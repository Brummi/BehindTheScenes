import math
from copy import copy
from typing import Optional, Union, Iterable, Sequence

import ignite.distributed as idist
import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import T_co, _collate_fn_t, _worker_init_fn_t
from torchvision.utils import make_grid

from datasets.data_util import make_datasets
from models.common.model.scheduler import make_scheduler
from models.common.render import NeRFRenderer
from models.bts.model.loss import ReconstructionLoss
from models.bts.trainer import get_metrics, BTSWrapper, BTSNet
from utils.array_operations import map_fn, unsqueezer, to
from utils.base_trainer import base_training
from utils.plotting import color_tensor


class EncoderDummy(nn.Module):
    def __init__(self, size, feat_dim, num_views=1) -> None:
        super().__init__()

        self.feats = nn.Parameter(torch.randn(num_views, feat_dim, *size))
        self.latent_size = feat_dim

    def forward(self, x):
        n = x.shape[0]
        return [self.feats.expand(n, -1, -1, -1)]


class DataloaderDummy(DataLoader):

    def __init__(self, dataset: Dataset[T_co], batch_size: Optional[int] = 1, shuffle: Optional[bool] = None,
                 sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None, num_workers: int = 0,
                 collate_fn: Optional[_collate_fn_t] = None, pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None, multiprocessing_context=None,
                 generator=None, *, prefetch_factor: int = 2, persistent_workers: bool = False,
                 pin_memory_device: str = ""):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory,
                         drop_last, timeout, worker_init_fn, multiprocessing_context, generator,
                         prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
                         pin_memory_device=pin_memory_device)

        self.element = to(map_fn(map_fn(dataset.__getitem__(0), torch.tensor), unsqueezer), "cuda:0")

    def _get_iterator(self):
        return iter([self.element])

    def __iter__(self):
        return super().__iter__()

    def __len__(self) -> int:
        return 1


class BTSWrapperOverfit(BTSWrapper):
    def __init__(self, renderer, config, eval_nvs=False, size=None) -> None:
        super().__init__(renderer, config, eval_nvs)

        self.encoder_dummy = EncoderDummy(size, config["encoder"]["d_out"], num_views=3)

        self.renderer.net.encoder = self.encoder_dummy
        self.renderer.net.flip_augmentation = False


def training(local_rank, config):
    return base_training(local_rank, config, get_dataflow, initialize, get_metrics, visualize)


def get_dataflow(config, logger=None):
    # - Get train/test datasets
    if idist.get_local_rank() > 0:
        # Ensure that only local rank 0 download the dataset
        # Thus each node will download a copy of the dataset
        idist.barrier()

    train_dataset, _ = make_datasets(config["data"])

    train_dataset.length = 1
    train_dataset._skip = config["data"].get("skip", 0)

    vis_dataset = copy(train_dataset)
    test_dataset = copy(train_dataset)

    vis_dataset.return_depth = True
    test_dataset.return_depth = True

    if idist.get_local_rank() == 0:
        # Ensure that only local rank 0 download the dataset
        idist.barrier()

    # Setup data loader also adapted to distributed config: nccl, gloo, xla-tpu
    train_loader = DataloaderDummy(train_dataset)
    test_loader = DataloaderDummy(test_dataset)
    vis_loader = DataloaderDummy(vis_dataset)

    return train_loader, test_loader, vis_loader


def initialize(config: dict, logger=None):
    arch = config["model_conf"].get("arch", "BTSNet")
    net = globals()[arch](config["model_conf"])
    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()

    mode = config.get("mode", "depth")

    model = BTSWrapperOverfit(
        renderer,
        config["model_conf"],
        mode == "nvs",
        size=config["data"].get("image_size", (192, 640))
    )

    model = idist.auto_model(model)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    optimizer = idist.auto_optim(optimizer)

    lr_scheduler = make_scheduler(config.get("scheduler", {}), optimizer)

    criterion = ReconstructionLoss(config["loss"], config["model_conf"].get("use_automasking", False))

    return model, optimizer, criterion, lr_scheduler


def visualize(engine: Engine, logger: TensorboardLogger, step: int, tag: str):
    print("Visualizing")

    data = engine.state.output["output"]
    writer = logger.writer

    images = torch.stack(data["imgs"], dim=1).detach()[0]
    recon_imgs = data["fine"][0]["rgb"].detach()[0]
    recon_depths = [f["depth"].detach()[0] for f in data["fine"]]

    depth_profile = data["coarse"][0]["alphas"].detach()[0]
    alphas = data["coarse"][0]["alphas"].detach()[0]
    invalids = data["coarse"][0]["invalid"].detach()[0]

    z_near = data["z_near"]
    z_far = data["z_far"]

    take = list(range(0, images.shape[0], 2))

    _, c, h, w = images.shape
    nv = recon_imgs.shape[0]

    images = images[take]
    images = images * .5 + .5

    recon_imgs = recon_imgs.view(nv, h, w, -1, c)
    recon_imgs = recon_imgs[take]
    # Aggregate recon_imgs by taking the mean
    recon_imgs = recon_imgs.mean(dim=-2).permute(0, 3, 1, 2)

    recon_mse = (((images - recon_imgs) ** 2) / 2).mean(dim=1).clamp(0, 1)
    recon_mse = color_tensor(recon_mse, cmap="plasma").permute(0, 3, 1, 2)

    recon_depths = [(1 / d[take] - 1 / z_far) / (1 / z_near - 1 / z_far) for d in recon_depths]
    recon_depths = [color_tensor(d.squeeze(1).clamp(0, 1), cmap="plasma").permute(0, 3, 1, 2) for d in recon_depths]

    depth_profile = depth_profile[take][:, [h//4, h//2, 3*h//4], :, :].view(len(take)*3, w, -1).permute(0, 2, 1)
    depth_profile = depth_profile.clamp_min(0) / depth_profile.max()
    depth_profile = color_tensor(depth_profile, cmap="plasma").permute(0, 3, 1, 2)

    alphas = alphas[take]

    alphas += 1e-5

    ray_density = alphas / alphas.sum(dim=-1, keepdim=True)
    ray_entropy = -(ray_density * torch.log(ray_density)).sum(-1) / (math.log2(alphas.shape[-1]))
    ray_entropy = color_tensor(ray_entropy, cmap="plasma").permute(0, 3, 1, 2)

    alpha_sum = (alphas.sum(dim=-1) / alphas.shape[-1]).clamp(-1)
    alpha_sum = color_tensor(alpha_sum, cmap="plasma").permute(0, 3, 1, 2)

    invalids = invalids[take]
    invalids = invalids.mean(-2).mean(-1)
    invalids = color_tensor(invalids, cmap="plasma").permute(0, 3, 1, 2)

    # Write images
    nrow = int(len(take) ** .5)

    images_grid = make_grid(images, nrow=nrow)
    recon_imgs_grid = make_grid(recon_imgs, nrow=nrow)
    recon_depths_grid = [make_grid(d, nrow=nrow) for d in recon_depths]
    depth_profile_grid = make_grid(depth_profile, nrow=nrow)
    ray_entropy_grid = make_grid(ray_entropy, nrow=nrow)
    alpha_sum_grid = make_grid(alpha_sum, nrow=nrow)
    recon_mse_grid = make_grid(recon_mse, nrow=nrow)
    invalids_grid = make_grid(invalids, nrow=nrow)

    writer.add_image(f"{tag}/input_im", images_grid.cpu(), global_step=step)
    writer.add_image(f"{tag}/recon_im", recon_imgs_grid.cpu(), global_step=step)
    for i, d in enumerate(recon_depths_grid):
        writer.add_image(f"{tag}/recon_depth_{i}", d.cpu(), global_step=step)
    writer.add_image(f"{tag}/depth_profile", depth_profile_grid.cpu(), global_step=step)
    writer.add_image(f"{tag}/ray_entropy", ray_entropy_grid.cpu(), global_step=step)
    writer.add_image(f"{tag}/alpha_sum", alpha_sum_grid.cpu(), global_step=step)
    writer.add_image(f"{tag}/recon_mse", recon_mse_grid.cpu(), global_step=step)
    writer.add_image(f"{tag}/invalids", invalids_grid.cpu(), global_step=step)
