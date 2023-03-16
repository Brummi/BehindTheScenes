import math

import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import lpips
import skimage.metrics

from datasets.data_util import make_test_dataset
from models.common.render import NeRFRenderer
from models.bts.model.image_processor import make_image_processor, RGBProcessor
from models.bts.model.loss import ReconstructionLoss
from models.bts.model.models_bts import BTSNet
from models.bts.model.ray_sampler import ImageRaySampler, PatchRaySampler, RandomRaySampler
from utils.base_evaluator import base_evaluation
from utils.metrics import MeanMetric
from utils.projection_operations import distance_to_z

IDX = 0


class BTSWrapper(nn.Module):
    def __init__(self, renderer, config, ) -> None:
        super().__init__()

        self.renderer = renderer

        self.z_near = config["z_near"]
        self.z_far = config["z_far"]
        self.ray_batch_size = config["ray_batch_size"]
        self.sampler = ImageRaySampler(self.z_near, self.z_far)

        self.lpips_vgg = lpips.LPIPS(net="vgg")

        self.depth_scaling = config.get("depth_scaling", None)

    @staticmethod
    def get_loss_metric_names():
        return ["loss", "loss_l2", "loss_mask", "loss_temporal"]

    def forward(self, data):
        data = dict(data)
        images = torch.stack(data["imgs"], dim=1)                           # n, v, c, h, w
        poses = torch.stack(data["poses"], dim=1)                           # n, v, 4, 4 w2c
        projs = torch.stack(data["projs"], dim=1)                           # n, v, 4, 4 (-1, 1)

        n, v, c, h, w = images.shape
        device = images.device

        # Use first frame as keyframe
        to_base_pose = torch.inverse(poses[:, :1, :, :])
        poses = to_base_pose.expand(-1, v, -1, -1) @ poses

        ids_encoder = [0]

        self.renderer.net.compute_grid_transforms(projs[:, ids_encoder], poses[:, ids_encoder])
        self.renderer.net.encode(images, projs, poses, ids_encoder=ids_encoder, ids_render=ids_encoder, )

        all_rays, all_rgb_gt = self.sampler.sample(images * .5 + .5, poses, projs)

        data["fine"] = []
        data["coarse"] = []

        self.renderer.net.set_scale(0)
        render_dict = self.renderer(all_rays, want_weights=True, want_alphas=True)

        if "fine" not in render_dict:
            render_dict["fine"] = dict(render_dict["coarse"])

        render_dict["rgb_gt"] = all_rgb_gt
        render_dict["rays"] = all_rays

        render_dict = self.sampler.reconstruct(render_dict)

        render_dict["coarse"]["depth"] = distance_to_z(render_dict["coarse"]["depth"], projs)
        render_dict["fine"]["depth"] = distance_to_z(render_dict["fine"]["depth"], projs)

        data["fine"].append(render_dict["fine"])
        data["coarse"].append(render_dict["coarse"])
        data["rgb_gt"] = render_dict["rgb_gt"]
        data["rays"] = render_dict["rays"]

        data["z_near"] = torch.tensor(self.z_near, device=images.device)
        data["z_far"] = torch.tensor(self.z_far, device=images.device)

        data.update(self.compute_depth_metrics(data))
        data.update(self.compute_nvs_metrics(data))

        globals()["IDX"] += 1

        return data

    def compute_depth_metrics(self, data):
        # TODO: This is only correct for batchsize 1!
        depth_gt = data["depths"][0]
        depth_pred = data["fine"][0]["depth"][:, :1]

        depth_pred = F.interpolate(depth_pred, depth_gt.shape[-2:])

        if self.depth_scaling == "median":
            mask = depth_gt > 0
            scaling = torch.median(depth_gt[mask]) / torch.median(depth_pred[mask])
            depth_pred = scaling * depth_pred
        elif self.depth_scaling == "l2":
            mask = depth_gt > 0
            depth_pred = depth_pred
            depth_gt_ = depth_gt[mask]
            depth_pred_ = depth_pred[mask]
            depth_pred_ = torch.stack((depth_pred_, torch.ones_like(depth_pred_)), dim=-1)
            x = torch.linalg.lstsq(depth_pred_.to(torch.float32), depth_gt_.unsqueeze(-1).to(torch.float32)).solution.squeeze()
            depth_pred = depth_pred * x[0] + x[1]

        depth_pred = torch.clamp(depth_pred, 1e-3, 80)
        mask = depth_gt != 0

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]

        thresh = torch.maximum((depth_gt / depth_pred), (depth_pred / depth_gt))
        a1 = (thresh < 1.25).to(torch.float)
        a2 = (thresh < 1.25 ** 2).to(torch.float)
        a3 = (thresh < 1.25 ** 3).to(torch.float)
        a1 = a1.mean()
        a2 = a2.mean()
        a3 = a3.mean()

        rmse = (depth_gt - depth_pred) ** 2
        rmse = rmse.mean() ** .5

        rmse_log = (torch.log(depth_gt) - torch.log(depth_pred)) ** 2
        rmse_log = rmse_log.mean() ** .5

        abs_rel = torch.abs(depth_gt - depth_pred) / depth_gt
        abs_rel = abs_rel.mean()

        sq_rel = ((depth_gt - depth_pred) ** 2) / depth_gt
        sq_rel = sq_rel.mean()

        metrics_dict = {
            "abs_rel": abs_rel,
            "sq_rel": sq_rel,
            "rmse": rmse,
            "rmse_log": rmse_log,
            "a1": a1,
            "a2": a2,
            "a3": a3
        }
        return metrics_dict

    def compute_nvs_metrics(self, data):
        # TODO: This is only correct for batchsize 1!
        # Following tucker et al. and others, we crop 5% on all sides

        # idx of stereo frame (the target frame is always the "stereo" frame).
        sf_id = data["rgb_gt"].shape[1] // 2

        imgs_gt = data["rgb_gt"][:, sf_id:sf_id+1]
        imgs_pred = data["fine"][0]["rgb"][:, sf_id:sf_id+1]

        imgs_gt = imgs_gt.squeeze(0).permute(0, 3, 1, 2)
        imgs_pred = imgs_pred.squeeze(0).squeeze(-2).permute(0, 3, 1, 2)

        n, c, h, w = imgs_gt.shape
        y0 = int(math.ceil(0.05 * h))
        y1 = int(math.floor(0.95 * h))
        x0 = int(math.ceil(0.05 * w))
        x1 = int(math.floor(0.95 * w))

        imgs_gt = imgs_gt[:, :, y0:y1, x0:x1]
        imgs_pred = imgs_pred[:, :, y0:y1, x0:x1]

        imgs_gt_np = imgs_gt.detach().squeeze().permute(1, 2, 0).cpu().numpy()
        imgs_pred_np = imgs_pred.detach().squeeze().permute(1, 2, 0).cpu().numpy()

        ssim_score = skimage.metrics.structural_similarity(imgs_pred_np, imgs_gt_np, multichannel=True, data_range=1)
        psnr_score = skimage.metrics.peak_signal_noise_ratio(imgs_pred_np, imgs_gt_np, data_range=1)
        lpips_score = self.lpips_vgg(imgs_pred, imgs_gt, normalize=False).mean()

        metrics_dict = {
            "ssim": ssim_score,
            "psnr": psnr_score,
            "lpips": lpips_score
        }
        return metrics_dict


def evaluation(local_rank, config):
    return base_evaluation(local_rank, config, get_dataflow, initialize, get_metrics)


def get_dataflow(config):
    test_dataset = make_test_dataset(config["data"])
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=config["num_workers"], shuffle=False, drop_last=False)

    return test_loader


def get_metrics(config, device):
    names = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "ssim", "psnr", "lpips"]
    metrics = {name: MeanMetric((lambda n: lambda x: x["output"][n])(name), device) for name in names}
    return metrics


def initialize(config: dict, logger=None):
    arch = config["model_conf"].get("arch", "BTSNet")
    net = globals()[arch](config["model_conf"])
    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()

    model = BTSWrapper(
        renderer,
        config["model_conf"]
    )

    return model


def visualize(engine: Engine, logger: TensorboardLogger, step: int, tag: str):
    pass
