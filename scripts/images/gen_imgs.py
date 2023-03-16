import sys
sys.path.append(".")

from scripts.inference_setup import *

import copy

import hydra
import torch

from models.bts.model import BTSNet, ImageRaySampler
from models.common.render import NeRFRenderer
from utils.array_operations import map_fn, unsqueezer
from utils.plotting import color_tensor


def main():

    s_img = True
    s_depth = True
    s_profile = True
    dry_run = True

    indices = [0, 1, 2, 3]

    task = "KITTI-360"
    assert task in ["KITTI-360", "KITTI-Raw", "RealEstate10K"]

    if task == "KITTI-360":
        dataset, config_path, cp_path, out_path, resolution, cam_incl_adjust = setup_kitti360("imgs")
    elif task == "KITTI-Raw":
        dataset, config_path, cp_path, out_path, resolution, cam_incl_adjust = setup_kittiraw("imgs")
    elif task == "RealEstate10K":
        dataset, config_path, cp_path, out_path, resolution, cam_incl_adjust = setup_re10k("imgs")
    else:
        raise ValueError(f"Invalid task: {task}")

    # Slightly hacky, but we need to load the config based on the task
    global config
    config = {}
    @hydra.main(version_base=None, config_path="../../configs", config_name=config_path)
    def main_dummy(cfg):
        global config
        config = copy.deepcopy(cfg)
    main_dummy()

    print("Setup folders")
    out_path.mkdir(exist_ok=True, parents=True)

    print('Loading checkpoint')
    cp = torch.load(cp_path, map_location=device)

    net = BTSNet(config["model_conf"])
    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()
    renderer.renderer.n_coarse = 64
    renderer.renderer.lindisp = True

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.renderer = renderer

    _wrapper = _Wrapper()

    _wrapper.load_state_dict(cp["model"], strict=False)
    renderer.to(device)
    renderer.eval()

    ray_sampler = ImageRaySampler(config["model_conf"]["z_near"], config["model_conf"]["z_far"], *resolution, norm_dir=False)

    with torch.no_grad():
        for idx in indices:
            data = dataset[idx]
            data_batch = map_fn(map_fn(data, torch.tensor), unsqueezer)

            images = torch.stack(data_batch["imgs"], dim=1).to(device)
            poses = torch.stack(data_batch["poses"], dim=1).to(device)
            projs = torch.stack(data_batch["projs"], dim=1).to(device)

            # Move coordinate system to input frame
            poses = torch.inverse(poses[:, :1, :, :]) @ poses

            net.encode(images, projs, poses, ids_encoder=[0], ids_render=[0])
            net.set_scale(0)

            img = images[0, 0].permute(1, 2, 0).cpu() * .5 + .5
            _, depth = render_poses(renderer, ray_sampler, poses[:, :1], projs[:, :1])

            if s_profile:
                profile = render_profile(net, cam_incl_adjust)
            else:
                profile = None

            depth = ((1 / depth - 1 / config["model_conf"]["z_far"]) / (1 / config["model_conf"]["z_near"] - 1 / config["model_conf"]["z_far"])).clamp(0, 1)

            print(f"Generated " + str(out_path / f"{idx:010d}"))

            if s_img:
                save_plot(img.numpy(), str(out_path / f"{idx:010d}_in.png"), dry_run=dry_run)
            if s_depth:
                save_plot(color_tensor(depth, "magma", norm=True).numpy(), str(out_path / f"{idx:010d}_depth.png"), dry_run=dry_run)
            if s_profile:
                save_plot(color_tensor(profile.cpu(), "magma", norm=True).numpy(), str(out_path / f"{idx:010d}_profile.png"), dry_run=dry_run)


if __name__ == '__main__':
    main()
