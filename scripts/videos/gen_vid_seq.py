import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from tqdm import tqdm

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
    s_profile = False
    dry_run = False

    task = "KITTI-360"
    assert task in ["KITTI-360", "KITTI-Raw", "RealEstate10K"]

    FROM = 1000
    TO = 1400
    assert 0 <= FROM < TO

    d_min = 3
    d_max = 40

    if task == "KITTI-360":
        dataset, config_path, cp_path, out_path, resolution, cam_incl_adjust = setup_kitti360("videos/seq", "2013_05_28_drive_0000_sync", "val_seq")
    elif task == "KITTI-Raw":
        dataset, config_path, cp_path, out_path, resolution, cam_incl_adjust = setup_kittiraw("videos/seq", "test")
    elif task == "RealEstate10K":
        dataset, config_path, cp_path, out_path, resolution, cam_incl_adjust = setup_re10k("videos/seq", "test")
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
    file_name = f"{FROM:05d}-{TO:05d}.mp4"

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

    # Change resolution to match final height
    if s_depth and s_img:
        OUT_RES.P_RES_ZX = (resolution[0] * 2, resolution[0] * 2)
    else:
        OUT_RES.P_RES_ZX = (resolution[0], resolution[0])

    frames = []

    with torch.no_grad():
        for idx in tqdm(range(FROM, TO)):
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

            img = img.numpy()

            if s_depth:
                _, depth = render_poses(renderer, ray_sampler, poses[:, :1], projs[:, :1])
                depth = 1 / depth
                depth = ((depth - 1 / d_max) / (1 / d_min - 1 / d_max)).clamp(0, 1)
                depth = color_tensor(depth, "magma", norm=False).numpy()
            else:
                depth = None

            if s_profile:
                profile = render_profile(net, cam_incl_adjust)
                profile = color_tensor(profile.cpu(), "magma", norm=True).numpy()
            else:
                profile = None

            if s_img and s_depth and s_profile:
                frame = np.concatenate((img, depth), axis=0)
                frame = np.concatenate((frame, profile), axis=1)
            elif s_img and s_depth:
                frame = np.concatenate((img, depth), axis=0)
            elif s_img and s_profile:
                frame = np.concatenate((img, profile), axis=1)
            elif s_img:
                frame = img
            elif s_depth:
                frame = depth
            elif s_profile:
                frame = profile
            else:
                frame = None
            frames.append(frame)

    frames = [(frame * 255).astype(np.uint8) for frame in frames]

    if not dry_run:
        video = ImageSequenceClip(frames, fps=10)
        video.write_videofile(str(out_path / file_name))
        video.close()

    print("Completed.")


if __name__ == '__main__':
    main()
