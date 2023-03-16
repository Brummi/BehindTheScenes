import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from scipy.spatial.transform import Rotation
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
    dry_run = False

    indices = [0]

    task = "KITTI-360"
    assert task in ["KITTI-360", "KITTI-Raw"]

    length = 30

    d_min = 3
    d_max = 40

    if task == "KITTI-360":
        dataset, config_path, cp_path, out_path, resolution, cam_incl_adjust = setup_kitti360("videos/transition", "test")
        z_top = 10
        y_top = -6
        t_near = 5
        t_far = 7
        target_angle = math.radians(85)
    elif task == "KITTI-Raw":
        dataset, config_path, cp_path, out_path, resolution, cam_incl_adjust = setup_kittiraw("videos/transition", "test")
        z_top = 14
        y_top = -8
        t_near = 8
        t_far = 10
        target_angle = math.radians(90)
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

    z_near = config["model_conf"]["z_near"]
    z_far = config["model_conf"]["z_far"]

    z_near = d_min
    z_far = d_max

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

            frames = []

            for i in tqdm(range(length + 5)):

                prog = (i / (length - 1)) ** 2
                prog = min(prog, 1)

                pose = torch.eye(4, device=device)
                angle = -target_angle * prog
                rotation = torch.tensor(Rotation.from_euler("x", angle, degrees=False).as_matrix(), device=device)
                pose[:3, :3] = rotation

                z = z_top - math.cos(-angle) * z_top
                y = math.sin(-angle) * y_top
                pose[1, 3] = y
                pose[2, 3] = z

                z_near_ = z_near * (1 - prog) + t_near * prog
                z_far_ = z_far * (1 - prog) + t_far * prog

                target_width = int(resolution[1] * (1 - prog) + resolution[0] * prog)
                pad_left = (resolution[1] - target_width) // 2
                pad_right = (resolution[1] - target_width) - pad_left

                projs_ = projs[:, :1].clone()
                projs_[0, 0, 1, 1] = projs_[0, 0, 1, 1] * (target_width / resolution[1])

                ray_sampler.width = target_width
                ray_sampler.z_near = z_near_
                ray_sampler.z_far = z_far_

                novel_view, depth = render_poses(renderer, ray_sampler, pose.view(1, 1, 4, 4), projs_[:, :1])

                depth = 1 / depth.squeeze()
                depth = ((depth - 1 / z_far_) / (1 / z_near_ - 1 / z_far_)).clamp(0, 1)

                novel_view = novel_view.squeeze(-2).squeeze(0)

                if i > 0:
                    depth_ = torch.zeros(*resolution, device=device)
                    depth_[:, pad_left:-pad_right] = depth
                    depth = depth_

                    novel_view_ = torch.zeros(*resolution, 3, device=device)
                    novel_view_[:, pad_left:-pad_right, :] = novel_view
                    novel_view = novel_view_

                novel_view = novel_view[:, :].cpu().numpy()
                depth = color_tensor(depth.cpu(), "magma", norm=False).numpy()

                novel_view = (novel_view * 255).astype(np.uint8)
                depth = (depth * 255).astype(np.uint8)

                if s_img and s_depth:
                    frame = np.concatenate((novel_view, depth), axis=0)
                elif s_img:
                    frame = novel_view
                elif s_depth:
                    frame = depth
                else:
                    frame = None

                frames.append(frame)

            file_name = f"{idx:010d}.mp4"

            if not dry_run:
                video = list(frames)

                video = ImageSequenceClip(video, fps=10)
                video.write_videofile(str(out_path / file_name))
                video.close()

    print("Completed.")


if __name__ == '__main__':
    main()
