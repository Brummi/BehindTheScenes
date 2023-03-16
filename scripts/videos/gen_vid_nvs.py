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
    dry_run = False

    indices = [1044]

    d_min = 3
    d_max = 40

    task = "KITTI-360"
    assert task in ["KITTI-360", "KITTI-Raw", "RealEstate10K"]

    cam_traj = "simple_movement.npy"

    if task == "KITTI-360":
        dataset, config_path, cp_path, out_path, resolution, cam_incl_adjust = setup_kitti360("videos/nvs", "2013_05_28_drive_0000_sync", "val_seq")
    elif task == "KITTI-Raw":
        dataset, config_path, cp_path, out_path, resolution, cam_incl_adjust = setup_kittiraw("videos/nvs", "val")
    elif task == "RealEstate10K":
        dataset, config_path, cp_path, out_path, resolution, cam_incl_adjust = setup_re10k("videos/nvs", "val")
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
    traj_folder = Path("scripts/videos/trajectories")

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

    camera_trj = np.load(str(traj_folder / cam_traj))
    poses_nv = torch.tensor(camera_trj, dtype=torch.float).to(device)
    poses_nv[:, :3, 3] *= .75
    poses_nv = poses_nv[::2, :, :]

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

            for pose in tqdm(poses_nv):
                novel_view, depth = render_poses(renderer, ray_sampler, pose.view(1, 1, 4, 4), projs[:, :1])

                novel_view = novel_view[0, :, :, 0].cpu().numpy()
                depth = ((1 / depth - 1 / d_max) / (1 / d_min - 1 / d_max)).clamp(0, 1)
                depth = color_tensor(depth, "magma", norm=False).numpy()

                if s_img and s_depth:
                    frame = np.concatenate((novel_view, depth), axis=0)
                elif s_img:
                    frame = novel_view
                elif s_depth:
                    frame = depth
                else:
                    frame = None

                frames.append(frame)

            frames = [(frame * 255).astype(np.uint8) for frame in frames]

            file_name = f"{idx:010d}_{cam_traj.split('.')[0]}.mp4"

            if not dry_run:
                video = list(frames)

                video = ImageSequenceClip(video, fps=10)
                video.write_videofile(str(out_path / file_name))
                video.close()

    print("Completed.")


if __name__ == '__main__':
    main()
