import sys
from argparse import ArgumentParser

sys.path.append(".")

from scripts.inference_setup import *

from hydra import compose, initialize
from omegaconf import OmegaConf

import torch

from models.bts.model import BTSNet, ImageRaySampler
from models.common.render import NeRFRenderer
from utils.array_operations import map_fn, unsqueezer
from utils.plotting import color_tensor


def main():
    parser = ArgumentParser("Generate density field from single image.")
    parser.add_argument("--img", "-i", required=True, help="Path to the image.")
    parser.add_argument("--plot", "-p", action="store_true", help="Plot rather than save images.")
    parser.add_argument("--model", "-m", help="Which pretrained model to use. (KITTI-360 (default), KITTI-Raw, RealEstate10K)", default="KITTI-360")

    args = parser.parse_args()

    s_img = True
    s_depth = True
    s_profile = True
    dry_run = args.plot

    model = args.model
    assert model in ["KITTI-360", "KITTI-Raw", "RealEstate10K"]

    if model == "KITTI-360":
        resolution = (192, 640)

        config_path = "exp_kitti_360"

        cp_path = Path(f"out/kitti_360/pretrained")
        cp_name = cp_path.name
        cp_path = next(cp_path.glob("training*.pt"))

        out_path = Path(f"media/img_custom/kitti-360_{cp_name}")

        cam_incl_adjust = torch.tensor(
            [[1.0000000, 0.0000000, 0.0000000, 0],
             [0.0000000, 0.9961947, -0.0871557, 0],
             [0.0000000, 0.0871557, 0.9961947, 0],
             [0.0000000, 000000000, 0.0000000, 1]
             ],
            dtype=torch.float32).view(1, 4, 4)

        proj = torch.tensor([
            [ 0.7849,  0.0000, -0.0312, 0],
            [ 0.0000,  2.9391,  0.2701, 0],
            [ 0.0000,  0.0000,  1.0000, 0],
            [ 0.0000,  0.0000,  0.0000, 1],
        ], dtype=torch.float32).view(1, 4, 4)
    elif model == "KITTI-Raw":
        resolution = (192, 640)
        config_path = "exp_kitti_raw"

        cp_path = Path(f"out/kitti_raw/pretrained")
        cp_name = cp_path.name
        cp_path = next(cp_path.glob("training*.pt"))

        out_path = Path(f"media/img_custom/kitti-raw_{cp_name}")

        cam_incl_adjust = None

        proj = torch.tensor([
            [ 1.1619,  0.0000, -0.0184, 0],
            [ 0.0000,  3.8482, -0.0781, 0],
            [ 0.0000,  0.0000,  1.0000, 0],
            [ 0.0000,  0.0000,  0.0000, 1]
        ], dtype=torch.float32).view(1, 4, 4)
    elif model == "RealEstate10K":
        resolution = (256, 384)

        config_path = "exp_re10k"

        cp_path = Path(f"out/re10k/pretrained")
        cp_name = cp_path.name
        cp_path = next(cp_path.glob("training*.pt"))

        out_path = Path(f"media/img_custom/re10k_{cp_name}")

        cam_incl_adjust = None
        proj = torch.tensor([
            [1.0056, 0.0000, 0.0000, 0],
            [0.0000, 1.7877, 0.0000, 0],
            [0.0000, 0.0000, 1.0000, 0],
            [0.0000, 0.0000, 0.0000, 1],
        ], dtype=torch.float32).view(1, 4, 4)
    else:
        raise ValueError(f"Invalid model: {model}")

    initialize(version_base=None, config_path="../../configs", job_name="gen_imgs")
    config = compose(config_name=config_path, overrides=[])

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

    print("Load input image")
    assert os.path.exists(args.img)
    img = cv2.cvtColor(cv2.imread(args.img), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    img = cv2.resize(img, (resolution[1], resolution[0]))
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device) * 2 - 1
    img_name = os.path.basename(args.img).split(".")[0]

    with torch.no_grad():
        poses = torch.eye(4).view(1, 1, 4, 4).to(device)
        projs = proj.view(1, 1, 4, 4).to(device)[:, :, :3, :3]

        net.encode(img, projs, poses, ids_encoder=[0], ids_render=[0])
        net.set_scale(0)

        img_save = img[0, 0].permute(1, 2, 0).cpu() * .5 + .5
        _, depth = render_poses(renderer, ray_sampler, poses[:, :1], projs[:, :1])

        if s_profile:
            profile = render_profile(net, cam_incl_adjust)
        else:
            profile = None

        depth = ((1 / depth - 1 / config["model_conf"]["z_far"]) / (1 / config["model_conf"]["z_near"] - 1 / config["model_conf"]["z_far"])).clamp(0, 1)

        print(f"Generated " + str(out_path / f"{img_name}"))

        if s_img:
            save_plot(img_save.numpy(), str(out_path / f"{img_name}_in.png"), dry_run=dry_run)
        if s_depth:
            save_plot(color_tensor(depth, "magma", norm=True).numpy(), str(out_path / f"{img_name}_depth.png"), dry_run=dry_run)
        if s_profile:
            save_plot(color_tensor(profile.cpu(), "magma", norm=True).numpy(), str(out_path / f"{img_name}_profile.png"), dry_run=dry_run)


if __name__ == '__main__':
    main()
