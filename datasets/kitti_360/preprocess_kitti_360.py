import argparse
import sys
sys.path.append(".")

from pathlib import Path
import subprocess

import cv2
import numpy as np
from tqdm import tqdm

from datasets.kitti_360.kitti_360_dataset import Kitti360Dataset

DRY_RUN = False


def main():
    parser = argparse.ArgumentParser("KITTI 360 Preprocessing")
    parser.add_argument("--data_path", "-d", type=str)
    parser.add_argument("--resolution", "-r", default=(192, 640))
    parser.add_argument("--fisheye_rotation", "-f", default=(0, -15))
    parser.add_argument("--only_fisheye", "-o", action="store_true")

    args = parser.parse_args()

    data_path = Path(args.data_path)
    resolution = args.resolution
    rotation = args.fisheye_rotation
    only_fisheye = args.only_fisheye

    print("Setting up dataset")
    dataset = Kitti360Dataset(
        data_path=data_path,
        pose_path=data_path / "data_poses",
        split_path=None,
        return_stereo=True,
        frame_count=1,
        fisheye_rotation=rotation,
        color_aug=False,
        return_segmentation=False,
    )

    print("Setting up folders...")

    for i in tqdm(range(len(dataset))):
        sequence, id, is_right = dataset._datapoints[i]

        if is_right:
            continue

        image_00 = data_path / "data_2d_raw" / sequence / "image_00" / f"data_{resolution[0]}x{resolution[1]}"
        image_01 = data_path / "data_2d_raw" / sequence / "image_01" / f"data_{resolution[0]}x{resolution[1]}"
        image_02 = data_path / "data_2d_raw" / sequence / "image_02" / f"data_{resolution[0]}x{resolution[1]}_{rotation[0]}x{rotation[1]}"
        image_03 = data_path / "data_2d_raw" / sequence / "image_03" / f"data_{resolution[0]}x{resolution[1]}_{rotation[0]}x{rotation[1]}"

        img_id = dataset._img_ids[sequence][id]

        if (image_00 / f"{img_id:010d}.png").exists():
            continue

        data = dataset[i]

        image_00.mkdir(exist_ok=True, parents=True)
        image_01.mkdir(exist_ok=True, parents=True)
        image_02.mkdir(exist_ok=True, parents=True)
        image_03.mkdir(exist_ok=True, parents=True)

        img_00 = (np.transpose(data["imgs"][0].numpy(), (1, 2, 0)) * .5 + .5) * 255.
        img_01 = (np.transpose(data["imgs"][1].numpy(), (1, 2, 0)) * .5 + .5) * 255.
        img_02 = (np.transpose(data["imgs"][2].numpy(), (1, 2, 0)) * .5 + .5) * 255.
        img_03 = (np.transpose(data["imgs"][3].numpy(), (1, 2, 0)) * .5 + .5) * 255.

        if not only_fisheye:
            cv2.imwrite(str(image_00 / f"{img_id:010d}.png"), cv2.cvtColor(img_00, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(image_01 / f"{img_id:010d}.png"), cv2.cvtColor(img_01, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(image_02 / f"{img_id:010d}.png"), cv2.cvtColor(img_02, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(image_03 / f"{img_id:010d}.png"), cv2.cvtColor(img_03, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
