import argparse
import json
import pickle
import sys
import time

from matplotlib import pyplot as plt

sys.path.append(".")

from pathlib import Path
import subprocess

import cv2
import numpy as np
from tqdm import tqdm

import tensorflow.compat.v1 as tf

tf.enable_eager_execution()

from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


DRY_RUN = False
PROCESS_POINTS = False
PROCESS_IMGS = False

# Transform waymo camera coordinate system (x forward, y left, z up) to the canonical model (x right, y down, z forward)
axis_swap = np.array([
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
], dtype=np.float32)

axis_swap_inv = np.linalg.inv(axis_swap)


def plot_image_with_projection(img, points, proj_mat, extrinsics):
    h, w, c = img.shape

    points_hom = np.concatenate((points, np.ones_like(points[:, :1])), axis=1)
    points_cam = ((proj_mat @ np.linalg.inv(extrinsics)[:3, :]) @ points_hom.T).T
    points_cam[:, :2] = points_cam[:, :2] / points_cam[:, 2:3]

    mask = (points_cam[:, 0] > -1) & (points_cam[:, 0] < 1) & (points_cam[:, 1] > -1) & (points_cam[:, 1] < 1) & (points_cam[:, 2] > 0)
    points_cam = points_cam[mask, :]

    plt.imshow(img)
    plt.scatter((points_cam[:, 0] * .5 + .5) * w, (points_cam[:, 1] * .5 + .5) * h, c=points_cam[:, 2], cmap="jet", marker=',', lw=0, s=1)
    plt.show()


def get_proj_mat(intrs, dims):
    h, w = dims
    return np.array([[intrs[0] * 2 / w, 0, intrs[2] * 2 / w - 1.], [0, intrs[1] * 2 / h, intrs[3] * 2 / h - 1], [0, 0, 1]], dtype=np.float32)


def get_dist_coeff(intrs):
    return np.array((intrs[4], intrs[5], intrs[6], intrs[7], intrs[8]), dtype=np.float32)


def precompute_undistort(proj_mat, dist_coeff, size):
    h, w = size
    proj_mat = np.array(proj_mat)
    proj_mat[0, 0] = proj_mat[0, 0] * .5 * w
    proj_mat[0, 2] = (proj_mat[0, 2] * .5 + .5) * w
    proj_mat[1, 1] = proj_mat[1, 1] * .5 * h
    proj_mat[1, 2] = (proj_mat[1, 2] * .5 + .5) * h
    map1, map2 = cv2.initUndistortRectifyMap(proj_mat, dist_coeff, np.eye(3), proj_mat, (w, h), cv2.CV_16SC2)
    return map1, map2


def undistort(img, proj_mat, dist_coeff):
    img = tf.image.decode_jpeg(img.image).numpy()
    h, w, c = img.shape

    proj_mat = np.array(proj_mat)
    proj_mat[0, 0] = proj_mat[0, 0] * .5 * w
    proj_mat[0, 2] = (proj_mat[0, 2] * .5 + .5) * w
    proj_mat[1, 1] = proj_mat[1, 1] * .5 * h
    proj_mat[1, 2] = (proj_mat[1, 2] * .5 + .5) * h

    img_ud = cv2.undistort(img, proj_mat, dist_coeff)
    return img_ud


def extract_frame(frame, undistortion_maps, proj_mats, dist_coeffs):
    # Images
    images = {img.name: img for img in frame.images}

    exposure = {img.name: img.shutter for img in frame.images}

    if PROCESS_POINTS:
        (range_images, camera_projections, _, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose)

        # 3d points in vehicle frame
        points_all = np.concatenate(points, axis=0)
    else:
        points_all = None

    if PROCESS_IMGS:
        # Undistorted images
        # imgs_undistorted = {name: undistort(img, proj_mats[name], dist_coeffs[name]) for name, img in images.items()}
        imgs_undistorted = {name: cv2.remap(tf.image.decode_jpeg(img.image).numpy(), undistortion_maps[name][0], undistortion_maps[name][1], cv2.INTER_LINEAR) for name, img in images.items()}
    else:
        imgs_undistorted = None

    # Vehicle to world transform
    vehicle_pose = np.array(list(frame.pose.transform), dtype=np.float32).reshape([4, 4])

    return imgs_undistorted, points_all, vehicle_pose, exposure


def setup_folders(seq_folder: Path, names):
    frames_folder = seq_folder / "frames"
    cam_folders = {name: frames_folder / f"cam_{name:02d}" for name in names}
    lidar_folder = seq_folder / "lidar"

    if not DRY_RUN:
        frames_folder.mkdir(exist_ok=True)
        for folder in cam_folders.values():
            folder.mkdir(exist_ok=True)

        lidar_folder.mkdir(exist_ok=True)

    return cam_folders, lidar_folder


def main():
    parser = argparse.ArgumentParser("Waymo Extraction")
    parser.add_argument("--data_in", "-i", type=str)
    parser.add_argument("--data_out", "-o", type=str)
    parser.add_argument("--resolution", "-r", default=(320, 480))

    args = parser.parse_args()

    data_in = Path(args.data_in)
    data_out = Path(args.data_out)
    resolution = args.resolution

    target_wh = (resolution[1], resolution[0])

    data_out.mkdir(exist_ok=True, parents=True)
    records = list(data_in.iterdir())

    records = [r for r in records if r.suffix == ".tfrecord"]

    total = 0

    sequence_meta = {}

    if DRY_RUN:
        print("#" * 40)
        print("DRY RUN - NOT WRITING ANY FILES")
        print("#" * 40)

    pbar = tqdm(records)
    for record in pbar:
        dataset = tf.data.TFRecordDataset(record, compression_type='')

        base_data = next(iter(dataset))
        base_frame = open_dataset.Frame()
        base_frame.ParseFromString(bytearray(base_data.numpy()))

        seq_folder = data_out / base_frame.context.name
        seq_folder.mkdir(exist_ok=True)

        names = [img.name for img in base_frame.images]
        camera_calibrations = {cc.name: cc for cc in base_frame.context.camera_calibrations}
        intrinsics = {name: np.array(list(cc.intrinsic), dtype=np.float32) for name, cc in camera_calibrations.items()}
        # Image dimensions
        dims = {name: (cc.height, cc.width) for name, cc in camera_calibrations.items()}

        # Camera to vehicle transform
        extrinsics = {name: np.array(list(cc.extrinsic.transform), dtype=np.float32).reshape([4, 4]) @ axis_swap_inv for name, cc in camera_calibrations.items()}
        # Camera K matrix (normalized for [-1, 1] x [-1, 1]
        proj_mats = {name: get_proj_mat(cc, dims[name]) for name, cc in intrinsics.items()}
        # Distortion coefficients
        dist_coeffs = {name: get_dist_coeff(cc) for name, cc in intrinsics.items()}

        target_calibration = {
            "dims": resolution,
            "extrinsics": extrinsics,
            "proj_mats": proj_mats,
        }

        cam_folders, lidar_folder = setup_folders(seq_folder, names)

        if not DRY_RUN:
            with (seq_folder / "calibration.pkl").open(mode="wb") as f:
                pickle.dump(target_calibration, f)

        poses = []
        exposures = []

        num_frames = 0

        undistortion_maps = {name: precompute_undistort(proj_mats[name], dist_coeffs[name], dims[name]) for name in names}

        t = time.time()

        dt_parse_avg = -1
        dt_proc_avg = -1
        dt_write_avg = -1
        dt_total_avg = -1

        for i, data in enumerate(dataset):
            pbar.set_description(f"[{i:010d}, Parse: {dt_parse_avg:.2f}s, Proc: {dt_proc_avg:.2f}s, Write: {dt_write_avg:.2f}s, Total: {dt_total_avg:.2f}s]")

            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            dt_parse = time.time() - t
            t = time.time()

            imgs, points, pose, exposure = extract_frame(frame, undistortion_maps, proj_mats, dist_coeffs)

            if PROCESS_IMGS:
                imgs = {name: cv2.resize(img, target_wh, interpolation=cv2.INTER_LINEAR) for name, img in imgs.items()}

            poses.append(pose)
            exposures.append(exposure)

            dt_proc = time.time() - t
            t = time.time()

            if not DRY_RUN and PROCESS_IMGS:
                file_name = f"{i:010d}"

                for name, img in imgs.items():
                    cv2.imwrite(str(cam_folders[name] / f"{file_name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            if not DRY_RUN and PROCESS_POINTS:
                np.save(str(lidar_folder / f"{file_name}.npy"), points)

            dt_write = time.time() - t
            t = time.time()

            dt_total = dt_parse + dt_proc + dt_write

            if dt_total_avg < 0:
                dt_parse_avg = dt_parse
                dt_proc_avg = dt_proc
                dt_write_avg = dt_write
                dt_total_avg = dt_total
            else:
                dt_parse_avg = dt_parse_avg * .9 + dt_parse * .1
                dt_proc_avg = dt_proc_avg * .9 + dt_proc * .1
                dt_write_avg = dt_write_avg * .9 + dt_write * .1
                dt_total_avg = dt_total_avg * .9 + dt_total * .1
            num_frames += 1

        poses = np.array(poses, dtype=np.float32)
        exposures = {name: np.array([exp[name] for exp in exposures]) for name in exposures[0].keys()}

        if not DRY_RUN:
            np.save(seq_folder / f"poses.npy", poses)
            with (seq_folder / "exposures.pkl").open(mode="wb") as f:
                pickle.dump(exposures, f)

        sequence_meta[base_frame.context.name] = num_frames

        total += num_frames

    if not DRY_RUN:
        with (data_out / "meta_data.json").open("w") as f:
            json.dump(sequence_meta, f, indent=4)

    print(f"Extracted {total} frames.")


if __name__ == "__main__":
    main()
