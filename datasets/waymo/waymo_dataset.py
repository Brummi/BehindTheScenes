import os
import pickle
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

from utils.augmentation import get_color_aug_fn


class WaymoDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 mode: str,
                 split_path: Optional[str],
                 target_image_size: tuple = (320, 480),
                 return_45: bool = True,
                 return_90: bool = True,
                 return_depth: bool = False,
                 frame_count: int = 2,
                 keyframe_offset: int = 0,
                 dilation: int = 1,
                 offset_45: int = 5,
                 offset_90: int = 10,
                 color_aug: bool = False,
                 correct_exposure: bool = False,
                 ):
        self.data_path = str(Path(data_path) / mode)
        self.split_path = split_path
        self.target_image_size = target_image_size

        self.return_45 = return_45
        self.return_90 = return_90
        self.return_depth = return_depth

        self.frame_count = frame_count
        self.keyframe_offset = keyframe_offset
        self.dilation = dilation
        self.offset_45 = offset_45
        self.offset_90 = offset_90

        self.color_aug = color_aug

        self.correct_exposure = correct_exposure

        self._sequences = self._get_sequences(self.data_path)

        self._calibs = self._load_calibs(self.data_path, self._sequences)
        self._poses = self._load_poses(self.data_path, self._sequences)
        self._exposures = self._load_exposures(self.data_path, self._sequences)

        self._left_offset = ((self.frame_count - 1) // 2 + self.keyframe_offset) * self.dilation

        if self.split_path is not None:
            self._datapoints = self._load_split(self.split_path)
        else:
            self._datapoints = self._full_split(self._sequences, self._poses)

        self._skip = 0
        self.length = len(self._datapoints)

    @staticmethod
    def _get_sequences(data_path: str):
        all_sequences = []

        seqs_path = Path(data_path)
        for seq in seqs_path.iterdir():
            if not seq.is_dir():
                continue
            all_sequences.append(seq.name)

        return all_sequences

    @staticmethod
    def _full_split(sequences: list, poses: dict):
        datapoints = []
        for seq in sorted(sequences):
            ids = [id for id in range(len(poses[seq]))]
            datapoints_seq = [(seq, id) for id in ids]
            datapoints.extend(datapoints_seq)
        return datapoints

    @staticmethod
    def _load_split(split_path: str):
        with open(split_path, "r") as f:
            lines = f.readlines()

        def split_line(l):
            segments = l.split(" ")
            seq = segments[0]
            id = int(segments[1])
            return seq, id

        return list(map(split_line, lines))

    @staticmethod
    def _load_calibs(data_path: str, sequences: list):
        data_path = Path(data_path)

        calibs = {}

        for seq in sequences:
            seq_folder = data_path / seq

            with (seq_folder / "calibration.pkl").open(mode="rb") as f:
                seq_calib = pickle.load(f)

            calibs[seq] = seq_calib

        return calibs

    @staticmethod
    def _load_poses(data_path: str, sequences: list):
        poses = {}

        for seq in sequences:
            pose_file = Path(data_path) / seq / f"poses.npy"
            seq_poses = np.load(str(pose_file))

            poses[seq] = seq_poses

        return poses

    @staticmethod
    def _load_exposures(data_path: str, sequences: list):
        exposures = {}

        for seq in sequences:
            exposure_file = Path(data_path) / seq / f"exposures.pkl"
            with exposure_file.open(mode="rb") as f:
                seq_exposures = pickle.load(f)

            exposures[seq] = seq_exposures

        return exposures

    def get_exposure_correction(self, exp_0, exp_45_l, exp_45_r, exp_90_l, exp_90_r):
        median_exposure = np.median(np.concatenate((exp_0, exp_45_l, exp_45_r, exp_90_l, exp_90_r)))
        corr_0 = [median_exposure / exp for exp in exp_0]
        corr_45_l = [median_exposure / exp for exp in exp_45_l]
        corr_45_r = [median_exposure / exp for exp in exp_45_r]
        corr_90_l = [median_exposure / exp for exp in exp_90_l]
        corr_90_r = [median_exposure / exp for exp in exp_90_r]
        return corr_0, corr_45_l, corr_45_r, corr_90_l, corr_90_r

    def load_images(self, seq: str, ids: list, ids_45: Optional[list], ids_90: Optional[list]):
        imgs_0 = []
        imgs_45_l, imgs_45_r = [], []
        imgs_90_l, imgs_90_r = [], []

        for id in ids:
            img = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, seq, "frames", "cam_01", f"{id:010d}.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            imgs_0 += [img]

        for id in ids_45:
            img_left = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, seq, "frames", "cam_02", f"{id:010d}.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            imgs_45_l += [img_left]
        for id in ids_45:
            img_right = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, seq, "frames", "cam_03", f"{id:010d}.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            imgs_45_r += [img_right]

        for id in ids_90:
            img_left = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, seq, "frames", "cam_04", f"{id:010d}.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            imgs_90_l += [img_left]
        for id in ids_90:
            img_right = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, seq, "frames", "cam_05", f"{id:010d}.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            imgs_90_r += [img_right]

        return imgs_0, imgs_45_l, imgs_45_r, imgs_90_l, imgs_90_r

    def process_img(self, img: np.array, color_aug_fn=None, exposure_correction_factor=None):
        if self.target_image_size and (self.target_image_size[0] != img.shape[0] or self.target_image_size[1] != img.shape[1]):
            img = cv2.resize(img, (self.target_image_size[1], self.target_image_size[0]), interpolation=cv2.INTER_LINEAR)

        if exposure_correction_factor is not None:
            img = img ** 2.2
            img *= exposure_correction_factor
            img = img ** (1 / 2.2)
            img = np.clip(img, 0, 1)

        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img)

        if color_aug_fn is not None:
            img = color_aug_fn(img)

        img = img * 2 - 1
        return img

    def load_depth(self, seq, id):
        points = np.load(os.path.join(self.data_path, seq, "lidar", f"{id:010d}.npy")).astype(dtype=np.float32).reshape(-1, 3)

        points_hom = np.concatenate((points, np.ones_like(points[:, :1])), axis=1)
        points_cam = ((self._calibs[seq]["proj_mats"][1] @ np.linalg.inv(self._calibs[seq]["extrinsics"][1])[:3, :]) @ points_hom.T).T
        points_cam[:, :2] = points_cam[:, :2] / points_cam[:, 2:3]

        mask = (points_cam[:, 0] > -1) & (points_cam[:, 0] < 1) & (points_cam[:, 1] > -1) & (points_cam[:, 1] < 1) & (points_cam[:, 2] > 0)
        points_cam = points_cam[mask, :]

        # project to image
        depth = np.zeros(self.target_image_size)
        depth[
            ((points_cam[:, 1] * .5 + .5) * self.target_image_size[0]).astype(np.int).clip(0, self.target_image_size[0] - 1),
            ((points_cam[:, 0] * .5 + .5) * self.target_image_size[1]).astype(np.int).clip(0, self.target_image_size[1] - 1)] = points_cam[:, 2]

        depth[depth < 0] = 0

        return depth[None, :, :]

    def __getitem__(self, index: int):
        _start_time = time.time()

        if index >= self.length:
            raise IndexError()

        if self._skip != 0:
            index += self._skip

        sequence, id = self._datapoints[index]
        seq_len = self._poses[sequence].shape[0]

        ids = [id] + [max(min(i, seq_len-1), 0) for i in range(id - self._left_offset, id - self._left_offset + self.frame_count * self.dilation, self.dilation) if i != id]
        ids_45 = [max(min(id + self.offset_45, seq_len-1), 0) for id in ids]
        ids_90 = [max(min(id + self.offset_90, seq_len-1), 0) for id in ids]

        if not self.return_45:
            ids_45 = []
        if not self.return_90:
            ids_90 = []

        if self.color_aug:
            color_aug_fn = get_color_aug_fn(ColorJitter.get_params(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1, 0.1)))
        else:
            color_aug_fn = None

        if self.correct_exposure:
            exp_0 = self._exposures[sequence][1][ids]
            exp_45_l = self._exposures[sequence][2][ids_45]
            exp_45_r = self._exposures[sequence][3][ids_45]
            exp_90_l = self._exposures[sequence][4][ids_90]
            exp_90_r = self._exposures[sequence][5][ids_90]
            exp_0, exp_45_l, exp_45_r, exp_90_l, exp_90_r = self.get_exposure_correction(exp_0, exp_45_l, exp_45_r, exp_90_l, exp_90_r)
        else:
            exp_0 = [None for _ in ids]
            exp_45_l = [None for _ in ids_45]
            exp_45_r = [None for _ in ids_45]
            exp_90_l = [None for _ in ids_90]
            exp_90_r = [None for _ in ids_90]

        _start_time_loading = time.time()
        imgs_0, imgs_45_l, imgs_45_r, imgs_90_l, imgs_90_r = self.load_images(sequence, ids, ids_45, ids_90)
        _loading_time = np.array(time.time() - _start_time_loading)

        _start_time_processing = time.time()
        imgs_0 = [self.process_img(img, color_aug_fn=color_aug_fn, exposure_correction_factor=exp_c) for img, exp_c in zip(imgs_0, exp_0)]
        imgs_45_l = [self.process_img(img, color_aug_fn=color_aug_fn, exposure_correction_factor=exp_c) for img, exp_c in zip(imgs_45_l, exp_45_l)]
        imgs_45_r = [self.process_img(img, color_aug_fn=color_aug_fn, exposure_correction_factor=exp_c) for img, exp_c in zip(imgs_45_r, exp_45_r)]
        imgs_90_l = [self.process_img(img, color_aug_fn=color_aug_fn, exposure_correction_factor=exp_c) for img, exp_c in zip(imgs_90_l, exp_90_l)]
        imgs_90_r = [self.process_img(img, color_aug_fn=color_aug_fn, exposure_correction_factor=exp_c) for img, exp_c in zip(imgs_90_r, exp_90_r)]
        _processing_time = np.array(time.time() - _start_time_processing)

        # These poses are camera to world !!
        poses_0 = [self._poses[sequence][i, :, :] @ self._calibs[sequence]["extrinsics"][1] for i in ids]
        poses_45_l = [self._poses[sequence][i, :, :] @ self._calibs[sequence]["extrinsics"][2] for i in ids_45]
        poses_45_r = [self._poses[sequence][i, :, :] @ self._calibs[sequence]["extrinsics"][3] for i in ids_45]
        poses_90_l = [self._poses[sequence][i, :, :] @ self._calibs[sequence]["extrinsics"][4] for i in ids_90]
        poses_90_r = [self._poses[sequence][i, :, :] @ self._calibs[sequence]["extrinsics"][5] for i in ids_90]

        projs_0 = [self._calibs[sequence]["proj_mats"][1] for _ in ids]
        projs_45_l = [self._calibs[sequence]["proj_mats"][2] for _ in ids_45]
        projs_45_r = [self._calibs[sequence]["proj_mats"][3] for _ in ids_45]
        projs_90_l = [self._calibs[sequence]["proj_mats"][4] for _ in ids_90]
        projs_90_r = [self._calibs[sequence]["proj_mats"][5] for _ in ids_90]

        imgs = imgs_0 + imgs_45_l + imgs_45_r + imgs_90_l + imgs_90_r
        projs = projs_0 + projs_45_l + projs_45_r + projs_90_l + projs_90_r
        poses = poses_0 + poses_45_l + poses_45_r + poses_90_l + poses_90_r

        if self.return_depth:
            depths = [self.load_depth(sequence, ids[0])]
        else:
            depths = []

        bboxes_3d = []
        segs = []

        _proc_time = np.array(time.time() - _start_time)

        data = {
            "imgs": imgs,
            "projs": projs,
            "poses": poses,
            "depths": depths,
            "3d_bboxes": bboxes_3d,
            "segs": segs,
            "t__get_item__": np.array([_proc_time])
        }

        return data

    def __len__(self) -> int:
        return self.length
