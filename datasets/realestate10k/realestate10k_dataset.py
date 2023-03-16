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


class RealEstate10kDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 split_path: Optional[str]=None,
                 target_image_size=(256, 384),
                 frame_count=2,
                 dilation=1,
                 keyframe_offset=0,
                 color_aug=False
                 ):
        self.data_path = data_path
        self.split_path = split_path
        self.target_image_size = target_image_size
        self.frame_count = frame_count
        self.dilation = dilation
        self.keyframe_offset = keyframe_offset
        self.color_aug = color_aug

        if self.split_path is None:
            self.split = "train"
        else:
            self.split = "test"

        self._seq_data = self._load_seq_data(self.data_path, self.split)
        self._seq_keys = list(self._seq_data.keys())

        if isinstance(self.dilation, int):
            self._left_offset = ((self.frame_count - 1) // 2 + self.keyframe_offset) * self.dilation
            dilation = self.dilation
        else:
            self._left_offset = 0
            dilation = 0

        if self.split == "train":
            self._key_id_pairs = self._full_index(self._seq_keys, self._seq_data, self._left_offset, (self.frame_count-1) * dilation)
        else:
            self._key_id_pairs = self._load_index(split_path)

        self._skip = 0
        self.length = len(self._key_id_pairs)

    @staticmethod
    def _load_seq_data(data_path, split):
        file_path = Path(data_path) / f"{split}.pickle"
        with open(file_path, "rb") as f:
            seq_data = pickle.load(f)
        return seq_data

    @staticmethod
    def _full_index(seq_keys, seq_data, left_offset, extra_frames):
        key_id_pairs = []
        for k in seq_keys:
            seq_len = len(seq_data[k]["timestamps"])
            seq_key_id_pairs = [(k, i + left_offset) for i in range(seq_len - extra_frames)]
            key_id_pairs += seq_key_id_pairs
        return key_id_pairs

    @staticmethod
    def _load_index(index_path):
        def get_key_id(s):
            parts = s.split(" ")
            key = parts[0]
            id = int(parts[1])
            return key, id

        with open(index_path, "r") as f:
            lines = f.readlines()
        key_id_pairs = list(map(get_key_id, lines))
        return key_id_pairs

    def load_images(self, key, ids):
        imgs = []

        for id in ids:
            timestamp = self._seq_data[key]["timestamps"][id]
            img = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, "frames", self.split, key, f"{timestamp}.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            imgs += [img]

        return imgs

    def process_img(self, img: np.array, color_aug_fn=None):
        if self.target_image_size:
            img = cv2.resize(img, (self.target_image_size[1], self.target_image_size[0]), interpolation=cv2.INTER_LINEAR)
        img = np.transpose(img, (2, 0, 1))

        if color_aug_fn is not None:
            img = color_aug_fn(torch.tensor(img))

        img = img * 2 - 1
        return img

    @staticmethod
    def process_pose(pose):
        pose = np.concatenate((pose.astype(np.float32), np.array([[0, 0, 0, 1]], dtype=np.float32)), axis=0)
        pose = np.linalg.inv(pose)
        return pose

    @staticmethod
    def process_projs(proj):
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = 2 * proj[0]
        K[1, 1] = 2 * proj[1]
        K[0, 2] = 2 * proj[2] - 1
        K[1, 2] = 2 * proj[3] - 1
        return K

    def __getitem__(self, index: int):
        _start_time = time.time()

        if index >= self.length:
            raise IndexError()

        if self._skip != 0:
            index += self._skip

        if self.color_aug:
            color_aug_fn = get_color_aug_fn(ColorJitter.get_params(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1, 0.1)))
        else:
            color_aug_fn = None

        key, index = self._key_id_pairs[index]
        seq_len = len(self._seq_data[key]["timestamps"])

        if self.dilation == "random":
            dilation = torch.randint(1, 30, (1,)).item()
            left_offset = self._left_offset
            if self.frame_count > 2:
                left_offset = dilation * (self.frame_count // 2)
        else:
            dilation = self.dilation
            left_offset = self._left_offset

        ids = [index] + [max(min(i, seq_len-1), 0) for i in range(index - left_offset, index - left_offset + self.frame_count * dilation, dilation) if i != index]

        imgs = self.load_images(key, ids)
        imgs = [self.process_img(img, color_aug_fn=color_aug_fn) for img in imgs]

        # These poses are camera to world !!
        poses = [self.process_pose(self._seq_data[key]["poses"][i, :, :]) for i in ids]
        projs = [self.process_projs(self._seq_data[key]["intrinsics"][i, :]) for i in ids]

        depths = [np.ones_like(imgs[0][:1, :, :])]

        _proc_time = np.array(time.time() - _start_time)

        data = {
            "imgs": imgs,
            "projs": projs,
            "poses": poses,
            "depths": depths,
            "t__get_item__": np.array([_proc_time])
        }

        return data

    def __len__(self) -> int:
        return self.length
