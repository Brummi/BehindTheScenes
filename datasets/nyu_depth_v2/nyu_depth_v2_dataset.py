
import os

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class NYUDepthV2Dataset(Dataset):
    def __init__(self, data_path, target_image_size):
        super().__init__()
        self.data_path = data_path
        self.target_image_size = target_image_size

        self._data = self._load_data(self.data_path)
        self.length = self._data.get("images").shape[0]

    @staticmethod
    def _load_data(data_path):
        data = h5py.File(os.path.join(data_path, "nyu_depth_v2_labeled.mat"))
        return data

    def process_img(self, img):
        img = np.transpose(img, (2, 1, 0)).astype(np.float32) / 255.
        if self.target_image_size:
            img = cv2.resize(img, (self.target_image_size[1], self.target_image_size[0]), interpolation=cv2.INTER_LINEAR)

        img = np.transpose(img, (2, 0, 1))

        img = img * 2 - 1
        return img

    def process_depth(self, depth):
        depth = np.transpose(depth, (1, 0))
        if self.target_image_size:
            depth = cv2.resize(depth, (self.target_image_size[1], self.target_image_size[0]), interpolation=cv2.INTER_NEAREST)
        depth = depth[None, :, :]
        return depth

    def __getitem__(self, index):
        img = np.array(self._data.get("images")[index])
        depth = np.array(self._data.get("depths")[index])

        img = self.process_img(img)
        depth = self.process_depth(depth)

        poses = torch.eye(4, dtype=torch.float32)
        projs = torch.eye(3, dtype=torch.float32)

        data_dict = {
            "imgs": [img],
            "depths": [depth],
            "poses": [poses],
            "projs": [projs]
        }
        return data_dict

    def __len__(self):
        return self.length