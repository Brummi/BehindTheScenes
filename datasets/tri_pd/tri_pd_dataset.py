import json
import os
import random
import sys
import time
from logging import Logger
from typing import Optional

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
import math
from scipy.spatial import transform

banned_scenes = ['scene_000100','scene_000002','scene_000008','scene_000012','scene_000018','scene_000029',
'scene_000038','scene_000040','scene_000043','scene_000044','scene_000049','scene_000050','scene_000053','scene_000063',
'scene_000079','scene_000090','scene_000094','scene_000100','scene_000103','scene_000106','scene_000111','scene_000112',
'scene_000124','scene_000125','scene_000127','scene_000148','scene_000159','scene_000166','scene_000169',
'scene_000170','scene_000171','scene_000187', 'scene_000191','scene_000200','scene_000202','scene_000217',
'scene_000218','scene_000225','scene_000229','scene_000232','scene_000236','scene_000237','scene_000245',
'scene_000249', "scene_000196", "scene_000148", "scene_000156"
]

BASE_SIZE = (1216, 1936)


class TRIPDDataset(Dataset):
    def __init__(self, split='train', root=None, supervision='moving', frame_count=5, random_select=True, random_select_factor=1, load_flow=False, load_segs=True, logger: Optional[Logger]=None):
        super(TRIPDDataset, self).__init__()
        self.root_dir = root
        self.frame_count = frame_count
        self.random_select = random_select
        self.random_select_factor = random_select_factor
        self.load_flow = load_flow
        self.load_segs = load_segs
        self.logger = logger

        self.files = os.listdir(self.root_dir)
        self.files.sort()
        if split == 'train':
            self.files = self.files[1:]
        elif split == 'eval':
            self.files = self.files[0:1]
        else:
            self.files = self.files
        self.annotation = None 
        if supervision == 'moving':
            self.annotation = 'moving_masks'
        elif supervision == 'all':
            self.annotation = 'ari_masks'
        else:
            raise ValueError("Need to choose either moving masks, or all masks.")

        self.real_files = []
        self.mask_files = []
        self.flow_files = []
        self.calibrations = []
        for f in self.files:
            if f in banned_scenes or not f.startswith("scene"):
                continue

            calib_path = os.path.join(self.root_dir, f + "/calibration")
            calib_path = os.path.join(calib_path, os.listdir(calib_path)[0])
            with open(calib_path) as calib_file:
                calib_data = json.load(calib_file)
            for i in [1, 5, 6, 7, 8, 9]:
                if os.path.exists(os.path.join(self.root_dir, f + f'/rgb/camera_0{i}')):
                    self.real_files.append(f + f'/rgb/camera_0{i}')
                    self.mask_files.append(f + f'/{self.annotation}/camera_0{i}')
                    self.flow_files.append(f + f'/motion_vectors_2d/camera_0{i}')
                local_pose, projection_mat = self.get_camera(calib_data, i)
                self.calibrations += [{"local_pose": local_pose, "projection_mat": projection_mat}]
        self.img_transform = transforms.Compose([
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def log(self, msg, mode="info"):
        if self.logger is not None:
            if mode == "debug":
                self.logger.debug(msg)
            elif mode == "error":
                self.logger.error(msg)
            else:
                self.logger.info(msg)
        else:
            if mode == "error":
                out = sys.stderr
            else:
                out = sys.stdout
            print(msg, file=out)

    @staticmethod
    def get_image_name(i):
        i += 1
        i *= 5
        return f"{i:018d}.png"

    @staticmethod
    def get_camera(calib_data, i):
        cam_idx = calib_data["names"].index(f"camera_0{i}")
        extrinsics = calib_data["extrinsics"][cam_idx]
        intrinsics = calib_data["intrinsics"][cam_idx]

        rot_mat = transform.Rotation.from_quat(list(extrinsics["rotation"].values())).as_matrix()
        local_pose = np.concatenate([rot_mat, np.array(list(extrinsics["translation"].values())).reshape((-1, 1))], axis=-1)

        projection_matrix = np.eye(4)
        projection_matrix[0, 0] = intrinsics["fx"] * 2 / BASE_SIZE[1]
        projection_matrix[1, 1] = intrinsics["fy"] * 2 / BASE_SIZE[0]
        projection_matrix[0, 2] = intrinsics["cx"] / BASE_SIZE[1] - .5
        projection_matrix[1, 2] = intrinsics["cy"] / BASE_SIZE[0] - .5

        return local_pose, projection_matrix

    def load_files(self, real_idx, path, mask_path=None, flow_path=None):
        ims = [cv2.imread(os.path.join(os.path.join(self.root_dir, path), self.get_image_name(idd))) for idd in real_idx]
        if self.load_segs and mask_path is not None:
            segs = [cv2.imread(os.path.join(os.path.join(self.root_dir, mask_path), self.get_image_name(idd)), -1) for idd in real_idx]
        else:
            segs = []
        if self.load_flow and flow_path is not None:
            flows = [cv2.imread(os.path.join(os.path.join(self.root_dir, flow_path), self.get_image_name(idd)), -1) for idd in real_idx]
        else:
            flows = []
        return ims, segs, flows

    def __getitem__(self, index):
        _start_time = time.time()

        if self.random_select:
            rand_id = random.randint(0, 190) + 1
            real_idx = [rand_id + j for j in range(self.frame_count)]
            index %= len(self.real_files)
        else:
            offset = index % 190
            index = index // 190
            real_idx = [offset + j for j in range(self.frame_count)]

        path = self.real_files[index]
        mask_path = self.mask_files[index]
        flow_path = self.flow_files[index]

        ims, segs, flows = self.load_files(real_idx, path, mask_path, flow_path)

        if any(im is None for im in ims) or any(seg is None for seg in segs) or any(flow is None for flow in flows):
            self.log(f"Sample around {str(path)} is broken. Skipping.", mode="error")
            return self.__getitem__((index + 1) % len(self))

        projs = []
        poses = []

        for i in range(len(ims)):
            image = ims[i]
            downsampling_ratio = 0.58
            crop = 158
            width = int(math.ceil(image.shape[1] * downsampling_ratio))
            height = int(math.ceil(image.shape[0] * downsampling_ratio))
            dim = (width, height)

            image = cv2.resize(image, dim, interpolation = cv2.INTER_LINEAR)
            image = image[crop:, :, :]
            image = torch.Tensor(image).float()
            image = image / 255.0
            image = image.permute(2, 0, 1)
            image = self.img_transform(image)
            ims[i] = image

        for i in range(len(segs)):
            seg = segs[i]
            seg = cv2.resize(seg, dim, interpolation = cv2.INTER_NEAREST)
            seg = seg[crop:,:]

            values, indices, counts = np.unique(seg, return_inverse=True, return_counts=True)
            to_eliminate = counts <= 50
            mapping = np.arange(len(values))
            mapping[to_eliminate] = 0
            _h, _w = seg.shape
            seg = mapping[indices].reshape((_h, _w))

            seg = seg.astype(np.int)
            segs[i] = seg

        for i in range(len(flows)):
            flow = flows[i]
            flow = cv2.resize(flow, dim, interpolation=cv2.INTER_NEAREST)
            flow = flow[crop:, :]
            flow = flow.astype(np.float)
            flows[i] = flow
            # TODO redo

        for i in range(self.frame_count):
            projs.append(self.calibrations[index]["projection_mat"])
            poses.append(self.calibrations[index]["local_pose"])

        _proc_time = np.array(time.time() - _start_time)

        data = {
            'imgs': ims,
            'segs': segs,
            "flows": flows,
            "projs": projs,
            "poses": poses,
            "t__get_item__": _proc_time
        }

        return data

    def __len__(self):
        factor = self.random_select_factor if self.random_select else 190
        return len(self.real_files) * factor
