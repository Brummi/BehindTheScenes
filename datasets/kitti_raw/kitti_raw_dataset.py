import os
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

from utils.array_operations import apply_crop
from utils.augmentation import get_color_aug_fn

# This could also be retrieved from
BASE_SIZES = {
    "2011_09_26": (375, 1242),
    "2011_09_28": (370, 1224),
    "2011_09_29": (374, 1238),
    "2011_09_30": (370, 1226),
    "2011_10_03": (376, 1241),
}


class KittiRawDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 pose_path: str,
                 split_path: str,
                 target_image_size=(192, 640),
                 return_stereo=False,
                 return_depth=False,
                 frame_count=2,
                 keyframe_offset=0,
                 dilation=1,
                 keep_aspect_ratio=False,
                 eigen_depth=True,
                 color_aug=False
                 ):
        self.data_path = data_path
        self.pose_path = pose_path
        self.split_path = split_path
        self.target_image_size = target_image_size
        self.return_stereo = return_stereo
        self.return_depth = return_depth
        self.frame_count = frame_count
        self.dilation = dilation
        self.keyframe_offset = keyframe_offset
        self.keep_aspect_ratio = keep_aspect_ratio
        self.eigen_depth = eigen_depth
        self.color_aug = color_aug

        self._sequences = self._get_sequences(self.data_path)
        self._seq_lengths = {(day, seq): length for day, seq, length in self._sequences}

        self._calibs = self._load_calibs(self.data_path, self.target_image_size, keep_aspect_ratio)
        self._poses = self._load_poses(self.pose_path, self._sequences)

        self._datapoints = self._load_split(self.split_path)

        self._left_offset = ((self.frame_count - 1) // 2 + self.keyframe_offset) * self.dilation

        self._skip = 0
        self.length = len(self._datapoints)

    @staticmethod
    def _get_sequences(data_path):
        all_sequences = []

        data_path = Path(data_path)
        for day in data_path.iterdir():
            if not day.is_dir():
                continue
            day_sequences = [seq for seq in day.iterdir() if seq.is_dir()]
            lengths = [len(list((seq / "image_02" / "data").iterdir())) for seq in day_sequences]
            day_sequences = [(day.name, seq.name, length) for seq, length in zip(day_sequences, lengths)]
            all_sequences.extend(day_sequences)

        return all_sequences

    @staticmethod
    def _load_split(split_path):
        with open(split_path, "r") as f:
            lines = f.readlines()

        def split_line(l):
            segments = l.split(" ")
            day, sequence = segments[0].split("/")
            # (day, sequence, id, is_right)
            return day, sequence, int(segments[1]), segments[2] == "r"

        return list(map(split_line, lines))

    @staticmethod
    def _load_calibs(data_path, target_image_size, keep_aspect_ratio):
        calibs = {}

        for day in BASE_SIZES.keys():
            day_folder = Path(data_path) / day
            cam_calib_file = day_folder / "calib_cam_to_cam.txt"
            velo_calib_file = day_folder / "calib_velo_to_cam.txt"

            cam_calib_file_data = {}
            with open(cam_calib_file, 'r') as f:
                for line in f.readlines():
                    key, value = line.split(':', 1)
                    try:
                        cam_calib_file_data[key] = np.array([float(x) for x in value.split()], dtype=np.float32)
                    except ValueError:
                        pass
            velo_calib_file_data = {}
            with open(velo_calib_file, 'r') as f:
                for line in f.readlines():
                    key, value = line.split(':', 1)
                    try:
                        velo_calib_file_data[key] = np.array([float(x) for x in value.split()], dtype=np.float32)
                    except ValueError:
                        pass

            im_size = BASE_SIZES[day]

            # Create 3x4 projection matrices
            P_rect_l = np.reshape(cam_calib_file_data['P_rect_02'], (3, 4))
            P_rect_r = np.reshape(cam_calib_file_data['P_rect_03'], (3, 4))

            R_rect = np.eye(4, dtype=np.float32)
            R_rect[:3, :3] = cam_calib_file_data['R_rect_00'].reshape(3, 3)

            T_v2c = np.hstack((velo_calib_file_data['R'].reshape(3, 3), velo_calib_file_data['T'][..., np.newaxis]))
            T_v2c = np.vstack((T_v2c, np.array([0, 0, 0, 1.0], dtype=np.float32)))

            P_v2cl = P_rect_l @ R_rect @ T_v2c
            P_v2cr = P_rect_r @ R_rect @ T_v2c

            # Compute the rectified extrinsics from cam0 to camN
            T_l = np.eye(4, dtype=np.float32)
            T_l[0, 3] = P_rect_l[0, 3] / P_rect_l[0, 0]
            T_r = np.eye(4, dtype=np.float32)
            T_r[0, 3] = P_rect_r[0, 3] / P_rect_r[0, 0]

            K = P_rect_l[:3, :3]

            if keep_aspect_ratio:
                r_orig = im_size[0] / im_size[1]
                r_target = target_image_size[0] / target_image_size[1]

                if r_orig >= r_target:
                    new_height = r_target * im_size[1]
                    crop_height = im_size[0] - ((im_size[0] - new_height) // 2) * 2
                    box = ((im_size[0] - new_height) // 2, 0, crop_height, int(im_size[1]))

                    c_x = K[0, 2] / im_size[1]
                    c_y = (K[1, 2] - (im_size[0] - new_height) / 2) / new_height

                    rescale = im_size[1] / target_image_size[1]

                else:
                    new_width = im_size[0] / r_target
                    crop_width = im_size[1] - ((im_size[1] - new_width) // 2) * 2
                    box = (0, (im_size[1] - new_width) // 2, im_size[0], crop_width)

                    c_x = (K[0, 2] - (im_size[1] - new_width) / 2) / new_width
                    c_y = K[1, 2] / im_size[0]

                    rescale = im_size[0] / target_image_size[0]

                f_x = (K[0, 0] / target_image_size[1]) / rescale
                f_y = (K[1, 1] / target_image_size[0]) / rescale

                box = tuple([int(x) for x in box])

            else:
                f_x = K[0, 0] / im_size[1]
                f_y = K[1, 1] / im_size[0]

                c_x = K[0, 2] / im_size[1]
                c_y = K[1, 2] / im_size[0]

                box = None

            # Replace old K with new K
            K[0, 0] = f_x * 2.
            K[1, 1] = f_y * 2.
            K[0, 2] = c_x * 2 - 1
            K[1, 2] = c_y * 2 - 1

            # Invert to get camera to center transformation, not center to camera
            T_r = np.linalg.inv(T_r)
            T_l = np.linalg.inv(T_l)

            calibs[day] = {
                "K": K,
                "T_l": T_l,
                "T_r": T_r,
                "P_v2cl": P_v2cl,
                "P_v2cr": P_v2cr,
                "crop": box
            }

        return calibs

    @staticmethod
    def _load_poses(pose_path, sequences):
        poses = {}

        for day, seq, _ in sequences:
            pose_file = Path(pose_path) / day / f"{seq}.txt"

            poses_seq = []
            try:
                with open(pose_file, 'r') as f:
                    lines = f.readlines()

                    for line in lines:
                        T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                        T_w_cam0 = T_w_cam0.reshape(3, 4)
                        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                        poses_seq.append(T_w_cam0)

            except FileNotFoundError:
                print(f'Ground truth poses are not avaialble for sequence {seq}.')

            poses_seq = np.array(poses_seq, dtype=np.float32)

            poses[(day, seq)] = poses_seq
        return poses

    def load_images(self, day, seq, ids, load_left, load_right):
        imgs_left = []
        imgs_right = []

        for id in ids:
            if load_left:
                img = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, day, seq, "image_02", "data", f"{id:010d}.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                imgs_left += [img]

            if load_right:
                img = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, day, seq, "image_03", "data", f"{id:010d}.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                imgs_right += [img]

        return imgs_left, imgs_right

    def process_img(self, img: np.array, crop_box=None, color_aug_fn=None):
        if crop_box:
            img = apply_crop(img, crop_box)
        if self.target_image_size:
            img = cv2.resize(img, (self.target_image_size[1], self.target_image_size[0]), interpolation=cv2.INTER_LINEAR)
        img = np.transpose(img, (2, 0, 1))

        if color_aug_fn is not None:
            img = color_aug_fn(torch.tensor(img))

        img = img * 2 - 1
        return img

    def load_depth(self, day, seq, id, P):
        size = BASE_SIZES[day]

        points = np.fromfile(os.path.join(self.data_path, day, seq, "velodyne_points", "data", f"{id:010d}.bin"), dtype=np.float32).reshape(-1, 4)
        points[:, 3] = 1.0

        points = points[points[:, 0] >= 0, :]

        # project the points to the camera
        velo_pts_im = np.dot(P, points.T).T
        velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

        # Use velodyne depth instead of reprojected depth
        # TODO: Consider
        # velo_pts_im[:, 2] = points[:, 0]

        # check if in bounds
        # use minus 1 to get the exact same value as KITTI matlab code
        velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
        velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
        val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
        val_inds = val_inds & (velo_pts_im[:, 0] < size[1]) & (velo_pts_im[:, 1] < size[0])
        velo_pts_im = velo_pts_im[val_inds, :]

        # project to image
        depth = np.zeros(size)
        depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

        # find the duplicate points and choose the closest depth
        inds = velo_pts_im[:, 1] * (size[1] - 1) + velo_pts_im[:, 0] - 1
        dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
        for dd in dupe_inds:
            pts = np.where(inds == dd)[0]
            x_loc = int(velo_pts_im[pts[0], 0])
            y_loc = int(velo_pts_im[pts[0], 1])
            depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
        depth[depth < 0] = 0

        if self.eigen_depth:
            mask = np.logical_and(depth > 1e-3, depth < 80)
            crop = np.array([0.40810811 * size[0], 0.99189189 * size[0], 0.03594771 * size[1], 0.96405229 * size[1]]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
            depth[~mask] = 0

        return depth[None, :, :]

    def __getitem__(self, index: int):
        _start_time = time.time()

        if index >= self.length:
            raise IndexError()

        if self._skip != 0:
            index += self._skip

        day, sequence, seq_id, is_right = self._datapoints[index]
        seq_len = self._seq_lengths[(day, sequence)]

        load_left = (not is_right) or self.return_stereo
        load_right = is_right or self.return_stereo

        calibs = self._calibs[day]

        ids = [seq_id] + [max(min(i, seq_len-1), 0) for i in range(seq_id - self._left_offset, seq_id - self._left_offset + self.frame_count * self.dilation, self.dilation) if i != seq_id]

        if self.color_aug:
            color_aug_fn = get_color_aug_fn(ColorJitter.get_params(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1, 0.1)))
        else:
            color_aug_fn = None
        imgs_left, imgs_right = self.load_images(day, sequence, ids, load_left, load_right)

        imgs_left = [self.process_img(img, calibs["crop"], color_aug_fn=color_aug_fn) for img in imgs_left]
        imgs_right = [self.process_img(img, calibs["crop"], color_aug_fn=color_aug_fn) for img in imgs_right]

        # These poses are camera to world !!
        poses_left = [self._poses[(day, sequence)][i, :, :] @ calibs["T_l"] for i in ids] if load_left else []
        poses_right = [self._poses[(day, sequence)][i, :, :] @ calibs["T_r"] for i in ids] if load_right else []

        projs_left = [calibs["K"] for _ in ids] if load_left else []
        projs_right = [calibs["K"] for _ in ids] if load_right else []

        imgs = imgs_left + imgs_right if not is_right else imgs_right + imgs_left
        projs = projs_left + projs_right if not is_right else projs_right + projs_left
        poses = poses_left + poses_right if not is_right else poses_right + poses_left

        if self.return_depth:
            depths = [self.load_depth(day, sequence, ids[0], calibs["P_v2cl" if not is_right else "P_v2cr"])]
        else:
            depths = []

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
