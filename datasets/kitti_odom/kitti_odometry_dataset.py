import os
import time

import cv2
import numpy as np
from torch.utils.data import Dataset

from utils.array_operations import apply_crop

BASE_SIZES = {
    "00": (376, 1241),
    "01": (376, 1241),
    "02": (376, 1241),
    "03": (375, 1242),
    "04": (370, 1226),
    "05": (370, 1226),
    "06": (370, 1226),
    "07": (370, 1226),
    "08": (370, 1226),
    "09": (370, 1226),
    "10": (370, 1226),
    "11": (370, 1226),
    "12": (370, 1226),
    "13": (376, 1241),
    "14": (376, 1241),
    "15": (376, 1241),
    "16": (376, 1241),
    "17": (376, 1241),
    "18": (376, 1241),
    "19": (376, 1241),
    "20": (376, 1241),
    "21": (376, 1241),
}


class KittiOdometryDataset(Dataset):
    def __init__(self, base_path: str, frame_count=2, sequences=None, target_image_size=(256, 512), dilation=1, return_stereo=False, custom_pose_path=None, keyframe_offset=0):
        """
        Dataset implementation for KITTI Odometry.
        :param dataset_dir: Top level folder for KITTI Odometry (should contain folders sequences, poses, poses_dso (if available)
        :param frame_count: Number of frames used per sample (excluding the keyframe). By default, the keyframe is in the middle of those frames. (Default=2)
        :param sequences: Which sequences to use. Should be tuple of strings, e.g. ("00", "01", ...)
        :param depth_folder: The folder within the sequence folder that contains the depth information (e.g. sequences/00/{depth_folder})
        :param target_image_size: Desired image size (correct processing of depths is only guaranteed for default value). (Default=(256, 512))
        :param max_length: Maximum length per sequence. Useful for splitting up sequences and testing. (Default=None)
        :param dilation: Spacing between the frames (Default 1)
        :param offset_d: Index offset for frames (offset_d=0 means keyframe is centered). (Default=0)
        :param use_color: Use color (camera 2) or greyscale (camera 0) images (default=True)
        :param use_dso_poses: Use poses provided by d(v)so instead of KITTI poses. Requires poses_dso folder. (Default=True)
        :param use_color_augmentation: Use color jitter augmentation. The same transformation is applied to all frames in a sample. (Default=False)
        :param lidar_depth: Use depth information from (annotated) velodyne data. (Default=False)
        :param dso_depth: Use depth information from d(v)so. (Default=True)
        :param annotated_lidar: If lidar_depth=True, then this determines whether to use annotated or non-annotated depth maps. (Default=True)
        :param return_stereo: Return additional stereo frame. Only used during training. (Default=False)
        :param return_mvobj_mask: Return additional moving object mask. Only used during training. If return_mvobj_mask=2, then the mask is returned as target instead of the depthmap. (Default=False)
        :param use_index_mask: Use the listed index masks (if a sample is listed in one of the masks, it is not used). (Default=())
        """
        self.dataset_dir = base_path
        self.frame_count = frame_count
        self.sequences = sequences
        self.target_image_size = target_image_size
        self.dilation = dilation
        self.return_stereo = return_stereo
        self.custom_pose_path = custom_pose_path
        self.keyframe_offset = keyframe_offset

        if self.sequences is None:
            self.sequences = [f"{i:02d}" for i in range(11)]
        self._sequence_folders = [os.path.join(self.dataset_dir, "sequences", sequence) for sequence in self.sequences]
        self._sequences_files_cam2 = [list(sorted(os.listdir(os.path.join(sequence_folder, "image_2")))) for sequence_folder in self._sequence_folders]
        self._sequences_files_cam3 = [list(sorted(os.listdir(os.path.join(sequence_folder, "image_3")))) for sequence_folder in self._sequence_folders]

        extra_frames = frame_count * dilation
        self._sequence_lengths = [len(sequence_files_cam2) - extra_frames for sequence_files_cam2 in self._sequences_files_cam2]
        self._calibs = self._load_calibs(self._sequence_folders, self.target_image_size)
        self._poses = self._load_poses(self.dataset_dir, self.sequences, self.custom_pose_path)

        self._keyframe_idx = self.frame_count // 2 + self.keyframe_offset
        assert 0 <= self._keyframe_idx < self.frame_count

        self.length = sum(self._sequence_lengths)

        self._skip = 0

    @staticmethod
    def _load_calibs(sequence_folders, target_image_size):
        calibs = []
        for seq_folder in sequence_folders:
            seq = seq_folder[-2:]
            im_size = BASE_SIZES[seq]

            calib_file = os.path.join(seq_folder, "calib.txt")
            calib_file_data = {}
            with open(calib_file, 'r') as f:
                for line in f.readlines():
                    key, value = line.split(':', 1)
                    try:
                        calib_file_data[key] = np.array([float(x) for x in value.split()], dtype=np.float32)
                    except ValueError:
                        pass
            # Create 3x4 projection matrices
            P_rect_20 = np.reshape(calib_file_data['P2'], (3, 4))
            P_rect_30 = np.reshape(calib_file_data['P3'], (3, 4))

            # Compute the rectified extrinsics from cam0 to camN
            T_0 = np.eye(4, dtype=np.float32)
            T_0[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
            T_1 = np.eye(4, dtype=np.float32)
            T_1[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

            # Poses are centered around the left camera
            # T_0 = np.linalg.inv(T_0)
            T_1 = np.linalg.inv(T_1) @ T_0
            T_0 = np.eye(4, dtype=np.float32)

            K = P_rect_20[:3, :3]

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

            f_x = K[0, 0] / target_image_size[1] / rescale
            f_y = K[1, 1] / target_image_size[0] / rescale

            box = tuple([int(x) for x in box])

            # Replace old K with new K
            K[0, 0] = f_x * 2.
            K[1, 1] = f_y * 2.
            K[0, 2] = c_x * 2 - 1
            K[1, 2] = c_y * 2 - 1

            calibs.append({
                "K": K,
                "T_0": T_0,
                "T_1": T_1,
                "crop": box
            })

        return calibs

    @staticmethod
    def _load_poses(dataset_dir, sequences, custom_pose_path=None):
        if custom_pose_path is None:
            pose_path = os.path.join(dataset_dir, "poses")
        else:
            pose_path = custom_pose_path
        poses = []
        for seq in sequences:
            pose_file = os.path.join(pose_path, seq + '.txt')

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

            poses.append(poses_seq)
        return poses

    def get_sequence_index(self, index: int):
        for dataset_index, dataset_size in enumerate(self._sequence_lengths):
            if index >= dataset_size:
                index = index - dataset_size
            else:
                return dataset_index, index
        return None, None

    def get_index(self, sequence, index):
        for i in range(len(self.sequences)):
            if int(self.sequences[i]) != sequence:
                index += self._sequence_lengths[i]
            else:
                break
        return index

    def load_files(self, seq, ids, load_stereo):
        imgs = []
        imgs_stereo = []

        seq_folder = self._sequence_folders[seq]

        for id in ids:
            img = cv2.cvtColor(cv2.imread(os.path.join(seq_folder, "image_2", self._sequences_files_cam2[seq][id])), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            imgs += [img]

            if load_stereo:
                img = cv2.cvtColor(cv2.imread(os.path.join(seq_folder, "image_3", self._sequences_files_cam3[seq][id])), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                imgs_stereo += [img]

        return imgs, imgs_stereo

    def process_img(self, img: np.array, crop_box=None):
        if crop_box:
            img = apply_crop(img, crop_box)
        if self.target_image_size:
            img = cv2.resize(img, (self.target_image_size[1], self.target_image_size[0]), interpolation=cv2.INTER_LINEAR)
        img = np.transpose(img, (2, 0, 1)) * 2 - 1
        return img

    def __getitem__(self, index: int):
        _start_time = time.time()

        if self._skip > 0:
            index += self._skip

        sequence_index, index = self.get_sequence_index(index)
        if sequence_index is None:
            raise IndexError()

        calibs = self._calibs[sequence_index]

        ids = [index + i * self.dilation for i in range(self.frame_count)]

        imgs, imgs_stereo = self.load_files(sequence_index, ids, self.return_stereo)

        imgs = [self.process_img(img, calibs["crop"]) for img in imgs]
        imgs_stereo = [self.process_img(img, calibs["crop"]) for img in imgs_stereo]

        # These poses are camera to world !!
        poses = [self._poses[sequence_index][i, :, :] @ calibs["T_0"] for i in ids]
        poses_stereo = [self._poses[sequence_index][i, :, :] @ calibs["T_1"] for i in ids] if self.return_stereo else []

        projs = [calibs["K"] for _ in ids]
        projs_stereo = list(projs) if self.return_stereo else []

        imgs = [imgs[self._keyframe_idx]] + imgs[:self._keyframe_idx] + imgs[self._keyframe_idx+1:]
        poses = [poses[self._keyframe_idx]] + poses[:self._keyframe_idx] + poses[self._keyframe_idx+1:]
        projs = [projs[self._keyframe_idx]] + projs[:self._keyframe_idx] + projs[self._keyframe_idx+1:]

        if self.return_stereo:
            imgs_stereo = [imgs_stereo[self._keyframe_idx]] + imgs_stereo[:self._keyframe_idx] + imgs_stereo[self._keyframe_idx + 1:]
            poses_stereo = [poses_stereo[self._keyframe_idx]] + poses_stereo[:self._keyframe_idx] + poses_stereo[self._keyframe_idx + 1:]
            projs_stereo = [projs_stereo[self._keyframe_idx]] + projs_stereo[:self._keyframe_idx] + projs_stereo[self._keyframe_idx + 1:]

        _proc_time = np.array(time.time() - _start_time)

        data = {
            "imgs": imgs + imgs_stereo,
            "projs": projs + projs_stereo,
            "poses": poses + poses_stereo,
            "sequence": np.array([sequence_index], np.int32),
            "ids": np.array(ids, np.int32),
            "t__get_item__": np.array([_proc_time])
        }

        return data

    def __len__(self) -> int:
        return self.length