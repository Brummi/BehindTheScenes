import os
import glob
import numpy as np
import cv2

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision.transforms import transforms

from utils.array_operations import apply_crop

ID_TO_CLASS = {
    0: "unlabeled",
    1: "ambiguous",
    2: "sky",
    3: "road",
    4: "sidewalk",
    5: "railtrack",
    6: "terrain",
    7: "tree",
    8: "vegetation",
    9: "building",
    10: "infrastructure",
    11: "fence",
    12: "billboard",
    13: "trafficlight",
    14: "trafficsign",
    15: "mobilebarrier",
    16: "firehydrant",
    17: "chair",
    18: "trash",
    19: "trashcan",
    20: "person",
    21: "animal",
    22: "bicycle",
    23: "motorcycle",
    24: "car",
    25: "van",
    26: "bus",
    27: "truck",
    28: "trailer",
    29: "train",
    30: "plane",
    31: "boat",
}

DYNAMIC_CLASS_IDS = np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])

BASE_SIZE = (1080, 1920)
BASE_CROP = (180, 0, 720, 1920)


class ViperDataset(Dataset):
    def __init__(self, base_path: str, frame_count=2, every_nth=10, target_size=(128, 192), load_flow=True, load_segs=True) -> None:
        super().__init__()

        self.base_path = base_path
        self.frame_count = frame_count
        self.every_nth = every_nth
        self.target_size = target_size
        self.load_flow = load_flow
        self.load_segs = load_segs

        self.sequences = [p for p in sorted(os.listdir(os.path.join(self.base_path, "img")))]

        self.img_paths = [[p for p in sorted(glob.glob(os.path.join(self.base_path, "img", s, "*.jpg")))] for s in self.sequences]

        self.flow_paths = [[p for p in sorted(glob.glob(os.path.join(self.base_path, "flow", s, "*.npz")))] for s in self.sequences]

        self.segmentation_paths = [[p for p in sorted(glob.glob(os.path.join(self.base_path, "inst", s, "*.png")))] for s in self.sequences]

        self.projection_mats, self.view_mats = self.load_cameras()

        self.sequence_lengths = [(len(seq_img_paths) - frame_count + 1) // self.every_nth for seq_img_paths in self.img_paths]

        self.crop, cam_scale_mat = self.precompute_crop()
        self.projection_mats = [p_mat @ cam_scale_mat for p_mat in self.projection_mats]

        self.img_transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def load_cameras(self):
        camera_path = os.path.join(self.base_path, "camera")

        projection_mats = []
        view_mats = []

        for s in self.sequences:
            camera_raw = np.loadtxt(os.path.join(camera_path, f"{s}.csv"), delimiter=",", skiprows=1)
            projection_mats_seq = np.reshape(camera_raw[:, 1:17], (-1, 4, 4))
            view_mats_seq = np.reshape(camera_raw[:, 17:], (-1, 4, 4))

            projection_mats += [projection_mats_seq]
            view_mats += [view_mats_seq]

        return projection_mats, view_mats

    def precompute_crop(self):
        h_ratio = self.target_size[0] / BASE_CROP[2]
        w_ratio = self.target_size[1] / BASE_CROP[3]

        if h_ratio < w_ratio:
            h_new = int(self.target_size[0] / w_ratio)
            w_new = BASE_CROP[3]
            crop = ((BASE_CROP[2] - h_new) // 2, 0, h_new, w_new)
        else:
            h_new = BASE_CROP[2]
            w_new = int(self.target_size[1] / h_ratio)
            crop = (0, (BASE_CROP[3] - w_new) // 2, h_new, w_new)

        cam_y_scale = BASE_SIZE[0] / h_new
        cam_x_scale = BASE_SIZE[1] / w_new

        cam_scale_mat = np.eye(4)
        cam_scale_mat[0, 0] = cam_x_scale
        cam_scale_mat[1, 1] = cam_y_scale

        # Crop (y, x, height, width)
        return crop, cam_scale_mat

    def load_files(self, seq_index, index):
        imgs = []
        segs = []

        for i in range(self.frame_count):
            img = cv2.cvtColor(cv2.imread(self.img_paths[seq_index][index * self.every_nth + i]), cv2.COLOR_BGR2RGB).astype(float) / 255
            imgs += [img]

            if self.load_segs:
                seg = cv2.cvtColor(cv2.imread(self.segmentation_paths[seq_index][index * self.every_nth + i]), cv2.COLOR_BGR2RGB)
                segs += [seg]

        if self.load_flow:
            flow = np.load(self.flow_paths[seq_index][index])
            flow = np.stack((flow["u"], flow["v"]), axis=-1)
            flow[np.isnan(flow)] = 0
            flows = [flow]
        else:
            flows = []

        data = {
            "imgs": imgs,
            "segs": segs,
            "flows": flows
        }

        return data

    def crop_and_scale(self, data):
        data = dict(data)
        w_h = (self.target_size[1], self.target_size[0])
        data["imgs"] = [cv2.resize(apply_crop(apply_crop(img, BASE_CROP), self.crop), w_h, interpolation=cv2.INTER_LINEAR) for img in data["imgs"]]
        data["segs"] = [cv2.resize(apply_crop(apply_crop(seg, BASE_CROP), self.crop), w_h, interpolation=cv2.INTER_NEAREST) for seg in data["segs"]]
        data["flows"] = [cv2.resize(apply_crop(apply_crop(np.concatenate((flow, np.zeros((BASE_SIZE[0], BASE_SIZE[1], 1))), axis=-1), BASE_CROP), self.crop), w_h, interpolation=cv2.INTER_NEAREST)[:, :, :2] for flow in data["flows"]]

        return data

    def process_segs(self, data):
        data = dict(data)
        segs = []
        for seg in data["segs"]:
            cls = np.any(seg[:, :, :1] == DYNAMIC_CLASS_IDS, axis=-1).astype(int)
            iid = seg[:, :, 1] * 256 + seg[:, :, 2]
            seg = np.stack((cls, iid), axis=-1)
            segs += [seg]
        data["segs"] = segs
        return data

    def __getitem__(self, index) -> T_co:
        if index >= len(self):
            raise IndexError
        seq_index = 0
        while seq_index < len(self.sequence_lengths) and index >= self.sequence_lengths[seq_index]:
            index -= self.sequence_lengths[seq_index]
            seq_index += 1

        data = self.load_files(seq_index, index)
        data = self.crop_and_scale(data)
        data = self.process_segs(data)

        # put everything into the right format
        data["imgs"] = [np.transpose(img, (2, 0, 1)) for img in data["imgs"]]
        data["segs"] = [np.transpose(seg, (2, 0, 1)) for seg in data["segs"]]
        data["flows"] = [np.transpose(flow, (2, 0, 1)) for flow in data["flows"]]

        data["projs"] = [self.projection_mats[seq_index][index + i] for i in range(self.frame_count)]
        data["poses"] = [self.view_mats[seq_index][index + i] for i in range(self.frame_count)]

        data["meta-data"] = np.array([seq_index, index])

        # transformations
        data["imgs"] = [self.img_transform(img) for img in data["imgs"]]

        return data

    def __len__(self):
        return sum(self.sequence_lengths)