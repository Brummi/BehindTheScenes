import os

from datasets.kitti_360.kitti_360_dataset import Kitti360Dataset
from datasets.kitti_odom.kitti_odometry_dataset import KittiOdometryDataset
from datasets.kitti_raw.kitti_raw_dataset import KittiRawDataset
from datasets.nyu_depth_v2.nyu_depth_v2_dataset import NYUDepthV2Dataset
from datasets.realestate10k.realestate10k_dataset import RealEstate10kDataset
from datasets.waymo.waymo_dataset import WaymoDataset


def make_datasets(config):
    type = config.get("type", "KITTI_Raw")
    if type == "KITTI_Odometry":
        train_dataset = KittiOdometryDataset(
            base_path=config["data_path"],
            frame_count=config.get("data_fc", 1),
            target_image_size=config.get("image_size", (128, 256)),
            return_stereo=config.get("data_stereo", False),
            sequences=config.get("train_sequences", ("00",)),
            custom_pose_path=config.get("custom_pose_path", None),
            keyframe_offset=0 #-(config.get("data_fc", 1) // 2)
        )
        test_dataset = KittiOdometryDataset(
            base_path=config["data_path"],
            frame_count=config.get("data_fc", 1),
            target_image_size=config.get("image_size", (128, 256)),
            return_stereo=config.get("data_stereo", False),
            sequences=config.get("val_sequences", ("00",)),
            custom_pose_path=config.get("custom_pose_path", None),
            keyframe_offset=0 #-(config.get("data_fc", 1) // 2)
        )
        return train_dataset, test_dataset

    elif type == "KITTI_Raw":
        train_dataset = KittiRawDataset(
            data_path=config["data_path"],
            pose_path=config["pose_path"],
            split_path=os.path.join(config["split_path"], "train_files.txt"),
            target_image_size=config.get("image_size", (192, 640)),
            frame_count=config.get("data_fc", 1),
            return_stereo=config.get("data_stereo", False),
            keyframe_offset=config.get("keyframe_offset", 0),
            dilation=config.get("dilation", 1),
            color_aug=config.get("color_aug", False)
        )
        test_dataset = KittiRawDataset(
            data_path=config["data_path"],
            pose_path=config["pose_path"],
            split_path=os.path.join(config["split_path"], "val_files.txt"),
            target_image_size=config.get("image_size", (192, 640)),
            frame_count=config.get("data_fc", 1),
            return_stereo=config.get("data_stereo", False),
            keyframe_offset=config.get("keyframe_offset", 0),
            dilation=config.get("dilation", 1),
        )
        return train_dataset, test_dataset

    elif type == "KITTI_360":
        if config.get("split_path", None) is None:
            train_split_path = None
            test_split_path = None
        else:
            train_split_path = os.path.join(config["split_path"], "train_files.txt")
            test_split_path = os.path.join(config["split_path"], "val_files.txt")

        train_dataset = Kitti360Dataset(
            data_path=config["data_path"],
            pose_path=config["pose_path"],
            split_path=train_split_path,
            target_image_size=tuple(config.get("image_size", (192, 640))),
            frame_count=config.get("data_fc", 3),
            return_stereo=config.get("data_stereo", True),
            return_fisheye=config.get("data_fisheye", True),
            return_3d_bboxes=config.get("data_3d_bboxes", False),
            return_segmentation=config.get("data_segmentation", False),
            keyframe_offset=config.get("keyframe_offset", 0),
            dilation=config.get("dilation", 1),
            fisheye_rotation=config.get("fisheye_rotation", 0),
            fisheye_offset=config.get("fisheye_offset", 1),
            color_aug=config.get("color_aug", False),
            is_preprocessed=config.get("is_preprocessed", False)
        )
        test_dataset = Kitti360Dataset(
            data_path=config["data_path"],
            pose_path=config["pose_path"],
            split_path=test_split_path,
            target_image_size=tuple(config.get("image_size", (192, 640))),
            frame_count=config.get("data_fc", 3),
            return_stereo=config.get("data_stereo", True),
            return_fisheye=config.get("data_fisheye", True),
            return_3d_bboxes=config.get("data_3d_bboxes", False),
            return_segmentation=config.get("data_segmentation", False),
            keyframe_offset=config.get("keyframe_offset", 0),
            fisheye_rotation=config.get("fisheye_rotation", 0),
            fisheye_offset=config.get("fisheye_offset", 1),
            dilation=config.get("dilation", 1),
            is_preprocessed=config.get("is_preprocessed", False)
        )
        return train_dataset, test_dataset

    elif type == "RealEstate10k":
        train_dataset = RealEstate10kDataset(
            data_path=config["data_path"],
            split_path=None,
            target_image_size=config.get("image_size", (256, 384)),
            frame_count=config.get("data_fc", 2),
            keyframe_offset=0, #-(config.get("data_fc", 1) // 2),
            dilation=config.get("dilation", 10),
            color_aug=config.get("color_aug", False)
        )
        test_dataset = RealEstate10kDataset(
            data_path=config["data_path"],
            split_path=os.path.join(config["split_path"], "val_files.txt"),
            target_image_size=config.get("image_size", (256, 384)),
            frame_count=config.get("data_fc", 2),
            keyframe_offset=0, #-(config.get("data_fc", 1) // 2),
            dilation=config.get("dilation", 10),
            color_aug=False
        )
        return train_dataset, test_dataset

    elif type == "Waymo":
        if config.get("split_path", None) is None:
            train_split_path = None
            test_split_path = None
        else:
            train_split_path = os.path.join(config["split_path"], "train_files.txt")
            test_split_path = os.path.join(config["split_path"], "val_files.txt")

        train_dataset = WaymoDataset(
            data_path=config["data_path"],
            mode="training",
            split_path=train_split_path,
            target_image_size=tuple(config.get("image_size", (320, 480))),
            frame_count=config.get("data_fc", 2),
            keyframe_offset=config.get("keyframe_offset", 0),
            return_45=config.get("return_45", True),
            return_90=config.get("return_90", True),
            offset_45=config.get("offset_45", 5),
            offset_90=config.get("offset_90", 10),
            dilation=config.get("dilation", 1),
            color_aug=config.get("color_aug", True),
            correct_exposure=config.get("correct_exposure", True),
        )
        test_dataset = WaymoDataset(
            data_path=config["data_path"],
            mode="validation",
            split_path=test_split_path,
            target_image_size=tuple(config.get("image_size", (320, 480))),
            frame_count=config.get("data_fc", 2),
            keyframe_offset=config.get("keyframe_offset", 0),
            return_45=config.get("return_45", True),
            return_90=config.get("return_90", True),
            offset_45=config.get("offset_45", 5),
            offset_90=config.get("offset_90", 10),
            dilation=config.get("dilation", 1),
            color_aug=False,
            return_depth=True,
            correct_exposure=config.get("correct_exposure", True),
        )
        return train_dataset, test_dataset

    else:
        raise NotImplementedError(f"Unsupported dataset type: {type}")


def make_test_dataset(config):
    type = config.get("type", "KITTI_Raw")
    if type == "KITTI_Raw":
        test_dataset = KittiRawDataset(
            data_path=config["data_path"],
            pose_path=config["pose_path"],
            split_path=os.path.join(config["split_path"], "test_files.txt"),
            target_image_size=config.get("image_size", (192, 640)),
            return_depth=True,
            frame_count=1,
            return_stereo=config.get("data_stereo", False),
            keyframe_offset=0
        )
        return test_dataset
    elif type == "KITTI_360":
        test_dataset = Kitti360Dataset(
            data_path=config["data_path"],
            pose_path=config["pose_path"],
            split_path=os.path.join(config.get("split_path", None), "test_files.txt"),
            target_image_size=tuple(config.get("image_size", (192, 640))),
            frame_count=config.get("data_fc", 1),
            return_stereo=config.get("data_stereo", False),
            return_fisheye=config.get("data_fisheye", False),
            return_3d_bboxes=config.get("data_3d_bboxes", False),
            return_segmentation=config.get("data_segmentation", False),
            keyframe_offset=0,
            fisheye_rotation=config.get("fisheye_rotation", 0),
            fisheye_offset=config.get("fisheye_offset", 1),
            dilation=config.get("dilation", 1),
            is_preprocessed=config.get("is_preprocessed", False)
        )
        return test_dataset
    elif type == "RealEstate10k":
        test_dataset = RealEstate10kDataset(
            data_path=config["data_path"],
            split_path=os.path.join(config["split_path"], "test_files.txt"),
            target_image_size=config.get("image_size", (256, 384)),
            frame_count=config.get("data_fc", 2),
            keyframe_offset=0,
            dilation=config.get("dilation", 10),
            color_aug=False
        )
        return test_dataset
    elif type == "NYU_Depth_V2":
        test_dataset = NYUDepthV2Dataset(
            data_path=config["data_path"],
            target_image_size=config.get("image_size", (256, 384)),
        )
        return test_dataset
    else:
        raise NotImplementedError(f"Unsupported dataset type: {type}")