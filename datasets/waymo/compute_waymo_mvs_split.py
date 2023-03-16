import argparse
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm

from datasets.waymo.waymo_dataset import WaymoDataset

tf.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset


DRY_RUN = False


def check_time_of_day(records, seq):
    record = records / f"segment-{seq}_with_camera_labels.tfrecord"
    dataset = tf.data.TFRecordDataset(record, compression_type='')
    base_data = next(iter(dataset))
    base_frame = open_dataset.Frame()
    base_frame.ParseFromString(bytearray(base_data.numpy()))
    time_of_day = base_frame.context.stats.time_of_day
    return time_of_day


def main():
    parser = argparse.ArgumentParser("Waymo MVS Split")
    parser.add_argument("--data_path", "-d", type=str)
    parser.add_argument("--out_path", "-o", type=str)
    parser.add_argument("--min_d", type=float, default=.5)
    parser.add_argument("--day_only", action="store_true")
    parser.add_argument("--train_records", type=str, default="")
    parser.add_argument("--val_records", type=str, default="")
    parser.add_argument("--test_records", type=str, default="")

    args = parser.parse_args()

    data_path = Path(args.data_path)
    out_path = Path(args.out_path)
    min_d = args.min_d
    day_only = args.day_only
    train_records = Path(args.train_records)
    val_records = Path(args.val_records)
    test_records = Path(args.test_records)

    print("Setting up folders...")
    Path(out_path).mkdir(parents=True, exist_ok=True)

    splits = [
        ("training", "train", train_records),
        ("validation", "val", val_records),
        ("testing", "test", test_records)
    ]

    for name, short_name, records in splits:
        dataset = WaymoDataset(str(data_path), name, None, return_45=False, return_90=False, frame_count=1)

        files = []

        for seq in tqdm(dataset._sequences):
            if day_only:
                tod = check_time_of_day(records, seq)
                if tod != "Day":
                    continue

            poses = dataset._poses[seq]
            positions = poses[:, :3, 3]
            dists = np.linalg.norm(positions[1:, :] - positions[:-1, :], axis=-1)
            valid = dists > min_d
            valid = valid[1:] | valid[:-1]

            indices = valid.nonzero()[0]

            files += [f"{seq} {i+1:010d}" for i in indices]

        print(f"{short_name}: {len(files)} filtered data samples from {len(dataset)} data samples.")

        with open(out_path / f"{short_name}_files.txt", "w") as f:
            f.write("\n".join(files))


if __name__ == "__main__":
    main()
