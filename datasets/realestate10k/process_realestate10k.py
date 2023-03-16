import argparse
import pickle
from pathlib import Path

from tqdm import tqdm

import numpy as np


def load_data(mode_path: Path, failures_path: Path):
    seq_files = sorted(mode_path.glob('*.txt'))
    with open(failures_path, "r") as f:
        failures = set(l[:-1] for l in f.readlines())

    skipped = 0

    seq_data = {}

    for seq_file in tqdm(seq_files):
        seq_name = seq_file.stem

        if seq_name in failures:
            # print(f"Skipping sequence {seq_name} because the download had failed.")
            skipped += 1
            continue

        try:
            timestamps = np.loadtxt(seq_file, dtype=int, skiprows=1).reshape((-1, 19))[:, 0]
        except:
            print(seq_file)
            pass
        data = np.loadtxt(seq_file, dtype=float, skiprows=1).reshape((-1, 19))[:, 1:]
        intrinsics = data[:, :6]
        pose = np.reshape(data[:, 6:], (-1, 3, 4))

        seq_data[seq_name] = {
            "timestamps": timestamps,
            "intrinsics": intrinsics,
            "poses": pose
        }

    print(f"Skipped {skipped} sequences because the video download had failed.")

    return seq_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str)
    parser.add_argument("-d", "--data_path", type=str)

    args = parser.parse_args()
    mode = args.mode
    data_path = Path(args.data_path)

    if mode not in ["test", "train"]:
        raise ValueError(f"Invalid split mode: {mode}")

    mode_path = data_path / mode
    failures_path = data_path / f"failed_videos_{mode}.txt"

    print("Loading data")
    data = load_data(mode_path, failures_path)

    print()
    print("Saving data into a single file.")
    with open(data_path / f"{mode}.pickle", "wb") as f:
        pickle.dump(data, f)

    print()
    print("Done")


if __name__ == "__main__":
    main()
