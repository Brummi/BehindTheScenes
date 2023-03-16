import argparse
import json
import pickle

from pathlib import Path

import numpy as np

DRY_RUN = False


# Corresponds to Tulsiani et al.
def main():
    parser = argparse.ArgumentParser("KITTI Raw NVS Split")
    parser.add_argument("--data_path", "-d", type=str)
    parser.add_argument("--out_path", "-o", type=str)
    parser.add_argument("--split_path", "-s", type=str)

    args = parser.parse_args()

    data_path = Path(args.data_path)
    out_path = Path(args.out_path)
    split_path = Path(args.split_path)

    print("Setting up folders...")
    Path(out_path).mkdir(parents=True, exist_ok=True)

    file_path = Path(data_path) / f"test.pickle"
    with open(file_path, "rb") as f:
        seq_data = pickle.load(f)

    modes = ["val", "test"]

    def get_key_id(sample):
        key = sample["sequence_id"]
        if key not in seq_data:
            return None
        timestamp = int(sample["src_img_obj"]["frame_ts"])
        id = np.searchsorted(seq_data[key]["timestamps"], timestamp)
        return key, id

    for mode in modes:
        split_file_name = "test_pairs.json" if mode == "test" else "validation_pairs.json"
        with open(split_path / split_file_name, "r") as f:
            lines = f.readlines()
            split_data = list(map(json.loads, lines))

        print(f"Loaded {len(split_data)} samples.")

        key_id_pairs = list(map(get_key_id, split_data))
        key_id_pairs_filtered = [pair for pair in key_id_pairs if pair is not None]
        print(f"Skipped {len(key_id_pairs)-len(key_id_pairs_filtered)} frames.")

        key_id_strs = [f"{key} {id}" for key, id in key_id_pairs_filtered]

        print(f"Processed {len(split_data)} samples.")

        with open(out_path / f"{mode}_files.txt", "w") as f:
            f.write("\n".join(key_id_strs))


if __name__ == "__main__":
    main()