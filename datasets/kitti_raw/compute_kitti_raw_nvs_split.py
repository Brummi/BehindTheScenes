import argparse

from pathlib import Path

import numpy as np

DRY_RUN = False


def raw_city_sequences():
  """Sequence names for city sequences in kitti raw data.
  Returns:
    seq_names: list of names
  """
  seq_names = [
      '2011_09_26_drive_0001_sync',
      '2011_09_26_drive_0002_sync',
      '2011_09_26_drive_0005_sync',
      '2011_09_26_drive_0009_sync',
      '2011_09_26_drive_0011_sync',
      '2011_09_26_drive_0013_sync',
      '2011_09_26_drive_0014_sync',
      '2011_09_26_drive_0017_sync',
      '2011_09_26_drive_0018_sync',
      '2011_09_26_drive_0048_sync',
      '2011_09_26_drive_0051_sync',
      '2011_09_26_drive_0056_sync',
      '2011_09_26_drive_0057_sync',
      '2011_09_26_drive_0059_sync',
      '2011_09_26_drive_0060_sync',
      '2011_09_26_drive_0084_sync',
      '2011_09_26_drive_0091_sync',
      '2011_09_26_drive_0093_sync',
      '2011_09_26_drive_0095_sync',
      '2011_09_26_drive_0096_sync',
      '2011_09_26_drive_0104_sync',
      '2011_09_26_drive_0106_sync',
      '2011_09_26_drive_0113_sync',
      '2011_09_26_drive_0117_sync',
      '2011_09_28_drive_0001_sync',
      '2011_09_28_drive_0002_sync',
      '2011_09_29_drive_0026_sync',
      '2011_09_29_drive_0071_sync',
  ]
  return seq_names


# Corresponds to Tulsiani et al.
def main():
    parser = argparse.ArgumentParser("KITTI Raw NVS Split")
    parser.add_argument("--data_path", "-d", type=str)
    parser.add_argument("--out_path", "-o", type=str)

    args = parser.parse_args()

    data_path = Path(args.data_path)
    out_path = Path(args.out_path)

    print("Setting up folders...")
    Path(out_path).mkdir(parents=True, exist_ok=True)

    seqs = raw_city_sequences()

    # This seems to be very strange, but the original repo does it. https://github.com/google/layered-scene-inference/blob/master/lsi/data/kitti/data.py
    rng = np.random.RandomState(0)
    rng.shuffle(seqs)

    n_all = len(seqs)
    n_train = int(round(0.7 * n_all))
    n_val = int(round(0.15 * n_all))

    train_seq_paths = seqs[0:n_train]
    val_seq_paths = seqs[n_train:(n_train+n_val)]
    test_seq_paths = seqs[(n_train+n_val):n_all]

    print(f"Found sequences: {len(train_seq_paths)} (train), {len(val_seq_paths)} (val), {len(test_seq_paths)} (test).")

    names = ("train", "val", "test")

    for name, seqs in zip(names, (train_seq_paths, val_seq_paths, test_seq_paths)):
        split_file = out_path / f"{name}_files.txt"

        print(f"Collecting files for {name} split.")

        lines = []

        for seq in seqs:

            seq_day = seq[:10]
            seq_path = data_path / seq_day / seq / "image_02" / "data"

            seq_lines = [f"{seq_day}/{seq} {i:010d} l" for i, img_path in enumerate(sorted(seq_path.iterdir()))]

            if name == "train":
                seq_lines += [f"{seq_day}/{seq} {i:010d} r" for i, img_path in enumerate(sorted(seq_path.iterdir()))]

            print(f">> Found {len(seq_lines)} files in {seq}.")

            lines.extend(seq_lines)

        print(f">> Found {len(lines)} files in total for {name} split.")

        with open(split_file, "w") as f:
            f.writelines("\n".join(lines))


if __name__ == "__main__":
    main()