import argparse

from pathlib import Path

DRY_RUN = False

CUT = [7, 9, 10, 15, 19, 31, 43, 69, 87, 107, 118, 154, 156, 167, 168, 170, 171, 172, 173, 174, 178, 179, 180, 181,
         182, 183, 184, 185, 187, 188, 193, 194, 195, 196, 201, 202, 203, 209, 210, 212, 213, 214, 215, 216, 217, 218,
         219, 220, 221, 222, 224, 225, 226, 229, 230, 231, 234, 235, 236, 237, 238, 256, 257, 258, 267, 278, 283, 293,
         294, 295, 296, 297, 298, 299, 310, 315, 317, 318, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333,
         334, 335, 336, 337, 340, 341, 349, 353, 354, 361, 362, 365, 366, 368, 371, 372, 376, 380, 386, 387, 394, 402,
         403, 404, 411, 414, 415, 416, 420, 438, 441, 448, 452, 456, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482,
         484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 519, 520, 554, 562, 593, 594,
         596]


def check_integrity(data_path, seq, img_id):
    persp = data_path / "data_2d_raw" / seq / "image_00" / "data_rect" / f"{img_id:010d}.png"
    fish = data_path / "data_2d_raw" / seq / "image_02" / "data_rgb" / f"{img_id:010d}.png"

    return fish.exists() and persp.exists()


def main():
    parser = argparse.ArgumentParser("KITTI Raw NVS Split")
    parser.add_argument("--data_path", "-d", type=str)
    parser.add_argument("--out_path", "-o", type=str)
    parser.add_argument("--offset", type=int, default=20)

    args = parser.parse_args()

    data_path = Path(args.data_path)
    out_path = Path(args.out_path)
    offset = args.offset

    print("Setting up folders...")
    Path(out_path).mkdir(parents=True, exist_ok=True)

    segmentation_train_file = data_path / "data_2d_semantics" / "train" / "2013_05_28_drive_train_frames.txt"
    segmentation_val_file = data_path / "data_2d_semantics" / "train" / "2013_05_28_drive_val_frames.txt"

    with open(segmentation_train_file, "r") as f:
        train_lines = f.readlines()
    with open(segmentation_val_file, "r") as f:
        val_lines = f.readlines()

    train_files = []
    val_files = []
    test_files = []

    invalid = 0

    for i in range(len(train_lines)):
        parts = train_lines[i].split(" ")
        img_path = parts[0]

        parts = img_path.split("/")
        sequence = parts[1]
        img_id = int(parts[-1][-14:-4])

        if not check_integrity(data_path, sequence, img_id):
            invalid += 1
            continue

        train_files.append(f"{sequence} {img_id:010d} l")
        train_files.append(f"{sequence} {img_id:010d} r")

    for i in range(0, len(val_lines)):
        parts = val_lines[i].split(" ")
        img_path = parts[0]
        seg_path = parts[1][:-1]

        parts = img_path.split("/")
        sequence = parts[1]
        img_id = int(parts[-1][-14:-4])

        is_test = (i % offset) == 0

        if not check_integrity(data_path, sequence, img_id):
            invalid += 1
            continue

        if not is_test:
            val_files.append(f"{sequence} {img_id:010d} l")
        else:
            test_files.append(f"{sequence} {img_id:010d} l")

    print(f"Found: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)} test files.")
    print(f"Found: {invalid} invalids.")

    test_files = [s for i, s in enumerate(test_files) if not i in CUT]

    print(f"{len(CUT)} test files removed. {len(test_files)} remaining.")

    train_file = out_path / f"train_files.txt"
    val_file = out_path / f"val_files.txt"
    test_file = out_path / f"test_files.txt"

    with open(train_file, "w") as f:
        f.writelines("\n".join(train_files))

    with open(val_file, "w") as f:
        f.writelines("\n".join(val_files))

    with open(test_file, "w") as f:
        f.writelines("\n".join(test_files))


if __name__ == "__main__":
    main()