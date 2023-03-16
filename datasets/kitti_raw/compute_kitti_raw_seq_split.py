import argparse

from pathlib import Path

DRY_RUN = False


# Corresponds to Tulsiani et al.
def main():
    parser = argparse.ArgumentParser("KITTI Raw Sequence Split")
    parser.add_argument("--data_path", "-d", type=str)
    parser.add_argument("--out_path", "-o", type=str)
    parser.add_argument("--seq", "-s", type=str)

    args = parser.parse_args()

    data_path = Path(args.data_path)
    out_path = Path(args.out_path)

    print("Setting up folders...")
    Path(out_path).mkdir(parents=True, exist_ok=True)

    seq = args.seq

    split_file = out_path / f"{seq}_files.txt"

    print(f"Collecting files for {seq} split.")

    lines = []


    seq_day = seq[:10]
    seq_path = data_path / seq_day / seq / "image_02" / "data"

    seq_lines = [f"{seq_day}/{seq} {i:010d} l" for i, img_path in enumerate(sorted(seq_path.iterdir()))]

    print(f">> Found {len(seq_lines)} files in {seq}.")

    lines.extend(seq_lines)

    print(f">> Found {len(lines)} files in total for {seq}.")

    with open(split_file, "w") as f:
        f.writelines("\n".join(lines))


if __name__ == "__main__":
    main()