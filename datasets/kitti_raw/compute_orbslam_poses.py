import argparse

from pathlib import Path
import subprocess

DRY_RUN = False


def run_command(cmd):
    print(">>", " ".join(cmd))
    if not DRY_RUN:
        subprocess.run(" ".join(cmd), shell=True, check=True)


def get_sequences(data_path):
    all_sequences = []

    data_path = Path(data_path)
    for day in data_path.iterdir():
        if not day.is_dir():
            continue
        day_sequences = [seq for seq in day.iterdir() if seq.is_dir()]
        day_sequences = [(day.name, seq.name) for seq in day_sequences]
        all_sequences.extend(day_sequences)

    return all_sequences


def main():
    """
    Requires a modified version of the ORB-SLAM 3 executable.
    Pre-computed ORB-SLAM3 poses are provided in the orb-slam_poses folder.
    """
    parser = argparse.ArgumentParser("KITTI Raw ORB-SLAM script")
    parser.add_argument("--exe_path", "-e", type=str)
    parser.add_argument("--vocab_path", "-v", type=str)
    parser.add_argument("--data_path", "-d", type=str)
    parser.add_argument("--settings_path", "-s", type=str)
    parser.add_argument("--out_path", "-o", type=str)

    args = parser.parse_args()

    exe_path = args.exe_path
    vocab_path = args.vocab_path
    data_path = args.data_path
    settings_path = args.settings_path
    out_path = args.out_path

    print("Setting up folders...")
    Path(out_path).mkdir(parents=True, exist_ok=True)

    print("Listing sequences...")
    sequences = get_sequences(data_path)
    print(f"Found {len(sequences)} sequences.")

    for i, (day, seq) in enumerate(sequences):
        print(f"Processing {day}/{seq} ({i+1}/{len(sequences)})...")

        seq_path = Path(data_path) / day / seq
        seq_settings_path = Path(settings_path) / f"{day}.yaml"
        seq_out_path = Path(out_path) / day
        seq_out_path.mkdir(exist_ok=True, parents=False)
        seq_out_path = seq_out_path / f"{seq}.txt"

        if seq_out_path.exists():
            print(f"Poses for {str(seq_out_path)} are already computed - skipping.")
            continue

        command = [exe_path, str(vocab_path), str(seq_settings_path), str(seq_path), str(seq_out_path)]
        run_command(command)


if __name__ == "__main__":
    main()