"""Pre-download NYU Depth v2 to local cache.

Usage:
    python -m experiment7_depth.download_nyu --data_dir ./data/nyu
"""

import argparse
from pathlib import Path


def download_nyu(data_dir: str):
    """Download NYU Depth v2 via HuggingFace datasets."""
    from datasets import load_dataset

    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Downloading NYU Depth v2 (train split)...")
    train_ds = load_dataset("sayakpaul/nyu_depth_v2", split="train")
    train_path = out / "train"
    train_ds.save_to_disk(str(train_path))
    print(f"  Saved {len(train_ds)} train images to {train_path}")

    print("Downloading NYU Depth v2 (validation split)...")
    val_ds = load_dataset("sayakpaul/nyu_depth_v2", split="validation")
    val_path = out / "validation"
    val_ds.save_to_disk(str(val_path))
    print(f"  Saved {len(val_ds)} validation images to {val_path}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Download NYU Depth v2")
    parser.add_argument("--data_dir", type=str, default="./data/nyu")
    args = parser.parse_args()
    download_nyu(args.data_dir)


if __name__ == "__main__":
    main()
