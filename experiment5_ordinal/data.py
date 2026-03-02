"""UTKFace dataset loading, preprocessing, and train/val/test splitting.

Downloads UTKFace dataset, filters ages 0-99, creates stratified splits,
and provides PyTorch DataLoaders with appropriate augmentation.
"""

import os
import re
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


# UTKFace download URLs (aligned & cropped faces)
UTKFACE_URLS = [
    "https://drive.google.com/uc?export=download&id=0BxYys69jI14kYVM3aVhKS1VhRUk",  # part1
    "https://drive.google.com/uc?export=download&id=0BxYys69jI14kSVdSMUlMOHJfbUE",  # part2
    "https://drive.google.com/uc?export=download&id=0BxYys69jI14kU0I3YUNlUGtMNHM",  # part3
]

# ImageNet normalization (since we use pretrained ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(input_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform(input_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class UTKFaceDataset(Dataset):
    """UTKFace dataset for age estimation.

    Filenames follow the pattern: [age]_[gender]_[race]_[date&time].jpg
    Ages are integer years. We filter to ages 0-99 for clean L=100 binning.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        max_age: int = 99,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.max_age = max_age

        self.samples = []  # list of (filepath, age)
        self._load_samples()

    def _load_samples(self):
        """Scan directory for valid UTKFace images."""
        pattern = re.compile(r"^(\d+)_\d+_\d+_\d+\.jpg\.chip\.jpg$|^(\d+)_\d+_\d+_\d+\.jpg$")

        img_dir = self.root_dir
        # Handle nested directory structures
        if (self.root_dir / "UTKFace").is_dir():
            img_dir = self.root_dir / "UTKFace"
        elif (self.root_dir / "utkface").is_dir():
            img_dir = self.root_dir / "utkface"

        for fname in sorted(os.listdir(img_dir)):
            match = pattern.match(fname)
            if match:
                age = int(match.group(1) or match.group(2))
                if 0 <= age <= self.max_age:
                    self.samples.append((img_dir / fname, age))

        if len(self.samples) == 0:
            # Try alternative: just parse leading integer from any jpg
            for fname in sorted(os.listdir(img_dir)):
                if not fname.lower().endswith(".jpg"):
                    continue
                parts = fname.split("_")
                if parts and parts[0].isdigit():
                    age = int(parts[0])
                    if 0 <= age <= self.max_age:
                        self.samples.append((img_dir / fname, age))

        print(f"Loaded {len(self.samples)} images from {img_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, age = self.samples[idx]
        image = Image.open(filepath).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, age


def get_age_decade(age: int) -> int:
    """Map age to decade bucket for stratification."""
    return min(age // 10, 9)  # 0-9, 10-19, ..., 90-99


def get_difficulty_tier(age: int) -> str:
    """Classify age into difficulty tier per spec §8."""
    if 20 <= age < 40:
        return "easy"
    elif (0 <= age < 5) or (60 <= age < 80):
        return "medium"
    else:  # 5-19 or 80-99
        return "hard"


def create_splits(
    dataset: UTKFaceDataset,
    split_seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Create stratified 80/10/10 train/val/test splits.

    Stratified by age decade to ensure proportional representation.
    Split is deterministic (seed 42) and identical across all runs.

    Returns:
        (train_indices, val_indices, test_indices)
    """
    indices = list(range(len(dataset)))
    ages = [dataset.samples[i][1] for i in indices]
    decades = [get_age_decade(a) for a in ages]

    # First split: 80% train, 20% temp
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, random_state=split_seed, stratify=decades
    )

    # Second split: 50/50 of temp → 10% val, 10% test
    temp_decades = [decades[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=split_seed, stratify=temp_decades
    )

    return train_idx, val_idx, test_idx


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    input_size: int = 224,
    num_workers: int = 4,
    split_seed: int = 42,
    L: int = 100,
) -> tuple[DataLoader, DataLoader, DataLoader, UTKFaceDataset]:
    """Create train/val/test DataLoaders.

    Args:
        data_dir: Path to UTKFace images
        batch_size: Batch size for all splits
        input_size: Image input size (default 224)
        num_workers: DataLoader workers
        split_seed: Seed for train/val/test split (always 42)
        L: Number of age bins (100 or 2)

    Returns:
        (train_loader, val_loader, test_loader, full_dataset)
    """
    max_age = L - 1 if L == 100 else 99

    # Load full dataset without transforms (we'll apply per-split)
    full_dataset = UTKFaceDataset(data_dir, transform=None, max_age=max_age)
    train_idx, val_idx, test_idx = create_splits(full_dataset, split_seed)

    print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Create transform-aware subsets
    train_transform = get_train_transform(input_size)
    eval_transform = get_eval_transform(input_size)

    train_set = TransformedSubset(full_dataset, train_idx, train_transform, L=L)
    val_set = TransformedSubset(full_dataset, val_idx, eval_transform, L=L)
    test_set = TransformedSubset(full_dataset, test_idx, eval_transform, L=L)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, full_dataset


class TransformedSubset(Dataset):
    """Subset of a dataset with per-split transforms and optional L=2 binning."""

    def __init__(self, dataset: UTKFaceDataset, indices: list[int],
                 transform: transforms.Compose, L: int = 100):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.L = L

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        filepath, age = self.dataset.samples[self.indices[idx]]
        image = Image.open(filepath).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if self.L == 2:
            # Binary binning: 0 = ages 0-49, 1 = ages 50-99
            target = 0 if age < 50 else 1
        else:
            # Direct: bin = age (for L=100, age 0-99 maps to bin 0-99)
            target = min(age, self.L - 1)

        return image, target, age  # return raw age too for difficulty analysis

    def get_ages(self) -> list[int]:
        """Get raw ages for all samples in this subset."""
        return [self.dataset.samples[i][1] for i in self.indices]


def download_utkface(output_dir: str):
    """Download UTKFace dataset.

    Note: The Google Drive links may require manual download.
    This function provides instructions if automatic download fails.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UTKFace Dataset Download")
    print("=" * 60)
    print()
    print("Please download the UTKFace dataset manually:")
    print("1. Go to: https://susanqq.github.io/UTKFace/")
    print("2. Download the 'Aligned&Cropped Faces' archive(s)")
    print("3. Extract to:", output_dir)
    print()
    print("Expected structure after extraction:")
    print(f"  {output_dir}/")
    print(f"  {output_dir}/[age]_[gender]_[race]_[datetime].jpg")
    print(f"  OR")
    print(f"  {output_dir}/UTKFace/[age]_[gender]_[race]_[datetime].jpg")
    print()
    print("Alternative: Use kaggle CLI:")
    print("  pip install kaggle")
    print("  kaggle datasets download -d jangedoo/utkface-new -p", output_dir)
    print("  unzip", f"{output_dir}/utkface-new.zip -d {output_dir}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UTKFace data preparation")
    parser.add_argument("--output_dir", type=str, default="./data/utkface",
                        help="Directory for UTKFace images")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing dataset and print statistics")
    args = parser.parse_args()

    if args.verify:
        dataset = UTKFaceDataset(args.output_dir)
        train_idx, val_idx, test_idx = create_splits(dataset)
        ages = [dataset.samples[i][1] for i in range(len(dataset))]
        print(f"\nTotal images: {len(dataset)}")
        print(f"Age range: {min(ages)} - {max(ages)}")
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        # Print age distribution
        decade_counts = {}
        for a in ages:
            d = get_age_decade(a)
            decade_counts[d] = decade_counts.get(d, 0) + 1
        print("\nAge decade distribution:")
        for d in sorted(decade_counts):
            print(f"  {d*10}-{d*10+9}: {decade_counts[d]}")

        # Print difficulty tier distribution
        tier_counts = {"easy": 0, "medium": 0, "hard": 0}
        for a in ages:
            tier_counts[get_difficulty_tier(a)] += 1
        print("\nDifficulty tier distribution:")
        for tier, count in tier_counts.items():
            print(f"  {tier}: {count}")
    else:
        download_utkface(args.output_dir)
