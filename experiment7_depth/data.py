"""NYU Depth v2 data loading for dense depth estimation."""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_H, INPUT_W = 352, 256


def bin_depth(depth_m, bin_edges, K):
    """Bin depth values. depth_m: (...). Returns integer bins in [0, K-1]."""
    return (torch.bucketize(depth_m, bin_edges) - 1).clamp(0, K - 1)


def bin_to_meters(bin_idx, bin_centers):
    """Convert bin index back to meters. bin_idx: (...) long."""
    return bin_centers[bin_idx]


def apply_horizontal_flip(image, target_bin, target_m, valid_mask):
    """Atomically flip all tensors along width axis.

    Args:
        image: (3, H, W)
        target_bin: (H, W)
        target_m: (H, W)
        valid_mask: (H, W)

    Returns:
        Tuple of flipped (image, target_bin, target_m, valid_mask)
    """
    return (
        torch.flip(image, dims=[-1]),
        torch.flip(target_bin, dims=[-1]),
        torch.flip(target_m, dims=[-1]),
        torch.flip(valid_mask, dims=[-1]),
    )


class NYUDepthDataset(Dataset):
    """NYU Depth v2 dataset for dense depth estimation."""

    def __init__(self, hf_dataset, K, train=False, split_seed=42):
        """
        Args:
            hf_dataset: HuggingFace dataset with 'image' and 'depth_map' columns
            K: number of depth bins
            train: if True, apply augmentation
            split_seed: unused here, kept for API consistency
        """
        self.dataset = hf_dataset
        self.K = K
        self.train = train
        self.bin_edges = torch.linspace(0.5, 10.0, K + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

        # Color jitter for training
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Load image and depth as PIL
        image_pil = item["image"].convert("RGB")
        depth_pil = item["depth_map"]

        # Resize image (bilinear) and depth (nearest)
        image_pil = image_pil.resize((INPUT_W, INPUT_H), Image.BILINEAR)
        depth_pil = depth_pil.resize((INPUT_W, INPUT_H), Image.NEAREST)

        # Convert to tensors
        image = TF.to_tensor(image_pil)  # (3, H, W), float32 [0, 1]
        depth_np = np.array(depth_pil, dtype=np.float32)
        depth_m = torch.from_numpy(depth_np)  # (H, W)

        # Valid mask: non-zero original AND within [0.5, 10.0] strictly
        valid_mask = (depth_m > 0) & (depth_m > 0.5) & (depth_m < 10.0)

        # Clip depth to [0.5, 10.0]
        depth_m = depth_m.clamp(0.5, 10.0)

        # Bin depth
        target_bin = bin_depth(depth_m, self.bin_edges, self.K)

        # Training augmentation
        if self.train:
            # Color jitter (before normalization)
            image = self.color_jitter(image)

            # Random horizontal flip
            if random.random() < 0.5:
                image, target_bin, depth_m, valid_mask = apply_horizontal_flip(
                    image, target_bin, depth_m, valid_mask
                )

        # Normalize
        image = self.normalize(image)

        return {
            "image": image,
            "target_bin": target_bin,
            "target_m": depth_m,
            "valid_mask": valid_mask,
            "image_idx": idx,
        }


def get_dataloaders(
    data_dir: str,
    batch_size: int,
    K: int,
    num_workers: int = 4,
    split_seed: int = 42,
) -> tuple:
    """Build train, val, test dataloaders.

    Split strategy:
    - NYU's train (47K) is split 90/10 into train/val deterministically by split_seed.
    - NYU's validation (654) is used as the test set.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    from datasets import load_from_disk

    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "validation")

    # Load from disk (pre-downloaded)
    full_train = load_from_disk(train_path)
    test_hf = load_from_disk(val_path)

    # Split train 90/10
    split = full_train.train_test_split(test_size=0.1, seed=split_seed)
    train_hf = split["train"]
    val_hf = split["test"]

    train_dataset = NYUDepthDataset(train_hf, K=K, train=True)
    val_dataset = NYUDepthDataset(val_hf, K=K, train=False)
    test_dataset = NYUDepthDataset(test_hf, K=K, train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
