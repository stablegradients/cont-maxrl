"""Tests for data loading and preprocessing.

Covers:
- Dataset loading and sample counts
- Stratified split properties
- Age decade distribution preservation
- Difficulty tier assignment
- L=2 binning correctness
- Transform properties (shape, normalization)
- DataLoader batch correctness
- Split determinism (always same with seed 42)
"""

import pytest
import torch

from experiment5_ordinal.data import (
    UTKFaceDataset,
    TransformedSubset,
    create_splits,
    get_age_decade,
    get_dataloaders,
    get_difficulty_tier,
    get_eval_transform,
    get_train_transform,
)
from experiment5_ordinal.tests.conftest import requires_data


class TestDifficultyTier:
    """Test difficulty tier assignment."""

    def test_easy_range(self):
        for age in [20, 25, 30, 35, 39]:
            assert get_difficulty_tier(age) == "easy"

    def test_medium_range(self):
        for age in [0, 1, 2, 3, 4, 60, 65, 70, 75, 79]:
            assert get_difficulty_tier(age) == "medium"

    def test_hard_range(self):
        for age in [5, 10, 15, 19, 80, 85, 90, 95, 99]:
            assert get_difficulty_tier(age) == "hard"

    def test_boundary_exclusivity(self):
        """Age 20 is easy (not hard), age 40 is NOT easy (it's hard 40-49 is hard)."""
        assert get_difficulty_tier(20) == "easy"
        assert get_difficulty_tier(40) != "easy"
        assert get_difficulty_tier(5) == "hard"
        assert get_difficulty_tier(4) == "medium"
        assert get_difficulty_tier(60) == "medium"
        assert get_difficulty_tier(80) == "hard"

    def test_all_ages_covered(self):
        """Every age 0-99 should map to a tier."""
        for age in range(100):
            tier = get_difficulty_tier(age)
            assert tier in ("easy", "medium", "hard"), f"Age {age} has no tier"


class TestAgeDecade:
    """Test age decade mapping."""

    def test_decade_boundaries(self):
        assert get_age_decade(0) == 0
        assert get_age_decade(9) == 0
        assert get_age_decade(10) == 1
        assert get_age_decade(99) == 9

    def test_clamped_at_9(self):
        """Ages ≥90 should all map to decade 9."""
        assert get_age_decade(90) == 9
        assert get_age_decade(99) == 9


@requires_data
class TestDatasetLoading:
    """Tests that require the UTKFace dataset."""

    def test_dataset_loads(self, data_dir):
        ds = UTKFaceDataset(data_dir)
        assert len(ds) > 20000, f"Expected >20K images, got {len(ds)}"

    def test_age_range(self, data_dir):
        ds = UTKFaceDataset(data_dir)
        ages = [s[1] for s in ds.samples]
        assert min(ages) >= 0
        assert max(ages) <= 99

    def test_sample_format(self, data_dir):
        """Each sample should be (filepath, age)."""
        ds = UTKFaceDataset(data_dir)
        filepath, age = ds.samples[0]
        assert filepath.exists()
        assert isinstance(age, int)
        assert 0 <= age <= 99


@requires_data
class TestSplits:
    """Test data splitting properties."""

    def test_split_sizes(self, data_dir):
        """80/10/10 split proportions."""
        ds = UTKFaceDataset(data_dir)
        train, val, test = create_splits(ds)
        total = len(ds)

        assert abs(len(train) / total - 0.8) < 0.02
        assert abs(len(val) / total - 0.1) < 0.02
        assert abs(len(test) / total - 0.1) < 0.02

    def test_no_overlap(self, data_dir):
        """Splits should not share indices."""
        ds = UTKFaceDataset(data_dir)
        train, val, test = create_splits(ds)

        train_set = set(train)
        val_set = set(val)
        test_set = set(test)

        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_complete_coverage(self, data_dir):
        """All indices should be in exactly one split."""
        ds = UTKFaceDataset(data_dir)
        train, val, test = create_splits(ds)
        all_idx = set(train) | set(val) | set(test)
        assert all_idx == set(range(len(ds)))

    def test_deterministic(self, data_dir):
        """Same seed should produce same splits."""
        ds = UTKFaceDataset(data_dir)
        t1, v1, te1 = create_splits(ds, split_seed=42)
        t2, v2, te2 = create_splits(ds, split_seed=42)
        assert t1 == t2
        assert v1 == v2
        assert te1 == te2

    def test_different_seed_different_split(self, data_dir):
        """Different seed should produce different splits."""
        ds = UTKFaceDataset(data_dir)
        t1, _, _ = create_splits(ds, split_seed=42)
        t2, _, _ = create_splits(ds, split_seed=99)
        assert t1 != t2

    def test_stratification(self, data_dir):
        """Each decade should be proportionally represented in each split."""
        ds = UTKFaceDataset(data_dir)
        train, val, test = create_splits(ds)

        def decade_dist(indices):
            counts = {}
            for i in indices:
                d = get_age_decade(ds.samples[i][1])
                counts[d] = counts.get(d, 0) + 1
            total = sum(counts.values())
            return {d: c / total for d, c in counts.items()}

        train_dist = decade_dist(train)
        val_dist = decade_dist(val)
        full_dist = decade_dist(range(len(ds)))

        # Each decade fraction should be within 2% of full distribution
        for decade in full_dist:
            if decade in train_dist:
                assert abs(train_dist[decade] - full_dist[decade]) < 0.02, (
                    f"Decade {decade}: train={train_dist[decade]:.3f}, "
                    f"full={full_dist[decade]:.3f}"
                )
            if decade in val_dist:
                assert abs(val_dist[decade] - full_dist[decade]) < 0.03


@requires_data
class TestTransformedSubset:
    """Test TransformedSubset with actual data."""

    def test_l100_targets(self, data_dir):
        """L=100: target = age (0-99)."""
        ds = UTKFaceDataset(data_dir)
        train_idx, _, _ = create_splits(ds)
        subset = TransformedSubset(ds, train_idx[:10], get_eval_transform(), L=100)
        for i in range(10):
            _, target, age = subset[i]
            assert target == min(age, 99)

    def test_l2_targets(self, data_dir):
        """L=2: target = 0 if age < 50, else 1."""
        ds = UTKFaceDataset(data_dir)
        train_idx, _, _ = create_splits(ds)
        subset = TransformedSubset(ds, train_idx[:100], get_eval_transform(), L=2)
        for i in range(100):
            _, target, age = subset[i]
            expected = 0 if age < 50 else 1
            assert target == expected, f"Age {age}: got target {target}, expected {expected}"

    def test_output_shape(self, data_dir):
        """Images should be (3, 224, 224) after transforms."""
        ds = UTKFaceDataset(data_dir)
        train_idx, _, _ = create_splits(ds)
        subset = TransformedSubset(ds, train_idx[:5], get_eval_transform(224), L=100)
        img, target, age = subset[0]
        assert img.shape == (3, 224, 224)


@requires_data
class TestDataLoaders:
    """Test DataLoader creation."""

    def test_loader_batch_shape(self, data_dir):
        """Check batch dimensions."""
        train_loader, _, _, _ = get_dataloaders(
            data_dir, batch_size=8, num_workers=0, L=100,
        )
        images, targets, ages = next(iter(train_loader))
        assert images.shape == (8, 3, 224, 224)
        assert targets.shape == (8,)
        assert len(ages) == 8

    def test_targets_in_range(self, data_dir):
        """All targets should be in [0, L-1]."""
        _, val_loader, _, _ = get_dataloaders(
            data_dir, batch_size=64, num_workers=0, L=100,
        )
        for images, targets, ages in val_loader:
            assert targets.min() >= 0
            assert targets.max() <= 99
            break

    def test_l2_loader(self, data_dir):
        """L=2 loader should have binary targets."""
        train_loader, _, _, _ = get_dataloaders(
            data_dir, batch_size=32, num_workers=0, L=2,
        )
        images, targets, ages = next(iter(train_loader))
        assert set(targets.unique().tolist()).issubset({0, 1})
