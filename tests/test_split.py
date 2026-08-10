"""Tests for reproducible train/val/test splitting."""
from src.data.split import create_splits, load_split_ids, official_split_ids


def test_split_reproducible(synthetic_vindr, tmp_path):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    create_splits(synthetic_vindr, a_dir, val_fraction=0.25, seed=42)
    create_splits(synthetic_vindr, b_dir, val_fraction=0.25, seed=42)
    a = load_split_ids(a_dir)
    b = load_split_ids(b_dir)
    for split in ("train", "val", "test"):
        assert a[split] == b[split], f"split {split} not reproducible"


def test_splits_disjoint_and_cover(synthetic_vindr, tmp_path):
    splits = create_splits(synthetic_vindr, tmp_path / "s", val_fraction=0.25, seed=42)
    official = official_split_ids(synthetic_vindr)

    assert set(splits["train"]).isdisjoint(set(splits["val"]))
    assert set(splits["train"]) | set(splits["val"]) == set(official["train"])
    assert len(splits["train"]) + len(splits["val"]) == len(official["train"])


def test_test_split_untouched(synthetic_vindr, tmp_path):
    splits = create_splits(synthetic_vindr, tmp_path / "s", val_fraction=0.25, seed=42)
    official = official_split_ids(synthetic_vindr)
    assert set(splits["test"]) == set(official["test"])
    # test ids never appear in train/val
    assert set(splits["test"]).isdisjoint(set(splits["train"]) | set(splits["val"]))


def test_val_size_consistent(synthetic_vindr, tmp_path):
    splits = create_splits(synthetic_vindr, tmp_path / "s", val_fraction=0.5, seed=42)
    total = len(splits["train"]) + len(splits["val"])
    assert len(splits["val"]) == max(1, round(0.5 * total))


def test_different_seed_different_split(synthetic_vindr, tmp_path):
    s1 = create_splits(synthetic_vindr, tmp_path / "s1", val_fraction=0.5, seed=1)
    s2 = create_splits(synthetic_vindr, tmp_path / "s2", val_fraction=0.5, seed=2)
    # Different seeds select a different validation subset (deterministic given
    # the fixed seeds, so this is not flaky).
    assert set(s1["val"]) != set(s2["val"])
