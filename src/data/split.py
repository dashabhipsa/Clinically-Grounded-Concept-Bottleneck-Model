"""Reproducible train/validation/test splits for VinDr-CXR.

The *official test set* is kept untouched. The *official training set* is
split into train/validation using a fixed seed (default 42), so splits are
fully reproducible.

Split files (one column ``image_id``):

    data/splits/train.csv
    data/splits/val.csv
    data/splits/test.csv
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.annotations import find_image_ids, load_boxes, load_image_labels, load_metadata

SPLIT_FILES = ["train.csv", "val.csv", "test.csv"]


def official_split_ids(root: str | Path) -> Dict[str, List[str]]:
    """Return the official VinDr-CXR train/test image id lists.

    The image-id universe is taken from ``metadata.csv`` (which covers the
    whole dataset) plus the ids found in the annotation files, so images
    without any finding are still covered. Membership:

    * train: ids appearing in the train box/label files
    * test:  ids appearing in the test box/label files, plus any remaining
      unclassified images (no-findings test images)
    """
    root = Path(root)
    annotations = root / "annotations"

    train_union = set(
        find_image_ids(
            load_boxes(annotations / "train.csv"),
            load_image_labels(annotations / "image_labels_train.csv"),
        )
    )
    test_union = set(
        find_image_ids(
            load_boxes(annotations / "test.csv"),
            load_image_labels(annotations / "image_labels_test.csv"),
        )
    )

    universe = set(train_union) | set(test_union)
    metadata_path = root / "metadata.csv"
    if metadata_path.exists():
        meta = load_metadata(metadata_path)
        universe |= {str(x) for x in meta["image_id"].astype(str).unique()}

    remaining = universe - train_union - test_union
    if remaining:
        # No-findings images not listed in any annotation file: assign to the
        # official test set (VinDr-CXR has only a train and a test split).
        test_union |= remaining

    return {
        "train": sorted(train_union),
        "test": sorted(test_union),
    }


def create_splits(
    root: str | Path,
    splits_dir: str | Path,
    val_fraction: float = 0.1,
    seed: int = 42,
    overwrite: bool = False,
) -> Dict[str, List[str]]:
    """Build train/val/test split CSVs.

    Returns a dict ``{split: [image_ids]}``.
    """
    root = Path(root)
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    for name in SPLIT_FILES:
        target = splits_dir / name
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"{target} already exists. Pass overwrite=True to regenerate."
            )

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    official = official_split_ids(root)
    official_train = np.asarray(official["train"])
    official_test = np.asarray(official["test"])

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(official_train))
    n_val = max(1, int(round(val_fraction * len(official_train))))
    val_ids = [str(i) for i in np.sort(official_train[perm[:n_val]])]
    train_ids = [str(i) for i in np.sort(official_train[perm[n_val:]])]
    test_ids = [str(i) for i in official_test]

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    for name, ids in splits.items():
        _write_ids(splits_dir / f"{name}.csv", ids)

    return splits


def _write_ids(path: Path, ids: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_id"])
        for image_id in ids:
            writer.writerow([image_id])


def load_split_ids(splits_dir: str | Path) -> Dict[str, List[str]]:
    """Load previously generated split CSVs into ``{split: [ids]}``."""
    splits_dir = Path(splits_dir)
    if not (splits_dir / "train.csv").exists():
        raise FileNotFoundError(
            f"Split files not found in {splits_dir}. Run scripts/prepare_data.py first."
        )
    splits: Dict[str, List[str]] = {}
    for name in SPLIT_FILES:
        path = splits_dir / name
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype={"image_id": str})
        splits[name.replace(".csv", "")] = [str(i) for i in df["image_id"].tolist()]
    return splits


def ensure_splits_exist(root: str | Path, splits_dir: str | Path, cfg) -> Dict[str, List[str]]:
    """Create splits if missing; otherwise load them. Used by train scripts."""
    splits_dir = Path(splits_dir)
    if not (splits_dir / "train.csv").exists():
        return create_splits(
            root,
            splits_dir,
            val_fraction=cfg.get("data.val_fraction", 0.1),
            seed=cfg.get("data.split_seed", 42),
        )
    return load_split_ids(splits_dir)
