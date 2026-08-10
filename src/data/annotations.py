"""Parsing of VinDr-CXR annotation files.

VinDr-CXR provides (per official split):

* ``annotations/train.csv`` / ``annotations/test.csv``
  Local (bounding-box) findings. Columns:
  ``image_id, class_id, class_name, x_max, x_min, y_max, y_min, rad_id``.
  Coordinates are normalized to [0, 1] relative to the image.

* ``annotations/image_labels_train.csv`` / ``annotations/image_labels_test.csv``
  Global diagnostic labels. Columns: ``image_id, class_name``.

* ``metadata.csv``
  ``image_id, patient_id, study_id``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Canonical VinDr-CXR classes (normalized form of the class names used in
# the annotation files).
VINDR_CLASSES: List[str] = [
    "aortic_enlargement",
    "atelectasis",
    "calcification",
    "cardiomegaly",
    "consolidation",
    "ild",
    "infiltration",
    "lung_opacity",
    "nodule_mass",
    "other_lesion",
    "pleural_effusion",
    "pleural_thickening",
    "pneumothorax",
    "pulmonary_fibrosis",
]

BOX_COLUMNS = ["image_id", "class_id", "class_name", "x_max", "x_min", "y_max", "y_min", "rad_id"]
LABEL_COLUMNS = ["image_id", "class_name"]


def normalize_class_name(name: str) -> str:
    """Map a VinDr class name to a canonical key.

    Examples:
        "Pleural effusion" -> "pleural_effusion"
        "Nodule/Mass"      -> "nodule_mass"
        "Lung opacity"     -> "lung_opacity"
    """
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


@dataclass
class Box:
    """A single local finding bounding box (coordinates normalized to [0, 1])."""

    image_id: str
    class_id: int
    class_name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    rad_id: int = 0

    def __post_init__(self) -> None:
        for attr in ("x_min", "y_min", "x_max", "y_max"):
            value = float(getattr(self, attr))
            object.__setattr__(self, attr, value)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Box coordinate {attr}={value} for image {self.image_id} "
                    f"outside [0, 1]. VinDr-CXR coordinates are normalized."
                )

    @property
    def canonical_class(self) -> str:
        return normalize_class_name(self.class_name)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "rad_id": self.rad_id,
        }


def load_boxes(path: str | Path) -> pd.DataFrame:
    """Load a VinDr-CXR bounding-box annotation file, validating columns.

    ``image_id`` is kept as a string so leading zeros (e.g. ``"0001"``,
    which also appear in the DICOM filenames) are preserved.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    df = pd.read_csv(path, dtype={"image_id": str})
    missing = [c for c in BOX_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    return df


def load_image_labels(path: str | Path) -> pd.DataFrame:
    """Load a VinDr-CXR global image-label file, validating columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")
    df = pd.read_csv(path, dtype={"image_id": str})
    missing = [c for c in LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    return df


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Load VinDr-CXR ``metadata.csv`` (image_id, patient_id, study_id)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    df = pd.read_csv(path, dtype={"image_id": str})
    if "image_id" not in df.columns:
        raise ValueError(f"{path}: missing required column 'image_id'")
    return df


def group_boxes(df: pd.DataFrame) -> Dict[str, List[Box]]:
    """Group a box annotation DataFrame by image_id -> list[Box]."""
    boxes_by_image: Dict[str, List[Box]] = {}
    for row in df.itertuples(index=False):
        box = Box(
            image_id=str(row.image_id),
            class_id=int(row.class_id),
            class_name=str(row.class_name),
            x_min=float(row.x_min),
            y_min=float(row.y_min),
            x_max=float(row.x_max),
            y_max=float(row.y_max),
            rad_id=int(row.rad_id),
        )
        boxes_by_image.setdefault(box.image_id, []).append(box)
    return boxes_by_image


def find_image_ids(*dataframes: pd.DataFrame) -> List[str]:
    """Sorted unique image ids across the given annotation DataFrames."""
    ids: set[str] = set()
    for df in dataframes:
        ids.update(str(x) for x in df["image_id"].astype(str).unique())
    return sorted(ids)


def concepts_from_boxes(
    boxes_df: pd.DataFrame,
    concept_names: List[str],
    image_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Multi-hot concept matrix from local (box) findings.

    A concept is *present* for an image if at least one bounding box of that
    class exists in its annotations. Images without any matching box get 0.

    Returns a DataFrame indexed by image_id with one column per concept.
    """
    if not concept_names:
        raise ValueError("concept_names must be non-empty")
    boxes_by_image = group_boxes(boxes_df)
    if image_ids is None:
        image_ids = sorted(boxes_by_image.keys())

    rows = {}
    for image_id in image_ids:
        present = {box.canonical_class for box in boxes_by_image.get(image_id, [])}
        rows[image_id] = {c: (1 if c in present else 0) for c in concept_names}
    return pd.DataFrame.from_dict(rows, orient="index", columns=concept_names)


def image_labels_to_multihot(
    labels_df: pd.DataFrame,
    class_names: List[str],
    image_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Multi-hot matrix from the global image-label file.

    Returns a DataFrame indexed by image_id with one column per class in
    ``class_names`` (0/1). Missing rows are treated as 0.
    """
    if not class_names:
        raise ValueError("class_names must be non-empty")
    if image_ids is None:
        image_ids = find_image_ids(labels_df)

    labels_df = labels_df.copy()
    labels_df["image_id"] = labels_df["image_id"].astype(str)
    labels_df["canonical"] = labels_df["class_name"].map(normalize_class_name)

    known = set(class_names)
    unknown = sorted(set(labels_df["canonical"]) - known)
    if unknown:
        print(
            f"[annotations] ignoring labels not in requested classes: {unknown}"
        )

    df = labels_df[labels_df["canonical"].isin(known)]
    pivoted = df.pivot_table(
        index="image_id", columns="canonical", values="class_name", aggfunc="count"
    ).fillna(0)
    pivoted = (pivoted > 0).astype(int)
    return pivoted.reindex(index=image_ids, columns=class_names, fill_value=0)


def concept_prevalence(concept_matrix: pd.DataFrame) -> pd.Series:
    """Fraction of positive examples per concept."""
    return concept_matrix.mean(axis=0)
