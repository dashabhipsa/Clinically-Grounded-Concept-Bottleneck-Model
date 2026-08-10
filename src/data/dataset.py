"""PyTorch Dataset for the VinDr-CXR chest X-ray images.

Each sample returns:

    {
        "image":     float tensor [3, H, W] (normalized),
        "concepts":  float tensor [num_concepts]  (multi-hot local findings),
        "diagnosis": float tensor [num_diagnoses] (multi-hot global labels),
        "boxes":     list[Box] (preserved for later phases),
        "image_id":  str,
    }
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.annotations import (
    Box,
    concepts_from_boxes,
    group_boxes,
    image_labels_to_multihot,
    load_boxes,
    load_image_labels,
)
from src.data.preprocessing import dicom_to_image, resize_image
from src.data.transforms import build_transform


class VinDrCXRDataset(Dataset):
    """Iterate over a list of VinDr-CXR image ids.

    Parameters
    ----------
    root:
        Root of the VinDr-CXR dataset (see config ``data.root``).
    image_ids:
        Ordered list of image ids belonging to this split.
    concepts:
        Concept class names (canonical form) used for the concept layer.
    diagnoses:
        Diagnosis class names (canonical form) used for the diagnosis head.
    split:
        One of "train", "val", "test". ``"test"`` reads annotations from the
        official ``test`` files; anything else uses the official ``train``
        files (validation ids are a subset of the official training set).
    image_size:
        Square size the image is resized to before normalization.
    transform:
        Optional callable (PIL image -> tensor). If None, the default
        non-augmented normalized transform is used.
    """

    def __init__(
        self,
        root: str | Path,
        image_ids: List[str],
        concepts: List[str],
        diagnoses: List[str],
        split: str = "train",
        image_size: int = 512,
        transform=None,
    ):
        if not concepts:
            raise ValueError("`concepts` must be non-empty")
        if not diagnoses:
            raise ValueError("`diagnoses` must be non-empty")
        if not image_ids:
            raise ValueError("`image_ids` must be non-empty")

        self.root = Path(root)
        self.image_ids = [str(i) for i in image_ids]
        self.concepts = list(concepts)
        self.diagnoses = list(diagnoses)
        self.split = split
        self.image_size = int(image_size)
        self.transform = transform

        images_dir = self.root / "images"
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        # Official test images live in the test annotation files; validation
        # ids are a subset of the official training set.
        is_test = split == "test"
        boxes_df = load_boxes(self.root / "annotations" / ("test.csv" if is_test else "train.csv"))
        labels_df = load_image_labels(
            self.root / "annotations" / (f"image_labels_{split}.csv" if is_test else "image_labels_train.csv")
        )

        self.concept_df = concepts_from_boxes(boxes_df, self.concepts, image_ids=self.image_ids)
        self.diagnosis_df = image_labels_to_multihot(
            labels_df, self.diagnoses, image_ids=self.image_ids
        )
        self.boxes_by_image = group_boxes(boxes_df)

    def __len__(self) -> int:
        return len(self.image_ids)

    def _image_path(self, image_id: str) -> Path:
        return self.root / "images" / f"{image_id}.dicom"

    def _default_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return build_transform(self.image_size, mean, std, train=False, augment=False)

    def __getitem__(self, index: int) -> dict:
        image_id = self.image_ids[index]

        img = dicom_to_image(self._image_path(image_id))
        if img.size != (self.image_size, self.image_size):
            img = resize_image(img, self.image_size)

        transform = self.transform if self.transform is not None else self._default_transform()
        image = transform(img)

        concepts = torch.as_tensor(
            self.concept_df.loc[image_id].to_numpy(dtype="float32")
        )
        diagnosis = torch.as_tensor(
            self.diagnosis_df.loc[image_id].to_numpy(dtype="float32")
        )
        boxes: List[Box] = list(self.boxes_by_image.get(image_id, []))

        return {
            "image": image,
            "concepts": concepts,
            "diagnosis": diagnosis,
            "boxes": boxes,
            "image_id": image_id,
        }

    def class_counts(self, target: str = "diagnosis") -> pd.Series:
        """Positive-example counts per class (for diagnostics)."""
        df = self.diagnosis_df if target == "diagnosis" else self.concept_df
        return df.sum(axis=0)


def collate_vindr(batch: List[dict]) -> dict:
    """Collate function preserving variable-length boxes and string ids."""
    images = torch.stack([b["image"] for b in batch])
    concepts = torch.stack([b["concepts"] for b in batch])
    diagnosis = torch.stack([b["diagnosis"] for b in batch])
    boxes = [b["boxes"] for b in batch]
    image_ids = [b["image_id"] for b in batch]
    return {
        "image": images,
        "concepts": concepts,
        "diagnosis": diagnosis,
        "boxes": boxes,
        "image_id": image_ids,
    }
