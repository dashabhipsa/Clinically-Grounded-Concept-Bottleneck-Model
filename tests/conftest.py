"""Shared fixtures: a small synthetic VinDr-CXR tree with valid DICOMs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

CONCEPT_NAMES = [
    "cardiomegaly",
    "pleural_effusion",
    "consolidation",
    "lung_opacity",
    "edema",
    "atelectasis",
    "pneumothorax",
    "nodule_mass",
]

DIAGNOSIS_NAMES = [
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

BOX_COLS = ["image_id", "class_id", "class_name", "x_max", "x_min", "y_max", "y_min", "rad_id"]


def make_dicom(path: Path, shape=(64, 64), seed: int = 0) -> None:
    rng = np.random.RandomState(seed)
    arr = (rng.rand(*shape) * 65535).astype("<u2")

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns = shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"
    ds.PatientName = "Synthetic"
    ds.PixelData = arr.tobytes()
    try:  # pydicom >= 3.0
        ds.save_as(str(path), enforce_file_format=True)
    except TypeError:  # pydicom 2.x
        ds.save_as(str(path), write_like_original=False)


@pytest.fixture
def synthetic_vindr(tmp_path):
    root = tmp_path / "vindr-cxr"
    images = root / "images"
    images.mkdir(parents=True)
    annotations = root / "annotations"
    annotations.mkdir()

    train_ids = [f"{i:04d}" for i in range(1, 7)]   # 0001..0006
    test_ids = [f"{i:04d}" for i in range(7, 10)]    # 0007..0009
    all_ids = train_ids + test_ids
    for i, image_id in enumerate(all_ids):
        make_dicom(images / f"{image_id}.dicom", seed=i)

    # --- local (bounding-box) findings, official train set ---
    train_boxes = pd.DataFrame(
        [
            # image_id, class_name, x_min, y_min, x_max, y_max
            ("0001", "Cardiomegaly", 0.20, 0.20, 0.50, 0.55),
            ("0001", "Pleural effusion", 0.10, 0.30, 0.40, 0.60),
            ("0002", "Lung opacity", 0.25, 0.15, 0.60, 0.45),
            ("0003", "Nodule/Mass", 0.30, 0.35, 0.45, 0.50),
            ("0003", "Atelectasis", 0.05, 0.05, 0.20, 0.25),
            ("0004", "Cardiomegaly", 0.10, 0.40, 0.35, 0.70),
            ("0005", "Pneumothorax", 0.50, 0.10, 0.90, 0.40),
            ("0006", "Consolidation", 0.20, 0.20, 0.55, 0.50),
        ],
        columns=["image_id", "class_name", "x_min", "y_min", "x_max", "y_max"],
    )
    train_boxes.insert(1, "class_id", range(len(train_boxes)))
    train_boxes["rad_id"] = 0
    train_boxes.to_csv(annotations / "train.csv", index=False, columns=BOX_COLS)

    # --- local findings, official test set (0008 has none) ---
    test_boxes = pd.DataFrame(
        [
            ("0007", "Pleural effusion", 0.10, 0.10, 0.40, 0.50),
            ("0009", "Cardiomegaly", 0.30, 0.20, 0.60, 0.55),
        ],
        columns=["image_id", "class_name", "x_min", "y_min", "x_max", "y_max"],
    )
    test_boxes.insert(1, "class_id", range(len(test_boxes)))
    test_boxes["rad_id"] = 0
    test_boxes.to_csv(annotations / "test.csv", index=False, columns=BOX_COLS)

    # --- global diagnostic labels ---
    train_labels = pd.DataFrame(
        [
            ("0001", "Cardiomegaly"),
            ("0001", "Pleural effusion"),
            ("0002", "Lung opacity"),
            ("0003", "Nodule/Mass"),
            ("0003", "Atelectasis"),
            ("0004", "Cardiomegaly"),
            ("0005", "Pneumothorax"),
            ("0006", "Consolidation"),
        ],
        columns=["image_id", "class_name"],
    )
    train_labels.to_csv(annotations / "image_labels_train.csv", index=False)

    test_labels = pd.DataFrame(
        [
            ("0007", "Pleural effusion"),
            ("0009", "Cardiomegaly"),
        ],
        columns=["image_id", "class_name"],
    )
    test_labels.to_csv(annotations / "image_labels_test.csv", index=False)

    # --- metadata ---
    metadata = pd.DataFrame(
        {"image_id": all_ids, "patient_id": all_ids, "study_id": [f"s{i}" for i in range(len(all_ids))]}
    )
    metadata.to_csv(root / "metadata.csv", index=False)

    return root
