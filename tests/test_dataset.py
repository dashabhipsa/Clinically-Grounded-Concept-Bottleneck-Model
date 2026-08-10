"""Tests for the PyTorch Dataset loader and collate function."""
import torch

from src.data.dataset import VinDrCXRDataset, collate_vindr
from src.data.split import create_splits, load_split_ids

from conftest import CONCEPT_NAMES, DIAGNOSIS_NAMES


def _build_ds(root, tmp_path, split="train"):
    splits_dir = tmp_path / "splits"
    create_splits(root, splits_dir, val_fraction=0.25, seed=42)
    splits = load_split_ids(splits_dir)
    ids = splits["val"] if split == "val" else splits[split]
    return VinDrCXRDataset(
        root, ids, CONCEPT_NAMES, DIAGNOSIS_NAMES, split=split, image_size=64
    )


def test_dataset_item_structure(synthetic_vindr, tmp_path):
    ds = _build_ds(synthetic_vindr, tmp_path, "train")
    item = ds[0]
    assert set(item.keys()) == {"image", "concepts", "diagnosis", "boxes", "image_id"}
    assert item["image"].shape == (3, 64, 64)
    assert item["concepts"].shape == (len(CONCEPT_NAMES),)
    assert item["diagnosis"].shape == (len(DIAGNOSIS_NAMES),)
    assert item["concepts"].dtype == torch.float32
    assert item["diagnosis"].dtype == torch.float32
    assert isinstance(item["image_id"], str)
    assert isinstance(item["boxes"], list)


def test_dataset_multilabel_concepts(synthetic_vindr, tmp_path):
    from src.data.split import official_split_ids

    ids = official_split_ids(synthetic_vindr)["train"]
    ds = VinDrCXRDataset(
        synthetic_vindr, ids, CONCEPT_NAMES, DIAGNOSIS_NAMES,
        split="train", image_size=64,
    )
    idx = ds.image_ids.index("0001")
    item = ds[idx]
    concepts = item["concepts"].tolist()
    idx = {c: i for i, c in enumerate(CONCEPT_NAMES)}
    assert concepts[idx["cardiomegaly"]] == 1.0
    assert concepts[idx["pleural_effusion"]] == 1.0
    # edema is not annotated anywhere -> 0
    assert concepts[idx["edema"]] == 0.0


def test_dataset_missing_findings(synthetic_vindr, tmp_path):
    ds = VinDrCXRDataset(
        synthetic_vindr, ["0008"], CONCEPT_NAMES, DIAGNOSIS_NAMES,
        split="test", image_size=64,
    )
    item = ds[0]
    assert (item["concepts"] == 0).all()
    assert (item["diagnosis"] == 0).all()
    assert item["boxes"] == []


def test_boxes_preserved(synthetic_vindr, tmp_path):
    ds = _build_ds(synthetic_vindr, tmp_path, "train")
    # Find an item that carries at least one bounding box.
    idx = next(i for i in range(len(ds)) if ds[i]["boxes"])
    item = ds[idx]
    assert len(item["boxes"]) >= 1
    box = item["boxes"][0]
    assert box.image_id == item["image_id"]
    assert 0.0 <= box.x_min <= 1.0


def test_collate_vindr(synthetic_vindr, tmp_path):
    ds = _build_ds(synthetic_vindr, tmp_path, "train")
    batch = collate_vindr([ds[0], ds[1], ds[2]])
    assert batch["image"].shape == (3, 3, 64, 64)
    assert batch["concepts"].shape == (3, len(CONCEPT_NAMES))
    assert batch["diagnosis"].shape == (3, len(DIAGNOSIS_NAMES))
    assert len(batch["boxes"]) == 3
    assert isinstance(batch["image_id"], list)


def test_transform_reproducible(synthetic_vindr, tmp_path):
    from src.data.transforms import build_transform

    transform = build_transform(64, train=True, augment=True)
    ds1 = VinDrCXRDataset(
        synthetic_vindr, ["0001"], CONCEPT_NAMES, DIAGNOSIS_NAMES,
        split="train", image_size=64, transform=transform,
    )
    torch.manual_seed(0)
    a = ds1[0]["image"]
    torch.manual_seed(0)
    b = ds1[0]["image"]
    assert torch.allclose(a, b)
