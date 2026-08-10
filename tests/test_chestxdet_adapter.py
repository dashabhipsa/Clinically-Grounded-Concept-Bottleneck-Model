"""Unit tests for the remote ChestX-Det adapter's decoding logic.

These tests never hit the network: ``load_id2label`` is monkeypatched and
the label/mask decoding is exercised with synthetic PIL images.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import src.data.chestxdet as cxd

FAKE_ID2LABEL = {
    255: "Background",
    1: "Atelectasis",
    3: "Cardiomegaly",
    4: "Consolidation",
    13: "Pneumothorax",
}


@pytest.fixture(autouse=True)
def _fake_id2label(monkeypatch):
    monkeypatch.setattr(cxd, "load_id2label", lambda repo_id="x": dict(FAKE_ID2LABEL))


def make_label_map(shape=(64, 64)):
    arr = np.full(shape, 255, dtype=np.uint16)
    arr[10:20, 10:20] = 3   # cardiomegaly blob
    arr[40:50, 40:50] = 13  # pneumothorax blob
    return arr


def test_chestx_det_classes_excludes_background():
    assert cxd.chestx_det_classes() == ["atelectasis", "cardiomegaly", "consolidation", "pneumothorax"]


def test_image_to_grayscale_handles_rgba_and_l():
    rgba = Image.new("RGBA", (16, 16), (200, 100, 50, 255))
    out = cxd.image_to_grayscale(rgba)
    assert out.mode == "L"
    assert np.asarray(out).min() == np.asarray(out).max()  # flat colour


def test_decode_label_map_roundtrip_and_resize():
    pil = Image.fromarray(make_label_map())
    out = cxd.decode_label_map(pil, size=32)
    assert out.shape == (32, 32)
    assert out.dtype == np.uint16
    assert 3 in np.unique(out) and 13 in np.unique(out)


def test_label_map_to_multihot():
    vec = cxd.label_map_to_multihot(make_label_map(), [1, 3, 4, 13])
    assert vec.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_class_masks_from_label():
    masks = cxd.class_masks_from_label(make_label_map(), [3, 13])
    assert masks.shape == (2, 64, 64)
    assert masks[0].sum() == 100  # 10x10 cardiomegaly blob
    assert masks[1].sum() == 100  # 10x10 pneumothorax blob


def test_instances_and_boxes_from_mask():
    label_map = make_label_map()
    mask = np.zeros((64, 64, 3), dtype=np.uint8)
    mask[10:20, 10:20] = (128, 0, 0)    # instance 1
    mask[40:50, 40:50] = (0, 255, 255)  # instance 2
    boxes = cxd.boxes_from_mask_instances(mask, label_map, "img-0", FAKE_ID2LABEL)
    assert len(boxes) == 2
    by_class = {b.canonical_class: b for b in boxes}
    assert set(by_class) == {"cardiomegaly", "pneumothorax"}
    b = by_class["cardiomegaly"]
    assert b.class_id == 3
    assert b.class_name == "Cardiomegaly"
    assert abs(b.x_min - 10 / 63) < 1e-9
    assert abs(b.x_max - 19 / 63) < 1e-9
    assert abs(b.y_min - 10 / 63) < 1e-9
    assert abs(b.y_max - 19 / 63) < 1e-9


def test_background_only_instance_skipped():
    label_map = np.full((32, 32), 255, dtype=np.uint16)
    mask = np.zeros((32, 32, 3), dtype=np.uint8)
    mask[5:15, 5:15] = (1, 2, 3)
    boxes = cxd.boxes_from_mask_instances(mask, label_map, "img-0", FAKE_ID2LABEL)
    assert boxes == []


def test_dataset_decode_row_matches_pipeline_interface():
    ds = cxd.ChestXDetRemoteDataset(split="train", image_size=32, max_samples=1)
    row = {
        "image": Image.fromarray(make_label_map().astype(np.uint8)),
        "label": Image.fromarray(make_label_map()),
        "mask": Image.fromarray(
            np.zeros((64, 64, 3), dtype=np.uint8)
        ),
    }
    out = ds.decode_row(row, "train-00000", ds._default_transform())
    assert set(out) == {"image", "concepts", "diagnosis", "boxes", "mask", "image_id"}
    assert out["image"].shape == (3, 32, 32)
    assert out["concepts"].shape == (len(ds.concept_names),)
    assert out["concepts"][1] == 1.0 and out["concepts"][0] == 0.0  # cardiomegaly present
    assert out["diagnosis"].tolist() == out["concepts"].tolist()
    assert out["mask"].shape == (len(ds.concept_names), 32, 32)
    assert out["image_id"] == "train-00000"
    assert isinstance(out["boxes"], list)
