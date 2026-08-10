"""Tests for bounding-box -> activation-mask handling and coordinate
transforms after image resizing."""
import numpy as np
import pytest
import torch

from src.data.annotations import Box
from src.data.spatial import (
    boxes_to_masks,
    boxes_to_masks_tensor,
    concept_presence_from_masks,
    normalized_box_to_pixels,
    smooth_masks,
)

from conftest import CONCEPT_NAMES

NAME_TO_IDX = {name: i for i, name in enumerate(CONCEPT_NAMES)}


def _box(name="cardiomegaly", x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75):
    return Box(
        image_id="0001", class_id=0, class_name=name,
        x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, rad_id=0,
    )


def test_normalized_box_to_pixels_after_resize():
    """Normalized coords map to the same fractional location at any grid."""
    # (0.25, 0.25, 0.75, 0.75) on a 16x16 grid -> pixels 4..11.
    box = _box()
    x0, y0, x1, y1 = normalized_box_to_pixels(box, 16, 16)
    assert (x0, y0, x1, y1) == (4, 4, 11, 11)

    # Same fractional box on a 32x32 grid -> pixels 8..23 (uniform scale).
    x0, y0, x1, y1 = normalized_box_to_pixels(box, 32, 32)
    assert (x0, y0, x1, y1) == (8, 8, 23, 23)


def test_normalized_box_to_pixels_clips_out_of_range():
    # Box validation rejects out-of-range coords at construction, so inject
    # them directly to test the clamping in normalized_box_to_pixels.
    box = _box()
    object.__setattr__(box, "x_min", -0.1)
    object.__setattr__(box, "x_max", 1.5)
    object.__setattr__(box, "y_min", 0.4)
    object.__setattr__(box, "y_max", 1.2)
    x0, y0, x1, y1 = normalized_box_to_pixels(box, 16, 16)
    assert x0 >= 0 and x1 <= 15
    assert y0 >= 0 and y1 <= 15
    assert x0 <= x1 and y0 <= y1


def test_boxes_to_masks_shape_and_fill():
    masks = boxes_to_masks([_box()], CONCEPT_NAMES, cam_height=16, cam_width=16)
    assert masks.shape == (len(CONCEPT_NAMES), 16, 16)
    k = NAME_TO_IDX["cardiomegaly"]
    # Inside the box -> 1; outside -> 0.
    assert masks[k, 4, 4] == 1.0
    assert masks[k, 11, 11] == 1.0
    assert masks[k, 0, 0] == 0.0
    # Concepts without boxes stay all-zero.
    assert (masks[NAME_TO_IDX["edema"]] == 0).all()


def test_boxes_to_masks_multiple_boxes_union():
    box_a = _box(name="cardiomegaly", x_min=0.0, y_min=0.0, x_max=0.3, y_max=0.3)
    box_b = _box(name="cardiomegaly", x_min=0.6, y_min=0.6, x_max=1.0, y_max=1.0)
    masks = boxes_to_masks([box_a, box_b], CONCEPT_NAMES, 16, 16)
    k = NAME_TO_IDX["cardiomegaly"]
    assert masks[k, 0, 0] == 1.0
    assert masks[k, 15, 15] == 1.0
    # Gap in between stays zero.
    assert masks[k, 8, 8] == 0.0


def test_boxes_to_masks_unknown_concept_ignored():
    box = _box(name="not_a_concept")
    masks = boxes_to_masks([box], CONCEPT_NAMES, 8, 8)
    assert (masks == 0).all()


def test_boxes_to_masks_tensor_device_and_smoothing():
    boxes_list = [[_box()], [], [_box(name="pneumothorax")]]
    masks = boxes_to_masks_tensor(boxes_list, CONCEPT_NAMES, 8, 8)
    assert isinstance(masks, torch.Tensor)
    assert masks.shape == (3, len(CONCEPT_NAMES), 8, 8)
    # Empty sample -> no concept present.
    presence = concept_presence_from_masks(masks)
    assert not bool(presence[1].any())
    assert bool(presence[0, NAME_TO_IDX["cardiomegaly"]])

    smooth = boxes_to_masks_tensor(boxes_list, CONCEPT_NAMES, 8, 8, smooth_sigma=1.0)
    assert smooth.shape == masks.shape
    assert float(smooth[0, NAME_TO_IDX["cardiomegaly"]].max()) <= 1.0
    # Gaussian smoothing is normalized but zero-padded at the borders, so the
    # smoothed mask retains (almost) all the original mass.
    hard_sum = float(masks[0, NAME_TO_IDX["cardiomegaly"]].sum())
    soft_sum = float(smooth[0, NAME_TO_IDX["cardiomegaly"]].sum())
    assert soft_sum == pytest.approx(hard_sum, rel=0.05)


def test_smooth_masks_preserves_shape_and_is_soft():
    mask = torch.zeros(1, 1, 16, 16)
    mask[0, 0, 6:10, 6:10] = 1.0
    out = smooth_masks(mask, sigma=1.0)
    assert out.shape == mask.shape
    # Smoothing spreads energy beyond the hard box and softens the edges.
    assert float(out[0, 0, 4, 4]) > 0.0
    assert float(out[0, 0, 4, 4]) < 0.5
    # Box interior stays (nearly) saturated.
    assert float(out[0, 0, 8, 8]) > 0.8
    # sigma <= 0 returns the input unchanged.
    assert torch.equal(smooth_masks(mask, sigma=0.0), mask)
