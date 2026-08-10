"""Tests for spatial fidelity metrics: IoU, Pointing Game, ELS."""
import numpy as np
import pytest

from src.data.annotations import Box
from src.evaluation.spatial_metrics import (
    box_iou,
    cam_binary_mask,
    els,
    evaluate_spatial_metrics,
    mask_bbox,
    pointing_hit,
)

from conftest import CONCEPT_NAMES

NAME_TO_IDX = {name: i for i, name in enumerate(CONCEPT_NAMES)}


def _box(name="cardiomegaly", x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75):
    return Box(
        image_id="0001", class_id=0, class_name=name,
        x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, rad_id=0,
    )


def test_box_iou_perfect_overlap():
    assert box_iou((2, 2, 6, 6), (2, 2, 6, 6)) == pytest.approx(1.0)


def test_box_iou_disjoint():
    assert box_iou((0, 0, 2, 2), (8, 8, 10, 10)) == pytest.approx(0.0)


def test_box_iou_known_partial():
    # GT 4x4 at (0,0), pred 4x4 at (2,2): intersection 2x2, union 4x4+4x4-4=28.
    gt = (0, 0, 3, 3)
    pred = (2, 2, 5, 5)
    expected = (4) / (16 + 16 - 4)
    assert box_iou(gt, pred) == pytest.approx(expected)


def test_cam_binary_mask_and_bbox():
    cam = np.zeros((8, 8))
    cam[2:6, 3:7] = 1.0
    mask = cam_binary_mask(cam, threshold=0.5)
    assert mask is not None
    assert mask_bbox(mask) == (3, 2, 6, 5)
    # All-zero CAM -> no mask.
    assert cam_binary_mask(np.zeros((4, 4))) is None
    assert mask_bbox(None) is None


def test_pointing_game_hit_and_miss():
    cam = np.zeros((8, 8))
    cam[2, 3] = 10.0  # strongest activation at (x=3, y=2)
    gt = (3, 2, 4, 3)  # contains (3,2)
    assert pointing_hit(cam, gt) == 1
    gt_out = (0, 0, 1, 1)
    assert pointing_hit(cam, gt_out) == 0
    # Zero activation -> no hit.
    assert pointing_hit(np.zeros((8, 8)), gt_out) == 0


def test_els_energy_ratios():
    cam = np.ones((8, 8))
    # Box covering the whole map -> ELS 1.
    assert els(cam, (0, 0, 7, 7)) == pytest.approx(1.0)
    # Box covering exactly half the map -> ELS 0.5.
    assert els(cam, (0, 0, 7, 3)) == pytest.approx(0.5)
    # Zero activation -> NaN.
    assert np.isnan(els(np.zeros((8, 8)), (0, 0, 7, 7)))


def test_evaluate_spatial_metrics_per_class_and_macro():
    k_cardio = NAME_TO_IDX["cardiomegaly"]
    k_effusion = NAME_TO_IDX["pleural_effusion"]
    # 2 images, K concept maps at 8x8.
    maps = np.zeros((2, len(CONCEPT_NAMES), 8, 8))
    # Image 0: cardiomegaly evidence exactly inside its box.
    maps[0, k_cardio, 2:6, 2:6] = 1.0
    # Image 1: pleural effusion evidence half inside its box (16 units inside,
    # 16 units outside) -> ELS = 0.5.
    maps[1, k_effusion, 2:6, 2:6] = 1.0
    maps[1, k_effusion, 6:8, 0:8] = 1.0

    boxes = [
        [_box(name="cardiomegaly", x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75)],
        [_box(name="pleural_effusion", x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75)],
    ]

    result = evaluate_spatial_metrics(maps, boxes, CONCEPT_NAMES)
    pc = result["per_class"]

    assert pc["cardiomegaly"]["num_instances"] == 1
    assert pc["cardiomegaly"]["iou"] == pytest.approx(1.0)
    assert pc["cardiomegaly"]["pointing_game"] == pytest.approx(1.0)
    assert pc["cardiomegaly"]["els"] == pytest.approx(1.0)

    assert pc["pleural_effusion"]["num_instances"] == 1
    # Energy split evenly inside/outside -> ELS 0.5.
    assert pc["pleural_effusion"]["els"] == pytest.approx(0.5)
    # Predicted box = bbox of thresholded CAM = (0, 2, 7, 7), area 48;
    # GT box = (2, 2, 5, 5), area 16; intersection area 16 -> IoU 16/48.
    iou_effusion = 16 / 48
    assert pc["pleural_effusion"]["iou"] == pytest.approx(iou_effusion)

    # Macro = mean over the two scored concepts.
    macro = result["macro"]
    assert macro["iou"] == pytest.approx((1.0 + iou_effusion) / 2)
    assert macro["els"] == pytest.approx((1.0 + 0.5) / 2)
    assert macro["pointing_game"] == pytest.approx(1.0)


def test_evaluate_spatial_metrics_skips_unscored_concepts():
    maps = np.zeros((1, len(CONCEPT_NAMES), 4, 4))
    maps[0, NAME_TO_IDX["cardiomegaly"], 1:3, 1:3] = 1.0
    boxes = [[_box(name="cardiomegaly", x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75)]]
    result = evaluate_spatial_metrics(maps, boxes, CONCEPT_NAMES)
    pc = result["per_class"]
    # Unscored concepts report nan but macro averages only the scored ones.
    assert np.isnan(pc["edema"]["iou"])
    assert not np.isnan(result["macro"]["iou"])


def test_evaluate_spatial_metrics_no_boxes():
    maps = np.zeros((1, len(CONCEPT_NAMES), 4, 4))
    result = evaluate_spatial_metrics(maps, [[]], CONCEPT_NAMES)
    assert np.isnan(result["macro"]["iou"])
    assert np.isnan(result["macro"]["els"])
    assert np.isnan(result["macro"]["pointing_game"])
