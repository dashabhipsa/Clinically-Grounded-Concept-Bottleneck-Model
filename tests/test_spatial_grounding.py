"""Tests for the SpatialGroundingLoss (soft grounding via ELS)."""
import numpy as np
import pytest
import torch

from src.data.annotations import Box
from src.models.spatial_cbm import build_spatial_cbm
from src.training.spatial_losses import SpatialGroundingLoss, build_spatial_loss_fn
from src.utils.config import Config

from conftest import CONCEPT_NAMES, DIAGNOSIS_NAMES

NUM_CONCEPTS = len(CONCEPT_NAMES)
NUM_DIAGNOSES = len(DIAGNOSIS_NAMES)


def _config(grounding_weight=0.5, box_corruption="correct"):
    return Config(
        {
            "seed": 42,
            "experiment": "spatial_cbm",
            "model": {
                "encoder": "densenet121",
                "pretrained": False,
                "concept_activation": "none",
                "concept_pool": "avg",
                "grounding": {
                    "weight": grounding_weight,
                    "background_weight": 0.0,
                    "mask_smoothing_sigma": 0.0,
                    "box_corruption": box_corruption,
                },
            },
            "training": {"lambda_concept": 1.0, "lambda_diagnosis": 1.0},
            "concepts": CONCEPT_NAMES,
            "diagnoses": DIAGNOSIS_NAMES,
        }
    )


def _box(name="cardiomegaly", x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75):
    return Box(
        image_id="0001", class_id=0, class_name=name,
        x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, rad_id=0,
    )


def _labels(batch_size):
    return {
        "concepts": torch.zeros(batch_size, NUM_CONCEPTS),
        "diagnosis": torch.zeros(batch_size, NUM_DIAGNOSES),
    }


def _batch(boxes_list):
    return {"boxes": boxes_list}


def _loss_value(config, model, x, labels, batch=None):
    loss_fn = build_spatial_loss_fn(config)
    return float(loss_fn(model, x, labels, batch=batch)[0].detach())


def test_grounding_loss_zero_when_evidence_inside_box():
    cfg = _config(grounding_weight=1.0)
    model = build_spatial_cbm(cfg)
    with torch.no_grad():
        # Perfect concept head: activation energy at every pixel. With a box
        # spanning (0.25..0.75) on a 2x2 CAM grid all pixels lie inside the
        # box -> ELS = 1 -> grounding term = 0.
        model.spatial_head.conv.weight.zero_()
        model.spatial_head.conv.bias.zero_()
        model.spatial_head.conv.bias[0] = 5.0
        model.spatial_head.conv.bias[1] = 5.0

    x = torch.randn(1, 3, 64, 64)
    labels = _labels(1)
    batch = _batch([[_box(), _box(name="pleural_effusion")]])
    loss = _loss_value(cfg, model, x, labels, batch)
    loss_no_grounding = _loss_value(cfg, model, x, labels, None)
    # Grounding term is exactly 0 -> both losses agree.
    assert loss == pytest.approx(loss_no_grounding, abs=1e-4)


def test_grounding_loss_penalizes_out_of_box_evidence():
    cfg = _config(grounding_weight=1.0)
    model = build_spatial_cbm(cfg)
    with torch.no_grad():
        model.spatial_head.conv.weight.zero_()
        model.spatial_head.conv.bias.zero_()
        # Put energy at every pixel. A box limited to the top-left half of
        # the 2x2 grid (0..0.5 -> pixel 0) means most energy is OUTSIDE.
        model.spatial_head.conv.bias[0] = 5.0

    # Box only in the top-left quadrant -> on 2x2 grid pixel (0,0).
    box = _box(x_min=0.0, y_min=0.0, x_max=0.5, y_max=0.5)
    x = torch.randn(1, 3, 64, 64)
    labels = _labels(1)
    batch = _batch([[box]])
    loss = _loss_value(cfg, model, x, labels, batch)
    loss_no_grounding = _loss_value(cfg, model, x, labels, None)
    assert loss > loss_no_grounding + 1e-3


def test_grounding_loss_ignores_concepts_without_boxes():
    cfg = _config(grounding_weight=1.0)
    model = build_spatial_cbm(cfg)
    with torch.no_grad():
        model.spatial_head.conv.weight.zero_()
        model.spatial_head.conv.bias.zero_()
        model.spatial_head.conv.bias[0] = 5.0

    x = torch.randn(1, 3, 64, 64)
    labels = _labels(1)
    # No boxes at all -> grounding term is skipped.
    batch = _batch([[]])
    assert _loss_value(cfg, model, x, labels, batch) == pytest.approx(
        _loss_value(cfg, model, x, labels, None), abs=1e-4
    )


def test_grounding_weight_scales_the_term():
    model = build_spatial_cbm(_config(grounding_weight=1.0))
    with torch.no_grad():
        model.spatial_head.conv.weight.zero_()
        model.spatial_head.conv.bias.zero_()
        model.spatial_head.conv.bias[0] = 5.0
        model.spatial_head.conv.bias[1] = 5.0

    x = torch.randn(1, 3, 64, 64)
    labels = _labels(1)
    batch = _batch([
        [
            _box(x_min=0.0, y_min=0.0, x_max=0.5, y_max=0.5),
            _box(name="pleural_effusion", x_min=0.0, y_min=0.0,
                 x_max=0.5, y_max=0.5),
        ]
    ])

    loss_base = _loss_value(_config(grounding_weight=1.0), model, x, labels, None)
    loss_w05 = _loss_value(_config(grounding_weight=0.5), model, x, labels, batch)
    loss_w1 = _loss_value(_config(grounding_weight=1.0), model, x, labels, batch)
    # Grounding term scales linearly with grounding_weight.
    assert loss_w05 - loss_base == pytest.approx((loss_w1 - loss_base) * 0.5, rel=1e-3)


def test_box_corruption_none_drops_supervision():
    cfg = _config(grounding_weight=1.0, box_corruption="none")
    model = build_spatial_cbm(cfg)
    with torch.no_grad():
        model.spatial_head.conv.weight.zero_()
        model.spatial_head.conv.bias.zero_()
        model.spatial_head.conv.bias[0] = 5.0

    x = torch.randn(1, 3, 64, 64)
    labels = _labels(1)
    batch = _batch([[_box()]])
    # "none" removes all boxes -> grounding term is skipped.
    assert _loss_value(cfg, model, x, labels, batch) == pytest.approx(
        _loss_value(cfg, model, x, labels, None), abs=1e-4
    )


def test_box_corruption_shifted_is_deterministic():
    model = build_spatial_cbm(_config(grounding_weight=1.0, box_corruption="shifted"))
    with torch.no_grad():
        model.spatial_head.conv.weight.zero_()
        model.spatial_head.conv.bias.zero_()
        model.spatial_head.conv.bias[0] = 5.0

    x = torch.randn(1, 3, 64, 64)
    labels = _labels(1)
    batch = _batch([[_box()]])
    la = _loss_value(
        _config(grounding_weight=1.0, box_corruption="shifted"),
        model, x, labels, batch,
    )
    lb = _loss_value(
        _config(grounding_weight=1.0, box_corruption="shifted"),
        model, x, labels, batch,
    )
    assert la == pytest.approx(lb, abs=1e-6)


def test_build_spatial_loss_fn_config():
    cfg = _config(grounding_weight=0.25)
    loss_fn = build_spatial_loss_fn(cfg)
    assert isinstance(loss_fn, SpatialGroundingLoss)
    assert loss_fn.grounding_weight == 0.25
