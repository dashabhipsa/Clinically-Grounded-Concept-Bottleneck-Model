"""Tests for the Spatially Grounded CBM forward pass, activation-map
dimensions, and bottleneck (no-bypass) integrity."""
import pytest
import torch

from src.models import build_model
from src.models.cbm import DiagnosisHead
from src.models.spatial_cbm import (
    SpatialCBMOutput,
    SpatialConceptHead,
    SpatiallyGroundedCBM,
    build_spatial_cbm,
)
from src.utils.config import Config

from conftest import CONCEPT_NAMES, DIAGNOSIS_NAMES

NUM_CONCEPTS = len(CONCEPT_NAMES)
NUM_DIAGNOSES = len(DIAGNOSIS_NAMES)


def _config(concept_activation="sigmoid", concept_pool="avg"):
    return Config(
        {
            "experiment": "spatial_cbm",
            "model": {
                "encoder": "densenet121",
                "pretrained": False,
                "concept_activation": concept_activation,
                "concept_pool": concept_pool,
            },
            "concepts": CONCEPT_NAMES,
            "diagnoses": DIAGNOSIS_NAMES,
        }
    )


def test_spatial_concept_head_outputs_shapes():
    head = SpatialConceptHead(in_channels=128, num_concepts=NUM_CONCEPTS, pool="avg")
    feature_map = torch.randn(2, 128, 8, 8)
    cams, logits = head(feature_map)
    assert cams.shape == (2, NUM_CONCEPTS, 8, 8)
    assert logits.shape == (2, NUM_CONCEPTS)


def test_spatial_concept_head_maps_are_non_negative():
    head = SpatialConceptHead(in_channels=64, num_concepts=5)
    cams, _ = head(torch.randn(3, 64, 6, 6))
    assert bool((cams >= 0.0).all())


def test_spatial_concept_head_max_pool():
    head = SpatialConceptHead(in_channels=32, num_concepts=4, pool="max")
    _, logits = head(torch.randn(2, 32, 8, 8))
    assert logits.shape == (2, 4)


def test_spatial_concept_head_unknown_pool_raises():
    with pytest.raises(ValueError):
        SpatialConceptHead(in_channels=32, num_concepts=4, pool="median")


def test_spatial_cbm_forward_returns_all_outputs():
    model = build_spatial_cbm(_config())
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert isinstance(out, SpatialCBMOutput)
    assert out.concept_logits.shape == (2, NUM_CONCEPTS)
    assert out.concept_probs.shape == (2, NUM_CONCEPTS)
    assert out.diagnosis_logits.shape == (2, NUM_DIAGNOSES)
    # DenseNet-121 at 64x64 -> 2x2 feature map (downsampling x32).
    assert out.activation_maps.shape == (2, NUM_CONCEPTS, 2, 2)
    assert bool((out.concept_probs >= 0.0).all())
    assert bool((out.concept_probs <= 1.0).all())


def test_spatial_cbm_true_concepts_intervention_path():
    model = build_spatial_cbm(_config(concept_activation="none"))
    x = torch.randn(2, 3, 64, 64)
    true_concepts = torch.rand(2, NUM_CONCEPTS)
    out = model(x, concepts=true_concepts)
    assert torch.allclose(out.diagnosis_logits, model.diagnosis_head(true_concepts), atol=1e-5)


def test_spatial_cbm_no_encoder_to_diagnosis_bypass():
    """The diagnosis head must receive ONLY the concept vector."""
    model = build_spatial_cbm(_config(concept_activation="none"))

    # Structural check: DiagnosisHead is an affine map with in_features == K,
    # so raw encoder features cannot reach it.
    assert isinstance(model.diagnosis_head, DiagnosisHead)
    assert model.diagnosis_head.fc.in_features == NUM_CONCEPTS
    assert model.diagnosis_head.fc.out_features == NUM_DIAGNOSES
    assert model.encoder.feature_dim != NUM_CONCEPTS

    # Functional check: zero concept output must zero the diagnosis output,
    # independent of the input image.
    with torch.no_grad():
        model.spatial_head.conv.weight.zero_()
        model.spatial_head.conv.bias.zero_()
        model.diagnosis_head.fc.bias.zero_()

    x1 = torch.randn(2, 3, 64, 64)
    x2 = torch.randn(2, 3, 64, 64) * 5.0
    out1 = model(x1)
    out2 = model(x2)
    assert torch.allclose(out1.diagnosis_logits, out2.diagnosis_logits, atol=1e-6)
    assert torch.allclose(out1.diagnosis_logits, torch.zeros(2, NUM_DIAGNOSES), atol=1e-6)


def test_build_model_factory_spatial_cbm():
    cfg = _config()
    cfg.set("experiment", "spatial_cbm")
    model = build_model(cfg)
    assert isinstance(model, SpatiallyGroundedCBM)
