"""Tests for model forward passes and the CBM bottleneck integrity."""
import torch

from src.models import build_model
from src.models.blackbox import BlackBoxClassifier, build_blackbox
from src.models.cbm import ConceptBottleneckModel, build_cbm
from src.models.encoder import ImageEncoder
from src.utils.config import Config

from conftest import CONCEPT_NAMES, DIAGNOSIS_NAMES

NUM_CONCEPTS = len(CONCEPT_NAMES)
NUM_DIAGNOSES = len(DIAGNOSIS_NAMES)


def _base_config(concept_activation="sigmoid"):
    return Config(
        {
            "experiment": "blackbox",
            "model": {"encoder": "densenet121", "pretrained": False, "concept_activation": concept_activation},
            "concepts": CONCEPT_NAMES,
            "diagnoses": DIAGNOSIS_NAMES,
        }
    )


def test_encoder_forward_shape():
    encoder = ImageEncoder(name="densenet121", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    embedding, feature_map = encoder(x)
    assert embedding.shape == (2, encoder.feature_dim)
    assert feature_map.shape == (2, encoder.feature_dim, 7, 7)


def test_blackbox_forward_shape():
    model = build_blackbox(_base_config())
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, NUM_DIAGNOSES)


def test_cbm_forward_shape():
    model = build_cbm(_base_config())
    x = torch.randn(2, 3, 224, 224)
    logits_c, logits_d = model(x)
    assert logits_c.shape == (2, NUM_CONCEPTS)
    assert logits_d.shape == (2, NUM_DIAGNOSES)


def test_cbm_true_concepts_intervention_path():
    model = build_cbm(_base_config())
    x = torch.randn(2, 3, 224, 224)
    true_concepts = torch.rand(2, NUM_CONCEPTS)
    logits_c, logits_d = model(x, concepts=true_concepts)
    # diagnosis logits must equal the linear map applied to the concept vector
    assert torch.allclose(logits_d, model.diagnosis_head(true_concepts), atol=1e-5)


def test_cbm_bottleneck_integrity_no_encoder_to_diagnosis():
    """The diagnosis head must receive ONLY the concept vector.

    1. Structurally: ``DiagnosisHead`` in_features == num_concepts (so raw
       encoder features cannot be passed to it).
    2. Functionally: if the concept head outputs zero, the diagnosis logits
       depend only on that zero vector and not on the input image.
    """
    # Use concept_activation="none" so z = logits_c exactly: zero concept
    # output must yield exactly zero diagnosis output.
    model = build_cbm(_base_config(concept_activation="none"))

    assert isinstance(model.diagnosis_head.fc, torch.nn.Linear)
    assert model.diagnosis_head.fc.in_features == NUM_CONCEPTS
    assert model.diagnosis_head.fc.out_features == NUM_DIAGNOSES
    assert model.encoder.feature_dim != NUM_CONCEPTS

    # Freeze concept head at zero output and zero the diagnosis bias.
    with torch.no_grad():
        model.concept_head.fc.weight.zero_()
        model.concept_head.fc.bias.zero_()
        model.diagnosis_head.fc.bias.zero_()

    x1 = torch.randn(2, 3, 224, 224)
    x2 = torch.randn(2, 3, 224, 224) * 5.0
    _, logits_d1 = model(x1)
    _, logits_d2 = model(x2)
    # Any image now maps to the same (zero) diagnosis logits -> the diagnosis
    # output is a function of the concept vector alone.
    assert torch.allclose(logits_d1, logits_d2, atol=1e-6)
    assert torch.allclose(logits_d1, torch.zeros(2, NUM_DIAGNOSES), atol=1e-6)


def test_build_model_factory():
    blackbox = build_model(_base_config())
    assert isinstance(blackbox, BlackBoxClassifier)

    cbm_cfg = _base_config()
    cbm_cfg.set("experiment", "cbm")
    cbm = build_model(cbm_cfg)
    assert isinstance(cbm, ConceptBottleneckModel)


def test_unknown_encoder_raises():
    cfg = _base_config()
    cfg.set("model.encoder", "not_a_backbone")
    import pytest

    with pytest.raises(ValueError):
        build_blackbox(cfg)
