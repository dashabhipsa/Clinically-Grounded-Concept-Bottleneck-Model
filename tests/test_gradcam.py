"""Tests for the Grad-CAM post-hoc baseline."""
import torch

from src.explainability.gradcam import GradCAM, find_last_conv, visualize_cam
from src.models.blackbox import build_blackbox
from src.utils.config import Config

from conftest import CONCEPT_NAMES, DIAGNOSIS_NAMES


def _config():
    return Config(
        {
            "experiment": "blackbox",
            "model": {"encoder": "densenet121", "pretrained": False},
            "concepts": CONCEPT_NAMES,
            "diagnoses": DIAGNOSIS_NAMES,
        }
    )


def test_find_last_conv_is_conv2d():
    model = build_blackbox(_config())
    layer = find_last_conv(model.encoder.features)
    assert layer is not None
    assert isinstance(layer, torch.nn.Conv2d)


def test_gradcam_mask_shape_and_range(tmp_path):
    model = build_blackbox(_config())
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    cam = GradCAM(model)
    mask = cam.generate(x)
    cam.close()
    assert mask.shape == (64, 64)
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0


def test_visualize_cam_writes_file(tmp_path):
    model = build_blackbox(_config())
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    cam = GradCAM(model)
    mask = cam.generate(x)
    cam.close()
    out = tmp_path / "cam.png"
    visualize_cam(x, mask, out)
    assert out.exists()
    assert out.stat().st_size > 0
