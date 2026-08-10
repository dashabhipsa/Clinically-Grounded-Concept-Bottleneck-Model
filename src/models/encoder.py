"""Image encoder (feature extractor) with a small registry of backbones."""
from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


def _build_backbone(builder: Callable, pretrained: bool) -> nn.Module:
    """Version-tolerant construction of a torchvision backbone.

    Newer torchvision exposes ``builder(weights=...)``; older versions use
    the deprecated ``pretrained=`` keyword.
    """
    if hasattr(builder, "DEFAULT"):
        weights = builder.DEFAULT if pretrained else None
        return builder(weights=weights)
    return builder(pretrained=pretrained)


# name -> (builder, feature_dim). The feature_dim is the global-pooled
# embedding size fed into classification heads.
ENCODER_REGISTRY: dict = {
    "densenet121": (tv_models.densenet121, 1024),
    "densenet169": (tv_models.densenet169, 1664),
    "densenet201": (tv_models.densenet201, 1920),
    "resnet50": (tv_models.resnet50, 2048),
    "resnet101": (tv_models.resnet101, 2048),
}


def build_feature_extractor(
    name: str, pretrained: bool = True
) -> Tuple[nn.Module, int]:
    """Return ``(feature_extractor, feature_dim)`` for a registry backbone.

    The feature extractor outputs a tensor ``[B, C, H, W]`` (feature map)
    that is subsequently global-pooled to form the embedding.
    """
    key = name.lower()
    if key not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder {name!r}. Supported: {sorted(ENCODER_REGISTRY)}"
        )
    builder, feature_dim = ENCODER_REGISTRY[key]
    model = _build_backbone(builder, pretrained)

    if key.startswith("densenet"):
        # torchvision DenseNet: `features` ends with a BatchNorm; the
        # trailing ReLU + global pooling are applied manually.
        extractor = model.features
    elif key.startswith("resnet"):
        # Everything up to (and including) the last residual stage.
        extractor = nn.Sequential(*list(model.children())[:-2])
    else:  # pragma: no cover - registry guards this
        raise ValueError(f"Unsupported architecture family for {key}")

    return extractor, feature_dim


class ImageEncoder(nn.Module):
    """Backbone feature extractor.

    ``forward(x)`` returns ``(embedding, feature_map)``:

    * ``embedding``  -> ``[B, feature_dim]`` (adaptive-average-pooled)
    * ``feature_map`` -> ``[B, C, H, W]`` (used by Grad-CAM in later code)
    """

    def __init__(self, name: str = "densenet121", pretrained: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()
        self.name = name
        self.features, self.feature_dim = build_feature_extractor(name, pretrained)
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.features(x)
        feature_map = F.relu(feature_map, inplace=True)
        embedding = F.adaptive_avg_pool2d(feature_map, (1, 1)).flatten(1)
        return embedding, feature_map
