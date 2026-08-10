"""Spatially Grounded Concept Bottleneck Model (Phase 2).

The proposed architecture:

    Chest X-ray
        |
        v
    Vision Encoder .......................... (DenseNet-121 features)
        |
        v
    Spatial Feature Map [B, C, H, W]
        |
        v
    Concept-specific Spatial Heads .......... (1x1 conv, one channel per concept)
        |
        v
    Concept Activation Maps [B, K, H, W] .... (ReLU -> non-negative evidence)
        |
        v
    Concept Scores [B, K] ................... (global pooling of each CAM)
        |
        v
    Concept Bottleneck z = activation(z) .... (sigmoid -> concept probabilities)
        |
        v
    DiagnosisHead [B, D] .................... (affine map on z ONLY)

CRITICAL: there is NO direct encoder -> diagnosis connection. The diagnosis
head receives ONLY the concept vector ``z`` (structurally enforced by
``DiagnosisHead(fc.in_features == num_concepts)`` and verified by tests).

The spatial evidence for each concept comes directly from the concept-
specific model head (the per-concept activation map), NOT from post-hoc
Grad-CAM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cbm import DiagnosisHead
from src.models.encoder import ImageEncoder

_ACTIVATIONS = {
    "sigmoid": torch.sigmoid,
    "none": lambda z: z,
}

_POOLS = ("avg", "max")


@dataclass
class SpatialCBMOutput:
    """Rich forward-pass output of the Spatially Grounded CBM.

    For every concept the model produces, as required:
      1. ``concept_logits``   -- pre-activation concept score
      2. ``concept_probs``    -- concept probability (sigmoid of logits)
      3. ``activation_maps``  -- concept-specific spatial activation map
    plus the diagnosis logits (computed from the concept vector only).
    """

    concept_logits: torch.Tensor
    concept_probs: torch.Tensor
    activation_maps: torch.Tensor
    diagnosis_logits: torch.Tensor


class SpatialConceptHead(nn.Module):
    """Concept-specific spatial heads over the encoder feature map.

    A single ``1x1`` convolution produces one channel per concept; each
    channel is the *concept activation map* (CAM). A ReLU keeps the evidence
    non-negative (so energies and ELS are meaningful), and the concept score
    is obtained by pooling each CAM (``avg`` by default, ``max`` optional).
    """

    def __init__(self, in_channels: int, num_concepts: int, pool: str = "avg"):
        super().__init__()
        if pool not in _POOLS:
            raise ValueError(
                f"Unknown concept_pool {pool!r}; choose from {_POOLS}"
            )
        self.in_channels = int(in_channels)
        self.num_concepts = int(num_concepts)
        self.pool = pool
        self.conv = nn.Conv2d(in_channels, num_concepts, kernel_size=1)

    def forward(
        self, feature_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(activation_maps [B, K, H, W], concept_logits [B, K])``."""
        cam = self.conv(feature_map)
        cam = F.relu(cam, inplace=False)
        if self.pool == "avg":
            logits = F.adaptive_avg_pool2d(cam, 1).flatten(1)
        else:
            logits = F.adaptive_max_pool2d(cam, 1).flatten(1)
        return cam, logits


class SpatiallyGroundedCBM(nn.Module):
    """Encoder -> spatial concept heads -> concept bottleneck -> diagnosis."""

    def __init__(
        self,
        encoder: ImageEncoder,
        num_concepts: int,
        num_diagnoses: int,
        concept_activation: str = "sigmoid",
        concept_pool: str = "avg",
    ):
        super().__init__()
        if concept_activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown concept_activation {concept_activation!r}; "
                f"choose from {sorted(_ACTIVATIONS)}"
            )
        self.encoder = encoder
        self.spatial_head = SpatialConceptHead(
            encoder.feature_dim, num_concepts, pool=concept_pool
        )
        self.diagnosis_head = DiagnosisHead(num_concepts, num_diagnoses)
        self.activation = _ACTIVATIONS[concept_activation]
        self.num_concepts = num_concepts
        self.num_diagnoses = num_diagnoses

    def forward(
        self, x: torch.Tensor, concepts: torch.Tensor | None = None
    ) -> SpatialCBMOutput:
        """Forward pass; returns a :class:`SpatialCBMOutput`.

        If ``concepts`` (a multi-hot batch) is provided, the *true* concept
        vector is substituted at the bottleneck (standard CBM evaluation /
        intervention experiments), as in the Phase 1 CBM.
        """
        _, feature_map = self.encoder(x)
        activation_maps, concept_logits = self.spatial_head(feature_map)

        if concepts is not None:
            z = concepts
        else:
            z = self.activation(concept_logits)

        diagnosis_logits = self.diagnosis_head(z)
        return SpatialCBMOutput(
            concept_logits=concept_logits,
            concept_probs=self.activation(concept_logits),
            activation_maps=activation_maps,
            diagnosis_logits=diagnosis_logits,
        )


def build_spatial_cbm(config) -> SpatiallyGroundedCBM:
    """Build a Spatially Grounded CBM from a config object."""
    encoder = ImageEncoder(
        name=config.model.encoder,
        pretrained=bool(config.model.pretrained),
        freeze_backbone=bool(config.get("model.freeze_backbone", False)),
    )
    return SpatiallyGroundedCBM(
        encoder=encoder,
        num_concepts=len(config.concepts),
        num_diagnoses=len(config.diagnoses),
        concept_activation=str(config.get("model.concept_activation", "sigmoid")),
        concept_pool=str(config.get("model.concept_pool", "avg")),
    )
