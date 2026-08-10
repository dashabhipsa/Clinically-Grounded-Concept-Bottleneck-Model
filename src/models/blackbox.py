"""Black-box diagnostic classifier: X-ray -> encoder -> diagnosis."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoder import ImageEncoder


class BlackBoxClassifier(nn.Module):
    """Direct image-to-diagnosis model (no interpretable bottleneck)."""

    def __init__(self, encoder: ImageEncoder, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding, _ = self.encoder(x)
        return self.classifier(embedding)


def build_blackbox(config) -> BlackBoxClassifier:
    """Build a black-box model from a config object."""
    encoder = ImageEncoder(
        name=config.model.encoder,
        pretrained=bool(config.model.pretrained),
        freeze_backbone=bool(config.get("model.freeze_backbone", False)),
    )
    num_diagnoses = len(config.diagnoses)
    return BlackBoxClassifier(encoder=encoder, num_classes=num_diagnoses)
