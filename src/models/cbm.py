"""Standard Concept Bottleneck Model (CBM).

    X-ray
      |--> ImageEncoder ------> embedding
                                   |
                                   v
                             ConceptHead (k logits)
                                   |
                                   v
                          z = activation(logits_c)   <- concept vector
                                   |
                                   v
                            DiagnosisHead
                                   |
                                   v
                             diagnosis logits

CRITICAL: there is NO direct encoder -> diagnosis connection. The diagnosis
head receives ONLY the concept vector ``z = [z1, ..., zk]``. This is enforced
structurally (``DiagnosisHead`` is an affine map on the concept vector) and
verified by ``tests/test_models.py``.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.models.encoder import ImageEncoder

_ACTIVATIONS = {
    "sigmoid": torch.sigmoid,
    "none": lambda z: z,
}


class ConceptHead(nn.Module):
    """Linear map from encoder embedding to concept logits."""

    def __init__(self, in_features: int, num_concepts: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_concepts)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.fc(embedding)


class DiagnosisHead(nn.Module):
    """Linear map from the *concept vector* to diagnosis logits.

    ``in_features == num_concepts`` by construction; this guarantees the
    bottleneck cannot be bypassed by the raw encoder features.
    """

    def __init__(self, num_concepts: int, num_diagnoses: int):
        super().__init__()
        self.fc = nn.Linear(num_concepts, num_diagnoses)

    def forward(self, concept_vector: torch.Tensor) -> torch.Tensor:
        return self.fc(concept_vector)


class ConceptBottleneckModel(nn.Module):
    """ImageEncoder -> ConceptHead -> (concept vector) -> DiagnosisHead."""

    def __init__(
        self,
        encoder: ImageEncoder,
        num_concepts: int,
        num_diagnoses: int,
        concept_activation: str = "sigmoid",
    ):
        super().__init__()
        if concept_activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown concept_activation {concept_activation!r}; "
                f"choose from {sorted(_ACTIVATIONS)}"
            )
        self.encoder = encoder
        self.concept_head = ConceptHead(encoder.feature_dim, num_concepts)
        self.diagnosis_head = DiagnosisHead(num_concepts, num_diagnoses)
        self.activation = _ACTIVATIONS[concept_activation]
        self.num_concepts = num_concepts
        self.num_diagnoses = num_diagnoses

    def forward(
        self, x: torch.Tensor, concepts: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(concept_logits, diagnosis_logits)``.

        If ``concepts`` (a multi-hot batch) is provided, the *true* concept
        vector is substituted at the bottleneck (standard CBM evaluation /
        future intervention experiments). Phase 1 training does not use this.
        """
        embedding, _ = self.encoder(x)
        logits_c = self.concept_head(embedding)

        if concepts is not None:
            z = concepts
        else:
            z = self.activation(logits_c)

        logits_d = self.diagnosis_head(z)
        return logits_c, logits_d


def build_cbm(config) -> ConceptBottleneckModel:
    """Build a standard CBM from a config object."""
    encoder = ImageEncoder(
        name=config.model.encoder,
        pretrained=bool(config.model.pretrained),
        freeze_backbone=bool(config.get("model.freeze_backbone", False)),
    )
    return ConceptBottleneckModel(
        encoder=encoder,
        num_concepts=len(config.concepts),
        num_diagnoses=len(config.diagnoses),
        concept_activation=str(config.get("model.concept_activation", "sigmoid")),
    )
