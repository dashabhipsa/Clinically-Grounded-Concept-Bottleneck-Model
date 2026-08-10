"""Spatial grounding loss for the Spatially Grounded CBM.

Loss = lambda_c * BCE(concepts) + lambda_d * BCE(diagnosis)
     + grounding_weight * L_grounding

``L_grounding`` is the *soft* grounding objective (Evidence Localization
Score): for every image x concept pair that carries a radiologist box, it
maximizes the fraction of concept activation energy that falls inside the
box:

    ELS      = energy_inside_box / total_activation_energy
    L_g      = 1 - ELS          (averaged over boxed (image, concept) pairs)

Concepts without a box do not contribute (no arbitrary target), which keeps
grounding decoupled from the concept-presence prediction. Optionally a
background term can push the energy of box-less concepts toward zero.

The ``grounding_weight`` is configurable (ablations 0.0 / 0.1 / 0.25 / 0.5 /
1.0). Box supervision can additionally be corrupted deterministically
(``box_corruption``) for the Phase 2 corruption experiments.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.data.box_corruption import corrupt_boxes
from src.data.spatial import boxes_to_masks_tensor, concept_presence_from_masks


class SpatialGroundingLoss:
    """Combined classification + soft spatial-grounding objective."""

    def __init__(
        self,
        concept_names: List[str],
        grounding_weight: float = 0.5,
        lambda_concept: float = 1.0,
        lambda_diagnosis: float = 1.0,
        background_weight: float = 0.0,
        mask_smoothing_sigma: float = 0.0,
        box_corruption: str = "correct",
        seed: int = 42,
        eps: float = 1e-8,
    ):
        self.concept_names = list(concept_names)
        self.num_concepts = len(self.concept_names)
        self.grounding_weight = float(grounding_weight)
        self.lambda_concept = float(lambda_concept)
        self.lambda_diagnosis = float(lambda_diagnosis)
        self.background_weight = float(background_weight)
        self.mask_smoothing_sigma = float(mask_smoothing_sigma)
        self.box_corruption = str(box_corruption or "correct")
        self.eps = float(eps)
        # Deterministic corruption: one RNG seeded from the run seed.
        self.rng = np.random.RandomState(int(seed))
        self.criterion_c = nn.BCEWithLogitsLoss()
        self.criterion_d = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------ #
    def grounding_term(
        self, activation_maps: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """Soft grounding loss: ``1 - ELS`` over (image, concept) with a box."""
        has_box = concept_presence_from_masks(masks)  # [B, K]
        if not bool(has_box.any()):
            return torch.zeros((), device=activation_maps.device)

        energy_total = activation_maps.sum(dim=(2, 3)) + self.eps  # [B, K]
        energy_in = (activation_maps * masks).sum(dim=(2, 3))      # [B, K]
        els = energy_in / energy_total
        grounding = (1.0 - els)[has_box].mean()

        if self.background_weight > 0.0:
            no_box = ~has_box
            if bool(no_box.any()):
                h, w = activation_maps.shape[2], activation_maps.shape[3]
                mean_energy = (energy_total / (h * w))[no_box].mean()
                grounding = grounding + self.background_weight * mean_energy
        return grounding

    # ------------------------------------------------------------------ #
    def _corrupt_boxes(self, boxes: list) -> list:
        if self.box_corruption in ("", "correct"):
            return list(boxes)
        return corrupt_boxes(boxes, self.box_corruption, rng=self.rng)

    # ------------------------------------------------------------------ #
    def __call__(self, model, x, labels, batch=None):
        """``loss_fn(model, x, labels, batch=None) -> (loss, outputs)``.

        ``batch`` must carry ``"boxes"`` (list of per-sample box lists) so the
        grounding masks can be built at CAM resolution. When ``batch`` is None
        (or ``grounding_weight`` is 0) only the classification loss applies.
        """
        out = model(x)

        loss = self.lambda_concept * self.criterion_c(
            out.concept_logits, labels["concepts"]
        )
        loss = loss + self.lambda_diagnosis * self.criterion_d(
            out.diagnosis_logits, labels["diagnosis"]
        )

        if batch is not None and self.grounding_weight > 0.0:
            h, w = out.activation_maps.shape[2], out.activation_maps.shape[3]
            boxes = [self._corrupt_boxes(sample) for sample in batch["boxes"]]
            masks = boxes_to_masks_tensor(
                boxes, self.concept_names, h, w,
                device=out.activation_maps.device,
                smooth_sigma=self.mask_smoothing_sigma,
            )
            grounding = self.grounding_term(out.activation_maps, masks)
            loss = loss + self.grounding_weight * grounding

        outputs = {
            "concepts": out.concept_logits,
            "diagnosis": out.diagnosis_logits,
        }
        return loss, outputs


def build_spatial_loss_fn(config):
    """Return a ``SpatialGroundingLoss`` configured from ``config``."""
    return SpatialGroundingLoss(
        concept_names=list(config.concepts),
        grounding_weight=float(config.get("model.grounding.weight", 0.5)),
        lambda_concept=float(config.get("training.lambda_concept", 1.0)),
        lambda_diagnosis=float(config.get("training.lambda_diagnosis", 1.0)),
        background_weight=float(
            config.get("model.grounding.background_weight", 0.0)
        ),
        mask_smoothing_sigma=float(
            config.get("model.grounding.mask_smoothing_sigma", 0.0)
        ),
        box_corruption=str(config.get("model.grounding.box_corruption", "correct")),
        seed=int(config.get("seed", 42)),
    )
