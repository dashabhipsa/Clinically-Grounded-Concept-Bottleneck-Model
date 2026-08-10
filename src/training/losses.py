"""Loss functions for multi-label chest X-ray training."""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def build_criterion(pos_weight: float | None = None) -> nn.Module:
    """Multi-label BCEWithLogitsLoss (optionally class-balanced)."""
    if pos_weight is not None:
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    return nn.BCEWithLogitsLoss()


def _outputs_from_tuple(output, experiment: str) -> dict:
    if isinstance(output, tuple):
        logits_c, logits_d = output
        return {"concepts": logits_c, "diagnosis": logits_d}
    return {"diagnosis": output}


def build_loss_fn(experiment: str, config) -> Callable:
    """Return ``loss_fn(model, x, labels, batch=None) -> (loss, outputs)``.

    ``labels`` is a dict of target tensors keyed by task
    (``"diagnosis"`` / ``"concepts"``); ``outputs`` is a dict of logits
    tensors keyed by task. For ``"spatial_cbm"`` the returned loss also
    consumes ``batch["boxes"]`` for the soft spatial-grounding term.
    """
    if str(experiment).lower() == "spatial_cbm":
        from src.training.spatial_losses import build_spatial_loss_fn

        return build_spatial_loss_fn(config)

    lambda_c = float(config.get("training.lambda_concept", 1.0))
    lambda_d = float(config.get("training.lambda_diagnosis", 1.0))
    criterion_c = build_criterion()
    criterion_d = build_criterion()

    def loss_fn(model, x, labels, batch=None):
        output = model(x)
        outputs = _outputs_from_tuple(output, experiment)
        if experiment == "cbm":
            loss = lambda_c * criterion_c(outputs["concepts"], labels["concepts"])
            loss = loss + lambda_d * criterion_d(outputs["diagnosis"], labels["diagnosis"])
        else:
            loss = criterion_d(outputs["diagnosis"], labels["diagnosis"])
        return loss, outputs

    return loss_fn
