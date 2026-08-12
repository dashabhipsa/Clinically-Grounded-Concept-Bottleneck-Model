"""Loss functions for multi-label chest X-ray training."""
from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn


def build_criterion(
    pos_weight: Sequence[float] | None = None, device=None
) -> nn.Module:
    """Multi-label BCEWithLogitsLoss (optionally class-balanced).

    ``pos_weight`` is one weight per class (``negatives/positives``), used to
    counter the majority-collapse induced by sparse multi-label targets.
    """
    if pos_weight is not None:
        pw = torch.as_tensor(list(pos_weight), dtype=torch.float32)
        if device is not None:
            pw = pw.to(device)
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    return nn.BCEWithLogitsLoss()


def _outputs_from_tuple(output, experiment: str) -> dict:
    if isinstance(output, tuple):
        logits_c, logits_d = output
        return {"concepts": logits_c, "diagnosis": logits_d}
    return {"diagnosis": output}


def build_loss_fn(
    experiment: str,
    config,
    pos_weight_concepts: Sequence[float] | None = None,
    pos_weight_diagnosis: Sequence[float] | None = None,
) -> Callable:
    """Return ``loss_fn(model, x, labels, batch=None) -> (loss, outputs)``.

    ``labels`` is a dict of target tensors keyed by task
    (``"diagnosis"`` / ``"concepts"``); ``outputs`` is a dict of logits
    tensors keyed by task. For ``"spatial_cbm"`` the returned loss also
    consumes ``batch["boxes"]`` for the soft spatial-grounding term.

    ``pos_weight_*`` are per-class ``negatives/positives`` ratios (from the
    *training* set) used to class-balance the BCE terms.
    """
    if str(experiment).lower() == "spatial_cbm":
        from src.training.spatial_losses import build_spatial_loss_fn

        return build_spatial_loss_fn(
            config,
            pos_weight_concepts=pos_weight_concepts,
            pos_weight_diagnosis=pos_weight_diagnosis,
        )

    lambda_c = float(config.get("training.lambda_concept", 1.0))
    lambda_d = float(config.get("training.lambda_diagnosis", 1.0))

    def loss_fn(model, x, labels, batch=None):
        output = model(x)
        outputs = _outputs_from_tuple(output, experiment)
        criterion_c = build_criterion(pos_weight_concepts, x.device)
        criterion_d = build_criterion(pos_weight_diagnosis, x.device)
        if experiment == "cbm":
            loss = lambda_c * criterion_c(outputs["concepts"], labels["concepts"])
            loss = loss + lambda_d * criterion_d(outputs["diagnosis"], labels["diagnosis"])
        else:
            loss = criterion_d(outputs["diagnosis"], labels["diagnosis"])
        return loss, outputs

    return loss_fn
