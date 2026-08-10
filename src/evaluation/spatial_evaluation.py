"""Spatial evaluation orchestration: predict + metric dispatch for the
Spatially Grounded CBM.

The prediction pass runs the model over a loader and collects, for every
image, the concept activation maps (from the concept-specific head) and the
radiologist bounding boxes. Classification metrics (diagnosis + concept) and
spatial metrics (IoU, Pointing Game, ELS) are then computed together.
"""
from __future__ import annotations

from src.evaluation.evaluation import build_metric_fn, predict


def evaluate_spatial_model(model, loader, device, config):
    """Return ``(metrics, results)`` for a Spatially Grounded CBM.

    ``metrics`` contains ``"diagnosis"``, ``"concepts"`` and ``"spatial"``
    (with macro ``diagnosis_auroc_macro`` / ``concept_auroc_macro`` /
    ``spatial_els_macro``); ``results`` is the raw prediction dict (including
    ``"activation_maps"`` and ``"boxes"`` for downstream visualization).
    """
    results = predict(model, loader, device)
    metric_fn = build_metric_fn(str(config.get("experiment", "spatial_cbm")), config)
    return metric_fn(results), results
