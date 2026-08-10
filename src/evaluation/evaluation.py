"""Model evaluation: inference, metric computation, output persistence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch

from src.evaluation.classification_metrics import compute_multilabel_metrics, sigmoid
from src.evaluation.spatial_metrics import evaluate_spatial_metrics
from src.models.spatial_cbm import SpatialCBMOutput


def predict(model, loader, device) -> Dict[str, np.ndarray]:
    """Run inference over a loader.

    Returns a dict with keys ``"diagnosis_logits"``, ``"diagnosis_labels"``,
    and (for CBMs) ``"concepts_logits"`` / ``"concepts_labels"`` plus
    ``"image_ids"``. For the Spatially Grounded CBM it additionally includes
    ``"activation_maps"`` ``[N, K, H, W]`` and ``"boxes"`` (the per-image
    radiologist bounding boxes) so spatial metrics can be computed.
    """
    model.eval()
    collected: Dict[str, List] = {}
    image_ids: List[str] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            output = model(x)
            if isinstance(output, SpatialCBMOutput):
                collected.setdefault("concepts_logits", []).append(
                    output.concept_logits.detach().float().cpu().numpy()
                )
                collected.setdefault("diagnosis_logits", []).append(
                    output.diagnosis_logits.detach().float().cpu().numpy()
                )
                collected.setdefault("activation_maps", []).append(
                    output.activation_maps.detach().float().cpu().numpy()
                )
            elif isinstance(output, tuple):
                logits_c, logits_d = output
                collected.setdefault("concepts_logits", []).append(
                    logits_c.detach().float().cpu().numpy()
                )
                collected.setdefault("diagnosis_logits", []).append(
                    logits_d.detach().float().cpu().numpy()
                )
            else:
                collected.setdefault("diagnosis_logits", []).append(
                    output.detach().float().cpu().numpy()
                )
            for key in ("diagnosis", "concepts"):
                if key in batch:
                    collected.setdefault(f"{key}_labels", []).append(
                        batch[key].cpu().numpy()
                    )
            if "boxes" in batch:
                collected.setdefault("boxes", []).append(batch["boxes"])
            image_ids.extend(batch["image_id"])

    results: Dict[str, np.ndarray] = {}
    for key, values in collected.items():
        if key == "boxes":
            results[key] = [sample for batch_boxes in values for sample in batch_boxes]
        else:
            results[key] = np.concatenate(values, axis=0)
    results["image_ids"] = image_ids
    return results


def build_metric_fn(experiment: str, config) -> Callable:
    """Return ``metric_fn(results) -> metrics dict`` for trainer/eval use.

    The returned metrics dict contains:

    * ``"diagnosis"``  -> per-class + macro metrics
    * ``"diagnosis_auroc_macro"`` -> top-level monitor value
    * for CBMs, ``"concepts"`` and ``"concept_auroc_macro"`` as well.
    * for ``"spatial_cbm"``, ``"spatial"`` (per-class + macro IoU /
      Pointing Game / ELS) whenever ``results`` carries activation maps and
      boxes (during training the val-loop result only has logits, so spatial
      metrics are skipped there and computed at test time).
    """
    threshold = float(config.get("evaluation.threshold", 0.5))
    diagnosis_names = list(config.diagnoses)
    concept_names = list(config.concepts)

    def metric_fn(results: Dict[str, np.ndarray]) -> Dict:
        metrics: Dict = {}
        y_d = results["diagnosis_labels"]
        s_d = results["diagnosis_logits"]
        diagnosis_metrics = compute_multilabel_metrics(
            y_d, sigmoid(s_d), diagnosis_names, threshold
        )
        metrics["diagnosis"] = diagnosis_metrics
        metrics["diagnosis_auroc_macro"] = diagnosis_metrics["macro"]["auroc"]

        if experiment in ("cbm", "spatial_cbm"):
            y_c = results["concepts_labels"]
            s_c = results["concepts_logits"]
            concept_metrics = compute_multilabel_metrics(
                y_c, sigmoid(s_c), concept_names, threshold
            )
            metrics["concepts"] = concept_metrics
            metrics["concept_auroc_macro"] = concept_metrics["macro"]["auroc"]

        if experiment == "spatial_cbm" and "activation_maps" in results and "boxes" in results:
            spatial = evaluate_spatial_metrics(
                results["activation_maps"],
                results["boxes"],
                concept_names,
                cam_threshold=float(config.get("evaluation.cam_threshold", 0.5)),
            )
            metrics["spatial"] = spatial
            metrics["spatial_els_macro"] = spatial["macro"]["els"]
        return metrics

    return metric_fn


def save_metrics_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


def save_predictions(
    results: Dict[str, np.ndarray],
    diagnosis_names: List[str],
    concept_names: List[str],
    path: str | Path,
) -> None:
    """Save per-image predicted probabilities and ground truth to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    probs_d = sigmoid(results["diagnosis_logits"])
    y_d = results["diagnosis_labels"]
    has_concepts = "concepts_logits" in results
    probs_c = sigmoid(results["concepts_logits"]) if has_concepts else None
    y_c = results.get("concepts_labels")

    for i, image_id in enumerate(results["image_ids"]):
        row = {"image_id": image_id}
        for j, name in enumerate(diagnosis_names):
            row[f"{name}_prob"] = float(probs_d[i, j])
            row[f"{name}_true"] = int(y_d[i, j])
        if has_concepts:
            for j, name in enumerate(concept_names):
                row[f"{name}_concept_prob"] = float(probs_c[i, j])
                row[f"{name}_concept_true"] = int(y_c[i, j])
        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)


def results_to_markdown_table(summary: dict) -> str:
    """Render the diagnosis concept table for the report (values only after
    experiments run; otherwise N/A)."""
    exp = summary.get("experiment", "?")
    test = summary.get("test", {})
    diag_macro = test.get("diagnosis", {}).get("macro", {})
    concept_macro = test.get("concepts", {}).get("macro", {})
    return (
        f"| {exp:12s} | {diag_macro.get('auroc', 'N/A'):>15s} | "
        f"{diag_macro.get('auprc', 'N/A'):>15s} | "
        f"{concept_macro.get('auroc', 'N/A') if exp == 'cbm' else 'N/A':>13s} |"
    )
