"""Multi-label classification metrics (AUROC, AUPRC, F1, precision,
recall, sensitivity, specificity) with per-class and macro aggregation."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

METRIC_NAMES = ["auroc", "auprc", "f1", "precision", "recall", "sensitivity", "specificity"]


def sigmoid(z) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def compute_binary_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Metrics for a single binary class.

    ``y_score`` are logits or probabilities; the prediction threshold is
    applied after sigmoid transformation.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    probs = sigmoid(y_score).ravel()
    y_pred = (probs >= threshold).astype(int)

    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    result: Dict[str, float] = {}

    if n_pos > 0 and n_neg > 0:
        result["auroc"] = float(roc_auc_score(y_true, probs))
        result["auprc"] = float(average_precision_score(y_true, probs))
    else:
        result["auroc"] = float("nan")
        result["auprc"] = float("nan")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    result["precision"] = float(precision)
    result["recall"] = float(recall)
    result["sensitivity"] = float(recall)
    result["f1"] = float(f1)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    result["accuracy"] = float((tp + tn) / max(tp + tn + fp + fn, 1))
    return result


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> Dict:
    """Per-class and macro metrics for a multi-label problem.

    Returns ``{"per_class": {name: {...}}, "macro": {...}}``. Macro values
    are NaN-mean averages (classes without both label values are skipped).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_score = y_score[:, None]

    num_classes = y_true.shape[1]
    names = list(class_names) if class_names else [f"class_{i}" for i in range(num_classes)]
    if len(names) != num_classes:
        raise ValueError("class_names length must match number of columns")

    per_class: Dict[str, Dict[str, float]] = {}
    macro_values: Dict[str, List[float]] = {m: [] for m in METRIC_NAMES}

    for i in range(num_classes):
        row = compute_binary_metrics(y_true[:, i], y_score[:, i], threshold=threshold)
        per_class[names[i]] = row
        for metric in METRIC_NAMES:
            if not np.isnan(row[metric]):
                macro_values[metric].append(row[metric])

    macro = {m: float(np.nanmean(v)) if v else float("nan") for m, v in macro_values.items()}
    return {"per_class": per_class, "macro": macro}
