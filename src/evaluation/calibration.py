"""Calibration metrics (ECE / Brier / reliability curves) using only
dependencies already present in the repository (numpy, sklearn, matplotlib).

* **Expected Calibration Error (ECE)** -- mean |accuracy - confidence| over
  probability bins.
* **Brier Score** -- mean squared error between probabilities and labels.
* **Reliability diagrams** -- accuracy vs. confidence per bin.

Calibration is computed for both diagnosis and concept predictions. If
``sklearn.calibration.calibration_curve`` is unavailable the module degrades
gracefully (ECE via numpy only).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.evaluation.classification_metrics import sigmoid

try:  # pragma: no cover - availability guard
    from sklearn.calibration import calibration_curve as _sk_calibration_curve
    from sklearn.metrics import brier_score_loss as _brier_score_loss
    _HAS_SKLEARN_CALIBRATION = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN_CALIBRATION = False


def _brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    if _HAS_SKLEARN_CALIBRATION:
        return float(_brier_score_loss(y_true, probs))
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    probs = np.asarray(probs, dtype=np.float64).ravel()
    return float(np.mean((probs - y_true) ** 2))


def reliability_bins(
    y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> Dict[str, np.ndarray]:
    """Bin probabilities and return confidence, accuracy, count per bin."""
    probs = np.asarray(probs, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    n_bins = int(n_bins)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf = np.zeros(n_bins)
    acc = np.zeros(n_bins)
    cnt = np.zeros(n_bins)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        cnt[i] = int(mask.sum())
        if cnt[i] > 0:
            conf[i] = float(probs[mask].mean())
            acc[i] = float(y_true[mask].mean())
    return {"confidence": conf, "accuracy": acc, "count": cnt, "edges": edges}


def expected_calibration_error(
    y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> float:
    """ECE with count-weighted bins (over N bins)."""
    bins = reliability_bins(y_true, probs, n_bins)
    total = max(float(bins["count"].sum()), 1.0)
    ece = np.sum(
        bins["count"] * np.abs(bins["confidence"] - bins["accuracy"])
    ) / total
    return float(ece)


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    return _brier(y_true, probs)


def per_class_calibration(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    n_bins: int = 10,
) -> Dict:
    """ECE + Brier per class and macro average (skips degenerate classes)."""
    y_true = np.asarray(y_true, dtype=int)
    probs = sigmoid(y_score)
    num_classes = y_true.shape[1]
    names = list(class_names) if class_names else [f"class_{i}" for i in range(num_classes)]
    if len(names) != num_classes:
        raise ValueError("class_names length must match number of columns")

    per_class: Dict[str, Dict[str, float]] = {}
    ece_values, brier_values = [], []
    for i in range(num_classes):
        ece = expected_calibration_error(y_true[:, i], probs[:, i], n_bins)
        br = brier_score(y_true[:, i], probs[:, i])
        per_class[names[i]] = {"ece": ece, "brier": br}
        ece_values.append(ece)
        brier_values.append(br)
    return {
        "per_class": per_class,
        "macro": {
            "ece": float(np.mean(ece_values)),
            "brier": float(np.mean(brier_values)),
        },
    }


def calibration_results(
    results: Dict[str, np.ndarray],
    diagnosis_names: Sequence[str],
    concept_names: Optional[Sequence[str]] = None,
    n_bins: int = 10,
) -> Dict:
    """Calibration for diagnosis (and concepts where present) predictions."""
    out: Dict = {
        "diagnosis": per_class_calibration(
            results["diagnosis_labels"],
            results["diagnosis_logits"],
            diagnosis_names,
            n_bins,
        )
    }
    if concept_names is not None and "concepts_logits" in results and "concepts_labels" in results:
        out["concepts"] = per_class_calibration(
            results["concepts_labels"],
            results["concepts_logits"],
            concept_names,
            n_bins,
        )
    return out


def plot_reliability_diagram(
    results: Dict[str, np.ndarray],
    task: str,
    class_names: Sequence[str],
    save_path: str | Path,
    n_bins: int = 10,
    max_classes: int = 8,
) -> Path:
    """Reliability diagram: per-class accuracy vs. confidence (max 8 classes).

    ``task`` is "diagnosis" or "concepts"; the matching labels/logits are
    read from ``results``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(results[f"{task}_labels"], dtype=int)
    probs = sigmoid(results[f"{task}_logits"])
    names = list(class_names)
    classes = list(range(min(max_classes, probs.shape[1])))

    ncols = 4
    nrows = int(np.ceil(len(classes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")

    for plot_i, i in enumerate(classes):
        ax = axes[plot_i]
        ax.axis("on")
        y = y_true[:, i]
        p = probs[:, i]
        bins = reliability_bins(y, p, n_bins)
        ax.plot(bins["confidence"], bins["accuracy"], "o-", color="#1f77b4",
                label="model", markersize=3)
        ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{names[i]}\nECE={expected_calibration_error(y, p, n_bins):.3f}",
                     fontsize=9)
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        if plot_i == 0:
            ax.legend(fontsize=7, loc="lower right")

    for ax in axes[len(classes):]:
        ax.axis("off")
    fig.suptitle(f"{task.title()} reliability diagram")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
