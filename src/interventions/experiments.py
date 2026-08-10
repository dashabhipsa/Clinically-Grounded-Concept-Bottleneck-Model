"""Phase 3 analysis experiments built on the concept bottleneck.

These functions operate on ``predict()`` results (numpy arrays) and a trained
model. They reuse the bottleneck machinery:

* :func:`oracle_concept_experiment` -- predicted vs. ground-truth concepts.
* :func:`concept_error_robustness`   -- controlled concept corruption at eval.
* :func:`concept_error_recovery`     -- fix predicted concepts with GT, measure
  diagnostic improvement.
* :func:`concept_completeness`       -- how many concepts are needed to retain
  diagnostic information (concept subsets by masking out-of-subset concepts).

Every result is computed from actual model predictions -- nothing is inferred
from medical knowledge.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.evaluation.classification_metrics import (
    compute_multilabel_metrics,
    sigmoid,
)
from src.interventions.runner import (
    concept_probs_from_logits,
    diagnosis_probs_from_concept_matrix,
)

_THRESHOLD = 0.5


def _metric_sets(
    y_true: np.ndarray,
    scores: np.ndarray,
    class_names: Sequence[str],
    threshold: float = _THRESHOLD,
) -> Dict:
    """Per-class + macro metrics for one score matrix."""
    return compute_multilabel_metrics(
        np.asarray(y_true, dtype=int),
        np.asarray(scores, dtype=np.float64),
        list(class_names),
        threshold,
    )


# --------------------------------------------------------------------------- #
# 1. Oracle concept experiment
# --------------------------------------------------------------------------- #
def oracle_concept_experiment(
    model,
    results: Dict[str, np.ndarray],
    diagnosis_names: Sequence[str],
    threshold: float = _THRESHOLD,
) -> Dict:
    """Compare diagnosis performance using predicted vs. ground-truth concepts.

    The diagnosis is always computed from the *concept vector* via the
    diagnosis head; only the source of the concept vector differs:

    A. ``z = activation(concept_logits)`` (predicted concepts)
    B. ``z = ground-truth concept labels`` (oracle concepts)

    Returns ``{"predicted_concepts": {...}, "oracle_concepts": {...},
    "gap": {...}}`` where each entry contains diagnosis per-class + macro
    metrics (AUROC / AUPRC / F1 / sensitivity / specificity).

    .. warning::
        This is an *analysis experiment* estimating how much diagnostic
        performance is limited by imperfect concept prediction. It is NOT a
        clinically realistic deployment scenario: ground-truth concepts are
        not available at inference time.
    """
    y_d = np.asarray(results["diagnosis_labels"], dtype=int)
    y_c = np.asarray(results["concepts_labels"], dtype=int)
    z_pred = concept_probs_from_logits(model, results["concepts_logits"])

    p_pred = diagnosis_probs_from_concept_matrix(model, z_pred)
    p_oracle = diagnosis_probs_from_concept_matrix(model, y_c.astype(np.float32))

    predicted = _metric_sets(y_d, p_pred, diagnosis_names, threshold)
    oracle = _metric_sets(y_d, p_oracle, diagnosis_names, threshold)

    gap: Dict[str, float] = {}
    for metric in ("auroc", "auprc", "f1", "sensitivity", "specificity"):
        a = oracle["macro"].get(metric, float("nan"))
        b = predicted["macro"].get(metric, float("nan"))
        gap[metric] = float(a) - float(b) if (np.isfinite(a) and np.isfinite(b)) else float("nan")

    return {
        "predicted_concepts": predicted,
        "oracle_concepts": oracle,
        "gap": gap,
    }


def summarize_oracle_experiment(result: Dict) -> Dict:
    """Flatten oracle results into a row suitable for a summary CSV."""
    row: Dict[str, float] = {}
    for variant in ("predicted_concepts", "oracle_concepts"):
        macro = result[variant].get("macro", {})
        for metric in ("auroc", "auprc", "f1", "sensitivity", "specificity"):
            row[f"{variant}_{metric}"] = macro.get(metric, float("nan"))
    for metric, value in result.get("gap", {}).items():
        row[f"gap_{metric}"] = value
    return row


# --------------------------------------------------------------------------- #
# 2. Concept error robustness
# --------------------------------------------------------------------------- #
def corrupt_concepts(
    z: np.ndarray,
    fraction: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Corrupt ``fraction`` of concept values per sample.

    A corrupted concept's value is flipped: ``z_k -> 1 - z_k`` (the opposite
    evidence). Deterministic for a given ``rng``. ``fraction=0`` leaves the
    vector unchanged.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"expected [B, K] concept matrix, got {z.shape}")
    mask = rng.uniform(size=z.shape) < float(fraction)
    z_mod = z.copy()
    z_mod[mask] = 1.0 - z_mod[mask]
    return z_mod


def concept_error_robustness(
    model,
    results: Dict[str, np.ndarray],
    diagnosis_names: Sequence[str],
    fractions: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    seed: int = 42,
    threshold: float = _THRESHOLD,
) -> pd.DataFrame:
    """Diagnostic metrics under increasing deterministic concept corruption.

    Returns a DataFrame with one row per corruption fraction:
    ``fraction, seed, auroc, auprc, f1`` (macro over diagnoses). Each level
    uses a fresh ``RandomState(seed + level_index)`` so results are
    reproducible and the corruption pattern differs across levels.
    """
    y_d = np.asarray(results["diagnosis_labels"], dtype=int)
    z_pred = concept_probs_from_logits(model, results["concepts_logits"])

    rows: List[Dict] = []
    for level, fraction in enumerate(fractions):
        rng = np.random.RandomState(int(seed) + level)
        z_corrupt = corrupt_concepts(z_pred, fraction, rng)
        p_d = diagnosis_probs_from_concept_matrix(model, z_corrupt)
        metrics = _metric_sets(y_d, p_d, diagnosis_names, threshold)
        macro = metrics["macro"]
        rows.append(
            {
                "fraction": float(fraction),
                "seed": int(seed),
                "auroc": macro.get("auroc", float("nan")),
                "auprc": macro.get("auprc", float("nan")),
                "f1": macro.get("f1", float("nan")),
                "sensitivity": macro.get("sensitivity", float("nan")),
                "specificity": macro.get("specificity", float("nan")),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. Concept error recovery
# --------------------------------------------------------------------------- #
def concept_error_recovery(
    model,
    results: Dict[str, np.ndarray],
    diagnosis_names: Sequence[str],
    threshold: float = _THRESHOLD,
) -> Dict:
    """Replace *incorrect* predicted concepts with GT values and re-measure.

    Procedure (per test image):

    1. Predict concepts ``z``.
    2. Compare with ground-truth concepts (threshold 0.5) -> errors.
    3. Replace erroneous concepts with their ground-truth values.
    4. Recompute diagnosis through the diagnosis head.
    5. Measure diagnostic improvement.

    Returns a dict with macro metrics for ``original``, ``corrected`` and the
    per-image ``improvement`` data frame.

    .. note::
        This uses ground-truth concept values at evaluation time; it estimates
        an upper bound on how much diagnostic performance is lost to concept
        errors.
    """
    y_d = np.asarray(results["diagnosis_labels"], dtype=int)
    y_c = np.asarray(results["concepts_labels"], dtype=int)
    z_pred = concept_probs_from_logits(model, results["concepts_logits"])

    p_original = diagnosis_probs_from_concept_matrix(model, z_pred)

    pred_bin = (z_pred >= float(threshold)).astype(int)
    errors = pred_bin != y_c
    z_corrected = z_pred.copy()
    z_corrected[errors] = y_c[errors].astype(np.float32)
    p_corrected = diagnosis_probs_from_concept_matrix(model, z_corrected)

    original = _metric_sets(y_d, p_original, diagnosis_names, threshold)
    corrected = _metric_sets(y_d, p_corrected, diagnosis_names, threshold)

    macro_orig = original["macro"]
    macro_corr = corrected["macro"]
    improvement = {
        metric: (
            float(macro_corr.get(metric, float("nan")))
            - float(macro_orig.get(metric, float("nan")))
            if np.isfinite(macro_orig.get(metric))
            and np.isfinite(macro_corr.get(metric))
            else float("nan")
        )
        for metric in ("auroc", "auprc", "f1", "sensitivity", "specificity")
    }

    image_ids = results.get("image_ids", [])
    num_errors_per_image = errors.sum(axis=1)
    per_image = pd.DataFrame(
        {
            "image_id": image_ids,
            "num_concept_errors": num_errors_per_image,
            "num_concepts": errors.shape[1],
        }
    )
    return {
        "original": original,
        "corrected": corrected,
        "improvement": improvement,
        "per_image": per_image,
    }


def summarize_recovery(result: Dict) -> Dict:
    row: Dict[str, float] = {}
    for variant in ("original", "corrected"):
        macro = result[variant].get("macro", {})
        for metric in ("auroc", "auprc", "f1", "sensitivity", "specificity"):
            row[f"{variant}_{metric}"] = macro.get(metric, float("nan"))
    for metric, value in result.get("improvement", {}).items():
        row[f"improvement_{metric}"] = value
    row["total_concept_errors"] = int(result["per_image"]["num_concept_errors"].sum())
    row["n_images"] = int(len(result["per_image"]))
    return row


# --------------------------------------------------------------------------- #
# 4. Concept completeness
# --------------------------------------------------------------------------- #
def concept_subsets(
    concept_names: Sequence[str],
    sizes: Sequence[int] = (5, 8, 12, 22),
    min_concepts: int = 1,
) -> List[Dict]:
    """Configurable concept subsets that actually exist.

    Only concepts in ``concept_names`` are considered and only sizes that fit
    the available set are produced. Larger requested sizes than available are
    silently dropped (never fabricated).
    """
    names = list(concept_names)
    subsets: List[Dict] = []
    for size in sizes:
        if int(size) > len(names) or int(size) < int(min_concepts):
            continue
        subset = names[: int(size)]
        subsets.append({"size": int(size), "concepts": subset})
    if len(names) not in [s["size"] for s in subsets]:
        subsets.append({"size": len(names), "concepts": names})
    return subsets


def concept_completeness(
    model,
    results: Dict[str, np.ndarray],
    concept_names: Sequence[str],
    diagnosis_names: Sequence[str],
    sizes: Sequence[int] = (5, 8, 12, 22),
    threshold: float = _THRESHOLD,
) -> Dict:
    """Diagnostic + concept metrics for progressively larger concept subsets.

    For each subset size, the concepts outside the subset are *masked to zero*
    in the bottleneck before the diagnosis head is evaluated (the encoder and
    trained weights are reused -- no retraining). Diagnostic metrics are
    computed on the masked bottleneck output; concept metrics use only the
    in-subset concept predictions.

    The research question is *how many concepts are sufficient to preserve
    diagnostic information*; the results do not assume that more concepts
    imply better performance.

    Returns ``{"subsets": [...], "summary": DataFrame}``.
    """
    y_d = np.asarray(results["diagnosis_labels"], dtype=int)
    y_c = np.asarray(results["concepts_labels"], dtype=int)
    z_pred = concept_probs_from_logits(model, results["concepts_logits"])
    logits_c = np.asarray(results["concepts_logits"], dtype=np.float64)

    subsets = concept_subsets(concept_names, sizes)
    summary_rows: List[Dict] = []
    subset_results: List[Dict] = []
    names = list(concept_names)
    index = {n: i for i, n in enumerate(names)}

    for subset in subsets:
        sub_names = subset["concepts"]
        cols = [index[n] for n in sub_names]

        mask = np.zeros((1, len(names)), dtype=np.float32)
        mask[0, cols] = 1.0
        z_masked = z_pred * mask

        p_d = diagnosis_probs_from_concept_matrix(model, z_masked)
        diag_metrics = _metric_sets(y_d, p_d, diagnosis_names, threshold)

        concept_metrics = _metric_sets(
            y_c[:, cols], logits_c[:, cols], sub_names, threshold
        )

        macro_d = diag_metrics["macro"]
        macro_c = concept_metrics["macro"]
        summary_rows.append(
            {
                "n_concepts": subset["size"],
                "diagnosis_auroc": macro_d.get("auroc", float("nan")),
                "diagnosis_auprc": macro_d.get("auprc", float("nan")),
                "diagnosis_f1": macro_d.get("f1", float("nan")),
                "concept_auroc": macro_c.get("auroc", float("nan")),
                "concept_auprc": macro_c.get("auprc", float("nan")),
                "concept_f1": macro_c.get("f1", float("nan")),
            }
        )
        subset_results.append(
            {
                "size": subset["size"],
                "concepts": sub_names,
                "diagnosis": diag_metrics,
                "concepts_metrics": concept_metrics,
            }
        )

    return {"subsets": subset_results, "summary": pd.DataFrame(summary_rows)}
