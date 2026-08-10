"""Systematic failure analysis.

Every test image is categorised by the combination of diagnostic, concept and
localization correctness:

===================  ========================================================
Category             Description
===================  ========================================================
``A``                correct diagnosis + correct concept + correct localization
``B``                correct diagnosis + correct concept + incorrect localization
``C``                correct diagnosis + incorrect concept
``D``                incorrect diagnosis + correct concept
``E``                incorrect diagnosis + incorrect concept + incorrect localization
===================  ========================================================

Category rules (thresholds 0.5):

* *diagnosis correct*  -- every present diagnosis is detected, OR
  (multi-label) the diagnosis prediction matches GT on all classes; see
  ``diagnosis_correct``.
* *concept correct*    -- concept predictions match GT on all classes.
* *localization correct* -- mean ELS over boxed (image, concept) pairs >=
  ``els_threshold`` for the images that actually have boxes.

Images without boxes are excluded from the spatial subset but still get a
category based on diagnosis/concept correctness (A/B -> C, D -> E mapped).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.evaluation.classification_metrics import sigmoid
from src.evaluation.spatial_metrics import els, normalized_box_to_pixels


def diagnosis_correct(y_true: np.ndarray, y_pred: np.ndarray) -> bool:
    """Exact multi-label match on the diagnosis vector."""
    return bool(np.array_equal(y_true, y_pred))


def concept_correct(y_true: np.ndarray, y_pred: np.ndarray) -> bool:
    """Exact multi-label match on the concept vector."""
    return bool(np.array_equal(y_true, y_pred))


def localization_correct(
    activation_map: np.ndarray,
    boxes,
    concept_names: Sequence[str],
    threshold: float = 0.5,
) -> Optional[bool]:
    """True when mean ELS of boxed concepts >= threshold (None if no boxes)."""
    if boxes is None or len(boxes) == 0:
        return None
    cam_h, cam_w = activation_map.shape[-2], activation_map.shape[-1]
    index = {name: i for i, name in enumerate(concept_names)}
    els_values = []
    for box in boxes:
        name = box.class_name.strip().lower().replace("/", "_").replace(" ", "_")
        k = index.get(name)
        if k is None:
            continue
        gt = normalized_box_to_pixels(box, cam_h, cam_w)
        els_values.append(els(activation_map[k], gt))
    if not els_values:
        return None
    return float(np.mean(els_values)) >= float(threshold)


def categorize_row(
    diag_correct: bool,
    concept_correct: bool,
    loc_correct: Optional[bool],
) -> str:
    """Map correctness flags to a failure-analysis category letter."""
    if diag_correct and concept_correct and loc_correct is True:
        return "A"
    if diag_correct and concept_correct and loc_correct is False:
        return "B"
    if diag_correct and not concept_correct:
        return "C"
    if not diag_correct and concept_correct:
        return "D"
    return "E"


def run_failure_analysis(
    results: Dict[str, np.ndarray],
    diagnosis_names: Sequence[str],
    concept_names: Sequence[str],
    threshold: float = 0.5,
    els_threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute a per-image failure-analysis table.

    Requires ``results`` with diagnosis/concept logits+labels and (for the
    spatial subset) ``activation_maps`` and ``boxes``.
    """
    image_ids = list(results.get("image_ids", []))
    y_d = np.asarray(results["diagnosis_labels"], dtype=int)
    y_c = np.asarray(results["concepts_labels"], dtype=int)
    p_d = sigmoid(results["diagnosis_logits"])
    p_c = sigmoid(results["concepts_logits"])
    pred_d = (p_d >= threshold).astype(int)
    pred_c = (p_c >= threshold).astype(int)

    has_spatial = "activation_maps" in results and "boxes" in results
    maps = results.get("activation_maps")
    boxes_list = results.get("boxes")

    rows: List[Dict] = []
    counts = {c: 0 for c in "ABCDE"}
    for i, image_id in enumerate(image_ids):
        d_ok = diagnosis_correct(y_d[i], pred_d[i])
        c_ok = concept_correct(y_c[i], pred_c[i])
        loc_ok: Optional[bool] = None
        if has_spatial:
            loc_ok = localization_correct(
                maps[i], boxes_list[i], concept_names, threshold=els_threshold
            )
        category = categorize_row(d_ok, c_ok, loc_ok)
        counts[category] += 1
        rows.append(
            {
                "image_id": image_id,
                "diagnosis_correct": int(d_ok),
                "concept_correct": int(c_ok),
                "localization_correct": None if loc_ok is None else int(loc_ok),
                "category": category,
            }
        )
    df = pd.DataFrame(rows)
    counts_df = pd.DataFrame(
        [{"category": c, "count": counts[c]} for c in "ABCDE"]
    )
    return df.assign(_counts=None), counts_df


def save_failure_analysis(
    results: Dict[str, np.ndarray],
    diagnosis_names: Sequence[str],
    concept_names: Sequence[str],
    csv_path: str | Path,
    threshold: float = 0.5,
    els_threshold: float = 0.5,
) -> pd.DataFrame:
    """Run + save the failure analysis (``outputs/metrics/failure_analysis.csv``)."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df, counts = run_failure_analysis(
        results, diagnosis_names, concept_names,
        threshold=threshold, els_threshold=els_threshold,
    )
    df.to_csv(csv_path, index=False)
    return df


def plot_failure_analysis(
    counts: pd.DataFrame,
    save_path: str | Path,
) -> Path:
    """Bar chart of category counts (``outputs/figures/failure_analysis.png``)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    categories = ["A", "B", "C", "D", "E"]
    values = [int(counts.loc[counts["category"] == c, "count"].sum() or 0) for c in categories]
    labels = [
        "A: diag+concept+loc",
        "B: diag+concept, loc err",
        "C: diag ok, concept err",
        "D: diag err, concept ok",
        "E: all incorrect",
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(categories, values, color="#4c72b0")
    ax.set_ylabel("Number of test images")
    ax.set_title("Failure analysis: category counts")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, str(int(v)),
                ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(values) * 1.15 if values and max(values) > 0 else 1)
    for i, c in enumerate(categories):
        ax.text(i, -max(values) * 0.22 if values and max(values) > 0 else -0.05,
                labels[i], ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
