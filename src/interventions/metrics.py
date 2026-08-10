"""Intervention metrics: per-image and per-concept diagnosis probability
changes caused by a concept intervention.

For every concept we record, for each diagnosis:

* ``original_prob``      -- diagnosis probability before the intervention
* ``intervened_prob``    -- diagnosis probability after the intervention
* ``delta_prob``         -- intervened - original
* ``abs_delta``          -- |delta_prob|
* ``relative_change``    -- delta / original, where mathematically valid
                            (NaN when the original probability is 0)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


class InterventionMetrics:
    """Compute and aggregate diagnosis-probability deltas from interventions."""

    @staticmethod
    def rows_for_intervention(
        image_id: str,
        concept: str,
        mode: str,
        value: float,
        delta: float,
        diagnosis_names: Sequence[str],
        probs_before: np.ndarray,
        probs_after: np.ndarray,
    ) -> List[Dict]:
        """One CSV row per diagnosis for a single intervention."""
        p_before = np.asarray(probs_before, dtype=np.float64).reshape(-1)
        p_after = np.asarray(probs_after, dtype=np.float64).reshape(-1)
        if len(p_before) != len(p_after):
            raise ValueError("probs_before and probs_after must have same length")
        rows: List[Dict] = []
        for j, name in enumerate(diagnosis_names):
            original = float(p_before[j])
            intervened = float(p_after[j])
            delta_prob = intervened - original
            rel = (
                (delta_prob / original)
                if abs(original) > 1e-12
                else float("nan")
            )
            rows.append(
                {
                    "image_id": image_id,
                    "concept": concept,
                    "mode": mode,
                    "value": value,
                    "delta": delta,
                    "diagnosis": name,
                    "original_prob": original,
                    "intervened_prob": intervened,
                    "delta_prob": delta_prob,
                    "abs_delta": abs(delta_prob),
                    "relative_change": rel,
                }
            )
        return rows

    # ------------------------------------------------------------------ #
    @staticmethod
    def per_concept_summary(records: List[Dict]) -> pd.DataFrame:
        """Aggregate intervention records per (concept, diagnosis) pair."""
        df = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
        if df.empty:
            return df
        grouped = (
            df.groupby(["concept", "diagnosis"], as_index=False)
            .agg(
                n_images=("abs_delta", "size"),
                mean_abs_delta=("abs_delta", "mean"),
                mean_delta=("delta_prob", "mean"),
                mean_original=("original_prob", "mean"),
                mean_intervened=("intervened_prob", "mean"),
            )
        )
        return grouped

    @staticmethod
    def concept_totals(records: List[Dict]) -> pd.DataFrame:
        """Aggregate intervention records per concept (across diagnoses)."""
        df = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
        if df.empty:
            return df
        grouped = (
            df.groupby(["concept", "mode"], as_index=False)
            .agg(
                n_images=("abs_delta", "size"),
                mean_abs_delta=("abs_delta", "mean"),
                mean_delta=("delta_prob", "mean"),
            )
        )
        return grouped


def save_interventions_csv(records: List[Dict], path: str | Path) -> Path:
    """Persist intervention records (``outputs/metrics/interventions.csv``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
    return path


def save_dependency_matrix(
    matrix: np.ndarray,
    concept_names: Sequence[str],
    diagnosis_names: Sequence[str],
    path: str | Path,
    metric: str = "abs_change",
) -> Path:
    """Save the concept-diagnosis dependency matrix as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        matrix, index=list(concept_names), columns=list(diagnosis_names)
    )
    df.index.name = f"concept_{metric}"
    df.to_csv(path)
    return path


def relative_change(original: float, changed: float) -> float:
    if abs(float(original)) <= 1e-12:
        return float("nan")
    return (float(changed) - float(original)) / float(original)


def describe_delta(original: float, changed: float) -> str:
    delta = changed - original
    if delta > 0:
        return f"{original:.3f} -> {changed:.3f} (+{delta:.3f})"
    if delta < 0:
        return f"{original:.3f} -> {changed:.3f} ({delta:.3f})"
    return f"{original:.3f} -> {changed:.3f} (no change)"


def _finite_mean(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")
