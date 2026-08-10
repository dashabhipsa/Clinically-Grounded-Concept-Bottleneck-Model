"""Figure helpers for Phase 3 analysis (interventions).

Consumes the outputs of :mod:`src.interventions.experiments` and
:mod:`src.interventions.runner` directly; every value plotted is a real
model-computed number.

* :func:`plot_dependency_matrix` -- ``|abs_change|`` from
  ``run_dependency_analysis``.
* :func:`plot_robustness`        -- ``concept_error_robustness`` frame.
* :func:`plot_completeness`      -- ``concept_completeness`` summary frame.
* :func:`plot_recovery_bars`     -- ``concept_error_recovery`` result.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def _save(fig, save_path: Path, dpi: int = 150) -> Path:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return save_path


def plot_dependency_matrix(
    matrix: np.ndarray,
    concept_names: Sequence[str],
    diagnosis_names: Sequence[str],
    save_path: str | Path,
    title: str = "Concept -> diagnosis dependency (mean |delta P|)",
) -> Path:
    """Heatmap of per-(concept, diagnosis) intervention effect."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix, dtype=np.float64)
    k, d = matrix.shape
    if k != len(concept_names) or d != len(diagnosis_names):
        raise ValueError("matrix shape does not match the name lists")
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * d + 3), max(5, 0.45 * k + 2)))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(d))
    ax.set_xticklabels(list(diagnosis_names), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(k))
    ax.set_yticklabels(list(concept_names), fontsize=8)
    for i in range(k):
        for j in range(d):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if matrix[i, j] > matrix.max() * 0.6 else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="mean |diagnosis probability change|")
    fig.tight_layout()
    return _save(fig, Path(save_path))


def plot_robustness(
    frames: Dict[str, pd.DataFrame],
    models: Sequence[str],
    save_path: str | Path,
    metric: str = "auroc",
    title: str = "Diagnostic AUROC under concept corruption",
) -> Path:
    """Line plot of a diagnostic metric vs. corruption fraction per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model in models:
        df = frames.get(model)
        if df is None or df.empty or metric not in df.columns:
            continue
        ax.plot(df["fraction"], df[metric], marker="o", label=model)
    ax.set_xlabel("concept corruption fraction")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save(fig, Path(save_path))


def plot_completeness(
    summary: pd.DataFrame,
    save_path: str | Path,
    title: str = "Concept completeness (masked bottleneck)",
) -> Path:
    """Diagnostic + concept AUROC vs. number of concepts in the subset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if summary.empty or "n_concepts" not in summary.columns:
        raise ValueError("summary frame must contain 'n_concepts'")
    x = summary["n_concepts"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if "diagnosis_auroc" in summary.columns:
        ax.plot(x, summary["diagnosis_auroc"], marker="o", label="diagnosis AUROC")
    if "concept_auroc" in summary.columns:
        ax.plot(x, summary["concept_auroc"], marker="s", label="concept AUROC")
    ax.set_xlabel("number of concepts in subset")
    ax.set_ylabel("AUROC (macro)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save(fig, Path(save_path))


def plot_recovery_bars(
    recovery: Dict,
    save_path: str | Path,
    title: str = "Diagnostic metrics: original vs. corrected concepts",
) -> Path:
    """Grouped bar chart for ``concept_error_recovery`` results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["auroc", "auprc", "f1", "sensitivity", "specificity"]
    orig = recovery.get("original", {}).get("macro", {})
    corr = recovery.get("corrected", {}).get("macro", {})
    labels = [m.upper() for m in metrics]
    x = np.arange(len(metrics))
    vals_orig = [orig.get(m, float("nan")) for m in metrics]
    vals_corr = [corr.get(m, float("nan")) for m in metrics]
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, vals_orig, width, label="original (predicted concepts)")
    ax.bar(x + width / 2, vals_corr, width, label="corrected (GT concepts)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max([v for v in vals_orig + vals_corr if np.isfinite(v)] or [1]) * 1.15)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, Path(save_path))
