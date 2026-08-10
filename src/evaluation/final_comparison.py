"""Final four-way model comparison aggregation.

Collates already-computed evaluation results for:

A. Black-box classifier
B. Black-box + Grad-CAM
C. Standard CBM
D. Spatially Grounded CBM

into a single table with DIAGNOSTIC / CONCEPT / SPATIAL metrics. Metrics that
are not applicable to a model are reported as ``N/A`` (never fabricated).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

MODEL_ORDER = ["blackbox", "blackbox_gradcam", "cbm", "spatial_cbm"]
COLUMNS = [
    "model",
    "diagnosis_auroc",
    "diagnosis_auprc",
    "diagnosis_f1",
    "diagnosis_sensitivity",
    "diagnosis_specificity",
    "concept_auroc",
    "concept_auprc",
    "concept_f1",
    "spatial_iou",
    "spatial_pointing_game",
    "spatial_els",
]


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(fv):
        return "N/A"
    return f"{fv:.{digits}f}"


def collect_model_metrics(
    model_metrics: Dict[str, Dict],
    model_key: str,
) -> Dict[str, Optional[float]]:
    """Extract the canonical Phase-3 metric row from a metrics dict.

    ``model_metrics`` has the shape produced by ``build_metric_fn``: nested
    ``diagnosis`` / ``concepts`` / ``spatial`` dicts with ``macro`` entries.
    """
    row: Dict[str, Optional[float]] = {"model": model_key}

    diag = (model_metrics.get("diagnosis") or {}).get("macro", {})
    row["diagnosis_auroc"] = diag.get("auroc")
    row["diagnosis_auprc"] = diag.get("auprc")
    row["diagnosis_f1"] = diag.get("f1")
    row["diagnosis_sensitivity"] = diag.get("sensitivity")
    row["diagnosis_specificity"] = diag.get("specificity")

    concepts = (model_metrics.get("concepts") or {}).get("macro", {})
    row["concept_auroc"] = concepts.get("auroc")
    row["concept_auprc"] = concepts.get("auprc")
    row["concept_f1"] = concepts.get("f1")

    spatial = (model_metrics.get("spatial") or {}).get("macro", {})
    row["spatial_iou"] = spatial.get("iou")
    row["spatial_pointing_game"] = spatial.get("pointing_game")
    row["spatial_els"] = spatial.get("els")

    return row


def final_comparison_table(
    metrics_by_model: Dict[str, Dict],
    order: Sequence[str] = MODEL_ORDER,
) -> pd.DataFrame:
    """Assemble the final comparison DataFrame (missing -> None -> N/A)."""
    rows: List[Dict] = []
    for key in order:
        if key in metrics_by_model:
            rows.append(collect_model_metrics(metrics_by_model[key], key))
        else:
            rows.append({"model": key, **{c: None for c in COLUMNS[1:]}})
    return pd.DataFrame(rows, columns=COLUMNS)


def final_comparison_csv(
    metrics_by_model: Dict[str, Dict],
    path: str | Path,
    order: Sequence[str] = MODEL_ORDER,
) -> Path:
    """Write ``outputs/metrics/final_comparison.csv``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = final_comparison_table(metrics_by_model, order)
    df.to_csv(path, index=False)
    return path


def final_comparison_markdown(
    metrics_by_model: Dict[str, Dict],
    path: str | Path,
    order: Sequence[str] = MODEL_ORDER,
) -> Path:
    """Write ``outputs/metrics/final_comparison.md`` (publication-ready)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = final_comparison_table(metrics_by_model, order)

    lines = [
        "# Final Model Comparison",
        "",
        "Values are computed from actual experiment metrics. Cells marked N/A "
        "are not applicable for that model (e.g. no concept bottleneck, or no "
        "spatial evidence).",
        "",
        "| Model | Diagnosis AUROC | Diagnosis AUPRC | Diagnosis F1 | "
        "Diagnosis Sens. | Diagnosis Spec. | Concept AUROC | Concept AUPRC | "
        "Concept F1 | Spatial IoU | Pointing Game | ELS |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        model_cell = str(row["model"])
        lines.append(
            "| " + " | ".join(
                [model_cell] + [_fmt(row[c]) for c in COLUMNS[1:]]
            ) + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def final_comparison_tex(
    metrics_by_model: Dict[str, Dict],
    path: str | Path,
    order: Sequence[str] = MODEL_ORDER,
) -> Path:
    """Write ``outputs/metrics/final_comparison.tex`` (LaTeX table)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = final_comparison_table(metrics_by_model, order)

    label_cols = ["diagnosis_auroc", "diagnosis_auprc", "concept_auroc",
                  "spatial_iou", "spatial_pointing_game", "spatial_els"]
    header = " & ".join(
        ["Model", "Diag AUROC", "Diag AUPRC", "Concept AUROC",
         "IoU", "Pointing", "ELS"]
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Final model comparison (values from actual experiments; "
        r"N/A means not applicable).}",
        r"\label{tab:final_comparison}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        cells = [f"\\texttt{{{row['model']}}}"] + [_fmt(row[c]) for c in label_cols]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
