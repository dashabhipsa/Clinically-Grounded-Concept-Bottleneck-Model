"""Final comparison + master report assembly.

Collates the per-model analysis bundles produced by
:mod:`src.reporting.analysis` into:

* ``final_comparison.{csv,md,tex}``  -- four-way model comparison (reusing
  ``src.evaluation.final_comparison``),
* ``report.md``                      -- the master Phase 3 report.

Cells that are not applicable or not yet computed are rendered as ``N/A``;
nothing is invented.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.evaluation.final_comparison import (
    collect_model_metrics,
    final_comparison_csv,
    final_comparison_markdown,
    final_comparison_tex,
)


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(fv):
        return "N/A"
    return f"{fv:.{digits}f}"


def _macro(metrics: Optional[Dict], block: str, metric: str):
    if not metrics:
        return None
    return (metrics.get(block) or {}).get("macro", {}).get(metric)


def assemble_metrics_by_model(bundles: Sequence[Dict]) -> Dict[str, Dict]:
    """Map each canonical model key to its ``collect_model_metrics`` row.

    ``blackbox_gradcam`` is the black-box model viewed with Grad-CAM: it has
    no separate checkpoint and inherits the black-box diagnostic metrics.
    """
    rows: Dict[str, Dict] = {}
    for bundle in bundles:
        experiment = bundle.get("experiment")
        if experiment is None:
            continue
        metrics = bundle.get("metrics") or {}
        rows[experiment] = collect_model_metrics(metrics, experiment)
    if "blackbox" in rows and "blackbox_gradcam" not in rows:
        rows["blackbox_gradcam"] = dict(rows["blackbox"])
        rows["blackbox_gradcam"]["model"] = "blackbox_gradcam"
    return rows


def write_final_comparison(
    metrics_by_model: Dict[str, Dict], out_dir: str | Path
) -> List[Path]:
    """Write CSV / Markdown / LaTeX versions of the final comparison."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [
        final_comparison_csv(metrics_by_model, out_dir / "final_comparison.csv"),
        final_comparison_markdown(metrics_by_model, out_dir / "final_comparison.md"),
        final_comparison_tex(metrics_by_model, out_dir / "final_comparison.tex"),
    ]
    return written


# --------------------------------------------------------------------------- #
# Report sections
# --------------------------------------------------------------------------- #
def _status_table(bundles: Sequence[Dict], discovered: Dict[str, object]) -> str:
    lines = ["## 1. Experiment status", "",
             "| Model | Checkpoint found | run_id | Status | Notes |",
             "|---|---|---|---|---|"]
    keys = ("blackbox", "blackbox_gradcam", "cbm", "spatial_cbm")
    bundle_by_key = {b.get("experiment"): b for b in bundles}
    for key in keys:
        model = discovered.get(key)
        if model is None:
            lines.append(f"| {key} | no | - | pending | no checkpoint on disk |")
            continue
        bundle = bundle_by_key.get(key, {})
        errors = bundle.get("errors") or []
        status = "computed" if bundle.get("metrics") else "errored"
        note = "; ".join(errors[:2]) if errors else "metrics + analysis written"
        lines.append(
            f"| {key} | yes | {model.run_id} | {status} | {note} |"
        )
    lines.append("")
    return "\n".join(lines)


def _metrics_section(bundle: Dict) -> str:
    metrics = bundle.get("metrics") or {}
    experiment = bundle.get("experiment", "?")
    run_id = bundle.get("run_id", "?")
    lines = [f"### {experiment} — run {run_id}", ""]

    diag = metrics.get("diagnosis", {}).get("macro", {})
    if diag:
        lines += [
            "**Diagnosis (macro):**",
            "",
            "| Metric | Value |",
            "|---|---|",
        ]
        for name in ("auroc", "auprc", "f1", "precision", "recall",
                     "sensitivity", "specificity"):
            lines.append(f"| {name} | {_fmt(diag.get(name))} |")
        lines.append("")

    concepts = metrics.get("concepts", {}).get("macro", {})
    if concepts:
        lines += ["**Concepts (macro):**", "",
                  "| Metric | Value |", "|---|---|"]
        for name in ("auroc", "auprc", "f1"):
            lines.append(f"| {name} | {_fmt(concepts.get(name))} |")
        lines.append("")

    spatial = metrics.get("spatial", {}).get("macro", {})
    if spatial:
        lines += ["**Spatial (macro):**", "",
                  "| Metric | Value |", "|---|---|"]
        for name in ("iou", "pointing_game", "els"):
            lines.append(f"| {name} | {_fmt(spatial.get(name))} |")
        lines.append("")

    outputs = bundle.get("outputs") or {}
    figures = outputs.get("figures") or []
    if figures:
        lines.append("**Qualitative figures:**")
        for path in figures[:20]:
            lines.append(f"- `{Path(path).name}`")
        if len(figures) > 20:
            lines.append(f"- ... and {len(figures) - 20} more")
        lines.append("")

    errors = bundle.get("errors") or []
    if errors:
        lines.append("**Errors (analysis steps skipped):**")
        for error in errors:
            lines.append(f"- {error}")
        lines.append("")
    return "\n".join(lines)


def _interventions_section(bundle: Dict) -> str:
    interventions = bundle.get("interventions")
    experiment = bundle.get("experiment", "?")
    if not interventions:
        return (
            f"### Interventions — {experiment}\n\n"
            f"Not computed (not applicable for `{experiment}` or analysis failed).\n"
        )
    lines = [f"### Interventions — {experiment}", ""]

    oracle = interventions.get("oracle")
    if oracle:
        pred = oracle.get("predicted_concepts", {}).get("macro", {})
        gt = oracle.get("oracle_concepts", {}).get("macro", {})
        lines += ["**Oracle concepts** (diagnosis metrics, predicted vs GT concept vector):",
                  "",
                  "| Metric | Predicted concepts | Oracle (GT) concepts | Gap |",
                  "|---|---|---|---|"]
        for name in ("auroc", "auprc", "f1"):
            lines.append(
                f"| {name} | {_fmt(pred.get(name))} | {_fmt(gt.get(name))} | "
                f"{_fmt(oracle.get('gap', {}).get(name))} |"
            )
        lines.append("")

    robustness = interventions.get("robustness")
    if robustness is not None and not robustness.empty:
        lines += ["**Concept error robustness** (diagnosis AUROC under corruption):",
                  "",
                  "| Fraction | AUROC | AUPRC | F1 |",
                  "|---|---|---|---|"]
        for _, row in robustness.iterrows():
            lines.append(
                f"| {row['fraction']:.2f} | {_fmt(row.get('auroc'))} | "
                f"{_fmt(row.get('auprc'))} | {_fmt(row.get('f1'))} |"
            )
        lines.append("")

    recovery = interventions.get("recovery")
    if recovery:
        imp = recovery.get("improvement", {})
        orig = recovery.get("original", {}).get("macro", {})
        corr = recovery.get("corrected", {}).get("macro", {})
        lines += ["**Concept error recovery** (fix predicted concepts with GT):",
                  "",
                  "| Metric | Original | Corrected | Improvement |",
                  "|---|---|---|---|"]
        for name in ("auroc", "auprc", "f1", "sensitivity", "specificity"):
            lines.append(
                f"| {name} | {_fmt(orig.get(name))} | {_fmt(corr.get(name))} | "
                f"{_fmt(imp.get(name))} |"
            )
        lines.append("")

    completeness = interventions.get("completeness")
    if completeness and not completeness["summary"].empty:
        lines += ["**Concept completeness** (masked bottleneck):",
                  "",
                  "| n_concepts | Diagnosis AUROC | Concept AUROC |",
                  "|---|---|---|"]
        for _, row in completeness["summary"].iterrows():
            lines.append(
                f"| {row['n_concepts']} | {_fmt(row.get('diagnosis_auroc'))} | "
                f"{_fmt(row.get('concept_auroc'))} |"
            )
        lines.append("")

    dependency = interventions.get("dependency")
    if dependency:
        abs_change = np.asarray(dependency["abs_change"])
        top = np.argsort(abs_change, axis=None)[::-1][:5]
        k_idx = [int(i) // abs_change.shape[1] for i in top]
        d_idx = [int(i) % abs_change.shape[1] for i in top]
        concept_names = list(bundle["model"].config.concepts)
        diagnosis_names = list(bundle["model"].config.diagnoses)
        lines += ["**Top concept -> diagnosis dependencies** (mean |delta P| under 'remove'):",
                  "",
                  "| Concept | Diagnosis | |delta P| |",
                  "|---|---|---|"]
        for k, d in zip(k_idx, d_idx):
            lines.append(
                f"| {concept_names[k]} | {diagnosis_names[d]} | "
                f"{abs_change[k, d]:.4f} |"
            )
        lines.append("")

    if not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _failure_section(bundle: Dict) -> str:
    experiment = bundle.get("experiment", "?")
    df = bundle.get("failure_df")
    if df is None:
        return (
            f"### Failure analysis — {experiment}\n\n"
            f"Not computed (not applicable for `{experiment}`).\n"
        )
    counts = df["category"].value_counts().reindex(
        ["A", "B", "C", "D", "E"], fill_value=0
    )
    lines = [
        f"### Failure analysis — {experiment}",
        "",
        "| Category | Meaning | Images |",
        "|---|---|---|",
    ]
    meanings = {
        "A": "correct diag + concept + localization",
        "B": "correct diag + concept, incorrect localization",
        "C": "correct diag, incorrect concept",
        "D": "incorrect diag, correct concept",
        "E": "incorrect diag + concept + localization",
    }
    for cat in ["A", "B", "C", "D", "E"]:
        lines.append(f"| {cat} | {meanings[cat]} | {int(counts.get(cat, 0))} |")
    lines.append("")
    return "\n".join(lines)


def _pending_section(metrics_by_model: Dict[str, Dict], bundles: Sequence[Dict]) -> str:
    keys = ("blackbox", "blackbox_gradcam", "cbm", "spatial_cbm")
    missing = [k for k in keys if k not in metrics_by_model]
    lines = ["## 4. Pending / not computed", ""]
    if missing:
        lines.append(
            "The following model keys have no checkpoint on disk yet; every "
            "cell they would fill is reported as N/A above:"
        )
        for key in missing:
            lines.append(f"- `{key}`")
    else:
        lines.append("All four model keys were found on disk.")
    lines.append("")
    lines.append(
        "To fill in the pending cells: run `scripts/train_blackbox.py`, "
        "`scripts/train_cbm.py` and `scripts/train_spatial_cbm.py`, then re-run "
        "`python scripts/run_phase3_report.py`."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Master report
# --------------------------------------------------------------------------- #
def render_master_report(
    bundles: Sequence[Dict],
    discovered: Dict[str, object],
    out_dir: str | Path,
    timestamp: Optional[str] = None,
) -> Path:
    """Write ``report.md`` aggregating every computed analysis."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    metrics_by_model = assemble_metrics_by_model(bundles)
    write_final_comparison(metrics_by_model, out_dir)

    sections: List[str] = [
        "# Phase 3 Report — Intervention, Robustness and Failure Analysis",
        "",
        f"*Generated: {timestamp}*",
        "",
        "> **Code and pipeline status**: this report is produced by "
        "`scripts/run_phase3_report.py` and reuses the Phase 1/2 evaluation "
        "modules. **Values marked N/A are experiments that have not been run "
        "or that do not apply to the model**; no number is estimated. Run the "
        "training scripts first, then re-run this report.",
        "",
    ]
    sections.append(_status_table(bundles, discovered))
    sections.append("## 2. Final model comparison")
    sections.append("")
    sections.append("The table below mirrors `final_comparison.md`.")
    sections.append("")

    comparison_md = (out_dir / "final_comparison.md")
    if comparison_md.exists():
        sections.append(comparison_md.read_text(encoding="utf-8"))
    sections.append("")

    sections.append("## 3. Per-model analysis")
    sections.append("")
    for key in ("blackbox", "cbm", "spatial_cbm"):
        bundle = next((b for b in bundles if b.get("experiment") == key), None)
        if bundle is None:
            sections.append(f"### {key}\n\nNo checkpoint on disk — pending.\n")
            continue
        sections.append(_metrics_section(bundle))
        sections.append(_interventions_section(bundle))
        sections.append(_failure_section(bundle))

    sections.append(_pending_section(metrics_by_model, bundles))

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return report_path
