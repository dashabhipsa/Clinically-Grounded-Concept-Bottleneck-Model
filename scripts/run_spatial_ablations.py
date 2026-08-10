"""Run the Phase 2 grounding-weight ablations.

    A. Standard CBM
    B. Spatial CBM, grounding_weight = 0.1
    C. Spatial CBM, grounding_weight = 0.25
    D. Spatial CBM, grounding_weight = 0.5
    E. Spatial CBM, grounding_weight = 1.0

Each experiment is trained from its config under ``configs/spatial_ablations/``
and evaluated on the official test split; results are aggregated into
``outputs/metrics/spatial_ablations/summary.csv`` (plus a markdown table).

Usage:
    python scripts/run_spatial_ablations.py \
        --base configs/base.yaml \
        --ablations-dir configs/spatial_ablations \
        [--root /path/to/vindr-cxr] [--override training.epochs=30]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.cbm import build_cbm  # noqa: E402
from src.models.spatial_cbm import build_spatial_cbm  # noqa: E402
from src.training.trainer import run_experiment  # noqa: E402
from src.utils.config import apply_overrides, load_config, resolve  # noqa: E402
from src.utils.logging import setup_logger  # noqa: E402

ABLATIONS = [
    ("cbm", "cbm.yaml"),
    ("spatial_0.1", "spatial_0.1.yaml"),
    ("spatial_0.25", "spatial_0.25.yaml"),
    ("spatial_0.5", "spatial_0.5.yaml"),
    ("spatial_1.0", "spatial_1.0.yaml"),
]

SUMMARY_COLUMNS = [
    "ablation", "experiment", "seed",
    "diagnosis_auroc_macro", "diagnosis_auprc_macro", "diagnosis_f1_macro",
    "diagnosis_sensitivity_macro", "diagnosis_specificity_macro",
    "concept_auroc_macro", "concept_auprc_macro", "concept_f1_macro",
    "spatial_iou_macro", "spatial_pointing_game_macro", "spatial_els_macro",
    "checkpoint_dir",
]


def _extract(summary: dict) -> dict:
    test = summary.get("test", {})
    diag = test.get("diagnosis", {}).get("macro", {})
    concepts = test.get("concepts", {}).get("macro", {})
    spatial = test.get("spatial", {}).get("macro", {})
    return {
        "diagnosis_auroc_macro": diag.get("auroc"),
        "diagnosis_auprc_macro": diag.get("auprc"),
        "diagnosis_f1_macro": diag.get("f1"),
        "diagnosis_sensitivity_macro": diag.get("sensitivity"),
        "diagnosis_specificity_macro": diag.get("specificity"),
        "concept_auroc_macro": concepts.get("auroc"),
        "concept_auprc_macro": concepts.get("auprc"),
        "concept_f1_macro": concepts.get("f1"),
        "spatial_iou_macro": spatial.get("iou"),
        "spatial_pointing_game_macro": spatial.get("pointing_game"),
        "spatial_els_macro": spatial.get("els"),
        "checkpoint_dir": summary.get("checkpoint_dir"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the grounding-weight ablations.")
    parser.add_argument("--base", default="configs/base.yaml")
    parser.add_argument("--ablations-dir", default="configs/spatial_ablations")
    parser.add_argument("--root", default=None, help="Override data.root.")
    parser.add_argument("--override", action="append", default=[],
                        help="key=value config override (repeatable).")
    parser.add_argument("--only", default=None,
                        help="Comma-separated ablation names to run (e.g. spatial_0.5).")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    ablations_dir = Path(args.ablations_dir)
    logger = setup_logger(
        "ablations",
        log_dir=resolve(project_root, "outputs/logs"),
        log_file="spatial_ablations.log",
    )

    only = {s.strip() for s in args.only.split(",")} if args.only else None
    rows = []
    for name, cfg_file in ABLATIONS:
        if only and name not in only:
            continue
        cfg = load_config(args.base, ablations_dir / cfg_file)
        apply_overrides(cfg, args.override)
        if args.root:
            cfg.set("data.root", args.root)

        builder = build_cbm if cfg.get("experiment") == "cbm" else build_spatial_cbm
        logger.info("=" * 70)
        logger.info("Running ablation %s (experiment=%s)", name, cfg.get("experiment"))
        summary = run_experiment(cfg, builder, logger, run_id=f"ablation_{name}")
        rows.append({"ablation": name, "experiment": summary["experiment"],
                     "seed": summary.get("seed"), **_extract(summary)})

    out_dir = resolve(project_root, "outputs/metrics") / "spatial_ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Ablation summary written -> %s", csv_path)

    md_path = out_dir / "summary.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("| Ablation | Diag AUROC | Diag AUPRC | Concept AUROC | ELS | IoU | Pointing |\n")
        fh.write("|---|---|---|---|---|---|---|\n")
        for row in rows:
            fmt = lambda v: "N/A" if v is None else f"{v:.4f}"
            fh.write(
                f"| {row['ablation']} | {fmt(row['diagnosis_auroc_macro'])} | "
                f"{fmt(row['diagnosis_auprc_macro'])} | {fmt(row['concept_auroc_macro'])} | "
                f"{fmt(row['spatial_els_macro'])} | {fmt(row['spatial_iou_macro'])} | "
                f"{fmt(row['spatial_pointing_game_macro'])} |\n"
            )
    logger.info("Ablation markdown written -> %s", md_path)


if __name__ == "__main__":
    main()
