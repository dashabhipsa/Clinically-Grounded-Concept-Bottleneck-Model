"""Run the Phase 2 spatial-supervision corruption experiments.

Training-time box supervision is corrupted deterministically (via the run
seed): correct / shifted / random / no box supervision. Each condition is
trained from ``configs/spatial_corruption/<mode>.yaml`` and evaluated on the
official test split (using the *uncorrupted* GT boxes). Results are
aggregated into ``outputs/metrics/spatial_corruption/summary.csv``.

Usage:
    python scripts/run_box_corruption.py \
        --base configs/base.yaml \
        --corruption-dir configs/spatial_corruption \
        [--root /path/to/vindr-cxr] [--override training.epochs=30]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.spatial_cbm import build_spatial_cbm  # noqa: E402
from src.training.trainer import run_experiment  # noqa: E402
from src.utils.config import apply_overrides, load_config, resolve  # noqa: E402
from src.utils.logging import setup_logger  # noqa: E402

CORRUPTIONS = ["correct", "shifted", "random", "none"]

SUMMARY_COLUMNS = [
    "mode", "seed", "grounding_weight",
    "diagnosis_auroc_macro", "diagnosis_auprc_macro",
    "concept_auroc_macro",
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
        "concept_auroc_macro": concepts.get("auroc"),
        "spatial_iou_macro": spatial.get("iou"),
        "spatial_pointing_game_macro": spatial.get("pointing_game"),
        "spatial_els_macro": spatial.get("els"),
        "checkpoint_dir": summary.get("checkpoint_dir"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the box-corruption experiments.")
    parser.add_argument("--base", default="configs/base.yaml")
    parser.add_argument("--corruption-dir", default="configs/spatial_corruption")
    parser.add_argument("--root", default=None, help="Override data.root.")
    parser.add_argument("--override", action="append", default=[],
                        help="key=value config override (repeatable).")
    parser.add_argument("--only", default=None,
                        help="Comma-separated corruption modes to run (e.g. shifted).")
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    corruption_dir = Path(args.corruption_dir)
    logger = setup_logger(
        "corruption",
        log_dir=resolve(project_root, "outputs/logs"),
        log_file="box_corruption.log",
    )

    only = {s.strip() for s in args.only.split(",")} if args.only else None
    rows = []
    for mode in CORRUPTIONS:
        if only and mode not in only:
            continue
        cfg = load_config(args.base, corruption_dir / f"{mode}.yaml")
        apply_overrides(cfg, args.override)
        if args.root:
            cfg.set("data.root", args.root)

        logger.info("=" * 70)
        logger.info("Running corruption mode: %s", mode)
        summary = run_experiment(cfg, build_spatial_cbm, logger, run_id=f"corrupt_{mode}")
        rows.append({
            "mode": mode,
            "seed": summary.get("seed"),
            "grounding_weight": cfg.get("model.grounding.weight"),
            **_extract(summary),
        })

    out_dir = resolve(project_root, "outputs/metrics") / "spatial_corruption"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Corruption summary written -> %s", csv_path)


if __name__ == "__main__":
    main()
