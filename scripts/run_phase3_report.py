"""Phase 3 final experiment + report pipeline.

Collates every Phase 3 analysis for the trained checkpoints found under
``outputs/checkpoints``:

1. diagnostic / concept / spatial metrics (Phase 1+2 evaluation),
2. intervention experiments (oracle, robustness, recovery, completeness,
   dependency) for CBM-style models,
3. A-E failure analysis,
4. qualitative evidence figures (Grad-CAM / concept-evidence / intervention),
5. structured fact-only explanations,

then writes the final comparison and a master markdown report under
``outputs/reports/phase3/``.

**This script does not train models and does not fabricate results.** Models
that have no checkpoint on disk are reported as N/A (pending).

Usage:
    python scripts/run_phase3_report.py \
        [--base configs/base.yaml] \
        [--checkpoints-dir outputs/checkpoints] \
        [--out-dir outputs/reports/phase3] \
        [--root /path/to/vindr-cxr] \
        [--override data.batch_size=16] \
        [--device auto] [--max-images 200] [--qualitative 6] [--explanations 3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.reporting.analysis import run_model_analysis  # noqa: E402
from src.reporting.discovery import discover_report_models  # noqa: E402
from src.reporting.report import render_master_report  # noqa: E402
from src.training.trainer import resolve_device  # noqa: E402
from src.utils.config import apply_overrides, load_config, resolve  # noqa: E402
from src.utils.logging import setup_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 report pipeline.")
    parser.add_argument("--base", default="configs/base.yaml",
                        help="Base config (used for data/splits defaults).")
    parser.add_argument("--checkpoints-dir", default="outputs/checkpoints",
                        help="Directory containing trained checkpoints.")
    parser.add_argument("--out-dir", default="outputs/reports/phase3",
                        help="Where the report and analysis artifacts go.")
    parser.add_argument("--root", default=None,
                        help="Override data.root for all checkpoints.")
    parser.add_argument("--override", action="append", default=[],
                        help="key=value config override (repeatable).")
    parser.add_argument("--device", default=None,
                        help="torch device; default: config's device (auto).")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=200,
                        help="Cap for per-image intervention records / dependency.")
    parser.add_argument("--qualitative", type=int, default=6,
                        help="Number of qualitative evidence figures to write.")
    parser.add_argument("--explanations", type=int, default=3,
                        help="Number of structured explanations to write.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    logger = setup_logger(
        "phase3_report",
        log_dir=resolve(project_root, "outputs/logs"),
        log_file="phase3_report.log",
    )

    base_cfg = load_config(args.base)
    set_seed(int(base_cfg.get("seed", 42)))

    checkpoints_dir = resolve(project_root, args.checkpoints_dir)
    out_dir = resolve(project_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Discovering checkpoints under %s", checkpoints_dir)
    discovered = discover_report_models(checkpoints_dir, logger)
    if not discovered:
        logger.warning(
            "No trained checkpoints found under %s. The report will be written "
            "with every metric as N/A (pending) -- nothing is fabricated. "
            "Train the models first with scripts/train_*.py.",
            checkpoints_dir,
        )

    bundles = []
    for key, model in discovered.items():
        cfg = model.config
        apply_overrides(cfg, args.override)
        if args.root:
            cfg.set("data.root", args.root)
        if args.batch_size:
            cfg.set("data.batch_size", args.batch_size)

        device = resolve_device(args.device or cfg.get("device", "auto"))
        logger.info("=" * 70)
        logger.info("Analyzing %s (run %s, %s)", key, model.run_id, device)

        bundle = run_model_analysis(
            model,
            device=device,
            project_root=project_root,
            out_dir=out_dir,
            split=args.split,
            batch_size=args.batch_size,
            max_images=args.max_images,
            n_qualitative=args.qualitative,
            n_explanations=args.explanations,
            logger=logger,
        )
        bundles.append(bundle)
        errors = bundle.get("errors") or []
        logger.info(
            "[%s] done; %d errors", model.run_id, len(errors)
        )
        for error in errors:
            logger.warning("    - %s", error)

    logger.info("Assembling final comparison + master report -> %s", out_dir)
    report_path = render_master_report(bundles, discovered, out_dir)
    logger.info("Report written -> %s", report_path)

    # Summary line clearly separating code status from experimental results.
    n_models = len(discovered)
    if n_models:
        logger.info(
            "Pipeline complete: %d/%d model keys analyzed; artifacts in %s. "
            "See %s. (All values are computed from real model outputs; "
            "missing keys are reported as N/A.)",
            len(bundles), 4, out_dir, report_path,
        )
    else:
        logger.info(
            "Pipeline complete (no checkpoints): report written to %s with "
            "all metrics pending/N/A. Train and re-run to populate.",
            report_path,
        )


if __name__ == "__main__":
    main()
