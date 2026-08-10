"""Train the Standard Concept Bottleneck Model.

    X-ray -> DenseNet-121 -> ConceptHead -> concept vector z -> DiagnosisHead

There is NO direct encoder -> diagnosis connection.

Usage:
    python scripts/train_cbm.py --config configs/base.yaml --config configs/cbm.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.cbm import build_cbm  # noqa: E402
from src.training.trainer import run_experiment  # noqa: E402
from src.utils.config import apply_overrides, load_config, resolve  # noqa: E402
from src.utils.logging import setup_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the standard CBM.")
    parser.add_argument("--config", action="append", required=True,
                        help="Config file(s); base first, experiment second.")
    parser.add_argument("--root", default=None, help="Override data.root.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--override", action="append", default=[],
                        help="key=value config override (repeatable).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(*args.config)
    apply_overrides(cfg, args.override)
    if args.root:
        cfg.set("data.root", args.root)

    project_root = Path(__file__).resolve().parents[1]
    logger = setup_logger(
        "train_cbm",
        log_dir=resolve(project_root, cfg.get("outputs.log_dir", "outputs/logs")),
        log_file="cbm.log",
    )

    summary = run_experiment(cfg, build_cbm, logger, run_id=args.run_id)
    logger.info("CBM run complete: %s", summary["run_id"])


if __name__ == "__main__":
    main()
