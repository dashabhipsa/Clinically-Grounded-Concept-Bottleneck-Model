"""Train the black-box diagnostic baseline (DenseNet-121 -> diagnosis).

Usage:
    python scripts/train_blackbox.py --config configs/base.yaml --config configs/blackbox.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import VinDrCXRDataset  # noqa: E402
from src.data.split import load_split_ids  # noqa: E402
from src.data.transforms import build_transform  # noqa: E402
from src.explainability.gradcam import generate_gradcam_examples  # noqa: E402
from src.models.blackbox import build_blackbox  # noqa: E402
from src.training.trainer import resolve_device, run_experiment  # noqa: E402
from src.utils.config import apply_overrides, load_config, resolve  # noqa: E402
from src.utils.logging import setup_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the black-box diagnostic baseline.")
    parser.add_argument("--config", action="append", required=True,
                        help="Config file(s); base first, experiment second.")
    parser.add_argument("--root", default=None, help="Override data.root.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-gradcam", action="store_true",
                        help="Skip post-hoc Grad-CAM example generation.")
    parser.add_argument("--override", action="append", default=[],
                        help="key=value config override (repeatable).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(*args.config)
    apply_overrides(cfg, args.override)
    if args.root:
        cfg.set("data.root", args.root)
    if args.no_gradcam:
        cfg.set("evaluation.gradcam.enabled", False)

    project_root = Path(__file__).resolve().parents[1]
    logger = setup_logger(
        "train_blackbox",
        log_dir=resolve(project_root, cfg.get("outputs.log_dir", "outputs/logs")),
        log_file="blackbox.log",
    )

    def post_train(model, config, device, project_root, run_id, logger):
        gc_cfg = config.get("evaluation.gradcam", {})
        if not gc_cfg.get("enabled", True):
            logger.info("Grad-CAM disabled; skipping.")
            return
        splits = load_split_ids(resolve(project_root, config.get("data.splits_dir", "data/splits")))
        val_ds = VinDrCXRDataset(
            Path(config.data.root), splits["val"], list(config.concepts),
            list(config.diagnoses), split="val",
            image_size=int(config.data.image_size),
            transform=build_transform(
                int(config.data.image_size),
                list(config.get("data.normalize.mean", [0.485, 0.456, 0.406])),
                list(config.get("data.normalize.std", [0.229, 0.224, 0.225])),
                train=False,
            ),
        )
        out_dir = resolve(project_root, config.get("outputs.figures_dir", "outputs/figures")) / "gradcam" / run_id
        num_examples = int(gc_cfg.get("num_examples", 8))
        written = generate_gradcam_examples(
            model, val_ds, resolve_device(config.get("device", "auto")),
            out_dir=out_dir, num_examples=num_examples,
        )
        logger.info("Grad-CAM: wrote %d examples to %s", len(written), out_dir)

    summary = run_experiment(
        cfg, build_blackbox, logger, run_id=args.run_id, post_train=post_train
    )
    logger.info("Black-box run complete: %s", summary["run_id"])


if __name__ == "__main__":
    main()
