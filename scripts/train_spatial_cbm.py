"""Train the Spatially Grounded Concept Bottleneck Model (Phase 2).

    Chest X-ray -> DenseNet-121 -> spatial feature map
               -> concept-specific spatial heads -> concept activation maps
               -> concept scores -> concept bottleneck z -> diagnosis

There is NO direct encoder -> diagnosis connection. Concept evidence comes
from the concept-specific heads (never post-hoc Grad-CAM).

Usage:
    python scripts/train_spatial_cbm.py \
        --config configs/base.yaml --config configs/spatial_cbm.yaml \
        [--root /path/to/vindr-cxr] [--run-id spatial_0.5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.spatial_cbm import build_spatial_cbm  # noqa: E402
from src.training.trainer import run_experiment  # noqa: E402
from src.utils.config import apply_overrides, load_config, resolve  # noqa: E402
from src.utils.logging import setup_logger  # noqa: E402


def spatial_post_train(model, config, device, project_root, run_id, logger):
    """After training: generate concept-evidence figures on the test split.

    Both successful (high ELS) and failure (low ELS) examples are saved under
    ``outputs/figures/spatial/<run_id>/``.
    """
    from torch.utils.data import DataLoader

    from src.data.dataset import VinDrCXRDataset, collate_vindr
    from src.data.split import load_split_ids
    from src.data.transforms import build_transform
    from src.evaluation.spatial_visualization import generate_spatial_examples

    splits = load_split_ids(resolve(project_root, config.get("data.splits_dir", "data/splits")))
    test_ids = splits["test"]
    image_size = int(config.data.image_size)
    mean = list(config.get("data.normalize.mean", [0.485, 0.456, 0.406]))
    std = list(config.get("data.normalize.std", [0.229, 0.224, 0.225]))
    transform = build_transform(image_size, mean, std, train=False)
    dataset = VinDrCXRDataset(
        config.data.root, test_ids, list(config.concepts), list(config.diagnoses),
        split="test", image_size=image_size, transform=transform,
    )
    loader = DataLoader(
        dataset, batch_size=int(config.data.batch_size), shuffle=False,
        num_workers=0, collate_fn=collate_vindr,
    )
    figures = resolve(project_root, config.get("outputs.figures_dir", "outputs/figures")) / "spatial" / run_id
    written = generate_spatial_examples(
        model, dataset, device, out_dir=figures, mean=mean, std=std,
    )
    logger.info("Spatial concept-evidence figures written: %d -> %s", len(written), figures)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Spatially Grounded CBM.")
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
        "train_spatial_cbm",
        log_dir=resolve(project_root, cfg.get("outputs.log_dir", "outputs/logs")),
        log_file="spatial_cbm.log",
    )

    summary = run_experiment(
        cfg, build_spatial_cbm, logger,
        run_id=args.run_id, post_train=spatial_post_train,
    )
    logger.info("Spatial CBM run complete: %s", summary["run_id"])


if __name__ == "__main__":
    main()
