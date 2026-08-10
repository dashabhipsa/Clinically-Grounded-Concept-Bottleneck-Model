"""Evaluate a saved checkpoint on a chosen split.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/blackbox_<ts>/best_model.pth \
        --split test [--root /path/to/vindr-cxr] [--gradcam]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data.dataset import VinDrCXRDataset  # noqa: E402
from src.data.split import load_split_ids  # noqa: E402
from src.data.transforms import build_transform  # noqa: E402
from src.evaluation.evaluation import (  # noqa: E402
    build_metric_fn,
    predict,
    save_metrics_json,
    save_predictions,
)
from src.explainability.gradcam import generate_gradcam_examples  # noqa: E402
from src.models import build_model  # noqa: E402
from src.training.trainer import resolve_device  # noqa: E402
from src.utils.config import Config, apply_overrides, load_config, resolve, save_config  # noqa: E402
from src.utils.logging import setup_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved model checkpoint.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a saved checkpoint (*.pth).")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--root", default=None, help="Override data.root.")
    parser.add_argument("--out-dir", default=None,
                        help="Where to write metrics/predictions (default: next to checkpoint).")
    parser.add_argument("--gradcam", action="store_true",
                        help="Generate Grad-CAM examples (black-box models only).")
    parser.add_argument("--num-gradcam", type=int, default=8)
    parser.add_argument("--config", action="append", default=[],
                        help="Optional config files to merge under the checkpoint config.")
    parser.add_argument("--override", action="append", default=[],
                        help="key=value config override (repeatable).")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = Config(checkpoint["config"])
    if args.config:
        for extra in args.config:
            cfg.update(load_config(extra))
    apply_overrides(cfg, args.override)
    if args.root:
        cfg.set("data.root", args.root)
    if args.batch_size:
        cfg.set("data.batch_size", args.batch_size)

    project_root = Path(__file__).resolve().parents[1]
    logger = setup_logger(
        "evaluate",
        log_dir=resolve(project_root, cfg.get("outputs.log_dir", "outputs/logs")),
        log_file="evaluate.log",
    )
    set_seed(int(cfg.get("seed", 42)))

    data_root = Path(cfg.data.root)
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    splits = load_split_ids(resolve(project_root, cfg.get("data.splits_dir", "data/splits")))
    split_ids = splits.get(args.split)
    if not split_ids:
        raise ValueError(f"Unknown split {args.split!r}; available: {sorted(splits)}")

    device = resolve_device(cfg.get("device", "auto"))
    transform = build_transform(
        int(cfg.data.image_size),
        list(cfg.get("data.normalize.mean", [0.485, 0.456, 0.406])),
        list(cfg.get("data.normalize.std", [0.229, 0.224, 0.225])),
        train=False,
    )
    dataset = VinDrCXRDataset(
        data_root, split_ids, list(cfg.concepts), list(cfg.diagnoses),
        split=args.split, image_size=int(cfg.data.image_size), transform=transform,
    )
    from torch.utils.data import DataLoader

    from src.data.dataset import collate_vindr
    loader = DataLoader(dataset, batch_size=int(cfg.data.batch_size), shuffle=False,
                        num_workers=0, collate_fn=collate_vindr)

    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded %s (epoch %s) from %s", type(model).__name__,
                checkpoint.get("epoch"), ckpt_path)

    results = predict(model, loader, device)
    metric_fn = build_metric_fn(str(cfg.get("experiment", "blackbox")), cfg)
    metrics = metric_fn(results)

    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": cfg.get("experiment", "blackbox"),
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "num_samples": len(split_ids),
        **metrics,
    }
    metrics_path = out_dir / f"metrics_{args.split}.json"
    save_metrics_json(summary, metrics_path)
    preds_path = out_dir / f"predictions_{args.split}.csv"
    save_predictions(results, list(cfg.diagnoses), list(cfg.concepts), preds_path)
    save_config(cfg, out_dir / f"config_{args.split}.yaml")
    logger.info("Saved %s and %s", metrics_path, preds_path)

    if args.gradcam and cfg.get("experiment", "blackbox") == "blackbox":
        figures = resolve(project_root, cfg.get("outputs.figures_dir", "outputs/figures")) / "gradcam" / ckpt_path.parent.name
        written = generate_gradcam_examples(
            model, dataset, device, out_dir=figures, num_examples=args.num_gradcam,
        )
        logger.info("Grad-CAM: wrote %d examples to %s", len(written), figures)


if __name__ == "__main__":
    main()
