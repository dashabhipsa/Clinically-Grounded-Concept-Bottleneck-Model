"""Prepare the VinDr-CXR dataset: validate layout, build reproducible
train/val/test splits and print per-class statistics.

Usage:
    python scripts/prepare_data.py --config configs/base.yaml --root /path/to/vindr-cxr
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.annotations import (  # noqa: E402
    concepts_from_boxes,
    find_image_ids,
    image_labels_to_multihot,
    load_boxes,
    load_image_labels,
)
from src.data.preprocessing import inspect_dicom  # noqa: E402
from src.data.split import create_splits, official_split_ids  # noqa: E402
from src.utils.config import apply_overrides, load_config, resolve  # noqa: E402
from src.utils.logging import setup_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare VinDr-CXR data and splits.")
    parser.add_argument("--config", action="append", required=True,
                        help="Config file(s); base first, experiment second.")
    parser.add_argument("--root", default=None, help="Override data.root.")
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate existing split CSVs.")
    parser.add_argument("--validate-samples", type=int, default=5,
                        help="Number of random DICOMs to verify are loadable.")
    parser.add_argument("--override", action="append", default=[],
                        help="key=value config override (repeatable).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(*args.config)
    apply_overrides(cfg, args.override)

    if args.root:
        cfg.set("data.root", args.root)
    if args.val_fraction is not None:
        cfg.set("data.val_fraction", args.val_fraction)
    if args.seed is not None:
        cfg.set("data.split_seed", args.seed)
    if args.splits_dir:
        cfg.set("data.splits_dir", args.splits_dir)

    project_root = Path(__file__).resolve().parents[1]
    data_root = Path(cfg.data.root)
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {data_root}. Please download VinDr-CXR and "
            "point config 'data.root' (or --root) at it."
        )

    logger = setup_logger("prepare_data", log_dir=resolve(project_root, cfg.outputs.log_dir))
    logger.info("VinDr-CXR root: %s", data_root)
    if not (data_root / "images").is_dir():
        raise FileNotFoundError(f"Missing images directory: {data_root / 'images'}")

    official = official_split_ids(data_root)
    logger.info(
        "Official split sizes -> train: %d, test: %d",
        len(official["train"]), len(official["test"]),
    )

    splits_dir = resolve(project_root, cfg.get("data.splits_dir", "data/splits"))
    splits = create_splits(
        data_root,
        splits_dir,
        val_fraction=cfg.get("data.val_fraction", 0.1),
        seed=cfg.get("data.split_seed", 42),
        overwrite=args.overwrite,
    )
    for name, ids in splits.items():
        logger.info("Split %-5s -> %d images -> %s/%s.csv", name, len(ids), splits_dir, name)

    # ---- per-class statistics ----------------------------------------- #
    concept_names = list(cfg.concepts)
    diagnosis_names = list(cfg.diagnoses)

    train_boxes = load_boxes(data_root / "annotations" / "train.csv")
    test_boxes = load_boxes(data_root / "annotations" / "test.csv")
    train_labels = load_image_labels(data_root / "annotations" / "image_labels_train.csv")
    test_labels = load_image_labels(data_root / "annotations" / "image_labels_test.csv")

    all_ids = find_image_ids(train_boxes, test_boxes, train_labels, test_labels)
    concept_matrix = concepts_from_boxes(train_boxes, concept_names, image_ids=all_ids)
    diagnosis_matrix = image_labels_to_multihot(train_labels, diagnosis_names, image_ids=all_ids)

    logger.info("Concept prevalence (from local findings, whole official set):")
    for concept in concept_names:
        logger.info("  %-20s %6.1f%%", concept, 100.0 * float(concept_matrix[concept].mean()))
    logger.info("Diagnosis prevalence (from global labels):")
    for diag in diagnosis_names:
        logger.info("  %-20s %6.1f%%", diag, 100.0 * float(diagnosis_matrix[diag].mean()))

    if args.validate_samples > 0:
        import random
        random.seed(cfg.get("seed", 42))
        sample_ids = random.sample(all_ids, min(args.validate_samples, len(all_ids)))
        ok = 0
        for image_id in sample_ids:
            try:
                info = inspect_dicom(data_root / "images" / f"{image_id}.dicom")
                ok += 1
                logger.info("  image %s OK: %s", image_id, info)
            except Exception as exc:  # pragma: no cover - depends on data
                logger.error("  image %s FAILED: %s", image_id, exc)
        logger.info("Validated %d/%d sample DICOMs.", ok, len(sample_ids))

    logger.info("Done. Splits written to %s", splits_dir)


if __name__ == "__main__":
    main()
