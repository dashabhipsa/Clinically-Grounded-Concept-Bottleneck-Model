"""Preliminary ChestX-Det subset experiment (fast-track).

Trains three models on a deterministic subset of the remote ChestX-Det
dataset (``natealberti/ChestX-Det``, Hugging Face -- NOT VinDr-CXR):

    1. Black-box baseline
    2. Standard Concept Bottleneck Model
    3. Spatially Grounded CBM (Strategy-B boxes as grounding supervision)

then evaluates on a held-out subset of the ChestX-Det ``test`` split
(diagnosis AUROC/AUPRC/F1/sensitivity/specificity; concept AUROC/AUPRC/F1;
spatial IoU / Pointing Game / ELS), runs the oracle concept-intervention
experiment on both CBMs, writes real checkpoints / predictions / metrics /
figures, and emits a summary report clearly labelled
"Preliminary ChestX-Det subset experiment".

ALL caches, temporary files, checkpoints and outputs are placed on the D:
drive (under the project root). Results are computed from real model outputs;
nothing is fabricated.

Usage:
    python scripts/train_chestxdet.py \
        [--base configs/chestxdet_base.yaml] \
        [--only blackbox,cbm,spatial_cbm] \
        [--epochs 3] \
        [--override key=value]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# D:-drive-only caches. MUST run before importing torch/datasets/hub.
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CACHE_ROOT = PROJECT_ROOT / ".cache"
for _sub in ("hf", "datasets", "hub", "torch", "tmp"):
    (_CACHE_ROOT / _sub).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_CACHE_ROOT / "hf"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE_ROOT / "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_CACHE_ROOT / "hub"))
os.environ.setdefault("TORCH_HOME", str(_CACHE_ROOT / "torch"))
os.environ["TMP"] = str(_CACHE_ROOT / "tmp")
os.environ["TEMP"] = str(_CACHE_ROOT / "tmp")
os.environ["TMPDIR"] = str(_CACHE_ROOT / "tmp")
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data import chestxdet as cxd  # noqa: E402
from src.data.transforms import build_transform  # noqa: E402
from src.evaluation.evaluation import (  # noqa: E402
    build_metric_fn,
    predict,
    save_metrics_json,
    save_predictions,
)
from src.evaluation.qualitative import generate_qualitative_examples  # noqa: E402
from src.evaluation.spatial_visualization import generate_spatial_examples  # noqa: E402
from src.explainability.gradcam import generate_gradcam_examples  # noqa: E402
from src.interventions.experiments import (  # noqa: E402
    oracle_concept_experiment,
    summarize_oracle_experiment,
)
from src.models import build_model  # noqa: E402
from src.training.losses import build_loss_fn  # noqa: E402
from src.training.trainer import Trainer, resolve_device  # noqa: E402
from src.utils.config import apply_overrides, load_config, resolve  # noqa: E402
from src.utils.logging import setup_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

MAPPING_CONFIG = PROJECT_ROOT / "configs" / "chestxdet_concepts.yaml"


# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #
class DecodedListDataset:
    """Random-access wrapper over a materialized list of decoded samples."""

    def __init__(self, items: List[dict], concepts: List[str]):
        self.items = list(items)
        self.concepts = list(concepts)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def collate_chestxdet(batch: List[dict]) -> dict:
    """Collate preserving variable-length boxes / string ids (mask optional)."""
    out = {
        "image": torch.stack([b["image"] for b in batch]),
        "concepts": torch.stack([b["concepts"] for b in batch]),
        "diagnosis": torch.stack([b["diagnosis"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
    }
    if "mask" in batch[0]:
        out["mask"] = torch.stack([b["mask"] for b in batch])
    return out


def materialize_split(
    *,
    repo_id: str,
    split: str,
    offset: int,
    n: int,
    image_size: int,
    transform,
    concept_names: List[str],
    concept_mapping: Dict[str, List[str]],
    box_strategy: str,
    min_component_area_fraction: float,
    logger=None,
    attempts: int = 8,
) -> DecodedListDataset:
    """Stream ``n`` decoded samples from the remote split into a list."""
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            ds = cxd.ChestXDetRemoteDataset(
                repo_id=repo_id,
                split=split,
                image_size=image_size,
                transform=transform,
                concept_names=concept_names,
                concept_mapping=concept_mapping,
                box_strategy=box_strategy,
                min_component_area_fraction=min_component_area_fraction,
                max_samples=n,
                offset=offset,
            )
            items: List[dict] = []
            for sample in ds:
                sample.pop("mask", None)  # not needed downstream; saves RAM
                items.append(sample)
                if len(items) >= n:
                    break
            if len(items) < n:
                raise RuntimeError(
                    f"only {len(items)}/{n} samples streamed for {split}@{offset}"
                )
            if logger:
                logger.info("materialized %s split: %d samples", split, len(items))
            return DecodedListDataset(items, concept_names)
        except Exception as exc:  # transient HF stream errors
            last_exc = exc
            if logger:
                logger.warning(
                    "stream attempt %d failed (%s: %s)",
                    attempt + 1, type(exc).__name__, str(exc)[:120],
                )
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"failed to materialize split {split}@{offset}: {last_exc}")


def make_loader(
    dataset, batch_size: int, shuffle: bool = False, seed: int = 42
) -> DataLoader:
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_chestxdet,
        generator=gen,
    )


# --------------------------------------------------------------------------- #
# Per-model train + evaluate + intervene + figure
# --------------------------------------------------------------------------- #
def run_one_model(
    cfg,
    model_name: str,
    train_loader,
    val_loader,
    test_loader,
    test_dataset,
    device,
    out_dir: Path,
    logger,
) -> Dict:
    """Train one model, evaluate on test, run interventions, write figures."""
    experiment = str(cfg.get("experiment"))
    ckpt_dir = out_dir / "checkpoints" / model_name
    metrics_dir = out_dir / "metrics" / model_name
    predictions_dir = out_dir / "predictions"
    figures_dir = out_dir / "figures" / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    threshold = float(cfg.get("evaluation.threshold", 0.5))
    diagnosis_names = list(cfg.diagnoses)
    concept_names = list(cfg.concepts)
    mean = list(cfg.get("data.normalize.mean", [0.485, 0.456, 0.406]))
    std = list(cfg.get("data.normalize.std", [0.229, 0.224, 0.225]))

    logger.info("=" * 70)
    logger.info("Training %s (%s) | epochs=%d | lr=%.1e",
                model_name, experiment, int(cfg.training.epochs),
                float(cfg.training.learning_rate))

    model = build_model(cfg).to(device)
    loss_fn = build_loss_fn(experiment, cfg)
    metric_fn = build_metric_fn(experiment, cfg)

    trainer = Trainer(
        model=model, config=cfg, device=device,
        train_loader=train_loader, val_loader=val_loader,
        loss_fn=loss_fn, metric_fn=metric_fn, logger=logger,
        checkpoint_dir=ckpt_dir,
    )

    t0 = time.time()
    history, best_epoch = trainer.fit()
    train_time = time.time() - t0
    logger.info("[%s] training done in %.1fs (best epoch %d)", model_name, train_time, best_epoch)

    checkpoint = torch.load(ckpt_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    t0 = time.time()
    results = predict(model, test_loader, device)
    metrics = metric_fn(results)
    eval_time = time.time() - t0

    bundle: Dict = {
        "experiment": experiment,
        "model_name": model_name,
        "best_epoch": int(best_epoch),
        "train_time_sec": round(train_time, 2),
        "eval_time_sec": round(eval_time, 2),
        "checkpoint_dir": str(ckpt_dir),
        "metrics_json": str(metrics_dir / "metrics.json"),
        "predictions_csv": str(predictions_dir / f"{model_name}_test.csv"),
        "figures": [],
        "test_diagnosis_macro": metrics.get("diagnosis", {}).get("macro", {}),
        "test_concepts_macro": metrics.get("concepts", {}).get("macro", {}),
        "test_spatial_macro": metrics.get("spatial", {}).get("macro", {}),
    }
    save_metrics_json(
        {
            "experiment": experiment,
            "model_name": model_name,
            "split": "test",
            "num_samples": len(results.get("image_ids", [])),
            "best_epoch": best_epoch,
            "train_time_sec": train_time,
            **metrics,
        },
        metrics_dir / "metrics.json",
    )
    save_predictions(
        results, diagnosis_names, concept_names,
        predictions_dir / f"{model_name}_test.csv",
    )
    logger.info(
        "[%s] test diag AUROC macro=%.4f | concept AUROC macro=%s",
        model_name,
        bundle["test_diagnosis_macro"].get("auroc", float("nan")),
        _fmt(bundle["test_concepts_macro"].get("auroc")),
    )
    if bundle["test_spatial_macro"]:
        logger.info(
            "[%s] spatial IoU=%.4f | pointing=%.4f | ELS=%.4f",
            model_name,
            bundle["test_spatial_macro"].get("iou", float("nan")),
            bundle["test_spatial_macro"].get("pointing_game", float("nan")),
            bundle["test_spatial_macro"].get("els", float("nan")),
        )

    # ---- concept intervention experiment (CBMs only) ---------------------- #
    if experiment in ("cbm", "spatial_cbm") and "concepts_logits" in results:
        int_dir = out_dir / "interventions" / model_name
        int_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        oracle = oracle_concept_experiment(model, results, diagnosis_names, threshold)
        bundle["oracle"] = summarize_oracle_experiment(oracle)
        bundle["interventions_csv"] = str(int_dir / "oracle.csv")
        _write_dict_row(bundle["oracle"], int_dir / "oracle.csv")
        bundle["intervention_time_sec"] = round(time.time() - t0, 2)
        logger.info(
            "[%s] oracle intervention: diag AUROC gap=%.4f",
            model_name, bundle["oracle"].get("gap_auroc", float("nan")),
        )

    # ---- qualitative figures ---------------------------------------------- #
    try:
        t0 = time.time()
        written: List[Path] = []
        if experiment == "blackbox":
            written = [
                Path(p) for p in generate_gradcam_examples(
                    model, test_dataset, device, figures_dir, num_examples=4,
                    mean=mean, std=std,
                )
            ]
        elif experiment == "cbm":
            written = generate_qualitative_examples(
                model, test_dataset, device, figures_dir,
                diagnosis_names, concept_names,
                num_success=2, num_failure=2, mean=mean, std=std,
            )
        else:
            written = [
                Path(p) for p in generate_spatial_examples(
                    model, test_dataset, device, figures_dir,
                    num_success=2, num_failure=2, mean=mean, std=std,
                )
            ]
            written += generate_qualitative_examples(
                model, test_dataset, device, figures_dir,
                diagnosis_names, concept_names,
                num_success=2, num_failure=2, mean=mean, std=std,
            )
        bundle["figures"] = [str(p) for p in written]
        bundle["figures_time_sec"] = round(time.time() - t0, 2)
        logger.info("[%s] %d figures written -> %s", model_name, len(written), figures_dir)
    except Exception as exc:  # defensive: figures must not kill the run
        logger.warning("[%s] figures failed: %s", model_name, exc)
        bundle["figure_error"] = str(exc)

    return bundle


def _write_dict_row(row: Dict, path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _fmt(value, digits: int = 4):
    if value is None:
        return "N/A"
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{fv:.{digits}f}" if np.isfinite(fv) else "N/A"


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(summary: Dict, env: Dict, report_path: Path) -> None:
    rows = summary["models"]
    md: List[str] = []
    md.append("# Preliminary ChestX-Det subset experiment\n")
    md.append(
        "Preliminary results on a deterministic subset of the remote "
        "**ChestX-Det** dataset (`natealberti/ChestX-Det`, Hugging Face) -- "
        "**not** VinDr-CXR. All values below are computed from actually run "
        "model outputs; nothing is fabricated.\n"
    )

    md.append("## Environment\n")
    md.append("| Item | Value |")
    md.append("|---|---|")
    for key, value in env.items():
        md.append(f"| {key} | {value} |")

    ds = summary["dataset"]
    md.append("\n## Dataset\n")
    md.append(
        f"| Item | Value |\n|---|---|\n"
        f"| Source | `{ds['repo_id']}` (Hugging Face) |\n"
        f"| Train subset | {ds['n_train']} |\n"
        f"| Validation subset | {ds['n_val']} |\n"
        f"| Test subset | {ds['n_test']} |\n"
        f"| Image size | {ds['image_size']}x{ds['image_size']} |\n"
        f"| Concepts | {len(ds['concepts'])} ({ds['n_mapped']}/8 mapped, "
        f"{ds['n_unmapped']} all-zero) |\n"
        f"| Box strategy | {ds['box_strategy']} |\n"
    )
    md.append("Mapped concepts: " + ", ".join(f"`{c}`" for c in ds["mapped"]) + ".\n")
    md.append(
        "Unavailable concepts (all-zero vectors): "
        + ", ".join(f"`{c}`" for c in ds["unmapped"]) + ".\n"
    )

    md.append("## Training time (CPU)\n")
    md.append("| Model | Best epoch | Train (s) | Eval (s) | Total run (s) |")
    md.append("|---|---|---|---|---|")
    for name in summary["order"]:
        b = rows[name]
        md.append(
            f"| {name} | {b['best_epoch']} | {_fmt(b.get('train_time_sec'))} | "
            f"{_fmt(b.get('eval_time_sec'))} | {_fmt(b.get('total_time_sec'))} |"
        )
    md.append(f"\n**Total wall time: {_fmt(summary['total_time_sec'])} s**\n")

    md.append("\n## Diagnosis metrics (test subset, macro)\n")
    md.append(
        "| Model | AUROC | AUPRC | F1 | Sensitivity | Specificity |"
    )
    md.append("|---|---|---|---|---|---|")
    for name in summary["order"]:
        m = rows[name]["test_diagnosis_macro"]
        md.append(
            f"| {name} | {_fmt(m.get('auroc'))} | {_fmt(m.get('auprc'))} | "
            f"{_fmt(m.get('f1'))} | {_fmt(m.get('sensitivity'))} | "
            f"{_fmt(m.get('specificity'))} |"
        )

    md.append("\n## Concept metrics (test subset, macro)\n")
    md.append("| Model | AUROC | AUPRC | F1 |")
    md.append("|---|---|---|---|")
    for name in summary["order"]:
        m = rows[name].get("test_concepts_macro") or {}
        md.append(
            f"| {name} | {_fmt(m.get('auroc'))} | {_fmt(m.get('auprc'))} | "
            f"{_fmt(m.get('f1'))} |"
        )

    md.append("\n## Spatial fidelity metrics (test subset, macro)\n")
    md.append("| Model | IoU | Pointing Game | ELS |")
    md.append("|---|---|---|---|")
    for name in summary["order"]:
        m = rows[name].get("test_spatial_macro") or {}
        md.append(
            f"| {name} | {_fmt(m.get('iou'))} | "
            f"{_fmt(m.get('pointing_game'))} | {_fmt(m.get('els'))} |"
        )

    md.append("\n## Oracle concept intervention (test subset)\n")
    md.append(
        "Diagnosis metrics recomputed through the bottleneck using "
        "ground-truth concepts instead of predicted ones (`gap` = oracle - predicted).\n"
    )
    md.append("| Model | Pred AUROC | Oracle AUROC | Gap AUROC | Gap AUPRC | Gap F1 |")
    md.append("|---|---|---|---|---|---|")
    for name in summary["order"]:
        o = rows[name].get("oracle")
        if not o:
            md.append(f"| {name} | N/A | N/A | N/A | N/A | N/A |")
            continue
        md.append(
            f"| {name} | {_fmt(o.get('predicted_concepts_auroc'))} | "
            f"{_fmt(o.get('oracle_concepts_auroc'))} | "
            f"{_fmt(o.get('gap_auroc'))} | {_fmt(o.get('gap_auprc'))} | "
            f"{_fmt(o.get('gap_f1'))} |"
        )

    md.append("\n## Artifacts (checkpoints / metrics / predictions / figures)\n")
    md.append("| Model | Checkpoint | Metrics | Predictions | Figures |")
    md.append("|---|---|---|---|---|")
    for name in summary["order"]:
        b = rows[name]
        figs = ", ".join(f"`{Path(p).name}`" for p in b.get("figures", [])[:3]) or "none"
        md.append(
            f"| {name} | `{b['checkpoint_dir']}` | `{b['metrics_json']}` | "
            f"`{b['predictions_csv']}` | {figs} |"
        )
    md.append("\n## Limitations\n")
    for limitation in summary["limitations"]:
        md.append(f"- {limitation}")

    md.append("\n---\n*Generated by `scripts/train_chestxdet.py`.*\n")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(md), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChestX-Det fast-track experiment.")
    parser.add_argument("--base", default="configs/chestxdet_base.yaml")
    parser.add_argument(
        "--only", default="blackbox,cbm,spatial_cbm",
        help="Comma-separated subset of models to train.",
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training.epochs for all models.")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t_start = time.time()

    logger = setup_logger(
        "train_chestxdet",
        log_dir=resolve(PROJECT_ROOT, "outputs/logs"),
        log_file="chestxdet_fasttrack.log",
    )

    base_cfg = load_config(args.base)
    apply_overrides(base_cfg, args.override)
    if args.epochs:
        base_cfg.set("training.epochs", args.epochs)
    seed = int(base_cfg.get("seed", 42))
    set_seed(seed)

    repo_id = str(base_cfg.get("chestxdet.repo_id", cxd.CHESTX_DET_REPO))
    n_train = int(base_cfg.get("chestxdet.n_train", 500))
    n_val = int(base_cfg.get("chestxdet.n_val", 100))
    n_test = int(base_cfg.get("chestxdet.n_test", 200))
    train_split_size = int(cxd.SPLIT_SIZES.get("train", 3025))
    if n_train + n_val > train_split_size:
        n_original = n_train
        n_train = max(0, train_split_size - n_val)
        logger.warning(
            "n_train + n_val (%d + %d) exceeds the %d-sample train split; "
            "capping n_train to %d (val split is the contiguous slice after train)",
            n_original, n_val, train_split_size, n_train,
        )
    box_strategy = str(base_cfg.get("chestxdet.box_strategy", "B_label_components"))
    min_area_frac = float(base_cfg.get("chestxdet.min_component_area_fraction", 0.0005))
    image_size = int(base_cfg.get("data.image_size", 256))
    batch_size = int(base_cfg.get("data.batch_size", 8))
    mean = list(base_cfg.get("data.normalize.mean", [0.485, 0.456, 0.406]))
    std = list(base_cfg.get("data.normalize.std", [0.229, 0.224, 0.225]))

    with open(MAPPING_CONFIG, encoding="utf-8") as fh:
        mapping_raw = yaml.safe_load(fh)
    concept_mapping: Dict[str, List[str]] = mapping_raw["concept_mapping"]
    concept_names = list(base_cfg.concepts)
    mapped = [c for c in concept_names if concept_mapping.get(c)]
    unmapped = [c for c in concept_names if not concept_mapping.get(c)]

    device = resolve_device(base_cfg.get("device", "auto"))
    logger.info("Device: %s (cuda=%s) | seed=%d", device, torch.cuda.is_available(), seed)
    logger.info(
        "ChestX-Det subset: train=%d val=%d test=%d | concepts=%d (%d mapped, %d all-zero) | box strategy=%s",
        n_train, n_val, n_test, len(concept_names), len(mapped), len(unmapped), box_strategy,
    )

    transform = build_transform(image_size, mean, std, train=False, augment=False)
    logger.info("Materializing deterministic ChestX-Det subsets (first run downloads to .cache on D:)...")
    train_ds = materialize_split(
        repo_id=repo_id, split="train", offset=0, n=n_train,
        image_size=image_size, transform=transform,
        concept_names=concept_names, concept_mapping=concept_mapping,
        box_strategy=box_strategy, min_component_area_fraction=min_area_frac,
        logger=logger,
    )
    val_ds = materialize_split(
        repo_id=repo_id, split="train", offset=n_train, n=n_val,
        image_size=image_size, transform=transform,
        concept_names=concept_names, concept_mapping=concept_mapping,
        box_strategy=box_strategy, min_component_area_fraction=min_area_frac,
        logger=logger,
    )
    test_ds = materialize_split(
        repo_id=repo_id, split="test", offset=0, n=n_test,
        image_size=image_size, transform=transform,
        concept_names=concept_names, concept_mapping=concept_mapping,
        box_strategy=box_strategy, min_component_area_fraction=min_area_frac,
        logger=logger,
    )

    n_boxes = sum(1 for item in test_ds.items for _ in item["boxes"])
    n_boxed_images = sum(1 for item in test_ds.items if item["boxes"])
    logger.info(
        "test subset boxes: %d boxes across %d/%d images",
        n_boxes, n_boxed_images, len(test_ds),
    )

    train_loader = make_loader(train_ds, batch_size, shuffle=True, seed=seed)
    val_loader = make_loader(val_ds, batch_size, shuffle=False, seed=seed)
    test_loader = make_loader(test_ds, batch_size, shuffle=False, seed=seed)

    out_dir = resolve(PROJECT_ROOT, base_cfg.get("outputs.chestxdet_dir", "outputs/chestxdet_fasttrack"))
    out_dir.mkdir(parents=True, exist_ok=True)

    experiment_map = {
        "blackbox": "configs/chestxdet_blackbox.yaml",
        "cbm": "configs/chestxdet_cbm.yaml",
        "spatial_cbm": "configs/chestxdet_spatial_cbm.yaml",
    }
    order = [m for m in ("blackbox", "cbm", "spatial_cbm") if m in args.only.split(",")]
    if not order:
        raise ValueError("--only must include at least one of blackbox,cbm,spatial_cbm")

    rows: Dict[str, Dict] = {}
    for model_name in order:
        cfg = load_config(args.base, experiment_map[model_name])
        apply_overrides(cfg, args.override)
        if args.epochs:
            cfg.set("training.epochs", args.epochs)
        cfg.set("seed", seed)
        bundle = run_one_model(
            cfg, model_name,
            train_loader, val_loader, test_loader, test_ds,
            device, out_dir, logger,
        )
        bundle["total_time_sec"] = round(time.time() - t_start, 2)
        rows[model_name] = bundle

    summary = {
        "experiment": "Preliminary ChestX-Det subset experiment",
        "environment": {
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)",
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
        },
        "dataset": {
            "repo_id": repo_id,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "image_size": image_size,
            "batch_size": batch_size,
            "concepts": concept_names,
            "mapped": mapped,
            "unmapped": unmapped,
            "n_mapped": len(mapped),
            "n_unmapped": len(unmapped),
            "box_strategy": box_strategy,
            "test_boxes_total": n_boxes,
            "test_images_with_boxes": n_boxed_images,
            "test_images_total": len(test_ds),
        },
        "models": rows,
        "order": order,
        "total_time_sec": round(time.time() - t_start, 2),
        "limitations": [
            "CPU-only environment (no GPU available); torch "
            f"{torch.__version__} is the CPU build.",
            "Only 6 of 8 project concepts map to ChestX-Det classes; "
            "`lung_opacity` and `edema` concept vectors are all-zero.",
            "ChestX-Det has no independent global diagnosis labels: "
            "diagnosis vectors are derived from the same pixel-level findings "
            "as the concepts, so diagnosis and concept tasks are not "
            "independent (unlike VinDr-CXR).",
            "Spatial annotations are sparse and the `mask`/`label` fields are "
            "only partially consistent (see reports/chestxdet_validation.md); "
            "Strategy-B boxes come from connected components of the label maps.",
            "Preliminary subset sizes (500/100/200) and 3 epochs; results are "
            "NOT comparable to full-dataset numbers.",
        ],
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    report_path = resolve(PROJECT_ROOT, base_cfg.get("outputs.report_dir", "outputs/reports")) / "chestxdet_fasttrack.md"
    write_report(summary, summary["environment"], report_path)

    logger.info("=" * 70)
    logger.info("Experiment complete. Total wall time: %.1fs", summary["total_time_sec"])
    logger.info("Summary  -> %s", summary_path)
    logger.info("Report   -> %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
