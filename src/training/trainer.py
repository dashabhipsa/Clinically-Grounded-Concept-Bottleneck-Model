"""Training loop with mixed precision, scheduler, early stopping and
checkpointing, plus the top-level ``run_experiment`` entry point."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import VinDrCXRDataset, collate_vindr
from src.data.split import ensure_splits_exist
from src.data.transforms import build_transform
from src.evaluation.evaluation import build_metric_fn, predict, save_metrics_json, save_predictions
from src.models import build_model
from src.training.losses import build_loss_fn
from src.utils.config import Config, resolve, save_config
from src.utils.logging import get_logger
from src.utils.seed import seed_worker, set_seed


def resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class Trainer:
    """Generic multi-label trainer for black-box and CBM models."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: Config,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Callable,
        metric_fn: Callable,
        logger=None,
        checkpoint_dir: str | Path | None = None,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.logger = logger or get_logger("trainer")

        training = config.training
        self.epochs = int(training.epochs)
        lr = float(training.learning_rate)
        wd = float(training.weight_decay)
        self.patience = int(training.patience)
        self.monitor_metric = str(training.get("monitor_metric", "diagnosis_auroc_macro"))
        self.maximize = True

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd
        )

        scheduler_name = str(training.get("scheduler", "reduce_on_plateau"))
        if scheduler_name == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5,
                patience=max(2, self.patience // 2),
            )
        elif scheduler_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )
        else:
            raise ValueError(f"Unknown scheduler {scheduler_name!r}")

        self.use_amp = bool(training.get("mixed_precision", True)) and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        self.early_stopping = bool(training.get("early_stopping", True)) and self.patience > 0

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.best_metric = -np.inf
        self.best_epoch = 0
        self.history: List[Dict] = []

    # ------------------------------------------------------------------ #
    def fit(self) -> tuple:
        """Run the full training loop; returns (history, best_epoch)."""
        logger = self.logger
        logger.info(
            "Training %s on %s | %d epochs | lr=%.1e | wd=%.1e | amp=%s",
            type(self.model).__name__, self.device, self.epochs,
            float(self.config.training.learning_rate),
            float(self.config.training.weight_decay), self.use_amp,
        )
        wait = 0
        start = time.time()
        for epoch in range(1, self.epochs + 1):
            _, train_loss = self._run_epoch(self.train_loader, training=True)
            results, val_loss = self._run_epoch(self.val_loader, training=False)
            metrics = self.metric_fn(results)
            monitor = float(metrics.get(self.monitor_metric, np.nan))
            if np.isfinite(monitor):
                score, shown = monitor, monitor
            else:
                # Degenerate validation set (e.g. AUROC undefined): fall back
                # to minimizing validation loss.
                score, shown = -float(val_loss), np.nan

            improved = score > self.best_metric + 1e-6
            if improved:
                self.best_metric = score
                self.best_epoch = epoch
                wait = 0
                if self.checkpoint_dir:
                    self._save_checkpoint(self.checkpoint_dir / "best_model.pth", epoch, metrics)
            else:
                wait += 1

            if self.checkpoint_dir:
                self._save_checkpoint(self.checkpoint_dir / "last_model.pth", epoch, metrics)

            lr_now = self.optimizer.param_groups[0]["lr"]
            self._step_scheduler(score, epoch)
            logger.info(
                "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | %s=%.4f | lr=%.2e | %s",
                epoch, self.epochs, train_loss, val_loss, self.monitor_metric,
                shown, lr_now, "BEST" if improved else " ",
            )
            self.history.append({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "learning_rate": lr_now, **metrics,
            })

            if self.early_stopping and wait >= self.patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs).",
                    epoch, self.patience,
                )
                break

        elapsed = time.time() - start
        logger.info(
            "Training finished in %.1fs | best epoch %d | best %s=%.4f",
            elapsed, self.best_epoch, self.monitor_metric, self.best_metric,
        )
        if self.checkpoint_dir:
            self._save_history(self.checkpoint_dir / "training_history.csv")
            best_path = self.checkpoint_dir / "best_model.pth"
            if not best_path.exists():
                logger.warning(
                    "No monitored improvement during training; saving last_model.pth "
                    "as best_model.pth so evaluation has a checkpoint."
                )
                self._save_checkpoint(best_path, self.best_epoch or 1, {})
        return self.history, self.best_epoch

    # ------------------------------------------------------------------ #
    def _run_epoch(self, loader: DataLoader, training: bool):
        self.model.train(training)
        running_loss = 0.0
        num_samples = 0
        collected = {}

        iterator = tqdm(loader, desc="train" if training else "val", leave=False)
        for batch in iterator:
            x = batch["image"].to(self.device, non_blocking=True)
            labels = {}
            for key in ("diagnosis", "concepts"):
                if key in batch:
                    labels[key] = batch[key].to(self.device, non_blocking=True).float()

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                with torch.autocast(
                    device_type=self.device.type, enabled=self.use_amp and training
                ):
                    loss, outputs = self.loss_fn(self.model, x, labels, batch=batch)
                if training:
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()

            bs = x.size(0)
            running_loss += float(loss.item()) * bs
            num_samples += bs

            for key, value in outputs.items():
                collected.setdefault(f"{key}_logits", []).append(
                    value.detach().float().cpu().numpy()
                )
            for key, value in labels.items():
                collected.setdefault(f"{key}_labels", []).append(
                    value.detach().float().cpu().numpy()
                )
            iterator.set_postfix(loss=float(loss.item()))

        merged = {k: np.concatenate(v, axis=0) for k, v in collected.items()}
        return merged, running_loss / max(num_samples, 1)

    # ------------------------------------------------------------------ #
    def _step_scheduler(self, metric: float, epoch: int) -> None:
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def _save_checkpoint(self, path: Path, epoch: int, metrics: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "config": self.config.to_dict(),
            },
            path,
        )

    def _save_history(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.history)
        df.to_csv(path, index=False)


# ------------------------------------------------------------------------ #
def build_dataloaders(config, splits, device, logger):
    """Create train/val/test datasets and loaders from config."""
    root = Path(config.data.root)
    concept_names = list(config.concepts)
    diagnosis_names = list(config.diagnoses)
    image_size = int(config.data.image_size)
    mean = list(config.get("data.normalize.mean", [0.485, 0.456, 0.406]))
    std = list(config.get("data.normalize.std", [0.229, 0.224, 0.225]))
    augment = bool(config.get("data.augment", True))
    num_workers = int(config.get("data.num_workers", 4))
    batch_size = int(config.data.batch_size)

    train_ds = VinDrCXRDataset(
        root, splits["train"], concept_names, diagnosis_names, split="train",
        image_size=image_size,
        transform=build_transform(image_size, mean, std, train=True, augment=augment),
    )
    val_ds = VinDrCXRDataset(
        root, splits["val"], concept_names, diagnosis_names, split="val",
        image_size=image_size,
        transform=build_transform(image_size, mean, std, train=False),
    )
    test_ds = VinDrCXRDataset(
        root, splits["test"], concept_names, diagnosis_names, split="test",
        image_size=image_size,
        transform=build_transform(image_size, mean, std, train=False),
    )
    logger.info(
        "Datasets: train=%d val=%d test=%d | concepts=%d diagnoses=%d | image_size=%d",
        len(train_ds), len(val_ds), len(test_ds),
        len(concept_names), len(diagnosis_names), image_size,
    )

    generator = torch.Generator().manual_seed(int(config.seed))
    common = dict(num_workers=num_workers, collate_fn=collate_vindr)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        worker_init_fn=seed_worker, generator=generator, **common,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, **common
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, **common
    )
    return train_loader, val_loader, test_loader


def run_experiment(
    config: Config,
    model_builder: Callable,
    logger,
    run_id: str | None = None,
    post_train: Callable | None = None,
) -> dict:
    """End-to-end train + evaluate + persist for one experiment.

    ``model_builder(config) -> nn.Module`` selects black-box vs. CBM.
    Returns a summary dict of the run.
    """
    experiment = str(config.get("experiment", "blackbox"))
    seed = int(config.get("seed", 42))
    set_seed(seed)

    project_root = Path(__file__).resolve().parents[2]
    data_root = Path(config.data.root)
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {data_root}. Set 'data.root' in the config "
            "to your VinDr-CXR download."
        )

    splits_dir = resolve(project_root, config.get("data.splits_dir", "data/splits"))
    splits = ensure_splits_exist(data_root, splits_dir, config)

    device = resolve_device(config.get("device", "auto"))
    logger.info("Device: %s | seed: %d", device, seed)

    train_loader, val_loader, test_loader = build_dataloaders(config, splits, device, logger)

    model = model_builder(config)
    model = model.to(device)

    loss_fn = build_loss_fn(experiment, config)
    metric_fn = build_metric_fn(experiment, config)

    if run_id is None:
        run_id = f"{experiment}_{time.strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = resolve(project_root, config.get("outputs.checkpoint_dir", "outputs/checkpoints")) / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, checkpoint_dir / "config.yaml")

    trainer = Trainer(
        model=model, config=config, device=device,
        train_loader=train_loader, val_loader=val_loader,
        loss_fn=loss_fn, metric_fn=metric_fn, logger=logger,
        checkpoint_dir=checkpoint_dir,
    )
    history, best_epoch = trainer.fit()

    logger.info("Evaluating best checkpoint on the official test set...")
    checkpoint = torch.load(checkpoint_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_results = predict(model, test_loader, device)
    test_metrics = metric_fn(test_results)

    metrics_dir = resolve(project_root, config.get("outputs.metrics_dir", "outputs/metrics")) / run_id
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": experiment,
        "run_id": run_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val": trainer.best_metric,
        "test": test_metrics,
    }
    save_metrics_json(summary, metrics_dir / "metrics.json")

    predictions_dir = resolve(project_root, config.get("outputs.predictions_dir", "outputs/predictions")) / run_id
    predictions_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(
        test_results,
        list(config.diagnoses),
        list(config.concepts),
        predictions_dir / "predictions_test.csv",
    )

    summary["checkpoint_dir"] = str(checkpoint_dir)
    logger.info("Saved metrics -> %s", metrics_dir / "metrics.json")
    logger.info("Saved predictions -> %s", predictions_dir / "predictions_test.csv")

    if post_train is not None:
        post_train(model, config, device, project_root, run_id, logger)

    return summary
