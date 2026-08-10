"""Checkpoint discovery for the Phase 3 report pipeline.

Scans the experiment checkpoint directory for trained models, reads the config
embedded in each checkpoint, and classifies the experiment type. The report
pipeline consumes only the models actually found on disk -- a missing model is
reported as ``N/A`` in the report, never fabricated.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from src.utils.config import Config

VALID_EXPERIMENTS = ("blackbox", "cbm", "spatial_cbm")

# Canonical keys used by the final comparison table (``final_comparison.py``).
CANONICAL_MODEL_KEYS = ("blackbox", "blackbox_gradcam", "cbm", "spatial_cbm")


@dataclass
class DiscoveredModel:
    """One trained checkpoint plus its embedded configuration."""

    experiment: str          # blackbox | cbm | spatial_cbm
    run_id: str              # checkpoint directory name
    checkpoint_path: Path    # best_model.pth (or last_model.pth fallback)
    checkpoint_dir: Path
    config: Config

    @property
    def canonical_key(self) -> str:
        return self.experiment


def _checkpoint_candidates(checkpoint_dir: Path) -> List[Path]:
    """Prefer ``best_model.pth``, fall back to ``last_model.pth``."""
    best = checkpoint_dir / "best_model.pth"
    if best.exists():
        return [best]
    last = checkpoint_dir / "last_model.pth"
    if last.exists():
        return [last]
    return []


def discover_checkpoints(
    checkpoints_dir: str | Path,
    logger: Optional[logging.Logger] = None,
) -> List[DiscoveredModel]:
    """Return every valid trained checkpoint under ``checkpoints_dir``.

    A checkpoint is considered valid if it embeds a config whose ``experiment``
    is one of ``blackbox`` / ``cbm`` / ``spatial_cbm``. Invalid or unreadable
    checkpoint directories are skipped with a warning.
    """
    checkpoints_dir = Path(checkpoints_dir)
    if not checkpoints_dir.exists():
        if logger:
            logger.warning("Checkpoint directory not found: %s", checkpoints_dir)
        return []

    found: List[DiscoveredModel] = []
    for child in sorted(checkpoints_dir.iterdir()):
        if not child.is_dir():
            continue
        candidates = _checkpoint_candidates(child)
        if not candidates:
            continue
        ckpt_path = candidates[0]
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg = Config(checkpoint.get("config") or {})
        except Exception as exc:  # pragma: no cover - defensive
            if logger:
                logger.warning("Skipping unreadable checkpoint %s: %s", ckpt_path, exc)
            continue

        experiment = str(cfg.get("experiment", "")).lower()
        if experiment not in VALID_EXPERIMENTS:
            if logger:
                logger.warning(
                    "Skipping %s: unknown experiment %r", ckpt_path, experiment
                )
            continue

        found.append(
            DiscoveredModel(
                experiment=experiment,
                run_id=child.name,
                checkpoint_path=ckpt_path,
                checkpoint_dir=child,
                config=cfg,
            )
        )

    # Deterministic ordering by (experiment, run_id).
    found.sort(key=lambda m: (m.experiment, m.run_id))
    return found


def select_one_per_experiment(
    models: List[DiscoveredModel],
) -> Dict[str, DiscoveredModel]:
    """Pick the first checkpoint for each canonical model key.

    ``blackbox_gradcam`` is not a separate checkpoint -- it is the black-box
    model viewed with Grad-CAM -- so it is intentionally left out of this
    selection and mapped to the black-box metrics in the report.
    """
    selected: Dict[str, DiscoveredModel] = {}
    for model in models:
        key = model.canonical_key
        if key not in selected:
            selected[key] = model
    return selected


def discover_report_models(
    checkpoints_dir: str | Path,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, DiscoveredModel]:
    """One-stop helper: discover then select one model per experiment."""
    return select_one_per_experiment(discover_checkpoints(checkpoints_dir, logger))
