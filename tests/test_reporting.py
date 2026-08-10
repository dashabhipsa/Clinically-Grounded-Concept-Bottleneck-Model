"""Tests for the Phase 3 reporting pipeline.

These tests verify the *pipeline* (discovery, analysis, report) using a tiny
random-weight model on the synthetic dataset. They deliberately assert that
missing checkpoints produce N/A (never fabricated numbers), and that a real
checkpoint produces real artifacts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data.split import create_splits
from src.models.spatial_cbm import build_spatial_cbm
from src.reporting.analysis import run_model_analysis
from src.reporting.discovery import (
    discover_checkpoints,
    discover_report_models,
    select_one_per_experiment,
)
from src.reporting.report import (
    assemble_metrics_by_model,
    render_master_report,
)
from src.utils.config import Config

from conftest import CONCEPT_NAMES, DIAGNOSIS_NAMES


def _config(root: Path, splits_dir: Path, experiment: str = "spatial_cbm"):
    return Config(
        {
            "seed": 42,
            "experiment": experiment,
            "data": {
                "root": str(root),
                "image_size": 64,
                "batch_size": 2,
                "splits_dir": str(splits_dir),
                "normalize": {"mean": [0.485, 0.456, 0.406],
                              "std": [0.229, 0.224, 0.225]},
            },
            "model": {
                "encoder": "densenet121",
                "pretrained": False,
                "concept_activation": "sigmoid",
                "concept_pool": "avg",
            },
            "evaluation": {"threshold": 0.5, "cam_threshold": 0.5},
            "concepts": CONCEPT_NAMES,
            "diagnoses": DIAGNOSIS_NAMES,
        }
    )


def _write_checkpoint(cfg: Config, model, ckpt_path: Path) -> Path:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": cfg.to_dict()},
        ckpt_path,
    )
    return ckpt_path


# --------------------------------------------------------------------------- #
def test_discovery_and_selection(tmp_path):
    ckpts = tmp_path / "checkpoints"
    cfg = _config(tmp_path, tmp_path / "splits")

    model = build_spatial_cbm(cfg)
    valid = _write_checkpoint(cfg, model, ckpts / "spatial_a" / "best_model.pth")

    # A garbage directory must be skipped, not crash.
    (ckpts / "not_a_model").mkdir(parents=True)
    (ckpts / "not_a_model" / "best_model.pth").write_bytes(b"not a torch file")

    found = discover_checkpoints(ckpts)
    assert len(found) == 1
    assert found[0].checkpoint_path == valid
    assert found[0].experiment == "spatial_cbm"

    selected = discover_report_models(ckpts)
    assert set(selected) == {"spatial_cbm"}
    assert selected["spatial_cbm"].run_id == "spatial_a"


def test_select_one_per_experiment_keeps_first():
    fake_cfgs = Config({})
    models = []
    for run, exp in (("a", "blackbox"), ("b", "blackbox"), ("c", "cbm")):
        models.append(
            type(
                "DM", (),
                {
                    "experiment": exp,
                    "canonical_key": exp,
                    "run_id": run,
                    "checkpoint_path": Path(f"{run}.pth"),
                    "checkpoint_dir": Path(run),
                    "config": fake_cfgs,
                },
            )()
        )
    selected = select_one_per_experiment(models)
    assert selected["blackbox"].run_id == "a"
    assert selected["cbm"].run_id == "c"


# --------------------------------------------------------------------------- #
def test_report_pipeline_spatial(synthetic_vindr, tmp_path):
    splits_dir = tmp_path / "splits"
    create_splits(synthetic_vindr, splits_dir, val_fraction=0.2, seed=42)

    ckpts = tmp_path / "checkpoints"
    cfg = _config(synthetic_vindr, splits_dir, experiment="spatial_cbm")
    model = build_spatial_cbm(cfg)
    ckpt_path = _write_checkpoint(cfg, model, ckpts / "spatial_test" / "best_model.pth")

    out_dir = tmp_path / "report"
    discovered = discover_report_models(ckpts)
    assert "spatial_cbm" in discovered

    bundle = run_model_analysis(
        discovered["spatial_cbm"],
        device=torch.device("cpu"),
        project_root=tmp_path,
        out_dir=out_dir,
        split="test",
        max_images=10,
        n_qualitative=2,
        n_explanations=2,
    )
    assert bundle["metrics"] is not None, bundle.get("errors")
    assert not bundle["errors"], bundle["errors"]

    run_out = out_dir / "spatial_test"
    assert (run_out / "metrics.json").exists()
    assert (run_out / "interventions").is_dir()
    assert (run_out / "failure_analysis.csv").exists()
    assert (run_out / "failure_analysis.png").exists()
    assert (run_out / "figures").is_dir()
    assert (run_out / "explanations").is_dir()

    # Interventions must contain all experiment artifacts.
    int_dir = run_out / "interventions"
    for name in ("oracle.csv", "robustness.csv", "robustness.png",
                 "recovery.csv", "completeness.csv", "dependency_matrix.csv",
                 "dependency_matrix.png", "interventions.csv"):
        assert (int_dir / name).exists(), name

    # Failure analysis CSV has one row per test image.
    import pandas as pd

    fa = pd.read_csv(run_out / "failure_analysis.csv")
    assert len(fa) == 3  # synthetic test split (0007, 0008, 0009)

    # Report assembly: real model present, missing models are N/A.
    metrics_by_model = assemble_metrics_by_model([bundle])
    assert "spatial_cbm" in metrics_by_model
    assert "blackbox" not in metrics_by_model

    report_path = render_master_report([bundle], discovered, out_dir)
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "## 1. Experiment status" in text
    assert "spatial_cbm" in text
    assert "pending" in text  # blackbox/cbm are pending
    assert (out_dir / "final_comparison.csv").exists()
    assert (out_dir / "final_comparison.md").exists()

    comparison = (out_dir / "final_comparison.md").read_text(encoding="utf-8")
    assert "| spatial_cbm |" in comparison
    assert "| blackbox | N/A |" in comparison


def test_report_no_checkpoints_marks_everything_na(tmp_path):
    empty = tmp_path / "empty_ckpts"
    empty.mkdir(parents=True)
    out_dir = tmp_path / "report"

    discovered = discover_report_models(empty)
    assert discovered == {}

    report_path = render_master_report([], discovered, out_dir)
    text = report_path.read_text(encoding="utf-8")
    assert "blackbox | no | - | pending" in text
    assert (out_dir / "final_comparison.md").exists()
    comparison = (out_dir / "final_comparison.md").read_text(encoding="utf-8")
    for key in ("blackbox", "blackbox_gradcam", "cbm", "spatial_cbm"):
        assert f"| {key} | N/A |" in comparison
