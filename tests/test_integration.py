"""End-to-end integration test: run_experiment on a tiny synthetic dataset."""
import json

from src.models.blackbox import build_blackbox
from src.models.cbm import build_cbm
from src.training.trainer import run_experiment
from src.utils.config import Config
from src.utils.logging import setup_logger

from conftest import CONCEPT_NAMES, DIAGNOSIS_NAMES


def _cfg(synthetic_vindr, tmp_path, experiment="blackbox"):
    return Config(
        {
            "seed": 42,
            "device": "cpu",
            "experiment": experiment,
            "data": {
                "root": str(synthetic_vindr),
                "image_size": 64,
                "batch_size": 2,
                "num_workers": 0,
                "val_fraction": 0.25,
                "split_seed": 42,
                "splits_dir": str(tmp_path / "splits"),
                "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                "augment": False,
            },
            "concepts": CONCEPT_NAMES,
            "diagnoses": DIAGNOSIS_NAMES,
            "model": {
                "encoder": "densenet121",
                "pretrained": False,
                "concept_activation": "sigmoid",
            },
            "training": {
                "epochs": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "patience": 1,
                "mixed_precision": False,
                "early_stopping": True,
                "scheduler": "reduce_on_plateau",
                "lambda_concept": 1.0,
                "lambda_diagnosis": 1.0,
                "monitor_metric": "diagnosis_auroc_macro",
            },
            "evaluation": {"threshold": 0.5},
            "outputs": {
                "checkpoint_dir": str(tmp_path / "ckpt"),
                "metrics_dir": str(tmp_path / "metrics"),
                "predictions_dir": str(tmp_path / "preds"),
                "figures_dir": str(tmp_path / "figs"),
                "log_dir": str(tmp_path / "logs"),
            },
        }
    )


def test_run_experiment_blackbox(synthetic_vindr, tmp_path):
    cfg = _cfg(synthetic_vindr, tmp_path, "blackbox")
    logger = setup_logger("test_int_blackbox", log_dir=tmp_path / "logs", log_file="bb.log")
    summary = run_experiment(cfg, build_blackbox, logger, run_id="smoke_bb")

    assert (tmp_path / "ckpt" / "smoke_bb" / "best_model.pth").exists()
    assert (tmp_path / "ckpt" / "smoke_bb" / "last_model.pth").exists()
    assert (tmp_path / "ckpt" / "smoke_bb" / "training_history.csv").exists()

    metrics = json.loads((tmp_path / "metrics" / "smoke_bb" / "metrics.json").read_text())
    assert metrics["experiment"] == "blackbox"
    assert "diagnosis" in metrics["test"]
    assert "concepts" not in metrics["test"]

    preds = (tmp_path / "preds" / "smoke_bb" / "predictions_test.csv").read_text()
    assert "image_id" in preds
    assert "cardiomegaly_prob" in preds
    assert summary["checkpoint_dir"]


def test_run_experiment_cbm(synthetic_vindr, tmp_path):
    cfg = _cfg(synthetic_vindr, tmp_path, "cbm")
    logger = setup_logger("test_int_cbm", log_dir=tmp_path / "logs", log_file="cbm.log")
    summary = run_experiment(cfg, build_cbm, logger, run_id="smoke_cbm")

    assert (tmp_path / "ckpt" / "smoke_cbm" / "best_model.pth").exists()
    metrics = json.loads((tmp_path / "metrics" / "smoke_cbm" / "metrics.json").read_text())
    assert metrics["experiment"] == "cbm"
    assert "concepts" in metrics["test"]
    assert "concept_auroc_macro" in metrics["test"]
    assert "diagnosis" in metrics["test"]
    assert summary["checkpoint_dir"]
