"""Per-model Phase 3 analysis.

For one trained checkpoint this module:

1. runs the standard Phase 1/2 evaluation (``build_metric_fn`` + ``predict``),
2. runs the intervention experiments (oracle, robustness, recovery,
   completeness, dependency) for CBM-style models,
3. runs the A-E failure analysis,
4. generates qualitative evidence figures (Grad-CAM for the black-box,
   concept-evidence / intervention figures for the CBMs),
5. writes structured, fact-only explanations.

Every artifact is computed from actual model outputs. When a computation is
not applicable to a model (e.g. concept interventions on a black-box) it is
skipped and recorded as such -- never estimated.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import VinDrCXRDataset, collate_vindr
from src.data.split import load_split_ids
from src.data.transforms import build_transform
from src.evaluation.evaluation import build_metric_fn, predict, save_metrics_json
from src.evaluation.failure_analysis import run_failure_analysis, plot_failure_analysis
from src.evaluation.plots import (
    plot_completeness,
    plot_dependency_matrix,
    plot_recovery_bars,
    plot_robustness,
)
from src.evaluation.qualitative import generate_qualitative_examples
from src.evaluation.spatial_visualization import generate_spatial_examples
from src.explainability.gradcam import generate_gradcam_examples
from src.explainability.structured_explanation import (
    attach_intervention,
    build_structured_explanation,
    render_markdown,
)
from src.interventions.experiments import (
    concept_completeness,
    concept_error_recovery,
    concept_error_robustness,
    oracle_concept_experiment,
    summarize_oracle_experiment,
    summarize_recovery,
)
from src.interventions.intervention import build_intervention
from src.interventions.metrics import (
    InterventionMetrics,
    save_dependency_matrix,
    save_interventions_csv,
)
from src.interventions.runner import (
    InterventionRunner,
    concept_probs_from_logits,
    run_dependency_analysis,
)
from src.models import build_model
from src.reporting.discovery import DiscoveredModel
from src.utils.config import Config, resolve

CBM_LIKE = ("cbm", "spatial_cbm")

DEFAULT_CORRUPTION_FRACTIONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
DEFAULT_SUBSET_SIZES = (5, 8, 12, 22)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def slice_results(results: Dict[str, np.ndarray], n: Optional[int]) -> Dict[str, np.ndarray]:
    """Return the first ``n`` rows of every array in ``results`` (n=None: all)."""
    if n is None:
        return results
    sliced: Dict[str, np.ndarray] = {}
    for key, value in results.items():
        if key == "image_ids":
            sliced[key] = list(value[:n])
        elif key == "boxes":
            sliced[key] = list(value[:n])
        elif isinstance(value, np.ndarray):
            sliced[key] = value[:n]
        else:
            sliced[key] = value
    return sliced


def build_test_dataloader(
    cfg: Config,
    project_root: Path,
    split: str = "test",
    batch_size: Optional[int] = None,
) -> tuple:
    """Build a dataset + DataLoader for a split from the checkpoint config."""
    data_root = Path(cfg.data.root)
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {data_root}. Override with --root or fix data.root."
        )
    splits_dir = resolve(project_root, cfg.get("data.splits_dir", "data/splits"))
    splits = load_split_ids(splits_dir)
    split_ids = splits.get(split)
    if not split_ids:
        raise ValueError(
            f"Unknown split {split!r}; available: {sorted(splits)}"
        )

    image_size = int(cfg.data.image_size)
    mean = list(cfg.get("data.normalize.mean", [0.485, 0.456, 0.406]))
    std = list(cfg.get("data.normalize.std", [0.229, 0.224, 0.225]))
    transform = build_transform(image_size, mean, std, train=False)

    dataset = VinDrCXRDataset(
        data_root, split_ids, list(cfg.concepts), list(cfg.diagnoses),
        split=split, image_size=image_size, transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size or cfg.get("data.batch_size", 16)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_vindr,
    )
    return dataset, loader


def load_model_from_checkpoint(discovered: DiscoveredModel, device: torch.device):
    """Instantiate the right model for ``discovered`` and load its weights."""
    model = build_model(discovered.config).to(device)
    checkpoint = torch.load(
        discovered.checkpoint_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _safe_float(value, digits: int = 4):
    if value is None:
        return None
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    return round(fv, digits) if np.isfinite(fv) else None


# --------------------------------------------------------------------------- #
# Intervention analysis
# --------------------------------------------------------------------------- #
def _run_interventions(
    model,
    results: Dict[str, np.ndarray],
    cfg: Config,
    device: torch.device,
    out_dir: Path,
    concept_names: Sequence[str],
    diagnosis_names: Sequence[str],
    max_images: Optional[int] = 200,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """Run all concept-intervention experiments for one CBM-style model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    concept_names = list(concept_names)
    diagnosis_names = list(diagnosis_names)
    threshold = float(cfg.get("evaluation.threshold", 0.5))
    bundle: Dict = {}

    # --- oracle concepts ------------------------------------------------ #
    oracle = oracle_concept_experiment(model, results, diagnosis_names, threshold)
    bundle["oracle"] = oracle
    _write_dict(summarize_oracle_experiment(oracle), out_dir / "oracle.csv")

    # --- robustness ------------------------------------------------------ #
    robustness = concept_error_robustness(
        model, results, diagnosis_names,
        fractions=DEFAULT_CORRUPTION_FRACTIONS, threshold=threshold,
    )
    bundle["robustness"] = robustness
    robustness.to_csv(out_dir / "robustness.csv", index=False)
    plot_robustness(
        {cfg.get("experiment", "model"): robustness},
        [cfg.get("experiment", "model")],
        out_dir / "robustness.png",
        metric="auroc",
    )

    # --- recovery -------------------------------------------------------- #
    recovery = concept_error_recovery(model, results, diagnosis_names, threshold)
    bundle["recovery"] = recovery
    _write_dict(summarize_recovery(recovery), out_dir / "recovery.csv")
    plot_recovery_bars(recovery, out_dir / "recovery.png")

    # --- completeness ---------------------------------------------------- #
    completeness = concept_completeness(
        model, results, concept_names, diagnosis_names,
        sizes=DEFAULT_SUBSET_SIZES, threshold=threshold,
    )
    bundle["completeness"] = completeness
    completeness["summary"].to_csv(out_dir / "completeness.csv", index=False)
    plot_completeness(completeness["summary"], out_dir / "completeness.png")

    # --- dependency matrix ----------------------------------------------- #
    z_subset = concept_probs_from_logits(model, results["concepts_logits"])
    z_mean = torch.as_tensor(z_subset[:max_images].mean(axis=0, keepdims=True))
    dependency = run_dependency_analysis(
        model, z_mean, concept_names, diagnosis_names, mode="remove", device=device,
    )
    bundle["dependency"] = dependency
    save_dependency_matrix(
        dependency["abs_change"], concept_names, diagnosis_names,
        out_dir / "dependency_matrix.csv", metric="abs_change",
    )
    plot_dependency_matrix(
        dependency["abs_change"], concept_names, diagnosis_names,
        out_dir / "dependency_matrix.png",
    )

    # --- per-image intervention records ---------------------------------- #
    records = _collect_intervention_records(
        model, results, cfg, device, concept_names, diagnosis_names,
        max_images=max_images, logger=logger,
    )
    bundle["records"] = records
    if records:
        save_interventions_csv(records, out_dir / "interventions.csv")

    return bundle


def _collect_intervention_records(
    model,
    results: Dict[str, np.ndarray],
    cfg: Config,
    device: torch.device,
    concept_names: Sequence[str],
    diagnosis_names: Sequence[str],
    max_images: Optional[int],
    logger: Optional[logging.Logger] = None,
) -> List[Dict]:
    """Per-image 'remove' interventions on every concept (records for CSV)."""
    concept_names = list(concept_names)
    diagnosis_names = list(diagnosis_names)
    subset = slice_results(results, max_images)
    z = concept_probs_from_logits(model, subset["concepts_logits"])
    image_ids = subset["image_ids"]

    records: List[Dict] = []
    try:
        runner = InterventionRunner(model, concept_names, diagnosis_names, device=device)
    except TypeError as exc:
        if logger:
            logger.warning("Skipping intervention records: %s", exc)
        return records

    for i, image_id in enumerate(image_ids):
        z_i = torch.as_tensor(z[i], dtype=torch.float32).unsqueeze(0)
        for k, name in enumerate(concept_names):
            intervention = runner.intervene(
                z_i, build_intervention("remove", name)
            )
            records.extend(
                InterventionMetrics.rows_for_intervention(
                    image_id=image_id,
                    concept=name,
                    mode="remove",
                    value=0.0,
                    delta=0.0,
                    diagnosis_names=diagnosis_names,
                    probs_before=intervention["diagnosis_probs_before"],
                    probs_after=intervention["diagnosis_probs_after"],
                )
            )
    return records


def _write_dict(row: Dict, path: Path) -> Path:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return path


# --------------------------------------------------------------------------- #
# Qualitative evidence + structured explanations
# --------------------------------------------------------------------------- #
def _run_qualitative(
    model,
    dataset,
    device: torch.device,
    cfg: Config,
    out_dir: Path,
    diagnosis_names: Sequence[str],
    concept_names: Sequence[str],
    n_figures: int = 6,
    logger: Optional[logging.Logger] = None,
) -> List[Path]:
    """Generate evidence figures appropriate for the model type."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mean = list(cfg.get("data.normalize.mean", [0.485, 0.456, 0.406]))
    std = list(cfg.get("data.normalize.std", [0.229, 0.224, 0.225]))
    experiment = str(cfg.get("experiment", "blackbox"))
    written: List[Path] = []

    if experiment == "blackbox":
        try:
            written += [
                Path(p)
                for p in generate_gradcam_examples(
                    model, dataset, device, out_dir=out_dir,
                    num_examples=max(1, n_figures), mean=mean, std=std,
                )
            ]
        except Exception as exc:  # pragma: no cover - defensive
            if logger:
                logger.warning("Grad-CAM examples failed: %s", exc)
    else:
        half = max(1, n_figures // 2)
        if experiment == "spatial_cbm":
            try:
                written += [
                    Path(p)
                    for p in generate_spatial_examples(
                        model, dataset, device, out_dir=out_dir,
                        num_success=half, num_failure=n_figures - half,
                        mean=mean, std=std,
                    )
                ]
            except Exception as exc:  # pragma: no cover - defensive
                if logger:
                    logger.warning("Spatial examples failed: %s", exc)
        try:
            written += generate_qualitative_examples(
                model, dataset, device, out_dir=out_dir,
                diagnosis_names=diagnosis_names, concept_names=concept_names,
                num_success=half, num_failure=n_figures - half,
                mean=mean, std=std,
            )
        except Exception as exc:  # pragma: no cover - defensive
            if logger:
                logger.warning("Qualitative examples failed: %s", exc)
    return written


def _run_structured_explanations(
    model,
    dataset,
    results: Dict[str, np.ndarray],
    device: torch.device,
    cfg: Config,
    out_dir: Path,
    diagnosis_names: Sequence[str],
    concept_names: Sequence[str],
    n_explanations: int = 3,
    logger: Optional[logging.Logger] = None,
) -> List[Path]:
    """Write fact-only markdown explanations for a few test images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    concept_names = list(concept_names)
    diagnosis_names = list(diagnosis_names)
    threshold = float(cfg.get("evaluation.threshold", 0.5))
    cam_threshold = float(cfg.get("evaluation.cam_threshold", 0.5))

    has_spatial = "activation_maps" in results
    maps = results.get("activation_maps")
    boxes = results.get("boxes")
    image_ids = results.get("image_ids", [])

    written: List[Path] = []
    n = min(n_explanations, len(image_ids))
    for i in range(n):
        activation_map = (
            np.asarray(maps[i]) if has_spatial and maps is not None else None
        )
        boxes_i = boxes[i] if boxes is not None and i < len(boxes) else None
        explanation = build_structured_explanation(
            image_id=image_ids[i],
            diagnosis_logits=results["diagnosis_logits"][i],
            concept_logits=results["concepts_logits"][i],
            diagnosis_names=diagnosis_names,
            concept_names=concept_names,
            activation_map=activation_map,
            boxes=boxes_i,
            cam_threshold=cam_threshold,
        )

        # Attach a real 'remove' intervention on the top detected concept.
        detected = explanation.detected_concepts()
        if detected:
            concept = detected[0]
            try:
                runner = InterventionRunner(
                    model, concept_names, diagnosis_names, device=device
                )
                z_i = concept_probs_from_logits(
                    model, results["concepts_logits"][i : i + 1]
                )
                z_t = torch.as_tensor(z_i, dtype=torch.float32)
                outcome = runner.intervene(z_t, build_intervention("remove", concept))
                explanation = attach_intervention(
                    explanation, concept, "remove",
                    probs_before=outcome["diagnosis_probs_before"],
                    probs_after=outcome["diagnosis_probs_after"],
                    diagnosis_names=diagnosis_names,
                    value=0.0,
                )
            except (KeyError, TypeError, ValueError) as exc:
                if logger:
                    logger.warning(
                        "Could not attach intervention for %s: %s", image_ids[i], exc
                    )

        path = out_dir / f"{image_ids[i]}.md"
        path.write_text(render_markdown(explanation) + "\n", encoding="utf-8")
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
# Top-level per-model analysis
# --------------------------------------------------------------------------- #
def run_model_analysis(
    discovered: DiscoveredModel,
    device: torch.device,
    project_root: Path,
    out_dir: str | Path,
    split: str = "test",
    batch_size: Optional[int] = None,
    max_images: Optional[int] = 200,
    n_qualitative: int = 6,
    n_explanations: int = 3,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """Run every applicable Phase 3 analysis for one checkpoint.

    Returns a bundle dict describing what was computed and where it was
    written. Analyses that are not applicable to the model type are skipped
    (never fabricated).
    """
    out_dir = Path(out_dir)
    run_out = out_dir / discovered.run_id
    run_out.mkdir(parents=True, exist_ok=True)
    cfg = discovered.config
    experiment = discovered.experiment
    concept_names = list(cfg.concepts)
    diagnosis_names = list(cfg.diagnoses)

    bundle: Dict = {
        "experiment": experiment,
        "run_id": discovered.run_id,
        "model": discovered,
        "metrics": None,
        "results": None,
        "failure_df": None,
        "interventions": None,
        "outputs": {"dir": run_out, "figures": []},
        "errors": [],
    }
    log = logger or logging.getLogger("reporting")

    try:
        dataset, loader = build_test_dataloader(cfg, project_root, split, batch_size)
    except Exception as exc:
        bundle["errors"].append(f"data loading failed: {exc}")
        log.error("data loading failed for %s: %s", discovered.run_id, exc)
        return bundle

    try:
        model = load_model_from_checkpoint(discovered, device)
    except Exception as exc:
        bundle["errors"].append(f"model loading failed: {exc}")
        log.error("model loading failed for %s: %s", discovered.run_id, exc)
        return bundle

    # 1. Standard evaluation -------------------------------------------------
    results = predict(model, loader, device)
    metric_fn = build_metric_fn(experiment, cfg)
    metrics = metric_fn(results)
    bundle["metrics"] = metrics
    bundle["results"] = results
    metrics_path = run_out / "metrics.json"
    save_metrics_json(
        {
            "experiment": experiment,
            "run_id": discovered.run_id,
            "split": split,
            "num_samples": len(results.get("image_ids", [])),
            **metrics,
        },
        metrics_path,
    )
    bundle["outputs"]["metrics_json"] = metrics_path
    log.info("[%s] metrics computed (diag AUROC macro=%s)", discovered.run_id,
             _safe_float(metrics.get("diagnosis_auroc_macro")))

    # 2. Interventions (CBM-like models only) --------------------------------
    if experiment in CBM_LIKE and "concepts_logits" in results:
        int_dir = run_out / "interventions"
        try:
            interventions = _run_interventions(
                model, results, cfg, device, int_dir,
                concept_names, diagnosis_names,
                max_images=max_images, logger=log,
            )
            bundle["interventions"] = interventions
            log.info("[%s] interventions written -> %s", discovered.run_id, int_dir)
        except Exception as exc:
            bundle["errors"].append(f"interventions failed: {exc}")
            log.error("interventions failed for %s: %s", discovered.run_id, exc)

    # 3. Failure analysis (CBM-like models only) -----------------------------
    if experiment in CBM_LIKE and "concepts_logits" in results:
        try:
            df, counts = run_failure_analysis(
                results, diagnosis_names, concept_names,
                threshold=float(cfg.get("evaluation.threshold", 0.5)),
                els_threshold=float(cfg.get("evaluation.cam_threshold", 0.5)),
            )
            bundle["failure_df"] = df
            df.to_csv(run_out / "failure_analysis.csv", index=False)
            plot_failure_analysis(counts, run_out / "failure_analysis.png")
            log.info("[%s] failure analysis written", discovered.run_id)
        except Exception as exc:
            bundle["errors"].append(f"failure analysis failed: {exc}")
            log.error("failure analysis failed for %s: %s", discovered.run_id, exc)

    # 4. Qualitative figures ---------------------------------------------------
    try:
        fig_dir = run_out / "figures"
        figures = _run_qualitative(
            model, dataset, device, cfg, fig_dir,
            diagnosis_names, concept_names,
            n_figures=n_qualitative, logger=log,
        )
        bundle["outputs"]["figures"] = figures
        log.info("[%s] %d qualitative figures written", discovered.run_id, len(figures))
    except Exception as exc:
        bundle["errors"].append(f"qualitative figures failed: {exc}")
        log.error("qualitative figures failed for %s: %s", discovered.run_id, exc)

    # 5. Structured explanations (CBM-like models only) -----------------------
    if experiment in CBM_LIKE and "concepts_logits" in results:
        try:
            exp_dir = run_out / "explanations"
            written = _run_structured_explanations(
                model, dataset, results, device, cfg, exp_dir,
                diagnosis_names, concept_names,
                n_explanations=n_explanations, logger=log,
            )
            bundle["outputs"]["explanations"] = written
            log.info("[%s] %d explanations written", discovered.run_id, len(written))
        except Exception as exc:
            bundle["errors"].append(f"structured explanations failed: {exc}")
            log.error("structured explanations failed for %s: %s", discovered.run_id, exc)

    return bundle
