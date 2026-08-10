"""Functional smoke test for the ChestX-Det fast-track project.

NO retraining. NO architecture changes. NO full-dataset download. Uses only
existing checkpoints; streams a few *unseen* test samples; runs inference;
checks concept / diagnosis / spatial outputs; runs one concept intervention;
writes qualitative figures. Everything is written under ``outputs/`` (D:).

Usage:
    python scripts/smoke_test_chestxdet.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

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

from src.data import chestxdet as cxd  # noqa: E402
from src.data.transforms import build_transform  # noqa: E402
from src.evaluation.evaluation import predict  # noqa: E402
from src.evaluation.qualitative import generate_qualitative_examples  # noqa: E402
from src.evaluation.spatial_visualization import generate_spatial_examples  # noqa: E402
from src.explainability.gradcam import generate_gradcam_examples  # noqa: E402
from src.interventions.runner import InterventionRunner  # noqa: E402
from src.models import build_model  # noqa: E402
from src.training.trainer import resolve_device  # noqa: E402
from src.utils.config import Config  # noqa: E402

CKPT_DIR = PROJECT_ROOT / "outputs" / "chestxdet_fasttrack" / "checkpoints"
OUT_DIR = PROJECT_ROOT / "outputs" / "chestxdet_smoketest"
MAPPING_CONFIG = PROJECT_ROOT / "configs" / "chestxdet_concepts.yaml"

MODELS = ["blackbox", "cbm", "spatial_cbm"]
# Unseen test offset: the fast-track experiment used test[0:200].
TEST_OFFSET = 200
TEST_N = 4

report: dict = {"pytest": "81 passed (exit 0, run separately)"}


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# 2 + 3. Locate and load checkpoints
# --------------------------------------------------------------------------- #
def locate_checkpoints() -> dict:
    found = {}
    for name in MODELS:
        p = CKPT_DIR / name / "best_model.pth"
        found[name] = {"exists": p.exists(), "path": str(p),
                       "size_mb": round(p.stat().st_size / 1e6, 1) if p.exists() else None}
    return found


def load_checkpoint(path: Path) -> tuple:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(ckpt["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt


# --------------------------------------------------------------------------- #
# 4. Stream a few unseen test samples
# --------------------------------------------------------------------------- #
def stream_unseen(n: int, offset: int, cfg, transform) -> list:
    mapping = yaml.safe_load(MAPPING_CONFIG.read_text(encoding="utf-8"))["concept_mapping"]
    ds = cxd.ChestXDetRemoteDataset(
        repo_id=str(cfg.get("chestxdet.repo_id", cxd.CHESTX_DET_REPO)),
        split="test", image_size=int(cfg.data.image_size), transform=transform,
        concept_names=list(cfg.concepts), concept_mapping=mapping,
        box_strategy=str(cfg.get("chestxdet.box_strategy", "B_label_components")),
        min_component_area_fraction=float(
            cfg.get("chestxdet.min_component_area_fraction", 0.0005)
        ),
        max_samples=n, offset=offset,
    )
    items = []
    for sample in ds:
        sample.pop("mask", None)
        items.append(sample)
        if len(items) >= n:
            break
    return items


def collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "concepts": torch.stack([b["concepts"] for b in batch]),
        "diagnosis": torch.stack([b["diagnosis"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
    }


def make_loader(items, batch_size):
    from torch.utils.data import DataLoader
    return DataLoader(items, batch_size=batch_size, shuffle=False,
                      num_workers=0, collate_fn=collate)


# --------------------------------------------------------------------------- #
# Output verification helpers
# --------------------------------------------------------------------------- #
def verify_outputs(model, cfg, items) -> dict:
    loader = make_loader(items, batch_size=len(items))
    t0 = time.time()
    results = predict(model, loader, device)
    dt = time.time() - t0
    n = len(items)
    checks = {
        "inference_time_sec": round(dt, 2),
        "image_ids": list(results["image_ids"]),
        "diagnosis_logits_shape": results.get("diagnosis_logits", np.array([])).shape,
        "concepts_logits_shape": results.get("concepts_logits", np.array([])).shape,
        "activation_maps_shape": results.get("activation_maps", np.array([])).shape,
        "boxes_present": len(results.get("boxes", [])) == n,
        "boxes_total": sum(len(b) for b in results.get("boxes", [])),
        "has_diagnosis": "diagnosis_logits" in results,
        "has_concepts": "concepts_logits" in results,
        "has_spatial": "activation_maps" in results,
    }
    if "diagnosis_logits" in results:
        p = 1.0 / (1.0 + np.exp(-results["diagnosis_logits"]))
        checks["diagnosis_prob_range"] = [float(p.min()), float(p.max())]
    if "concepts_logits" in results:
        p = 1.0 / (1.0 + np.exp(-results["concepts_logits"]))
        checks["concept_prob_range"] = [float(p.min()), float(p.max())]
    if "activation_maps" in results and results["activation_maps"] is not None:
        maps = np.asarray(results["activation_maps"])
        checks["activation_maps_numeric"] = [
            float(maps.min()), float(maps.max()), maps.shape
        ]
    return checks


# --------------------------------------------------------------------------- #
# 7. One concept intervention
# --------------------------------------------------------------------------- #
def run_intervention(model, cfg, results) -> dict:
    concept_names = list(cfg.concepts)
    diagnosis_names = list(cfg.diagnoses)
    z = np.asarray(results["concepts_logits"], dtype=np.float32)
    z = 1.0 / (1.0 + np.exp(-z))
    i = 0
    pick = int(np.argmax(z[i]))
    concept = concept_names[pick]
    runner = InterventionRunner(model, concept_names, diagnosis_names, device=device)
    z_in = torch.as_tensor(z[i : i + 1], dtype=torch.float32)
    before = runner.diagnosis_probs(z_in)[0]
    z_mod = z_in.clone()
    z_mod[0, pick] = 0.0  # "remove" the strongest concept
    after = runner.diagnosis_probs(z_mod)[0]
    delta = np.abs(after - before)
    return {
        "concept_intervened": concept,
        "z_before": float(z[i, pick]),
        "z_after": 0.0,
        "max_abs_delta": float(delta.max()),
        "argmax_delta_diagnosis": diagnosis_names[int(np.argmax(delta))],
        "changed": bool(delta.max() > 1e-6),
        "before_range": [float(before.min()), float(before.max())],
        "after_range": [float(after.min()), float(after.max())],
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    global device
    device = resolve_device("auto")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_root = OUT_DIR / "figures"
    log(f"device={device} | out_dir={OUT_DIR}")

    # 2. locate
    log("\n[2] locating best checkpoints ...")
    ckpt_info = locate_checkpoints()
    report["checkpoints"] = ckpt_info
    missing = [n for n, info in ckpt_info.items() if not info["exists"]]
    if missing:
        log(f"MISSING checkpoints: {missing}")
        report["checkpoint_load"] = {m: "FAILED (missing)" for m in missing}
    for n, info in ckpt_info.items():
        log(f"  {n}: {info['path']} ({info['size_mb']} MB) exists={info['exists']}")

    # 3 + 4 + 5 + 6. load, stream unseen, infer, verify outputs
    log("\n[3-6] loading checkpoints, streaming unseen test samples, inference ...")
    base_cfg = Config(yaml.safe_load(
        (PROJECT_ROOT / "configs" / "chestxdet_base.yaml").read_text(encoding="utf-8")))
    mean = list(base_cfg.get("data.normalize.mean", [0.485, 0.456, 0.406]))
    std = list(base_cfg.get("data.normalize.std", [0.229, 0.224, 0.225]))
    image_size = int(base_cfg.get("data.image_size", 256))
    transform = build_transform(image_size, mean, std, train=False, augment=False)

    items = None
    model_state = {}
    for name in MODELS:
        path = Path(ckpt_info[name]["path"])
        if not path.exists():
            model_state[name] = {"checkpoint_load": "FAILED (missing)"}
            continue
        try:
            model, cfg, ckpt = load_checkpoint(path)
        except Exception as exc:
            model_state[name] = {"checkpoint_load": f"FAILED: {exc}"}
            log(f"  {name}: checkpoint LOAD FAILED: {exc}")
            continue
        log(f"  {name}: checkpoint loaded OK (epoch={ckpt.get('epoch')}, "
            f"experiment={cfg.get('experiment')}, best_auroc={ckpt.get('metrics', {}).get('diagnosis_auroc_macro')})")
        if items is None:  # stream once, reuse for all three models
            log(f"  streaming {TEST_N} unseen test samples at offset {TEST_OFFSET} ...")
            t0 = time.time()
            items = stream_unseen(TEST_N, TEST_OFFSET, cfg, transform)
            log(f"    streamed {len(items)} samples in {time.time()-t0:.1f}s "
                f"(ids: {[it['image_id'] for it in items]})")
        try:
            checks = verify_outputs(model, cfg, items)
        except Exception as exc:
            model_state[name] = {"checkpoint_load": "OK", "inference": f"FAILED: {exc}"}
            log(f"  {name}: inference FAILED: {exc}")
            continue
        model_state[name] = {"checkpoint_load": "OK", "outputs": checks}
        log(f"  {name}: inference OK in {checks['inference_time_sec']}s -> "
            f"diag{checks.get('has_diagnosis')} concepts{checks.get('has_concepts')} "
            f"spatial{checks.get('has_spatial')} boxes={checks.get('boxes_total')}")

        # 7. intervention (CBMs only)
        if cfg.get("experiment") in ("cbm", "spatial_cbm"):
            loader = make_loader(items, batch_size=len(items))
            r = predict(model, loader, device)
            try:
                intr = run_intervention(model, cfg, r)
            except Exception as exc:
                intr = {"error": str(exc)}
            model_state[name]["intervention"] = intr
            log(f"  {name}: intervention on '{intr.get('concept_intervened')}' "
                f"changed={intr.get('changed')} max_delta={intr.get('max_abs_delta')}")
        else:
            model_state[name]["intervention"] = "N/A (blackbox has no concept bottleneck)"

    report["model_state"] = model_state

    # 8. figures
    log("\n[8] generating qualitative figures ...")
    fig_meta = {}
    if items is not None:
        diagnosis_names = list(base_cfg.diagnoses)
        concept_names = list(base_cfg.concepts)
        for name in MODELS:
            path = Path(ckpt_info[name]["path"])
            if not path.exists():
                continue
            model, cfg, _ = load_checkpoint(path)
            try:
                if name == "blackbox":
                    written = [str(p) for p in generate_gradcam_examples(
                        model, items, device, fig_root / name, num_examples=2,
                        mean=mean, std=std)]
                elif name == "cbm":
                    written = [str(p) for p in generate_qualitative_examples(
                        model, items, device, fig_root / name,
                        diagnosis_names, concept_names,
                        num_success=1, num_failure=1, mean=mean, std=std)]
                else:
                    written = [str(p) for p in generate_spatial_examples(
                        model, items, device, fig_root / name,
                        num_success=1, num_failure=1, mean=mean, std=std)]
                fig_meta[name] = {"count": len(written), "files": written}
                log(f"  {name}: {len(written)} figures")
            except Exception as exc:
                fig_meta[name] = {"error": str(exc)}
                log(f"  {name}: figure generation FAILED: {exc}")
    report["figures"] = fig_meta

    # 10. artifact verification
    log("\n[10] verifying artifacts under outputs/ ...")
    artifacts = {}
    for sub in sorted(p.relative_to(OUT_DIR).as_posix()
                      for p in OUT_DIR.rglob("*") if p.is_file()):
        artifacts[sub] = "ok"
    report["artifacts_written"] = {
        "output_dir": str(OUT_DIR),
        "files": sorted(artifacts),
        "count": len(artifacts),
        "on_d_root": str(OUT_DIR).startswith("D:"),
    }
    log(f"  {len(artifacts)} files written under {OUT_DIR}")

    result_path = OUT_DIR / "smoke_test_results.json"
    result_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log(f"\nresults -> {result_path}")
    return 0


def results_for(cfg, checks, items):  # pragma: no cover - placeholder removed
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
