"""Streamlit dashboard for the clinically grounded CBM project.

Two top-level views, matching the medical-AI reference design:

* **Doctor View** (default) -- upload a chest X-ray, click *Analyze X-ray*,
  and see clinical findings, concept-specific evidence maps, the diagnosis
  from the concept bottleneck, and a deterministic, model-derived
  explanation. The best available Spatial CBM checkpoint is selected
  automatically; no technical settings are exposed.
* **Researcher View** -- the full technical dashboard: configuration
  (model/checkpoint/dataset/split/batch/max images), Overview metric cards,
  Image Explorer, Failure Analysis, Intervention Experiments and stored
  Reports. Every number shown is an actual model output -- the dashboard
  never invents one.

Run with:

    pip install -r dashboard/requirements.txt
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Make the local ``ui`` helper module importable.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# D:-drive-only caches (same guard as the ChestX-Det fast-track run). MUST be
# set before torch / datasets / huggingface-hub are imported so no cache or
# download lands on C:.
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

# Load pyarrow's native dataset DLLs before torch/matplotlib allocate, so the
# ``pyarrow._dataset`` native module never has to bind lazily at runtime.
import pyarrow  # noqa: E402
import pyarrow.dataset  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import ui  # noqa: E402

from src.data.dataset import VinDrCXRDataset, collate_vindr  # noqa: E402
from src.data.spatial import normalized_box_to_pixels  # noqa: E402
from src.data.split import load_split_ids  # noqa: E402
from src.data.transforms import build_transform, unnormalize  # noqa: E402
from src.evaluation.classification_metrics import sigmoid  # noqa: E402
from src.explainability.structured_explanation import (  # noqa: E402
    build_structured_explanation,
    render_markdown,
)
from src.interventions.intervention import build_intervention  # noqa: E402
from src.interventions.runner import InterventionRunner  # noqa: E402
from src.models import build_model  # noqa: E402
from src.reporting.discovery import discover_checkpoints  # noqa: E402
from src.training.trainer import resolve_device  # noqa: E402
from src.utils.config import Config  # noqa: E402

st.set_page_config(page_title="Clinically Grounded CBM", layout="wide")
ui.inject_css()

CHESTX_DET_REPO = "natealberti/ChestX-Det"
MAPPING_CONFIG = PROJECT_ROOT / "configs" / "chestxdet_concepts.yaml"
DEFAULT_DATA_ROOT = CHESTX_DET_REPO
DEFAULT_CKPT_DIR = str(
    PROJECT_ROOT / "outputs" / "chestxdet_fasttrack" / "checkpoints"
)

# Clinical concepts available in ChestX-Det (lung_opacity and edema are NOT
# available there and are intentionally excluded from the Doctor View).
DOCTOR_CONCEPTS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "pleural_effusion",
    "nodule_mass",
    "pneumothorax",
]
CONCEPT_DISPLAY = {
    "atelectasis": "Atelectasis",
    "cardiomegaly": "Cardiomegaly",
    "consolidation": "Consolidation",
    "pleural_effusion": "Pleural Effusion",
    "nodule_mass": "Nodule / Mass",
    "pneumothorax": "Pneumothorax",
}
CONCEPT_ICON = {
    "atelectasis": "🫁",
    "cardiomegaly": "❤️",
    "consolidation": "🌫️",
    "pleural_effusion": "💧",
    "nodule_mass": "⚪",
    "pneumothorax": "🎈",
}
# Descriptive-only text for the "About this Concept" box. These are neutral
# educational statements, not diagnoses.
CONCEPT_ABOUT = {
    "atelectasis": "Atelectasis refers to partial collapse or incomplete "
        "expansion of lung tissue. On a chest X-ray it typically appears as an "
        "area of increased opacity, often with associated volume loss.",
    "cardiomegaly": "Cardiomegaly describes an enlarged cardiac silhouette on "
        "a chest X-ray, commonly assessed through the cardiothoracic ratio.",
    "consolidation": "Consolidation is a region of lung filled with fluid or "
        "cells instead of air, appearing as a homogeneous opacity that may "
        "obscure underlying vessels.",
    "pleural_effusion": "A pleural effusion is fluid within the pleural space, "
        "usually seen as blunting of the costophrenic angle or a basal opacity.",
    "nodule_mass": "A nodule or mass is a focal opacity within the lung. Size, "
        "margins and characteristics determine its clinical significance.",
    "pneumothorax": "Pneumothorax is the presence of air in the pleural space, "
        "seen as a visceral pleural line with absent lung markings beyond it.",
}
DOCTOR_THRESHOLD = 0.5
DOCTOR_DISCLAIMER = "Research prototype — Not for clinical diagnosis or treatment decisions."
UPLOAD_TYPES = ["png", "jpg", "jpeg", "dcm"]
# No patient context is collected in the Doctor View; used by the explanation
# and report builders so they still produce valid output.
EMPTY_PATIENT = {"age": None, "sex": "", "indication": ""}
# Fixed display order for the Clinical Concepts table (reference UI order).
CONCEPT_ROW_ORDER = [
    "cardiomegaly",
    "pleural_effusion",
    "consolidation",
    "atelectasis",
    "pneumothorax",
    "nodule_mass",
]


# --------------------------------------------------------------------------- #
# Upload decoding
# --------------------------------------------------------------------------- #
def decode_uploaded(uploaded) -> "Image.Image":
    """Decode an uploaded file (PNG/JPG/JPEG or DICOM) to a grayscale image.

    DICOM files are normalized with the project's own
    ``src.data.preprocessing`` pipeline (window/invert), matching VinDr-CXR.
    Temporary DICOM files are written under the D:-drive cache and removed.
    """
    from PIL import Image

    from src.data.chestxdet import image_to_grayscale

    name = (uploaded.name or "").lower()
    is_dicom = name.endswith(".dcm") or (
        (uploaded.type or "").startswith("application/dicom")
    )
    if is_dicom:
        from src.data.preprocessing import dicom_to_image

        tmp_dir = _CACHE_ROOT / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"upload_{uuid4().hex}.dcm"
        tmp_path.write_bytes(uploaded.getvalue())
        try:
            return dicom_to_image(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
    img = Image.open(io.BytesIO(uploaded.getvalue()))
    return image_to_grayscale(img)


# --------------------------------------------------------------------------- #
# Cached heavy objects
# --------------------------------------------------------------------------- #
def _is_chestxdet(cfg) -> bool:
    """True when a checkpoint was trained on the remote ChestX-Det adapter."""
    return bool(cfg.get("chestxdet.repo_id"))


class _ListDataset:
    """Random-access wrapper over a materialized list of decoded samples."""

    def __init__(self, items):
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def _chestxdet_split_params(cfg, split: str):
    """Map dashboard split names onto the fast-track streaming subsets.

    Returns ``(remote_split, offset, n)``. The fast-track run carved the
    validation set out of the remote ``train`` split (offset 500, 100 rows).
    """
    n_train = int(cfg.get("chestxdet.n_train", 500))
    n_val = int(cfg.get("chestxdet.n_val", 100))
    n_test = int(cfg.get("chestxdet.n_test", 200))
    if split == "val":
        return "train", n_train, n_val
    if split == "test":
        return "test", 0, n_test
    return "train", 0, n_train


def _load_concept_mapping() -> dict:
    """Project concept -> ChestX-Det class names from the validation config."""
    import yaml

    if not MAPPING_CONFIG.exists():
        return {}
    with open(MAPPING_CONFIG, "r", encoding="utf-8") as fh:
        return (yaml.safe_load(fh) or {}).get("concept_mapping", {})


def _materialize_chestxdet(cfg, repo_id: str, split: str, transform, max_images=None):
    """Stream the requested ChestX-Det subset into an in-memory list dataset.

    Mirrors ``scripts/train_chestxdet.py``'s materialize_split: only the
    fast-track subset (500/100/200 rows) is streamed, nothing else is
    downloaded, and no training is performed.
    """
    from src.data import chestxdet as cxd

    remote_split, offset, n = _chestxdet_split_params(cfg, split)
    if max_images and int(max_images) > 0:
        n = min(n, int(max_images))

    ds = cxd.ChestXDetRemoteDataset(
        repo_id=str(repo_id) or str(cfg.get("chestxdet.repo_id")),
        split=remote_split,
        image_size=int(cfg.get("data.image_size", 256)),
        transform=transform,
        concept_names=list(cfg.concepts),
        concept_mapping=_load_concept_mapping(),
        box_strategy=str(cfg.get("chestxdet.box_strategy", "B_label_components")),
        min_component_area_fraction=float(
            cfg.get("chestxdet.min_component_area_fraction", 0.0005)
        ),
        max_samples=n,
        offset=offset,
    )
    items: list = []
    for sample in ds:
        items.append(sample)
        if len(items) >= n:
            break
    if not items:
        raise RuntimeError(f"no ChestX-Det samples streamed for split {split!r}")
    return _ListDataset(items)


@st.cache_resource(show_spinner="Loading model and running predictions...")
def load_model_bundle(ckpt_path: str, data_root: str, split: str,
                      batch_size: int, max_images: int):
    """Load a checkpoint, build its dataset and run predictions (once)."""
    ckpt_path = Path(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = Config(checkpoint["config"])

    device = resolve_device(cfg.get("device", "auto"))
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_size = int(cfg.get("data.image_size", 256))
    mean = list(cfg.get("data.normalize.mean", [0.485, 0.456, 0.406]))
    std = list(cfg.get("data.normalize.std", [0.229, 0.224, 0.225]))
    transform = build_transform(image_size, mean, std, train=False)

    if _is_chestxdet(cfg):
        from src.data import chestxdet as cxd

        dataset = _materialize_chestxdet(
            cfg, data_root, split, transform, max_images
        )
        collate_fn = cxd.chestx_det_collate
    else:
        cfg.set("data.root", data_root)

        from src.utils.config import resolve as resolve_cfg

        splits = load_split_ids(resolve_cfg(PROJECT_ROOT, cfg.data.splits_dir))
        split_ids = splits.get(split)
        if not split_ids:
            raise ValueError(f"unknown split {split!r}")

        if max_images and max_images > 0:
            split_ids = split_ids[: int(max_images)]

        dataset = VinDrCXRDataset(
            data_root, split_ids, list(cfg.concepts), list(cfg.diagnoses),
            split=split, image_size=image_size, transform=transform,
        )
        collate_fn = collate_vindr

    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )
    results = predict_all(model, loader, device)
    return {
        "cfg": cfg,
        "model": model,
        "device": device,
        "dataset": dataset,
        "results": results,
        "mean": mean,
        "std": std,
        "image_size": image_size,
    }


def predict_all(model, loader, device) -> dict:
    """Local predict that also keeps the raw forward output per image."""
    from src.evaluation.evaluation import predict as repo_predict

    return repo_predict(model, loader, device)


# --------------------------------------------------------------------------- #
# Doctor View helpers
# --------------------------------------------------------------------------- #
def _select_best_spatial_cbm(checkpoints_dir=DEFAULT_CKPT_DIR):
    """Auto-select the best compatible Spatial CBM checkpoint on disk.

    Only Spatial CBM checkpoints qualify (they produce concept probabilities
    AND concept-specific activation maps). Among multiple candidates the one
    with the best stored ``diagnosis_auroc_macro`` validation metric wins;
    ties are broken deterministically by run id.
    """
    models = [
        m for m in discover_checkpoints(checkpoints_dir)
        if m.experiment == "spatial_cbm"
    ]
    if not models:
        raise RuntimeError(
            "No Spatial CBM checkpoint found. Train one with "
            "`scripts/train_spatial_cbm.py` first."
        )

    def metric(m):
        best = float("-inf")
        for cand in (
            m.checkpoint_dir.parent.parent / "metrics" / m.run_id / "metrics.json",
            m.checkpoint_dir.parent.parent / "metrics" / m.experiment / "metrics.json",
        ):
            if cand.exists():
                try:
                    data = json.loads(cand.read_text(encoding="utf-8"))
                    best = max(best, float(data.get("diagnosis_auroc_macro", best)))
                except (ValueError, OSError):
                    continue
        return best

    return max(models, key=lambda m: (metric(m), m.run_id))


@st.cache_resource(show_spinner="Loading the Spatial CBM model...")
def load_doctor_bundle():
    """Load the auto-selected best Spatial CBM checkpoint (once)."""
    info = _select_best_spatial_cbm()
    checkpoint = torch.load(
        str(info.checkpoint_path), map_location="cpu", weights_only=False
    )
    cfg = Config(checkpoint["config"])
    device = resolve_device(cfg.get("device", "auto"))
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_size = int(cfg.get("data.image_size", 256))
    mean = list(cfg.get("data.normalize.mean", [0.485, 0.456, 0.406]))
    std = list(cfg.get("data.normalize.std", [0.229, 0.224, 0.225]))
    transform = build_transform(image_size, mean, std, train=False)
    return {
        "model": model,
        "cfg": cfg,
        "device": device,
        "transform": transform,
        "mean": mean,
        "std": std,
        "image_size": image_size,
        "checkpoint": info,
    }


def run_doctor_inference(image, bundle: dict) -> dict:
    """Run the Spatial CBM on one X-ray image and return fact-only results.

    Preprocessing mirrors ``ChestXDetRemoteDataset.decode_row``: grayscale,
    resize to the training image size, normalize with the training statistics.
    """
    from PIL import Image

    from src.data.chestxdet import image_to_grayscale

    img = image_to_grayscale(image)
    image_size = int(bundle["image_size"])
    display = np.asarray(
        img.resize((image_size, image_size), Image.BILINEAR)
    ).astype(np.uint8)

    x = bundle["transform"](img).unsqueeze(0).to(bundle["device"])
    with torch.no_grad():
        out = bundle["model"](x)

    cfg = bundle["cfg"]
    concept_names = list(cfg.concepts)
    probs_c = torch.sigmoid(out.concept_logits[0]).detach().cpu().numpy()
    probs_d = torch.sigmoid(out.diagnosis_logits[0]).detach().cpu().numpy()
    maps = out.activation_maps[0].detach().cpu().numpy()

    index = {name: i for i, name in enumerate(concept_names)}
    findings = []
    for concept in DOCTOR_CONCEPTS:
        k = index.get(concept)
        if k is None:
            continue
        prob = float(probs_c[k])
        findings.append({
            "Concept": CONCEPT_DISPLAY[concept],
            "Probability": prob,
            "Evidence": "Detected (map below)" if prob >= DOCTOR_THRESHOLD else "—",
            "_name": concept,
            "_k": k,
        })
    findings.sort(key=lambda r: r["Probability"], reverse=True)

    diagnosis = []
    for concept in DOCTOR_CONCEPTS:
        k = index.get(concept)
        if k is None:
            continue
        diagnosis.append({
            "Diagnosis": CONCEPT_DISPLAY[concept],
            "Probability": float(probs_d[k]),
        })
    diagnosis.sort(key=lambda r: r["Probability"], reverse=True)

    detected = [r["Concept"] for r in findings if r["Probability"] >= DOCTOR_THRESHOLD]

    evidence: dict = {}
    overlays: dict = {}
    for r in findings:
        name = r["_name"]
        cam = upscale_cam(maps[r["_k"]], (image_size, image_size))
        evidence[name] = cam
        overlays[name] = ui.overlay_image(display, cam)

    return {
        "display": display,
        "findings": findings,
        "diagnosis": diagnosis,
        "detected": detected,
        "evidence": evidence,
        "overlays": overlays,
        "threshold": DOCTOR_THRESHOLD,
    }


# --------------------------------------------------------------------------- #
# Viewing tools (window / level / invert) -- radiology-style display control
# --------------------------------------------------------------------------- #
def apply_view_controls(img: np.ndarray, invert: bool, window: float,
                        level: float) -> np.ndarray:
    """Re-map a grayscale uint8 image with window/level (and optional invert).

    Mirrors standard DICOM windowing: values outside ``[level - window/2,
    level + window/2]`` clip to black/white.
    """
    arr = np.asarray(img, dtype=np.float32)
    if invert:
        arr = 255.0 - arr
    window = max(float(window), 1.0)
    lo = float(level) - window / 2.0
    hi = lo + window
    arr = (arr - lo) / max(hi - lo, 1e-6) * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def adjusted_overlays(result: dict, invert: bool, window: float,
                      level: float) -> dict:
    """Re-composite evidence overlays onto a window/level-adjusted base."""
    base = apply_view_controls(result["display"], invert, window, level)
    return {
        name: ui.overlay_image(base, result["evidence"][name])
        for name in result["evidence"]
    }


# --------------------------------------------------------------------------- #
# Plain-language, patient-aware interpretation (no LLM -- rule based)
# --------------------------------------------------------------------------- #
def _build_doctor_explanation(result: dict, patient: dict) -> str:
    """Deterministic explanation written for clinicians, from model outputs."""
    detected = result["detected"]
    findings = result["findings"]
    threshold = result["threshold"]
    near = [
        r for r in findings
        if r["Probability"] < threshold
        and r["Probability"] >= 0.5 * threshold
    ]

    parts = []
    if detected:
        top = findings[0]
        lead = ", ".join(detected[:2])
        parts.append(
            f"The model identifies changes most consistent with {lead}."
        )
        if len(detected) > 2:
            parts.append(
                f"Additional findings: {', '.join(detected[2:])}."
            )
    else:
        parts.append(
            "No finding reached the model's decision threshold; the study is "
            "reported as unremarkable."
        )

    if near:
        names = [r["Concept"] for r in near]
        parts.append(
            f"Near-threshold values were noted for {', '.join(names)} — "
            "correlate with the clinical context."
        )

    top_diag = result["diagnosis"][0]
    if top_diag is not None and top_diag["Probability"] >= threshold:
        parts.append(
            f"The highest-scoring diagnosis from the model is "
            f"{top_diag['Diagnosis']} (probability "
            f"{top_diag['Probability']:.0%})."
        )

    indication = (patient or {}).get("indication") or ""
    if indication.strip():
        parts.append(
            "These model findings should be correlated with the reported "
            "clinical indication."
        )
    return " ".join(parts)


def _build_impression(result: dict) -> str:
    """Radiology-style one-line impression built from model outputs."""
    detected = result["detected"]
    threshold = result["threshold"]
    if detected:
        top = result["findings"][0]
        body = (
            f"Findings are present and most consistent with "
            f"{', '.join(detected)}. Highest model confidence: "
            f"{top['Concept']} ({top['Probability']:.0%}). "
            "Clinical correlation is recommended."
        )
        return body
    top_diag = result["diagnosis"][0]
    if top_diag is not None and top_diag["Probability"] >= threshold:
        body = (
            f"No significant abnormality was detected at the model threshold. "
            f"Highest diagnosis score: {top_diag['Diagnosis']} "
            f"({top_diag['Probability']:.0%})."
        )
        return body
    return (
        "No significant abnormality detected on this study. Correlate with "
        "the clinical presentation."
    )


def _build_report_html(result: dict, patient: dict, impression: str) -> str:
    """Printable, self-contained HTML radiology report (open → print to PDF)."""
    from datetime import datetime

    findings_rows = "".join(
        f"<tr><td>{r['Concept']}</td>"
        f"<td>{r['Probability']:.0%}</td>"
        f"<td>{ui.status_pill(r['Probability'], result['threshold'])}</td></tr>"
        for r in result["findings"]
    )
    diag_rows = "".join(
        f"<tr><td>{r['Diagnosis']}</td><td>{r['Probability']:.0%}</td></tr>"
        for r in result["diagnosis"]
    )
    age = f"{int(patient.get('age') or 0)} y" if patient.get("age") else "—"
    sex = patient.get("sex") or "—"
    indication = patient.get("indication") or "—"
    detected = ", ".join(result["detected"]) or "None"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Chest X-ray Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #1e293b; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }}
  .header {{ border-bottom: 3px solid #2563eb; padding-bottom: 0.6rem; margin-bottom: 1.2rem; }}
  .header h1 {{ margin: 0; font-size: 1.25rem; color: #0f172a; }}
  .header p {{ margin: 0.15rem 0; color: #64748b; font-size: 0.85rem; }}
  h2 {{ font-size: 0.95rem; color: #2563eb; text-transform: uppercase; letter-spacing: 0.06em; margin: 1.1rem 0 0.35rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th, td {{ text-align: left; padding: 0.35rem 0.5rem; border-bottom: 1px solid #e2e8f0; }}
  th {{ color: #64748b; font-weight: 600; font-size: 0.78rem; text-transform: uppercase; }}
  .impression {{ background: #eff6ff; border-left: 4px solid #2563eb; padding: 0.7rem 1rem; border-radius: 8px; margin-top: 0.8rem; }}
  .foot {{ margin-top: 2rem; color: #94a3b8; font-size: 0.75rem; border-top: 1px solid #e2e8f0; padding-top: 0.6rem; }}
</style></head>
<body>
  <div class="header">
    <h1>Chest X-ray Interpretation — Research Prototype</h1>
    <p><strong>Patient:</strong> {age}, {sex} &nbsp;|&nbsp;
       <strong>Study:</strong> Chest X-ray (PA) &nbsp;|&nbsp;
       <strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    <p><strong>Clinical indication:</strong> {indication}</p>
  </div>

  <h2>Impression</h2>
  <div class="impression"><p>{impression}</p></div>

  <h2>Findings (model concept probabilities)</h2>
  <table><tr><th>Finding</th><th>Probability</th><th>Status</th></tr>{findings_rows}</table>

  <h2>Diagnosis scores</h2>
  <table><tr><th>Diagnosis</th><th>Probability</th></tr>{diag_rows}</table>

  <h2>Detected findings</h2>
  <p>{detected}</p>

  <div class="foot">
    AI-assisted analysis is a research prototype and not validated for clinical
    use. This report must be reviewed by a qualified clinician. The predictions
    above are direct outputs of the model; no clinical decision should be made
    from this report alone.
  </div>
</body></html>"""


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def to_display_image(x: torch.Tensor, mean, std) -> np.ndarray:
    img = unnormalize(x.detach().cpu(), mean=mean, std=std)
    img = img[0].clamp(0, 1).numpy()
    img = np.transpose(img, (1, 2, 0))
    return (img * 255.0).astype(np.uint8)


def draw_image_with_boxes(ax, img: np.ndarray, boxes) -> None:
    ax.imshow(img, cmap="gray")
    h, w = img.shape[:2]
    for box in boxes:
        x0, y0, x1, y1 = normalized_box_to_pixels(box, h, w)
        ax.add_patch(
            Rectangle((x0, y0), x1 - x0 + 1, y1 - y0 + 1,
                      fill=False, edgecolor="lime", linewidth=1.5)
        )
        ax.text(x0, y0 - 3, box.class_name, color="lime", fontsize=7)
    ax.axis("off")


def upscale_cam(cam: np.ndarray, size) -> np.ndarray:
    t = torch.as_tensor(cam, dtype=torch.float32)[None, None]
    t = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
    up = t[0, 0].numpy()
    lo, hi = float(up.min()), float(up.max())
    if hi - lo > 1e-8:
        up = (up - lo) / (hi - lo)
    return up


# --------------------------------------------------------------------------- #
# Doctor View
# --------------------------------------------------------------------------- #
def _set_selected(name: str) -> None:
    """Point the evidence view at a given concept."""
    st.session_state["doctor_selected"] = name


def _img_to_data_uri(arr: np.ndarray) -> str:
    """Encode a uint8 RGB image as a base64 data URI for inline HTML."""
    from PIL import Image

    im = Image.fromarray(np.asarray(arr).astype(np.uint8))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _evidence_area(cam: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of the (normalized) activation map above a fixed threshold."""
    cam = np.asarray(cam, dtype=np.float32)
    if cam.size == 0:
        return 0.0
    return float((cam > threshold).mean())


def _doctor_upload_card():
    """Section 1 — upload card with thumbnail, filename and Analyze button."""
    uploaded = st.session_state.get("doctor_upload")
    with st.container(border=True):
        st.markdown(ui.card_title("1. Upload Chest X-ray", "blue"), unsafe_allow_html=True)
        st.file_uploader(
            "Chest X-ray (PNG, JPG, JPEG, DICOM)",
            type=UPLOAD_TYPES, key="doctor_upload", label_visibility="collapsed",
        )
        if uploaded is not None:
            try:
                preview = decode_uploaded(uploaded).resize((80, 80))
                pc, meta = st.columns([1, 2.1])
                pc.image(np.asarray(preview), width=80)
                meta.markdown(f"**{uploaded.name}**")
                meta.caption(f"{len(uploaded.getvalue()) // 1024} KB · uploaded")
                meta.caption("Replace by dropping a new file.")
            except Exception as exc:
                st.error(f"Could not read image: {exc}")
        else:
            st.caption(
                "Drag & drop image here or browse. "
                "Accepted: PNG, JPG, JPEG, DICOM."
            )
        analyze = st.button(
            "▶  Analyze X-ray", type="primary", key="doctor_analyze",
            use_container_width=True, disabled=uploaded is None,
        )
    return analyze


def _doctor_xray_card(result, uploaded):
    """Center — the uploaded chest X-ray, large and undistorted, with
    optional radiology viewing tools."""
    with st.container(border=True):
        st.markdown(ui.card_title("Uploaded Chest X-ray", "blue"), unsafe_allow_html=True)
        inv = st.session_state.get("doctor_invert", False)
        win = st.session_state.get("doctor_window", 256)
        lvl = st.session_state.get("doctor_level", 128)
        if result is not None:
            st.image(
                apply_view_controls(result["display"], inv, win, lvl),
                width=300, caption="Model input (grayscale, resized)",
            )
        elif uploaded is not None:
            try:
                preview = decode_uploaded(uploaded)
                st.image(
                    apply_view_controls(np.asarray(preview), inv, win, lvl),
                    width=300, caption="Uploaded X-ray",
                )
            except Exception:
                st.markdown('<div class="placeholder">Preview unavailable.</div>',
                            unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="placeholder">Upload an X-ray to see it here.</div>',
                unsafe_allow_html=True,
            )

        if uploaded is not None or result is not None:
            with st.expander("🖥 Viewing tools (window / level)"):
                c1, c2, c3 = st.columns([1, 1, 1])
                c1.checkbox("Invert", key="doctor_invert")
                c2.slider("Window", 1, 256, 256, key="doctor_window")
                c3.slider("Level", 0, 255, 128, key="doctor_level")
                st.markdown(
                    '<div class="tool-hint">Adjust brightness/contrast as in a '
                    "standard PACS viewer. Overlays update automatically.</div>",
                    unsafe_allow_html=True,
                )


def _doctor_overall_card(result):
    """Top-right — Overall Analysis: Download Report at the top-right,
    prediction summary, and the six concept rows with probability bars."""
    with st.container(border=True):
        h0, h1 = st.columns([1.6, 1.0], vertical_alignment="center")
        with h0:
            st.markdown(ui.card_title("Overall Analysis", "purple"), unsafe_allow_html=True)
        with h1:
            if result is not None:
                html = _build_report_html(
                    result, EMPTY_PATIENT, _build_impression(result)
                )
                st.download_button(
                    "📄 Download Report", data=html,
                    file_name="chest_xray_report.html", mime="text/html",
                    use_container_width=True,
                )
        if result is None:
            st.markdown(
                '<div class="pred-placeholder"><div class="big">Awaiting '
                "analysis</div>Upload an X-ray and click Analyze to see the "
                "overall analysis.</div>",
                unsafe_allow_html=True,
            )
            return

        top = result["diagnosis"][0]
        if result["detected"]:
            st.markdown(
                f'<div class="prediction-big">{top["Diagnosis"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="confidence">Model confidence: '
                f'{top["Probability"]:.0%}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(ui.status_banner(True), unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="prediction-big">No significant abnormality</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="confidence">Highest diagnosis score: '
                f'{top["Probability"]:.0%}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(ui.status_banner(False), unsafe_allow_html=True)

        st.markdown('<div class="section-label">Concept status</div>',
                    unsafe_allow_html=True)
        by_name = {r["_name"]: r for r in result["findings"]}
        ordered = [c for c in CONCEPT_ROW_ORDER if c in by_name]
        for name in ordered:
            r = by_name[name]
            st.markdown(
                ui.concept_row_html(
                    CONCEPT_ICON[name], r["Concept"],
                    ui.status_pill(r["Probability"], result["threshold"]),
                    r["Probability"], result["threshold"],
                ),
                unsafe_allow_html=True,
            )
        st.caption(
            "Probabilities represent model estimates and not clinical "
            "probabilities."
        )


def _doctor_concept_cards(result):
    """Six concept cards in one row — evidence image, name, probability and
    status badge. Clicking a card selects it (clear active state)."""
    with st.container(border=True):
        st.markdown(ui.card_title("Clinical Concepts", "blue"), unsafe_allow_html=True)
        st.caption("Click any concept to view its visual evidence and details.")
        if result is None:
            st.markdown(
                '<div class="placeholder">Concept cards appear after analysis.</div>',
                unsafe_allow_html=True,
            )
            return
        by_name = {r["_name"]: r for r in result["findings"]}
        ordered = [c for c in CONCEPT_ROW_ORDER if c in by_name]
        inv = st.session_state.get("doctor_invert", False)
        win = st.session_state.get("doctor_window", 256)
        lvl = st.session_state.get("doctor_level", 128)
        overlays = adjusted_overlays(result, inv, win, lvl)
        selected = st.session_state.get("doctor_selected")

        cols = st.columns(6, gap="small")
        for j, name in enumerate(ordered):
            r = by_name[name]
            active = name == selected
            uri = _img_to_data_uri(overlays[name])
            with cols[j]:
                st.markdown(
                    ui.concept_card_html(
                        r["Concept"], r["Probability"],
                        ui.status_pill(r["Probability"], result["threshold"]),
                        uri, active=active,
                    ),
                    unsafe_allow_html=True,
                )
                st.button(
                    "✓ Selected" if active else "View",
                    key=f"cc_{name}",
                    on_click=_set_selected, args=(name,),
                    use_container_width=True,
                    type="primary" if active else "secondary",
                )


def _doctor_concept_detail(result):
    """Selected concept — dynamic title, Original X-ray vs Model Evidence,
    deterministic 'What does this mean?' stats and an About box."""
    with st.container(border=True):
        if result is None:
            st.markdown(ui.card_title("Concept Details", "purple"), unsafe_allow_html=True)
            st.markdown(
                '<div class="placeholder">Select a concept above to see its '
                "detail.</div>",
                unsafe_allow_html=True,
            )
            return
        by_name = {r["_name"]: r for r in result["findings"]}
        selected = st.session_state.get("doctor_selected")
        if selected not in by_name:
            selected = CONCEPT_ROW_ORDER[0]
            st.session_state["doctor_selected"] = selected
        r = by_name[selected]

        st.markdown(
            ui.card_title(f"{CONCEPT_DISPLAY[selected]} — Concept Details", "purple"),
            unsafe_allow_html=True,
        )

        inv = st.session_state.get("doctor_invert", False)
        win = st.session_state.get("doctor_window", 256)
        lvl = st.session_state.get("doctor_level", 128)
        base = apply_view_controls(result["display"], inv, win, lvl)
        overlay = adjusted_overlays(result, inv, win, lvl)[selected]

        col_x, col_e = st.columns(2, gap="medium")
        with col_x:
            st.markdown('<div class="detail-label">Original X-ray</div>',
                        unsafe_allow_html=True)
            st.image(base, width=300)
        with col_e:
            st.markdown('<div class="detail-label">Model Evidence</div>',
                        unsafe_allow_html=True)
            st.image(overlay, width=300)
            st.image(ui.legend_image(), width=200)
            st.markdown(
                '<div class="legend-labels"><span>Low</span><span>High</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("**What does this mean?**")
        detected = r["Probability"] >= result["threshold"]
        area = _evidence_area(result["evidence"][selected])
        st.markdown(
            f'<div class="stat-grid">'
            f'<div class="stat-item"><div class="stat-label">Model Probability</div>'
            f'<div class="stat-value">{r["Probability"]:.0%}</div></div>'
            f'<div class="stat-item"><div class="stat-label">Status</div>'
            f'<div class="stat-value">{"Detected" if detected else "Not detected"}</div></div>'
            f'<div class="stat-item"><div class="stat-label">Evidence Area (approx.)</div>'
            f'<div class="stat-value">{area:.0%} of image</div></div>'
            f'<div class="stat-item"><div class="stat-label">Evidence</div>'
            f'<div class="stat-value">{ui.status_pill(r["Probability"], result["threshold"])}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if detected:
            st.markdown(
                f"The model **detects {r['Concept']}** — the highlighted regions "
                f"are where the Spatial CBM localizes the finding in the X-ray."
            )
        else:
            st.markdown(
                f"The model **does not detect {r['Concept']}** at its decision "
                f"threshold. The map is shown for completeness; no localized "
                f"finding is attributed to this concept."
            )
        st.caption(
            "Orange/red regions mark where the Spatial CBM localizes the "
            "concept in the image."
        )

        st.markdown(
            f'<div class="about-box"><div class="about-label">About this '
            f"Concept</div><p>{CONCEPT_ABOUT.get(selected, '')}</p>"
            '<p style="color:#64748b;font-size:0.8rem;">Descriptive only — '
            "not a diagnosis.</p></div>",
            unsafe_allow_html=True,
        )


def _doctor_why_card(result):
    """Why this prediction? — deterministic, traceable to the concept
    bottleneck. Lists the concepts driving the prediction."""
    with st.container(border=True):
        st.markdown(ui.card_title("Why this prediction?", "purple"), unsafe_allow_html=True)
        if result is None:
            st.markdown(
                '<div class="placeholder">The explanation appears after '
                "analysis.</div>",
                unsafe_allow_html=True,
            )
            return
        st.markdown(
            ui.impression_html("Impression", _build_impression(result)),
            unsafe_allow_html=True,
        )
        st.markdown(_build_doctor_explanation(result, EMPTY_PATIENT))
        st.markdown('<div class="section-label">Concepts driving the prediction</div>',
                    unsafe_allow_html=True)
        by_name = {r["_name"]: r for r in result["findings"]}
        driving = [r for r in result["findings"] if r["Probability"] >= result["threshold"]]
        if not driving:
            driving = sorted(result["findings"],
                             key=lambda x: x["Probability"], reverse=True)[:3]
        for r in driving:
            st.markdown(
                ui.concept_row_html(
                    CONCEPT_ICON[r["_name"]], r["Concept"],
                    ui.status_pill(r["Probability"], result["threshold"]),
                    r["Probability"], result["threshold"],
                ),
                unsafe_allow_html=True,
            )
        st.caption(
            "Traceable to the concept bottleneck. Language is deterministic — "
            "no LLM is used."
        )


def render_doctor_view():
    """Doctor View: 3-column top section (Upload | X-ray | Overall Analysis),
    then six concept cards, concept details, why-this-prediction, disclaimer."""
    if "doctor_result" not in st.session_state:
        st.session_state["doctor_result"] = None
    if "doctor_selected" not in st.session_state:
        st.session_state["doctor_selected"] = None
    if "doctor_file_id" not in st.session_state:
        st.session_state["doctor_file_id"] = None

    col_upload, col_xray, col_overall = st.columns(
        [0.8, 1.15, 1.4], gap="medium", vertical_alignment="top"
    )
    with col_upload:
        uploaded = st.session_state.get("doctor_upload")
        analyze = _doctor_upload_card()

    # Reset results whenever a new file replaces the previous one.
    if uploaded is not None:
        file_id = uploaded.file_id
        if st.session_state.get("doctor_file_id") != file_id:
            st.session_state["doctor_file_id"] = file_id
            st.session_state["doctor_result"] = None
            st.session_state["doctor_selected"] = None

    if analyze and uploaded is not None:
        try:
            with st.spinner("Analyzing the X-ray with the Spatial CBM..."):
                bundle = load_doctor_bundle()
                img = decode_uploaded(uploaded)
                result = run_doctor_inference(img, bundle)
                st.session_state["doctor_result"] = result
                st.session_state["doctor_selected"] = result["findings"][0]["_name"]
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

    result = st.session_state.get("doctor_result")

    with col_xray:
        _doctor_xray_card(result, uploaded)
    with col_overall:
        _doctor_overall_card(result)

    _doctor_concept_cards(result)
    _doctor_concept_detail(result)
    _doctor_why_card(result)

    st.markdown(f"**{DOCTOR_DISCLAIMER}**")


# --------------------------------------------------------------------------- #
# Researcher View — technical configuration (no sidebar)
# --------------------------------------------------------------------------- #
def _researcher_config():
    """Technical controls for the Researcher View (rendered in the body)."""
    data_root = st.text_input("Dataset root", value=DEFAULT_DATA_ROOT)
    ckpt_dir = st.text_input("Checkpoint directory", value=DEFAULT_CKPT_DIR)

    models = discover_checkpoints(ckpt_dir)
    labels = {
        m.run_id: f"{m.experiment} ({m.run_id})" for m in models
    }
    if not models:
        st.warning(
            "No checkpoints found under the configured directory. Train with "
            "`scripts/train_*.py` first."
        )
        return None, None, None, None

    c1, c2 = st.columns(2)
    selected = c1.selectbox(
        "Checkpoint", list(labels.keys()), format_func=lambda k: labels[k]
    )
    model = next(m for m in models if m.run_id == selected)
    split = c2.selectbox("Split", ["test", "val", "train"])

    c3, c4 = st.columns(2)
    batch_size = c3.number_input("Batch size", 1, 64, 8, 1)
    max_images = c4.number_input("Max images to scan", 1, 5000, 200, 1)

    if _is_chestxdet(model.config):
        root_valid = bool(str(data_root).strip())
        if not root_valid:
            st.warning("Dataset root (Hugging Face repo id) is empty.")
    else:
        root_valid = Path(data_root).exists()
        if not root_valid:
            st.warning("Dataset root does not exist on disk.")
    load_clicked = st.button(
        "Load model", disabled=not root_valid, type="primary",
    )
    return model, data_root, split, (load_clicked, batch_size, max_images)


# --------------------------------------------------------------------------- #
# Tabs (Researcher View)
# --------------------------------------------------------------------------- #
def _fmt_metric(value) -> str:
    if value is None:
        return "N/A"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if f != f:  # NaN
        return "N/A"
    return f"{f:.4f}"


def _render_metric_cards(metrics: dict):
    rows = [
        ("Diagnosis AUROC", metrics.get("diagnosis_auroc_macro")),
        ("Diagnosis AUPRC", metrics.get("diagnosis", {}).get("macro", {}).get("auprc")),
        ("Concept AUROC", metrics.get("concept_auroc_macro")),
        ("Concept F1", metrics.get("concepts", {}).get("macro", {}).get("f1")),
        ("Spatial IoU", metrics.get("spatial", {}).get("macro", {}).get("iou")),
        ("Pointing Game", metrics.get("spatial", {}).get("macro", {}).get("pointing_game")),
        ("ELS", metrics.get("spatial", {}).get("macro", {}).get("els")),
    ]
    cols = st.columns(len(rows))
    for col, (label, val) in zip(cols, rows):
        col.markdown(ui.metric_card_html(label, _fmt_metric(val)),
                     unsafe_allow_html=True)


def tab_overview(bundle):
    st.subheader("Overview")
    report_csv = PROJECT_ROOT / "outputs" / "reports" / "phase3" / "final_comparison.csv"
    if report_csv.exists():
        st.markdown(
            "Comparison from the latest `scripts/run_phase3_report.py` run:"
        )
        st.dataframe(pd.read_csv(report_csv).set_index("model"),
                     use_container_width=True)

    if bundle is not None:
        cfg = bundle["cfg"]
        results = bundle["results"]
        from src.evaluation.evaluation import build_metric_fn

        metric_fn = build_metric_fn(str(cfg.get("experiment")), cfg)
        metrics = metric_fn(results)
        st.markdown("**Live metrics (actual model outputs)**")
        _render_metric_cards(metrics)
        with st.expander("Full live metrics"):
            st.json(metrics, expanded=False)
    else:
        st.info("Load a model first (Researcher View configuration) to see live metrics.")


def tab_explorer(bundle):
    st.subheader("Image Explorer")
    if bundle is None:
        st.info("Load a model first (Researcher View configuration).")
        return

    results = bundle["results"]
    image_ids = results["image_ids"]
    index = {rid: i for i, rid in enumerate(image_ids)}
    selected_id = st.selectbox("Image", image_ids, format_func=lambda x: str(x))
    i = index[selected_id]

    concept_names = list(bundle["cfg"].concepts)
    diagnosis_names = list(bundle["cfg"].diagnoses)
    threshold = float(bundle["cfg"].get("evaluation.threshold", 0.5))

    x = bundle["dataset"][i]["image"].unsqueeze(0).to(bundle["device"])
    img = to_display_image(x, bundle["mean"], bundle["std"])
    boxes = results["boxes"][i] if "boxes" in results else []
    probs_d = sigmoid(results["diagnosis_logits"][i])
    probs_c = (
        sigmoid(results["concepts_logits"][i]) if "concepts_logits" in results else None
    )

    col_img, col_probs = st.columns([1, 1])
    with col_img:
        fig, ax = plt.subplots(figsize=(5, 5))
        draw_image_with_boxes(ax, img, boxes)
        ax.set_title(f"{selected_id} — boxes: "
                     f"{[b.class_name for b in boxes] or 'none'}")
        st.pyplot(fig)
        plt.close(fig)

    with col_probs:
        st.markdown("**Diagnosis probabilities**")
        ddf = pd.DataFrame(
            {"diagnosis": diagnosis_names, "probability": probs_d}
        ).sort_values("probability", ascending=False)
        st.dataframe(ddf.style.format({"probability": "{:.3f}"}),
                     height=min(420, 30 * len(ddf) + 40))

        if probs_c is not None:
            st.markdown("**Concept probabilities**")
            cdf = pd.DataFrame(
                {"concept": concept_names, "probability": probs_c}
            ).sort_values("probability", ascending=False)
            st.dataframe(cdf.style.format({"probability": "{:.3f}"}),
                         height=min(260, 30 * len(cdf) + 40))

    # Spatial evidence
    if "activation_maps" in results and results["activation_maps"] is not None:
        maps = np.asarray(results["activation_maps"])
        st.markdown("**Concept evidence maps (top 4 by probability)**")
        if probs_c is not None:
            top = np.argsort(probs_c)[::-1][:4]
        else:
            top = np.arange(min(4, len(concept_names)))
        cols = st.columns(4)
        for j, k in enumerate(top):
            with cols[j]:
                cam = maps[i, k]
                up = upscale_cam(cam, img.shape[:2])
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(img, cmap="gray", alpha=0.5)
                ax.imshow(up, cmap="jet", alpha=0.55)
                for box in boxes:
                    bx0, by0, bx1, by1 = normalized_box_to_pixels(
                        box, img.shape[0], img.shape[1]
                    )
                    ax.add_patch(Rectangle((bx0, by0), bx1 - bx0 + 1, by1 - by0 + 1,
                                           fill=False, edgecolor="white",
                                           linewidth=1.0))
                ax.axis("off")
                ax.set_title(concept_names[k])
                st.pyplot(fig)
                plt.close(fig)

    # Structured explanation
    if probs_c is not None:
        st.markdown("**Structured explanation (facts from model outputs)**")
        activation_map = (
            np.asarray(results["activation_maps"])[i]
            if "activation_maps" in results else None
        )
        explanation = build_structured_explanation(
            image_id=str(selected_id),
            diagnosis_logits=results["diagnosis_logits"][i],
            concept_logits=results["concepts_logits"][i],
            diagnosis_names=diagnosis_names,
            concept_names=concept_names,
            activation_map=activation_map,
            boxes=boxes or None,
            cam_threshold=float(bundle["cfg"].get("evaluation.cam_threshold", 0.5)),
        )
        st.markdown(render_markdown(explanation))

        st.markdown("**Intervention playground**")
        _render_intervention_playground(bundle, i, concept_names, diagnosis_names)


def _render_intervention_playground(bundle, i, concept_names, diagnosis_names):
    from src.interventions.runner import concept_probs_from_logits

    model = bundle["model"]
    device = bundle["device"]
    results = bundle["results"]
    z = concept_probs_from_logits(model, results["concepts_logits"][i : i + 1])

    concept = st.selectbox("Concept to intervene", concept_names)
    mode = st.selectbox("Mode", ["remove", "activate", "set", "increase", "decrease"])
    value = 0.5 if mode == "set" else 0.0
    if mode == "set":
        value = st.slider("Value", 0.0, 1.0, 0.5)
    delta = st.slider("Delta", -0.5, 0.5, 0.1) if mode in ("increase", "decrease") else 0.0

    if st.button("Apply intervention"):
        runner = InterventionRunner(model, concept_names, diagnosis_names, device=device)
        spec = build_intervention(mode, concept, value=value, delta=abs(delta))
        outcome = runner.intervene(
            torch.as_tensor(z, dtype=torch.float32), spec
        )
        before = outcome["diagnosis_probs_before"][0]
        after = outcome["diagnosis_probs_after"][0]
        df = pd.DataFrame(
            {
                "diagnosis": diagnosis_names,
                "before": before,
                "after": after,
                "delta": after - before,
            }
        ).sort_values("delta", key=lambda s: s.abs(), ascending=False)
        st.dataframe(
            df.style.format({"before": "{:.3f}", "after": "{:.3f}", "delta": "{:+.3f}"}),
            height=min(420, 30 * len(df) + 40),
        )


def tab_failure(bundle):
    st.subheader("Failure Analysis")
    if bundle is None:
        st.info("Load a model first (Researcher View configuration).")
        return
    cfg = bundle["cfg"]
    if str(cfg.get("experiment")) not in ("cbm", "spatial_cbm"):
        st.info("Failure analysis applies to CBM models only.")
        return

    from src.evaluation.failure_analysis import run_failure_analysis

    df, counts = run_failure_analysis(
        bundle["results"],
        list(cfg.diagnoses),
        list(cfg.concepts),
        threshold=float(cfg.get("evaluation.threshold", 0.5)),
        els_threshold=float(cfg.get("evaluation.cam_threshold", 0.5)),
    )
    st.dataframe(df)

    values = [
        int(counts.loc[counts["category"] == c, "count"].sum() or 0)
        for c in ["A", "B", "C", "D", "E"]
    ]
    labels = [
        "A: diag+concept+loc",
        "B: diag+concept, loc err",
        "C: diag ok, concept err",
        "D: diag err, concept ok",
        "E: all incorrect",
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="#4c72b0")
    ax.set_ylabel("images")
    ax.set_title("Failure analysis category counts")
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig)
    plt.close(fig)


def tab_interventions(bundle):
    st.subheader("Intervention Experiments")
    if bundle is None:
        st.info("Load a model first (Researcher View configuration).")
        return
    cfg = bundle["cfg"]
    if str(cfg.get("experiment")) not in ("cbm", "spatial_cbm"):
        st.info("Interventions apply to CBM models only.")
        return

    if st.button("Run intervention experiments on loaded images"):
        from src.interventions.experiments import (
            concept_completeness,
            concept_error_recovery,
            concept_error_robustness,
            oracle_concept_experiment,
        )
        from src.interventions.runner import (
            concept_probs_from_logits,
            run_dependency_analysis,
        )

        model = bundle["model"]
        device = bundle["device"]
        results = bundle["results"]
        concept_names = list(cfg.concepts)
        diagnosis_names = list(cfg.diagnoses)
        threshold = float(cfg.get("evaluation.threshold", 0.5))
        with st.spinner("Computing..."):
            oracle = oracle_concept_experiment(model, results, diagnosis_names, threshold)
            robustness = concept_error_robustness(model, results, diagnosis_names,
                                                  threshold=threshold)
            recovery = concept_error_recovery(model, results, diagnosis_names, threshold)
            completeness = concept_completeness(model, results, concept_names,
                                                diagnosis_names, threshold=threshold)
            z_mean = torch.as_tensor(
                concept_probs_from_logits(model, results["concepts_logits"]).mean(
                    axis=0, keepdims=True
                )
            )
            dependency = run_dependency_analysis(
                model, z_mean, concept_names, diagnosis_names,
                mode="remove", device=device,
            )

        st.markdown("**Oracle concepts**")
        st.json({
            k: v for k, v in oracle.get("gap", {}).items()
        }, expanded=False)

        st.markdown("**Robustness (diagnosis AUROC under concept corruption)**")
        st.dataframe(robustness.style.format({"auroc": "{:.4f}"}))

        st.markdown("**Recovery**")
        st.json(recovery.get("improvement", {}), expanded=False)

        st.markdown("**Completeness**")
        st.dataframe(completeness["summary"].style.format(
            {"diagnosis_auroc": "{:.4f}", "concept_auroc": "{:.4f}"}
        ))

        st.markdown("**Dependency matrix (mean |delta diagnosis P|)**")
        dep_df = pd.DataFrame(
            dependency["abs_change"],
            index=concept_names,
            columns=diagnosis_names,
        )
        st.dataframe(dep_df.style.format("{:.4f}"))


def tab_reports(_bundle):
    st.subheader("Reports / Metrics")
    root = PROJECT_ROOT / "outputs"
    exts = {".csv", ".json", ".md", ".html", ".png"}
    files = sorted(
        (
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in exts
            and not any(seg in p.parts for seg in ("checkpoints", "figures", ".cache"))
        ),
        key=lambda p: str(p).lower(),
    )
    if not files:
        st.info("No report files found under outputs/.")
        return
    options = {str(p.relative_to(root)): p for p in files}
    selected = st.selectbox("Select a stored report / metrics file", list(options.keys()))
    path = options[selected]
    st.caption(f"`{path}`")
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            st.dataframe(pd.read_csv(path), use_container_width=True)
        elif suffix == ".json":
            st.json(json.loads(path.read_text(encoding="utf-8")), expanded=False)
        elif suffix == ".md":
            st.markdown(path.read_text(encoding="utf-8"))
        elif suffix == ".html":
            components.html(path.read_text(encoding="utf-8"), height=600, scrolling=True)
        elif suffix == ".png":
            st.image(str(path))
    except Exception as exc:
        st.error(f"Could not preview `{path.name}`: {exc}")


# --------------------------------------------------------------------------- #
# Researcher View
# --------------------------------------------------------------------------- #
def render_researcher_view():
    st.markdown(
        '<div class="researcher-banner">🔬 Researcher View — '
        "Inspect → Evaluate → Analyze → Intervene</div>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(ui.card_title("Model Configuration", "purple"), unsafe_allow_html=True)
        st.caption("Technical settings — every number shown is an actual model output.")
        model, data_root, split, sidebar_opts = _researcher_config()
    if model is None:
        st.write("Configure the dataset root and checkpoint directory, then "
                 "train a model with `scripts/train_*.py`.")
        return

    load_clicked, batch_size, max_images = sidebar_opts
    bundle = None
    if load_clicked:
        try:
            bundle = load_model_bundle(
                str(model.checkpoint_path), data_root, split,
                int(batch_size), int(max_images),
            )
        except Exception as exc:
            st.error(f"Could not load model: {exc}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", "Image Explorer", "Failure Analysis", "Interventions", "Reports"]
    )
    with tab1:
        tab_overview(bundle)
    with tab2:
        tab_explorer(bundle)
    with tab3:
        tab_failure(bundle)
    with tab4:
        tab_interventions(bundle)
    with tab5:
        tab_reports(bundle)

    st.caption("Research prototype. Not for clinical use.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _render_header() -> None:
    """Compact header: brand on the left, prototype badge on the right."""
    col_l, col_r = st.columns([1.8, 1], vertical_alignment="center")
    with col_l:
        st.markdown(ui.header_left_html(), unsafe_allow_html=True)
    with col_r:
        st.markdown(
            f'<div style="display:flex;justify-content:flex-end;">'
            f"{ui.prototype_badge_html()}</div>",
            unsafe_allow_html=True,
        )


def main():
    _render_header()
    tab_doctor, tab_researcher = st.tabs(
        ["🩺 Doctor View", "🔬 Researcher View"]
    )
    with tab_doctor:
        render_doctor_view()
    with tab_researcher:
        render_researcher_view()


if __name__ == "__main__":
    main()
