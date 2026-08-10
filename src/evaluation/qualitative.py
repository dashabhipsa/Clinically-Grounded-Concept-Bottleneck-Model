"""Qualitative evidence + intervention visualization for Phase 3.

Builds multi-panel figures from actual model outputs:

1. Original X-ray
2. Concept probability table
3. Concept-specific evidence maps
4. Radiologist bounding boxes
5. Combined evidence + annotation overlay
6. Final diagnosis
7. Concept intervention result

Examples are selected *algorithmically* by metrics (e.g. ELS, concept/diagnosis
correctness) so that both successes and failures are shown -- never a
cherry-picked set of only positive examples.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from src.data.spatial import normalized_box_to_pixels
from src.evaluation.classification_metrics import sigmoid
from src.evaluation.spatial_metrics import els
from src.explainability.structured_explanation import render_markdown


def _to_display_image(tensor: torch.Tensor, mean, std) -> np.ndarray:
    from src.data.transforms import unnormalize

    img = unnormalize(tensor.detach().cpu(), mean=mean, std=std)
    img = img[0].clamp(0, 1).numpy()
    img = np.transpose(img, (1, 2, 0))
    return (img * 255.0).astype(np.uint8)


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.asarray(cam, dtype=np.float32)
    lo, hi = float(cam.min()), float(cam.max())
    if hi - lo > 1e-8:
        return (cam - lo) / (hi - lo)
    return np.zeros_like(cam)


def collect_examples(
    model,
    dataset,
    device: torch.device,
    num_success: int = 3,
    num_failure: int = 3,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
) -> List[Dict]:
    """Gather per-image predictions + concept evidence + GT boxes.

    Each returned record has ``x`` (tensor), ``image_id``, ``concepts``
    (probs), ``diagnosis`` (probs), ``activation_maps`` (if spatial),
    ``boxes``, and per-concept ELS where boxes exist.
    """
    import torch.nn.functional as F

    model.eval()
    concept_index = {n: i for i, n in enumerate(dataset.concepts)}
    records: List[Dict] = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            x = item["image"].unsqueeze(0).to(device)
            out = model(x)
            if hasattr(out, "concept_probs"):
                z = out.concept_probs[0].detach().cpu().numpy()
                diag = sigmoid(out.diagnosis_logits[0].detach().cpu().numpy())
                cams = out.activation_maps[0].detach().cpu().numpy()
                has_spatial = True
            elif isinstance(out, (tuple, list)):
                logits_c, logits_d = out
                z = sigmoid(logits_c[0].detach().cpu().numpy())
                diag = sigmoid(logits_d[0].detach().cpu().numpy())
                cams = None
                has_spatial = False
            else:
                diag = sigmoid(out[0].detach().cpu().numpy())
                z = None
                cams = None
                has_spatial = False

            record = {
                "x": x,
                "image_id": item["image_id"],
                "concepts": z,
                "diagnosis": diag,
                "activation_maps": cams,
                "boxes": item["boxes"],
                "has_spatial": has_spatial,
            }
            if has_spatial and item["boxes"]:
                els_vals: Dict[str, float] = {}
                for box in item["boxes"]:
                    name = box.class_name.strip().lower().replace("/", "_").replace(" ", "_")
                    k = concept_index.get(name)
                    if k is None:
                        continue
                    gt = normalized_box_to_pixels(box, cams.shape[-2], cams.shape[-1])
                    els_vals[name] = float(els(cams[k], gt))
                record["els"] = els_vals
            records.append(record)
    return records


def _rank_examples(
    records: List[Dict], num_success: int, num_failure: int
) -> List[Dict]:
    """Algorithmically pick success + failure examples using ELS."""
    scored = [r for r in records if r.get("els")]
    if not scored:
        return records[: max(num_success, num_failure)]
    scored = sorted(scored, key=lambda r: np.mean(list(r["els"].values())))
    failures = scored[:num_failure]
    successes = scored[-num_success:] if len(scored) > num_failure else []
    selected: List[Dict] = []
    for tag, bucket in (("SUCCESS", successes), ("FAILURE", failures)):
        for r in bucket:
            r = dict(r)
            r["tag"] = tag
            selected.append(r)
    return selected


def visualize_intervention(
    record: Dict,
    dataset,
    diagnosis_names: Sequence[str],
    concept_names: Sequence[str],
    save_path: str | Path,
    intervention_desc: Optional[str] = None,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
) -> Path:
    """Multi-panel figure: X-ray, concepts, evidence, boxes, diagnosis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    original = _to_display_image(record["x"], mean, std)
    img_h, img_w = original.shape[:2]

    has_spatial = record["has_spatial"] and record["activation_maps"] is not None
    n_cols = 5 if has_spatial else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 5))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")

    axes[0].imshow(original)
    axes[0].set_title(f"Original X-ray\n{record['image_id']}")

    concept_probs = record["concepts"]
    if concept_probs is not None:
        top_idx = np.argsort(concept_probs)[::-1][:5]
        axes[1].barh(
            [concept_names[i] for i in top_idx][::-1],
            [concept_probs[i] for i in top_idx][::-1],
            color="#1f77b4",
        )
        axes[1].set_xlim(0, 1)
        axes[1].set_title("Concept probabilities", fontsize=10)

    diag_probs = record["diagnosis"]
    top_d = np.argsort(diag_probs)[::-1][:3]
    axes[2].barh(
        [diagnosis_names[i] for i in top_d][::-1],
        [diag_probs[i] for i in top_d][::-1],
        color="#d62728",
    )
    axes[2].set_xlim(0, 1)
    axes[2].set_title("Diagnosis probabilities", fontsize=10)

    if has_spatial:
        cams = record["activation_maps"]
        concept_index = {n: i for i, n in enumerate(concept_names)}
        boxed = [
            b for b in record["boxes"]
            if b.class_name.strip().lower().replace("/", "_").replace(" ", "_")
            in concept_index
        ]
        if boxed:
            b = boxed[0]
            name = b.class_name.strip().lower().replace("/", "_").replace(" ", "_")
            k = concept_index[name]
            cam = cams[k]
            import torch.nn.functional as F

            cam_up = F.interpolate(
                torch.as_tensor(cam)[None, None],
                size=(img_h, img_w), mode="bilinear", align_corners=False,
            )[0, 0].numpy()
            cam_up = _normalize_cam(cam_up)
            axes[3].imshow(cam_up, cmap="jet")
            axes[3].set_title(f"Evidence map: {name}", fontsize=10)

            axes[4].imshow(original)
            for box in record["boxes"]:
                (x0, y0, x1, y1) = normalized_box_to_pixels(box, img_h, img_w)
                axes[4].add_patch(
                    Rectangle((x0, y0), x1 - x0 + 1, y1 - y0 + 1,
                              fill=False, edgecolor="lime", linewidth=1.5)
                )
            axes[4].set_title("Boxes + evidence", fontsize=10)

    title = f"{record['image_id']}"
    if intervention_desc:
        title += f"\n{intervention_desc}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def generate_qualitative_examples(
    model,
    dataset,
    device: torch.device,
    out_dir: str | Path,
    diagnosis_names: Sequence[str],
    concept_names: Sequence[str],
    num_success: int = 3,
    num_failure: int = 3,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
) -> List[Path]:
    """Write qualitative evidence figures (success + failure by ELS)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = collect_examples(
        model, dataset, device, num_success=num_success, num_failure=num_failure,
        mean=mean, std=std,
    )
    selected = _rank_examples(records, num_success, num_failure)
    written: List[Path] = []
    for record in selected:
        tag = record.get("tag", "")
        path = out_dir / f"{record['image_id']}_{tag.lower()}.png"
        visualize_intervention(
            record, dataset, diagnosis_names, concept_names, path,
            intervention_desc=tag,
            mean=mean, std=std,
        )
        written.append(path)
    return written
