"""Concept-evidence visualization for the Spatially Grounded CBM.

For selected test images we produce a five-panel figure:

    Original X-ray | Concept probability | Model concept evidence map |
    Radiologist bounding box | Combined overlay

Both *successful* (high ELS) and *failure* (low ELS) examples are generated.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from src.data.annotations import Box, normalize_class_name
from src.data.spatial import normalized_box_to_pixels
from src.data.transforms import unnormalize
from src.evaluation.spatial_metrics import els


def _to_display_image(tensor: torch.Tensor, mean, std) -> np.ndarray:
    """Normalized [1, 3, H, W] tensor -> uint8 RGB array."""
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


def visualize_concept_evidence(
    image_tensor: torch.Tensor,
    cam: np.ndarray,
    boxes: Sequence[Box],
    image_id: str,
    concept_name: str,
    concept_prob: float,
    concept_true: float,
    els_value: float,
    save_path: str | Path,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    title: Optional[str] = None,
    dpi: int = 150,
) -> Path:
    """Save a five-panel concept-evidence figure.

    Panels: (1) original X-ray, (2) concept probability, (3) model concept
    evidence map, (4) radiologist bounding box, (5) combined overlay.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    original = _to_display_image(image_tensor, mean, std)
    img_h, img_w = original.shape[:2]

    cam_t = torch.as_tensor(np.asarray(cam, dtype=np.float32))[None, None]
    cam_up = F.interpolate(
        cam_t, size=(img_h, img_w), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    cam_up = _normalize_cam(cam_up)
    heatmap = (cm.jet(cam_up)[..., :3] * 255.0).astype(np.uint8)
    overlay = (
        0.5 * original.astype(np.float32) + 0.5 * heatmap.astype(np.float32)
    ).astype(np.uint8)

    concept_boxes = [
        b for b in boxes if normalize_class_name(b.class_name) == concept_name
    ]
    box_pixels = [normalized_box_to_pixels(b, img_h, img_w) for b in concept_boxes]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    for ax in axes:
        ax.axis("off")

    axes[0].imshow(original)
    axes[0].set_title("Original X-ray")

    axes[1].barh([0], [float(concept_prob)], color="#1f77b4", height=0.6)
    axes[1].set_xlim(0, 1)
    axes[1].set_yticks([])
    axes[1].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[1].axvline(0.5, color="gray", ls="--", lw=0.8)
    axes[1].set_title(
        "Concept probability\n"
        f"p={concept_prob:.2f}  true={int(concept_true)}  ELS={els_value:.2f}",
        fontsize=10,
    )

    axes[2].imshow(heatmap)
    axes[2].set_title("Model evidence map")

    axes[3].imshow(original)
    for (x0, y0, x1, y1) in box_pixels:
        axes[3].add_patch(
            Rectangle(
                (x0, y0), x1 - x0 + 1, y1 - y0 + 1,
                fill=False, edgecolor="lime", linewidth=1.5,
            )
        )
    axes[3].set_title("Radiologist bounding box")

    axes[4].imshow(overlay)
    for (x0, y0, x1, y1) in box_pixels:
        axes[4].add_patch(
            Rectangle(
                (x0, y0), x1 - x0 + 1, y1 - y0 + 1,
                fill=False, edgecolor="lime", linewidth=1.5,
            )
        )
    axes[4].set_title("Combined overlay")

    fig.suptitle(
        title or f"{image_id} | {concept_name} | p={concept_prob:.2f} | ELS={els_value:.2f}"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


def generate_spatial_examples(
    model,
    dataset,
    device: torch.device,
    out_dir: str | Path,
    num_success: int = 3,
    num_failure: int = 3,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
) -> list:
    """Generate concept-evidence figures on the first boxed images.

    Returns the list of written file paths. Among all (image, concept)
    instances with a radiologist box, the ``num_success`` highest-ELS are
    tagged "SUCCESS" and the ``num_failure`` lowest-ELS "FAILURE".
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    concept_index = {name: i for i, name in enumerate(dataset.concepts)}
    examples = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            if not item["boxes"]:
                continue
            x = item["image"].unsqueeze(0).to(device)
            out = model(x)
            cams = out.activation_maps[0].detach().cpu().numpy()
            probs = out.concept_probs[0].detach().cpu().numpy()
            for box in item["boxes"]:
                name = normalize_class_name(box.class_name)
                k = concept_index.get(name)
                if k is None:
                    continue
                cam = cams[k]
                gt = normalized_box_to_pixels(box, cam.shape[0], cam.shape[1])
                examples.append(
                    {
                        "x": x,
                        "cam": cam,
                        "boxes": item["boxes"],
                        "image_id": item["image_id"],
                        "concept_name": name,
                        "prob": float(probs[k]),
                        "true": float(item["concepts"][k].item()),
                        "els": float(els(cam, gt)),
                    }
                )

    if not examples:
        return []

    examples.sort(key=lambda e: e["els"])
    failures = examples[:num_failure]
    successes = examples[-num_success:] if len(examples) > num_failure else []

    selected = []
    seen = set()
    for tag, bucket in (("SUCCESS", successes), ("FAILURE", failures)):
        for e in bucket:
            key = (e["image_id"], e["concept_name"])
            if key in seen:
                continue
            seen.add(key)
            selected.append((tag, e))

    written = []
    for tag, e in selected:
        path = out_dir / f"{e['image_id']}_{e['concept_name']}_{tag.lower()}.png"
        visualize_concept_evidence(
            e["x"], e["cam"], e["boxes"], e["image_id"], e["concept_name"],
            e["prob"], e["true"], e["els"], path,
            mean=mean, std=std,
            title=f"{tag} | {e['image_id']} | {e['concept_name']} "
                  f"| p={e['prob']:.2f} | ELS={e['els']:.2f}",
        )
        written.append(str(path))
    return written
