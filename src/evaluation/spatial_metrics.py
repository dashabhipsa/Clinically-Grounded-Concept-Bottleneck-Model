"""Spatial fidelity metrics: IoU, Pointing Game, Evidence Localization Score.

Only (image, concept) pairs that actually carry a radiologist bounding box
are scored. Reported per concept and macro-averaged.

* **IoU**            -- IoU between the radiologist box and the predicted box
                        (the bounding box of the thresholded activation map).
* **Pointing Game**  -- fraction of positive instances whose *single most
                        activated location* (CAM argmax) lies inside the box.
* **ELS**            -- Evidence Localization Score: fraction of the total
                        concept activation energy inside the radiologist box.

.. note::
    The evidence comes from the concept-specific model head (the CAM), never
    from post-hoc Grad-CAM.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.data.annotations import Box, normalize_class_name
from src.data.spatial import normalized_box_to_pixels


def cam_binary_mask(
    cam: np.ndarray, threshold: float = 0.5
) -> Optional[np.ndarray]:
    """Threshold a CAM at ``threshold * max`` -> binary mask (or None)."""
    cam = np.asarray(cam)
    hi = float(cam.max()) if cam.size else 0.0
    if hi <= 0.0:
        return None
    return (cam >= hi * float(threshold)).astype(np.uint8)


def mask_bbox(
    mask: Optional[np.ndarray],
) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box ``(x0, y0, x1, y1)`` of the thresholded region (or None)."""
    if mask is None:
        return None
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def box_iou(gt: Tuple[int, ...], pred: Tuple[int, ...]) -> float:
    """IoU of two inclusive pixel-coordinate boxes ``(x0, y0, x1, y1)``."""
    x0, y0 = max(gt[0], pred[0]), max(gt[1], pred[1])
    x1, y1 = min(gt[2], pred[2]), min(gt[3], pred[3])
    inter = max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)
    area_gt = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
    area_pred = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)
    union = area_gt + area_pred - inter
    return float(inter / union) if union > 0 else 0.0


def pointing_hit(cam: np.ndarray, gt: Tuple[int, ...]) -> int:
    """Pointing Game hit: is the CAM argmax inside the box?"""
    cam = np.asarray(cam)
    h, w = cam.shape
    if cam.size == 0 or float(cam.max()) <= 0.0:
        return 0
    idx = int(cam.reshape(-1).argmax())
    x, y = idx % w, idx // w
    x0, y0, x1, y1 = gt
    return int(x0 <= x <= x1 and y0 <= y <= y1)


def els(cam: np.ndarray, gt: Tuple[int, ...]) -> float:
    """Evidence Localization Score: box energy / total energy."""
    cam = np.asarray(cam)
    h, w = cam.shape
    total = float(cam.sum())
    if total <= 0.0:
        return float("nan")
    x0, y0, x1, y1 = gt
    ys, xs = np.ogrid[:h, :w]
    inside = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    return float(cam[inside].sum()) / total


def evaluate_spatial_metrics(
    activation_maps: np.ndarray,
    boxes: Sequence[Sequence[Box]],
    concept_names: Sequence[str],
    cam_threshold: float = 0.5,
) -> Dict:
    """Aggregate IoU / Pointing Game / ELS per concept and macro.

    ``activation_maps``: ``[N, K, H, W]`` concept activation maps.
    ``boxes``: length-``N`` list of per-image lists of :class:`Box` (GT).
    Returns ``{"per_class": {name: {...}}, "macro": {...}}``.
    """
    activation_maps = np.asarray(activation_maps)
    if activation_maps.ndim != 4:
        raise ValueError(
            f"activation_maps must be [N, K, H, W], got {activation_maps.shape}"
        )
    if len(boxes) != activation_maps.shape[0]:
        raise ValueError(
            f"boxes ({len(boxes)}) must match the map batch "
            f"({activation_maps.shape[0]})"
        )

    concept_names = list(concept_names)
    num, _, cam_h, cam_w = activation_maps.shape
    index = {name: i for i, name in enumerate(concept_names)}

    per_instance: Dict[str, Dict[str, List[float]]] = {
        name: {"iou": [], "pointing": [], "els": []} for name in concept_names
    }

    for i in range(num):
        for box in boxes[i]:
            name = normalize_class_name(box.class_name)
            k = index.get(name)
            if k is None:
                continue
            cam = activation_maps[i, k]
            gt = normalized_box_to_pixels(box, cam_h, cam_w)
            pred = mask_bbox(cam_binary_mask(cam, cam_threshold))
            per_instance[name]["iou"].append(
                box_iou(gt, pred) if pred is not None else 0.0
            )
            per_instance[name]["pointing"].append(pointing_hit(cam, gt))
            per_instance[name]["els"].append(els(cam, gt))

    per_class: Dict[str, Dict] = {}
    for name in concept_names:
        data = per_instance[name]
        num_inst = len(data["iou"])
        per_class[name] = {
            "num_instances": num_inst,
            "iou": float(np.mean(data["iou"])) if num_inst else float("nan"),
            "pointing_game": (
                float(np.mean(data["pointing"])) if num_inst else float("nan")
            ),
            # CAMs may be all-zero for some positive instances (ELS undefined
            # there); NaN-aware mean keeps the rest of the instances scored.
            "els": (
                float(np.nanmean(data["els"]))
                if any(np.isfinite(v) for v in data["els"]) else float("nan")
            ),
        }

    macro: Dict[str, float] = {}
    for metric in ("iou", "pointing_game", "els"):
        values = [
            per_class[n][metric]
            for n in concept_names
            if per_class[n]["num_instances"] > 0
        ]
        finite = [v for v in values if np.isfinite(v)]
        macro[metric] = float(np.nanmean(finite)) if finite else float("nan")
    return {"per_class": per_class, "macro": macro}
