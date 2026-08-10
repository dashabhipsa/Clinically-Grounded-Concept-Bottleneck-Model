"""Bounding-box <-> activation-mask handling and coordinate transforms.

VinDr-CXR box coordinates are normalized to [0, 1] in each axis. Because the
pipeline square-resizes every image to ``(image_size, image_size)``, a
normalized coordinate maps to the same *fractional* location in any grid. A
box can therefore be rasterized directly at feature-map (CAM) resolution with
a uniform scale:

    x_pixel = round(x_norm * (grid_w - 1))
    y_pixel = round(y_norm * (grid_h - 1))

This is the coordinate transformation applied after image resizing: boxes are
stored normalized, so no aspect-ratio bookkeeping is needed between the
resized image and the downsampled feature map.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.data.annotations import Box, normalize_class_name


def normalized_box_to_pixels(
    box: Box, height: int, width: int
) -> Tuple[int, int, int, int]:
    """Map a normalized ``Box`` to pixel coordinates ``(x0, y0, x1, y1)``.

    Coordinates are computed on a ``(height, width)`` grid (typically the CAM
    resolution), clamped to the grid and ordered so ``x0 <= x1`` and
    ``y0 <= y1`` (a fully clipped box collapses to a single pixel).
    """
    height, width = int(height), int(width)
    if height < 1 or width < 1:
        raise ValueError(f"grid must be at least 1x1, got {width}x{height}")

    x0 = int(round(float(box.x_min) * (width - 1)))
    x1 = int(round(float(box.x_max) * (width - 1)))
    y0 = int(round(float(box.y_min) * (height - 1)))
    y1 = int(round(float(box.y_max) * (height - 1)))

    x0, x1 = sorted((max(x0, 0), min(x1, width - 1)))
    y0, y1 = sorted((max(y0, 0), min(y1, height - 1)))
    return x0, y0, x1, y1


def boxes_to_masks(
    boxes: Sequence[Box],
    concept_names: Sequence[str],
    cam_height: int,
    cam_width: int,
) -> np.ndarray:
    """Rasterize boxes into per-concept binary masks ``[K, H, W]``.

    ``masks[k, y, x] == 1`` if pixel ``(x, y)`` lies inside at least one box
    of concept ``k``. Concepts without any box in ``boxes`` stay all-zero.
    Boxes whose class is not part of ``concept_names`` are ignored.
    """
    concept_names = list(concept_names)
    if not concept_names:
        raise ValueError("concept_names must be non-empty")
    index = {name: i for i, name in enumerate(concept_names)}

    masks = np.zeros(
        (len(concept_names), int(cam_height), int(cam_width)), dtype=np.float32
    )
    for box in boxes:
        k = index.get(normalize_class_name(box.class_name))
        if k is None:
            continue
        x0, y0, x1, y1 = normalized_box_to_pixels(box, cam_height, cam_width)
        masks[k, y0:y1 + 1, x0:x1 + 1] = 1.0
    return masks


def smooth_masks(
    masks: torch.Tensor, sigma: float = 1.0
) -> torch.Tensor:
    """Gaussian-smooth binary masks -> soft grounding targets.

    ``masks`` may be ``[B, K, H, W]``, ``[K, H, W]`` or ``[H, W]``; a
    symmetric Gaussian kernel (radius ``ceil(3*sigma)``) is applied
    per-channel with zero boundary padding.
    """
    if sigma is None or float(sigma) <= 0.0:
        return masks
    masks = torch.as_tensor(masks, dtype=torch.float32)
    ndim = masks.ndim
    if ndim == 2:
        masks = masks[None, None]
    elif ndim == 3:
        masks = masks[None]
    elif ndim != 4:
        raise ValueError(f"masks must be 2D..4D, got {masks.shape}")

    radius = int(math.ceil(3.0 * float(sigma)))
    size = 2 * radius + 1
    kernel_1d = torch.exp(
        -(torch.arange(-radius, radius + 1, dtype=torch.float32) ** 2)
        / (2.0 * float(sigma) ** 2)
    )
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    channels = masks.shape[1]
    kernel = kernel_2d.view(1, 1, size, size).repeat(channels, 1, 1, 1)

    smoothed = F.conv2d(masks, kernel, padding=radius, groups=channels)

    if ndim == 2:
        smoothed = smoothed[0, 0]
    elif ndim == 3:
        smoothed = smoothed[0]
    return smoothed


def boxes_to_masks_tensor(
    boxes_list: Sequence[Sequence[Box]],
    concept_names: Sequence[str],
    cam_height: int,
    cam_width: int,
    device: Optional[torch.device] = None,
    smooth_sigma: float = 0.0,
) -> torch.Tensor:
    """Batch of masks ``[B, K, H, W]`` from a batch of box lists."""
    masks = np.stack(
        [
            boxes_to_masks(sample, concept_names, cam_height, cam_width)
            for sample in boxes_list
        ]
    )
    tensor = torch.as_tensor(masks, dtype=torch.float32)
    if float(smooth_sigma or 0.0) > 0.0:
        tensor = smooth_masks(tensor, sigma=float(smooth_sigma))
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def concept_presence_from_masks(masks: torch.Tensor) -> torch.Tensor:
    """Boolean ``[B, K]`` tensor: does image b have a box for concept k?"""
    return masks.sum(dim=(2, 3)) > 0.5
