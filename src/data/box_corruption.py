"""Deterministic spatial-supervision corruption.

These corrupt the *bounding boxes* used as grounding supervision during
training (the concept multi-hot labels and the evaluation-time boxes are left
untouched). All corruption is deterministic given the configured random seed:
each run seeds a ``numpy`` ``RandomState`` once and applies the same sequence
of perturbations to the same batch ordering.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from src.data.annotations import Box


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def shift_box(box: Box, dx: float, dy: float) -> Box:
    """Shift a box by normalized offsets, clipping to [0, 1]."""
    return Box(
        image_id=box.image_id,
        class_id=box.class_id,
        class_name=box.class_name,
        x_min=_clamp01(box.x_min + dx),
        x_max=_clamp01(box.x_max + dx),
        y_min=_clamp01(box.y_min + dy),
        y_max=_clamp01(box.y_max + dy),
        rad_id=box.rad_id,
    )


def random_box(box: Box, rng: np.random.RandomState) -> Box:
    """Place a same-size box at a random (deterministic) location."""
    w, h = max(float(box.width), 1e-4), max(float(box.height), 1e-4)
    x0 = float(rng.uniform(0.0, max(1e-4, 1.0 - w)))
    y0 = float(rng.uniform(0.0, max(1e-4, 1.0 - h)))
    return Box(
        image_id=box.image_id,
        class_id=box.class_id,
        class_name=box.class_name,
        x_min=x0,
        x_max=_clamp01(x0 + w),
        y_min=y0,
        y_max=_clamp01(y0 + h),
        rad_id=box.rad_id,
    )


def corrupt_boxes(
    boxes: Sequence[Box],
    mode: str = "correct",
    rng: Optional[np.random.RandomState] = None,
    shift: Sequence[float] = (0.2, 0.2),
) -> list:
    """Apply a corruption mode to a list of boxes.

    Modes
    -----
    "correct" : return the boxes unchanged.
    "none"    : drop all box supervision (returns ``[]`` -> grounding is 0).
    "shifted" : translate each box by ``rng.uniform(-shift, +shift)``.
    "random"  : relocate each box at a deterministic random position.
    """
    mode = str(mode or "correct").lower()
    if mode == "correct":
        return list(boxes)
    if mode == "none":
        return []
    if rng is None:
        rng = np.random
    if mode == "shifted":
        return [
            shift_box(b, float(rng.uniform(-shift[0], shift[0])),
                      float(rng.uniform(-shift[1], shift[1])))
            for b in boxes
        ]
    if mode == "random":
        return [random_box(b, rng) for b in boxes]
    raise ValueError(f"Unknown corruption mode {mode!r}; "
                     f"choose from 'correct', 'none', 'shifted', 'random'.")
