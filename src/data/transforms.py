"""Image transforms for the VinDr-CXR pipeline.

Transforms operate on PIL images (grayscale 'L') and return normalized
3-channel float tensors suitable for ImageNet-pretrained encoders.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


class ToRGBTensor:
    """Convert a PIL image to a tensor and replicate a single channel to RGB."""

    def __call__(self, img):
        tensor = F.to_tensor(img)
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "ToRGBTensor()"


def build_transform(
    image_size: int,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
    train: bool = False,
    augment: bool = True,
) -> T.Compose:
    """Build the (optionally augmented) train or eval transform."""
    ops: List = [T.Resize((int(image_size), int(image_size)), interpolation=T.InterpolationMode.BILINEAR)]
    if train and augment:
        ops += [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=5, translate=(0.03, 0.03)),
        ]
    ops += [ToRGBTensor(), T.Normalize(mean=list(mean), std=list(std))]
    return T.Compose(ops)


def unnormalize(
    tensor: torch.Tensor,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Inverse of the normalization used in ``build_transform``."""
    mean_t = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std_t = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return tensor * std_t + mean_t
