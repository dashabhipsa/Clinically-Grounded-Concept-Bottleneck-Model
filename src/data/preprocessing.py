"""DICOM reading and grayscale image preprocessing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from PIL import Image


def read_dicom(path: str | Path) -> np.ndarray:
    """Read a DICOM file and return the pixel array as float32.

    Applies PhotometricInterpretation inversion (MONOCHROME1) and the
    RescaleSlope/RescaleIntercept windowing where present.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DICOM file not found: {path}")
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)

    # MONOCHROME1 = higher values are darker (invert for display).
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        arr = arr.max() - arr

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    if slope != 1.0 or intercept != 0.0:
        arr = arr * slope + intercept

    return arr


def normalize_grayscale(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize a grayscale array to [0, 1]."""
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def array_to_image(arr: np.ndarray) -> Image.Image:
    """Convert a normalized grayscale array in [0, 1] to a PIL 'L' image."""
    normalized = normalize_grayscale(arr)
    return Image.fromarray((normalized * 255.0).astype(np.uint8))


def dicom_to_image(path: str | Path) -> Image.Image:
    """Full pipeline: DICOM -> normalized grayscale PIL image."""
    return array_to_image(read_dicom(path))


def resize_image(img: Image.Image, size: int, interpolation: int = Image.BILINEAR) -> Image.Image:
    """Square-resize a PIL image to ``(size, size)``."""
    return img.resize((int(size), int(size)), resample=interpolation)


def inspect_dicom(path: str | Path) -> dict:
    """Return basic metadata about a DICOM file (for debugging/validation)."""
    ds = pydicom.dcmread(path)
    return {
        "shape": tuple(ds.pixel_array.shape),
        "photometric": getattr(ds, "PhotometricInterpretation", None),
        "bits_allocated": getattr(ds, "BitsAllocated", None),
        "modality": getattr(ds, "Modality", None),
        "slope": getattr(ds, "RescaleSlope", None),
        "intercept": getattr(ds, "RescaleIntercept", None),
    }
