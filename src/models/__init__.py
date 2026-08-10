"""Model factories."""
from __future__ import annotations

from src.models.blackbox import BlackBoxClassifier, build_blackbox
from src.models.cbm import ConceptBottleneckModel, build_cbm
from src.models.encoder import ImageEncoder
from src.models.spatial_cbm import (
    SpatialCBMOutput,
    SpatiallyGroundedCBM,
    build_spatial_cbm,
)


def build_model(config) -> "torch.nn.Module":
    """Build a model from config based on ``config.experiment``.

    ``"blackbox"`` -> BlackBoxClassifier, ``"cbm"`` -> ConceptBottleneckModel,
    ``"spatial_cbm"`` -> SpatiallyGroundedCBM.
    """
    experiment = str(config.get("experiment", "blackbox")).lower()
    if experiment == "spatial_cbm":
        return build_spatial_cbm(config)
    if experiment == "cbm":
        return build_cbm(config)
    if experiment == "blackbox":
        return build_blackbox(config)
    raise ValueError(
        f"Unknown experiment {experiment!r}; "
        f"choose 'blackbox', 'cbm' or 'spatial_cbm'."
    )


__all__ = [
    "BlackBoxClassifier", "ConceptBottleneckModel", "ImageEncoder",
    "SpatialCBMOutput", "SpatiallyGroundedCBM",
    "build_blackbox", "build_cbm", "build_spatial_cbm", "build_model",
]
