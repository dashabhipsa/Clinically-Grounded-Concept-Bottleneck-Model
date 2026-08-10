"""Concept interventions operating directly on the bottleneck representation.

A Concept Bottleneck Model factorises the prediction as

    image --encoder--> (concept logits) --activation--> z --diagnosis_head--> diagnosis

An *intervention* edits the concept vector ``z`` BEFORE the diagnosis head and
recomputes the diagnosis as ``diagnosis_head(z_modified)``. The image encoder
is never re-run for the intervention step: given a single forward pass that
produces ``z``, every supported intervention is a pure transformation of the
concept vector followed by one affine diagnosis-head evaluation.

Supported interventions (configurable and reproducible):

=================  ==============================
Mode               Effect on concept k
=================  ==============================
``"remove"``       z_k -> 0
``"activate"``     z_k -> 1
``"set"``          z_k -> value
``"increase"``     z_k -> z_k + delta
``"decrease"``     z_k -> z_k - delta
``"correct"``      z_k -> ground-truth value (when valid)
=================  ==============================
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch

INTERVENTION_MODES = (
    "remove", "activate", "set", "increase", "decrease", "correct",
)


class ConceptIntervention:
    """A single editable operation on one concept of the bottleneck vector."""

    def __init__(
        self,
        mode: str,
        concept: str,
        value: float = 0.0,
        delta: float = 0.0,
    ):
        mode = str(mode).lower()
        if mode not in INTERVENTION_MODES:
            raise ValueError(
                f"Unknown intervention mode {mode!r}; choose from "
                f"{sorted(INTERVENTION_MODES)}"
            )
        self.mode = mode
        self.concept = str(concept)
        self.value = float(value)
        self.delta = float(delta)

    def apply(
        self,
        z: torch.Tensor,
        index: int,
        ground_truth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return a copy of ``z`` with the intervention applied at column ``index``.

        ``ground_truth`` may be a full ``[..., K]`` tensor (the GT concept
        column is picked), a vector/scalar of per-sample GT values, or None.
        The "correct" mode requires it; other modes ignore it.
        """
        z = torch.as_tensor(z, dtype=torch.float32)
        z_mod = z.clone()
        if self.mode == "remove":
            z_mod[..., index] = 0.0
        elif self.mode == "activate":
            z_mod[..., index] = 1.0
        elif self.mode == "set":
            z_mod[..., index] = float(self.value)
        elif self.mode == "increase":
            z_mod[..., index] = z_mod[..., index] + float(self.delta)
        elif self.mode == "decrease":
            z_mod[..., index] = z_mod[..., index] - float(self.delta)
        elif self.mode == "correct":
            if ground_truth is None:
                raise ValueError(
                    "The 'correct' intervention requires a ground_truth tensor."
                )
            gt = torch.as_tensor(ground_truth, dtype=torch.float32)
            if gt.ndim == z.ndim and gt.shape[-1] == z.shape[-1]:
                gt = gt[..., index]
            z_mod[..., index] = gt
        return z_mod

    def describe(self) -> str:
        if self.mode in ("remove", "activate", "correct"):
            return f"{self.concept}={self.mode}"
        if self.mode == "set":
            return f"{self.concept}={self.value:.4g}"
        sign = "+" if self.mode == "increase" else "-"
        return f"{self.concept}{sign}{abs(self.delta):.4g}"

    def __repr__(self) -> str:
        return (
            f"ConceptIntervention(mode={self.mode!r}, concept={self.concept!r}, "
            f"value={self.value!r}, delta={self.delta!r})"
        )


def build_intervention(
    mode: str,
    concept: str,
    value: float = 0.0,
    delta: float = 0.0,
) -> ConceptIntervention:
    """Factory for :class:`ConceptIntervention` (config-driven, reproducible)."""
    return ConceptIntervention(mode, concept, value=value, delta=delta)


def apply_interventions(
    z: torch.Tensor,
    interventions: Sequence[ConceptIntervention],
    concept_names: Sequence[str],
    ground_truth: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Chain a list of interventions onto a single concept vector ``z``.

    Each intervention is applied to the copy produced by the previous one, so
    multiple concepts can be edited before the diagnosis head is re-evaluated.
    """
    z_mod = torch.as_tensor(z, dtype=torch.float32).clone()
    index = {name: i for i, name in enumerate(concept_names)}
    for intervention in interventions:
        if intervention.concept not in index:
            raise KeyError(
                f"Concept {intervention.concept!r} not in provided names."
            )
        z_mod = intervention.apply(
            z_mod, index[intervention.concept], ground_truth=ground_truth
        )
    return z_mod


def intervene_vector(
    z: torch.Tensor,
    mode: str,
    concept: str,
    concept_names: Sequence[str],
    value: float = 0.0,
    delta: float = 0.0,
    ground_truth: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convenience wrapper: build one intervention and apply it."""
    spec = build_intervention(mode, concept, value=value, delta=delta)
    return apply_interventions(
        z, [spec], concept_names, ground_truth=ground_truth
    )


def concept_indices(concept_names: Sequence[str]) -> dict:
    return {name: i for i, name in enumerate(list(concept_names))}


def validate_concepts(names: List[str]) -> None:
    if not names:
        raise ValueError("concept_names must be non-empty")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate concept names in {names}")
