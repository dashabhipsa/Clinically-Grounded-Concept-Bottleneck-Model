"""Intervention runner: recompute diagnosis from a modified concept vector
without re-running the image encoder.

Both the standard CBM and the Spatially Grounded CBM expose a
``diagnosis_head`` that is an affine map on the concept vector ``z``. The
runner therefore:

1. runs the image encoder exactly once to obtain ``z``,
2. applies the requested intervention to ``z``,
3. evaluates ``diagnosis_head(z_modified)``.

The encoder is never invoked again during steps 2-3, which is the property the
tests verify (``test_encoder_not_rerun_after_intervention``).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.evaluation.classification_metrics import sigmoid
from src.interventions.intervention import ConceptIntervention, build_intervention


def _activation_fn(model) -> Callable:
    act = getattr(model, "activation", torch.sigmoid)
    if callable(act):
        return act
    if str(act) in ("sigmoid", torch.sigmoid):
        return torch.sigmoid
    return torch.sigmoid


def concept_probs_from_logits(model, logits_np: np.ndarray) -> np.ndarray:
    """Map raw concept logits to the concept vector z (the bottleneck input)."""
    logits = torch.as_tensor(np.asarray(logits_np, dtype=np.float32))
    act = _activation_fn(model)
    return act(logits).cpu().numpy()


def diagnosis_probs_from_concept_matrix(
    model, z_np: np.ndarray, device=None
) -> np.ndarray:
    """Sigmoid(diagnosis_head(z)) from a numpy concept matrix.

    Only the (cheap) diagnosis head is evaluated here -- the encoder is not
    involved.
    """
    z = torch.as_tensor(np.asarray(z_np, dtype=np.float32))
    if device is None:
        params = list(model.parameters())
        device = params[0].device if params else torch.device("cpu")
    with torch.no_grad():
        logits = model.diagnosis_head(z.to(device))
        return torch.sigmoid(logits).cpu().numpy()


def extract_concept_probs(output) -> torch.Tensor:
    """Pull the concept-vector ``z`` out of a model forward-pass output."""
    if hasattr(output, "concept_probs"):
        return output.concept_probs
    if isinstance(output, (tuple, list)) and len(output) >= 1:
        return output[0]
    raise TypeError(f"Cannot extract concept probs from {type(output)}")


class InterventionRunner:
    """Reusable bottleneck-intervention machinery for any CBM-style model."""

    def __init__(
        self,
        model,
        concept_names: Sequence[str],
        diagnosis_names: Sequence[str],
        device=None,
    ):
        if not hasattr(model, "diagnosis_head"):
            raise TypeError(
                "InterventionRunner requires a model with a 'diagnosis_head' "
                "that maps the concept vector to diagnosis logits."
            )
        self.model = model
        self.concept_names = list(concept_names)
        self.diagnosis_names = list(diagnosis_names)
        self.concept_index = {n: i for i, n in enumerate(self.concept_names)}
        self.device = device

    # ------------------------------------------------------------------ #
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, object]:
        """Run the encoder once and return ``(concept_probs z, raw output)``."""
        out = self.model(x)
        return extract_concept_probs(out), out

    def diagnosis_probs(self, z: torch.Tensor) -> np.ndarray:
        """Sigmoid(diagnosis_head(z)); no encoder involved."""
        z_np = z.detach().cpu().numpy() if torch.is_tensor(z) else np.asarray(z)
        return diagnosis_probs_from_concept_matrix(self.model, z_np, self.device)

    # ------------------------------------------------------------------ #
    def intervene(
        self,
        z: torch.Tensor,
        intervention: ConceptIntervention,
        ground_truth: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Apply one intervention to ``z`` and recompute diagnosis probs.

        Returns a dict with the modified vector and the diagnosis probabilities
        before and after the intervention.
        """
        index = self.concept_index[intervention.concept]
        z_mod = intervention.apply(z, index, ground_truth=ground_truth)
        p_before = self.diagnosis_probs(z)
        p_after = self.diagnosis_probs(z_mod)
        return {
            "concept": intervention.concept,
            "mode": intervention.mode,
            "index": index,
            "z_before": z,
            "z_after": z_mod,
            "diagnosis_probs_before": p_before,
            "diagnosis_probs_after": p_after,
        }

    def intervene_many(
        self,
        z: torch.Tensor,
        interventions: Sequence[ConceptIntervention],
        ground_truth: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Chain multiple interventions, then recompute diagnosis once."""
        z_mod = z
        applied: List[ConceptIntervention] = []
        for spec in interventions:
            index = self.concept_index[spec.concept]
            z_mod = spec.apply(z_mod, index, ground_truth=ground_truth)
            applied.append(spec)
        p_before = self.diagnosis_probs(z)
        p_after = self.diagnosis_probs(z_mod)
        return {
            "interventions": applied,
            "z_before": z,
            "z_after": z_mod,
            "diagnosis_probs_before": p_before,
            "diagnosis_probs_after": p_after,
        }

    # ------------------------------------------------------------------ #
    def single_image(self, x: torch.Tensor):
        """Run one image through the encoder; return (z[1,K], output)."""
        z, out = self.predict(x)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        return z[0], out


def run_dependency_analysis(
    model,
    z: torch.Tensor,
    concept_names: Sequence[str],
    diagnosis_names: Sequence[str],
    mode: str = "remove",
    device=None,
) -> Dict[str, np.ndarray]:
    """Intervene each concept and measure the mean |delta diagnosis prob|.

    Returns ``{"abs_change": [K, D], "signed_change": [K, D]}`` where row k
    corresponds to intervening concept k and column d to diagnosis d. The
    matrix is computed from actual model predictions -- no medical priors.
    """
    runner = InterventionRunner(model, concept_names, diagnosis_names, device=device)
    z_np = z.detach().cpu().numpy() if torch.is_tensor(z) else np.asarray(z)
    p0 = diagnosis_probs_from_concept_matrix(model, z_np, device)
    k = len(concept_names)
    d = len(diagnosis_names)
    abs_change = np.zeros((k, d))
    signed_change = np.zeros((k, d))
    for i, name in enumerate(concept_names):
        spec = build_intervention(mode, name)
        p_i = runner.diagnosis_probs(spec.apply(z, i))
        signed_change[i] = p_i - p0
        abs_change[i] = np.abs(p_i - p0)
    return {"abs_change": abs_change, "signed_change": signed_change}


def numpy_to_probabilities(logits: np.ndarray) -> np.ndarray:
    return sigmoid(np.asarray(logits, dtype=np.float64))
