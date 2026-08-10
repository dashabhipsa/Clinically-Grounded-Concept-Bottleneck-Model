"""Structured, factual explanations produced from the model's own outputs.

The explanation is assembled strictly from:

1. concept predictions (probabilities),
2. concept-specific spatial evidence maps,
3. the diagnostic bottleneck output (diagnosis probabilities),
4. actual intervention results (edited concept -> recomputed diagnosis).

No LLM is used to invent medical reasoning; every number shown is a direct
model output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.data.spatial import normalized_box_to_pixels
from src.evaluation.classification_metrics import sigmoid
from src.evaluation.spatial_metrics import els


@dataclass
class StructuredExplanation:
    """All facts of one structured explanation (values from the model only)."""

    image_id: str
    diagnosis_probs: Dict[str, float]
    concept_probs: Dict[str, float]
    concept_threshold: float = 0.5
    spatial_evidence: Dict[str, Optional[float]] = field(default_factory=dict)
    intervention: Optional[Dict] = None

    @property
    def top_diagnosis(self) -> str:
        return max(self.diagnosis_probs, key=lambda k: self.diagnosis_probs[k])

    @property
    def top_diagnosis_prob(self) -> float:
        return self.diagnosis_probs[self.top_diagnosis]

    def detected_concepts(self) -> List[str]:
        return sorted(
            (k for k, v in self.concept_probs.items() if v >= self.concept_threshold),
            key=lambda k: self.concept_probs[k],
            reverse=True,
        )

    def to_dict(self) -> Dict:
        return {
            "image_id": self.image_id,
            "diagnosis_probs": self.diagnosis_probs,
            "top_diagnosis": self.top_diagnosis,
            "top_diagnosis_prob": self.top_diagnosis_prob,
            "concept_probs": self.concept_probs,
            "detected_concepts": self.detected_concepts(),
            "spatial_evidence": self.spatial_evidence,
            "intervention": self.intervention,
        }


def build_structured_explanation(
    image_id: str,
    diagnosis_logits: np.ndarray,
    concept_logits: np.ndarray,
    diagnosis_names: Sequence[str],
    concept_names: Sequence[str],
    activation_map: Optional[np.ndarray] = None,
    boxes=None,
    cam_threshold: float = 0.5,
) -> StructuredExplanation:
    """Assemble a :class:`StructuredExplanation` from raw model outputs.

    ``activation_map`` is the concept evidence ``[K, H, W]`` (Spatially
    Grounded CBM) or None (Standard CBM / black-box). When present and boxes
    are provided, per-concept ELS is computed and stored as spatial evidence.
    """
    p_d = sigmoid(np.asarray(diagnosis_logits, dtype=np.float64)).ravel()
    p_c = sigmoid(np.asarray(concept_logits, dtype=np.float64)).ravel()

    diag_probs = {n: float(p) for n, p in zip(list(diagnosis_names), p_d)}
    concept_probs = {n: float(p) for n, p in zip(list(concept_names), p_c)}

    spatial: Dict[str, Optional[float]] = {}
    if activation_map is not None and boxes is not None:
        index = {n: i for i, n in enumerate(list(concept_names))}
        cam = np.asarray(activation_map)
        cam_h, cam_w = cam.shape[-2], cam.shape[-1]
        for box in boxes:
            name = box.class_name.strip().lower().replace("/", "_").replace(" ", "_")
            k = index.get(name)
            if k is None:
                continue
            gt = normalized_box_to_pixels(box, cam_h, cam_w)
            spatial[name] = float(els(cam[k], gt))

    return StructuredExplanation(
        image_id=image_id,
        diagnosis_probs=diag_probs,
        concept_probs=concept_probs,
        concept_threshold=float(cam_threshold),
        spatial_evidence=spatial,
    )


def attach_intervention(
    explanation: StructuredExplanation,
    concept: str,
    mode: str,
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    diagnosis_names: Sequence[str],
    value: float = 0.0,
) -> StructuredExplanation:
    """Attach a real intervention result to an explanation."""
    p_before = sigmoid(np.asarray(probs_before, dtype=np.float64)).ravel()
    p_after = sigmoid(np.asarray(probs_after, dtype=np.float64)).ravel()
    delta = {
        n: float(b) - float(a)
        for n, a, b in zip(list(diagnosis_names), p_before, p_after)
    }
    explanation.intervention = {
        "concept": concept,
        "mode": mode,
        "value": value,
        "concept_before": explanation.concept_probs.get(concept),
        "diagnosis_delta": delta,
    }
    return explanation


def render_markdown(explanation: StructuredExplanation) -> str:
    """Render the explanation as text. Values are the model's own outputs."""
    lines: List[str] = []
    lines.append(f"### Structured explanation — image {explanation.image_id}")
    lines.append("")
    lines.append(f"**Diagnosis:** {explanation.top_diagnosis} — "
                 f"probability {explanation.top_diagnosis_prob:.3f}")
    lines.append("")
    lines.append("**Diagnosis probabilities (actual model output):**")
    for name, prob in sorted(explanation.diagnosis_probs.items(),
                             key=lambda kv: kv[1], reverse=True):
        lines.append(f"- {name}: {prob:.3f}")

    lines.append("")
    lines.append(f"**Detected concepts (>= {explanation.concept_threshold:.2f}):**")
    detected = explanation.detected_concepts()
    if detected:
        for name in detected:
            lines.append(f"- {name}: {explanation.concept_probs[name]:.3f}")
    else:
        lines.append("- (none above threshold)")

    if explanation.spatial_evidence:
        lines.append("")
        lines.append("**Spatial evidence (ELS of concept-specific maps):**")
        for name, els_val in sorted(explanation.spatial_evidence.items(),
                                    key=lambda kv: kv[1], reverse=True):
            lines.append(f"- {name}: ELS = {els_val:.3f}" if els_val is not None
                         else f"- {name}: ELS = N/A")

    if explanation.intervention is not None:
        lines.append("")
        lines.append("**Intervention (actual result):**")
        inv = explanation.intervention
        lines.append(
            f"- Concept: {inv['concept']} ({inv['mode']}); "
            f"value was {inv['concept_before']:.3f}"
            if inv.get("concept_before") is not None
            else f"- Concept: {inv['concept']} ({inv['mode']})"
        )
        for name, delta in sorted(
            inv["diagnosis_delta"].items(), key=lambda kv: abs(kv[1]), reverse=True
        ):
            sign = "+" if delta >= 0 else ""
            lines.append(f"  - {name}: {sign}{delta:.3f}")

    lines.append("")
    lines.append("_Research prototype only. Not intended for clinical diagnosis._")
    return "\n".join(lines)
