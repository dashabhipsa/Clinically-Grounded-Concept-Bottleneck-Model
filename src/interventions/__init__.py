"""Phase 3 concept-intervention engine and analysis experiments."""
from __future__ import annotations

from src.interventions.experiments import (
    concept_completeness,
    concept_error_recovery,
    concept_error_robustness,
    concept_subsets,
    corrupt_concepts,
    oracle_concept_experiment,
    summarize_oracle_experiment,
    summarize_recovery,
)
from src.interventions.intervention import (
    ConceptIntervention,
    INTERVENTION_MODES,
    apply_interventions,
    build_intervention,
    concept_indices,
    intervene_vector,
    validate_concepts,
)
from src.interventions.metrics import (
    InterventionMetrics,
    relative_change,
    save_dependency_matrix,
    save_interventions_csv,
)
from src.interventions.runner import (
    InterventionRunner,
    concept_probs_from_logits,
    diagnosis_probs_from_concept_matrix,
    extract_concept_probs,
    run_dependency_analysis,
)

__all__ = [
    "ConceptIntervention",
    "InterventionRunner",
    "InterventionMetrics",
    "INTERVENTION_MODES",
    "apply_interventions",
    "build_intervention",
    "concept_indices",
    "concept_probs_from_logits",
    "concept_completeness",
    "concept_error_recovery",
    "concept_error_robustness",
    "concept_subsets",
    "corrupt_concepts",
    "diagnosis_probs_from_concept_matrix",
    "extract_concept_probs",
    "intervene_vector",
    "oracle_concept_experiment",
    "relative_change",
    "run_dependency_analysis",
    "save_dependency_matrix",
    "save_interventions_csv",
    "summarize_oracle_experiment",
    "summarize_recovery",
    "validate_concepts",
]
