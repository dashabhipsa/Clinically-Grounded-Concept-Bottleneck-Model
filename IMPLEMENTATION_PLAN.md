# Implementation Plan

## Research title

**Beyond Concept Prediction: Clinically Grounded and Spatially Faithful Concept Bottlenecks for Chest X-ray Diagnosis**

## Project goal (3 phases)

Build a Concept Bottleneck Model for chest X-ray diagnosis in which concepts are (a) clinically grounded radiographic findings and (b) spatially faithful to the underlying pathology. Compare against a black-box baseline and a standard (non-spatial) CBM.

---

## Phase 1 — Foundation (this phase)

**Goal:** a clean, reproducible baseline stack with no spatial grounding yet.

- [x] VinDr-CXR dataset pipeline (DICOM loading, normalization, transforms)
- [x] Annotation parser (bounding boxes, global labels, metadata)
- [x] Reproducible train/val/test split (seed 42, official test untouched)
- [x] Black-box diagnostic classifier (DenseNet-121 → diagnosis)
- [x] Grad-CAM post-hoc baseline
- [x] Standard Concept Bottleneck Model (no direct encoder → diagnosis)
- [x] Evaluation framework (per-class + macro AUROC/AUPRC/F1/precision/recall/sensitivity/specificity)
- [x] Experiment configuration system (YAML, overridable)
- [x] Unit tests (incl. CBM bottleneck-integrity test)
- [x] Docs: README, research notes, notebook

**Explicitly NOT in Phase 1:** spatial grounding, grounding loss, intervention, concept/box corruption, Streamlit dashboard.

---

## Phase 2 — Spatial grounding

Completed code (results pending until a real run):

- [x] Concept evidence localization: use the provided bounding boxes to supervise where each concept is detected (concept-specific spatial heads).
- [x] A **grounding loss** aligning concept evidence with box coordinates while preserving concept presence prediction.
- [x] Updated CBM variant (`GroundedConceptBottleneckModel`) replacing the global-pooled concept head.
- [x] Spatial fidelity metrics (localization accuracy, IoU, box containment) alongside concept AUROC.
- [x] Ablations: with vs. without grounding loss; different backbone pooling strategies.

## Phase 3 — Intervention & analysis

Completed code (results pending until the checkpoints are trained and
`scripts/run_phase3_report.py` is re-run):

- [x] **Intervention experiments**: replace predicted concepts with ground-truth concepts at test time; measure diagnosis AUROC under perfect and corrupted concept input.
- [x] **Corruption experiments**: concept / box corruption curves (robustness of the diagnosis head).
- [x] **Final report pipeline** (`scripts/run_phase3_report.py`): collates metrics, interventions (oracle / robustness / recovery / completeness / dependency), A-E failure analysis, qualitative evidence figures and structured explanations into `outputs/reports/phase3/`. Missing checkpoints are reported as N/A — never fabricated.
- [x] **Streamlit dashboard** (`dashboard/app.py`): interactively view images, boxes, concept evidence maps, diagnosis explanations, and run concept interventions.

---

## Dependency map

```
configs ────────────────────────────────┐
   │                                     │
   v                                     v
data (split -> dataset)          models (encoder -> blackbox / cbm)
   │                                     │
   └─────────────┬───────────────────────┘
                 v
             training (trainer, losses)
                 │
                 v
          evaluation (metrics, predict)
                 │
                 └───────────────┬──────────────┐
                                 v              v
                          explainability   outputs (checkpoints, metrics,
                          (gradcam)          predictions, figures, logs)
```

## Experiment protocol (Phase 1)

1. `scripts/prepare_data.py` once.
2. `scripts/train_blackbox.py` (baseline + Grad-CAM examples).
3. `scripts/train_cbm.py` (standard CBM).
4. `scripts/evaluate.py` per checkpoint on `test`.
5. Fill the comparison table in the README with actual results.

## Reproducibility

- Seed 42 for everything (python, numpy, torch, CUDA, DataLoader workers).
- All experiment settings live in YAML; the resolved config is saved next to each checkpoint.
- Splits are generated deterministically and stored as CSVs.
- No numbers are reported without a real run.

## Open questions / assumptions

See `reports/research_notes.md` (dataset assumptions, concept definitions, metric conventions).
