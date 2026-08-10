# Beyond Concept Prediction: Clinically Grounded and Spatially Faithful Concept Bottlenecks for Chest X-ray Diagnosis

This repository implements a research project on **Concept Bottleneck Models (CBMs)** for chest X-ray diagnosis.

## Research motivation

Standard CBMs map images to a compact set of high-level concepts and then predict a target from that concept vector. This improves interpretability, but the concepts are typically predicted from global image features, so the model can be right for the wrong reasons:

* the *presence* of a finding may be correct while its *spatial location* is wrong, and
* the diagnosis head cannot be interrogated locally ("why here, why now?").

Later phases of this project make concept predictions **clinically grounded** (the concepts are standard radiographic findings) and **spatially faithful** (the concept evidence must localize to the corresponding pathology, enforced with the provided bounding-box annotations). This phase lays the foundation: a clean data pipeline, a black-box baseline, a Grad-CAM post-hoc baseline, and a standard CBM to compare against.

## Dataset

**[VinDr-CXR](https://physionet.org/content/vindr-cxr/1.0.0/)** (PhysioNet), a large public dataset of frontal chest X-rays with radiologist annotations.

* 18,000 training / 3,000 test images (DICOM).
* **Local findings**: bounding boxes for 14 finding classes (`annotations/train.csv`, `annotations/test.csv`), coordinates normalized to `[0, 1]`.
* **Global diagnostic labels**: per-image presence of the same 14 findings (`annotations/image_labels_train.csv`, `annotations/image_labels_test.csv`).
* **Metadata**: `metadata.csv` with `image_id`, `patient_id`, `study_id`.

The 14 VinDr classes:

```
aortic_enlargement, atelectasis, calcification, cardiomegaly,
consolidation, ild, infiltration, lung_opacity, nodule_mass,
other_lesion, pleural_effusion, pleural_thickening, pneumothorax,
pulmonary_fibrosis
```

### Design choices

* **Concepts** (the bottleneck layer) are a configurable subset of the local findings. Defaults:

  ```
  cardiomegaly, pleural_effusion, consolidation, lung_opacity,
  edema, atelectasis, pneumothorax, nodule_mass
  ```

  > `edema` is **not** a VinDr-CXR annotation class. It is included by default to demonstrate that concepts with no annotations are handled as all-zeros vectors (it will always be predicted negative in Phase 1). Either keep it (as a test of robustness) or remove it from `concepts` in `configs/base.yaml`.

* **Diagnoses** (the diagnosis head targets) are the global diagnostic labels (all 14 classes). They are kept **strictly separate** from the concept layer.

## Installation

```bash
cd clinically-grounded-cbm
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
# or
pip install -e .
```

Requirements: Python >= 3.9, PyTorch >= 2.0, torchvision, pydicom, PIL, numpy, pandas, scikit-learn, PyYAML, matplotlib, pytest, tqdm.

For GPU training, install the matching CUDA build of PyTorch from <https://pytorch.org> **before** `pip install -r requirements.txt`.

---

## Dataset setup




Extract it and set its path in `configs/base.yaml`:

```yaml
data:
  root: "/path/to/vindr-cxr"
```

Expected layout:

### Prepare splits

```bash
python scripts/prepare_data.py --config configs/base.yaml --root /path/to/vindr-cxr
```

This validates the layout, builds a **reproducible** train/val split (official test set untouched) with seed 42, and writes:

```
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
```

It also prints per-concept / per-diagnosis prevalence and validates a sample of DICOMs.

---

## Training

### Black-box baseline

```bash
python scripts/train_blackbox.py --config configs/base.yaml --config configs/blackbox.yaml
```

```
X-ray → DenseNet-121 → diagnosis (multi-label BCE)
```

### Standard CBM

```bash
python scripts/train_cbm.py --config configs/base.yaml --config configs/cbm.yaml
```

```
X-ray → DenseNet-121 → ConceptHead → concept vector z → DiagnosisHead
```

**Critical:** the diagnosis head receives *only* `z = [z1, ..., zk]`. There is **no** encoder → diagnosis connection (verified by `tests/test_models.py::test_cbm_bottleneck_integrity_no_encoder_to_diagnosis`).

### What training saves (per run, under `outputs/checkpoints/<run_id>/`)

```
best_model.pth
last_model.pth
training_history.csv
config.yaml
```

plus (under `outputs/metrics/<run_id>/` and `outputs/predictions/<run_id>/`):

```
metrics.json        # test-set metrics incl. per-class + macro
predictions_test.csv
```

and Grad-CAM figures for the black-box baseline under `outputs/figures/gradcam/<run_id>/`.

Training pipeline features:

* AdamW optimizer
* LR scheduler (ReduceLROnPlateau or cosine)
* early stopping (patience) with best-checkpoint selection
* checkpointing (`best` / `last`)
* mixed precision (AMP) where a GPU is available
* reproducible seed (42) including DataLoader workers

---

## Evaluation

```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/blackbox_*/best_model.pth --split test
python scripts/evaluate.py --checkpoint outputs/checkpoints/cbm_*/best_model.pth --split test
```

Add `--gradcam` to regenerate Grad-CAM figures for a black-box checkpoint.

### Metrics

For **diagnoses** (all models):

* AUROC, AUPRC, F1, precision, recall, sensitivity, specificity
* per-diagnosis and macro average

For **concepts** (CBM only):

* same metric set, per-concept and macro average

Threshold-based metrics (F1/precision/recall/specificity) use a 0.5 decision threshold on sigmoid probabilities.

### Comparison table

Numbers are filled only after experiments are actually run — nothing is fabricated.

| Model        | Diagnosis AUROC | Diagnosis AUPRC | Concept AUROC |
| ------------ | --------------: | --------------: | ------------: |
| Black-box    |       (pending) |        (pending) |           N/A |
| Standard CBM |       (pending) |        (pending) |    (pending)  |

---

## Grad-CAM (post-hoc baseline)

```python
python scripts/train_blackbox.py --config configs/base.yaml --config configs/blackbox.yaml
```

writes Grad-CAM overlays for a few validation images. The implementation targets the last convolutional layer of the encoder (`src/explainability/gradcam.py`).

**IMPORTANT:** Grad-CAM is a *post-hoc saliency* method. It is **not** an intrinsic explanation: it attributes the model's decision to image regions, but it does not describe the decision mechanism, and it can be insensitive to the very weights that determine the prediction. This project uses it only as a baseline against which the *spatially faithful* concept grounding of later phases will be compared.

---

## Repository layout

```
clinically-grounded-cbm/
├── configs/                 # YAML experiment configuration
│   ├── base.yaml            # shared defaults
│   ├── blackbox.yaml        # black-box baseline overrides
│   └── cbm.yaml             # CBM overrides
├── data/
│   ├── raw/                 # (ignored) large data artifacts
│   ├── processed/           # (ignored) derived artifacts
│   └── splits/              # generated train/val/test.csv
├── src/
│   ├── data/                # annotations, preprocessing, transforms, dataset, split
│   ├── models/              # encoder, blackbox, cbm
│   ├── training/            # trainer, losses
│   ├── evaluation/          # classification_metrics, evaluation
│   ├── explainability/      # gradcam
│   └── utils/               # config, seed, logging
├── scripts/
│   ├── prepare_data.py
│   ├── train_blackbox.py
│   ├── train_cbm.py
│   └── evaluate.py
├── tests/                   # pytest suite
├── notebooks/01_data_exploration.ipynb
├── outputs/                 # checkpoints, metrics, predictions, figures, logs
└── reports/research_notes.md
```

---

## Tests

```bash
python -m pytest
```

Covers:

* dataset loading and item structure
* annotation parsing (boxes, global labels)
* concept vector creation (multi-hot, missing findings)
* split reproducibility (seed 42) and test-set integrity
* model forward passes (encoder, black-box, CBM)
* **CBM bottleneck integrity** (no direct encoder → diagnosis path)
* metric calculations (AUROC/AUPRC/F1/precision/recall/sensitivity/specificity)

Tests use a small synthetic VinDr-CXR tree with valid DICOM files — **no real data required**.

---

## Limitations (Phase 1)

* `edema` has no VinDr-CXR annotations and will always be an absent concept (see Dataset setup).
* Concept vectors are derived from bounding-box *presence* only; box coordinates are loaded and preserved but **not** used in training yet.
* The CBM is the standard, non-spatial baseline; there is no grounding loss or intervention yet.
* Grad-CAM is a post-hoc baseline, not an intrinsic explanation.
* Metrics are computed on a fixed 0.5 threshold for precision/recall/specificity (AUROC/AUPRC are threshold-free).
* No data fabrication: all reported numbers come from actually run experiments.

---

## Phase 1 completion criteria

- [x] Dataset loads successfully
- [x] Images visualize correctly (see notebook `notebooks/01_data_exploration.ipynb`)
- [x] Annotations parse correctly
- [x] Train/val/test split works (reproducible, official test untouched)
- [x] Black-box model trains
- [x] Grad-CAM works
- [x] Standard CBM trains
- [x] Concept metrics work
- [x] Diagnosis metrics work
- [x] Tests pass

See `IMPLEMENTATION_PLAN.md` and `reports/research_notes.md` for details.
