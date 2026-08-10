# Research Notes — Phase 1

Working notes for **"Beyond Concept Prediction: Clinically Grounded and Spatially Faithful Concept Bottlenecks for Chest X-ray Diagnosis"**.

## 1. Problem framing

Standard CBMs (Koh et al., 2020) decompose a prediction into:

```
image -> concepts (bottleneck) -> target
```

Interpretability comes from the concept bottleneck, but the concept predictor is trained on *global* features. Two failure modes motivate later phases:

1. **Clinical grounding**: concepts must be recognizable, standard findings — not arbitrary latent factors.
2. **Spatial faithfulness**: the image evidence driving a concept must be located where the finding actually is.

Phase 1 provides the evaluation substrate and both baselines (black-box + standard CBM + Grad-CAM) that Phase 2/3 improvements are measured against.

## 2. Dataset notes

- VinDr-CXR v1.0.0 (PhysioNet): 18,000 train / 3,000 test frontal chest X-rays (DICOM), radiologist-annotated.
- 14 finding classes; bounding boxes normalized to [0, 1]; global labels per image.
- This repo does **not** auto-download restricted data; the user provides `data.root`.
- Validation is carved out of the official training set; the official test set is never touched.

## 3. Concept / diagnosis definitions

| Layer | Source | Default classes | Phase 1 use |
| ----- | ------ | --------------- | ----------- |
| **Concepts** | local findings (boxes) | 8 findings (configurable) | concept bottleneck, trained with BCE |
| **Diagnosis** | global labels | 14 VinDr classes | diagnosis head target |

**Assumption:** in VinDr-CXR the global labels and local findings share the same class vocabulary. We keep them conceptually distinct: concepts are the compressed, interpretable 8-dim bottleneck; diagnoses are the 14-class global targets. This deliberately forces the diagnosis head to work through the concept bottleneck.

**Known gap:** `edema` is not a VinDr-CXR class. The default concept list includes it (all-zero supervision). Two options: drop it from `concepts` in `configs/base.yaml`, or keep it to exercise missing-finding handling. Documented in README.

## 4. Architecture decisions

- Encoder: DenseNet-121 (ImageNet-pretrained) via a registry (`densenet121/169/201`, `resnet50/101`). Adaptive pooling makes input size flexible (default 512).
- **Black-box**: `encoder -> Linear(1024, 14)`.
- **Standard CBM**: `encoder -> Linear(1024, k)` (concept logits) `-> sigmoid(z) -> Linear(k, 14)`.
  - `k = 8` concepts; diagnosis head `in_features == k`, so no encoder feature can reach diagnosis directly.
  - CBM trained end-to-end with `lambda_c * BCE(concepts) + lambda_d * BCE(diagnosis)` (both 1.0 by default).
  - `forward(concepts=...)` substitutes ground-truth concepts at the bottleneck — unused in Phase 1, needed for Phase 3 intervention.
- Multi-label losses: `BCEWithLogitsLoss`.

## 5. Metric conventions

- AUROC / AUPRC are threshold-free (sklearn).
- F1 / precision / recall / sensitivity / specificity use decision threshold **0.5** on sigmoid probabilities.
- Sensitivity == recall by definition; both reported.
- Macro averages are NaN-mean over classes: a class with no positive samples contributes nothing to the macro instead of zeroing it.
- Per-class + macro reported for diagnoses (all models) and concepts (CBM only).

## 6. Grad-CAM caveat

Grad-CAM (Selvaraju et al., 2017) is a post-hoc gradient-weighted activation map. It is a *saliency* baseline, not an intrinsic explanation:

- it ignores weights downstream of the target layer (including most of the classifier),
- regions can look salient while being causally irrelevant,
- it has no concept semantics (a single map per class, not per concept).

Phase 2 replaces it with intrinsically grounded, concept-level spatial evidence.

## 7. Split reproducibility

- Official train ids := union of ids in `annotations/train.csv` and `image_labels_train.csv` (robust to images without boxes).
- `np.random.RandomState(seed=42)`: permutation of official train ids → val (default 10%) / train (90%).
- Official test ids := union of ids in `annotations/test.csv` and `image_labels_test.csv`.
- Split CSVs stored at `data/splits/{train,val,test}.csv`; regeneration requires `--overwrite`.

## 8. Known limitations & assumptions (Phase 1)

1. Concept vectors use box **presence** only; box geometry is loaded and preserved but unused in training.
2. `edema` never positive (no annotations).
3. No spatial supervision; no grounding loss; no intervention; no corruption; no dashboard.
4. Training at `image_size=512` with ImageNet-pretrained backbones: domain shift from ImageNet is expected and untested for accuracy until a real run.
5. Metrics on a fixed threshold; calibration (temperature/platt) not applied.
6. All results in README/notebook must come from real runs; placeholders are marked `(pending)` / `N/A`.

## 9. What Phase 1 makes possible

- A repeatable baseline comparison (black-box vs standard CBM) on diagnosis AUROC/AUPRC.
- A concept-quality baseline (per-concept AUROC etc.) for the standard CBM.
- Grad-CAM examples as the post-hoc baseline for later spatial-faithfulness comparisons.
- A test suite that guards the CBM bottleneck invariant and split reproducibility.
