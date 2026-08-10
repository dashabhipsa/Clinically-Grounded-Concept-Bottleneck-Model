# ChestX-Det Pre-Training Validation

Validated the remote `natealberti/ChestX-Det` dataset (split `train`, `600`
deterministic streamed samples, images decoded at `256x256`).
No model was trained and no VinDr-CXR pipeline code was modified.

## Concept availability

### Available concepts (mapped)

| Project concept | ChestX-Det source classes |
|---|---|
| `atelectasis` | `Atelectasis` |
| `cardiomegaly` | `Cardiomegaly` |
| `consolidation` | `Consolidation` |
| `pleural_effusion` | `Effusion` |
| `nodule_mass` | `Mass`, `Nodule` |
| `pneumothorax` | `Pneumothorax` |

### Unavailable concepts (no ChestX-Det source)

- `lung_opacity` (concept vector stays all-zeros)
- `edema` (concept vector stays all-zeros)

ChestX-Det classes ignored: `Calcification`, `Diffuse Nodule`, `Emphysema`, `Fibrosis`, `Fracture`, `Pleural Thickening`.

## Spatial annotation coverage

| Metric | Value |
|---|---|
| Samples with non-zero mask | 81.83% |
| Samples with usable label map (>=1 disease-class pixel) | 78.67% |
| Mask instances per image (mean / max) | 1.662 / 6 |
| Mean mask pixels per image | 4661.3 |
| Mean disease-class pixels per image | 4738.5 |

Instances-per-image distribution: `{0: 109, 1: 166, 2: 191, 3: 95, 4: 32, 5: 6, 6: 1}`

## Box-generation strategy comparison

| Strategy | Images with >=1 box | % of samples | Total boxes | Boxes/image (mean) |
|---|---|---|---|---|
| A: boxes from mask instances | 266 | 44.33% | 408 | 0.68 |
| B: connected components of label maps | 463 | 77.17% | 1301 | 2.168 |

Box area (normalized, mean): A = `0.1631`, B = `0.055`.
Strategy B kept components with area fraction >= `0.0005` (0.050% of the image).

**Recommendation: strategy `B_label_components`.**

## Mask / label consistency

| Metric | Value |
|---|---|
| Mean fraction of mask pixels overlapping label foreground | 0.1762 |
| Mean fraction of label foreground covered by masks | 0.1655 |
| Mask instances with no assignable label class | 589 |
| Mask instances overlapping multiple label classes | 117 |
| Images with a mask but no label foreground | 108 |
| Images with label foreground but no mask | 89 |

## Label id 0 and background 255

- Label id `0` (unmapped in `id2label.json`): present in 56 images
  (9.33%), mean pixel fraction
  0.00141.
- Background `255`: mean pixel fraction 0.92629;
  fully-background images 18.50%.

## Limitations of using ChestX-Det for this project

- Only 6 of 8 project concepts are available; `lung_opacity` and `edema` have no
  ChestX-Det source and remain all-zeros.
- ChestX-Det has no independent global diagnosis labels: diagnosis vectors must be derived from
  pixel findings, so they are not comparable to VinDr-CXR radiologist global labels.
- Spatial annotations are sparse and partially inconsistent: masks and per-pixel label maps are
  not fully co-registered (see consistency table), and many images carry no mask at all.
- The `mask` and `label` fields can disagree; boxes derived from masks miss instances whose pixels
  only overlap background.
- ChestX-Det is a subset of NIH ChestX-ray14 (13 classes), not VinDr-CXR; it is not a drop-in
  replacement and any cross-dataset comparison must be clearly qualified.
