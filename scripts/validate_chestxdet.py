"""Pre-training validation of the remote ChestX-Det dataset adapter.

Streams a deterministic sample (first ``N`` rows) of the remote ``train``
split - without downloading the full dataset - and measures:

* concept availability (via ``configs/chestxdet_concepts.yaml``)
* spatial annotation coverage (masks, label maps, instances)
* mask/label consistency (overlap, disagreement cases)
* label-id-0 and background-255 frequency
* two box-generation strategies:
    A) boxes from mask instances (unique RGB per instance)
    B) boxes/regions from connected components of the class label maps

Writes ``outputs/metrics/chestxdet_validation.json``,
``outputs/metrics/chestxdet_validation.csv`` and
``reports/chestxdet_validation.md``.

Progress is saved incrementally to a JSON-lines file so interrupted runs
resume without re-streaming completed rows.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import chestxdet as cxd  # noqa: E402

REPO = cxd.CHESTX_DET_REPO
TARGET_N = int(os.environ.get("CHESTX_VAL_N", "600"))
WORK_SIZE = 256
MIN_COMPONENT_AREA_FRACTION = 0.0005  # ~33 px at 256x256
MAX_ATTEMPTS = 12
MIN_ACCEPTABLE_N = int(os.environ.get("CHESTX_VAL_MIN", "300"))

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "outputs" / "metrics"
REPORT_DIR = ROOT / "reports"
PROGRESS_FILE = METRICS_DIR / "chestxdet_validation_progress.jsonl"
JSON_OUT = METRICS_DIR / "chestxdet_validation.json"
CSV_OUT = METRICS_DIR / "chestxdet_validation.csv"
MD_OUT = REPORT_DIR / "chestxdet_validation.md"
CONFIG_MAPPING = ROOT / "configs" / "chestxdet_concepts.yaml"

ID2LABEL = cxd.load_id2label(REPO)
STRUCT_8 = np.ones((3, 3), dtype=np.uint8)


def label_components(label_map: np.ndarray, cls: int) -> list:
    """Connected components of one class in the label map -> boxes."""
    binary = label_map == cls
    if not binary.any():
        return []
    lab, n = ndimage.label(binary, structure=STRUCT_8)
    h, w = label_map.shape
    out = []
    for comp_id in range(1, int(n) + 1):
        ys, xs = np.nonzero(lab == comp_id)
        area = int(len(xs))
        x_min, x_max = xs.min() / (w - 1), xs.max() / (w - 1)
        y_min, y_max = ys.min() / (h - 1), ys.max() / (h - 1)
        out.append(
            {
                "class_id": int(cls),
                "class_name": ID2LABEL[cls],
                "x_min": float(x_min),
                "y_min": float(y_min),
                "x_max": float(x_max),
                "y_max": float(y_max),
                "area": float((x_max - x_min) * (y_max - y_min)),
                "area_fraction": float(area / (h * w)),
                "pixels": area,
            }
        )
    return out


def process_row(row: dict) -> dict:
    image_id = row["image_id"]
    img = row["image"]
    label_map = row["label"]
    mask = row["mask"]

    label_fg = (label_map != 255) & (label_map != 0)
    n_label_fg_px = int(label_fg.sum())
    classes_present = sorted(
        int(c) for c in np.unique(label_map[label_fg]) if int(c) in ID2LABEL
    )
    frac0 = float((label_map == 0).mean())
    frac255 = float((label_map == 255).mean())

    mask_fg = mask.any(axis=2)
    mask_present = bool(mask_fg.any())
    n_mask_px = int(mask_fg.sum())
    instances = cxd.instances_from_mask(mask)
    n_instances = len(instances)

    unassigned_instances = 0
    multi_class_instances = 0
    for inst in instances:
        cls = cxd.instance_class_id(inst["mask"], label_map, ID2LABEL)
        if cls is None:
            unassigned_instances += 1
            continue
        vals = np.unique(label_map[inst["mask"]])
        mapped = [int(v) for v in vals if int(v) in ID2LABEL and int(v) != 255]
        if len(mapped) > 1:
            multi_class_instances += 1

    boxes_a = cxd.boxes_from_mask_instances(mask, label_map, image_id, ID2LABEL)
    areas_a = [b.area for b in boxes_a]

    if mask_present and n_label_fg_px > 0:
        inter = int((mask_fg & label_fg).sum())
        overlap = float(inter) / n_mask_px
        coverage = float(inter) / n_label_fg_px
    elif mask_present and n_label_fg_px == 0:
        overlap, coverage = 0.0, 0.0
    else:
        overlap, coverage = None, None

    comps_b: list = []
    for cls in classes_present:
        comps_b.extend(label_components(label_map, cls))
    usable_b = [c for c in comps_b if c["area_fraction"] >= MIN_COMPONENT_AREA_FRACTION]
    areas_b = [c["area"] for c in usable_b]

    return {
        "image_id": image_id,
        "image_mode": img.mode,
        "image_size": list(img.size),
        "classes_present": classes_present,
        "n_classes_in_label": len(classes_present),
        "label_id0_present": bool((label_map == 0).any()),
        "label_id0_fraction": float(frac0),
        "background_fraction": float(frac255),
        "mask_present": mask_present,
        "mask_pixels": n_mask_px,
        "mask_instances": n_instances,
        "unassigned_instances": unassigned_instances,
        "multi_class_instances": multi_class_instances,
        "boxes_A": len(boxes_a),
        "boxes_A_area_mean": float(np.mean(areas_a)) if areas_a else None,
        "boxes_B": len(usable_b),
        "components_B_raw": len(comps_b),
        "boxes_B_area_mean": float(np.mean(areas_b)) if areas_b else None,
        "mask_overlap_label_fg": overlap,
        "label_fg_covered_by_mask": coverage,
        "label_fg_pixels": n_label_fg_px,
    }


def load_progress() -> dict:
    if not PROGRESS_FILE.exists():
        return {}
    records = {}
    for line in PROGRESS_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            records[rec["image_id"]] = rec
    return records


def stream_sample() -> dict:
    records = load_progress()
    for attempt in range(MAX_ATTEMPTS):
        missing = TARGET_N - len(records)
        print(
            f"[attempt {attempt}] already processed {len(records)}/{TARGET_N}",
            flush=True,
        )
        try:
            ds = cxd.ChestXDetRemoteDataset(split="train", image_size=WORK_SIZE)
            seen = 0
            for row in ds.iter_raw():
                image_id = row["image_id"]
                if image_id in records:
                    continue
                if len(records) >= TARGET_N:
                    break
                records[image_id] = process_row(row)
                seen += 1
                if (seen % 50) == 0:
                    print(f"  +{seen} new rows (total {len(records)})", flush=True)
                if len(records) % 100 == 0:
                    flush_progress(records)
            if len(records) >= TARGET_N:
                flush_progress(records)
                return records
        except Exception as exc:
            print(f"  stream error: {type(exc).__name__}: {str(exc)[:100]}", flush=True)
        flush_progress(records)
        time.sleep(3)
    if len(records) >= MIN_ACCEPTABLE_N:
        print(f"WARNING: only {len(records)} samples collected (min {MIN_ACCEPTABLE_N}).")
        flush_progress(records)
        return records
    raise RuntimeError(f"only {len(records)} samples streamed; aborting")


def flush_progress(records: dict) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_FILE.open("w", encoding="utf-8") as fh:
        for rec in records.values():
            fh.write(json.dumps(rec) + "\n")


def mean(vals: list) -> float | None:
    cleaned = []
    for v in vals:
        if v is None:
            continue
        try:
            if v != v:  # NaN from pandas float columns
                continue
        except (TypeError, ValueError):
            pass
        cleaned.append(v)
    return float(np.mean(cleaned)) if cleaned else None


def pct(vals: list) -> float:
    return 100.0 * sum(1 for v in vals if v) / len(vals)


def build_aggregates(records: dict) -> dict:
    df = pd.DataFrame(list(records.values()))
    n = len(df)
    images_with_label = df["n_classes_in_label"] > 0
    images_with_mask = df["mask_present"]
    images_with_boxes_a = df["boxes_A"] > 0
    images_with_boxes_b = df["boxes_B"] > 0
    images_with_id0 = df["label_id0_present"]
    label_but_no_mask = images_with_label & ~images_with_mask
    mask_but_no_label = images_with_mask & ~images_with_label

    class_frequency = {}
    for cls, name in sorted((int(k), v) for k, v in ID2LABEL.items() if int(k) != 255):
        class_frequency[name] = int((df["classes_present"].map(lambda l: cls in l)).sum())

    agg = {
        "dataset": REPO,
        "split": "train",
        "sample_count": int(n),
        "work_image_size": WORK_SIZE,
        "min_component_area_fraction": MIN_COMPONENT_AREA_FRACTION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "image_modes": df["image_mode"].value_counts().to_dict(),
        "image_sizes": {
            str(k): int(v)
            for k, v in df["image_size"].map(tuple).value_counts().items()
        },
        "class_frequency": class_frequency,
        "spatial_coverage": {
            "pct_images_with_nonzero_mask": round(float(pct(images_with_mask)), 2),
            "pct_images_with_usable_label_map": round(float(pct(images_with_label)), 2),
            "instances_per_image": {
                "mean": round(float(df["mask_instances"].mean()), 3),
                "max": int(df["mask_instances"].max()),
                "distribution": df["mask_instances"].value_counts().sort_index().to_dict(),
            },
            "mask_pixels_mean": round(float(df["mask_pixels"].mean()), 1),
            "label_fg_pixels_mean": round(float(df["label_fg_pixels"].mean()), 1),
        },
        "box_strategy": {
            "A_mask_instances": {
                "images_with_boxes": int(images_with_boxes_a.sum()),
                "pct_images_with_boxes": round(float(pct(images_with_boxes_a)), 2),
                "boxes_total": int(df["boxes_A"].sum()),
                "boxes_per_image_mean": round(float(df["boxes_A"].mean()), 3),
                "box_area_mean": round(float(mean(df["boxes_A_area_mean"].tolist())), 4)
                if df["boxes_A"].sum() else None,
            },
            "B_label_components": {
                "images_with_boxes": int(images_with_boxes_b.sum()),
                "pct_images_with_boxes": round(float(pct(images_with_boxes_b)), 2),
                "boxes_total": int(df["boxes_B"].sum()),
                "components_raw_total": int(df["components_B_raw"].sum()),
                "boxes_per_image_mean": round(float(df["boxes_B"].mean()), 3),
                "box_area_mean": round(float(mean(df["boxes_B_area_mean"].tolist())), 4)
                if df["boxes_B"].sum() else None,
            },
        },
        "mask_label_consistency": {
            "mean_mask_overlap_label_fg": (
                round(float(mean(df["mask_overlap_label_fg"].tolist())), 4)
                if mean(df["mask_overlap_label_fg"].tolist()) is not None else None
            ),
            "mean_label_fg_covered_by_mask": (
                round(float(mean(df["label_fg_covered_by_mask"].tolist())), 4)
                if mean(df["label_fg_covered_by_mask"].tolist()) is not None else None
            ),
            "unassigned_mask_instances_total": int(df["unassigned_instances"].sum()),
            "multi_class_mask_instances_total": int(df["multi_class_instances"].sum()),
            "images_with_mask_but_no_label": int(mask_but_no_label.sum()),
            "images_with_label_but_no_mask": int(label_but_no_mask.sum()),
        },
        "label_id0_frequency": {
            "images_with_id0": int(images_with_id0.sum()),
            "pct_images_with_id0": round(float(pct(images_with_id0)), 2),
            "mean_pixel_fraction": round(float(df["label_id0_fraction"].mean()), 5),
        },
        "background_255": {
            "mean_pixel_fraction": round(float(df["background_fraction"].mean()), 5),
            "pct_images_fully_background": round(
                float(100.0 * (df["background_fraction"] == 1.0).mean()), 2
            ),
        },
    }
    return agg


def recommend_strategy(agg: dict) -> str:
    a = agg["box_strategy"]["A_mask_instances"]
    b = agg["box_strategy"]["B_label_components"]
    if b["pct_images_with_boxes"] > a["pct_images_with_boxes"]:
        return "B_label_components"
    if a["pct_images_with_boxes"] > b["pct_images_with_boxes"]:
        return "A_mask_instances"
    if b["boxes_total"] >= a["boxes_total"]:
        return "B_label_components"
    return "A_mask_instances"


def write_csv(records: dict) -> None:
    df = pd.DataFrame(list(records.values()))
    cols = [
        "image_id", "image_mode", "image_size", "n_classes_in_label",
        "classes_present", "label_id0_present", "label_id0_fraction",
        "background_fraction", "mask_present", "mask_pixels",
        "mask_instances", "unassigned_instances", "multi_class_instances",
        "boxes_A", "boxes_A_area_mean", "boxes_B", "components_B_raw",
        "boxes_B_area_mean", "mask_overlap_label_fg",
        "label_fg_covered_by_mask", "label_fg_pixels",
    ]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(CSV_OUT, index=False)
    print(f"wrote {CSV_OUT}")


def write_md(agg: dict, recommendation: str) -> None:
    with open(CONFIG_MAPPING, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    mapping = cfg["concept_mapping"]
    available = list(mapping)
    unavailable = cfg["unavailable_concepts"]

    a = agg["box_strategy"]["A_mask_instances"]
    b = agg["box_strategy"]["B_label_components"]
    consistency = agg["mask_label_consistency"]

    md = f"""# ChestX-Det Pre-Training Validation

Validated the remote `{agg['dataset']}` dataset (split `{agg['split']}`, `{agg['sample_count']}`
deterministic streamed samples, images decoded at `{agg['work_image_size']}x{agg['work_image_size']}`).
No model was trained and no VinDr-CXR pipeline code was modified.

## Concept availability

### Available concepts (mapped)

| Project concept | ChestX-Det source classes |
|---|---|
"""
    for concept in available:
        md += f"| `{concept}` | {', '.join(f'`{s}`' for s in mapping[concept])} |\n"
    md += "\n### Unavailable concepts (no ChestX-Det source)\n\n"
    for concept in unavailable:
        md += f"- `{concept}` (concept vector stays all-zeros)\n"
    md += "\nChestX-Det classes ignored: " + ", ".join(
        f"`{c}`" for c in cfg["ignored_chestx_det_classes"]
    ) + ".\n"

    md += f"""
## Spatial annotation coverage

| Metric | Value |
|---|---|
| Samples with non-zero mask | {agg['spatial_coverage']['pct_images_with_nonzero_mask']:.2f}% |
| Samples with usable label map (>=1 disease-class pixel) | {agg['spatial_coverage']['pct_images_with_usable_label_map']:.2f}% |
| Mask instances per image (mean / max) | {agg['spatial_coverage']['instances_per_image']['mean']} / {agg['spatial_coverage']['instances_per_image']['max']} |
| Mean mask pixels per image | {agg['spatial_coverage']['mask_pixels_mean']} |
| Mean disease-class pixels per image | {agg['spatial_coverage']['label_fg_pixels_mean']} |

Instances-per-image distribution: `{agg['spatial_coverage']['instances_per_image']['distribution']}`

## Box-generation strategy comparison

| Strategy | Images with >=1 box | % of samples | Total boxes | Boxes/image (mean) |
|---|---|---|---|---|
| A: boxes from mask instances | {a['images_with_boxes']} | {a['pct_images_with_boxes']:.2f}% | {a['boxes_total']} | {a['boxes_per_image_mean']} |
| B: connected components of label maps | {b['images_with_boxes']} | {b['pct_images_with_boxes']:.2f}% | {b['boxes_total']} | {b['boxes_per_image_mean']} |

Box area (normalized, mean): A = `{a['box_area_mean']}`, B = `{b['box_area_mean']}`.
Strategy B kept components with area fraction >= `{agg['min_component_area_fraction']}` ({agg['min_component_area_fraction']*100:.3f}% of the image).

**Recommendation: strategy `{recommendation}`.**

## Mask / label consistency

| Metric | Value |
|---|---|
| Mean fraction of mask pixels overlapping label foreground | {consistency['mean_mask_overlap_label_fg']} |
| Mean fraction of label foreground covered by masks | {consistency['mean_label_fg_covered_by_mask']} |
| Mask instances with no assignable label class | {consistency['unassigned_mask_instances_total']} |
| Mask instances overlapping multiple label classes | {consistency['multi_class_mask_instances_total']} |
| Images with a mask but no label foreground | {consistency['images_with_mask_but_no_label']} |
| Images with label foreground but no mask | {consistency['images_with_label_but_no_mask']} |

## Label id 0 and background 255

- Label id `0` (unmapped in `id2label.json`): present in {agg['label_id0_frequency']['images_with_id0']} images
  ({agg['label_id0_frequency']['pct_images_with_id0']:.2f}%), mean pixel fraction
  {agg['label_id0_frequency']['mean_pixel_fraction']}.
- Background `255`: mean pixel fraction {agg['background_255']['mean_pixel_fraction']};
  fully-background images {agg['background_255']['pct_images_fully_background']:.2f}%.

## Limitations of using ChestX-Det for this project

- Only {len(available)} of 8 project concepts are available; `lung_opacity` and `edema` have no
  ChestX-Det source and remain all-zeros.
- ChestX-Det has no independent global diagnosis labels: diagnosis vectors must be derived from
  pixel findings, so they are not comparable to VinDr-CXR radiologist global labels.
- Spatial annotations are sparse and partially inconsistent: masks and per-pixel label maps are
  not fully co-registered (see consistency table), and many images carry no mask at all.
- The `mask` and `label` fields can disagree; boxes derived from masks miss instances whose pixels
  only overlap background.
- ChestX-Det is a subset of NIH ChestX-ray14 (13 classes), not VinDr-CXR; it is not a drop-in
  replacement and any cross-dataset comparison must be clearly qualified.
"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MD_OUT.write_text(md, encoding="utf-8")
    print(f"wrote {MD_OUT}")


def main() -> int:
    print(f"validating {REPO} train split (target {TARGET_N} samples)", flush=True)
    records = stream_sample()
    if len(records) < MIN_ACCEPTABLE_N:
        raise RuntimeError(f"insufficient samples ({len(records)}) for validation")

    records = {k: records[k] for k in sorted(records)}
    agg = build_aggregates(records)
    recommendation = recommend_strategy(agg)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"wrote {JSON_OUT}")
    write_csv(records)
    write_md(agg, recommendation)
    print(f"\n=== validation done: {len(records)} samples, recommended box strategy = {recommendation} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
