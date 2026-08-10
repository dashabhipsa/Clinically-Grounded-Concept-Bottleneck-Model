"""Sanity check: exercise the ChestX-Det remote adapter against HF.

Streams a small number of samples (no full download), verifies the decoded
image/label/mask/boxes/concepts, and prints exactly what the remote dataset
provides. Output is also written to a JSON file.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import chestxdet as cxd  # noqa: E402

OUT = Path(r"C:\Users\HP\AppData\Local\Temp\opencode\chestxdet_sanity.json")
N_SAMPLES = 8


def stream_rows(n: int):
    ds = cxd.ChestXDetRemoteDataset(split="train", image_size=256, max_samples=n)
    for attempt in range(8):
        try:
            return list(ds.iter_raw())
        except Exception as exc:  # flaky network
            print(f"[retry {attempt}] {type(exc).__name__}: {str(exc)[:100]}", flush=True)
            time.sleep(2)
    raise RuntimeError("could not stream remote dataset")


def main() -> int:
    id2label = cxd.load_id2label()
    classes = cxd.chestx_det_classes()
    print("id2label:", id2label)
    print("chestx_det_classes:", classes)

    rows = stream_rows(N_SAMPLES)
    print(f"\nstreamed {len(rows)} raw rows")

    report = {
        "id2label": {str(k): v for k, v in id2label.items()},
        "classes": classes,
        "samples": [],
    }

    n_with_mask = 0
    n_with_boxes = 0
    mask_instances = []

    for r in rows:
        img, label_map, mask = r["image"], r["label"], r["mask"]
        instances = cxd.instances_from_mask(mask)
        boxes = cxd.boxes_from_mask_instances(mask, label_map, r["image_id"], id2label)
        classes_in_label = sorted(int(v) for v in np.unique(label_map) if int(v) != 255)
        if instances:
            n_with_mask += 1
        if boxes:
            n_with_boxes += 1
        mask_instances.append(len(instances))

        print(f"\n--- {r['image_id']} (img {img.size}, mode {img.mode}) ---")
        print(f"  label classes present: {classes_in_label}")
        print(f"  mask instances: {len(instances)}; derived boxes: {len(boxes)}")
        for b in boxes:
            print(f"    {b.canonical_class:20s} x[{b.x_min:.3f},{b.x_max:.3f}] y[{b.y_min:.3f},{b.y_max:.3f}]")

        report["samples"].append(
            {
                "image_id": r["image_id"],
                "image_size": list(img.size),
                "image_mode": img.mode,
                "label_classes": classes_in_label,
                "mask_instances": len(instances),
                "boxes": [
                    {
                        "class": b.canonical_class,
                        "class_name": b.class_name,
                        "class_id": b.class_id,
                        "x_min": b.x_min,
                        "x_max": b.x_max,
                        "y_min": b.y_min,
                        "y_max": b.y_max,
                    }
                    for b in boxes
                ],
            }
        )

    print("\n================ SANITY SUMMARY ================")
    print(f"samples inspected : {len(rows)}")
    print(f"images with masks : {n_with_mask}/{len(rows)}")
    print(f"images with boxes : {n_with_boxes}/{len(rows)}")
    print(f"instances per img : {mask_instances}")
    report["summary"] = {
        "samples_inspected": len(rows),
        "images_with_masks": n_with_mask,
        "images_with_boxes": n_with_boxes,
        "instances_per_image": mask_instances,
    }

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
