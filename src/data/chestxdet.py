"""Remote ChestX-Det dataset adapter (Hugging Face Datasets).

ChestX-Det (``natealberti/ChestX-Det``) is a subset of NIH ChestX-ray14 with
~3500 images and 13 disease classes annotated by three radiologists. Each
row of the parquet-backed dataset provides:

* ``image``  - chest X-ray, 1024x1024 PIL image (mode ``L`` or ``RGBA``),
* ``label``  - 16-bit per-pixel class-id map (``255`` == background; ids
  ``1..13`` map to disease classes via ``id2label.json``; a raw id ``0``
  appears in a few images but is not present in ``id2label.json``),
* ``mask``   - RGB per-instance segmentation; each annotated instance is
  assigned a unique RGB triplet.

The adapter decodes these remote rows into the per-sample dict layout the
existing pipeline consumes (``image`` / ``concepts`` / ``diagnosis`` /
``boxes`` / ``image_id``), plus a per-class binary segmentation ``mask`` so
the existing spatial machinery can consume the spatial annotations.

ChestX-Det has *no independent global diagnostic labels*: the ``diagnosis``
vector is derived from the pixel-level findings (identical to ``concepts``),
which is not equivalent to VinDr-CXR's radiologist global labels.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import IterableDataset

from src.data.annotations import Box, normalize_class_name
from src.data.transforms import build_transform

CHESTX_DET_REPO = "natealberti/ChestX-Det"
BACKGROUND_ID = 255

# From the dataset card (splits in the parquet files).
SPLIT_SIZES = {"train": 3025, "test": 553}

_IMAGE_MEAN = [0.485, 0.456, 0.406]
_IMAGE_STD = [0.229, 0.224, 0.225]


@lru_cache(maxsize=1)
def load_id2label(repo_id: str = CHESTX_DET_REPO) -> Dict[int, str]:
    """Load and cache the remote ``id2label.json`` class map."""
    path = hf_hub_download(repo_id, "id2label.json", repo_type="dataset")
    raw = Path(path).read_text(encoding="utf-8")
    return {int(k): str(v) for k, v in __import__("json").loads(raw).items()}


def chestx_det_classes(repo_id: str = CHESTX_DET_REPO) -> List[str]:
    """Canonical (normalized) names of the 13 ChestX-Det disease classes."""
    id2label = load_id2label(repo_id)
    ids = sorted(i for i in id2label if i != BACKGROUND_ID)
    return [normalize_class_name(id2label[i]) for i in ids]


def _nearest_resize(arr: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize preserving integer class/instance ids."""
    if arr.ndim == 3:
        pil = Image.fromarray(arr)  # uint8 RGB mask
        pil = pil.resize((int(size), int(size)), resample=Image.NEAREST)
        return np.asarray(pil)
    pil = Image.fromarray(arr.astype(np.int32)).convert("I")  # class-id map
    pil = pil.resize((int(size), int(size)), resample=Image.NEAREST)
    return np.asarray(pil).astype(arr.dtype)


def image_to_grayscale(pil_img: Image.Image) -> Image.Image:
    """Normalize ChestX-Det images (L or RGBA) to 8-bit grayscale 'L'."""
    if pil_img.mode == "RGBA":
        background = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
        pil_img = Image.alpha_composite(background, pil_img)
    return pil_img.convert("L")


def decode_label_map(pil_label: Image.Image, size: Optional[int] = None) -> np.ndarray:
    """Decode the 16-bit per-pixel class-id map to a uint16 ``[H, W]`` array."""
    arr = np.asarray(pil_label)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.uint16)
    if size is not None:
        arr = _nearest_resize(arr, size)
    return arr


def decode_mask(pil_mask: Image.Image, size: Optional[int] = None) -> np.ndarray:
    """Decode the RGB per-instance mask to a uint8 ``[H, W, 3]`` array."""
    arr = np.asarray(pil_mask.convert("RGB"))
    if size is not None:
        arr = _nearest_resize(arr, size)
    return arr


def label_map_to_multihot(
    label_map: np.ndarray, class_ids: List[object]
) -> np.ndarray:
    """Multi-hot presence vector ``[K]`` from the per-pixel class-id map.

    Each entry of ``class_ids`` may be a single id, ``None`` (a project
    concept with no ChestX-Det source -> always absent), or a list of ids
    (e.g. ``nodule_mass`` <- ``Mass`` + ``Nodule``; present if *any* id
    appears in the label map).
    """
    present = set(np.unique(label_map).tolist())
    out = []
    for c in class_ids:
        if isinstance(c, (list, tuple)):
            hit = any(int(x) in present for x in c if x is not None)
        else:
            hit = c is not None and int(c) in present
        out.append(1.0 if hit else 0.0)
    return np.array(out, dtype=np.float32)


def class_masks_from_label(
    label_map: np.ndarray, class_ids: List[object]
) -> np.ndarray:
    """Per-class binary segmentation masks ``[K, H, W]`` from the id map.

    ``class_ids`` entries follow the same convention as
    :func:`label_map_to_multihot` (single id / ``None`` / list of ids).
    """
    h, w = label_map.shape
    masks = np.zeros((len(class_ids), int(h), int(w)), dtype=np.float32)
    for k, c in enumerate(class_ids):
        if isinstance(c, (list, tuple)):
            ids = [x for x in c if x is not None]
            if ids:
                masks[k] = np.isin(label_map, ids)
        elif c is not None:
            masks[k] = label_map == c
    return masks


def instances_from_mask(mask: np.ndarray) -> List[Dict]:
    """Split the RGB mask into per-instance binary masks.

    Every annotated instance owns a unique RGB triplet, so each non-black
    color is treated as one instance.
    """
    flat = mask.reshape(-1, 3)
    colors = np.unique(flat, axis=0)
    instances: List[Dict] = []
    for color in colors:
        if not color.any():
            continue
        inst = (mask[:, :, 0] == color[0]) & (
            mask[:, :, 1] == color[1]
        ) & (mask[:, :, 2] == color[2])
        instances.append(
            {"rgb": tuple(int(x) for x in color), "mask": inst}
        )
    return instances


def instance_class_id(
    instance_mask: np.ndarray, label_map: np.ndarray, id2label: Dict[int, str]
) -> Optional[int]:
    """Majority disease-class id covered by an instance's pixels.

    Background (``255``) is excluded; unmapped ids (e.g. ``0``) are only
    used as a fallback when the instance overlaps no mapped class.
    """
    vals, counts = np.unique(label_map[instance_mask], return_counts=True)
    mapped = [
        (int(v), int(c)) for v, c in zip(vals, counts)
        if int(v) != BACKGROUND_ID and int(v) in id2label
    ]
    if not mapped:
        return None
    return max(mapped, key=lambda pair: pair[1])[0]


def boxes_from_mask_instances(
    mask: np.ndarray,
    label_map: np.ndarray,
    image_id: str,
    id2label: Dict[int, str],
) -> List[Box]:
    """Derive normalized ``Box`` objects from the instance mask.

    Each instance (unique RGB color) yields one box: the tight bounding box
    of its pixels, normalized to ``[0, 1]`` like VinDr-CXR, with the class
    assigned by majority vote over the pixel-level label map.
    """
    h, w = label_map.shape
    boxes: List[Box] = []
    for inst in instances_from_mask(mask):
        cid = instance_class_id(inst["mask"], label_map, id2label)
        if cid is None or cid == BACKGROUND_ID:
            continue
        ys, xs = np.nonzero(inst["mask"])
        boxes.append(
            Box(
                image_id=image_id,
                class_id=int(cid),
                class_name=id2label[cid],
                x_min=float(xs.min()) / (w - 1),
                y_min=float(ys.min()) / (h - 1),
                x_max=float(xs.max()) / (w - 1),
                y_max=float(ys.max()) / (h - 1),
                rad_id=0,
            )
        )
    return boxes


def boxes_from_label_components(
    label_map: np.ndarray,
    image_id: str,
    id2label: Dict[int, str],
    id_to_concept: Optional[Dict[int, str]] = None,
    min_area_fraction: float = 0.0005,
    structure: Optional[np.ndarray] = None,
) -> List[Box]:
    """Derive normalized ``Box`` objects from connected components of the
    per-class label maps (Strategy B, selected in the pre-training validation).

    Each disease class present in ``label_map`` (excluding background ``255``
    and unmapped id ``0``) is binarized and split into connected components;
    every component whose pixel area fraction is >= ``min_area_fraction``
    yields one tight bounding box, normalized to ``[0, 1]`` like VinDr-CXR.

    ``class_name`` is the mapped project concept name when the class has a
    mapping (``id_to_concept``), otherwise the raw ChestX-Det class name, so
    downstream code that matches boxes to ``concept_names`` by canonical name
    works for the mapped concepts.
    """
    from scipy import ndimage

    h, w = label_map.shape
    structure = np.ones((3, 3), dtype=np.uint8) if structure is None else structure
    id_to_concept = id_to_concept or {}
    boxes: List[Box] = []
    for cls in np.unique(label_map).tolist():
        cls = int(cls)
        if cls in (BACKGROUND_ID, 0) or cls not in id2label:
            continue
        lab, n = ndimage.label(label_map == cls, structure=structure)
        for comp in range(1, int(n) + 1):
            ys, xs = np.nonzero(lab == comp)
            area = int(len(xs))
            if area / (h * w) < float(min_area_fraction):
                continue
            boxes.append(
                Box(
                    image_id=image_id,
                    class_id=cls,
                    class_name=id_to_concept.get(cls, id2label[cls]),
                    x_min=float(xs.min()) / (w - 1),
                    y_min=float(ys.min()) / (h - 1),
                    x_max=float(xs.max()) / (w - 1),
                    y_max=float(ys.max()) / (h - 1),
                    rad_id=0,
                )
            )
    return boxes


class ChestXDetRemoteDataset(IterableDataset):
    """Stream a split of the remote ChestX-Det dataset.

    Each yielded sample is a dict compatible with the existing pipeline:

    * ``image``     - float tensor ``[3, H, W]`` (normalized),
    * ``concepts``  - float tensor ``[K]``  (presence per requested concept),
    * ``diagnosis`` - float tensor ``[K]`` (same as concepts; ChestX-Det has
      no independent global labels),
    * ``boxes``     - list[Box] with normalized coordinates, derived from the
      per-instance RGB mask,
    * ``mask``      - float tensor ``[K, H, W]`` per-class segmentation,
    * ``image_id``  - ``"<split>-<index>"``.

    ``concept_names`` selects/orders the concept columns. Matching uses the
    canonical name, or the ``concept_mapping`` dict (project concept -> list
    of ChestX-Det class names, as in ``configs/chestxdet_concepts.yaml``)
    when provided; concepts with no ChestX-Det source stay all-zero.
    """

    def __init__(
        self,
        repo_id: str = CHESTX_DET_REPO,
        split: str = "train",
        image_size: int = 512,
        transform=None,
        concept_names: Optional[List[str]] = None,
        diagnosis_names: Optional[List[str]] = None,
        concept_mapping: Optional[Dict[str, List[str]]] = None,
        box_strategy: str = "B_label_components",
        min_component_area_fraction: float = 0.0005,
        streaming: bool = True,
        max_samples: Optional[int] = None,
        offset: int = 0,
    ):
        super().__init__()
        if split not in SPLIT_SIZES:
            raise ValueError(f"unknown split '{split}'; expected {sorted(SPLIT_SIZES)}")
        if box_strategy not in ("B_label_components", "A_mask_instances"):
            raise ValueError(
                f"unknown box_strategy '{box_strategy}'; choose from "
                f"'B_label_components', 'A_mask_instances'"
            )
        self.repo_id = repo_id
        self.split = split
        self.image_size = int(image_size)
        self.transform = transform
        self.streaming = streaming
        self.max_samples = max_samples
        self.offset = int(offset)
        self.box_strategy = box_strategy
        self.min_component_area_fraction = float(min_component_area_fraction)

        self.id2label = load_id2label(repo_id)
        self.class_ids = sorted(i for i in self.id2label if i != BACKGROUND_ID)
        all_by_name = {
            normalize_class_name(self.id2label[i]): i for i in self.class_ids
        }
        if concept_names is None:
            concept_names = [
                normalize_class_name(self.id2label[i]) for i in self.class_ids
            ]
        self.concept_names = list(concept_names)

        # project concept -> list of ChestX-Det class ids ([] if unmapped)
        self.concept_to_ids: Dict[str, List[int]] = {}
        for concept in self.concept_names:
            if concept_mapping:
                sources = concept_mapping.get(concept) or []
                ids = [
                    all_by_name[normalize_class_name(s)]
                    for s in sources
                    if normalize_class_name(s) in all_by_name
                ]
            else:
                cid = all_by_name.get(concept)
                ids = [cid] if cid is not None else []
            self.concept_to_ids[concept] = ids

        self.id_to_concept: Dict[int, str] = {}
        for concept, cids in self.concept_to_ids.items():
            for cid in cids:
                self.id_to_concept.setdefault(int(cid), concept)

        self.diagnosis_names = (
            self.concept_names if diagnosis_names is None else list(diagnosis_names)
        )

    def __len__(self) -> int:
        if self.max_samples is not None:
            return int(self.max_samples)
        return SPLIT_SIZES[self.split] - int(self.offset)

    def _default_transform(self):
        return build_transform(
            self.image_size, _IMAGE_MEAN, _IMAGE_STD, train=False, augment=False
        )

    def decode_row(
        self, row: dict, image_id: str, transform
    ) -> dict:
        """Decode one remote row into the pipeline-compatible sample dict."""
        img = image_to_grayscale(row["image"])
        image = transform(img)

        label_map = decode_label_map(row["label"], size=self.image_size)
        mask = decode_mask(row["mask"], size=self.image_size)

        id_lists = [self.concept_to_ids[name] for name in self.concept_names]
        concepts = label_map_to_multihot(label_map, id_lists)
        diagnosis = label_map_to_multihot(label_map, id_lists)

        boxes = self._boxes_for(label_map, mask, image_id)
        class_masks = class_masks_from_label(label_map, id_lists)

        return {
            "image": image,
            "concepts": torch.as_tensor(concepts, dtype=torch.float32),
            "diagnosis": torch.as_tensor(diagnosis, dtype=torch.float32),
            "boxes": boxes,
            "mask": torch.as_tensor(class_masks, dtype=torch.float32),
            "image_id": image_id,
        }

    def _boxes_for(self, label_map: np.ndarray, mask: np.ndarray, image_id: str):
        """Boxes for one sample using the configured strategy."""
        if self.box_strategy == "B_label_components":
            return boxes_from_label_components(
                label_map, image_id, self.id2label,
                id_to_concept=self.id_to_concept,
                min_area_fraction=self.min_component_area_fraction,
            )
        return boxes_from_mask_instances(mask, label_map, image_id, self.id2label)

    def _local_parquet(self):
        """Path to locally pre-downloaded parquet for this split, if any.

        Uses ``PROJECT_ROOT/.cache/chestxdet/data/`` populated by
        ``scripts/download_chestxdet.py``. Loading from disk avoids the flaky
        HF network streaming entirely.
        """
        project_root = Path(__file__).resolve().parents[2]
        cache_dir = project_root / ".cache" / "chestxdet" / "data"
        if self.split == "train":
            candidate = cache_dir / "train-00000-of-00003.parquet"
        else:
            candidate = cache_dir / "test-00000-of-00001.parquet"
        return candidate if candidate.exists() else None

    def __iter__(self) -> Iterator[dict]:
        local = self._local_parquet()
        if local is not None:
            ds = load_dataset("parquet", data_files=str(local), split="train", streaming=False)
        else:
            ds = load_dataset(self.repo_id, split=self.split, streaming=self.streaming)
        transform = self._default_transform() if self.transform is None else self.transform
        count = 0
        for row in ds:
            if count < self.offset:
                count += 1
                continue
            if self.max_samples is not None and (count - self.offset) >= self.max_samples:
                break
            yield self.decode_row(row, f"{self.split}-{count:05d}", transform)
            count += 1

    def iter_raw(self) -> Iterator[dict]:
        """Yield raw decoded arrays for inspection (no image transform)."""
        local = self._local_parquet()
        if local is not None:
            ds = load_dataset("parquet", data_files=str(local), split="train", streaming=False)
        else:
            ds = load_dataset(self.repo_id, split=self.split, streaming=self.streaming)
        count = 0
        for row in ds:
            if self.max_samples is not None and count >= self.max_samples:
                break
            label_map = decode_label_map(row["label"], size=self.image_size)
            mask = decode_mask(row["mask"], size=self.image_size)
            yield {
                "image": row["image"],
                "label": label_map,
                "mask": mask,
                "image_id": f"{self.split}-{count:05d}",
            }
            count += 1


def chestx_det_collate(batch: List[dict]) -> dict:
    """Collate preserving variable-length boxes and string ids."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "concepts": torch.stack([b["concepts"] for b in batch]),
        "diagnosis": torch.stack([b["diagnosis"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "mask": torch.stack([b["mask"] for b in batch]),
        "image_id": [b["image_id"] for b in batch],
    }
