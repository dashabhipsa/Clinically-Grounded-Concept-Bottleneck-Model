"""Pre-download the ChestX-Det parquet files to local disk.

The HF streaming pipeline (`load_dataset(..., streaming=True)`) is unreliable
on slow connections and keeps timing out on the large image parquet. This
script downloads the parquet shards once with `hf_hub_download`, which is
resumable and much more robust against flaky networks, so training can read
them from disk afterwards.

Usage:
    python scripts/download_chestxdet.py [--data-cache .cache/chestxdet]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# D:-drive-only caches, before importing huggingface_hub.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CACHE_ROOT = PROJECT_ROOT / ".cache"
for _sub in ("hf", "datasets", "hub", "torch", "tmp"):
    (_CACHE_ROOT / _sub).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_CACHE_ROOT / "hf"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE_ROOT / "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_CACHE_ROOT / "hub"))
os.environ["TMP"] = str(_CACHE_ROOT / "tmp")
os.environ["TEMP"] = str(_CACHE_ROOT / "tmp")
os.environ["TMPDIR"] = str(_CACHE_ROOT / "tmp")

import huggingface_hub  # noqa: E402

from huggingface_hub import constants as hf_constants  # noqa: E402

hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = 300

# The new Xet/CAS download backend keeps failing ("error decoding response
# body") on this connection. Force the classic HTTP downloader instead.
os.environ["HF_HUB_DISABLE_XET"] = "1"

REPO = "natealberti/ChestX-Det"
# Only the shard we need (train-00000 covers rows 0..~1008 -> our 500 train +
# 100 val slice) plus the single test shard. Downloads ~644 MB total instead
# of the full ~1.4 GB.
FILES = [
    "data/train-00000-of-00003.parquet",
    "data/test-00000-of-00001.parquet",
    "id2label.json",
]


def download_one(repo_id: str, filename: str, dest_dir: Path, attempts: int = 5):
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / filename
    if target.exists() and target.stat().st_size > 0:
        print(f"[skip] {filename} already present ({target.stat().st_size} bytes)")
        return target
    last = None
    for i in range(attempts):
        try:
            print(f"[{i + 1}/{attempts}] downloading {filename} ...", flush=True)
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=dest_dir,
            )
            print(f"[done] {filename} -> {path}", flush=True)
            return Path(path)
        except Exception as exc:
            last = exc
            print(f"  attempt {i + 1} failed: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            time.sleep(min(30, 5 * (i + 1)))
    raise RuntimeError(f"failed to download {filename}: {last}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-download ChestX-Det to disk.")
    parser.add_argument("--data-cache", default=str(_CACHE_ROOT / "chestxdet"))
    args = parser.parse_args()

    dest = Path(args.data_cache)
    for f in FILES:
        download_one(REPO, f, dest)
    print(f"\nAll files under {dest}:")
    for p in sorted(dest.rglob("*")):
        if p.is_file():
            print(f"  {p.stat().st_size:>12,}  {p.relative_to(dest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())