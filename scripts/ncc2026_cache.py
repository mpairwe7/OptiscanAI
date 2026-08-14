#!/usr/bin/env python3
"""Pre-decode RFMiD into uint8 arrays so ablation training is not I/O bound.

RFMiD ships multi-megapixel PNGs; decoding them every epoch dominates runtime
when DataLoader worker processes are unavailable. This caches each split once.

    train -> 256x256 (leaves room for RandomResizedCrop to 224)
    val/test -> 224x224 resized directly from the original, which is exactly
                the transform the production checkpoint was evaluated with.

Usage:
    python3 scripts/ncc2026_cache.py
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.ncc2026_infer import SPLITS  # noqa: E402

CACHE = REPO / "outputs/ncc2026/cache"
SIZES = {"train": 256, "val": 224, "test": 224}


def build(split: str) -> None:
    img_dir, labels_csv = SPLITS[split]
    size = SIZES[split]
    df = pd.read_csv(labels_csv, encoding="utf-8-sig")
    ids = [int(i) for i in df["ID"] if (img_dir / f"{int(i)}.png").exists()]
    df = df[df["ID"].isin(ids)].reset_index(drop=True)

    arr = np.zeros((len(ids), size, size, 3), dtype=np.uint8)

    def load(k: int) -> None:
        img = Image.open(img_dir / f"{ids[k]}.png").convert("RGB")
        arr[k] = np.asarray(img.resize((size, size), Image.BICUBIC), dtype=np.uint8)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(load, range(len(ids))))

    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / f"{split}_images_{size}.npy", arr)
    df.to_csv(CACHE / f"{split}_labels.csv", index=False)
    print(f"{split}: {arr.shape} cached in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    for s in ("train", "val", "test"):
        build(s)
