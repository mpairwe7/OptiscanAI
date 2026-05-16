#!/usr/bin/env python3
"""
Download pretrained ViT-Small weights for the MultiResolutionEncoder.
Tries PyTorch Hub first, then timm GitHub releases.

Usage:
    python3 scripts/download_pretrained_weights.py
"""
import sys
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path("pretrained_weights")

SOURCES = [
    {
        "name": "ViT-Small (PyTorch Hub)",
        "url": "https://download.pytorch.org/models/vit_small_patch16_224-15ec54c9.pth",
        "filename": "vit_small_patch16_224.pth",
    },
    {
        "name": "ViT-Small (timm GitHub Release)",
        "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_patch16_224-15ec54c9.pth",
        "filename": "vit_small_patch16_224-15ec54c9.pth",
    },
]


def download_progress(count, block_size, total_size):
    pct = int(count * block_size * 100 / max(total_size, 1))
    sys.stdout.write(f"\r  Progress: {min(pct, 100)}%")
    sys.stdout.flush()


def main():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PRETRAINED WEIGHTS DOWNLOADER")
    print("=" * 60)

    # Check if already downloaded
    for src in SOURCES:
        target = WEIGHTS_DIR / src["filename"]
        if target.exists():
            size_mb = target.stat().st_size / 1e6
            print(f"\n  Already exists: {target} ({size_mb:.1f} MB)")
            print("  Skipping download.")
            return str(target)

    # Try each source
    for src in SOURCES:
        target = WEIGHTS_DIR / src["filename"]
        print(f"\n  Downloading: {src['name']}")
        print(f"  URL: {src['url'][:60]}...")
        print(f"  Target: {target}")

        try:
            urllib.request.urlretrieve(src["url"], target, download_progress)
            size_mb = target.stat().st_size / 1e6
            print(f"\n  Downloaded: {target} ({size_mb:.1f} MB)")
            return str(target)
        except Exception as e:
            print(f"\n  Failed: {e}")

    print("\n  All sources failed. Models will use HuggingFace Hub or random init.")
    return None


if __name__ == "__main__":
    path = main()
    if path:
        print(f"\n  Weights ready at: {path}")
        print("  Models will auto-detect these on next training run.")
    print("=" * 60)
