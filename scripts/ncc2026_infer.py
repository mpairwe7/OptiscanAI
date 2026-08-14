#!/usr/bin/env python3
"""Run the production teacher over an RFMiD split and cache raw probabilities.

Produces the evidence base for the NCC 2026 camera-ready revision: every table
in the paper is computed from the ``.npz`` files this script writes, so the
numbers can be regenerated from the checkpoint at any time.

Usage:
    python3 scripts/ncc2026_infer.py --split test --variant fp32
    python3 scripts/ncc2026_infer.py --split test --variant fp32_tta
    python3 scripts/ncc2026_infer.py --split test --variant int8   # CPU dynamic quant
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("USE_PRETRAINED", "0")  # checkpoint supplies all weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ncc2026")

CKPT = REPO / "outputs/checkpoints/v2/final_with_thresholds.pth"
OUT_DIR = REPO / "outputs/ncc2026"

# The complete official RFMiD release (1920/640/640). The in-repo copy under
# data/rfmid_extracted/ is missing a handful of images, so prefer the full one.
KAGGLE_ROOT = Path.home() / (
    ".cache/kagglehub/datasets/mpairwelauben/multi-disease-retinal-eye-disease-dataset"
    "/versions/1/A. RFMiD_All_Classes_Dataset"
)


def _resolve_splits() -> dict[str, tuple[Path, Path]]:
    if KAGGLE_ROOT.is_dir():
        imgs = KAGGLE_ROOT / "1. Original Images"
        gts = KAGGLE_ROOT / "2. Groundtruths"
        return {
            "train": (imgs / "a. Training Set", gts / "a. RFMiD_Training_Labels.csv"),
            "val": (imgs / "b. Validation Set", gts / "b. RFMiD_Validation_Labels.csv"),
            "test": (imgs / "c. Testing Set", gts / "c. RFMiD_Testing_Labels.csv"),
        }
    return {
        "test": (
            REPO / "data/rfmid_extracted/Test_Set/Test",
            REPO / "data/rfmid_extracted/Test_Set/RFMiD_Testing_Labels.csv",
        ),
        "val": (
            REPO / "data/rfmid_extracted/Evaluation_Set/Validation",
            REPO / "data/rfmid_extracted/Evaluation_Set/RFMiD_Validation_Labels.csv",
        ),
    }


SPLITS = _resolve_splits()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RFMiDSplit(Dataset):
    """RFMiD split reading exactly the transform used during training/eval."""

    def __init__(self, img_dir: Path, labels_csv: Path, classes: list[str], img_size: int = 224):
        import torchvision.transforms as T

        df = pd.read_csv(labels_csv, encoding="utf-8-sig")
        keep = df["ID"].apply(lambda i: (img_dir / f"{int(i)}.png").exists())
        self.missing = [int(i) for i in df.loc[~keep, "ID"]]
        self.df = df[keep].reset_index(drop=True)
        self.img_dir = img_dir
        self.classes = classes
        self.labels = self.df[classes].to_numpy(dtype=np.float32)
        self.ids = self.df["ID"].to_numpy()
        self.transform = T.Compose(
            [
                T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        logger.info(
            "%s: %d images (%d label rows had no image on disk)",
            img_dir.name,
            len(self.df),
            len(self.missing),
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img = Image.open(self.img_dir / f"{int(self.ids[idx])}.png").convert("RGB")
        return self.transform(img), torch.from_numpy(self.labels[idx])


def build_model(num_classes: int, classes: list[str], ckpt: dict):
    from src.models.retinal_foundation_hybrid_v2 import RetinalFoundationHybridV2
    from src.models.vignn import ClinicalKnowledgeGraph

    kg = ClinicalKnowledgeGraph(classes)
    model = RetinalFoundationHybridV2(
        num_classes=num_classes,
        hidden_dim=512,
        clinical_knowledge_graph=kg,
        backbone="vit_large_patch16_224",
        img_size=224,
        use_lora=True,
        lora_rank=16,
        freeze_backbone=True,
    )
    incompatible = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        logger.warning(
            "state_dict mismatch — missing=%s unexpected=%s",
            incompatible.missing_keys[:5],
            incompatible.unexpected_keys[:5],
        )
    model.eval()
    return model


@torch.no_grad()
def run(model, loader, device, tta: bool = False):
    """Return (probs, labels). TTA averages sigmoid over 6 geometric views."""
    probs_all, labels_all = [], []
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        if tta:
            views = [
                x,
                x.flip(-1),
                x.flip(-2),
                x.rot90(1, [-2, -1]),
                x.rot90(2, [-2, -1]),
                x.rot90(3, [-2, -1]),
            ]
            p = torch.stack([torch.sigmoid(model(v).float()) for v in views]).mean(0)
        else:
            p = torch.sigmoid(model(x).float())
        probs_all.append(p.cpu().numpy())
        labels_all.append(y.numpy())
        if i % 10 == 0:
            done = sum(len(a) for a in probs_all)
            logger.info("  %d images  (%.2f s/img)", done, (time.time() - t0) / max(done, 1))
    return np.concatenate(probs_all), np.concatenate(labels_all)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=list(SPLITS))
    ap.add_argument("--variant", default="fp32", choices=["fp32", "fp32_tta", "int8", "fp32_cpu"])
    ap.add_argument("--device", default="cuda:7")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=32)
    # Worker processes need a unix socket the sandbox may forbid; 0 is safe.
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img_dir, labels_csv = SPLITS[args.split]

    t_load = time.time()
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    classes = ckpt["disease_columns"]
    logger.info("checkpoint loaded in %.1f s | %d classes", time.time() - t_load, len(classes))

    ds = RFMiDSplit(img_dir, labels_csv, classes)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    t_build = time.time()
    model = build_model(len(classes), classes, ckpt)
    logger.info("model built in %.1f s", time.time() - t_build)

    if args.variant in ("int8", "fp32_cpu"):
        # Both CPU paths use identical threading so the latency numbers compare.
        torch.set_num_threads(args.threads)
        device = torch.device("cpu")
        model = model.to(device)
        if args.variant == "int8":
            model = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            logger.info("dynamic INT8 quantization applied to all nn.Linear layers")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    t0 = time.time()
    probs, labels = run(model, loader, device, tta=(args.variant == "fp32_tta"))
    elapsed = time.time() - t0

    out = OUT_DIR / f"probs_{args.split}_{args.variant}.npz"
    np.savez_compressed(
        out,
        probs=probs,
        labels=labels,
        ids=ds.ids,
        classes=np.array(classes),
        thresholds=np.asarray(ckpt["thresholds"], dtype=np.float32),
    )
    logger.info(
        "wrote %s  probs=%s  %.1f s total (%.3f s/img)",
        out,
        probs.shape,
        elapsed,
        elapsed / len(ds),
    )
    (OUT_DIR / f"runmeta_{args.split}_{args.variant}.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "variant": args.variant,
                "n_images": int(len(ds)),
                "n_label_rows": int(len(pd.read_csv(labels_csv, encoding="utf-8-sig"))),
                "missing_images": [int(m) for m in ds.missing],
                "device": str(device),
                "seconds_total": round(elapsed, 2),
                "seconds_per_image": round(elapsed / len(ds), 4),
                "torch": torch.__version__,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
