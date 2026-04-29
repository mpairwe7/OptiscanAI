#!/usr/bin/env python3
"""Train the fundus gate MobileNetV3-Small binary classifier.

Enhanced training pipeline with:
- PGD adversarial augmentation for robustness
- JPEG compression simulation during training
- Train/val split with per-epoch metrics
- Post-training export to ONNX, TorchScript, INT8 quantization

Usage:
    python scripts/train_fundus_gate.py \
        --fundus-dir data/fundus_images \
        --non-fundus-dir data/non_fundus_images \
        --output-path weights/fundus_gate.pth \
        --epochs 20 \
        --export-onnx \
        --quantize-int8

Hard Negative Mining Guidance:
    The non-fundus directory should include diverse hard negatives:
    - Close-up eyes/skin (iris photos, dermatology images)
    - Medical illustrations and diagrams
    - AI-generated fundus-style images (synthetic fakes)
    - Smartphone photos with red-eye + vintage filters
    - Leaves, food, fabric with similar warm circular textures
    - Slit-lamp photographs, OCT scans
    - Bokeh/lens flare images, circular objects on dark backgrounds

    Recommended sources for positive (fundus) images:
    - RFMiD dataset (primary — already used in this project)
    - APTOS 2019 Blindness Detection (Kaggle)
    - EyePACS (Kaggle)
    - DRIVE retinal vessel dataset

    Target: ~1000 fundus + ~1000 non-fundus for >98% accuracy.
"""

import argparse
import io
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom augmentations
# ---------------------------------------------------------------------------


class JPEGCompression:
    """Simulate JPEG compression artifacts during training."""

    def __init__(self, quality_range: tuple[int, int] = (30, 95)):
        self.lo, self.hi = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        quality = np.random.randint(self.lo, self.hi + 1)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BinaryImageDataset(Dataset):
    """Fundus (label=1) vs non-fundus (label=0) dataset."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    def __init__(self, pos_dir: str, neg_dir: str, transform, indices=None):
        self.samples: list[tuple[str, float]] = []
        self.transform = transform

        for f in sorted(os.listdir(pos_dir)):
            if os.path.splitext(f)[1].lower() in self.EXTENSIONS:
                self.samples.append((os.path.join(pos_dir, f), 1.0))
        for f in sorted(os.listdir(neg_dir)):
            if os.path.splitext(f)[1].lower() in self.EXTENSIONS:
                self.samples.append((os.path.join(neg_dir, f), 0.0))

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            # Graceful failure: return black image
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        return self.transform(image), torch.tensor(label)


# ---------------------------------------------------------------------------
# PGD adversarial training
# ---------------------------------------------------------------------------


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    epsilon: float = 0.03,
    alpha: float = 0.01,
    steps: int = 7,
) -> torch.Tensor:
    """Projected Gradient Descent adversarial attack."""
    images_adv = images.clone().detach().requires_grad_(True)

    for _ in range(steps):
        outputs = model(images_adv).squeeze(-1)
        loss = criterion(outputs, labels)
        loss.backward()

        with torch.no_grad():
            perturbation = alpha * images_adv.grad.sign()
            images_adv = images_adv + perturbation
            # Project back to epsilon ball
            delta = torch.clamp(images_adv - images, min=-epsilon, max=epsilon)
            images_adv = torch.clamp(images + delta, min=0.0, max=1.0).detach().requires_grad_(True)

    return images_adv.detach()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s", device)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        JPEGCompression(quality_range=tuple(int(x) for x in args.jpeg_quality_range.split(","))),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Build full dataset to get indices, then split
    full_dataset = BinaryImageDataset(args.fundus_dir, args.non_fundus_dir, train_transform)
    n = len(full_dataset)
    labels = [s[1] for s in full_dataset.samples]

    train_idx, val_idx = train_test_split(
        range(n), test_size=args.val_split, stratify=labels, random_state=42
    )
    logger.info("Dataset: %d total (%d train, %d val)", n, len(train_idx), len(val_idx))

    train_dataset = BinaryImageDataset(args.fundus_dir, args.non_fundus_dir, train_transform, indices=train_idx)
    val_dataset = BinaryImageDataset(args.fundus_dir, args.non_fundus_dir, val_transform, indices=val_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    from src.data.fundus_gate_learned import LearnedFundusGate
    model = LearnedFundusGate(weights_path=None)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    for epoch in range(args.epochs):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for images, labels_batch in train_loader:
            images, labels_batch = images.to(device), labels_batch.to(device)

            # Clean loss
            logits = model(images).squeeze(-1)
            loss_clean = criterion(logits, labels_batch)

            # Adversarial loss (PGD)
            if args.pgd_epsilon > 0:
                images_adv = pgd_attack(
                    model, images, labels_batch, criterion,
                    epsilon=args.pgd_epsilon, steps=args.pgd_steps,
                )
                logits_adv = model(images_adv).squeeze(-1)
                loss_adv = criterion(logits_adv, labels_batch)
                loss = 0.5 * loss_clean + 0.5 * loss_adv
            else:
                loss = loss_clean

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # ── Validate ──
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for images, labels_batch in val_loader:
                images = images.to(device)
                logits = model(images).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels_batch.numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        preds = (all_probs >= 0.5).astype(float)

        acc = accuracy_score(all_labels, preds)
        prec = precision_score(all_labels, preds, zero_division=0)
        rec = recall_score(all_labels, preds, zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0

        avg_loss = train_loss / max(len(train_loader), 1)
        logger.info(
            "Epoch %d/%d — Loss: %.4f | Val Acc: %.4f | Prec: %.4f | Rec: %.4f | AUC: %.4f",
            epoch + 1, args.epochs, avg_loss, acc, prec, rec, auc,
        )

        if auc > best_auc:
            best_auc = auc
            os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.output_path)
            logger.info("  → Saved best model (AUC=%.4f) to %s", auc, args.output_path)

    logger.info("Training complete. Best AUC: %.4f", best_auc)

    # ── Post-training export ──
    model.load_state_dict(torch.load(args.output_path, map_location="cpu", weights_only=True))
    model.eval()
    model = model.cpu()

    stem = os.path.splitext(args.output_path)[0]

    if args.export_onnx:
        try:
            onnx_path = f"{stem}.onnx"
            dummy = torch.randn(1, 3, 224, 224)
            torch.onnx.export(
                model, dummy, onnx_path,
                input_names=["image"], output_names=["logit"],
                dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}},
                opset_version=17,
            )
            logger.info("ONNX exported to %s", onnx_path)
        except Exception as e:
            logger.warning("ONNX export failed: %s", e)

    if args.export_torchscript:
        try:
            ts_path = f"{stem}.pt"
            scripted = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
            scripted.save(ts_path)
            logger.info("TorchScript exported to %s", ts_path)
        except Exception as e:
            logger.warning("TorchScript export failed: %s", e)

    if args.quantize_int8:
        try:
            int8_path = f"{stem}_int8.pth"
            quantized = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
            torch.save(quantized.state_dict(), int8_path)
            logger.info("INT8 quantized model saved to %s", int8_path)
        except Exception as e:
            logger.warning("INT8 quantization failed: %s", e)

    # ── Latency benchmark ──
    logger.info("Benchmarking inference latency...")
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    latencies = []
    with torch.no_grad():
        for _ in range(10):  # warmup
            model(dummy)
        for _ in range(100):
            t0 = time.perf_counter()
            model(dummy)
            latencies.append((time.perf_counter() - t0) * 1000)

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    logger.info("Latency (CPU): p50=%.1fms, p95=%.1fms, p99=%.1fms", p50, p95, p99)


def main():
    parser = argparse.ArgumentParser(description="Train fundus gate binary classifier")
    parser.add_argument("--fundus-dir", required=True, help="Directory of fundus images")
    parser.add_argument("--non-fundus-dir", required=True, help="Directory of non-fundus images")
    parser.add_argument("--output-path", default="weights/fundus_gate.pth", help="Output weights path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pgd-epsilon", type=float, default=0.03, help="PGD attack epsilon (0 to disable)")
    parser.add_argument("--pgd-steps", type=int, default=7, help="PGD attack iterations")
    parser.add_argument("--jpeg-quality-range", default="30,95", help="JPEG compression quality range")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--device", default="auto", help="Device: auto|cpu|cuda")
    parser.add_argument("--export-onnx", action="store_true", help="Export to ONNX after training")
    parser.add_argument("--export-torchscript", action="store_true", help="Export to TorchScript")
    parser.add_argument("--quantize-int8", action="store_true", help="Export INT8 quantized model")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
