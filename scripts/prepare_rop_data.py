#!/usr/bin/env python3
"""
Prepare the ROP infant retinal dataset for integration.

Creates:
  1. data/fundus_gate_training/fundus/   — symlinks to ROP + RFMiD images (positive fundus)
  2. data/rop_processed/images/          — flat directory of all ROP images with metadata CSV
  3. data/rop_processed/labels.csv       — cleaned labels for future ROP classification task

Usage:
    python3 scripts/prepare_rop_data.py
"""

import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROP_BASE = Path("data/rop_infants")
ROP_IMAGES = Path(os.path.realpath("data/rop_infants/images")) / "images"
RFMID_TRAIN = Path(os.path.realpath("data/rfmid/1. Original Images/a. Training Set"))
OUTPUT_ROP = Path("data/rop_processed")
OUTPUT_GATE = Path("data/fundus_gate_training")


def prepare_rop_flat():
    """Flatten ROP images into a single directory with cleaned metadata."""
    OUTPUT_ROP.mkdir(parents=True, exist_ok=True)
    img_out = OUTPUT_ROP / "images"
    img_out.mkdir(exist_ok=True)

    records = []
    count = 0

    for patient_dir in sorted(ROP_IMAGES.iterdir()):
        if not patient_dir.is_dir():
            continue
        for img_path in patient_dir.rglob("*.jpg"):
            # Parse metadata from filename
            # Format: {ID}_{Sex}_{GA}_{BW}_{PA}_{DG}_{PF}_{Device}_{Serie}_{Num}.jpg
            name = img_path.stem
            parts = name.split("_")

            record = {"filename": img_path.name, "source_path": str(img_path)}
            if len(parts) >= 9:
                record["patient_id"] = parts[0]
                record["sex"] = parts[1]
                record["gestational_age"] = parts[2].replace("GA", "")
                record["birth_weight"] = parts[3].replace("BW", "")
                record["postconceptual_age"] = parts[4].replace("PA", "")
                record["diagnosis_code"] = parts[5].replace("DG", "")
                record["plus_form"] = parts[6].replace("PF", "")
                record["device"] = parts[7].replace("D", "")
                record["serie"] = parts[8].replace("S", "")

            # Symlink into flat dir
            link = img_out / img_path.name
            if not link.exists():
                link.symlink_to(img_path.resolve())

            records.append(record)
            count += 1

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_ROP / "labels.csv", index=False)
    logger.info(
        f"ROP processed: {count} images -> {img_out}, labels -> {OUTPUT_ROP / 'labels.csv'}"
    )
    return count


def prepare_fundus_gate_data():
    """Create fundus gate training directories.

    Positive examples: ROP infant fundus + RFMiD training fundus images
    Negative examples: must be provided by user (non-fundus images)
    """
    fundus_dir = OUTPUT_GATE / "fundus"
    non_fundus_dir = OUTPUT_GATE / "non_fundus"
    fundus_dir.mkdir(parents=True, exist_ok=True)
    non_fundus_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    # Symlink ROP images as positive fundus examples
    for img_path in ROP_IMAGES.rglob("*.jpg"):
        link = fundus_dir / f"rop_{img_path.name}"
        if not link.exists():
            link.symlink_to(img_path.resolve())
            count += 1

    # Symlink RFMiD training images as positive fundus examples
    if RFMID_TRAIN.exists():
        for img_path in RFMID_TRAIN.iterdir():
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                link = fundus_dir / f"rfmid_{img_path.name}"
                if not link.exists():
                    link.symlink_to(img_path.resolve())
                    count += 1

    logger.info(f"Fundus gate positive examples: {count} images -> {fundus_dir}")
    logger.info(f"Negative examples directory: {non_fundus_dir} (populate with non-fundus images)")
    logger.info(
        "To train the fundus gate:\n"
        "  1. Add 500+ non-fundus images to data/fundus_gate_training/non_fundus/\n"
        "     (natural scenes, other medical images, random photos)\n"
        '  2. Run: python3 -c "from src.data.fundus_gate_learned import train_fundus_gate; '
        "train_fundus_gate('data/fundus_gate_training/fundus', "
        "'data/fundus_gate_training/non_fundus')\""
    )
    return count


def main():
    logger.info("=== Preparing ROP Infant Retinal Dataset ===")

    if not ROP_IMAGES.exists():
        logger.error(f"ROP images not found at {ROP_IMAGES}")
        return

    n_rop = prepare_rop_flat()
    n_gate = prepare_fundus_gate_data()

    # Summary
    logger.info("")
    logger.info("=== INTEGRATION SUMMARY ===")
    logger.info(f"1. ROP flat dataset: {n_rop} images at data/rop_processed/")
    logger.info("   Labels: data/rop_processed/labels.csv")
    logger.info("   Use for: future ROP classification task")
    logger.info("")
    logger.info(
        f"2. Fundus gate training: {n_gate} positive examples at data/fundus_gate_training/fundus/"
    )
    logger.info(f"   Sources: {n_rop} ROP + {n_gate - n_rop} RFMiD training images")
    logger.info("   Need: Add non-fundus images to data/fundus_gate_training/non_fundus/")
    logger.info("")
    logger.info("3. RFMiD pipeline: no changes needed (already integrated via kagglehub)")


if __name__ == "__main__":
    main()
