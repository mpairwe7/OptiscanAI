"""
RetinalDiseaseDataset - PyTorch Dataset for RFMiD retinal fundus images.
Extracted and refactored from notebook cell 21.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class RetinalDiseaseDataset(Dataset):
    """Multi-label retinal disease dataset for 48-class classification."""

    def __init__(
        self,
        labels_df: pd.DataFrame,
        img_dir: str | Path,
        disease_columns: list[str],
        transform=None,
    ):
        self.labels_df = labels_df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.disease_columns = disease_columns
        self.transform = transform

        # Pre-convert labels to float32 array for speed
        self.labels_array = (
            self.labels_df[disease_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .values.astype(np.float32)
        )
        self.image_ids = self.labels_df["ID"].values

        # Validate that image directory exists
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        logger.info(
            f"Dataset initialized: {len(self)} samples, {len(disease_columns)} classes, "
            f"img_dir={self.img_dir}"
        )

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]

        # Try common extensions
        img_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = self.img_dir / f"{img_id}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        labels = torch.from_numpy(self.labels_array[idx])

        # Graceful handling of missing/corrupt images
        try:
            if img_path is None:
                raise FileNotFoundError(f"No image for ID {img_id}")
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError, SyntaxError, Exception) as e:
            logger.warning(f"Bad image {img_id}: {e} — using placeholder")
            image = Image.new("RGB", (224, 224), (0, 0, 0))
            labels = torch.zeros_like(labels)  # Zero labels for bad images

        if self.transform:
            image = self.transform(image)

        return image, labels

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for balanced training."""
        pos_counts = self.labels_array.sum(axis=0).clip(min=1)
        neg_counts = len(self) - pos_counts
        weights = neg_counts / pos_counts
        return torch.from_numpy(weights.astype(np.float32))

    def get_pos_weights(self) -> torch.Tensor:
        """Compute pos_weight for BCEWithLogitsLoss (handles class imbalance)."""
        pos_counts = self.labels_array.sum(axis=0).clip(min=1)
        neg_counts = len(self) - pos_counts
        pos_weights = neg_counts / pos_counts
        # Cap extreme weights
        pos_weights = np.clip(pos_weights, 0.5, 50.0)
        return torch.from_numpy(pos_weights.astype(np.float32))
