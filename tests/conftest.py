"""Shared pytest fixtures for retinal disease MLOps test suite."""
import sys
import os
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, "/home/developer/Mpairwe7/MLOPS_V1")

# Fix for environments where default pytest tmp_path base is owned by another user.
# Must be set BEFORE pytest collects tmp_path fixtures.
_custom_tmp = tempfile.mkdtemp(prefix="pytest_retinal_")
os.environ["TMPDIR"] = _custom_tmp
os.environ["TMP"] = _custom_tmp
os.environ["TEMP"] = _custom_tmp

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.datamodule import DISEASE_COLUMNS


@pytest.fixture
def _safe_tmpdir():
    """Create a safe temporary directory that works regardless of ownership."""
    d = Path(tempfile.mkdtemp(prefix="retinal_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_image():
    """Return a random 224x224 RGB PIL Image."""
    arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def sample_batch():
    """Return a (images, labels) batch for multi-label classification.

    images: (4, 3, 224, 224) float tensor
    labels: (4, 45) binary float tensor
    """
    images = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, 2, (4, 45)).float()
    return images, labels


@pytest.fixture
def disease_columns():
    """Return the canonical list of 45 disease column names."""
    return list(DISEASE_COLUMNS)


@pytest.fixture
def train_config():
    """Return a minimal config dict mirroring configs/train.yaml structure."""
    return {
        "data": {
            "dataset_name": "mpairwelauben/multi-disease-retinal-eye-disease-dataset",
            "data_dir": None,
            "img_size": 224,
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "train_split": 0.7,
            "val_split": 0.2,
            "test_split": 0.1,
            "prefetch_factor": 2,
        },
        "model": {
            "name": "vignn",
            "num_classes": 45,
            "hidden_dim": 64,
            "num_graph_layers": 1,
            "num_heads": 2,
            "dropout": 0.1,
            "num_patches": 196,
            "patch_embed_dim": 384,
            "pretrained_backbone": False,
        },
        "training": {
            "max_epochs": 2,
            "learning_rate": 3.0e-4,
            "weight_decay": 1.0e-4,
            "warmup_epochs": 1,
            "scheduler": "cosine",
            "label_smoothing": 0.05,
            "gradient_clip_val": 1.0,
            "loss": "focal",
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
            "early_stopping_patience": 3,
            "early_stopping_metric": "val/f1_macro",
            "early_stopping_mode": "max",
            "precision": "32",
            "seed": 42,
        },
        "augmentation": {
            "train": {
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            },
            "val": {
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            },
        },
    }


@pytest.fixture
def tmp_model_dir(_safe_tmpdir):
    """Provide a temporary directory for model artifacts."""
    model_dir = _safe_tmpdir / "model_artifacts"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def sample_labels_df(disease_columns):
    """Return a DataFrame with ID and 45 disease columns (20 rows, synthetic)."""
    rng = np.random.RandomState(42)
    n_rows = 20
    data = {"ID": [f"img_{i:04d}" for i in range(n_rows)]}
    for col in disease_columns:
        # Sparse binary labels: ~10% positive rate
        data[col] = rng.choice([0, 1], size=n_rows, p=[0.9, 0.1])
    df = pd.DataFrame(data)
    # Ensure at least one positive per class so pos_weights doesn't blow up
    for col in disease_columns:
        if df[col].sum() == 0:
            df.loc[0, col] = 1
    return df


@pytest.fixture
def sample_img_dir(_safe_tmpdir, sample_labels_df):
    """Create a temp directory with synthetic 224x224 PNG images matching IDs."""
    img_dir = _safe_tmpdir / "images"
    img_dir.mkdir()
    rng = np.random.RandomState(123)
    for img_id in sample_labels_df["ID"].values:
        arr = rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")
        img.save(img_dir / f"{img_id}.png")
    return img_dir
