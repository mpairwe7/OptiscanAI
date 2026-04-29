"""Tests for RetinalDiseaseDataset: creation, length, getitem, weights."""
import sys

sys.path.insert(0, "/home/developer/Mpairwe7/MLOPS_V1")

import pytest
import torch
import numpy as np
from torchvision import transforms

from src.data.dataset import RetinalDiseaseDataset
from src.data.datamodule import DISEASE_COLUMNS


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------

def test_dataset_creation(sample_labels_df, sample_img_dir, disease_columns):
    """Dataset should initialize without errors."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    ds = RetinalDiseaseDataset(
        labels_df=sample_labels_df,
        img_dir=sample_img_dir,
        disease_columns=disease_columns,
        transform=transform,
    )
    assert ds is not None
    assert len(ds.disease_columns) == len(disease_columns)


def test_dataset_creation_no_transform(sample_labels_df, sample_img_dir, disease_columns):
    """Dataset should work with transform=None (returns PIL images)."""
    ds = RetinalDiseaseDataset(
        labels_df=sample_labels_df,
        img_dir=sample_img_dir,
        disease_columns=disease_columns,
        transform=None,
    )
    img, labels = ds[0]
    # Without transform, img is a PIL Image
    from PIL import Image
    assert isinstance(img, Image.Image)
    assert isinstance(labels, torch.Tensor)


# ---------------------------------------------------------------------------
# Length
# ---------------------------------------------------------------------------

def test_dataset_len(sample_labels_df, sample_img_dir, disease_columns):
    """Dataset length should match the number of rows in the labels DataFrame."""
    ds = RetinalDiseaseDataset(
        labels_df=sample_labels_df,
        img_dir=sample_img_dir,
        disease_columns=disease_columns,
        transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
    )
    assert len(ds) == len(sample_labels_df)


# ---------------------------------------------------------------------------
# __getitem__
# ---------------------------------------------------------------------------

def test_dataset_getitem(sample_labels_df, sample_img_dir, disease_columns):
    """Each item should return (image_tensor, label_tensor) with correct shapes."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    ds = RetinalDiseaseDataset(
        labels_df=sample_labels_df,
        img_dir=sample_img_dir,
        disease_columns=disease_columns,
        transform=transform,
    )
    image, labels = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224), f"Image shape {image.shape} != (3, 224, 224)"
    assert isinstance(labels, torch.Tensor)
    assert labels.shape == (len(disease_columns),)
    assert labels.dtype == torch.float32


def test_dataset_getitem_all_items(sample_labels_df, sample_img_dir, disease_columns):
    """Iterating through all items should succeed without errors."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    ds = RetinalDiseaseDataset(
        labels_df=sample_labels_df,
        img_dir=sample_img_dir,
        disease_columns=disease_columns,
        transform=transform,
    )
    for i in range(len(ds)):
        image, labels = ds[i]
        assert image.shape == (3, 224, 224)
        assert labels.shape == (len(disease_columns),)


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def test_dataset_class_weights(sample_labels_df, sample_img_dir, disease_columns):
    """get_class_weights() should return a tensor of length 45."""
    ds = RetinalDiseaseDataset(
        labels_df=sample_labels_df,
        img_dir=sample_img_dir,
        disease_columns=disease_columns,
        transform=None,
    )
    weights = ds.get_class_weights()
    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (len(disease_columns),)
    assert weights.dtype == torch.float32
    # All weights should be positive (inverse frequency)
    assert (weights > 0).all(), "All class weights should be positive"


# ---------------------------------------------------------------------------
# Pos weights
# ---------------------------------------------------------------------------

def test_dataset_pos_weights(sample_labels_df, sample_img_dir, disease_columns):
    """get_pos_weights() should return capped values in [0.5, 50.0]."""
    ds = RetinalDiseaseDataset(
        labels_df=sample_labels_df,
        img_dir=sample_img_dir,
        disease_columns=disease_columns,
        transform=None,
    )
    pos_weights = ds.get_pos_weights()
    assert isinstance(pos_weights, torch.Tensor)
    assert pos_weights.shape == (len(disease_columns),)
    assert pos_weights.min().item() >= 0.5, "pos_weight should be >= 0.5 (capped)"
    assert pos_weights.max().item() <= 50.0, "pos_weight should be <= 50.0 (capped)"


# ---------------------------------------------------------------------------
# Missing image handling
# ---------------------------------------------------------------------------

def test_dataset_missing_image(sample_labels_df, _safe_tmpdir, disease_columns):
    """Dataset should handle missing images gracefully with a placeholder."""
    # Create an empty image directory
    empty_dir = _safe_tmpdir / "empty_imgs"
    empty_dir.mkdir()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    ds = RetinalDiseaseDataset(
        labels_df=sample_labels_df,
        img_dir=empty_dir,
        disease_columns=disease_columns,
        transform=transform,
    )
    # Should not raise - uses placeholder
    image, labels = ds[0]
    assert image.shape == (3, 224, 224)
    # Labels should be zeroed for missing images
    assert labels.sum().item() == 0.0
