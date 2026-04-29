"""Tests for loss functions: FocalLoss, AsymmetricLoss, build_loss factory."""
import sys

sys.path.insert(0, "/home/developer/Mpairwe7/MLOPS_V1")

import pytest
import torch

from src.training.losses import (
    AsymmetricLoss,
    FocalLoss,
    _smooth_targets,
    build_loss,
)


# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------

def test_focal_loss_forward():
    """FocalLoss should produce a scalar loss from random logits and targets."""
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.randn(8, 45)
    targets = torch.randint(0, 2, (8, 45)).float()
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0, "Loss must be scalar"
    assert loss.item() > 0, "Loss should be positive for random inputs"
    assert torch.isfinite(loss), "Loss must be finite"


def test_focal_loss_zero_for_perfect():
    """FocalLoss should be very small when predictions perfectly match targets."""
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    targets = torch.ones(4, 10)
    # Large positive logits => sigmoid close to 1 => matches targets=1
    logits = torch.ones(4, 10) * 10.0
    loss = loss_fn(logits, targets)
    assert loss.item() < 0.1, "Loss should be near zero for perfect predictions"


# ---------------------------------------------------------------------------
# AsymmetricLoss
# ---------------------------------------------------------------------------

def test_asymmetric_loss_forward():
    """AsymmetricLoss should produce a scalar loss from random logits and targets."""
    loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0)
    logits = torch.randn(8, 45)
    targets = torch.randint(0, 2, (8, 45)).float()
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0, "Loss must be scalar"
    assert loss.item() > 0, "Loss should be positive for random inputs"
    assert torch.isfinite(loss), "Loss must be finite"


def test_asymmetric_loss_downweights_easy_negatives():
    """Easy negatives should contribute less loss than hard negatives."""
    loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.0)
    targets = torch.zeros(1, 1)
    easy_negative = loss_fn(torch.tensor([[-6.0]]), targets)
    hard_negative = loss_fn(torch.tensor([[0.0]]), targets)
    assert easy_negative.item() < hard_negative.item()


# ---------------------------------------------------------------------------
# build_loss factory
# ---------------------------------------------------------------------------

def test_build_loss_focal():
    cfg = {"training": {"loss": "focal", "focal_alpha": 0.25, "focal_gamma": 2.0}}
    loss_fn = build_loss(cfg)
    assert isinstance(loss_fn, FocalLoss)


def test_build_loss_asymmetric():
    cfg = {"training": {"loss": "asymmetric"}}
    loss_fn = build_loss(cfg)
    assert isinstance(loss_fn, AsymmetricLoss)


def test_build_loss_bce():
    cfg = {"training": {"loss": "bce"}}
    loss_fn = build_loss(cfg)
    assert isinstance(loss_fn, torch.nn.BCEWithLogitsLoss)


def test_build_loss_unknown():
    cfg = {"training": {"loss": "unknown_loss"}}
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss(cfg)


# ---------------------------------------------------------------------------
# Label smoothing
# ---------------------------------------------------------------------------

def test_label_smoothing():
    """_smooth_targets should shift binary values toward 0.5."""
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
    smoothed = _smooth_targets(targets, smoothing=0.1)
    # 0.0 -> 0.0 * 0.9 + 0.5 * 0.1 = 0.05
    # 1.0 -> 1.0 * 0.9 + 0.5 * 0.1 = 0.95
    assert not torch.equal(targets, smoothed), "Smoothed targets should differ"
    assert smoothed[0].item() == pytest.approx(0.05, abs=1e-6)
    assert smoothed[1].item() == pytest.approx(0.95, abs=1e-6)


def test_label_smoothing_zero():
    """Smoothing of 0.0 should leave targets unchanged."""
    targets = torch.tensor([0.0, 1.0])
    smoothed = _smooth_targets(targets, smoothing=0.0)
    assert torch.equal(targets, smoothed)


# ---------------------------------------------------------------------------
# FocalLoss with pos_weight
# ---------------------------------------------------------------------------

def test_loss_with_pos_weight():
    """FocalLoss should accept a pos_weight tensor without error."""
    pos_weight = torch.ones(45) * 2.0
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)
    logits = torch.randn(4, 45)
    targets = torch.randint(0, 2, (4, 45)).float()
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_build_loss_with_pos_weight():
    """build_loss should pass pos_weight to FocalLoss and BCEWithLogitsLoss."""
    pos_weight = torch.ones(45) * 3.0
    cfg = {"training": {"loss": "focal"}}
    loss_fn = build_loss(cfg, pos_weight=pos_weight)
    assert isinstance(loss_fn, FocalLoss)
    # The pos_weight should be stored as a buffer
    assert loss_fn.pos_weight is not None


def test_build_loss_with_label_smoothing():
    """build_loss should propagate label_smoothing from config."""
    cfg = {"training": {"loss": "focal", "label_smoothing": 0.1}}
    loss_fn = build_loss(cfg)
    assert isinstance(loss_fn, FocalLoss)
    assert loss_fn.label_smoothing == 0.1
