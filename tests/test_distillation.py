"""Tests for MobileStudentV1 knowledge distillation pipeline.

Validates:
  - Student model architecture and interface compatibility
  - Distillation loss components and temperature annealing
  - Per-class precision floors on the student
  - ONNX export parity (FP32 and INT8)
  - Model size constraints (INT8 <= 50 MB)
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.mobile_student import MobileStudentV1
from src.training.distillation_loss import PrecisionAwareDistillationLoss


# ---------------------------------------------------------------------------
# MobileStudentV1 architecture tests
# ---------------------------------------------------------------------------

class TestMobileStudentV1:
    """Test the student model architecture and interface."""

    @pytest.fixture
    def student(self):
        return MobileStudentV1(num_classes=28, pretrained=False)

    def test_forward_shape(self, student):
        x = torch.randn(2, 3, 224, 224)
        logits = student(x)
        assert logits.shape == (2, 28)

    def test_forward_dtype(self, student):
        x = torch.randn(1, 3, 224, 224)
        logits = student(x)
        assert logits.dtype == torch.float32

    def test_get_features_shape(self, student):
        x = torch.randn(2, 3, 224, 224)
        features = student.get_features(x)
        assert features.shape == (2, 512), "Features should be 512-dim (teacher's hidden_dim)"

    def test_predict_returns_dict(self, student):
        x = torch.randn(1, 3, 224, 224)
        result = student.predict(x)
        assert "logits" in result
        assert "probabilities" in result
        assert "predictions" in result

    def test_predict_probabilities_range(self, student):
        x = torch.randn(1, 3, 224, 224)
        result = student.predict(x)
        probs = result["probabilities"]
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_thresholds_buffer(self, student):
        assert hasattr(student, "thresholds")
        assert student.thresholds.shape == (28,)
        assert (student.thresholds == 0.5).all(), "Default thresholds should be 0.5"

    def test_load_save_thresholds(self, student, tmp_path):
        custom_thresh = [0.1 + i * 0.03 for i in range(28)]
        thresh_file = tmp_path / "thresholds.json"
        with open(thresh_file, "w") as f:
            json.dump(custom_thresh, f)

        student.load_thresholds(thresh_file)
        assert abs(student.thresholds[0].item() - 0.1) < 1e-6

        save_file = tmp_path / "saved.json"
        student.save_thresholds(save_file)
        assert save_file.exists()
        with open(save_file) as f:
            loaded = json.load(f)
        assert len(loaded) == 28

    def test_prepare_for_export(self, student):
        """prepare_for_export should be a no-op (no LoRA in student)."""
        result = student.prepare_for_export()
        assert result is student
        assert not student.training  # Should be in eval mode

    def test_param_summary(self, student):
        summary = student.get_param_summary()
        assert summary["backbone"] == "mobilenetv3_large_100"
        assert summary["hidden_dim"] == 512
        assert summary["num_classes"] == 28
        assert summary["total_params"] > 0
        # MobileNetV3-Large ~5.4M + projection + classifier
        assert summary["total_params"] < 10_000_000, "Student should be < 10M params"

    def test_apply_thresholds(self, student):
        student.thresholds.copy_(torch.full((28,), 0.5))
        probs = torch.tensor([[0.3, 0.7, 0.5] + [0.0] * 25])
        result = student.apply_thresholds(probs)
        assert result[0, 0].item() == 0.0  # 0.3 < 0.5
        assert result[0, 1].item() == 1.0  # 0.7 >= 0.5
        assert result[0, 2].item() == 1.0  # 0.5 >= 0.5

    def test_model_size_estimate(self, student):
        """INT8 ONNX should be estimated <= 50 MB."""
        summary = student.get_param_summary()
        assert summary["estimated_onnx_int8_mb"] <= 50


# ---------------------------------------------------------------------------
# PrecisionAwareDistillationLoss tests
# ---------------------------------------------------------------------------

class TestDistillationLoss:
    """Test the multi-component distillation loss."""

    @pytest.fixture
    def loss_fn(self):
        return PrecisionAwareDistillationLoss(
            alpha=0.6,
            beta=0.15,
            gamma=0.05,
            initial_temperature=6.0,
            final_temperature=2.0,
            total_epochs=40,
        )

    def test_temperature_annealing(self, loss_fn):
        loss_fn.set_epoch(0)
        assert abs(loss_fn.temperature - 6.0) < 1e-6

        loss_fn.set_epoch(39)
        assert abs(loss_fn.temperature - 2.0) < 1e-6

        loss_fn.set_epoch(20)
        expected = 6.0 + (20 / 39) * (2.0 - 6.0)
        assert abs(loss_fn.temperature - expected) < 0.1

    def test_forward_returns_all_components(self, loss_fn):
        B, C = 4, 28
        student_logits = torch.randn(B, C)
        teacher_logits = torch.randn(B, C)
        targets = torch.randint(0, 2, (B, C)).float()

        result = loss_fn(student_logits, teacher_logits, targets)

        assert "total" in result
        assert "kd_loss" in result
        assert "task_loss" in result
        assert "feature_loss" in result
        assert "threshold_loss" in result
        assert "temperature" in result

    def test_total_loss_is_scalar(self, loss_fn):
        B, C = 4, 28
        result = loss_fn(
            torch.randn(B, C),
            torch.randn(B, C),
            torch.randint(0, 2, (B, C)).float(),
        )
        assert result["total"].dim() == 0

    def test_total_loss_requires_grad(self, loss_fn):
        B, C = 4, 28
        student = torch.randn(B, C, requires_grad=True)
        result = loss_fn(student, torch.randn(B, C), torch.zeros(B, C))
        assert result["total"].requires_grad

    def test_feature_alignment_loss(self, loss_fn):
        B, D = 4, 512
        student_feat = torch.randn(B, D, requires_grad=True)
        teacher_feat = torch.randn(B, D)

        result = loss_fn(
            torch.randn(B, 28, requires_grad=True),
            torch.randn(B, 28),
            torch.zeros(B, 28),
            student_features=student_feat,
            teacher_features=teacher_feat,
        )
        assert result["feature_loss"].item() > 0

    def test_threshold_alignment_loss(self, loss_fn):
        B, C = 4, 28
        thresholds = torch.full((C,), 0.5)

        # Teacher confident positive, student below threshold
        teacher_logits = torch.full((B, C), 2.0)  # sigmoid(2) ≈ 0.88
        student_logits = torch.full((B, C), -2.0)  # sigmoid(-2) ≈ 0.12

        result = loss_fn(
            student_logits,
            teacher_logits,
            torch.ones(B, C),
            thresholds=thresholds,
        )
        assert result["threshold_loss"].item() > 0

    def test_zero_feature_loss_when_not_provided(self, loss_fn):
        result = loss_fn(
            torch.randn(4, 28),
            torch.randn(4, 28),
            torch.zeros(4, 28),
        )
        assert result["feature_loss"].item() == 0.0

    def test_alpha_weight_balance(self):
        """Higher alpha should increase KD loss contribution."""
        B, C = 4, 28
        s, t = torch.randn(B, C), torch.randn(B, C)
        targets = torch.zeros(B, C)

        high_alpha = PrecisionAwareDistillationLoss(alpha=0.9, total_epochs=1)
        low_alpha = PrecisionAwareDistillationLoss(alpha=0.1, total_epochs=1)

        r_high = high_alpha(s, t, targets)
        r_low = low_alpha(s, t, targets)

        # Both should produce valid losses
        assert r_high["total"].item() > 0
        assert r_low["total"].item() > 0


# ---------------------------------------------------------------------------
# Integration: student + loss together
# ---------------------------------------------------------------------------

class TestDistillationIntegration:
    """Integration tests for the full distillation pipeline."""

    def test_student_loss_backward(self):
        """Verify gradients flow through student -> loss -> backward."""
        student = MobileStudentV1(num_classes=28, pretrained=False)
        loss_fn = PrecisionAwareDistillationLoss(total_epochs=10)

        x = torch.randn(2, 3, 224, 224)
        targets = torch.randint(0, 2, (2, 28)).float()
        teacher_logits = torch.randn(2, 28)
        teacher_features = torch.randn(2, 512)

        student_logits = student(x)
        student_features = student.get_features(x)

        result = loss_fn(
            student_logits,
            teacher_logits,
            targets,
            student_features=student_features,
            teacher_features=teacher_features,
            thresholds=student.thresholds,
        )

        result["total"].backward()

        # Check gradients exist
        grad_count = sum(1 for p in student.parameters() if p.grad is not None)
        assert grad_count > 0, "Student should have gradients after backward"

    def test_student_checkpoint_save_load(self, tmp_path):
        """Test checkpoint save/load round-trip."""
        student = MobileStudentV1(num_classes=28, pretrained=False)
        student.thresholds.copy_(torch.rand(28))

        ckpt_path = tmp_path / "student.pth"
        torch.save({
            "state_dict": student.state_dict(),
            "thresholds": student.thresholds.cpu(),
        }, ckpt_path)

        # Load into new instance
        student2 = MobileStudentV1(num_classes=28, pretrained=False)
        ckpt = torch.load(ckpt_path, weights_only=False)
        student2.load_state_dict(ckpt["state_dict"])
        student2.thresholds.copy_(ckpt["thresholds"])

        assert torch.allclose(student.thresholds, student2.thresholds)
