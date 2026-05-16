"""Precision-aware knowledge distillation loss for mobile student training.

Extends the base DistillationLoss (src/optimization/quantization.py) with:
  1. AsymmetricLossV2 as task loss (preserves precision-rescue behaviour)
  2. L2 feature alignment at the 512-dim bottleneck layer
  3. Threshold alignment penalty ensuring student respects precision floors
  4. Temperature annealing (T: 6.0 -> 2.0) for curriculum-style learning

Loss formulation:
    L = alpha * KD_loss(student, teacher, T)
      + (1 - alpha) * ASL_task_loss(student, targets)
      + beta * feature_alignment_loss(student_features, teacher_features)
      + gamma * threshold_alignment_loss(student, teacher, thresholds)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PrecisionAwareDistillationLoss(nn.Module):
    """Multi-component distillation loss optimized for precision-floor models.

    Parameters
    ----------
    alpha : float
        Weight for KD soft-target loss vs task loss. Default 0.6.
    beta : float
        Weight for feature alignment loss. Default 0.15.
    gamma : float
        Weight for threshold alignment loss. Default 0.05.
    initial_temperature : float
        Starting temperature for KD. Higher T forces student to learn
        inter-class soft relationships first. Default 6.0.
    final_temperature : float
        Ending temperature. Lower T sharpens to match hard predictions.
        Default 2.0.
    total_epochs : int
        Total training epochs for temperature annealing schedule.
    asl_gamma_neg : float
        ASL gamma_neg for suppressing easy negative FPs. Default 4.0.
    asl_gamma_pos : float
        ASL gamma_pos. Default 0.0 (never down-weight positives).
    asl_clip : float
        ASL clip threshold. Default 0.05.
    label_smoothing : float
        Label smoothing for ASL. Default 0.05.
    """

    def __init__(
        self,
        alpha: float = 0.6,
        beta: float = 0.15,
        gamma: float = 0.05,
        initial_temperature: float = 6.0,
        final_temperature: float = 2.0,
        total_epochs: int = 40,
        asl_gamma_neg: float = 4.0,
        asl_gamma_pos: float = 0.0,
        asl_clip: float = 0.05,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.total_epochs = total_epochs

        # ASL parameters
        self.asl_gamma_neg = asl_gamma_neg
        self.asl_gamma_pos = asl_gamma_pos
        self.asl_clip = asl_clip
        self.label_smoothing = label_smoothing

        self._current_epoch = 0

    @property
    def temperature(self) -> float:
        """Linear annealing from initial_temperature to final_temperature."""
        if self.total_epochs <= 1:
            return self.final_temperature
        progress = min(self._current_epoch / (self.total_epochs - 1), 1.0)
        return self.initial_temperature + progress * (
            self.final_temperature - self.initial_temperature
        )

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for temperature annealing."""
        self._current_epoch = epoch

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None,
        thresholds: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined distillation loss.

        Parameters
        ----------
        student_logits : Tensor [B, C]
            Raw logits from student model.
        teacher_logits : Tensor [B, C]
            Raw logits from teacher model (detached).
        targets : Tensor [B, C]
            Ground truth multi-label binary targets.
        student_features : Tensor [B, D], optional
            Student projected features (512-dim) for alignment.
        teacher_features : Tensor [B, D], optional
            Teacher global pool features (512-dim) for alignment.
        thresholds : Tensor [C], optional
            Per-class precision-floor thresholds for alignment loss.

        Returns
        -------
        dict with 'total', 'kd_loss', 'task_loss', 'feature_loss',
        'threshold_loss', 'temperature' keys.
        """
        T = self.temperature

        # --- 1. Knowledge Distillation Loss (temperature-scaled BCE) ---
        soft_teacher = torch.sigmoid(teacher_logits.detach() / T)
        soft_student = torch.sigmoid(student_logits / T)

        kd_loss = F.binary_cross_entropy(
            soft_student.clamp(1e-7, 1 - 1e-7),
            soft_teacher.clamp(1e-7, 1 - 1e-7),
            reduction="mean",
        ) * (T * T)

        # --- 2. Asymmetric Task Loss ---
        task_loss = self._asymmetric_loss(student_logits, targets)

        # --- 3. Feature Alignment Loss ---
        feature_loss = torch.tensor(0.0, device=student_logits.device)
        if student_features is not None and teacher_features is not None and self.beta > 0:
            feature_loss = F.mse_loss(student_features, teacher_features.detach())

        # --- 4. Threshold Alignment Loss ---
        threshold_loss = torch.tensor(0.0, device=student_logits.device)
        if thresholds is not None and self.gamma > 0:
            threshold_loss = self._threshold_alignment_loss(
                student_logits, teacher_logits.detach(), thresholds
            )

        # --- Combined ---
        total = (
            self.alpha * kd_loss
            + (1 - self.alpha) * task_loss
            + self.beta * feature_loss
            + self.gamma * threshold_loss
        )

        return {
            "total": total,
            "kd_loss": kd_loss.detach(),
            "task_loss": task_loss.detach(),
            "feature_loss": feature_loss.detach(),
            "threshold_loss": threshold_loss.detach(),
            "temperature": torch.tensor(T),
        }

    def _asymmetric_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """AsymmetricLossV2 matching the teacher's training loss."""
        # Label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        probs = torch.sigmoid(logits).clamp(min=1e-8, max=1 - 1e-8)
        probs_neg = 1.0 - probs

        # Probability shifting for easy negatives
        if self.asl_clip > 0:
            probs_neg = (probs_neg + self.asl_clip).clamp(max=1.0)

        loss_pos = targets * torch.log(probs)
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=1e-8))

        # Asymmetric focusing
        pt = probs * targets + probs_neg * (1 - targets)
        gamma = self.asl_gamma_pos * targets + self.asl_gamma_neg * (1 - targets)
        focal_weight = (1.0 - pt).clamp(min=0.0) ** gamma

        return -((loss_pos + loss_neg) * focal_weight).mean()

    def _threshold_alignment_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        thresholds: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize when student crosses thresholds differently than teacher.

        For each class, compute the margin from the threshold for both
        teacher and student. Penalize when the student's margin is in the
        wrong direction relative to the teacher's decision.
        """
        student_probs = torch.sigmoid(student_logits)
        teacher_probs = torch.sigmoid(teacher_logits)

        t = thresholds.unsqueeze(0)  # [1, C]

        # Teacher's decision: above or below threshold
        teacher_above = (teacher_probs >= t).float()

        # Student's distance from the threshold, signed by teacher's decision
        # If teacher says positive (above threshold), student should also be above
        student_margin = student_probs - t
        desired_sign = 2.0 * teacher_above - 1.0  # +1 for above, -1 for below
        signed_margin = student_margin * desired_sign

        # Penalize negative margins (student disagrees with teacher)
        penalties = F.relu(-signed_margin)

        return penalties.mean()
