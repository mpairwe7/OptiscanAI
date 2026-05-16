"""
Learning Rate Finder - exponential LR sweep to find optimal learning rate.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class LRFinder:
    """Exponential LR range test (Smith 2017)."""

    def __init__(self):
        self.lrs: list[float] = []
        self.losses: list[float] = []

    def find(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        min_lr: float = 1e-7,
        max_lr: float = 10.0,
        num_steps: int = 100,
    ) -> float:
        """Run LR sweep and return suggested LR."""
        # Save model state
        original_state = copy.deepcopy(model.state_dict())
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=min_lr)
        lr_mult = (max_lr / min_lr) ** (1 / num_steps)

        best_loss = float("inf")
        loader_iter = iter(train_loader)

        for step in range(num_steps):
            # Get batch (cycle if needed)
            try:
                images, targets = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                images, targets = next(loader_iter)

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            lr = min_lr * (lr_mult ** step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            self.lrs.append(lr)
            self.losses.append(loss.item())

            if loss.item() < best_loss:
                best_loss = loss.item()
            if loss.item() > best_loss * 4:
                break  # Diverging

        # Restore model
        model.load_state_dict(original_state)

        suggested = self.suggest_lr()
        logger.info(f"LR Finder: suggested LR = {suggested:.2e}")
        return suggested

    def suggest_lr(self) -> float:
        """Return LR at steepest descent (10x before minimum loss)."""
        if not self.losses:
            return 1e-4
        # Smooth losses
        smoothed = np.convolve(self.losses, np.ones(5) / 5, mode="valid")
        if len(smoothed) < 3:
            return self.lrs[np.argmin(self.losses)]
        # Steepest negative gradient
        gradients = np.gradient(smoothed)
        idx = np.argmin(gradients)
        # Map back to LR (offset by smoothing window)
        lr_idx = min(idx + 2, len(self.lrs) - 1)
        return self.lrs[lr_idx] / 10  # Use 10x lower than steepest point

    def plot(self, save_dir: Path):
        """Save LR finder plot."""
        from src.visualization.ieee_style import ieee_figure, ieee_style, save_ieee
        save_dir.mkdir(parents=True, exist_ok=True)
        with ieee_style():
            fig, ax = ieee_figure(1, 1, width="single")
            ax.plot(self.lrs[: len(self.losses)], self.losses, lw=1.5)
            suggested = self.suggest_lr()
            ax.axvline(suggested, ls="--", color="red", lw=1, label=f"Suggested: {suggested:.1e}")
            ax.set_xscale("log")
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel("Loss")
            ax.set_title("LR Range Test")
            ax.legend()
            save_ieee(fig, save_dir / "fig_lr_finder")
