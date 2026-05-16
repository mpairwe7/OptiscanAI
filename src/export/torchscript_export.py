"""
TorchScript export via tracing.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TorchScriptExporter:
    """Export via torch.jit.trace (no dynamic control flow in these models)."""

    @staticmethod
    def trace(
        model: nn.Module,
        output_path: str | Path,
        input_shape: tuple = (1, 3, 224, 224),
        device: str = "cpu",
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model = model.to(device).eval()
        dummy = torch.randn(*input_shape, device=device)

        with torch.no_grad():
            traced = torch.jit.trace(model, dummy)
        traced.save(str(output_path))

        size_mb = output_path.stat().st_size / 1e6
        logger.info(f"TorchScript exported: {output_path} ({size_mb:.1f} MB)")
        return output_path

    @staticmethod
    def validate(
        script_path: str | Path,
        model: nn.Module,
        input_shape: tuple = (1, 3, 224, 224),
        device: str = "cpu",
        rtol: float = 1e-4,
    ) -> dict:
        model = model.to(device).eval()
        dummy = torch.randn(*input_shape, device=device)

        with torch.no_grad():
            pt_out = model(dummy).cpu().numpy()

        loaded = torch.jit.load(str(script_path), map_location=device)
        with torch.no_grad():
            ts_out = loaded(dummy).cpu().numpy()

        max_diff = float(np.abs(pt_out - ts_out).max())
        return {"valid": max_diff < rtol, "max_diff": max_diff}
