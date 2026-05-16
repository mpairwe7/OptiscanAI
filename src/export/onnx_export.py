"""
ONNX export with validation and optimization.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ONNXExporter:
    """Export PyTorch model to ONNX with validation."""

    @staticmethod
    def export(
        model: nn.Module,
        output_path: str | Path,
        input_shape: tuple = (1, 3, 224, 224),
        opset_version: int = 17,
        device: str = "cpu",
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model = model.to(device).eval()
        dummy = torch.randn(*input_shape, device=device)

        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        size_mb = output_path.stat().st_size / 1e6
        logger.info(f"ONNX exported: {output_path} ({size_mb:.1f} MB)")
        return output_path

    @staticmethod
    def validate(
        onnx_path: str | Path,
        model: nn.Module,
        input_shape: tuple = (1, 3, 224, 224),
        device: str = "cpu",
        rtol: float = 1e-3,
    ) -> dict:
        """Validate ONNX output matches PyTorch within tolerance."""
        try:
            import onnxruntime as ort
        except ImportError:
            return {"valid": False, "error": "onnxruntime not installed"}

        model = model.to(device).eval()
        dummy = torch.randn(*input_shape, device=device)

        with torch.no_grad():
            pt_out = model(dummy).cpu().numpy()

        sess = ort.InferenceSession(str(onnx_path))
        ort_out = sess.run(None, {"input": dummy.cpu().numpy()})[0]

        max_diff = float(np.abs(pt_out - ort_out).max())
        matches = max_diff < rtol

        return {"valid": matches, "max_diff": max_diff, "rtol": rtol}
