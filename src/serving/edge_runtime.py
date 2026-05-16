"""Unified edge inference runtime for ONNX, CoreML, and quantized models.

Provides a single ``EdgeRuntime`` class that can load and run inference
across heterogeneous model formats while guaranteeing output schema
parity with ``ModelService.predict()``.

All heavy third-party imports (``onnxruntime``, ``coremltools``) are lazy
so that the module can be imported safely in any environment.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Project root for src imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy config / disease mapping imports
# ---------------------------------------------------------------------------
try:
    from backend.app.core.config import settings as _settings

    _EDGE_CFG = _settings.edge
except ImportError:
    _settings = None  # type: ignore[assignment]
    _EDGE_CFG = None
    logger.info(
        "backend.app.core.config not available -- "
        "EdgeRuntime will use constructor defaults"
    )

try:
    from backend.app.core.model_service import DISEASE_NAMES as _DISEASE_NAMES
    from src.data.datamodule import DISEASE_COLUMNS as _DISEASE_COLUMNS
except ImportError:
    _DISEASE_NAMES = {}
    _DISEASE_COLUMNS = []
    logger.info(
        "Disease name mappings not available -- "
        "EdgeRuntime will use numeric indices"
    )

# ---------------------------------------------------------------------------
# Standard preprocessing (identical to ModelService._transform)
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_edge_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


class EdgeRuntime:
    """Unified edge inference across ONNX, CoreML, and quantized PyTorch.

    Each format is loaded on-demand and kept in memory until the runtime
    instance is garbage-collected.  Output from every ``predict_*`` method
    follows the same schema as ``ModelService.predict()`` so callers do
    not need format-specific handling.

    Parameters
    ----------
    onnx_model_path:
        Override for ``settings.edge.onnx_model_path``.
    coreml_model_path:
        Override for ``settings.edge.coreml_model_path``.
    quantized_model_path:
        Override for ``settings.edge.quantized_model_path``.
    parity_tolerance:
        Numerical tolerance for parity validation.
    disease_columns:
        Ordered list of disease codes.  Defaults to ``DISEASE_COLUMNS``
        from the data module.
    disease_names:
        ``{code: full_name}`` mapping.  Defaults to ``DISEASE_NAMES``
        from the model service.
    """

    def __init__(
        self,
        onnx_model_path: Optional[str] = None,
        coreml_model_path: Optional[str] = None,
        quantized_model_path: Optional[str] = None,
        parity_tolerance: float = 1e-4,
        disease_columns: Optional[list[str]] = None,
        disease_names: Optional[dict[str, str]] = None,
    ) -> None:
        # Paths -- prefer explicit args, then settings, then defaults
        cfg = _EDGE_CFG
        self._onnx_path = onnx_model_path or (cfg.onnx_model_path if cfg else "models/export/model.onnx")
        self._coreml_path = coreml_model_path or (cfg.coreml_model_path if cfg else "models/export/model.mlpackage")
        self._quantized_path = quantized_model_path or (cfg.quantized_model_path if cfg else "models/export/model_int8.pth")
        self._parity_tolerance = parity_tolerance or (cfg.parity_tolerance if cfg else 1e-4)

        # Disease metadata
        self._disease_columns: list[str] = disease_columns or list(_DISEASE_COLUMNS)
        self._disease_names: dict[str, str] = disease_names or dict(_DISEASE_NAMES)

        # Loaded runtime handles
        self._onnx_session: Any = None
        self._coreml_model: Any = None
        self._quantized_model: Any = None
        self._quantized_precision: str = ""

        logger.info(
            "EdgeRuntime created (onnx=%s, coreml=%s, quantized=%s)",
            self._onnx_path,
            self._coreml_path,
            self._quantized_path,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_onnx(
        self,
        model_path: Optional[str] = None,
        providers: Optional[list[str]] = None,
    ) -> None:
        """Load an ONNX model via ``onnxruntime``.

        Parameters
        ----------
        model_path:
            Path to ``.onnx`` file.  Defaults to constructor / config value.
        providers:
            ONNX Runtime execution providers.  Defaults to
            ``["CUDAExecutionProvider", "CPUExecutionProvider"]``.
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for ONNX inference. "
                "Install with: pip install onnxruntime"
            ) from exc

        path = model_path or self._onnx_path
        if providers is None:
            available = ort.get_available_providers()
            providers = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

        self._onnx_session = ort.InferenceSession(str(path), providers=providers)
        logger.info(
            "ONNX model loaded: %s (providers=%s)",
            path,
            self._onnx_session.get_providers(),
        )

    def load_coreml(self, model_path: Optional[str] = None) -> None:
        """Load a CoreML model via ``coremltools``.

        Parameters
        ----------
        model_path:
            Path to ``.mlpackage`` or ``.mlmodel``.  Defaults to
            constructor / config value.
        """
        try:
            import coremltools as ct
        except ImportError as exc:
            raise ImportError(
                "coremltools is required for CoreML inference. "
                "Install with: pip install coremltools"
            ) from exc

        path = model_path or self._coreml_path
        self._coreml_model = ct.models.MLModel(str(path))
        logger.info("CoreML model loaded: %s", path)

    def load_quantized(
        self,
        model_path: Optional[str] = None,
        precision: str = "int8",
    ) -> None:
        """Load a quantized PyTorch model checkpoint.

        Parameters
        ----------
        model_path:
            Path to ``.pth`` checkpoint.  Defaults to constructor / config value.
        precision:
            Quantization precision tag (``"int8"``, ``"fp16"``).
        """
        path = model_path or self._quantized_path
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)

        # Support both raw state_dict and wrapped checkpoint formats
        if isinstance(checkpoint, torch.nn.Module):
            self._quantized_model = checkpoint
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # Need to rebuild model architecture -- try to import the builder
            try:
                from src.models.vignn import ClinicalKnowledgeGraph, create_vignn_model

                disease_cols = self._disease_columns
                kg = ClinicalKnowledgeGraph(disease_names=disease_cols)
                model = create_vignn_model(
                    num_classes=len(disease_cols),
                    clinical_knowledge_graph=kg,
                )
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)

                # Apply dynamic quantization if this is a non-quantized checkpoint
                if precision == "int8":
                    model = torch.ao.quantization.quantize_dynamic(
                        model.cpu().eval(),
                        {torch.nn.Linear},
                        dtype=torch.qint8,
                    )
                elif precision == "fp16":
                    model = model.half()

                self._quantized_model = model
            except Exception:
                logger.exception(
                    "Failed to rebuild model from state_dict -- "
                    "trying direct load"
                )
                self._quantized_model = checkpoint
        else:
            self._quantized_model = checkpoint

        if isinstance(self._quantized_model, torch.nn.Module):
            self._quantized_model.eval()

        self._quantized_precision = precision
        logger.info(
            "Quantized model loaded: %s (precision=%s)", path, precision
        )

    # ------------------------------------------------------------------
    # Internal preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Resize to 224x224, apply ImageNet normalisation, return (1,3,224,224) numpy array."""
        tensor = _edge_transform(image.convert("RGB")).unsqueeze(0)
        return tensor.numpy()

    def _preprocess_torch(self, image: Image.Image) -> torch.Tensor:
        """Same preprocessing, returning a PyTorch tensor."""
        return _edge_transform(image.convert("RGB")).unsqueeze(0)

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _build_result(
        self,
        probs: np.ndarray,
        threshold: Optional[float],
        elapsed_ms: float,
        format_name: str,
    ) -> dict:
        """Build a prediction result dict matching ModelService.predict() schema."""
        if threshold is None:
            threshold = 0.5
        threshold_val = float(threshold)

        disease_codes = self._disease_columns
        # Handle shape: probs might be (1, N) or (N,)
        if probs.ndim > 1:
            probs = probs[0]

        num_classes = len(disease_codes)
        if probs.shape[0] != num_classes:
            logger.warning(
                "Output size mismatch: model=%d, expected=%d. "
                "Truncating/padding.",
                probs.shape[0],
                num_classes,
            )
            if probs.shape[0] > num_classes:
                probs = probs[:num_classes]
            else:
                padded = np.zeros(num_classes, dtype=np.float32)
                padded[: probs.shape[0]] = probs
                probs = padded

        all_probs = {
            code: {
                "probability": float(probs[i]),
                "name": self._disease_names.get(code, code),
                "threshold": threshold_val,
            }
            for i, code in enumerate(disease_codes)
        }

        detected = [
            {
                "code": code,
                "name": self._disease_names.get(code, code),
                "probability": float(probs[i]),
                "threshold": threshold_val,
                "confidence": (
                    "high" if probs[i] > 0.8
                    else "medium" if probs[i] > 0.5
                    else "low"
                ),
            }
            for i, code in enumerate(disease_codes)
            if probs[i] > threshold_val
        ]
        detected.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "predictions": detected,
            "total_detected": len(detected),
            "all_probabilities": all_probs,
            "inference_ms": round(elapsed_ms, 2),
            "threshold": round(threshold_val, 4),
            "threshold_source": "scalar",
            "runtime_format": format_name,
        }

    # ------------------------------------------------------------------
    # Predict methods
    # ------------------------------------------------------------------

    def predict_onnx(
        self,
        image: Image.Image,
        threshold: Optional[float] = None,
    ) -> dict:
        """Run inference via ONNX Runtime.

        Raises ``RuntimeError`` if the ONNX model is not loaded.
        """
        if self._onnx_session is None:
            raise RuntimeError(
                "ONNX model not loaded. Call load_onnx() first."
            )

        input_array = self._preprocess(image)
        input_name = self._onnx_session.get_inputs()[0].name

        t0 = time.perf_counter()
        outputs = self._onnx_session.run(None, {input_name: input_array})
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Apply sigmoid to logits
        logits = outputs[0]
        probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))

        return self._build_result(
            probs.astype(np.float32), threshold, elapsed_ms, "onnx"
        )

    def predict_coreml(
        self,
        image: Image.Image,
        threshold: Optional[float] = None,
    ) -> dict:
        """Run inference via CoreML.

        Raises ``RuntimeError`` if the CoreML model is not loaded.
        """
        if self._coreml_model is None:
            raise RuntimeError(
                "CoreML model not loaded. Call load_coreml() first."
            )

        try:
            import coremltools as ct  # noqa: F401,F811 — optional dep, re-import for clarity
        except ImportError as exc:
            raise ImportError(
                "coremltools is required for CoreML inference."
            ) from exc

        # CoreML expects a PIL Image or dict input
        resized = image.convert("RGB").resize((224, 224))

        t0 = time.perf_counter()
        prediction = self._coreml_model.predict({"retinal_image": resized})
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Extract output -- CoreML models may name outputs differently
        output_key = list(prediction.keys())[0]
        raw_output = np.array(prediction[output_key]).flatten()

        # Apply sigmoid if outputs look like logits (range outside [0,1])
        if raw_output.min() < -0.5 or raw_output.max() > 1.5:
            probs = 1.0 / (1.0 + np.exp(-raw_output.astype(np.float64)))
        else:
            probs = raw_output

        return self._build_result(
            probs.astype(np.float32), threshold, elapsed_ms, "coreml"
        )

    def predict_quantized(
        self,
        image: Image.Image,
        threshold: Optional[float] = None,
    ) -> dict:
        """Run inference via quantized PyTorch model.

        Raises ``RuntimeError`` if the quantized model is not loaded.
        """
        if self._quantized_model is None:
            raise RuntimeError(
                "Quantized model not loaded. Call load_quantized() first."
            )

        tensor = self._preprocess_torch(image)
        if self._quantized_precision == "fp16":
            tensor = tensor.half()

        t0 = time.perf_counter()
        with torch.no_grad():
            output = self._quantized_model(tensor)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if isinstance(output, dict):
            output = output.get("logits", list(output.values())[0])

        probs = torch.sigmoid(output).cpu().numpy()

        return self._build_result(
            probs.astype(np.float32), threshold, elapsed_ms,
            f"quantized_{self._quantized_precision}",
        )

    # ------------------------------------------------------------------
    # Parity validation
    # ------------------------------------------------------------------

    def validate_parity(
        self,
        image: Image.Image,
        format_name: str,
        reference_output: dict,
        tolerance: Optional[float] = None,
    ) -> dict:
        """Validate that an edge format produces outputs within tolerance
        of a reference (typically the PyTorch FP32 model).

        Parameters
        ----------
        image:
            Input PIL image.
        format_name:
            One of ``"onnx"``, ``"coreml"``, ``"quantized"``.
        reference_output:
            Dict from ``ModelService.predict()`` or another predict method.
        tolerance:
            Max absolute difference.  Defaults to ``self._parity_tolerance``.

        Returns
        -------
        dict
            ``{"passed": bool, "max_diff": float, "tolerance": float,
              "format": str, "mismatched_classes": list}``
        """
        tol = tolerance if tolerance is not None else self._parity_tolerance

        # Run edge prediction
        predict_fn = {
            "onnx": self.predict_onnx,
            "coreml": self.predict_coreml,
            "quantized": self.predict_quantized,
        }.get(format_name)

        if predict_fn is None:
            return {
                "passed": False,
                "error": f"Unknown format: {format_name}",
                "max_diff": float("inf"),
                "tolerance": tol,
                "format": format_name,
                "mismatched_classes": [],
            }

        edge_result = predict_fn(image)

        # Compare probabilities
        ref_probs = reference_output.get("all_probabilities", {})
        edge_probs = edge_result.get("all_probabilities", {})

        max_diff = 0.0
        mismatched: list[dict[str, Any]] = []

        for code in self._disease_columns:
            ref_p = ref_probs.get(code, {}).get("probability", 0.0)
            edge_p = edge_probs.get(code, {}).get("probability", 0.0)
            diff = abs(ref_p - edge_p)
            max_diff = max(max_diff, diff)
            if diff > tol:
                mismatched.append({
                    "code": code,
                    "reference": round(ref_p, 6),
                    "edge": round(edge_p, 6),
                    "diff": round(diff, 6),
                })

        passed = max_diff <= tol

        return {
            "passed": passed,
            "max_diff": round(max_diff, 6),
            "tolerance": tol,
            "format": format_name,
            "mismatched_classes": mismatched,
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_loaded_formats(self) -> list[str]:
        """Return the list of currently loaded model formats."""
        loaded: list[str] = []
        if self._onnx_session is not None:
            loaded.append("onnx")
        if self._coreml_model is not None:
            loaded.append("coreml")
        if self._quantized_model is not None:
            loaded.append(f"quantized_{self._quantized_precision}")
        return loaded
