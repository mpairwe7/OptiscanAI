"""
Ray Serve deployment configuration for RetinalAI ViGNN model serving.

Provides dynamic batching, autoscaling, health checks, and structured
logging for production GPU/CPU inference.  All Ray imports are guarded
so the module can be imported safely in environments without Ray.

Launch (requires ``ray[serve]``):
    ray start --head
    python -m backend.app.serving.ray_serve_config

Or via the module-level helpers:
    from backend.app.serving.ray_serve_config import deploy_model, undeploy_model
    deploy_model("models/model_vignn_rank1.pth")
"""

from __future__ import annotations

import io
import logging
import os
import time
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guard Ray imports -- the module stays importable without ray installed
# ---------------------------------------------------------------------------

_RAY_AVAILABLE: bool = False

try:
    import ray
    from ray import serve

    _RAY_AVAILABLE = True
except ImportError:
    ray = None  # type: ignore[assignment]
    serve = None  # type: ignore[assignment]
    logger.info(
        "ray[serve] is not installed; " "Ray Serve deployment will not be available at runtime"
    )


# ---------------------------------------------------------------------------
# ImageNet preprocessing (matches training pipeline)
# ---------------------------------------------------------------------------

_IMG_SIZE: int = 224
_IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
_IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]


def _get_transform():
    """Build the torchvision transform chain (lazy import)."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((_IMG_SIZE, _IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def preprocess_single(image_bytes: bytes) -> torch.Tensor:
    """Convert raw image bytes to a single preprocessed tensor.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of a JPEG/PNG fundus image.

    Returns
    -------
    torch.Tensor
        Float tensor of shape ``(1, 3, 224, 224)`` ready for the model.
    """
    transform = _get_transform()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(img).unsqueeze(0)


# ---------------------------------------------------------------------------
# Autoscaling & deployment configuration (pulled from central settings)
# ---------------------------------------------------------------------------


def _load_serve_settings() -> Dict[str, Any]:
    """Load Ray Serve settings from the central config singleton.

    Returns a plain dict so we can feed it to the ``@serve.deployment``
    decorator without importing pydantic at decoration time.
    """
    try:
        from backend.app.core.config import settings

        rs = settings.ray
        return {
            "batch_max_size": rs.batch_max_size,
            "batch_timeout_s": rs.batch_timeout_s,
            "min_replicas": rs.min_replicas,
            "max_replicas": rs.max_replicas,
            "target_ongoing_requests": rs.target_ongoing_requests,
        }
    except Exception:
        logger.warning("Could not load RayServeSettings from config; using defaults")
        return {
            "batch_max_size": 16,
            "batch_timeout_s": 0.1,
            "min_replicas": 1,
            "max_replicas": 8,
            "target_ongoing_requests": 10,
        }


_SERVE_SETTINGS = _load_serve_settings()

_DEPLOYMENT_CONFIG: Dict[str, Any] = {
    "name": "retinalai-vignn",
    "num_replicas": _SERVE_SETTINGS["min_replicas"],
    "max_ongoing_requests": 100,
    "ray_actor_options": {
        "num_gpus": 0.5,
        "num_cpus": 2,
        "memory": 4 * 1024 * 1024 * 1024,  # 4 GiB
    },
    "autoscaling_config": {
        "min_replicas": _SERVE_SETTINGS["min_replicas"],
        "max_replicas": _SERVE_SETTINGS["max_replicas"],
        "target_ongoing_requests": _SERVE_SETTINGS["target_ongoing_requests"],
        "upscale_delay_s": 30,
        "downscale_delay_s": 300,
    },
    "health_check_period_s": 30,
    "health_check_timeout_s": 15,
    "graceful_shutdown_timeout_s": 60,
}


# ---------------------------------------------------------------------------
# Deployment class -- defined only when Ray is available
# ---------------------------------------------------------------------------

if _RAY_AVAILABLE:

    @serve.deployment(
        name=_DEPLOYMENT_CONFIG["name"],
        num_replicas=_DEPLOYMENT_CONFIG["num_replicas"],
        max_ongoing_requests=_DEPLOYMENT_CONFIG["max_ongoing_requests"],
        ray_actor_options=_DEPLOYMENT_CONFIG["ray_actor_options"],
        health_check_period_s=_DEPLOYMENT_CONFIG["health_check_period_s"],
        health_check_timeout_s=_DEPLOYMENT_CONFIG["health_check_timeout_s"],
        graceful_shutdown_timeout_s=_DEPLOYMENT_CONFIG["graceful_shutdown_timeout_s"],
    )
    class RetinalModelDeployment:
        """Ray Serve deployment for ViGNN retinal disease classification.

        Lifecycle
        ---------
        1. ``__init__`` loads the ViGNN model + ClinicalKnowledgeGraph onto
           the assigned device (GPU preferred, CPU fallback).
        2. Incoming requests hit ``__call__`` which delegates to the batched
           handler ``_handle_batch``.
        3. ``_handle_batch`` stacks individual images into a single tensor,
           runs one forward pass, then splits the results back.
        4. ``check_health`` runs a dummy forward pass; a raised exception
           tells Ray Serve to restart the replica.

        Parameters
        ----------
        model_path : str or None
            Path to a ``.pth`` checkpoint.  When *None* the path is read
            from ``settings.model_path``.
        device : str
            ``"auto"`` (default) picks CUDA when available.
        num_classes : int
            Must match the checkpoint (default 45).
        """

        def __init__(
            self,
            model_path: Optional[str] = None,
            device: str = "auto",
            num_classes: int = 45,
        ) -> None:
            # Resolve device
            if device == "auto":
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self._device = torch.device(device)

            self._num_classes = num_classes
            self._model: Optional[torch.nn.Module] = None
            self._disease_names: List[str] = []
            self._transform = _get_transform()

            # Runtime counters
            self._request_count: int = 0
            self._error_count: int = 0
            self._total_latency_ms: float = 0.0

            self._load_model(model_path)

            logger.info(
                "RetinalModelDeployment initialised",
                extra={
                    "device": str(self._device),
                    "num_classes": self._num_classes,
                    "model_path": model_path or "default",
                },
            )

        # -- model loading ------------------------------------------------

        def _load_model(self, model_path: Optional[str]) -> None:
            """Load the ViGNN checkpoint and knowledge graph."""
            from src.models.vignn import (
                create_knowledge_graph,
                create_vignn_model,
            )

            # Resolve path from settings when not supplied explicitly
            if model_path is None:
                try:
                    from backend.app.core.config import settings

                    model_path = settings.model_path
                except Exception:
                    model_path = "models/model_vignn_rank1.pth"

            kg = create_knowledge_graph()
            self._disease_names = kg.disease_names

            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                nc = checkpoint.get("num_classes", len(self._disease_names))
                self._num_classes = nc

                self._model = create_vignn_model(
                    num_classes=nc,
                    clinical_knowledge_graph=kg,
                )

                if "model_state_dict" in checkpoint:
                    self._model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                    logger.info(
                        "Loaded ViGNN weights from checkpoint",
                        extra={
                            "model_path": model_path,
                            "best_f1": checkpoint.get("best_f1"),
                            "best_auc": checkpoint.get("best_auc"),
                        },
                    )
                else:
                    logger.warning(
                        "Checkpoint has no model_state_dict key; "
                        "using randomly-initialised weights"
                    )
            else:
                logger.warning(
                    "Model file not found; creating un-trained ViGNN",
                    extra={"model_path": model_path},
                )
                self._model = create_vignn_model(
                    num_classes=self._num_classes,
                    clinical_knowledge_graph=kg,
                )

            self._model = self._model.to(self._device)
            self._model.eval()

            # Optional torch.compile on CUDA
            if self._device.type == "cuda":
                try:
                    self._model = torch.compile(self._model, mode="reduce-overhead")
                    logger.info("torch.compile applied (reduce-overhead)")
                except Exception as exc:
                    logger.debug("torch.compile unavailable: %s", exc)

        # -- preprocessing ------------------------------------------------

        def _preprocess_bytes(self, image_bytes: bytes) -> torch.Tensor:
            """Decode and normalise a single image.

            Returns
            -------
            torch.Tensor
                Shape ``(3, 224, 224)`` -- no batch dimension.
            """
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return self._transform(img)

        # -- batched inference --------------------------------------------

        @serve.batch(
            max_batch_size=_SERVE_SETTINGS["batch_max_size"],
            batch_wait_timeout_s=_SERVE_SETTINGS["batch_timeout_s"],
        )
        async def _handle_batch(self, image_list: List[bytes]) -> List[Dict[str, Any]]:
            """Process a dynamically-batched list of images.

            Ray Serve collects up to ``batch_max_size`` concurrent calls
            and delivers them here as a single list.  We stack the
            pre-processed tensors into one batch, run a single forward
            pass, then split the results back per-image.

            Parameters
            ----------
            image_list : list[bytes]
                Raw image bytes collected by Ray Serve batching.

            Returns
            -------
            list[dict]
                One result dict per image in the same order.
            """
            batch_start = time.perf_counter()
            batch_size = len(image_list)

            try:
                # Preprocess all images and stack into a single tensor
                tensors: List[torch.Tensor] = []
                for raw in image_list:
                    tensors.append(self._preprocess_bytes(raw))

                batch_tensor = torch.stack(tensors, dim=0).to(self._device)

                # Single forward pass for the entire batch
                with torch.no_grad():
                    logits = self._model(batch_tensor)
                    probs = torch.sigmoid(logits).cpu().numpy()

                # Split results back to individual dicts
                results: List[Dict[str, Any]] = []
                for idx in range(batch_size):
                    sample_probs = probs[idx]
                    predictions = []
                    for cls_idx, prob_val in enumerate(sample_probs):
                        if cls_idx < len(self._disease_names):
                            predictions.append(
                                {
                                    "disease": self._disease_names[cls_idx],
                                    "probability": float(prob_val),
                                }
                            )
                    predictions.sort(key=lambda d: d["probability"], reverse=True)

                    results.append(
                        {
                            "predictions": predictions,
                            "top_5": predictions[:5],
                            "num_classes": self._num_classes,
                        }
                    )

                batch_latency_ms = (time.perf_counter() - batch_start) * 1000
                self._request_count += batch_size
                self._total_latency_ms += batch_latency_ms

                logger.info(
                    "Batch inference completed",
                    extra={
                        "batch_size": batch_size,
                        "latency_ms": round(batch_latency_ms, 2),
                        "device": str(self._device),
                    },
                )

                return results

            except Exception as exc:
                self._error_count += batch_size
                logger.error(
                    "Batch inference failed: %s",
                    exc,
                    exc_info=True,
                    extra={"batch_size": batch_size},
                )
                return [
                    {"error": "inference_failed", "detail": str(exc)} for _ in range(batch_size)
                ]

        # -- public entry point -------------------------------------------

        async def __call__(self, request: Any) -> Dict[str, Any]:
            """Handle a single inference request.

            Accepts Starlette ``Request`` objects (multipart / raw body),
            plain ``bytes``, or a ``dict`` with a base64-encoded ``image``
            key.
            """
            try:
                if hasattr(request, "body"):
                    image_bytes: bytes = await request.body()
                elif isinstance(request, bytes):
                    image_bytes = request
                elif isinstance(request, dict):
                    import base64

                    image_bytes = base64.b64decode(request.get("image", ""))
                else:
                    return {
                        "error": "invalid_request",
                        "detail": "Expected raw bytes, Starlette Request, or dict with 'image' key",
                    }

                result = await self._handle_batch(image_bytes)
                return result

            except Exception as exc:
                self._error_count += 1
                logger.error("Request handling failed: %s", exc, exc_info=True)
                return {"error": "request_failed", "detail": str(exc)}

        # -- health check -------------------------------------------------

        def check_health(self) -> None:
            """Ray Serve health check.

            A normal return signals healthy; a raised exception triggers
            replica restart.  We run a small dummy tensor through the model
            to verify the full inference pipeline is functional.
            """
            dummy = torch.randn(1, 3, _IMG_SIZE, _IMG_SIZE, device=self._device)
            with torch.no_grad():
                output = self._model(dummy)

            if output is None or output.shape[0] != 1:
                raise RuntimeError("Health check failed: unexpected model output shape")

            logger.debug(
                "Health check passed",
                extra={
                    "total_requests": self._request_count,
                    "error_count": self._error_count,
                },
            )

        # -- metrics ------------------------------------------------------

        def get_metrics(self) -> Dict[str, Any]:
            """Return runtime metrics for monitoring dashboards.

            Returns
            -------
            dict
                Keys: ``total_requests``, ``error_count``, ``error_rate``,
                ``avg_latency_ms``, ``device``.
            """
            avg_latency = (
                self._total_latency_ms / self._request_count if self._request_count > 0 else 0.0
            )
            return {
                "total_requests": self._request_count,
                "error_count": self._error_count,
                "error_rate": round(self._error_count / max(self._request_count, 1), 4),
                "avg_latency_ms": round(avg_latency, 2),
                "device": str(self._device),
            }

    # Bind a default deployment handle for ``serve run`` CLI
    _default_deployment = RetinalModelDeployment.bind()


# ---------------------------------------------------------------------------
# Module-level deploy / undeploy helpers
# ---------------------------------------------------------------------------

# Track the running deployment application name for undeploy
_ACTIVE_APP_NAME: Optional[str] = None


def deploy_model(
    model_path: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    device: str = "auto",
    num_classes: int = 45,
    app_name: str = "retinalai_vignn",
    route_prefix: str = "/predict",
) -> None:
    """Deploy the ViGNN model on Ray Serve.

    Parameters
    ----------
    model_path : str or None
        Path to the ``.pth`` checkpoint.  Falls back to ``settings.model_path``.
    host : str
        Bind address for the HTTP proxy (default ``0.0.0.0``).
    port : int
        Port for the HTTP proxy (default ``8000``).
    device : str
        Torch device string (``"auto"``, ``"cpu"``, ``"cuda"``).
    num_classes : int
        Number of output classes.
    app_name : str
        Ray Serve application name.
    route_prefix : str
        HTTP route prefix for the deployment.

    Raises
    ------
    ImportError
        If Ray is not installed.
    RuntimeError
        If Ray initialisation fails.
    """
    global _ACTIVE_APP_NAME

    if not _RAY_AVAILABLE:
        raise ImportError("ray[serve] is required: pip install 'ray[serve]'")

    # Initialise Ray (idempotent)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
        logger.info("Ray cluster initialised")

    serve.start(http_options={"host": host, "port": port})

    bound = RetinalModelDeployment.bind(
        model_path=model_path,
        device=device,
        num_classes=num_classes,
    )
    serve.run(bound, name=app_name, route_prefix=route_prefix)

    _ACTIVE_APP_NAME = app_name

    logger.info(
        "Ray Serve deployment started",
        extra={
            "app_name": app_name,
            "route_prefix": route_prefix,
            "host": host,
            "port": port,
            "model_path": model_path or "default",
        },
    )


def undeploy_model(app_name: Optional[str] = None) -> None:
    """Undeploy the running RetinalAI model from Ray Serve.

    Parameters
    ----------
    app_name : str or None
        Application name to remove.  Defaults to the name used in the
        most recent ``deploy_model`` call.

    Raises
    ------
    ImportError
        If Ray is not installed.
    """
    global _ACTIVE_APP_NAME

    if not _RAY_AVAILABLE:
        raise ImportError("ray[serve] is required: pip install 'ray[serve]'")

    name = app_name or _ACTIVE_APP_NAME
    if name is None:
        logger.warning("No active deployment to undeploy")
        return

    try:
        serve.delete(name)
        logger.info("Deployment undeployed", extra={"app_name": name})
    except Exception as exc:
        logger.error("Failed to undeploy: %s", exc, extra={"app_name": name})
        raise
    finally:
        if name == _ACTIVE_APP_NAME:
            _ACTIVE_APP_NAME = None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    deploy_model()
