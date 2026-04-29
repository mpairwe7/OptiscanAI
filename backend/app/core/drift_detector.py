"""
Enhanced drift detection for RetinalAI production monitoring.

Wraps the core DataDriftDetector and ModelDriftDetector from
src/monitoring/drift, adding optional NannyML/Evidently integration,
automatic periodic checks, webhook alerting, and event-bus emission.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any

import numpy as np

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional third-party imports (graceful degradation)
# ---------------------------------------------------------------------------

_nannyml_available = False
_evidently_available = False

try:
    import nannyml  # noqa: F401

    _nannyml_available = True
except ImportError:
    pass

try:
    import evidently  # noqa: F401

    _evidently_available = True
except ImportError:
    pass


class EnhancedDriftDetector:
    """Production-grade drift detector combining multiple strategies.

    Integrates:
    * PSI + Kolmogorov-Smirnov (via ``DataDriftDetector``)
    * Confidence-drop monitoring (via ``ModelDriftDetector``)
    * NannyML UnivariateDriftCalculator (optional)
    * Evidently DataDriftPreset (optional)

    Automatically triggers a full drift check every ``check_interval``
    predictions recorded through :meth:`record_prediction`.
    """

    def __init__(self) -> None:
        cfg = settings.drift

        # Core detector parameters (detectors created after reference init)
        self._psi_threshold: float = cfg.psi_threshold
        self._ks_threshold: float = cfg.ks_threshold
        self._confidence_drop_threshold: float = cfg.confidence_drop_threshold
        self._window_size: int = cfg.window_size
        self._check_interval: int = cfg.check_interval

        # Optional backends
        self._nannyml_enabled: bool = cfg.nannyml_enabled and _nannyml_available
        self._evidently_enabled: bool = cfg.evidently_enabled and _evidently_available
        self._alert_webhook_url: str = cfg.alert_webhook_url

        if cfg.nannyml_enabled and not _nannyml_available:
            logger.warning(
                "NannyML requested but not installed. "
                "Install: pip install nannyml"
            )
        if cfg.evidently_enabled and not _evidently_available:
            logger.warning(
                "Evidently requested but not installed. "
                "Install: pip install evidently"
            )

        # Internal state
        self._data_detector = None  # created after reference init
        self._model_detector = None  # created after reference predictions init
        self._reference_stats: dict | None = None
        self._reference_pixel_values: np.ndarray | None = None
        self._reference_predictions: np.ndarray | None = None
        self._pixel_buffer: deque[np.ndarray] = deque(maxlen=self._window_size)
        self._prediction_counter: int = 0
        self._history: list[dict] = []
        self._total_alerts: int = 0

        logger.info(
            "EnhancedDriftDetector created (nannyml=%s, evidently=%s, "
            "check_interval=%d, window=%d)",
            self._nannyml_enabled,
            self._evidently_enabled,
            self._check_interval,
            self._window_size,
        )

    # ------------------------------------------------------------------
    # Reference initialisation
    # ------------------------------------------------------------------

    def initialize_reference(
        self,
        reference_pixel_values: np.ndarray,
        reference_predictions: np.ndarray | None = None,
    ) -> None:
        """Compute and store reference statistics from training data.

        Parameters
        ----------
        reference_pixel_values:
            1-D or 2-D array of per-image channel-mean pixel values from the
            training / validation set.
        reference_predictions:
            Optional 2-D array of shape ``(N, num_classes)`` of softmax
            prediction vectors produced by the model on the reference set.
        """
        from src.monitoring.drift import DataDriftDetector, ModelDriftDetector

        # Flatten to 1-D for histogram-based detection
        flat_pixels = np.asarray(reference_pixel_values).ravel().astype(np.float64)
        self._reference_stats = DataDriftDetector.compute_reference_stats(flat_pixels)
        self._reference_pixel_values = flat_pixels

        self._data_detector = DataDriftDetector(
            reference_stats=self._reference_stats,
            psi_threshold=self._psi_threshold,
            ks_threshold=self._ks_threshold,
        )

        if reference_predictions is not None:
            preds = np.asarray(reference_predictions)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            self._reference_predictions = preds
            self._model_detector = ModelDriftDetector(
                reference_predictions=preds,
                window_size=self._window_size,
            )

        logger.info(
            "Drift reference initialized: %d pixel samples, predictions=%s",
            len(flat_pixels),
            reference_predictions is not None,
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        pixel_values: np.ndarray,
        predictions: np.ndarray,
        inference_ms: float,  # noqa: ARG002 – kept for caller symmetry
    ) -> None:
        """Record a single inference for drift monitoring.

        Parameters
        ----------
        pixel_values:
            Pixel intensity values for the processed image (flattened or
            per-channel means).
        predictions:
            Model softmax output vector(s).  If 1-D, treated as a single
            prediction.
        inference_ms:
            Wall-clock inference time (logged, not used for drift maths).
        """
        flat = np.asarray(pixel_values).ravel().astype(np.float64)
        self._pixel_buffer.append(flat)

        # Feed the model-drift detector with the full prediction vector
        if self._model_detector is not None:
            preds = np.asarray(predictions)
            if preds.ndim == 1:
                preds = preds.reshape(1, -1)
            self._model_detector.record(preds)

        self._prediction_counter += 1

        # Auto-trigger periodic drift check
        if self._prediction_counter % self._check_interval == 0:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.check_drift())
                else:
                    loop.run_until_complete(self.check_drift())
            except RuntimeError:
                # No event loop available (e.g. plain sync context)
                logger.debug(
                    "Skipping auto drift check: no running event loop"
                )

    # ------------------------------------------------------------------
    # Full drift check
    # ------------------------------------------------------------------

    async def check_drift(self, force: bool = False) -> dict:  # noqa: ARG002
        """Run all enabled drift detectors and return a combined report.

        Parameters
        ----------
        force:
            Reserved for future use (e.g. skip minimum-sample guards).

        Returns
        -------
        dict
            Structured result with per-detector scores and overall severity.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        result: dict[str, Any] = {
            "overall_drift": False,
            "severity": "none",
            "timestamp": timestamp,
            "detectors": {
                "psi": None,
                "ks": None,
                "confidence": None,
                "nannyml": None,
                "evidently": None,
            },
        }

        severities: list[str] = []

        # -- Data drift (PSI + KS) ------------------------------------------
        if self._data_detector is not None and len(self._pixel_buffer) > 0:
            buffer_values = np.concatenate(list(self._pixel_buffer))
            data_report = self._data_detector.detect(buffer_values)

            psi_score = data_report.psi_scores.get("intensity", 0.0)
            ks_p = data_report.ks_p_values.get("intensity", 1.0)

            result["detectors"]["psi"] = {
                "score": psi_score,
                "threshold": self._psi_threshold,
                "drift": psi_score > self._psi_threshold,
            }
            result["detectors"]["ks"] = {
                "p_value": ks_p,
                "threshold": self._ks_threshold,
                "drift": ks_p < self._ks_threshold,
            }

            if data_report.drift_detected:
                severities.append(data_report.severity)

        # -- Model drift (confidence drop) -----------------------------------
        if self._model_detector is not None:
            model_report = self._model_detector.detect()

            # Parse confidence drop from the report details string
            conf_drop = 0.0
            if model_report.details and "dropped" in model_report.details:
                try:
                    conf_drop = float(
                        model_report.details.split("dropped")[-1].strip()
                    )
                except (ValueError, IndexError):
                    pass

            result["detectors"]["confidence"] = {
                "drop": conf_drop,
                "threshold": self._confidence_drop_threshold,
                "drift": model_report.drift_detected,
            }

            if model_report.drift_detected:
                severities.append(model_report.severity)

        # -- NannyML ---------------------------------------------------------
        nannyml_result = self._run_nannyml()
        if nannyml_result is not None:
            result["detectors"]["nannyml"] = nannyml_result
            if nannyml_result.get("drift_detected"):
                severities.append("warning")

        # -- Evidently -------------------------------------------------------
        evidently_result = self._run_evidently()
        if evidently_result is not None:
            result["detectors"]["evidently"] = evidently_result
            if evidently_result.get("dataset_drift"):
                severities.append("warning")

        # -- Aggregate severity ----------------------------------------------
        if "critical" in severities:
            result["severity"] = "critical"
            result["overall_drift"] = True
        elif "warning" in severities:
            result["severity"] = "warning"
            result["overall_drift"] = True

        # -- Post-check actions ----------------------------------------------
        if result["severity"] != "none":
            await self._send_alert(result)

        self._history.append(result)

        # Telemetry
        try:
            from backend.app.core.telemetry import record_drift_check

            record_drift_check()
        except Exception:
            pass

        logger.info(
            "Drift check complete: severity=%s, overall_drift=%s",
            result["severity"],
            result["overall_drift"],
        )

        return result

    # ------------------------------------------------------------------
    # Alerting
    # ------------------------------------------------------------------

    async def _send_alert(self, drift_report: dict) -> None:
        """Emit an event-bus event and optionally POST to a webhook."""
        self._total_alerts += 1

        # Event bus
        try:
            from src.agents.event_bus import Event, EventType, event_bus

            event = Event(
                type=EventType.DRIFT_DETECTED,
                source="enhanced_drift_detector",
                data=drift_report,
            )
            await event_bus.emit(event)
            logger.info("DRIFT_DETECTED event emitted on event bus")
        except Exception as exc:
            logger.error("Failed to emit drift event: %s", exc)

        # Webhook
        if self._alert_webhook_url:
            try:
                import httpx

                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        self._alert_webhook_url,
                        json=drift_report,
                    )
                    resp.raise_for_status()
                    logger.info(
                        "Drift alert posted to webhook (%d)",
                        resp.status_code,
                    )
            except ImportError:
                # Fall back to urllib (sync, but acceptable for a single POST)
                import json
                import urllib.request

                try:
                    payload = json.dumps(drift_report).encode()
                    req = urllib.request.Request(
                        self._alert_webhook_url,
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        logger.info(
                            "Drift alert posted via urllib (%d)",
                            resp.status,
                        )
                except Exception as exc:
                    logger.error("Webhook POST via urllib failed: %s", exc)
            except Exception as exc:
                logger.error("Webhook POST failed: %s", exc)

        logger.warning(
            "Drift alert raised: severity=%s (total alerts: %d)",
            drift_report.get("severity", "unknown"),
            self._total_alerts,
        )

    # ------------------------------------------------------------------
    # Optional backends
    # ------------------------------------------------------------------

    def _run_nannyml(self) -> dict | None:
        """Run NannyML UnivariateDriftCalculator on the pixel buffer.

        Returns a summary dict or ``None`` if NannyML is not enabled / fails.
        """
        if not self._nannyml_enabled:
            return None

        if self._reference_pixel_values is None or len(self._pixel_buffer) == 0:
            return None

        try:
            import pandas as pd
            from nannyml.drift.univariate import UnivariateDriftCalculator

            ref_df = pd.DataFrame({"pixel_intensity": self._reference_pixel_values})
            analysis_values = np.concatenate(list(self._pixel_buffer))
            analysis_df = pd.DataFrame({"pixel_intensity": analysis_values})

            calc = UnivariateDriftCalculator(
                column_names=["pixel_intensity"],
                chunk_size=max(len(analysis_df) // 5, 50),
            )
            calc.fit(ref_df)
            results = calc.calculate(analysis_df)

            # Extract per-chunk alerts
            drift_flags = results.filter(
                column_names=["pixel_intensity"],
                methods=["kolmogorov_smirnov"],
            ).to_df()

            any_drift = bool(drift_flags["alert"].any()) if "alert" in drift_flags.columns else False

            return {
                "method": "nannyml_univariate",
                "drift_detected": any_drift,
                "details": {
                    "n_chunks": len(drift_flags),
                    "n_alerts": int(drift_flags["alert"].sum()) if "alert" in drift_flags.columns else 0,
                },
            }
        except Exception as exc:
            logger.error("NannyML drift check failed: %s", exc, exc_info=True)
            return None

    def _run_evidently(self) -> dict | None:
        """Run Evidently DataDriftPreset on the pixel buffer vs reference.

        Returns a summary dict or ``None`` if Evidently is not enabled / fails.
        """
        if not self._evidently_enabled:
            return None

        if self._reference_pixel_values is None or len(self._pixel_buffer) == 0:
            return None

        try:
            import pandas as pd
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report

            ref_df = pd.DataFrame({"pixel_intensity": self._reference_pixel_values})
            analysis_values = np.concatenate(list(self._pixel_buffer))
            analysis_df = pd.DataFrame({"pixel_intensity": analysis_values})

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref_df, current_data=analysis_df)
            report_dict = report.as_dict()

            # Navigate the Evidently result structure
            metrics_results = report_dict.get("metrics", [])
            dataset_drift = False
            feature_drift_share = 0.0

            for m in metrics_results:
                result_data = m.get("result", {})
                if "dataset_drift" in result_data:
                    dataset_drift = bool(result_data["dataset_drift"])
                if "drift_share" in result_data:
                    feature_drift_share = float(result_data["drift_share"])

            return {
                "dataset_drift": dataset_drift,
                "feature_drift_share": feature_drift_share,
            }
        except Exception as exc:
            logger.error("Evidently drift check failed: %s", exc, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # History & status
    # ------------------------------------------------------------------

    def get_history(self, limit: int = 50) -> list[dict]:
        """Return the most recent *limit* drift check results."""
        return self._history[-limit:]

    def get_current_status(self) -> dict:
        """Return the latest drift check plus aggregate summary statistics."""
        last_check = self._history[-1] if self._history else None
        return {
            "total_checks": len(self._history),
            "total_alerts": self._total_alerts,
            "last_check_at": last_check["timestamp"] if last_check else None,
            "current_severity": last_check["severity"] if last_check else "none",
            "detectors_enabled": {
                "psi_ks": self._data_detector is not None,
                "confidence": self._model_detector is not None,
                "nannyml": self._nannyml_enabled,
                "evidently": self._evidently_enabled,
            },
        }


# ---------------------------------------------------------------------------
# Module-level singleton management
# ---------------------------------------------------------------------------

_detector: EnhancedDriftDetector | None = None


def init_drift_detector() -> None:
    """Create the module-level EnhancedDriftDetector singleton.

    Called during application startup (e.g. FastAPI lifespan). Respects
    ``settings.drift.enabled``; does nothing when drift detection is
    turned off.
    """
    global _detector

    if not settings.drift.enabled:
        logger.info("Drift detection disabled (DRIFT__ENABLED=false)")
        return

    _detector = EnhancedDriftDetector()
    logger.info("EnhancedDriftDetector initialized (singleton)")


def get_drift_detector() -> EnhancedDriftDetector | None:
    """Return the active drift detector, or ``None`` if not initialized."""
    return _detector
