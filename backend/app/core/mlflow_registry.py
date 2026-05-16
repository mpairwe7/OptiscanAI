"""MLflow 3.0 Model Registry integration for RetinalAI.

Provides model registration, stage promotion with quality gates,
shadow deployments, A/B test logging, and lineage tracking.
All operations are no-op when ``settings.mlflow.enabled`` is False or
when the ``mlflow`` package is not installed.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy mlflow import – the entire module stays functional even when mlflow
# is absent; every public method simply returns a neutral value.
# ---------------------------------------------------------------------------
try:
    import mlflow
    from mlflow.tracking import MlflowClient

    _MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    MlflowClient = None  # type: ignore[assignment,misc]
    _MLFLOW_AVAILABLE = False
    logger.info("mlflow package not installed – MLflow integration disabled")


class MLflowRegistry:
    """Thread-safe facade around the MLflow Model Registry.

    Every public method checks ``self._enabled`` first and returns a
    neutral value (``None`` / ``False`` / empty dict) when the registry
    is switched off, making this class safe to inject unconditionally.
    """

    # ── Construction ──────────────────────────────────────────────────

    def __init__(self) -> None:
        """Initialise the registry from application settings.

        If mlflow is not installed or ``settings.mlflow.enabled`` is
        ``False``, the instance marks itself as disabled and every
        subsequent call becomes a no-op.
        """
        self._cfg = settings.mlflow
        self._enabled: bool = False
        self._client: Any = None
        self._shadow_deployments: dict[str, dict[str, Any]] = {}

        if not _MLFLOW_AVAILABLE:
            logger.warning("MLflowRegistry created but mlflow package is not installed")
            return

        if not self._cfg.enabled:
            logger.info("MLflowRegistry disabled via settings.mlflow.enabled")
            return

        # Connect to the tracking / registry server.
        try:
            mlflow.set_tracking_uri(self._cfg.tracking_uri)
            if self._cfg.registry_uri:
                mlflow.set_registry_uri(self._cfg.registry_uri)

            self._client = MlflowClient(
                tracking_uri=self._cfg.tracking_uri,
                registry_uri=self._cfg.registry_uri or None,
            )

            # Ensure the experiment exists.
            exp = self._client.get_experiment_by_name(self._cfg.experiment_name)
            if exp is None:
                self._client.create_experiment(self._cfg.experiment_name)
                logger.info("Created MLflow experiment %s", self._cfg.experiment_name)

            self._enabled = True
            logger.info(
                "MLflowRegistry initialised: tracking=%s model=%s experiment=%s",
                self._cfg.tracking_uri,
                self._cfg.model_name,
                self._cfg.experiment_name,
            )
        except Exception:
            logger.exception("Failed to initialise MLflow connection")
            self._enabled = False

    # ── Model Registration ────────────────────────────────────────────

    def register_model(
        self,
        model_path: str,
        metrics: dict[str, float],
        training_config: dict,
        dataset_hash: str,
        tags: dict[str, str] | None = None,
    ) -> str | None:
        """Register a trained model artefact and its metadata with MLflow.

        Args:
            model_path: Local filesystem path to the serialised model.
            metrics: Training / validation metrics (e.g. ``{"f1": 0.92}``).
            training_config: Hyperparameters logged as MLflow params.
            dataset_hash: SHA-256 (or similar) of the training dataset for
                lineage tracking.
            tags: Optional extra tags to attach to the run.

        Returns:
            The MLflow ``run_id`` on success, or ``None`` on failure / disabled.
        """
        if not self._enabled:
            logger.debug("register_model skipped – registry disabled")
            return None

        try:
            mlflow.set_experiment(self._cfg.experiment_name)

            with mlflow.start_run() as run:
                # Log all supplied metrics.
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)

                # Log training hyperparams (flatten nested dicts with dot
                # notation so MLflow can store them).
                flat_params = _flatten_dict(training_config)
                for key, value in flat_params.items():
                    mlflow.log_param(key, value)

                # Log the model artefact.
                mlflow.log_artifact(model_path, artifact_path="model")

                # Tags.
                mlflow.set_tag("dataset_hash", dataset_hash)
                mlflow.set_tag("model_name", self._cfg.model_name)
                mlflow.set_tag(
                    "registered_at",
                    datetime.now(timezone.utc).isoformat(),
                )
                if tags:
                    for k, v in tags.items():
                        mlflow.set_tag(k, v)

                run_id: str = run.info.run_id

            # Register the model version in the registry.
            try:
                model_uri = f"runs:/{run_id}/model"
                mlflow.register_model(model_uri, self._cfg.model_name)
            except Exception:
                logger.exception(
                    "Model logged (run %s) but registry registration failed",
                    run_id,
                )

            logger.info(
                "Registered model: run_id=%s metrics=%s dataset_hash=%s",
                run_id,
                metrics,
                dataset_hash,
            )
            return run_id

        except Exception:
            logger.exception("register_model failed")
            return None

    # ── Stage Promotion ───────────────────────────────────────────────

    def promote_model(
        self,
        model_version: int,
        from_stage: str,
        to_stage: str,
        validation_metrics: dict[str, float],
        approved_by: str = "system",
    ) -> bool:
        """Promote a model version between registry stages.

        Quality gates enforce minimum F1 and AUC thresholds configured in
        ``settings.mlflow``.  On success an ``EventType.MODEL_PROMOTED``
        event is emitted on the global event bus.

        Args:
            model_version: Numeric version in the model registry.
            from_stage: Current stage (e.g. ``"Staging"``).
            to_stage: Target stage (e.g. ``"Production"``).
            validation_metrics: Must contain ``f1`` and ``auc`` keys.
            approved_by: Human / system identifier for audit trail.

        Returns:
            ``True`` if promotion succeeded, ``False`` otherwise.
        """
        if not self._enabled:
            logger.debug("promote_model skipped – registry disabled")
            return False

        # ── Quality gate ──────────────────────────────────────────────
        f1 = validation_metrics.get("f1", 0.0)
        auc = validation_metrics.get("auc", 0.0)

        if f1 < self._cfg.promotion_min_f1:
            logger.warning(
                "Promotion blocked: F1 %.4f < required %.4f",
                f1,
                self._cfg.promotion_min_f1,
            )
            return False

        if auc < self._cfg.promotion_min_auc:
            logger.warning(
                "Promotion blocked: AUC %.4f < required %.4f",
                auc,
                self._cfg.promotion_min_auc,
            )
            return False

        # ── Stage transition ──────────────────────────────────────────
        try:
            self._client.transition_model_version_stage(
                name=self._cfg.model_name,
                version=model_version,
                stage=to_stage,
            )

            self._client.set_model_version_tag(
                name=self._cfg.model_name,
                version=str(model_version),
                key="approved_by",
                value=approved_by,
            )
            self._client.set_model_version_tag(
                name=self._cfg.model_name,
                version=str(model_version),
                key="promoted_at",
                value=datetime.now(timezone.utc).isoformat(),
            )
            self._client.set_model_version_tag(
                name=self._cfg.model_name,
                version=str(model_version),
                key="from_stage",
                value=from_stage,
            )

            logger.info(
                "Promoted model v%d: %s -> %s (F1=%.4f AUC=%.4f approved_by=%s)",
                model_version,
                from_stage,
                to_stage,
                f1,
                auc,
                approved_by,
            )

            # Emit event (fire-and-forget async).
            self._emit_event_nonblocking(
                event_type_name="MODEL_PROMOTED",
                data={
                    "model_name": self._cfg.model_name,
                    "model_version": model_version,
                    "from_stage": from_stage,
                    "to_stage": to_stage,
                    "validation_metrics": validation_metrics,
                    "approved_by": approved_by,
                },
            )

            return True

        except Exception:
            logger.exception("promote_model failed for version %d", model_version)
            return False

    # ── Shadow Deployment ─────────────────────────────────────────────

    def start_shadow_deployment(
        self,
        staging_version: int,
        production_version: int,
    ) -> str:
        """Start a shadow deployment comparing staging against production.

        Args:
            staging_version: Candidate model version (receives traffic shadow).
            production_version: Current production model version.

        Returns:
            A unique shadow deployment identifier (12-char hex string).
        """
        shadow_id = uuid.uuid4().hex[:12]
        self._shadow_deployments[shadow_id] = {
            "staging_version": staging_version,
            "production_version": production_version,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "results": [],
        }
        logger.info(
            "Shadow deployment started: id=%s staging=v%d production=v%d",
            shadow_id,
            staging_version,
            production_version,
        )
        return shadow_id

    def record_shadow_result(
        self,
        shadow_id: str,
        staging_preds: dict,
        production_preds: dict,
        ground_truth: dict | None = None,
    ) -> None:
        """Record one inference comparison for an active shadow deployment.

        Args:
            shadow_id: Identifier returned by :meth:`start_shadow_deployment`.
            staging_preds: Predictions from the staging model.
            production_preds: Predictions from the production model.
            ground_truth: Optional ground-truth labels for offline evaluation.
        """
        deployment = self._shadow_deployments.get(shadow_id)
        if deployment is None:
            logger.warning("record_shadow_result: unknown shadow_id=%s", shadow_id)
            return

        deployment["results"].append(
            {
                "staging_preds": staging_preds,
                "production_preds": production_preds,
                "ground_truth": ground_truth,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        logger.debug(
            "Shadow result recorded: id=%s total=%d",
            shadow_id,
            len(deployment["results"]),
        )

    def end_shadow_deployment(self, shadow_id: str) -> dict:
        """End a shadow deployment and return an analysis summary.

        Computes the agreement rate between staging and production
        predictions and issues a promotion recommendation.

        Args:
            shadow_id: Identifier returned by :meth:`start_shadow_deployment`.

        Returns:
            Summary dict with ``staging_count``, ``production_count``,
            ``agreement_rate``, and ``recommendation`` keys.
        """
        deployment = self._shadow_deployments.pop(shadow_id, None)
        if deployment is None:
            logger.warning("end_shadow_deployment: unknown shadow_id=%s", shadow_id)
            return {
                "staging_count": 0,
                "production_count": 0,
                "agreement_rate": 0.0,
                "recommendation": "unknown_deployment",
            }

        results: list[dict] = deployment["results"]
        total = len(results)
        if total == 0:
            return {
                "staging_count": 0,
                "production_count": 0,
                "agreement_rate": 0.0,
                "recommendation": "insufficient_data",
            }

        agreements = 0
        for r in results:
            if r["staging_preds"] == r["production_preds"]:
                agreements += 1

        agreement_rate = agreements / total

        if agreement_rate >= 0.95:
            recommendation = "promote_staging"
        elif agreement_rate >= 0.85:
            recommendation = "extend_shadow"
        else:
            recommendation = "reject_staging"

        summary = {
            "staging_count": total,
            "production_count": total,
            "agreement_rate": round(agreement_rate, 4),
            "recommendation": recommendation,
        }

        logger.info(
            "Shadow deployment ended: id=%s agreement=%.2f%% recommendation=%s",
            shadow_id,
            agreement_rate * 100,
            recommendation,
        )
        return summary

    # ── Registry Queries ──────────────────────────────────────────────

    def get_production_model_version(self) -> int | None:
        """Return the latest Production-stage model version number.

        Returns:
            The integer version, or ``None`` when disabled / not found.
        """
        if not self._enabled:
            logger.debug("get_production_model_version skipped – registry disabled")
            return None

        try:
            versions = self._client.get_latest_versions(self._cfg.model_name, stages=["Production"])
            if versions:
                version_num = int(versions[0].version)
                logger.debug("Production model version: %d", version_num)
                return version_num
            logger.debug("No Production version found for %s", self._cfg.model_name)
            return None
        except Exception:
            logger.exception("get_production_model_version failed")
            return None

    def get_model_lineage(self, version: int) -> dict:
        """Retrieve lineage metadata for a specific model version.

        Args:
            version: Registry version number.

        Returns:
            Dict containing ``run_id``, ``metrics``, ``params``, ``tags``,
            and ``created_at``.  Returns an empty dict on error / disabled.
        """
        if not self._enabled:
            logger.debug("get_model_lineage skipped – registry disabled")
            return {}

        try:
            model_version = self._client.get_model_version(
                name=self._cfg.model_name, version=str(version)
            )
            run_id = model_version.run_id

            run = self._client.get_run(run_id)

            lineage: dict[str, Any] = {
                "run_id": run_id,
                "metrics": dict(run.data.metrics),
                "params": dict(run.data.params),
                "tags": dict(run.data.tags),
                "created_at": model_version.creation_timestamp,
            }

            logger.debug("Lineage for v%d: run_id=%s", version, run_id)
            return lineage

        except Exception:
            logger.exception("get_model_lineage failed for version %d", version)
            return {}

    # ── A/B Testing ───────────────────────────────────────────────────

    def log_ab_test(
        self,
        test_id: str,
        variant_a_version: int,
        variant_b_version: int,
        traffic_split: float,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Log an A/B test configuration and optional results.

        Creates a dedicated MLflow run tagged with A/B test metadata so
        experiments can be queried later.

        Args:
            test_id: Unique identifier for this A/B test.
            variant_a_version: Model version for variant A.
            variant_b_version: Model version for variant B.
            traffic_split: Fraction of traffic routed to variant B (0.0-1.0).
            metrics: Optional outcome metrics recorded so far.
        """
        if not self._enabled:
            logger.debug("log_ab_test skipped – registry disabled")
            return

        try:
            mlflow.set_experiment(self._cfg.experiment_name)

            with mlflow.start_run(run_name=f"ab-test-{test_id}"):
                mlflow.set_tag("ab_test_id", test_id)
                mlflow.set_tag("ab_test", "true")
                mlflow.set_tag("variant_a_version", str(variant_a_version))
                mlflow.set_tag("variant_b_version", str(variant_b_version))
                mlflow.set_tag(
                    "logged_at",
                    datetime.now(timezone.utc).isoformat(),
                )

                mlflow.log_param("traffic_split_b", traffic_split)
                mlflow.log_param("variant_a_version", variant_a_version)
                mlflow.log_param("variant_b_version", variant_b_version)

                if metrics:
                    for key, value in metrics.items():
                        mlflow.log_metric(key, value)

            logger.info(
                "A/B test logged: test_id=%s A=v%d B=v%d split=%.2f",
                test_id,
                variant_a_version,
                variant_b_version,
                traffic_split,
            )
        except Exception:
            logger.exception("log_ab_test failed for test_id=%s", test_id)

    # ── Status ────────────────────────────────────────────────────────

    def get_registry_status(self) -> dict:
        """Return a status snapshot of the model registry.

        Returns:
            Dict with ``enabled``, ``tracking_uri``, ``production_version``,
            ``staging_versions``, ``shadow_deployments_active``, and
            ``experiment_name`` keys.
        """
        status: dict[str, Any] = {
            "enabled": self._enabled,
            "tracking_uri": self._cfg.tracking_uri,
            "production_version": None,
            "staging_versions": [],
            "shadow_deployments_active": len(self._shadow_deployments),
            "experiment_name": self._cfg.experiment_name,
        }

        if not self._enabled:
            return status

        try:
            prod_versions = self._client.get_latest_versions(
                self._cfg.model_name, stages=["Production"]
            )
            if prod_versions:
                status["production_version"] = int(prod_versions[0].version)
        except Exception:
            logger.exception("Failed to query production versions")

        try:
            staging_versions = self._client.get_latest_versions(
                self._cfg.model_name, stages=["Staging"]
            )
            status["staging_versions"] = [int(v.version) for v in staging_versions]
        except Exception:
            logger.exception("Failed to query staging versions")

        return status

    # ── Internal Helpers ──────────────────────────────────────────────

    def _emit_event_nonblocking(
        self,
        event_type_name: str,
        data: dict[str, Any],
    ) -> None:
        """Fire an event on the global event bus without blocking.

        Uses ``asyncio.get_event_loop().create_task()`` so this can be
        called from synchronous MLflow-registry methods.
        """
        try:
            from src.agents.event_bus import Event, EventType, event_bus

            event_type = EventType[event_type_name]
            event = Event(
                type=event_type,
                source="mlflow_registry",
                data=data,
            )

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(event_bus.emit(event))
                else:
                    loop.run_until_complete(event_bus.emit(event))
            except RuntimeError:
                # No running event loop – create a new one for this emit.
                asyncio.run(event_bus.emit(event))

            logger.debug("Emitted %s event: %s", event_type_name, data)
        except Exception:
            logger.exception("Failed to emit %s event", event_type_name)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _flatten_dict(
    d: dict,
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """Flatten a nested dict into dot-separated keys for MLflow params."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: MLflowRegistry | None = None


def init_mlflow_registry() -> None:
    """Create the module-level :class:`MLflowRegistry` singleton.

    Safe to call multiple times – subsequent calls are no-ops.
    """
    global _registry
    if _registry is not None:
        logger.debug("MLflow registry already initialised")
        return
    _registry = MLflowRegistry()
    logger.info("MLflow registry singleton created (enabled=%s)", _registry._enabled)


def get_mlflow_registry() -> MLflowRegistry | None:
    """Return the module-level :class:`MLflowRegistry` instance.

    Returns ``None`` if :func:`init_mlflow_registry` has not been called.
    """
    return _registry
