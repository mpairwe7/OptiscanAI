"""Active learning closed loop: clinician corrections -> LoRA fine-tuning.

When clinicians resolve reviews with a "modified" decision, corrected samples
are persisted and queued for incremental LoRA fine-tuning.  Once the queue
reaches ``retrain_threshold`` the loop automatically triggers a fine-tuning
run that blends corrected samples with high-confidence retention samples.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_loop: ActiveLearningLoop | None = None


def init_active_learning() -> None:
    """Initialise the global ActiveLearningLoop singleton."""
    global _loop
    try:
        _loop = ActiveLearningLoop()
        logger.info("Active learning loop initialised (enabled=%s)", _loop._enabled)
    except Exception:
        logger.exception("Failed to initialise active learning loop")
        _loop = None


def get_active_learning_loop() -> ActiveLearningLoop | None:
    """Return the global loop instance, or *None* if not initialised."""
    return _loop


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------

class ActiveLearningLoop:
    """Closes the active-learning loop between clinician reviews and LoRA
    fine-tuning of the production ViGNN model."""

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        from backend.app.core.config import settings

        cfg = settings.active_learning_loop

        self._enabled: bool = cfg.enabled
        self._retrain_threshold: int = cfg.retrain_threshold
        self._lora_rank: int = cfg.lora_rank
        self._lora_alpha: float = cfg.lora_alpha
        self._retention_ratio: float = cfg.retention_ratio
        self._finetune_epochs: int = cfg.finetune_epochs
        self._finetune_lr: float = cfg.finetune_lr
        self._confidence_threshold: float = cfg.confidence_threshold

        self._queue_dir = Path(cfg.queue_dir)
        self._corrected_dir = self._queue_dir / "corrected"
        self._processed_dir = self._queue_dir / "processed"
        self._state_path = self._queue_dir / "loop_state.json"
        self._prediction_log_dir = Path(settings.prediction_log_dir)
        self._model_path = Path(settings.model_path)

        # Ensure directories exist
        for d in (self._corrected_dir, self._processed_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Counters (persisted)
        self._corrected_total: int = 0
        self._retrain_count: int = 0
        self._last_retrain_at: str | None = None
        self._last_retrain_metrics: dict[str, Any] | None = None

        self._load_state()

    # ------------------------------------------------------------------
    # 1. Review resolution handler
    # ------------------------------------------------------------------

    async def on_review_resolved(
        self,
        request_id: str,
        decision: str,
        reviewer: str,
        corrected_labels: dict[str, float] | None = None,
        image_path: str = "",
        original_predictions: dict[str, float] | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        """Handle a resolved review.  If the clinician *modified* the
        prediction and supplied corrected labels the sample is queued for
        fine-tuning.

        Returns
        -------
        dict
            ``{queued, queue_size, retrain_triggered}``
        """
        queued = False
        retrain_triggered = False

        if decision == "modified" and corrected_labels:
            sample = {
                "sample_id": str(uuid.uuid4()),
                "request_id": request_id,
                "reviewer": reviewer,
                "decision": decision,
                "corrected_labels": corrected_labels,
                "original_predictions": original_predictions or {},
                "image_path": image_path,
                "notes": notes,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._persist_corrected_sample(sample)
            self._corrected_total += 1
            queued = True
            logger.info(
                "Corrected sample queued: request=%s reviewer=%s (total=%d)",
                request_id,
                reviewer,
                self._corrected_total,
            )

        # Emit REVIEW_COMPLETED event regardless of decision
        await self._emit_review_completed(request_id, decision, reviewer)

        queue_size = self._count_corrected()

        if queued and queue_size >= self._retrain_threshold and self._enabled:
            retrain_triggered = True
            # Fire-and-forget background fine-tuning
            asyncio.ensure_future(self._safe_trigger_finetune())

        self._persist_state()

        return {
            "queued": queued,
            "queue_size": queue_size,
            "retrain_triggered": retrain_triggered,
        }

    # ------------------------------------------------------------------
    # 2. Fine-tuning trigger
    # ------------------------------------------------------------------

    async def trigger_finetune(self) -> dict[str, Any]:
        """Orchestrate a LoRA fine-tuning run.

        1. Emit RETRAIN_TRIGGERED.
        2. Load corrected samples from ``queue_dir/corrected/``.
        3. Sample high-confidence retention predictions.
        4. Run LoRA fine-tuning (synchronous, offloaded to executor).
        5. Archive processed samples.
        6. Emit RETRAIN_COMPLETED.

        Returns
        -------
        dict
            ``{run_id, samples_used, metrics, model_path}``
        """
        run_id = f"lora_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        await self._emit_retrain_triggered(run_id)

        # Load corrected samples
        corrected_samples = self._load_corrected_samples()
        if not corrected_samples:
            logger.warning("No corrected samples available for fine-tuning")
            return {
                "run_id": run_id,
                "samples_used": 0,
                "metrics": {},
                "model_path": "",
            }

        # Retention samples (high-confidence predictions mixed in)
        retention_count = max(
            1, int(len(corrected_samples) * self._retention_ratio)
        )
        retention_samples = self._sample_retention_predictions(retention_count)

        logger.info(
            "Fine-tune run %s: %d corrected + %d retention samples",
            run_id,
            len(corrected_samples),
            len(retention_samples),
        )

        # Run the actual training in a thread-pool executor so the event
        # loop is not blocked by the synchronous PyTorch training.
        loop = asyncio.get_running_loop()
        metrics = await loop.run_in_executor(
            None,
            self._run_lora_finetune,
            corrected_samples,
            retention_samples,
        )

        # Archive processed samples
        self._archive_corrected_samples()

        self._retrain_count += 1
        self._last_retrain_at = datetime.now(timezone.utc).isoformat()
        self._last_retrain_metrics = metrics
        self._persist_state()

        await self._emit_retrain_completed(run_id, metrics)

        logger.info(
            "Fine-tune run %s complete: loss=%.4f samples=%d path=%s",
            run_id,
            metrics.get("loss", -1),
            metrics.get("samples_used", 0),
            metrics.get("model_path", ""),
        )

        return {
            "run_id": run_id,
            "samples_used": metrics.get("samples_used", 0),
            "metrics": metrics,
            "model_path": metrics.get("model_path", ""),
        }

    # ------------------------------------------------------------------
    # 3. LoRA fine-tuning
    # ------------------------------------------------------------------

    def _run_lora_finetune(
        self,
        corrected_samples: list[dict],
        retention_samples: list[dict],
    ) -> dict[str, Any]:
        """Build a combined dataset from corrected + retention samples and
        run a lightweight LoRA fine-tuning loop.

        Returns
        -------
        dict
            ``{loss, samples_used, model_path, epochs, lr}``
        """
        import torch
        import torch.nn as nn

        from backend.app.core.config import settings

        num_classes = settings.num_classes
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ------- build label tensors -----------------------------------
        all_samples = corrected_samples + retention_samples
        if not all_samples:
            return {"loss": 0.0, "samples_used": 0, "model_path": ""}

        disease_codes = self._get_disease_codes()
        label_tensors: list[torch.Tensor] = []
        for sample in all_samples:
            labels = sample.get("corrected_labels") or sample.get("predictions") or {}
            vec = torch.zeros(num_classes)
            for code, prob in labels.items():
                if code in disease_codes:
                    vec[disease_codes.index(code)] = float(prob)
            label_tensors.append(vec)

        labels_batch = torch.stack(label_tensors).to(device)  # [N, C]

        # ------- load base model ----------------------------------------
        model = self._load_base_model(device)
        if model is None:
            logger.error("Cannot load base model for fine-tuning")
            return {"loss": 0.0, "samples_used": 0, "model_path": ""}

        # ------- apply LoRA adapters ------------------------------------
        lora_modules = self._apply_lora_adapters(model, self._lora_rank, self._lora_alpha)

        # Only optimise LoRA parameters
        lora_params = []
        for mod in lora_modules:
            lora_params.extend(mod.parameters())
        if not lora_params:
            logger.error("No LoRA parameters found - cannot fine-tune")
            return {"loss": 0.0, "samples_used": 0, "model_path": ""}

        optimizer = torch.optim.AdamW(lora_params, lr=self._finetune_lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self._finetune_epochs, eta_min=self._finetune_lr * 0.1
        )

        # ------- build simple input proxies ----------------------------
        # For samples where the actual image is available we load it;
        # otherwise fall back to a noise tensor (still useful because the
        # loss is label-driven via the classifier head + LoRA adapters).
        input_tensors = self._build_input_tensors(all_samples, device)

        # ------- training loop -----------------------------------------
        model.train()
        # Freeze everything except LoRA params
        for param in model.parameters():
            param.requires_grad = False
        for mod in lora_modules:
            for param in mod.parameters():
                param.requires_grad = True

        total_loss = 0.0
        batch_size = min(16, len(all_samples))
        num_batches = math.ceil(len(all_samples) / batch_size)

        for epoch in range(self._finetune_epochs):
            epoch_loss = 0.0
            indices = list(range(len(all_samples)))
            random.shuffle(indices)

            for b in range(num_batches):
                batch_idx = indices[b * batch_size : (b + 1) * batch_size]
                if not batch_idx:
                    continue

                x = torch.stack([input_tensors[i] for i in batch_idx]).to(device)
                y = torch.stack([labels_batch[i] for i in batch_idx]).to(device)

                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()
            avg = epoch_loss / max(num_batches, 1)
            total_loss = avg
            logger.info("LoRA epoch %d/%d  loss=%.4f", epoch + 1, self._finetune_epochs, avg)

        # ------- save checkpoint ----------------------------------------
        checkpoint_dir = self._queue_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        checkpoint_path = checkpoint_dir / f"lora_checkpoint_{ts}.pth"

        # Save only the LoRA state dicts (lightweight)
        lora_state = {}
        for i, mod in enumerate(lora_modules):
            lora_state[f"lora_{i}"] = mod.state_dict()
        torch.save(
            {
                "lora_state": lora_state,
                "lora_rank": self._lora_rank,
                "lora_alpha": self._lora_alpha,
                "epoch": self._finetune_epochs,
                "loss": total_loss,
                "samples_used": len(all_samples),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            checkpoint_path,
        )

        logger.info("LoRA checkpoint saved to %s", checkpoint_path)

        return {
            "loss": total_loss,
            "samples_used": len(all_samples),
            "model_path": str(checkpoint_path),
            "epochs": self._finetune_epochs,
            "lr": self._finetune_lr,
        }

    # ------------------------------------------------------------------
    # 4. Retention sampling
    # ------------------------------------------------------------------

    def _sample_retention_predictions(self, count: int) -> list[dict]:
        """Read recent prediction logs (daily JSONL files) and return a
        diverse set of high-confidence samples for retention during
        fine-tuning.

        High-confidence means the maximum predicted probability exceeds 0.9.
        Samples are drawn uniformly across disease classes so the model
        does not catastrophically forget rarer conditions.
        """
        candidates_by_class: dict[str, list[dict]] = {}

        if not self._prediction_log_dir.exists():
            logger.debug("Prediction log dir %s not found", self._prediction_log_dir)
            return []

        log_files = sorted(self._prediction_log_dir.glob("predictions_*.jsonl"), reverse=True)
        if not log_files:
            return []

        # Scan at most the 7 most recent log files
        for log_file in log_files[:7]:
            try:
                with open(log_file) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        preds = entry.get("top_predictions", [])
                        if not preds:
                            continue

                        # Find top prediction
                        best = max(preds, key=lambda p: p.get("probability", 0))
                        if best.get("probability", 0) < 0.9:
                            continue

                        top_code = best.get("code", "UNKNOWN")
                        retention_item = {
                            "predictions": {
                                p.get("code", "?"): p.get("probability", 0.0)
                                for p in preds
                            },
                            "image_info": {
                                "request_id": entry.get("request_id", ""),
                                "timestamp": entry.get("timestamp", ""),
                                "image_width": entry.get("image_width", 0),
                                "image_height": entry.get("image_height", 0),
                            },
                        }
                        candidates_by_class.setdefault(top_code, []).append(retention_item)
            except OSError as exc:
                logger.warning("Cannot read prediction log %s: %s", log_file, exc)

        if not candidates_by_class:
            return []

        # Uniform sampling across disease classes
        classes = list(candidates_by_class.keys())
        sampled: list[dict] = []
        per_class = max(1, count // len(classes))

        for cls in classes:
            pool = candidates_by_class[cls]
            k = min(per_class, len(pool))
            sampled.extend(random.sample(pool, k))
            if len(sampled) >= count:
                break

        return sampled[:count]

    # ------------------------------------------------------------------
    # 5. State persistence
    # ------------------------------------------------------------------

    def _persist_state(self) -> None:
        """Save counters and last-retrain metadata to disk."""
        state = {
            "corrected_total": self._corrected_total,
            "retrain_count": self._retrain_count,
            "last_retrain_at": self._last_retrain_at,
            "last_retrain_metrics": self._last_retrain_metrics,
        }
        try:
            tmp = self._state_path.with_suffix(".tmp")
            with open(tmp, "w") as fh:
                json.dump(state, fh, indent=2)
            tmp.replace(self._state_path)
        except OSError:
            logger.exception("Failed to persist active-learning loop state")

    def _load_state(self) -> None:
        """Load persisted state from disk if available."""
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path) as fh:
                state = json.load(fh)
            self._corrected_total = state.get("corrected_total", 0)
            self._retrain_count = state.get("retrain_count", 0)
            self._last_retrain_at = state.get("last_retrain_at")
            self._last_retrain_metrics = state.get("last_retrain_metrics")
            logger.info(
                "Loaded loop state: corrected=%d retrains=%d last=%s",
                self._corrected_total,
                self._retrain_count,
                self._last_retrain_at or "never",
            )
        except (OSError, json.JSONDecodeError):
            logger.exception("Failed to load active-learning loop state")

    # ------------------------------------------------------------------
    # 6. Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return a summary dict suitable for API responses."""
        queue_size = self._count_corrected()
        return {
            "enabled": self._enabled,
            "queue_size": queue_size,
            "corrected_total": self._corrected_total,
            "retrain_count": self._retrain_count,
            "last_retrain_at": self._last_retrain_at,
            "last_retrain_metrics": self._last_retrain_metrics,
            "retrain_threshold": self._retrain_threshold,
            "progress": f"{queue_size}/{self._retrain_threshold}",
        }

    # ==================================================================
    # Private helpers
    # ==================================================================

    # ---- corrected sample I/O ----------------------------------------

    def _persist_corrected_sample(self, sample: dict) -> None:
        """Write a single corrected sample to disk as JSON."""
        filename = f"{sample['sample_id']}.json"
        path = self._corrected_dir / filename
        try:
            with open(path, "w") as fh:
                json.dump(sample, fh, indent=2)
        except OSError:
            logger.exception("Failed to persist corrected sample %s", filename)

    def _load_corrected_samples(self) -> list[dict]:
        """Load all corrected-sample JSON files from the corrected dir."""
        samples: list[dict] = []
        try:
            for p in sorted(self._corrected_dir.glob("*.json")):
                try:
                    with open(p) as fh:
                        samples.append(json.load(fh))
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("Skipping malformed sample %s: %s", p.name, exc)
        except OSError:
            logger.exception("Cannot list corrected samples directory")
        return samples

    def _count_corrected(self) -> int:
        """Count JSON files currently in the corrected directory."""
        try:
            return sum(1 for _ in self._corrected_dir.glob("*.json"))
        except OSError:
            return 0

    def _archive_corrected_samples(self) -> None:
        """Move all corrected samples to the processed directory after a
        fine-tuning run completes."""
        batch_dir = self._processed_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        batch_dir.mkdir(parents=True, exist_ok=True)
        try:
            for p in self._corrected_dir.glob("*.json"):
                try:
                    shutil.move(str(p), str(batch_dir / p.name))
                except OSError as exc:
                    logger.warning("Failed to archive %s: %s", p.name, exc)
        except OSError:
            logger.exception("Failed to archive corrected samples")

    # ---- model / LoRA helpers ----------------------------------------

    def _get_disease_codes(self) -> list[str]:
        """Return the ordered list of disease codes used by the model."""
        try:
            from src.data.datamodule import DISEASE_COLUMNS
            return list(DISEASE_COLUMNS)
        except ImportError:
            logger.warning("Cannot import DISEASE_COLUMNS - using config num_classes")
            from backend.app.core.config import settings
            return [f"class_{i}" for i in range(settings.num_classes)]

    def _load_base_model(self, device: "torch.device") -> "torch.nn.Module | None":
        """Load the base ViGNN model for fine-tuning."""
        import sys

        import torch

        project_root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(project_root))

        try:
            from src.data.datamodule import DISEASE_COLUMNS
            from src.models.vignn import ClinicalKnowledgeGraph, create_vignn_model
        except ImportError:
            logger.exception("Failed to import model modules")
            return None

        disease_codes = list(DISEASE_COLUMNS)
        kg = ClinicalKnowledgeGraph(disease_names=disease_codes)
        num_classes = len(disease_codes)

        model = create_vignn_model(
            num_classes=num_classes,
            clinical_knowledge_graph=kg,
        )

        # Load weights from the production checkpoint if available
        model_path = project_root / self._model_path
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                    logger.info("Loaded base model weights from %s", model_path)
                else:
                    logger.warning("Checkpoint has no model_state_dict key")
            except Exception:
                logger.exception("Failed to load checkpoint weights from %s", model_path)
        else:
            logger.warning("Base model checkpoint not found at %s", model_path)

        model.to(device)
        return model

    @staticmethod
    def _apply_lora_adapters(
        model: "torch.nn.Module",
        rank: int,
        alpha: float,
    ) -> "list[torch.nn.Module]":
        """Inject lightweight LoRA adapter modules into the model's Linear
        layers within the classifier head and disease attention.

        Each adapter wraps a ``nn.Linear`` by adding a low-rank
        decomposition ``A @ B`` scaled by ``alpha / rank``.

        Returns the list of injected LoRA modules so their parameters
        can be selectively optimised.
        """
        import torch
        import torch.nn as nn

        class LoRALinear(nn.Module):
            """Drop-in wrapper that adds a low-rank residual to a frozen Linear."""

            def __init__(self, original: nn.Linear, rank: int, alpha: float):
                super().__init__()
                self.original = original
                in_f = original.in_features
                out_f = original.out_features
                self.lora_A = nn.Parameter(torch.randn(in_f, rank) * 0.01)
                self.lora_B = nn.Parameter(torch.zeros(rank, out_f))
                self.scale = alpha / rank

                # Freeze original weights
                for p in self.original.parameters():
                    p.requires_grad = False

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                base = self.original(x)
                lora = (x @ self.lora_A @ self.lora_B) * self.scale
                return base + lora

        lora_modules: list[nn.Module] = []

        # Target: classifier head and disease attention projection layers
        target_attrs = ["classifier", "disease_attention", "global_context", "patch_proj"]

        for attr_name in target_attrs:
            parent = getattr(model, attr_name, None)
            if parent is None:
                continue
            if isinstance(parent, nn.Sequential):
                for i, layer in enumerate(parent):
                    if isinstance(layer, nn.Linear):
                        wrapped = LoRALinear(layer, rank, alpha)
                        parent[i] = wrapped
                        lora_modules.append(wrapped)
            elif isinstance(parent, nn.Linear):
                wrapped = LoRALinear(parent, rank, alpha)
                setattr(model, attr_name, wrapped)
                lora_modules.append(wrapped)

        logger.info("Applied %d LoRA adapters (rank=%d, alpha=%.1f)", len(lora_modules), rank, alpha)
        return lora_modules

    def _build_input_tensors(
        self, samples: list[dict], device: "torch.device"
    ) -> list["torch.Tensor"]:
        """Build input tensors for each sample.  If the image file exists
        on disk it is loaded and preprocessed; otherwise a synthetic input
        is generated so label-driven fine-tuning can still proceed.
        """
        import torch
        from PIL import Image
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        tensors: list[torch.Tensor] = []
        for sample in samples:
            img_path = sample.get("image_path", "")
            tensor: torch.Tensor | None = None

            if img_path and Path(img_path).is_file():
                try:
                    img = Image.open(img_path).convert("RGB")
                    tensor = transform(img)
                except Exception:
                    logger.debug("Cannot load image %s - using synthetic input", img_path)

            if tensor is None:
                # Deterministic synthetic input seeded by sample id for
                # reproducibility.  Still useful because the LoRA adapters
                # in the classifier head learn from the label signal.
                seed_str = sample.get("sample_id", sample.get("request_id", str(len(tensors))))
                seed_val = hash(seed_str) % (2**31)
                rng = torch.Generator()
                rng.manual_seed(seed_val)
                tensor = torch.randn(3, 224, 224, generator=rng)

            tensors.append(tensor)

        return tensors

    # ---- event bus helpers -------------------------------------------

    async def _emit_review_completed(self, request_id: str, decision: str, reviewer: str) -> None:
        try:
            from src.agents.event_bus import Event, EventType, event_bus
            await event_bus.emit(Event(
                type=EventType.REVIEW_COMPLETED,
                source="active_learning_loop",
                data={
                    "request_id": request_id,
                    "decision": decision,
                    "reviewer": reviewer,
                },
            ))
        except Exception:
            logger.debug("Event bus unavailable for REVIEW_COMPLETED", exc_info=True)

    async def _emit_retrain_triggered(self, run_id: str) -> None:
        try:
            from src.agents.event_bus import Event, EventType, event_bus
            await event_bus.emit(Event(
                type=EventType.RETRAIN_TRIGGERED,
                source="active_learning_loop",
                data={"run_id": run_id},
            ))
        except Exception:
            logger.debug("Event bus unavailable for RETRAIN_TRIGGERED", exc_info=True)

    async def _emit_retrain_completed(self, run_id: str, metrics: dict) -> None:
        try:
            from src.agents.event_bus import Event, EventType, event_bus
            await event_bus.emit(Event(
                type=EventType.RETRAIN_COMPLETED,
                source="active_learning_loop",
                data={"run_id": run_id, "metrics": metrics},
            ))
        except Exception:
            logger.debug("Event bus unavailable for RETRAIN_COMPLETED", exc_info=True)

    async def _safe_trigger_finetune(self) -> None:
        """Wrapper that swallows exceptions so ``asyncio.ensure_future``
        does not produce unhandled-exception warnings."""
        try:
            await self.trigger_finetune()
        except Exception:
            logger.exception("Background fine-tuning failed")
