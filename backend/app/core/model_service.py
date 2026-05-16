"""Model loading and inference service - singleton lifecycle."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import settings
from src.data.datamodule import DISEASE_COLUMNS
from src.models.vignn import ClinicalKnowledgeGraph, create_vignn_model

# V2 disease columns (24 classes after ultra-rare filtering)
V2_DISEASE_COLUMNS = [
    "DR",
    "ARMD",
    "MH",
    "DN",
    "MYA",
    "BRVO",
    "TSLN",
    "ERM",
    "LS",
    "MS",
    "CSR",
    "ODC",
    "CRVO",
    "AH",
    "ODP",
    "ODE",
    "AION",
    "PT",
    "RT",
    "RS",
    "CRS",
    "EDN",
    "RPEC",
    "MHL",
]

logger = logging.getLogger(__name__)

# Disease full names
DISEASE_NAMES = {
    "DR": "Diabetic Retinopathy",
    "ARMD": "Age-Related Macular Degeneration",
    "MH": "Macular Hole",
    "DN": "Diabetic Neuropathy",
    "MYA": "Myopic Retinopathy",
    "BRVO": "Branch Retinal Vein Occlusion",
    "TSLN": "Tessellation",
    "ERM": "Epiretinal Membrane",
    "LS": "Laser Scars",
    "MS": "Macular Scars",
    "CSR": "Central Serous Retinopathy",
    "ODC": "Optic Disc Cupping",
    "CRVO": "Central Retinal Vein Occlusion",
    "TV": "Tortuous Vessels",
    "AH": "Asteroid Hyalosis",
    "ODP": "Optic Disc Pallor",
    "ODE": "Optic Disc Edema",
    "ST": "Optociliary Shunt Vessels",
    "AION": "Anterior Ischemic Optic Neuropathy",
    "PT": "Parafoveal Telangiectasia",
    "RT": "Retinal Traction",
    "RS": "Retinitis",
    "CRS": "Chorioretinal Scars",
    "EDN": "Exudative Detachment",
    "RPEC": "RPE Changes",
    "MHL": "Lamellar Macular Hole",
    "RP": "Retinitis Pigmentosa",
    "CWS": "Cotton Wool Spots",
    "CB": "Coats Disease",
    "ODPM": "Optic Disc Pit Maculopathy",
    "PRH": "Preretinal Hemorrhage",
    "MNF": "Myelinated Nerve Fibers",
    "HR": "Hemorrhagic Retinopathy",
    "CRAO": "Central Retinal Artery Occlusion",
    "TD": "Tilted Disc",
    "CME": "Cystoid Macular Edema",
    "PTCR": "Post-Traumatic Chorioretinopathy",
    "CF": "Choroidal Folds",
    "VH": "Vitreous Hemorrhage",
    "MCA": "Retinal Macroaneurysm",
    "VS": "Vasculitis",
    "BRAO": "Branch Retinal Artery Occlusion",
    "PLQ": "Optic Disc Drusen",
    "HPED": "Hemorrhagic PED",
    "CL": "Choroidal Lesion",
}

_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ModelService:
    """Manages model lifecycle - load once, serve many."""

    def __init__(self):
        self.model = None
        self.device = None
        self.disease_codes: list[str] = []
        self.kg = None
        self.default_thresholds: np.ndarray | None = None
        self.fundus_gate = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self):
        """Load model and knowledge graph."""
        device_str = settings.device
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        logger.info(f"Using device: {self.device}")

        self.disease_codes = DISEASE_COLUMNS
        self.kg = ClinicalKnowledgeGraph(disease_names=self.disease_codes)
        logger.info(
            f"Knowledge graph: {len(self.disease_codes)} diseases, "
            f"{self.kg.get_edge_count()} relationships"
        )

        model_path = PROJECT_ROOT / settings.model_path
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            model_name = checkpoint.get("model_name", "vignn")
            nc = len(self.disease_codes)
            logger.info(f"Loading model: {model_name} (from checkpoint)")

            if model_name == "hybrid_v2" or "disease_columns" in checkpoint:
                # V2 precision-rescue model
                from src.models.retinal_foundation_hybrid_v2 import create_hybrid_v2

                v2_columns = checkpoint.get("disease_columns", V2_DISEASE_COLUMNS)
                nc = checkpoint.get("num_classes", len(v2_columns))
                self.disease_codes = v2_columns
                self.kg = ClinicalKnowledgeGraph(disease_names=v2_columns)
                self.model = create_hybrid_v2(
                    num_classes=nc,
                    clinical_knowledge_graph=self.kg,
                    use_lora=False,  # LoRA already merged in checkpoint
                    freeze_backbone=False,
                )
                # Load per-class thresholds from v2 checkpoint
                if "thresholds" in checkpoint:
                    self.default_thresholds = np.asarray(
                        checkpoint["thresholds"], dtype=np.float32
                    )[:nc]
                    logger.info(f"Loaded v2 per-class thresholds ({nc} classes)")
                logger.info(f"Loaded HybridV2 model ({nc} classes)")
            elif model_name == "scene_graph_transformer":
                from src.models.scene_graph_transformer import SceneGraphTransformer

                self.model = SceneGraphTransformer(
                    num_classes=nc,
                    hidden_dim=384,
                    num_layers=3,
                    num_heads=4,
                    dropout=0.1,
                    clinical_knowledge_graph=self.kg,
                )
            elif model_name == "graphclip":
                from src.models.graphclip import GraphCLIP

                self.model = GraphCLIP(
                    num_classes=nc,
                    hidden_dim=384,
                    num_graph_layers=3,
                    num_heads=4,
                    dropout=0.1,
                    clinical_knowledge_graph=self.kg,
                )
            elif model_name == "visual_language_gnn":
                from src.models.visual_language_gnn import VisualLanguageGNN

                self.model = VisualLanguageGNN(
                    num_classes=nc,
                    hidden_dim=384,
                    num_layers=3,
                    num_heads=4,
                    dropout=0.1,
                    clinical_knowledge_graph=self.kg,
                )
            else:
                self.model = create_vignn_model(
                    num_classes=nc,
                    clinical_knowledge_graph=self.kg,
                )

            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                logger.info(f"Loaded weights from checkpoint (F1={checkpoint.get('best_f1', '?')})")
            threshold_values = checkpoint.get("decision_thresholds")
            if threshold_values is not None:
                threshold_array = np.asarray(threshold_values, dtype=np.float32).reshape(-1)
                if threshold_array.size == len(self.disease_codes):
                    self.default_thresholds = threshold_array
                    logger.info(
                        "Loaded %d learned per-class thresholds from checkpoint",
                        threshold_array.size,
                    )
                else:
                    logger.warning(
                        "Ignoring checkpoint thresholds: expected %d values, got %d",
                        len(self.disease_codes),
                        threshold_array.size,
                    )
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path} - running in demo mode")
            self.default_thresholds = None
            self._loaded = False

        # Load learned fundus gate if available
        gate_path = PROJECT_ROOT / "weights" / "fundus_gate.pth"
        if gate_path.exists():
            try:
                from src.data.fundus_gate_learned import LearnedFundusGate

                self.fundus_gate = LearnedFundusGate(weights_path=str(gate_path), threshold=0.5)
                self.fundus_gate.eval()
                logger.info(f"Learned fundus gate loaded from {gate_path}")
            except Exception as e:
                logger.warning(f"Failed to load fundus gate: {e}")

    def unload(self):
        """Release resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.default_thresholds = None
        self._loaded = False
        logger.info("Model unloaded")

    def _resolve_thresholds(self, threshold: Optional[float]) -> tuple[np.ndarray, float, str]:
        """Resolve scalar or learned per-class thresholds for inference."""
        if threshold is not None:
            scalar = float(threshold)
            return (
                np.full(len(self.disease_codes), scalar, dtype=np.float32),
                scalar,
                "scalar",
            )

        if self.default_thresholds is not None:
            return (
                self.default_thresholds,
                float(np.mean(self.default_thresholds)),
                "per_class",
            )

        return (
            np.full(len(self.disease_codes), 0.5, dtype=np.float32),
            0.5,
            "scalar",
        )

    @torch.no_grad()
    def predict(self, image: Image.Image, threshold: Optional[float] = None) -> dict:
        """Run inference on a PIL image with OTEL tracing and drift recording."""
        from backend.app.core.telemetry import get_tracer, record_prediction_metrics

        tracer = get_tracer()
        with tracer.start_as_current_span("retinalai.model.inference") as span:
            t0 = time.perf_counter()
            tensor = _transform(image.convert("RGB")).unsqueeze(0).to(self.device)
            thresholds, threshold_value, threshold_source = self._resolve_thresholds(threshold)

            if self.model is not None:
                output = self.model(tensor)
                probs = torch.sigmoid(output).cpu().numpy()[0]
            else:
                probs = np.random.rand(len(self.disease_codes))

            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Build results
            all_probs = {
                code: {
                    "probability": float(probs[i]),
                    "name": DISEASE_NAMES.get(code, code),
                    "threshold": float(thresholds[i]),
                }
                for i, code in enumerate(self.disease_codes)
            }
            detected = [
                {
                    "code": code,
                    "name": DISEASE_NAMES.get(code, code),
                    "probability": float(probs[i]),
                    "threshold": float(thresholds[i]),
                    "confidence": (
                        "high" if probs[i] > 0.8 else "medium" if probs[i] > 0.5 else "low"
                    ),
                }
                for i, code in enumerate(self.disease_codes)
                if probs[i] > thresholds[i]
            ]
            detected.sort(key=lambda x: x["probability"], reverse=True)

            # Clinical reasoning
            pred_dict = {code: float(probs[i]) for i, code in enumerate(self.disease_codes)}
            with tracer.start_as_current_span("retinalai.kg.clinical_reasoning"):
                refined = self.kg.apply_clinical_reasoning(pred_dict) if self.kg else pred_dict
                referral = (
                    self.kg.get_referral_priority([d["code"] for d in detected])
                    if self.kg and detected
                    else "FOLLOW_UP"
                )

            # OTEL span attributes
            max_conf = float(probs.max()) if len(probs) > 0 else 0.0
            span.set_attribute("model.name", settings.model_name)
            span.set_attribute("model.loaded", self._loaded)
            span.set_attribute("inference.duration_ms", round(elapsed_ms, 2))
            span.set_attribute("prediction.diseases_detected", len(detected))
            span.set_attribute("prediction.max_confidence", round(max_conf, 4))
            span.set_attribute("clinical.referral_priority", referral)
            span.set_attribute("threshold.source", threshold_source)

            # Record OTEL metrics
            record_prediction_metrics(
                inference_ms=elapsed_ms,
                diseases_detected=len(detected),
                max_confidence=max_conf,
                referral_priority=referral,
                model_version=settings.app_version,
            )

            # Phase 1: Drift recording
            try:
                from backend.app.core.drift_detector import get_drift_detector

                detector = get_drift_detector()
                if detector is not None:
                    pixel_vals = tensor.cpu().numpy().flatten()
                    detector.record_prediction(
                        pixel_values=pixel_vals,
                        predictions=probs,
                        inference_ms=elapsed_ms,
                    )
            except Exception:
                pass  # drift recording is best-effort

        return {
            "predictions": detected,
            "total_detected": len(detected),
            "all_probabilities": all_probs,
            "clinical": {
                "referral_priority": referral,
                "refined_predictions": {k: float(v) for k, v in list(refined.items())[:10]},
            },
            "inference_ms": round(elapsed_ms, 2),
            "model_loaded": self._loaded,
            "threshold": round(threshold_value, 4),
            "threshold_source": threshold_source,
            "per_class_thresholds": {
                code: float(thresholds[i]) for i, code in enumerate(self.disease_codes)
            },
        }


# Global singleton
model_service = ModelService()
