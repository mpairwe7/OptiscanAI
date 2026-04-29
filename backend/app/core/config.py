"""Application configuration with pydantic-settings.

All Phase 1-4 features are opt-in via nested settings with env_nested_delimiter='__'.
Example: TELEMETRY__ENABLED=true, MLFLOW__TRACKING_URI=http://mlflow:5000
"""
from pathlib import Path
from pydantic import BaseModel
from pydantic_settings import BaseSettings


# ── Phase 1: Observability & MLOps ──────────────────────────────────────────


class TelemetrySettings(BaseModel):
    """OpenTelemetry instrumentation (Phase 1)."""
    enabled: bool = False
    otlp_endpoint: str = "http://localhost:4317"
    otlp_protocol: str = "grpc"  # grpc | http
    service_name: str = "retinalai"
    sample_rate: float = 1.0
    log_correlation: bool = True
    metrics_export_interval_ms: int = 60000


class MLflowSettings(BaseModel):
    """MLflow 3.0 Model Registry (Phase 1)."""
    enabled: bool = False
    tracking_uri: str = "http://localhost:5000"
    registry_uri: str = ""  # defaults to tracking_uri
    model_name: str = "retinalai-vignn"
    experiment_name: str = "retinalai-production"
    promotion_min_f1: float = 0.0  # minimum F1 to allow promotion
    promotion_min_auc: float = 0.0  # minimum AUC-ROC


class ActiveLearningLoopSettings(BaseModel):
    """Active learning closed loop (Phase 1)."""
    enabled: bool = False
    retrain_threshold: int = 150
    queue_dir: str = "data/active_learning"
    lora_rank: int = 16
    lora_alpha: float = 16.0
    retention_ratio: float = 0.3  # high-confidence samples mixed in
    finetune_epochs: int = 5
    finetune_lr: float = 1e-4
    confidence_threshold: float = 0.65


class DriftSettings(BaseModel):
    """Enhanced drift detection (Phase 1)."""
    enabled: bool = True
    psi_threshold: float = 0.2
    ks_threshold: float = 0.05
    confidence_drop_threshold: float = 0.1
    window_size: int = 200
    check_interval: int = 100  # every N predictions
    nannyml_enabled: bool = False
    evidently_enabled: bool = False
    alert_webhook_url: str = ""


# ── Phase 2: Scalability & Security ─────────────────────────────────────────


class RayServeSettings(BaseModel):
    """Ray Serve model serving (Phase 2)."""
    enabled: bool = False
    serve_url: str = "http://localhost:8000"
    timeout_s: float = 30.0
    batch_max_size: int = 16
    batch_timeout_s: float = 0.1
    min_replicas: int = 1
    max_replicas: int = 8
    target_ongoing_requests: int = 10


class CanarySettings(BaseModel):
    """Canary release routing (Phase 2)."""
    enabled: bool = False
    primary_version: str = "default"
    canary_version: str = ""
    canary_weight: float = 0.0  # 0.0 = all primary
    sticky_sessions: bool = True


class CircuitBreakerSettings(BaseModel):
    """Circuit breaker for external services (Phase 2)."""
    failure_threshold: int = 5
    recovery_timeout_s: float = 60.0
    half_open_max_calls: int = 3


class MTLSSettings(BaseModel):
    """Mutual TLS between services (Phase 2)."""
    enabled: bool = False
    ca_cert_path: str = ""
    client_cert_path: str = ""
    client_key_path: str = ""
    verify_hostname: bool = True


class KafkaSettings(BaseModel):
    """Kafka for durable event streaming (Phase 2)."""
    enabled: bool = False
    bootstrap_servers: str = "localhost:9092"
    audit_topic: str = "retinalai.audit"
    events_topic: str = "retinalai.events"
    security_protocol: str = "PLAINTEXT"
    acks: str = "all"


class IcebergSettings(BaseModel):
    """Apache Iceberg for immutable audit tables (Phase 2)."""
    enabled: bool = False
    catalog_uri: str = ""
    warehouse: str = ""
    table_name: str = "retinalai.audit_events"


# ── Phase 3: Governance & Edge ───────────────────────────────────────────────


class EdgeSettings(BaseModel):
    """Edge inference formats (Phase 3)."""
    onnx_enabled: bool = False
    onnx_model_path: str = "models/export/model.onnx"
    coreml_enabled: bool = False
    coreml_model_path: str = "models/export/model.mlpackage"
    quantized_enabled: bool = False
    quantized_model_path: str = "models/export/model_int8.pth"
    parity_tolerance: float = 1e-4


class FairnessSettings(BaseModel):
    """Fairness dashboard (Phase 3)."""
    enabled: bool = False
    cache_ttl_s: int = 3600
    protected_attributes: list[str] = ["age_group", "sex", "ethnicity", "camera_device", "geography"]


class ModelCardSettings(BaseModel):
    """Automated model card generation (Phase 3)."""
    auto_generate: bool = False
    output_dir: str = "outputs/governance"


# ── Phase 4: Future-Proofing ────────────────────────────────────────────────


class FundusGateSettings(BaseModel):
    """Fundus gate v2 fusion settings."""
    enabled: bool = True
    version: str = "v2"                    # "v1" | "v2"
    learned_weight: float = 0.4            # weight for learned gate in fusion
    min_confidence: float = 0.70           # fusion confidence threshold
    model_path: str = "weights/fundus_gate.pth"
    visual_evidence: bool = False          # generate base64 heatmaps on rejection
    mc_dropout_samples: int = 5            # uncertainty estimation forward passes


class ResilienceSettings(BaseModel):
    """Graceful degradation (Phase 4)."""
    enabled: bool = False
    health_check_interval_s: float = 30.0
    fallback_cache_ttl_s: int = 300


class MultiModalSettings(BaseModel):
    """Multi-modal fusion skeleton (Phase 4)."""
    enabled: bool = False
    modalities: list[str] = ["fundus"]
    fusion_strategy: str = "concatenation"  # concatenation | cross_attention


class FederatedSettings(BaseModel):
    """Federated learning skeleton (Phase 4)."""
    enabled: bool = False
    framework: str = "flower"  # flower | nvflare
    server_address: str = "localhost:8080"
    local_epochs: int = 3
    dp_enabled: bool = False
    dp_epsilon: float = 10.0


# ── Main Settings ────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    app_name: str = "RetinalAI Clinical Screening Platform"
    app_version: str = "3.0.0"
    debug: bool = False

    # Deployment
    environment: str = "development"  # development | staging | production
    deployment_region: str = "default"

    # Model
    model_path: str = "models/model_vignn_rank1.pth"
    model_name: str = "vignn"
    num_classes: int = 45

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:3330",
        "http://localhost:8080",
        "http://localhost:8088",
        "https://mpairwe49-retinal-screening.hf.space",
        "https://huggingface.co",
    ]

    # GPU
    cuda_visible_devices: str = "0"
    device: str = "auto"  # auto | cpu | cuda

    # Uploads
    max_upload_size: int = 10 * 1024 * 1024  # 10 MB

    # Authentication
    auth_enabled: bool = False  # Set True in production
    jwt_secret: str = "change-me-in-production-use-env-var"
    jwt_expiry_seconds: int = 3600

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # json | text

    # Rate limiting
    rate_limit_per_minute: int = 60

    # Explainability
    explain_gradcam_enabled: bool = True
    explain_lime_default_samples: int = 300
    explain_shap_enabled: bool = True

    # Prediction logging
    prediction_log_dir: str = "logs/predictions"

    # Regulatory
    regulatory_mode: str = "research"  # research | ce_marked | fda_cleared

    # Agentic AI — Claude (primary)
    anthropic_api_key: str = ""
    anthropic_org_id: str = ""
    agent_model: str = "claude-sonnet-4-20250514"

    # Agentic AI — Groq (fallback)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_max_tokens: int = 4096
    groq_temperature: float = 0.3

    # Agent scheduling
    agent_monitor_interval: float = 60.0
    agent_governance_interval: float = 300.0

    # ── Nested feature settings (all opt-in, all default disabled) ──
    telemetry: TelemetrySettings = TelemetrySettings()
    mlflow: MLflowSettings = MLflowSettings()
    active_learning_loop: ActiveLearningLoopSettings = ActiveLearningLoopSettings()
    drift: DriftSettings = DriftSettings()
    ray: RayServeSettings = RayServeSettings()
    canary: CanarySettings = CanarySettings()
    circuit_breaker: CircuitBreakerSettings = CircuitBreakerSettings()
    mtls: MTLSSettings = MTLSSettings()
    kafka: KafkaSettings = KafkaSettings()
    iceberg: IcebergSettings = IcebergSettings()
    edge: EdgeSettings = EdgeSettings()
    fairness: FairnessSettings = FairnessSettings()
    model_card: ModelCardSettings = ModelCardSettings()
    fundus_gate: FundusGateSettings = FundusGateSettings()
    resilience: ResilienceSettings = ResilienceSettings()
    multimodal: MultiModalSettings = MultiModalSettings()
    federated: FederatedSettings = FederatedSettings()

    model_config = {
        "env_prefix": "",
        "env_file": ".env",
        "extra": "ignore",
        "env_nested_delimiter": "__",
    }


settings = Settings()
