"""Application configuration with pydantic-settings.

All Phase 1-4 features are opt-in via nested settings with env_nested_delimiter='__'.
Example: TELEMETRY__ENABLED=true, MLFLOW__TRACKING_URI=http://mlflow:5000
"""
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


class OfflineRAGSettings(BaseModel):
    """Offline RAG pipeline (Phase 5: Offline-First)."""
    enabled: bool = False
    index_dir: str = "data/offline_rag/index"
    source_dir: str = "data/offline_rag/source"
    bundles_dir: str = "data/offline_rag/bundles"
    embedder_path: str = "models/embedder/bge-m3-quantized.onnx"
    top_k: int = 5
    similarity_threshold: float = 0.45
    sync_interval_s: float = 3600.0
    compression: str = "gzip"  # gzip | zstd
    target_bundle_size_mb: int = 150


class QuantizationSettings(BaseModel):
    """Quantization pipeline and server optimization (Phase 5)."""
    enabled: bool = False
    active_format: str = ""                 # gguf_q4_k_m | awq_4bit | onnx_int8 | etc.
    models_dir: str = "outputs/quantized"
    torch_compile_enabled: bool = False
    torch_compile_mode: str = "max-autotune"  # default | reduce-overhead | max-autotune
    prefix_cache_enabled: bool = False
    speculative_decoding_enabled: bool = False
    speculative_draft_model: str = ""       # path to draft model for speculative decoding
    embedder_format: str = ""               # onnx_int8 | fp16 | etc.
    vllm_enabled: bool = False
    vllm_gpu_memory_fraction: float = 0.9
    max_faithfulness_drop: float = 0.04     # max 4% faithfulness drop from baseline
    max_p95_latency_ms: float = 1800.0      # target: <= 1.8s for full RAG


class VoiceFirstSettings(BaseModel):
    """Voice-first mobile experience (Phase 5)."""
    enabled: bool = False
    default_language: str = "en-ug"         # Ugandan English
    asr_model: str = "whisper-tiny"         # whisper-tiny | whisper-base
    asr_model_path: str = "models/voice/whisper-tiny.onnx"
    tts_engine: str = "piper"              # piper | sherpa
    tts_model_path: str = "models/voice/piper-en-ug.onnx"
    vad_sensitivity: float = 0.6            # 0.0-1.0, higher = more sensitive
    barge_in_enabled: bool = True
    speech_rate: float = 1.0
    max_recording_seconds: float = 30.0
    accent_adaptation_enabled: bool = False


class MobileBundleSettings(BaseModel):
    """Mobile bundle optimization (Phase 5)."""
    enabled: bool = False
    max_bundle_size_mb: int = 800
    target_model: str = "gemma-2-2b-q4_k_m"
    embedder_model: str = "bge-m3-4bit-onnx"
    faiss_index_path: str = "data/offline_rag/index/index.faiss"
    include_voice_models: bool = True
    min_android_sdk: int = 21
    min_ram_mb: int = 4096
    bundle_output_dir: str = "outputs/mobile_bundle"


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
    """Federated learning (Phase 4)."""
    enabled: bool = False
    framework: str = "flower"  # flower | nvflare
    server_address: str = "localhost:8080"
    local_epochs: int = 3
    dp_enabled: bool = False
    dp_epsilon: float = 10.0
    dp_delta: float = 1e-5
    dp_clip_norm: float = 1.0
    lora_only: bool = True
    secure_aggregation: bool = False
    num_sim_clients: int = 5


class DHIS2Settings(BaseModel):
    """DHIS2 Uganda health information system integration (Phase 3)."""
    enabled: bool = False
    base_url: str = "https://dhis2.health.go.ug"
    auth_method: str = "pat"  # pat | oauth2
    personal_access_token: str = ""
    oauth2_client_id: str = ""
    oauth2_client_secret: str = ""
    screening_program_id: str = ""
    data_set_id: str = ""
    queue_dir: str = "data/dhis2_queue"
    auto_flush_interval_s: float = 300.0
    request_timeout_s: float = 30.0


class MobileMoneySettings(BaseModel):
    """Mobile money integration — MTN MoMo + Airtel Money (Phase 3)."""
    enabled: bool = False
    mtn_api_key: str = ""
    mtn_api_secret: str = ""
    mtn_subscription_key: str = ""
    mtn_environment: str = "sandbox"  # sandbox | production
    mtn_callback_secret: str = ""  # HMAC shared secret for callback verification
    airtel_client_id: str = ""
    airtel_client_secret: str = ""
    airtel_callback_secret: str = ""
    default_transport_amount_ugx: int = 50000
    # FX rate for converting plan USD prices → UGX for mobile-money charges.
    # In production, read from an FX provider; for MVP this is a config knob.
    ugx_per_usd: int = 3800


class StripeSettings(BaseModel):
    """Stripe card-payment integration (subscription billing)."""
    enabled: bool = False
    api_key: str = ""  # sk_test_... or sk_live_...
    publishable_key: str = ""  # pk_test_... (exposed to frontend via /plans)
    webhook_secret: str = ""  # whsec_... — for stripe.Webhook.construct_event
    success_url: str = "https://www.optiscan.makstartup.com/app/checkout/success?session_id={CHECKOUT_SESSION_ID}"
    cancel_url: str = "https://www.optiscan.makstartup.com/app/billing"
    portal_return_url: str = "https://www.optiscan.makstartup.com/app/billing"

    # Stripe Price IDs — create these in the Stripe dashboard (one per plan + cycle)
    # then set via STRIPE__CLINICIAN_MONTHLY_PRICE_ID=price_xxx etc.
    clinician_monthly_price_id: str = ""
    clinician_annual_price_id: str = ""
    practice_monthly_price_id: str = ""
    practice_annual_price_id: str = ""

    # Extra-seat add-on prices (Practice tier). Per-seat, per cycle.
    # Recommended: USD 25/seat/mo, USD 250/seat/yr.
    practice_extra_seat_monthly_price_id: str = ""
    practice_extra_seat_annual_price_id: str = ""
    practice_extra_seat_monthly_cents: int = 2500
    practice_extra_seat_annual_cents: int = 25000


class FlutterwaveSettings(BaseModel):
    """Flutterwave pan-African aggregator (cards + MoMo)."""
    enabled: bool = False
    public_key: str = ""
    secret_key: str = ""
    encryption_key: str = ""
    secret_hash: str = ""  # for verif-hash header verification
    base_url: str = "https://api.flutterwave.com/v3"


class BillingSettings(BaseModel):
    """Subscription billing platform settings."""
    enabled: bool = False  # master switch — when False, predict.py skips quota
    free_scan_limit_monthly: int = 10
    free_period_days: int = 30  # rolling 30-day window for free-tier users
    quota_cache_ttl_s: int = 60  # in-process cache for (org, period) quota state
    annual_discount_pct: float = 0.17  # ~17% off (matches Grok)


class EmailSettings(BaseModel):
    """Outbound email — used for verification, magic links, password reset, invites."""
    enabled: bool = False
    provider: str = "console"  # console | smtp | sendgrid | resend
    from_address: str = "noreply@makstartup.com"
    from_name: str = "OptiscanAI"
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    sendgrid_api_key: str = ""
    resend_api_key: str = ""
    magic_link_ttl_seconds: int = 900  # 15 min
    verification_link_ttl_seconds: int = 86400  # 24 hr
    password_reset_ttl_seconds: int = 3600  # 1 hr
    invite_ttl_seconds: int = 604800  # 7 days


class DatabaseSettings(BaseModel):
    """Postgres connection (async, asyncpg driver)."""
    enabled: bool = False  # when False, billing/auth features are disabled
    url: str = "postgresql+asyncpg://optiscan:optiscan@localhost:5432/optiscan"
    pool_size: int = 5
    max_overflow: int = 10
    pool_pre_ping: bool = True
    echo: bool = False  # log all SQL — only for debugging


class AfricasTalkingSettings(BaseModel):
    """Africa's Talking SMS/USSD integration (Phase 3)."""
    enabled: bool = False
    api_key: str = ""
    username: str = "sandbox"
    sender_id: str = "RetinalAI"


class PrivacySettings(BaseModel):
    """Uganda PDP Act 2019 privacy compliance (Phase 3)."""
    enabled: bool = True
    consent_required: bool = True
    consent_storage_dir: str = "data/consent"
    data_retention_days: int = 730  # 2 years
    cross_border_allowed_countries: list[str] = ["UG", "KE", "TZ", "RW"]
    purpose_limitation_enabled: bool = True
    anonymize_exports: bool = True


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
        # Production — HTTPS canonical
        "https://www.optiscan.makstartup.com",
        "https://optiscan.makstartup.com",
        # Production — HTTP variants (the edge proxy upgrades to HTTPS, but
        # listing them keeps CORS happy during cert renewals or http-only probes).
        "http://www.optiscan.makstartup.com",
        "http://optiscan.makstartup.com",
        # Legacy
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
    offline_rag: OfflineRAGSettings = OfflineRAGSettings()
    quantization: QuantizationSettings = QuantizationSettings()
    voice_first: VoiceFirstSettings = VoiceFirstSettings()
    mobile_bundle: MobileBundleSettings = MobileBundleSettings()
    resilience: ResilienceSettings = ResilienceSettings()
    multimodal: MultiModalSettings = MultiModalSettings()
    federated: FederatedSettings = FederatedSettings()

    # Phase 3: Uganda health ecosystem
    dhis2: DHIS2Settings = DHIS2Settings()
    mobile_money: MobileMoneySettings = MobileMoneySettings()
    africastalking: AfricasTalkingSettings = AfricasTalkingSettings()
    privacy: PrivacySettings = PrivacySettings()

    # Phase 6: Subscription billing platform
    database: DatabaseSettings = DatabaseSettings()
    billing: BillingSettings = BillingSettings()
    email: EmailSettings = EmailSettings()
    stripe: StripeSettings = StripeSettings()
    flutterwave: FlutterwaveSettings = FlutterwaveSettings()

    # JWT (rewritten in core/auth.py — keys here)
    jwt_algorithm: str = "HS256"
    jwt_access_ttl_seconds: int = 900  # 15 min
    jwt_refresh_ttl_seconds: int = 2592000  # 30 days
    public_app_url: str = "https://www.optiscan.makstartup.com"  # for email links

    model_config = {
        "env_prefix": "",
        "env_file": ".env",
        "extra": "ignore",
        "env_nested_delimiter": "__",
    }


settings = Settings()
