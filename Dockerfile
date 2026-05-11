# ============================================================================
# GPU Production Dockerfile — Optimized for Crane Cloud
# Multi-stage: build deps in builder, copy only runtime to final
# ============================================================================

# --------------- Stage 1: Builder ---------------
FROM docker.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev \
    build-essential git curl ca-certificates \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

WORKDIR /app

# Install PyTorch CUDA FIRST — prevents double-download
# (pyproject.toml has torch>=2.0 which would pull CPU torch otherwise)
RUN /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Install project deps — torch already satisfied, won't re-download
COPY pyproject.toml /app/pyproject.toml
RUN /opt/venv/bin/pip install --no-cache-dir ".[deploy]" && \
    /opt/venv/bin/pip uninstall -y streamlit altair pydeck pyarrow 2>/dev/null; \
    /opt/venv/bin/pip install --no-cache-dir opencv-python-headless && \
    /opt/venv/bin/pip uninstall -y opencv-python 2>/dev/null; \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -name "*.pyc" -delete 2>/dev/null; \
    find /opt/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null; \
    true

# --------------- Stage 2: Runtime ---------------
FROM docker.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL maintainer="retinal-screening-team"
LABEL description="OptiscanAI — GPU Retinal Disease Screening API"
LABEL version="2.0.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/opt/venv/bin:/usr/local/cuda/bin:${PATH}"

# Runtime-only system deps (no build-essential, no git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 curl ca-certificates \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder (no compilers, no pip cache)
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy only application code
COPY src/ ./src/
COPY backend/ ./backend/
COPY configs/ ./configs/

# Create dirs and non-root user
RUN mkdir -p models/checkpoints logs uploads && \
    groupadd -r appuser && useradd -r -g appuser -d /app appuser && \
    chown -R appuser:appuser /app

USER appuser

ENV MODEL_PATH=models/model_vignn_rank1.pth \
    API_HOST=0.0.0.0 \
    API_PORT=8080 \
    CUDA_VISIBLE_DEVICES=0

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
