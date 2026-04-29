# ============================================================================
# GPU-Enabled Production Dockerfile
# NVIDIA CUDA 12.1 + FastAPI backend + UV package manager
# ============================================================================
FROM docker.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL maintainer="retinal-screening-team"
LABEL description="GPU-accelerated Retinal Disease Screening API"
LABEL version="2.0.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/opt/venv/bin:${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip \
    build-essential git wget curl ca-certificates \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install Python deps via pyproject.toml
COPY pyproject.toml /app/pyproject.toml
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir ".[deploy]"

# Install PyTorch with CUDA support
RUN /opt/venv/bin/pip install --no-cache-dir \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Copy application code
COPY src/ ./src/
COPY backend/ ./backend/
COPY configs/ ./configs/

# Create directories
RUN mkdir -p models/checkpoints logs uploads && chmod -R 755 /app

# Environment
ENV MODEL_PATH=${MODEL_PATH:-src/models/model_vignn_rank1.pth} \
    API_HOST=${API_HOST:-0.0.0.0} \
    API_PORT=${API_PORT:-8080} \
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
