# ============================================================================
# GPU-Enabled Production Dockerfile for Crane Cloud Deployment
# Uses NVIDIA CUDA base image with virtual environment
# ============================================================================
FROM docker.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Metadata
LABEL maintainer="retinal-screening-team"
LABEL description="GPU-accelerated Retinal Disease Screening API for Crane Cloud"
LABEL version="2.1.0"

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/opt/venv/bin:${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies including Python and build tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    build-essential \
    git \
    wget \
    ca-certificates \
    curl \
    gnupg \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Add ngrok apt repo and key (modern signed-by pattern)
RUN curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
    | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
    && echo "deb https://ngrok-agent.s3.amazonaws.com bookworm main" \
    | tee /etc/apt/sources.list.d/ngrok.list \
    && apt update \
    && apt install -y ngrok \
    && rm -rf /var/lib/apt/lists/*

# Ensure `python` points to python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Create virtual environment and install requirements
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r /app/requirements.txt

# Install PyTorch with CUDA support in virtual environment
RUN /opt/venv/bin/pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY deployment/ ./deployment/

# Create necessary directories with proper permissions
RUN mkdir -p models/checkpoints models/exports logs uploads \
    && chmod -R 755 /app

# Copy and setup entrypoint script
COPY start_gpu.sh /app/start.sh
RUN chmod +x /app/start.sh

# Set application environment variables
ENV MODEL_PATH=${MODEL_PATH:-models/model_vignn_rank1.pth} \
    API_HOST=${API_HOST:-0.0.0.0} \
    API_PORT=${API_PORT:-8080} \
    LOG_LEVEL=${LOG_LEVEL:-INFO} \
    STREAMLIT_SERVER_PORT=${STREAMLIT_SERVER_PORT:-8501} \
    STREAMLIT_SERVER_ADDRESS=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0} \
    STREAMLIT_SERVER_HEADLESS=${STREAMLIT_SERVER_HEADLESS:-true} \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=${STREAMLIT_BROWSER_GATHER_USAGE_STATS:-false} \
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    NGROK_REGION=${NGROK_REGION:-us} \
    NGROK_PORT=${NGROK_PORT:-8501} \
    NGROK_LOG_LEVEL=${NGROK_LOG_LEVEL:-stdout} \
    NGROK_BASIC_AUTH=${NGROK_BASIC_AUTH:-} \
    NGROK_OAUTH_PROVIDER=${NGROK_OAUTH_PROVIDER:-}

# Expose ports for API and Streamlit
EXPOSE 8080 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || curl -f http://localhost:8501 || exit 1

# Default command
CMD ["/app/start.sh"]
