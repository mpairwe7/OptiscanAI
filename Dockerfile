# ============================================================================
# GPU-Enabled Production Dockerfile for Crane Cloud Deployment
# Uses NVIDIA CUDA base image for GPU acceleration
# ============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Metadata
LABEL maintainer="retinal-screening-team"
LABEL description="GPU-accelerated Retinal Disease Screening API for Crane Cloud"
LABEL version="2.0.0"

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=${CUDA_HOME}/bin:${PATH} \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set working directory
WORKDIR /app

# Install system dependencies including Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA support first
RUN pip3 install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install remaining requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Install remaining requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Create necessary directories with proper permissions
RUN mkdir -p models/checkpoints models/exports logs uploads \
    && chmod -R 755 /app

# Expose port for API (Crane Cloud typically uses 80 or 8080)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application with uvicorn
CMD ["python3", "-m", "uvicorn", "src.api_server:app", \
    "--host", "0.0.0.0", \
    "--port", "8080", \
    "--workers", "1"]
