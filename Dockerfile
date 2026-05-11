# ============================================================================
# OptiscanAI — Full-Stack GPU Production Image
# 3-stage multi-stage build | OCI-compliant | Non-root | Graceful shutdown
# Process manager: supervisord → nginx(:8080) + uvicorn(:8081) + node(:3000)
# ============================================================================

# --------------- Stage 1: Python builder ---------------
FROM docker.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS builder
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev \
    build-essential git curl ca-certificates \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

WORKDIR /app

# Install PyTorch CUDA FIRST — prevents double-download from pyproject.toml
RUN pip install --upgrade pip && \
    pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Install project deps — torch already satisfied, skip re-download
COPY pyproject.toml /app/pyproject.toml
RUN pip install ".[deploy]" && \
    pip uninstall -y streamlit altair pydeck pyarrow 2>/dev/null; \
    pip install opencv-python-headless && \
    pip uninstall -y opencv-python 2>/dev/null; \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -name "*.pyc" -delete 2>/dev/null; \
    find /opt/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -name "*.dist-info" -type d -exec rm -rf {} + 2>/dev/null; \
    true

# --------------- Stage 2: Frontend builder ---------------
FROM node:20-slim AS frontend-builder

RUN npm install -g bun

WORKDIR /app/frontend

COPY frontend/package.json frontend/bun.lock* ./
RUN bun install --frozen-lockfile 2>/dev/null || bun install

COPY frontend/ ./
RUN NEXT_PUBLIC_API_URL="" bun run build \
    && mkdir -p /srv/nextjs/_next/static \
    && cp -r .next/static/* /srv/nextjs/_next/static/ \
    && cp -r public/* /srv/nextjs/ \
    && mkdir -p .next/standalone/.next \
    && cp -r .next/static .next/standalone/.next/static \
    && cp -r public .next/standalone/public

# --------------- Stage 3: Production runtime ---------------
FROM docker.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# OCI standard labels (https://github.com/opencontainers/image-spec/blob/main/annotations.md)
LABEL maintainer="retinal-screening-team"
LABEL org.opencontainers.image.title="OptiscanAI"
LABEL org.opencontainers.image.description="GPU Retinal Disease Screening Platform — Backend + Frontend"
LABEL org.opencontainers.image.version="2.0.0"
LABEL org.opencontainers.image.vendor="OptiscanAI"
LABEL org.opencontainers.image.source="https://github.com/mpairwe7/MLOPS_V1"
LABEL org.opencontainers.image.licenses="CC-BY-4.0"

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/opt/venv/bin:/usr/local/cuda/bin:${PATH}"

# Runtime deps — pinned Node.js 20 from nodesource without curl|bash
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3 curl ca-certificates gnupg \
       libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
       nginx supervisor \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
       | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
       > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get purge -y gnupg \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy Python venv from builder (no compilers, no pip cache)
COPY --from=builder /opt/venv /opt/venv

# Copy frontend build artifacts only
COPY --from=frontend-builder /srv/nextjs /srv/nextjs
COPY --from=frontend-builder /app/frontend/.next/standalone /app/frontend/.next/standalone

WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY backend/ ./backend/
COPY configs/ ./configs/

# Bake model weights (required for Crane Cloud — no volume mounts)
COPY models/model_vignn_rank1.pth ./models/model_vignn_rank1.pth
COPY weights/fundus_gate.pth ./weights/fundus_gate.pth

# Non-root user + directories in single layer (no extra chmod layer)
RUN groupadd -r optiscan && useradd -r -g optiscan -d /app -s /sbin/nologin optiscan \
    && mkdir -p models/checkpoints logs uploads \
    && chown -R optiscan:optiscan /app /srv/nextjs \
    && chown -R optiscan:optiscan /var/log/nginx /var/lib/nginx /run \
    && ln -sf /dev/stdout /var/log/nginx/access.log \
    && ln -sf /dev/stderr /var/log/nginx/error.log

# Nginx config — single port 8080 (non-privileged, Crane Cloud compatible)
RUN cat > /etc/nginx/sites-available/default <<'NGINX'
server {
    listen 8080;
    server_name _;
    client_max_body_size 15M;

    location /_next/static/ {
        root /srv/nextjs;
        expires 365d;
        access_log off;
        add_header Cache-Control "public, immutable";
    }

    location ~* \.(png|ico|svg|webmanifest|js)$ {
        root /srv/nextjs;
        try_files $uri @nextjs;
        expires 30d;
        access_log off;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8081/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }

    location /v1/ {
        proxy_pass http://127.0.0.1:8081/v1/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_read_timeout 120s;
    }

    location /health {
        proxy_pass http://127.0.0.1:8081/health;
        proxy_http_version 1.1;
    }

    location /docs {
        proxy_pass http://127.0.0.1:8081/docs;
        proxy_http_version 1.1;
    }

    location /openapi.json {
        proxy_pass http://127.0.0.1:8081/openapi.json;
        proxy_http_version 1.1;
    }

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_cache_bypass $http_upgrade;
    }

    location @nextjs {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
NGINX

# Supervisord — graceful shutdown with SIGTERM propagation
RUN cat > /etc/supervisor/conf.d/optiscan.conf <<'SUPER'
[supervisord]
nodaemon=true
logfile=/tmp/supervisord.log
pidfile=/tmp/supervisord.pid
user=root

[program:backend]
command=/opt/venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8081
directory=/app
user=optiscan
environment=MODEL_PATH="models/model_vignn_rank1.pth",API_PORT="8081",LOG_FORMAT="text"
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
autorestart=true
startretries=3
startsecs=5
stopwaitsecs=10
stopsignal=TERM

[program:frontend]
command=node server.js
directory=/app/frontend/.next/standalone
user=optiscan
environment=PORT="3000",HOSTNAME="127.0.0.1"
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
autorestart=true
startretries=3
startsecs=3
stopwaitsecs=5
stopsignal=TERM

[program:nginx]
command=nginx -g "daemon off;"
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
autorestart=true
startretries=3
startsecs=1
stopwaitsecs=5
stopsignal=QUIT
SUPER

ENV MODEL_PATH=models/model_vignn_rank1.pth \
    API_HOST=0.0.0.0 \
    API_PORT=8081 \
    CUDA_VISIBLE_DEVICES=0

EXPOSE 8080

STOPSIGNAL SIGTERM

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/optiscan.conf"]
