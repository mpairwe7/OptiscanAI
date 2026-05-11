# ============================================================================
# OptiscanAI — Full-Stack GPU Production Image
# Multi-stage: build deps + frontend in builder, slim runtime stage
# Process manager: supervisord → nginx(:8080) + uvicorn(:8081) + node(:3000)
# ============================================================================

# --------------- Stage 1: Builder (Python deps) ---------------
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

# --------------- Stage 2: Frontend build ---------------
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

# --------------- Stage 3: Runtime ---------------
FROM docker.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL maintainer="retinal-screening-team"
LABEL description="OptiscanAI — GPU Retinal Disease Screening Platform"
LABEL version="2.0.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/opt/venv/bin:/usr/local/cuda/bin:${PATH}"

# Runtime deps: python, nginx, supervisor, node (for Next.js SSR), curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 curl ca-certificates \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    nginx supervisor \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy Python venv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy frontend build from frontend-builder
COPY --from=frontend-builder /srv/nextjs /srv/nextjs
COPY --from=frontend-builder /app/frontend/.next/standalone /app/frontend/.next/standalone

WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY backend/ ./backend/
COPY configs/ ./configs/

# Create directories
RUN mkdir -p models/checkpoints logs uploads && chmod -R 755 /app

# --- Nginx config (port 8080 for Crane Cloud) ---
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

# --- Supervisord config (GPU-enabled) ---
RUN cat > /etc/supervisor/conf.d/optiscan.conf <<'SUPER'
[supervisord]
nodaemon=true
logfile=/tmp/supervisord.log
pidfile=/tmp/supervisord.pid

[program:backend]
command=/opt/venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8081
directory=/app
environment=MODEL_PATH="models/model_vignn_rank1.pth",API_PORT="8081",LOG_FORMAT="text"
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
autorestart=true
startretries=3
startsecs=5

[program:frontend]
command=node server.js
directory=/app/frontend/.next/standalone
environment=PORT="3000",HOSTNAME="127.0.0.1"
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
autorestart=true
startretries=3
startsecs=3

[program:nginx]
command=nginx -g "daemon off;"
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
autorestart=true
startretries=3
startsecs=1
SUPER

ENV MODEL_PATH=models/model_vignn_rank1.pth \
    API_HOST=0.0.0.0 \
    API_PORT=8081 \
    CUDA_VISIBLE_DEVICES=0

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/optiscan.conf"]
