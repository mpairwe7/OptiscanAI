#!/usr/bin/env bash
set -euo pipefail

echo " Starting Retinal Screening with ngrok tunneling..."
echo "================================================"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo " GPU Available:"
    nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader
else
    echo "  No GPU detected - running on CPU"
fi

# Set environment variables from Docker ENV or defaults
export STREAMLIT_SERVER_PORT=${STREAMLIT_SERVER_PORT:-8501}
export STREAMLIT_SERVER_ADDRESS=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}
export STREAMLIT_SERVER_HEADLESS=${STREAMLIT_SERVER_HEADLESS:-true}
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=${STREAMLIT_BROWSER_GATHER_USAGE_STATS:-false}

# Create necessary directories
mkdir -p /app/uploads
mkdir -p /app/models
mkdir -p /app/logs

# Configure ngrok with auth token
echo "🔐 Configuring ngrok with auth token..."
ngrok config add-authtoken "36LaG6DcP3gP52RloT2LjJ5JeRQ_SLBZDfTrvtV6ppGS1RVf" || true

# Optional: Set ngrok configuration
if [ -n "${NGROK_BASIC_AUTH:-}" ]; then
    echo "🔒 Setting ngrok basic auth..."
    ngrok config add-api-key "basic_auth" "$NGROK_BASIC_AUTH" || true
fi

if [ -n "${NGROK_OAUTH_PROVIDER:-}" ]; then
    echo "🔐 Setting ngrok OAuth provider..."
    ngrok config add-api-key "oauth_provider" "$NGROK_OAUTH_PROVIDER" || true
fi

# Start supervisord in background to manage app processes
echo "================================================"
echo " Starting Streamlit on port 8501..."
echo " Starting API server on port 8080..."
echo "================================================"

/usr/bin/supervisord -c /app/deployment/supervisord.conf &

# Wait for services to start
sleep 5

echo " Starting ngrok tunnel in foreground..."
echo "================================================"
echo " Services running in background"
echo " Ngrok tunnel will keep container alive"
echo " Ngrok Dashboard: http://localhost:4040"
echo "================================================"

# Run ngrok in foreground to keep container alive
NGROK_PORT=${NGROK_PORT:-8501}
exec ngrok http --region="${NGROK_REGION:-us}" --log="${NGROK_LOG_LEVEL:-stdout}" "${NGROK_PORT}"
