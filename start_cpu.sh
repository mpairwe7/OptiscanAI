#!/bin/bash
# ============================================================================
# CPU-Only Application Startup Script (No ngrok)
# ============================================================================

set -e

echo "Starting Retinal Disease Classification API (CPU)..."

# Set environment variables if not set
export API_HOST=${API_HOST:-0.0.0.0}
export API_PORT=${API_PORT:-8080}
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export STREAMLIT_SERVER_PORT=${STREAMLIT_SERVER_PORT:-8501}
export STREAMLIT_SERVER_ADDRESS=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}
export STREAMLIT_SERVER_HEADLESS=${STREAMLIT_SERVER_HEADLESS:-true}
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=${STREAMLIT_BROWSER_GATHER_USAGE_STATS:-false}

# Create necessary directories
mkdir -p /app/uploads
mkdir -p /app/models
mkdir -p /app/logs

echo "================================================"
echo " Starting Streamlit on port ${STREAMLIT_SERVER_PORT}..."
echo " Starting API server on port ${API_PORT}..."
echo "================================================"

# Start supervisord in foreground to manage app processes and keep container alive
exec /usr/bin/supervisord -c /app/deployment/supervisord.conf