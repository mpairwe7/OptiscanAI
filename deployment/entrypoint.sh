#!/bin/bash
set -e

echo "🚀 Starting Retinal Screening Application..."
echo "================================================"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU Available:"
    nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader
else
    echo "⚠️  No GPU detected - running on CPU"
fi

# Set environment variables
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Create necessary directories
mkdir -p /app/uploads
mkdir -p /app/models
mkdir -p /app/logs

# Start supervisord to manage multiple processes
echo "================================================"
echo "📊 Starting Streamlit on port 8501..."
echo "🔧 Starting API server on port 8080..."
echo "================================================"

exec /usr/bin/supervisord -c /app/deployment/supervisord.conf
