#!/bin/bash
# ============================================================================
# Application Startup Script
# ============================================================================

set -e

echo "Starting Retinal Disease Classification API..."

# Change to application directory
cd /app

# Set environment variables if not set
export API_HOST=${API_HOST:-0.0.0.0}
export API_PORT=${API_PORT:-8080}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

# Start the FastAPI server
echo "Starting FastAPI server on ${API_HOST}:${API_PORT}"
exec python src/api_server.py