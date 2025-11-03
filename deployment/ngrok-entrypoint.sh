#!/bin/bash
set -e

echo "🌐 Starting Retinal Screening with ngrok tunneling..."
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

# Check for ngrok auth token
if [ -z "$NGROK_AUTH_TOKEN" ]; then
    echo "⚠️  WARNING: NGROK_AUTH_TOKEN not set!"
    echo "    Set it with: export NGROK_AUTH_TOKEN=your_token"
    echo "    Get token from: https://dashboard.ngrok.com/get-started/your-authtoken"
fi

# Start ngrok in background if auth token is available
if [ ! -z "$NGROK_AUTH_TOKEN" ]; then
    echo "🔐 Configuring ngrok with auth token..."
    ngrok config add-authtoken "$NGROK_AUTH_TOKEN"
    
    echo "🌐 Starting ngrok tunnel..."
    ngrok http 8501 --log=stdout > /app/logs/ngrok.log 2>&1 &
    
    # Wait for ngrok to start and get URL
    sleep 3
    
    if command -v curl &> /dev/null; then
        NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"https://[^"]*' | grep -o 'https://[^"]*' | head -1)
        if [ ! -z "$NGROK_URL" ]; then
            echo "================================================"
            echo "✅ Ngrok tunnel established!"
            echo "🌐 Public URL: $NGROK_URL"
            echo "📊 Ngrok Dashboard: http://localhost:4040"
            echo "================================================"
        fi
    fi
else
    echo "⚠️  Ngrok not started - no auth token provided"
    echo "   Application will only be accessible locally"
fi

# Start supervisord to manage multiple processes
echo "================================================"
echo "📊 Starting Streamlit on port 8501..."
echo "🔧 Starting API server on port 8080..."
echo "================================================"

exec /usr/bin/supervisord -c /app/deployment/supervisord.conf
