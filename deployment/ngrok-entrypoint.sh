#!/bin/bash
# ============================================================================
# Ngrok-enabled Entrypoint for Retinal Screening API
# Supports multiple deployment modes: with/without ngrok
# ============================================================================

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Retinal Screening API - Starting...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"

# Configuration
API_PORT="${PORT:-8080}"
NGROK_AUTHTOKEN="${NGROK_AUTHTOKEN:-}"
NGROK_ENABLED="${NGROK_ENABLED:-false}"
NGROK_REGION="${NGROK_REGION:-us}"  # us, eu, ap, au, sa, jp, in
MODE="${1:-api}"  # api, api-with-ngrok, ngrok-only

echo -e "${BLUE}Mode:${NC} $MODE"
echo -e "${BLUE}API Port:${NC} $API_PORT"
echo -e "${BLUE}Ngrok Enabled:${NC} $NGROK_ENABLED"

# Function to start the API server
start_api() {
    echo -e "${GREEN}Starting FastAPI server on port $API_PORT...${NC}"
    cd /app
    exec python3 -m uvicorn src.api_server:app \
        --host 0.0.0.0 \
        --port "$API_PORT" \
        --workers 1 \
        --log-level info
}

# Function to start ngrok
start_ngrok() {
    echo -e "${GREEN}Starting ngrok tunnel...${NC}"
    
    # Check if authtoken is provided
    if [ -n "$NGROK_AUTHTOKEN" ]; then
        echo -e "${BLUE}Configuring ngrok with authtoken...${NC}"
        ngrok config add-authtoken "$NGROK_AUTHTOKEN"
    else
        echo -e "${YELLOW}Warning: No NGROK_AUTHTOKEN provided. Using free tier (limited features).${NC}"
    fi
    
    # Start ngrok in background
    echo -e "${BLUE}Creating ngrok tunnel to localhost:$API_PORT...${NC}"
    ngrok http "$API_PORT" \
        --region "$NGROK_REGION" \
        --log stdout \
        --log-level info \
        --log-format json &
    
    NGROK_PID=$!
    echo -e "${GREEN}Ngrok started with PID: $NGROK_PID${NC}"
    
    # Wait for ngrok to start and get the public URL
    sleep 3
    
    # Try to get the ngrok URL from the API
    for i in {1..10}; do
        NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | \
            python3 -c "import sys, json; data = json.load(sys.stdin); print(data['tunnels'][0]['public_url'] if data.get('tunnels') else '')" 2>/dev/null || echo "")
        
        if [ -n "$NGROK_URL" ]; then
            echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
            echo -e "${GREEN}  ✓ Ngrok tunnel established!${NC}"
            echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
            echo -e "${GREEN}Public URL:${NC} ${YELLOW}$NGROK_URL${NC}"
            echo -e "${GREEN}API Docs:${NC} ${YELLOW}$NGROK_URL/docs${NC}"
            echo -e "${GREEN}Health Check:${NC} ${YELLOW}$NGROK_URL/health${NC}"
            echo -e "${GREEN}Ngrok Dashboard:${NC} ${YELLOW}http://localhost:4040${NC}"
            echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
            
            # Save the URL to a file for easy access
            echo "$NGROK_URL" > /tmp/ngrok_url.txt
            break
        fi
        
        echo -e "${YELLOW}Waiting for ngrok to start... (attempt $i/10)${NC}"
        sleep 2
    done
    
    if [ -z "$NGROK_URL" ]; then
        echo -e "${YELLOW}Warning: Could not retrieve ngrok URL. Check logs at http://localhost:4040${NC}"
    fi
}

# Function to start both API and ngrok using supervisor
start_with_supervisor() {
    echo -e "${GREEN}Starting API and ngrok with supervisor...${NC}"
    
    # Configure ngrok if authtoken is provided
    if [ -n "$NGROK_AUTHTOKEN" ]; then
        ngrok config add-authtoken "$NGROK_AUTHTOKEN"
    fi
    
    # Start supervisor
    exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
}

# Main execution
case "$MODE" in
    "api")
        # Run API only (default mode for production)
        start_api
        ;;
    
    "api-with-ngrok")
        # Run both API and ngrok using supervisor
        NGROK_ENABLED=true
        start_with_supervisor
        ;;
    
    "ngrok-only")
        # Run only ngrok (assumes API is already running elsewhere)
        start_ngrok
        # Keep container running
        tail -f /dev/null
        ;;
    
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Usage: $0 [api|api-with-ngrok|ngrok-only]"
        exit 1
        ;;
esac
