#!/bin/bash
# Test CraneCloud Deployment
# Replace YOUR_APP_URL with your actual CraneCloud URL

APP_URL="YOUR_APP_URL.cranecloud.io"

echo "Testing CraneCloud Deployment..."
echo ""

echo "1. Health Check:"
curl -s "https://${APP_URL}/health" | python3 -m json.tool
echo ""

echo "2. Get Diseases:"
curl -s "https://${APP_URL}/diseases" | python3 -m json.tool | head -20
echo ""

echo "3. API Documentation:"
echo "   Open in browser: https://${APP_URL}/docs"
echo ""

echo "4. Check Response Time:"
time curl -s "https://${APP_URL}/health" > /dev/null
echo ""

echo "Done!"
