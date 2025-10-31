#!/bin/bash

# Quick Deploy Script - One Command Deployment
# Run this script to execute the full automated deployment

echo "🚀 Starting Automated Deployment Process..."
echo ""

# Navigate to project directory
cd "$(dirname "$0")"

# Run the main deployment script
./setup-automated-deployment.sh

echo ""
echo "📝 Next step: Push to GitHub to trigger CI/CD"
echo ""
echo "Run these commands:"
echo "  git add ."
echo "  git commit -m 'Automated Docker Hub and Crane Cloud deployment'"
echo "  git push origin main"
echo ""
echo "Then deploy on Crane Cloud: https://cranecloud.io"
