#!/bin/bash
# Deploy agent to norisor
# Usage: ./deploy.sh
#
# Pulls latest image from GHCR, updates config from git, restarts containers.
# Run from the agent-runtime directory on norisor.

set -e

echo "=== Agent Deploy ==="

# Pull latest config/state changes from git
echo "[1/4] Pulling latest from git..."
git pull origin main

# Log in to GHCR (uses GitHub token from env or prompts)
echo "[2/4] Pulling latest Docker image..."
docker compose pull agent

# Restart with new image
echo "[3/4] Restarting containers..."
docker compose up -d

# Verify
echo "[4/4] Verifying..."
sleep 3
docker compose ps

echo ""
echo "=== Deploy complete ==="
echo "Attach to agent: docker attach agent_001"
echo "View logs:       docker compose logs -f agent"
