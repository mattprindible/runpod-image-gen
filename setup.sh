#!/bin/bash
# Run this on a fresh pod to get up and running from GitHub.
# Safe to re-run - it just pulls latest changes and restarts.
set -e

REPO_URL="https://github.com/mattprindible/runpod-image-gen.git"
APP_DIR="/workspace/app"

echo "--- Syncing code from GitHub ---"
if [ -d "$APP_DIR/.git" ]; then
    cd "$APP_DIR" && GIT_TERMINAL_PROMPT=0 git pull
else
    GIT_TERMINAL_PROMPT=0 git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi

echo "--- Installing dependencies ---"
pip install -r server/requirements.txt --quiet

# Download model weights to /workspace/models (skips if already there)
echo "--- Checking model weights ---"
python server/download_model.py

echo ""
echo "Setup complete. Start the server with:"
echo "  cd $APP_DIR && uvicorn server.main:app --host 0.0.0.0 --port 8000"
