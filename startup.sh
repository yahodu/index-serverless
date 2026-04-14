#!/bin/bash
set -e

# Install dependencies
pip install -r requirements.txt

# Download InsightFace models
mkdir -p ~/.insightface/models
# Add model download commands if needed

# Start face server in background
python face_server.py > /var/log/face-server.log 2>&1 &

# Wait for face server to be ready
echo "Waiting for face server..."
timeout=60
while ! curl -s http://localhost:18000/health > /dev/null; do
    sleep 1
    timeout=$((timeout - 1))
    if [ $timeout -le 0 ]; then
        echo "Face server failed to start"
        exit 1
    fi
done

# Start PyWorker (foreground)
python worker.py
