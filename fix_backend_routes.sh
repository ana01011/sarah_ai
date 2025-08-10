#!/bin/bash

echo "Fixing Amesie AI Backend Routes..."

# Stop the current service
echo "1. Stopping current service..."
sudo systemctl stop amesie-ai-backend

# Check which directory we're working with
BACKEND_DIR="/opt/amesie_ai_backend"
if [ ! -d "$BACKEND_DIR" ]; then
    BACKEND_DIR="$HOME/sarah_ai/amesie_ai_backend"
fi

echo "Using backend directory: $BACKEND_DIR"

# Update the systemd service to use the correct main file
echo "2. Updating systemd service..."
sudo tee /etc/systemd/system/amesie-ai-backend.service > /dev/null << EOF
[Unit]
Description=Amesie AI Backend
After=network.target redis-server.service

[Service]
Type=simple
User=root
WorkingDirectory=$BACKEND_DIR
Environment="PATH=$BACKEND_DIR/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=$BACKEND_DIR/venv/bin/python $BACKEND_DIR/main_openhermes_cpu.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits
MemoryMax=28G
CPUQuota=700%

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
echo "3. Reloading systemd..."
sudo systemctl daemon-reload

# Start the service
echo "4. Starting service with correct main file..."
sudo systemctl start amesie-ai-backend

# Wait for service to start
echo "5. Waiting for service to initialize..."
sleep 5

# Check status
echo "6. Checking service status..."
sudo systemctl status amesie-ai-backend --no-pager

# Test the endpoints
echo -e "\n7. Testing endpoints..."

echo -e "\n   Testing /health..."
curl -s http://localhost:8000/health | python3 -m json.tool

echo -e "\n   Testing /api/v1/chat/roles..."
curl -s http://localhost:8000/api/v1/chat/roles | python3 -m json.tool | head -10

echo -e "\n   Testing from external IP..."
curl -s http://147.93.102.165/health
echo ""
curl -s http://147.93.102.165/api/v1/chat/roles | head -20

echo -e "\n✅ Service updated to use main_openhermes_cpu.py with proper API routes!"