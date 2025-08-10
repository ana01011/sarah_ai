#!/bin/bash

echo "Setting up Amesie AI Backend Service..."

# Configuration
BACKEND_DIR="/root/sarah_ai/amesie_ai_backend"
VENV_DIR="$BACKEND_DIR/venv"

# 1. Stop any conflicting processes
echo "1. Stopping any conflicting processes on port 8000..."
sudo lsof -ti:8000 | xargs -r sudo kill -9
sleep 2

# 2. Make sure Redis is running
echo "2. Starting Redis..."
sudo systemctl start redis-server
sudo systemctl enable redis-server

# 3. Create/activate virtual environment if needed
echo "3. Setting up Python environment..."
cd $BACKEND_DIR

if [ ! -d "$VENV_DIR" ]; then
    echo "   Creating virtual environment..."
    python3 -m venv venv
fi

# 4. Install dependencies
echo "4. Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements-light.txt
deactivate

# 5. Create systemd service
echo "5. Creating systemd service..."
sudo tee /etc/systemd/system/amesie-ai-backend.service > /dev/null << EOF
[Unit]
Description=Amesie AI Backend with OpenHermes
After=network.target redis-server.service

[Service]
Type=simple
User=root
WorkingDirectory=$BACKEND_DIR
Environment="PATH=$VENV_DIR/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTHONPATH=$BACKEND_DIR"
ExecStart=$VENV_DIR/bin/python $BACKEND_DIR/main_openhermes_cpu.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits for 32GB server
MemoryMax=28G
CPUQuota=700%

[Install]
WantedBy=multi-user.target
EOF

# 6. Reload systemd and start service
echo "6. Starting service..."
sudo systemctl daemon-reload
sudo systemctl enable amesie-ai-backend
sudo systemctl start amesie-ai-backend

# 7. Wait for service to initialize
echo "7. Waiting for service to start..."
for i in {1..10}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "   Service is up!"
        break
    fi
    echo "   Waiting... ($i/10)"
    sleep 2
done

# 8. Check service status
echo "8. Service status:"
sudo systemctl status amesie-ai-backend --no-pager

# 9. Test endpoints
echo -e "\n9. Testing API endpoints:"

echo -e "\n   a) Health check:"
curl -s http://localhost:8000/health | python3 -m json.tool || echo "Failed"

echo -e "\n   b) Available roles:"
curl -s http://localhost:8000/api/v1/chat/roles 2>/dev/null | python3 -m json.tool 2>/dev/null | head -15 || echo "No roles endpoint"

echo -e "\n   c) Metrics:"
curl -s http://localhost:8000/api/v1/metrics | python3 -m json.tool || echo "No metrics"

# 10. Test from external IP
echo -e "\n10. Testing from external IP (147.93.102.165):"
curl -s http://147.93.102.165/health || echo "External access failed"

echo -e "\n\n✅ Setup complete! Your Amesie AI Backend should now be running."
echo "   - Internal: http://localhost:8000"
echo "   - External: http://147.93.102.165"
echo ""
echo "To check logs: sudo journalctl -u amesie-ai-backend -f"
echo "To restart: sudo systemctl restart amesie-ai-backend"