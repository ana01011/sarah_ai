#!/bin/bash

# Full OpenHermes Model Deployment Script - FIXED VERSION
# For server with 32GB RAM, 8 vCores, 400GB storage
# Target: 147.93.102.165

set -e

echo "=========================================="
echo " Full OpenHermes Model Deployment"
echo " Server: 147.93.102.165 (32GB RAM)"
echo "=========================================="

# Configuration
SERVER_IP="147.93.102.165"
REMOTE_DIR="/opt/amesie_ai_backend"
SERVICE_NAME="amesie-ai"
MODEL_CACHE_DIR="/opt/models"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# We're already on the server, so skip the SSH parts
echo ""
print_status "Starting local deployment..."

# Get current directory
CURRENT_DIR=$(pwd)
print_status "Current directory: $CURRENT_DIR"

# Check if required files exist
echo ""
print_status "Checking required files..."

if [ ! -f "main_openhermes_cpu.py" ]; then
    print_error "main_openhermes_cpu.py not found!"
    echo "Please make sure you're in the amesie_ai_backend directory"
    exit 1
fi

if [ ! -f "requirements-cpu.txt" ]; then
    print_error "requirements-cpu.txt not found!"
    echo "Please make sure you have all required files"
    exit 1
fi

print_status "Required files found"

# Copy files to /opt/amesie_ai_backend if not already there
if [ "$CURRENT_DIR" != "/opt/amesie_ai_backend" ]; then
    echo ""
    print_status "Copying files to /opt/amesie_ai_backend..."
    mkdir -p /opt/amesie_ai_backend
    cp -r * /opt/amesie_ai_backend/ 2>/dev/null || true
    cd /opt/amesie_ai_backend
    print_status "Files copied to /opt/amesie_ai_backend"
else
    print_status "Already in /opt/amesie_ai_backend"
fi

# Setup server environment
echo ""
print_status "Setting up server environment..."

echo "Installing system dependencies..."
apt-get update -qq
apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    build-essential cmake \
    libopenblas-dev libomp-dev \
    redis-server nginx \
    htop iotop \
    git curl wget

# Create model cache directory
echo "Creating model cache directory..."
mkdir -p /opt/models
export HF_HOME=/opt/models
export TRANSFORMERS_CACHE=/opt/models

# Configure swap (important for large models)
echo "Configuring swap space..."
if [ ! -f /swapfile ]; then
    fallocate -l 16G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    echo "16GB swap space created"
else
    print_status "Swap already configured"
fi

# Optimize system for ML workloads
echo "Optimizing system settings..."
# Increase file descriptors
grep -q "nofile 65536" /etc/security/limits.conf || {
    echo "* soft nofile 65536" >> /etc/security/limits.conf
    echo "* hard nofile 65536" >> /etc/security/limits.conf
}

# Optimize memory settings
sysctl -w vm.swappiness=10
sysctl -w vm.vfs_cache_pressure=50
grep -q "vm.swappiness" /etc/sysctl.conf || echo "vm.swappiness=10" >> /etc/sysctl.conf
grep -q "vm.vfs_cache_pressure" /etc/sysctl.conf || echo "vm.vfs_cache_pressure=50" >> /etc/sysctl.conf

# Setup Python environment
echo ""
print_status "Setting up Python environment..."

# Remove old venv if exists
rm -rf venv

# Create new virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip and install wheel
pip install --upgrade pip wheel setuptools

# Install PyTorch CPU version first (to avoid downloading CUDA version)
echo ""
print_status "Installing PyTorch CPU version..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
echo ""
print_status "Installing other requirements..."
pip install transformers==4.36.2
pip install accelerate==0.25.0
pip install fastapi==0.104.1
pip install uvicorn[standard]==0.24.0
pip install psutil==5.9.6
pip install python-dotenv==1.0.0
pip install websockets==12.0
pip install httpx
pip install pydantic
pip install redis==5.0.1
pip install structlog==23.2.0
pip install prometheus-client==0.19.0

# Pre-download the model to avoid timeout during first run
echo ""
print_warning "Pre-downloading OpenHermes model (this may take 10-20 minutes)..."
print_warning "The model is about 14GB in size..."

python3 << 'PYTHONSCRIPT'
import os
os.environ['HF_HOME'] = '/opt/models'
os.environ['TRANSFORMERS_CACHE'] = '/opt/models'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Downloading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
    print("✓ Tokenizer downloaded successfully!")
except Exception as e:
    print(f"Error downloading tokenizer: {e}")
    print("You may need to download it manually later")

print("\nDownloading model (7B parameters, ~14GB)...")
print("This will take some time depending on your internet speed...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        "teknium/OpenHermes-2.5-Mistral-7B",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    print("✓ Model downloaded successfully!")
    del model
    del tokenizer
    print("✓ Memory cleared")
except Exception as e:
    print(f"Error downloading model: {e}")
    print("The model will be downloaded on first run")
PYTHONSCRIPT

# Create .env file
echo ""
print_status "Creating configuration file..."
cat > .env << 'EOF'
HF_HOME=/opt/models
TRANSFORMERS_CACHE=/opt/models
MODEL_NAME=teknium/OpenHermes-2.5-Mistral-7B
TORCH_NUM_THREADS=8
CORS_ORIGINS=*
EOF

# Start Redis
print_status "Starting Redis..."
systemctl enable redis-server
systemctl start redis-server || true

# Create systemd service
echo ""
print_status "Creating systemd service..."

cat > /etc/systemd/system/amesie-ai.service << 'EOF'
[Unit]
Description=Amesie AI Backend with Full OpenHermes Model
After=network.target redis-server.service
Wants=redis-server.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/amesie_ai_backend
Environment="PATH=/opt/amesie_ai_backend/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="HF_HOME=/opt/models"
Environment="TRANSFORMERS_CACHE=/opt/models"
Environment="OMP_NUM_THREADS=8"
Environment="MKL_NUM_THREADS=8"
Environment="TORCH_NUM_THREADS=8"

ExecStart=/opt/amesie_ai_backend/venv/bin/python /opt/amesie_ai_backend/main_openhermes_cpu.py

Restart=always
RestartSec=10
StartLimitInterval=200
StartLimitBurst=5

LimitNOFILE=65536
LimitNPROC=32768

MemoryMax=28G
MemorySwapMax=16G
CPUQuota=700%

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable amesie-ai

print_status "Service created"

# Configure nginx
echo ""
print_status "Configuring nginx..."

cat > /etc/nginx/sites-available/amesie-ai << 'EOF'
upstream amesie_backend {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name 147.93.102.165;
    
    proxy_read_timeout 300s;
    proxy_connect_timeout 75s;
    proxy_send_timeout 300s;
    
    client_max_body_size 100M;
    client_body_buffer_size 10M;
    
    location / {
        proxy_pass http://amesie_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_request_buffering off;
    }
    
    location /ws {
        proxy_pass http://amesie_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
    
    location /health {
        proxy_pass http://amesie_backend/health;
        proxy_cache_valid 200 5s;
        access_log off;
    }
}
EOF

ln -sf /etc/nginx/sites-available/amesie-ai /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

nginx -t && systemctl restart nginx

print_status "Nginx configured"

# Start the service
echo ""
print_status "Starting Amesie AI service..."

systemctl start amesie-ai

# Wait for service to initialize
echo "Waiting for model to load (this may take 2-3 minutes)..."
sleep 10

# Check status
if systemctl is-active --quiet amesie-ai; then
    print_status "Service started successfully!"
    echo ""
    echo "Recent logs:"
    journalctl -u amesie-ai -n 20 --no-pager
else
    print_error "Service failed to start. Checking logs..."
    journalctl -u amesie-ai -n 50 --no-pager
fi

# Test deployment
echo ""
print_status "Testing deployment..."
sleep 5

# Test health endpoint
if curl -s --max-time 10 http://localhost:8000/health | grep -q "healthy"; then
    print_status "API is responding!"
    echo ""
    curl -s http://localhost:8000/health | python3 -m json.tool || curl -s http://localhost:8000/health
else
    print_warning "API not responding yet. The model may still be loading."
    echo "Check status with: journalctl -u amesie-ai -f"
fi

# Print summary
echo ""
echo "=========================================="
echo -e "${GREEN} Deployment Complete!${NC}"
echo "=========================================="
echo ""
echo "Your Amesie AI Backend is now:"
echo "  Internal: http://localhost:8000"
echo "  External: http://147.93.102.165"
echo ""
echo "API Endpoints:"
echo "  Health: http://147.93.102.165/health"
echo "  Chat: http://147.93.102.165/api/v1/chat/completion"
echo "  Roles: http://147.93.102.165/api/v1/chat/roles"
echo "  Metrics: http://147.93.102.165/api/v1/metrics/dashboard"
echo ""
echo "Commands:"
echo "  View logs: journalctl -u amesie-ai -f"
echo "  Restart: systemctl restart amesie-ai"
echo "  Status: systemctl status amesie-ai"
echo "  Monitor: htop"
echo ""
print_warning "Note: First request will be slower as model warms up"
print_warning "Expected response time: 5-15 seconds on CPU"