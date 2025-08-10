#!/bin/bash

# Full OpenHermes Model Deployment Script
# For server with 32GB RAM, 8 vCores, 400GB storage
# Target: 147.93.102.165

set -e

echo "=========================================="
echo " Full OpenHermes Model Deployment"
echo " Server: 147.93.102.165 (32GB RAM)"
echo "=========================================="

# Configuration
SERVER_IP="147.93.102.165"
SERVER_USER="root"  # Change if different
REMOTE_DIR="/opt/amesie_ai_backend"
LOCAL_DIR="/workspace/amesie_ai_backend"
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

# Test connection
echo ""
print_status "Testing connection to server..."
if ! ssh -o ConnectTimeout=5 ${SERVER_USER}@${SERVER_IP} "echo 'Connected'" > /dev/null 2>&1; then
    print_error "Cannot connect to server ${SERVER_IP}"
    echo "Please ensure:"
    echo "  1. SSH access is configured"
    echo "  2. Server is online"
    echo "  3. Username is correct: ${SERVER_USER}"
    exit 1
fi
print_status "Connection successful"

# Create deployment package
echo ""
print_status "Creating deployment package..."
cd ${LOCAL_DIR}

# Create package with CPU-optimized files
tar -czf /tmp/amesie_full.tar.gz \
    main_openhermes_cpu.py \
    requirements-cpu.txt \
    .env.example \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc'

print_status "Package created"

# Copy to server
echo ""
print_status "Copying files to server..."
scp /tmp/amesie_full.tar.gz ${SERVER_USER}@${SERVER_IP}:/tmp/
ssh ${SERVER_USER}@${SERVER_IP} << 'ENDSSH'
mkdir -p /opt/amesie_ai_backend
cd /opt/amesie_ai_backend
tar -xzf /tmp/amesie_full.tar.gz
rm /tmp/amesie_full.tar.gz
ENDSSH
print_status "Files copied"

# Setup server environment
echo ""
print_status "Setting up server environment..."

ssh ${SERVER_USER}@${SERVER_IP} << 'ENDSSH'
set -e

cd /opt/amesie_ai_backend

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
fi

# Optimize system for ML workloads
echo "Optimizing system settings..."
# Increase file descriptors
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# Optimize memory settings
sysctl -w vm.swappiness=10
sysctl -w vm.vfs_cache_pressure=50
echo "vm.swappiness=10" >> /etc/sysctl.conf
echo "vm.vfs_cache_pressure=50" >> /etc/sysctl.conf

# Setup Python environment
echo "Setting up Python environment..."
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip and install wheel
pip install --upgrade pip wheel setuptools

# Install PyTorch CPU version first (to avoid downloading CUDA version)
echo "Installing PyTorch CPU version..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
echo "Installing requirements..."
pip install -r requirements-cpu.txt

# Pre-download the model to avoid timeout during first run
echo "Pre-downloading OpenHermes model (this may take 10-20 minutes)..."
python3 << 'PYTHONSCRIPT'
import os
os.environ['HF_HOME'] = '/opt/models'
os.environ['TRANSFORMERS_CACHE'] = '/opt/models'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")

print("Downloading model (7B parameters, ~14GB)...")
print("This will take some time...")
model = AutoModelForCausalLM.from_pretrained(
    "teknium/OpenHermes-2.5-Mistral-7B",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

print("Model downloaded successfully!")
del model
del tokenizer
PYTHONSCRIPT

# Create .env file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "HF_HOME=/opt/models" >> .env
    echo "TRANSFORMERS_CACHE=/opt/models" >> .env
    echo "MODEL_NAME=teknium/OpenHermes-2.5-Mistral-7B" >> .env
    echo "TORCH_NUM_THREADS=8" >> .env
fi

# Start Redis
systemctl enable redis-server
systemctl start redis-server

echo "Environment setup complete!"
ENDSSH

print_status "Server environment configured"

# Create optimized systemd service
echo ""
print_status "Creating systemd service..."

ssh ${SERVER_USER}@${SERVER_IP} << 'ENDSSH'
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

# Use the CPU-optimized script
ExecStart=/opt/amesie_ai_backend/venv/bin/python /opt/amesie_ai_backend/main_openhermes_cpu.py

# Restart policy
Restart=always
RestartSec=10
StartLimitInterval=200
StartLimitBurst=5

# Resource limits
LimitNOFILE=65536
LimitNPROC=32768

# Memory management
MemoryMax=28G
MemorySwapMax=16G

# CPU management
CPUQuota=700%

# Logging
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable amesie-ai
ENDSSH

print_status "Service created"

# Configure nginx with optimizations
echo ""
print_status "Configuring nginx..."

ssh ${SERVER_USER}@${SERVER_IP} << 'ENDSSH'
cat > /etc/nginx/sites-available/amesie-ai << 'EOF'
upstream amesie_backend {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name 147.93.102.165;
    
    # Increase timeouts for model inference
    proxy_read_timeout 300s;
    proxy_connect_timeout 75s;
    proxy_send_timeout 300s;
    
    # Increase buffer sizes
    client_max_body_size 100M;
    client_body_buffer_size 10M;
    
    # Main API
    location / {
        proxy_pass http://amesie_backend;
        proxy_http_version 1.1;
        
        # Keep-alive headers
        proxy_set_header Connection "";
        
        # Standard headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Buffering settings
        proxy_buffering off;
        proxy_request_buffering off;
    }
    
    # WebSocket endpoint
    location /ws {
        proxy_pass http://amesie_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
    
    # Health check endpoint (cached)
    location /health {
        proxy_pass http://amesie_backend/health;
        proxy_cache_valid 200 5s;
        access_log off;
    }
}
EOF

ln -sf /etc/nginx/sites-available/amesie-ai /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Optimize nginx
cat > /etc/nginx/conf.d/optimization.conf << 'EOF'
worker_processes auto;
worker_rlimit_nofile 65536;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

http {
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    keepalive_requests 100;
    
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript application/json application/javascript application/xml+rss application/rss+xml application/atom+xml image/svg+xml text/x-js text/x-cross-domain-policy application/x-font-ttf application/x-font-opentype application/vnd.ms-fontobject image/x-icon;
}
EOF

nginx -t && systemctl restart nginx
ENDSSH

print_status "Nginx configured"

# Start the service
echo ""
print_status "Starting Amesie AI service..."

ssh ${SERVER_USER}@${SERVER_IP} << 'ENDSSH'
systemctl start amesie-ai

# Wait for service to initialize
echo "Waiting for model to load (this may take 2-3 minutes)..."
sleep 10

# Check status
if systemctl is-active --quiet amesie-ai; then
    echo "Service started successfully!"
    journalctl -u amesie-ai -n 20 --no-pager
else
    echo "Service failed to start. Checking logs..."
    journalctl -u amesie-ai -n 50 --no-pager
fi
ENDSSH

# Test deployment
echo ""
print_status "Testing deployment..."
sleep 5

# Test health endpoint
if curl -s --max-time 10 http://${SERVER_IP}/health | grep -q "healthy"; then
    print_status "API is responding!"
    echo ""
    curl -s http://${SERVER_IP}/health | python3 -m json.tool
else
    print_warning "API not responding yet. The model may still be loading."
    echo "Check status with: ssh ${SERVER_USER}@${SERVER_IP} 'journalctl -u amesie-ai -f'"
fi

# Print summary
echo ""
echo "=========================================="
echo -e "${GREEN} Full Model Deployment Complete!${NC}"
echo "=========================================="
echo ""
echo "Server Details:"
echo "  IP: ${SERVER_IP}"
echo "  RAM: 32GB"
echo "  Model: OpenHermes-2.5-Mistral-7B"
echo "  Mode: Full model with CPU optimizations"
echo ""
echo "API Endpoints:"
echo "  Health: http://${SERVER_IP}/health"
echo "  Chat: http://${SERVER_IP}/api/v1/chat/completion"
echo "  Roles: http://${SERVER_IP}/api/v1/chat/roles"
echo "  Metrics: http://${SERVER_IP}/api/v1/metrics/dashboard"
echo "  WebSocket: ws://${SERVER_IP}/ws/chat"
echo ""
echo "Management Commands:"
echo "  View logs: ssh ${SERVER_USER}@${SERVER_IP} 'journalctl -u amesie-ai -f'"
echo "  Restart: ssh ${SERVER_USER}@${SERVER_IP} 'systemctl restart amesie-ai'"
echo "  Status: ssh ${SERVER_USER}@${SERVER_IP} 'systemctl status amesie-ai'"
echo "  Monitor: ssh ${SERVER_USER}@${SERVER_IP} 'htop'"
echo ""
echo "Performance Notes:"
echo "  - First request will be slower (model warming up)"
echo "  - Expect 5-15 seconds per response on CPU"
echo "  - Model uses ~14-20GB RAM when loaded"
echo "  - Responses limited to 256 tokens for performance"
echo ""
print_warning "The model may take 2-3 minutes to fully load on first start"