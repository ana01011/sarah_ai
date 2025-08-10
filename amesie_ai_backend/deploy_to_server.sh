#!/bin/bash

# Amesie AI Backend Deployment Script for Remote Server
# Target Server: 147.93.102.165

set -e  # Exit on error

echo "========================================"
echo " Amesie AI Backend Deployment Script"
echo " Target Server: 147.93.102.165"
echo "========================================"

# Configuration
SERVER_IP="147.93.102.165"
SERVER_USER="root"  # Change this to your server username
REMOTE_DIR="/opt/amesie_ai_backend"
LOCAL_DIR="/workspace/amesie_ai_backend"
SERVICE_NAME="amesie-ai"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if we can connect to the server
echo ""
print_status "Testing connection to server..."
if ssh -o ConnectTimeout=5 ${SERVER_USER}@${SERVER_IP} "echo 'Connection successful'" > /dev/null 2>&1; then
    print_status "Connection to server successful"
else
    print_error "Cannot connect to server. Please check:"
    echo "  1. Server IP is correct: ${SERVER_IP}"
    echo "  2. Username is correct: ${SERVER_USER}"
    echo "  3. SSH key is configured or password authentication is enabled"
    echo ""
    echo "To set up SSH key (recommended):"
    echo "  ssh-copy-id ${SERVER_USER}@${SERVER_IP}"
    exit 1
fi

# Create deployment package
echo ""
print_status "Creating deployment package..."
cd ${LOCAL_DIR}
tar -czf /tmp/amesie_backend.tar.gz \
    main_openhermes.py \
    requirements-light.txt \
    requirements-openhermes.txt \
    .env.example \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='server.log' \
    --exclude='nohup.out'

print_status "Deployment package created"

# Copy files to server
echo ""
print_status "Copying files to server..."
ssh ${SERVER_USER}@${SERVER_IP} "mkdir -p ${REMOTE_DIR}"
scp /tmp/amesie_backend.tar.gz ${SERVER_USER}@${SERVER_IP}:/tmp/
ssh ${SERVER_USER}@${SERVER_IP} "cd ${REMOTE_DIR} && tar -xzf /tmp/amesie_backend.tar.gz && rm /tmp/amesie_backend.tar.gz"
print_status "Files copied to server"

# Create setup script on server
echo ""
print_status "Setting up server environment..."

ssh ${SERVER_USER}@${SERVER_IP} << 'ENDSSH'
cd /opt/amesie_ai_backend

# Update system packages
echo "Updating system packages..."
apt-get update -qq

# Install Python 3.11 if not present
if ! command -v python3.11 &> /dev/null; then
    echo "Installing Python 3.11..."
    apt-get install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq
    apt-get install -y python3.11 python3.11-venv python3.11-dev
fi

# Install system dependencies
echo "Installing system dependencies..."
apt-get install -y build-essential nginx redis-server supervisor

# Create Python virtual environment
echo "Creating Python virtual environment..."
python3.11 -m venv venv

# Activate virtual environment and install dependencies
echo "Installing Python dependencies (lightweight version)..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-light.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file from template"
fi

# Start Redis
echo "Starting Redis..."
systemctl enable redis-server
systemctl start redis-server

echo "Server setup complete!"
ENDSSH

print_status "Server environment setup complete"

# Create systemd service
echo ""
print_status "Creating systemd service..."

ssh ${SERVER_USER}@${SERVER_IP} << ENDSSH
cat > /etc/systemd/system/${SERVICE_NAME}.service << 'EOF'
[Unit]
Description=Amesie AI Backend with OpenHermes
After=network.target redis-server.service

[Service]
Type=simple
User=root
WorkingDirectory=${REMOTE_DIR}
Environment="PATH=${REMOTE_DIR}/venv/bin"
ExecStart=${REMOTE_DIR}/venv/bin/python ${REMOTE_DIR}/main_openhermes.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and start service
systemctl daemon-reload
systemctl enable ${SERVICE_NAME}
systemctl restart ${SERVICE_NAME}

# Check service status
sleep 3
if systemctl is-active --quiet ${SERVICE_NAME}; then
    echo "Service started successfully"
else
    echo "Service failed to start. Check logs with: journalctl -u ${SERVICE_NAME} -n 50"
fi
ENDSSH

print_status "Systemd service created and started"

# Configure nginx reverse proxy
echo ""
print_status "Configuring nginx..."

ssh ${SERVER_USER}@${SERVER_IP} << 'ENDSSH'
cat > /etc/nginx/sites-available/amesie-ai << 'EOF'
server {
    listen 80;
    server_name 147.93.102.165;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
EOF

# Enable site and restart nginx
ln -sf /etc/nginx/sites-available/amesie-ai /etc/nginx/sites-enabled/
nginx -t && systemctl restart nginx
ENDSSH

print_status "Nginx configured"

# Configure firewall
echo ""
print_status "Configuring firewall..."

ssh ${SERVER_USER}@${SERVER_IP} << 'ENDSSH'
# Install ufw if not present
if ! command -v ufw &> /dev/null; then
    apt-get install -y ufw
fi

# Configure firewall rules
ufw allow 22/tcp   # SSH
ufw allow 80/tcp   # HTTP
ufw allow 443/tcp  # HTTPS (for future SSL)
ufw allow 8000/tcp # Direct API access (optional)

# Enable firewall (non-interactive)
echo "y" | ufw enable
ENDSSH

print_status "Firewall configured"

# Test the deployment
echo ""
print_status "Testing deployment..."

# Wait for service to be ready
sleep 5

# Test health endpoint
if curl -s http://${SERVER_IP}/health | grep -q "healthy"; then
    print_status "API is responding correctly!"
else
    print_warning "API might not be ready yet. Check manually:"
    echo "  curl http://${SERVER_IP}/health"
fi

# Print summary
echo ""
echo "========================================"
echo -e "${GREEN} Deployment Complete!${NC}"
echo "========================================"
echo ""
echo "Your Amesie AI Backend is now running at:"
echo "  Main URL: http://${SERVER_IP}"
echo "  Direct API: http://${SERVER_IP}:8000"
echo ""
echo "Available endpoints:"
echo "  Health Check: http://${SERVER_IP}/health"
echo "  Chat API: http://${SERVER_IP}/api/v1/chat/completion"
echo "  Roles: http://${SERVER_IP}/api/v1/chat/roles"
echo "  Metrics: http://${SERVER_IP}/api/v1/metrics/dashboard"
echo ""
echo "Useful commands on server:"
echo "  Check service status: systemctl status ${SERVICE_NAME}"
echo "  View logs: journalctl -u ${SERVICE_NAME} -f"
echo "  Restart service: systemctl restart ${SERVICE_NAME}"
echo "  Stop service: systemctl stop ${SERVICE_NAME}"
echo ""
echo "To enable OpenHermes model (requires GPU/high RAM):"
echo "  1. SSH to server: ssh ${SERVER_USER}@${SERVER_IP}"
echo "  2. cd ${REMOTE_DIR}"
echo "  3. source venv/bin/activate"
echo "  4. pip install -r requirements-openhermes.txt"
echo "  5. systemctl restart ${SERVICE_NAME}"
echo ""
print_warning "Note: Currently running in mock mode (no ML model loaded)"
print_warning "The OpenHermes model requires significant resources (GPU recommended)"