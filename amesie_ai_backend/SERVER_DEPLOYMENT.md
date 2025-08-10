# Server Deployment Guide for Amesie AI Backend

This guide explains how to deploy the Amesie AI Backend with OpenHermes on your server at **147.93.102.165**.

## Overview

The backend can run in two modes:
1. **Mock Mode** (default): Lightweight, no ML model required, returns simulated responses
2. **OpenHermes Mode**: Full AI capabilities using OpenHermes-2.5 model (requires GPU/high RAM)

## Prerequisites

### Server Requirements
- **OS**: Ubuntu 20.04+ or Debian 11+
- **RAM**: Minimum 4GB (Mock mode), 16GB+ recommended (OpenHermes mode)
- **Storage**: 10GB minimum, 50GB+ for OpenHermes model
- **GPU**: Optional but highly recommended for OpenHermes mode
- **Python**: 3.11+
- **Network**: Open ports 80, 443, 8000

## Quick Deployment (Automated)

### Option 1: Run from your local machine

```bash
# Make the script executable
chmod +x deploy_to_server.sh

# Edit the script to set your server username (default is root)
nano deploy_to_server.sh
# Change SERVER_USER="root" to your username if needed

# Run the deployment
./deploy_to_server.sh
```

## Manual Deployment (Step by Step)

### Step 1: Connect to Your Server

```bash
ssh root@147.93.102.165
```

### Step 2: Install System Dependencies

```bash
# Update package list
apt update

# Install Python 3.11
apt install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt update
apt install -y python3.11 python3.11-venv python3.11-dev

# Install other dependencies
apt install -y build-essential nginx redis-server git curl
```

### Step 3: Create Application Directory

```bash
mkdir -p /opt/amesie_ai_backend
cd /opt/amesie_ai_backend
```

### Step 4: Copy Application Files

From your local machine:
```bash
# Create a package
cd /workspace/amesie_ai_backend
tar -czf amesie_backend.tar.gz \
    main_openhermes.py \
    requirements-light.txt \
    requirements-openhermes.txt \
    .env.example

# Copy to server
scp amesie_backend.tar.gz root@147.93.102.165:/opt/amesie_ai_backend/

# Extract on server
ssh root@147.93.102.165 "cd /opt/amesie_ai_backend && tar -xzf amesie_backend.tar.gz"
```

### Step 5: Set Up Python Environment

On the server:
```bash
cd /opt/amesie_ai_backend

# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies (choose one):
# For lightweight deployment (Mock mode):
pip install -r requirements-light.txt

# For full OpenHermes deployment (requires GPU/high RAM):
# pip install -r requirements-openhermes.txt
```

### Step 6: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env

# Add/modify these settings:
# CORS_ORIGINS=http://your-frontend-domain.com,http://147.93.102.165
# MODEL_NAME=teknium/OpenHermes-2.5-Mistral-7B  # or use smaller model
```

### Step 7: Start Redis

```bash
systemctl enable redis-server
systemctl start redis-server

# Verify Redis is running
redis-cli ping
# Should return: PONG
```

### Step 8: Create Systemd Service

```bash
# Create service file
cat > /etc/systemd/system/amesie-ai.service << 'EOF'
[Unit]
Description=Amesie AI Backend with OpenHermes
After=network.target redis-server.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/amesie_ai_backend
Environment="PATH=/opt/amesie_ai_backend/venv/bin"
ExecStart=/opt/amesie_ai_backend/venv/bin/python /opt/amesie_ai_backend/main_openhermes.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
systemctl daemon-reload

# Enable and start service
systemctl enable amesie-ai
systemctl start amesie-ai

# Check status
systemctl status amesie-ai
```

### Step 9: Configure Nginx Reverse Proxy

```bash
# Create nginx configuration
cat > /etc/nginx/sites-available/amesie-ai << 'EOF'
server {
    listen 80;
    server_name 147.93.102.165;
    
    client_max_body_size 100M;
    
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
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
    
    location /ws {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 3600s;
    }
}
EOF

# Enable site
ln -sf /etc/nginx/sites-available/amesie-ai /etc/nginx/sites-enabled/

# Remove default site if exists
rm -f /etc/nginx/sites-enabled/default

# Test configuration
nginx -t

# Restart nginx
systemctl restart nginx
```

### Step 10: Configure Firewall

```bash
# Install ufw if not present
apt install -y ufw

# Configure rules
ufw allow 22/tcp   # SSH
ufw allow 80/tcp   # HTTP
ufw allow 443/tcp  # HTTPS
ufw allow 8000/tcp # Direct API (optional)

# Enable firewall
ufw --force enable

# Check status
ufw status
```

## Testing the Deployment

### From the server:
```bash
# Test local connection
curl http://localhost:8000/health

# Test through nginx
curl http://147.93.102.165/health
```

### From your local machine:
```bash
# Test health endpoint
curl http://147.93.102.165/health

# Test chat completion
curl -X POST http://147.93.102.165/api/v1/chat/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, how are you?",
    "role": "AI Assistant",
    "max_length": 200
  }'

# Test available roles
curl http://147.93.102.165/api/v1/chat/roles
```

## API Endpoints

Once deployed, your API will be available at:

- **Base URL**: `http://147.93.102.165`
- **Health Check**: `GET /health`
- **Chat Completion**: `POST /api/v1/chat/completion`
- **Available Roles**: `GET /api/v1/chat/roles`
- **System Metrics**: `GET /api/v1/metrics/system`
- **Dashboard Metrics**: `GET /api/v1/metrics/dashboard`
- **WebSocket Chat**: `ws://147.93.102.165/ws/chat`

## Monitoring and Maintenance

### View Logs
```bash
# Service logs
journalctl -u amesie-ai -f

# Nginx logs
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

### Restart Service
```bash
systemctl restart amesie-ai
```

### Update Code
```bash
cd /opt/amesie_ai_backend
# Copy new files
systemctl restart amesie-ai
```

## Enabling OpenHermes Model

By default, the system runs in mock mode. To enable the full OpenHermes model:

### Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 30GB+ free space for model

### Installation
```bash
cd /opt/amesie_ai_backend
source venv/bin/activate

# Install ML dependencies
pip install -r requirements-openhermes.txt

# For GPU support (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Restart service
systemctl restart amesie-ai
```

### Using Smaller Models (for limited resources)

Edit the `.env` file and use a smaller model:
```bash
# For limited resources (4-8GB RAM)
MODEL_NAME=microsoft/DialoGPT-medium

# Or
MODEL_NAME=google/flan-t5-base
```

## Troubleshooting

### Service Won't Start
```bash
# Check logs
journalctl -u amesie-ai -n 100

# Common issues:
# - Port 8000 already in use: Change port in main_openhermes.py
# - Python dependencies missing: Reinstall requirements
# - Redis not running: systemctl start redis-server
```

### High Memory Usage
- Switch to mock mode by not installing transformers
- Use a smaller model
- Add swap space:
```bash
fallocate -l 8G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

### API Not Responding
```bash
# Check if service is running
systemctl status amesie-ai

# Check nginx
systemctl status nginx
nginx -t

# Check firewall
ufw status
```

## Security Considerations

1. **Change default passwords** in `.env`
2. **Enable HTTPS** with Let's Encrypt:
   ```bash
   apt install certbot python3-certbot-nginx
   certbot --nginx -d your-domain.com
   ```
3. **Restrict CORS origins** in `.env`
4. **Use non-root user** for production
5. **Regular updates**:
   ```bash
   apt update && apt upgrade
   ```

## Performance Optimization

### For Mock Mode
- Already optimized for low resource usage
- Can handle 100+ requests/second

### For OpenHermes Mode
- Use GPU acceleration if available
- Consider model quantization (8-bit or 4-bit)
- Implement caching for common queries
- Use load balancer for multiple instances

## Support

For issues or questions:
1. Check logs: `journalctl -u amesie-ai -f`
2. Test endpoints manually
3. Verify all dependencies are installed
4. Check system resources: `htop` or `free -h`

---

**Note**: This deployment is configured for IP **147.93.102.165**. Update the IP address in configuration files if deploying to a different server.