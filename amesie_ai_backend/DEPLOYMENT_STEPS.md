# Step-by-Step Deployment Instructions

## Where You Are Now vs Where to Deploy

### Current Situation:
- **You have the code here**: `/workspace/amesie_ai_backend` (in this development environment)
- **You want to deploy to**: Your server at `147.93.102.165`
- **Target location on server**: `/opt/amesie_ai_backend`

## Method 1: Direct File Transfer (Simplest)

### Step 1: Connect to Your Server
Open a terminal on your local computer and connect:
```bash
ssh root@147.93.102.165
```

### Step 2: Create Directory on Server
Once connected to your server:
```bash
mkdir -p /opt/amesie_ai_backend
cd /opt/amesie_ai_backend
```

### Step 3: Download the Required Files Directly
Still on your server, create the files:

#### Create main_openhermes_cpu.py:
```bash
cat > /opt/amesie_ai_backend/main_openhermes_cpu.py << 'EOF'
# Copy the entire content of main_openhermes_cpu.py here
# (The file is too long to paste, use Method 2 or 3 instead)
EOF
```

## Method 2: Using GitHub (Recommended)

### Step 1: Push Code to GitHub
First, from this development environment:
```bash
cd /workspace
git add .
git commit -m "Add OpenHermes deployment files"
git push origin main
```

### Step 2: Clone on Your Server
On your server (after SSH):
```bash
# Install git if not already installed
apt update && apt install -y git

# Go to /opt directory
cd /opt

# Clone your repository
git clone https://github.com/ana01011/sarah_ai.git amesie_ai_backend

# Go to the backend directory
cd /opt/amesie_ai_backend/amesie_ai_backend
```

### Step 3: Run the Deployment Script
```bash
# Make script executable
chmod +x deploy_full_model.sh

# Run deployment
./deploy_full_model.sh
```

## Method 3: Direct Transfer via SCP (From Your Computer)

### Step 1: Download Files to Your Computer
If you have access to these files on your local computer:

### Step 2: Create a Deployment Package
On your local computer, create a folder with these files:
- main_openhermes_cpu.py
- requirements-cpu.txt
- deploy_full_model.sh
- .env.example

### Step 3: Transfer to Server
From your local computer:
```bash
# Create tar archive
tar -czf amesie_deployment.tar.gz main_openhermes_cpu.py requirements-cpu.txt deploy_full_model.sh .env.example

# Transfer to server
scp amesie_deployment.tar.gz root@147.93.102.165:/tmp/

# Connect to server
ssh root@147.93.102.165

# Extract files
mkdir -p /opt/amesie_ai_backend
cd /opt/amesie_ai_backend
tar -xzf /tmp/amesie_deployment.tar.gz
```

## Method 4: Quick Manual Setup (Copy-Paste Method)

### Step 1: Connect to Your Server
```bash
ssh root@147.93.102.165
```

### Step 2: Create Directory Structure
```bash
mkdir -p /opt/amesie_ai_backend
cd /opt/amesie_ai_backend
```

### Step 3: Install Basic Requirements
```bash
# Update system
apt update

# Install Python and essential tools
apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
apt install -y git curl wget nano build-essential
apt install -y redis-server nginx
```

### Step 4: Create Python Environment
```bash
cd /opt/amesie_ai_backend
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### Step 5: Install Python Packages
```bash
# Install PyTorch CPU version
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install transformers==4.36.2
pip install accelerate==0.25.0
pip install fastapi==0.104.1
pip install uvicorn[standard]==0.24.0
pip install psutil==5.9.6
pip install python-dotenv==1.0.0
pip install websockets==12.0
pip install httpx
pip install pydantic
```

### Step 6: Download the Model (This Takes Time!)
```bash
# Create models directory
mkdir -p /opt/models

# Set environment variables
export HF_HOME=/opt/models
export TRANSFORMERS_CACHE=/opt/models

# Download model using Python
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print('Starting download of OpenHermes-2.5-Mistral-7B...')
print('This will download about 14GB, please be patient...')

tokenizer = AutoTokenizer.from_pretrained('teknium/OpenHermes-2.5-Mistral-7B')
print('Tokenizer downloaded.')

model = AutoModelForCausalLM.from_pretrained(
    'teknium/OpenHermes-2.5-Mistral-7B',
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)
print('Model downloaded successfully!')
"
```

### Step 7: Create the Main Application File
```bash
# You'll need to create main_openhermes_cpu.py
# Either copy it from GitHub or create it manually
nano /opt/amesie_ai_backend/main_openhermes_cpu.py
# Then paste the content (it's too long to include here)
```

### Step 8: Create .env Configuration
```bash
cat > /opt/amesie_ai_backend/.env << 'EOF'
HF_HOME=/opt/models
TRANSFORMERS_CACHE=/opt/models
MODEL_NAME=teknium/OpenHermes-2.5-Mistral-7B
TORCH_NUM_THREADS=8
CORS_ORIGINS=*
EOF
```

### Step 9: Create Systemd Service
```bash
cat > /etc/systemd/system/amesie-ai.service << 'EOF'
[Unit]
Description=Amesie AI Backend with OpenHermes
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/amesie_ai_backend
Environment="PATH=/opt/amesie_ai_backend/venv/bin:/usr/bin"
Environment="HF_HOME=/opt/models"
Environment="TRANSFORMERS_CACHE=/opt/models"
ExecStart=/opt/amesie_ai_backend/venv/bin/python /opt/amesie_ai_backend/main_openhermes_cpu.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable amesie-ai
```

### Step 10: Configure Nginx
```bash
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
        proxy_read_timeout 300s;
    }
}
EOF

ln -sf /etc/nginx/sites-available/amesie-ai /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl restart nginx
```

### Step 11: Start the Service
```bash
systemctl start amesie-ai
systemctl status amesie-ai
```

## Verification Steps

After deployment, verify everything is working:

### 1. Check Service Status
```bash
systemctl status amesie-ai
```

### 2. Check Logs
```bash
journalctl -u amesie-ai -f
```

### 3. Test API
```bash
# From the server
curl http://localhost:8000/health

# From outside
curl http://147.93.102.165/health
```

## File Structure on Your Server

After deployment, your server should have:
```
/opt/
├── amesie_ai_backend/
│   ├── main_openhermes_cpu.py    # Main application
│   ├── requirements-cpu.txt       # Python dependencies
│   ├── .env                       # Configuration
│   ├── venv/                      # Python virtual environment
│   └── deploy_full_model.sh      # Deployment script
└── models/                        # Model cache (14GB+)
    └── hub/                       # Downloaded models
```

## Important Notes

1. **Model Download**: The OpenHermes model is about 14GB. First download will take 10-20 minutes depending on internet speed.

2. **Memory Requirements**: The model needs about 20GB RAM to run. Your 32GB server is perfect.

3. **First Start**: The first start will take 2-3 minutes as the model loads into memory.

4. **Storage Location**: 
   - Application: `/opt/amesie_ai_backend/`
   - Models: `/opt/models/`
   - Logs: `journalctl -u amesie-ai`

## Troubleshooting

If something goes wrong:

### Check Python Version
```bash
python3.11 --version
# Should show Python 3.11.x
```

### Check Memory
```bash
free -h
# Should show 32GB total
```

### Check Disk Space
```bash
df -h
# Need at least 30GB free
```

### Manual Test
```bash
cd /opt/amesie_ai_backend
source venv/bin/activate
python main_openhermes_cpu.py
# This will show any errors directly
```

## Summary

The easiest approach is:
1. Use GitHub to transfer the code (Method 2)
2. Or manually install everything step by step (Method 4)

The files should be saved in `/opt/amesie_ai_backend/` on your server.

The model will be downloaded to `/opt/models/` (about 14GB).

Your server at 147.93.102.165 with 32GB RAM is perfect for running this!