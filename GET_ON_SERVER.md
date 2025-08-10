# How to Get the OpenHermes Code on Your Server

## Current Situation
- **Branch with OpenHermes files**: `cursor/analyse-branch-and-repo-linkage-0afe`
- **Your server**: 147.93.102.165
- **Files you need**: All the OpenHermes deployment files are ready in this branch

## Step-by-Step Instructions

### 1️⃣ Connect to Your Server
```bash
ssh root@147.93.102.165
```

### 2️⃣ Clone or Update Your Repository

#### Option A: If you DON'T have the repo on your server yet:
```bash
# Go to /opt directory
cd /opt

# Clone the repository
git clone https://github.com/ana01011/sarah_ai.git amesie_ai_backend

# Go into the directory
cd amesie_ai_backend

# Switch to the branch with OpenHermes files
git checkout cursor/analyse-branch-and-repo-linkage-0afe
```

#### Option B: If you ALREADY have the repo on your server:
```bash
# Go to your repo directory (wherever you cloned it)
cd /path/to/your/sarah_ai  # or cd /opt/amesie_ai_backend

# Fetch latest changes
git fetch origin

# Switch to the OpenHermes branch
git checkout cursor/analyse-branch-and-repo-linkage-0afe

# Pull latest changes
git pull origin cursor/analyse-branch-and-repo-linkage-0afe
```

### 3️⃣ Go to the Backend Directory
```bash
cd amesie_ai_backend
```

### 4️⃣ Verify You Have the Right Files
```bash
# You should see these files:
ls -la | grep openhermes
```

You should see:
- `main_openhermes_cpu.py` - The CPU-optimized server
- `deploy_full_model.sh` - The automated deployment script
- `requirements-cpu.txt` - CPU-optimized dependencies
- `FULL_MODEL_DEPLOYMENT.md` - Documentation

### 5️⃣ Run the Deployment
```bash
# Make the script executable
chmod +x deploy_full_model.sh

# Run the deployment (this will set up everything!)
./deploy_full_model.sh
```

## 🚀 Quick One-Liner

If you want to do everything in one go (fresh install):
```bash
ssh root@147.93.102.165 'cd /opt && git clone https://github.com/ana01011/sarah_ai.git amesie_ai_backend && cd amesie_ai_backend && git checkout cursor/analyse-branch-and-repo-linkage-0afe && cd amesie_ai_backend && chmod +x deploy_full_model.sh && ./deploy_full_model.sh'
```

## 📝 What the Deployment Script Will Do

1. Install Python 3.11 and all system dependencies
2. Create Python virtual environment
3. Install PyTorch CPU version
4. Download the OpenHermes-2.5-Mistral-7B model (14GB)
5. Set up systemd service for auto-start
6. Configure nginx reverse proxy
7. Start the AI service

## ✅ After Deployment

Test if it's working:
```bash
# Check health
curl http://147.93.102.165/health

# Test chat
curl -X POST http://147.93.102.165/api/v1/chat/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, are you working?", "role": "AI Assistant"}'
```

## 🔍 If You're Not Sure Which Branch You're On

On your server:
```bash
# Check current branch
git branch

# List all branches
git branch -a

# The branch you need is:
# cursor/analyse-branch-and-repo-linkage-0afe
```

## ⚠️ Important Notes

- **Branch name**: `cursor/analyse-branch-and-repo-linkage-0afe`
- **This branch has**: All the OpenHermes deployment files
- **Model download**: Will take 10-20 minutes (14GB)
- **Total time**: About 30 minutes for complete setup
- **RAM needed**: Your 32GB is perfect!

## 🆘 Troubleshooting

If the branch doesn't exist:
```bash
# Make sure you fetch from origin
git fetch origin

# List remote branches
git branch -r | grep cursor

# Checkout the branch
git checkout -b cursor/analyse-branch-and-repo-linkage-0afe origin/cursor/analyse-branch-and-repo-linkage-0afe
```

That's it! The branch `cursor/analyse-branch-and-repo-linkage-0afe` has everything you need!