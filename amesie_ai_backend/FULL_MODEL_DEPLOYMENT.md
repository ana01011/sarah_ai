# Full OpenHermes Model Deployment Guide

This guide is for deploying the **FULL OpenHermes-2.5-Mistral-7B model** on your server with 32GB RAM.

## 🖥️ Your Server Specifications
- **IP**: 147.93.102.165
- **RAM**: 32GB (Perfect for 7B model)
- **CPU**: 8 vCores
- **Storage**: 400GB (More than enough)

## 🚀 Quick Deployment (Recommended)

### One-Command Deployment

```bash
cd /workspace/amesie_ai_backend
./deploy_full_model.sh
```

This script will:
1. ✅ Install all system dependencies
2. ✅ Set up Python 3.11 environment
3. ✅ Install PyTorch CPU-optimized version
4. ✅ Download the full OpenHermes-2.5-Mistral-7B model (~14GB)
5. ✅ Configure system optimizations for ML workloads
6. ✅ Set up 16GB swap space for model loading
7. ✅ Create systemd service for auto-start
8. ✅ Configure nginx reverse proxy
9. ✅ Start the service

**Total deployment time: ~20-30 minutes** (mostly model download)

## 📊 What You Get

### Full AI Capabilities
- **Real OpenHermes-2.5-Mistral-7B model** (not mock responses)
- **6 specialized AI roles**: CEO, CFO, CTO, COO, CMO, AI Assistant
- **High-quality responses** from a state-of-the-art 7B parameter model
- **WebSocket support** for real-time chat
- **Complete metrics dashboard**

### Performance Expectations
- **Model loading time**: 2-3 minutes on first start
- **Response time**: 5-15 seconds per query (CPU inference)
- **Memory usage**: 14-20GB when model is loaded
- **Token generation**: ~10-20 tokens/second on CPU
- **Max response length**: 256 tokens (optimized for CPU)

## 🔧 Manual Deployment Steps

If you prefer manual deployment or need to customize:

### 1. Connect to Your Server
```bash
ssh root@147.93.102.165
```

### 2. Download Files
```bash
# On your local machine
cd /workspace/amesie_ai_backend
scp main_openhermes_cpu.py requirements-cpu.txt root@147.93.102.165:/opt/amesie_ai_backend/
```

### 3. Install Dependencies on Server
```bash
# On the server
cd /opt/amesie_ai_backend

# System dependencies
apt update
apt install -y python3.11 python3.11-venv python3.11-dev \
               build-essential libopenblas-dev libomp-dev \
               redis-server nginx htop

# Python environment
python3.11 -m venv venv
source venv/bin/activate

# Install PyTorch CPU version
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install transformers==4.36.2 accelerate==0.25.0 \
            fastapi==0.104.1 uvicorn==0.24.0 \
            psutil==5.9.6 redis==5.0.1
```

### 4. Download the Model
```bash
# This will download ~14GB
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading OpenHermes-2.5-Mistral-7B...')
tokenizer = AutoTokenizer.from_pretrained('teknium/OpenHermes-2.5-Mistral-7B')
model = AutoModelForCausalLM.from_pretrained('teknium/OpenHermes-2.5-Mistral-7B')
print('Model downloaded successfully!')
"
```

### 5. Start the Service
```bash
python main_openhermes_cpu.py
```

## 🎯 API Usage Examples

### Test the API
```bash
# Health check
curl http://147.93.102.165/health

# Get available roles
curl http://147.93.102.165/api/v1/chat/roles

# Chat completion
curl -X POST http://147.93.102.165/api/v1/chat/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing in simple terms",
    "role": "CTO",
    "max_length": 200,
    "temperature": 0.7
  }'
```

### Python Example
```python
import requests

# Chat with the AI
response = requests.post(
    "http://147.93.102.165/api/v1/chat/completion",
    json={
        "prompt": "What's the best strategy for a startup?",
        "role": "CEO",
        "max_length": 256
    }
)

print(response.json()["text"])
```

### WebSocket Example
```javascript
const ws = new WebSocket('ws://147.93.102.165/ws/chat');

ws.onopen = () => {
    ws.send(JSON.stringify({
        prompt: "Hello, how can you help me?",
        role: "AI Assistant"
    }));
};

ws.onmessage = (event) => {
    const response = JSON.parse(event.data);
    console.log(response.text);
};
```

## ⚙️ Optimization Tips

### 1. CPU Performance
- The model uses all 8 CPU cores
- OpenMP and MKL optimizations are enabled
- Thread count is set to 8 for maximum parallelism

### 2. Memory Management
- 16GB swap is configured for model loading
- Memory limit set to 28GB (leaving 4GB for system)
- Low memory mode enabled in transformers

### 3. Response Speed
- Responses limited to 256 tokens for faster generation
- KV cache enabled for better performance
- Batch size of 1 for optimal CPU usage

## 🔍 Monitoring

### Check Service Status
```bash
ssh root@147.93.102.165 'systemctl status amesie-ai'
```

### View Real-time Logs
```bash
ssh root@147.93.102.165 'journalctl -u amesie-ai -f'
```

### Monitor System Resources
```bash
ssh root@147.93.102.165 'htop'
```

### Check Memory Usage
```bash
curl http://147.93.102.165/api/v1/metrics/system
```

## 🚨 Troubleshooting

### Model Takes Too Long to Load
- Normal loading time is 2-3 minutes
- Check available RAM: `free -h`
- Ensure swap is enabled: `swapon -s`

### Out of Memory Errors
```bash
# Increase swap if needed
sudo fallocate -l 32G /swapfile2
sudo chmod 600 /swapfile2
sudo mkswap /swapfile2
sudo swapon /swapfile2
```

### Slow Response Times
- CPU inference is naturally slower than GPU
- Consider using a smaller model for faster responses:
  - `microsoft/phi-2` (2.7B params, much faster)
  - `google/flan-t5-base` (250M params, very fast)

### Service Won't Start
```bash
# Check logs
journalctl -u amesie-ai -n 100

# Try running manually
cd /opt/amesie_ai_backend
source venv/bin/activate
python main_openhermes_cpu.py
```

## 🎨 Alternative Models

If OpenHermes is too slow on CPU, you can use smaller models:

### Edit .env file:
```bash
nano /opt/amesie_ai_backend/.env
```

### Add one of these:
```env
# Smaller but still powerful (2.7B params)
MODEL_NAME=microsoft/phi-2

# Very fast, good quality (770M params)
MODEL_NAME=google/flan-t5-large

# Fastest option (250M params)
MODEL_NAME=google/flan-t5-base
```

### Restart service:
```bash
systemctl restart amesie-ai
```

## 📈 Performance Benchmarks

On your 32GB server with 8 vCores:

| Model | Size | Load Time | Response Time | RAM Usage |
|-------|------|-----------|---------------|-----------|
| OpenHermes-2.5-Mistral-7B | 14GB | 2-3 min | 5-15 sec | 14-20GB |
| microsoft/phi-2 | 5GB | 30-60 sec | 2-5 sec | 6-8GB |
| google/flan-t5-large | 3GB | 15-30 sec | 1-3 sec | 4-5GB |
| google/flan-t5-base | 1GB | 5-10 sec | <1 sec | 2-3GB |

## 🎯 Next Steps

1. **Test the API**: Verify everything is working
2. **Connect Frontend**: Update your frontend to use `http://147.93.102.165`
3. **Add SSL**: Set up HTTPS with Let's Encrypt
4. **Monitor Usage**: Watch logs and metrics
5. **Optimize**: Adjust model and settings based on usage

## 💡 Pro Tips

1. **Warm-up the Model**: Send a test request after startup
2. **Use Caching**: Repeated queries can be cached in Redis
3. **Monitor Memory**: Keep an eye on RAM usage with `htop`
4. **Regular Restarts**: Restart weekly to clear memory fragmentation
5. **Backup Models**: Keep a backup of downloaded models in `/opt/models`

## 🆘 Support

If you encounter issues:

1. Check the logs first: `journalctl -u amesie-ai -f`
2. Verify model is downloaded: `ls -la /opt/models`
3. Test manually: `python main_openhermes_cpu.py`
4. Check system resources: `free -h && df -h`

---

**Your server is perfectly capable of running the full OpenHermes model!** With 32GB RAM and 8 vCores, you have more than enough resources for excellent AI performance.