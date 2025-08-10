# Amesie AI Backend - Simple Deployment Guide

This guide shows how to deploy and test the Amesie AI backend without Docker.

## Prerequisites

- Linux/Ubuntu system
- Python 3.11+ installed
- Internet connection for downloading dependencies

## Step-by-Step Deployment

### 1. Navigate to Backend Directory
```bash
cd /workspace/amesie_ai_backend
```

### 2. Install System Dependencies
```bash
# Install Python virtual environment support
sudo apt update
sudo apt install -y python3-venv

# Install Redis (for caching)
sudo apt install -y redis-server
```

### 3. Set Up Python Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 4. Install Python Dependencies

For simplified deployment (without heavy ML models):
```bash
# Install simplified requirements
pip install -r requirements-simple.txt

# Install testing dependencies
pip install requests
```

For full deployment (with ML models - requires more resources):
```bash
# Install full requirements (may take longer)
pip install -r requirements.txt
```

### 5. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env file to add your Mistral API key (optional)
# If you have a Mistral API key, add it to the .env file:
# MISTRAL_API_KEY=your_api_key_here
```

### 6. Start Redis
```bash
# Start Redis in background
redis-server --daemonize yes

# Verify Redis is running
redis-cli ping
# Should return: PONG
```

### 7. Start the Backend Server

For testing (simplified version):
```bash
# Run the simplified server
python main_simple.py
```

For production (full version):
```bash
# Run the full server
python app/main.py
```

To run in background:
```bash
# Start server in background with logging
nohup python main_simple.py > server.log 2>&1 &

# Check if server is running
curl http://localhost:8000/health
```

## Testing the Deployment

### Quick Test
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test available roles
curl http://localhost:8000/api/v1/chat/roles

# Test chat completion
curl -X POST http://localhost:8000/api/v1/chat/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how can you help me?", "role": "AI Assistant"}'
```

### Comprehensive Test
```bash
# Run the test suite
python test_api.py
```

## API Endpoints

- **Health Check**: `GET http://localhost:8000/health`
- **Available Roles**: `GET http://localhost:8000/api/v1/chat/roles`
- **Chat Completion**: `POST http://localhost:8000/api/v1/chat/completion`
- **Performance Metrics**: `GET http://localhost:8000/api/v1/metrics/performance`
- **System Metrics**: `GET http://localhost:8000/api/v1/metrics/system`
- **Dashboard Metrics**: `GET http://localhost:8000/api/v1/metrics/dashboard`
- **WebSocket Chat**: `ws://localhost:8000/ws/chat`

## Features

### Available AI Roles
- **CEO**: Company strategy, vision, leadership
- **CFO**: Finance, accounting, projections
- **CTO**: Technology, engineering, software architecture
- **COO**: Operations, process optimization, logistics
- **CMO**: Marketing, branding, campaigns
- **AI Assistant**: General questions, cross-domain support

### Mistral API Integration
The backend can work with or without Mistral API:
- **With Mistral API**: Real AI-powered responses using Mistral models
- **Without Mistral API**: Mock responses for testing and development

To use Mistral API, add your API key to the `.env` file:
```
MISTRAL_API_KEY=your_mistral_api_key_here
```

## Stopping the Server

### If running in foreground:
Press `Ctrl+C` to stop the server

### If running in background:
```bash
# Find the process
ps aux | grep main_simple.py

# Kill the process (replace PID with actual process ID)
kill PID
```

## Troubleshooting

### Port Already in Use
If port 8000 is already in use:
```bash
# Find what's using port 8000
lsof -i :8000

# Kill the process or change the port in main_simple.py
```

### Module Not Found Errors
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements-simple.txt
```

### Redis Connection Error
```bash
# Check if Redis is running
redis-cli ping

# If not, start Redis
redis-server --daemonize yes
```

## Performance Notes

- The simplified version (`main_simple.py`) uses mock data for metrics and can use Mistral API for chat
- Response times with Mistral API: 0.5-5 seconds depending on prompt complexity
- Response times with mock responses: < 0.1 seconds
- The server can handle multiple concurrent requests

## Security Notes

- Default CORS is configured for localhost:3000 and localhost:5173
- Change the SECRET_KEY in .env for production use
- Consider using HTTPS in production
- Implement proper authentication for production deployment

## Next Steps

1. **Frontend Integration**: Connect the React frontend to this backend
2. **Database Setup**: Add PostgreSQL for persistent storage
3. **Production Deployment**: Use a production WSGI server like Gunicorn
4. **Monitoring**: Set up Prometheus and Grafana for production monitoring
5. **Scaling**: Consider containerization with Docker for easier scaling

---

For more information, see the main README.md file in the repository root.