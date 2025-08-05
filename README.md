# Amesie AI - Production-Ready AI Platform

A comprehensive AI platform featuring a sophisticated React dashboard frontend and a production-ready FastAPI backend powered by Mistral 7B quantized model.

## 🚀 Features

### Frontend Dashboard
- **Real-time AI Chat**: Advanced chat interface with streaming responses
- **Live Metrics**: Real-time system performance monitoring
- **Neural Network Visualization**: Interactive model architecture display
- **Performance Charts**: Dynamic charts and analytics
- **Processing Pipeline**: Real-time pipeline monitoring
- **System Status**: Comprehensive system health monitoring
- **Theme System**: Multiple beautiful themes with smooth transitions
- **Responsive Design**: Works perfectly on all devices

### Backend API
- **Mistral 7B Quantized Model**: High-performance language model with 4-bit quantization
- **Real-time Chat API**: WebSocket-based streaming chat
- **Comprehensive Metrics**: Prometheus integration with Grafana dashboards
- **Production Monitoring**: Sentry error tracking, structured logging
- **Security**: JWT authentication, rate limiting, input validation
- **Scalable Architecture**: Redis caching, PostgreSQL database
- **Docker Support**: Complete containerization with Docker Compose

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React         │    │   FastAPI       │    │   Mistral 7B    │
│   Dashboard     │◄──►│   Backend       │◄──►│   Quantized     │
│   Frontend      │    │   (Python)      │    │   Model         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Redis Cache   │    │   PostgreSQL    │    │   Prometheus    │
│   & Sessions    │    │   Database      │    │   & Grafana     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.11+
- Docker and Docker Compose (optional)
- GPU with CUDA support (recommended for optimal performance)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd amesie-ai
```

### 2. Start the Backend
```bash
cd amesie_ai_backend

# Option 1: Using Docker (recommended)
docker-compose up -d

# Option 2: Manual setup
cp .env.example .env
# Edit .env with your configuration
pip install -r requirements.txt
./start.sh
```

### 3. Start the Frontend
```bash
# In the root directory
npm install
npm run dev
```

### 4. Access the Application
- **Frontend Dashboard**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus Metrics**: http://localhost:9090

## 📊 Dashboard Features

### Real-time AI Chat
- Streaming responses from Mistral 7B model
- Voice input support
- File upload capabilities
- Message reactions and suggestions
- Export and share functionality

### Live Metrics Dashboard
- **System Metrics**: CPU, memory, disk, GPU utilization
- **AI Performance**: Model accuracy, throughput, latency
- **Neural Network Visualization**: Interactive model architecture
- **Processing Pipeline**: Real-time pipeline monitoring
- **Performance Charts**: Dynamic analytics and trends

### Advanced Features
- **Multi-theme Support**: Beautiful themes with smooth transitions
- **Real-time Updates**: WebSocket-based live data
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Export Capabilities**: Download reports and data
- **System Monitoring**: Comprehensive health checks

## 🔧 Backend API

### Chat Endpoints
- `POST /api/v1/chat/completion` - Text completion
- `POST /api/v1/chat/conversation` - Multi-turn conversation
- `POST /api/v1/chat/stream` - Streaming chat (WebSocket)
- `GET /api/v1/chat/models` - Model information

### Metrics Endpoints
- `GET /api/v1/metrics/performance` - Performance metrics
- `GET /api/v1/metrics/system` - System resource usage
- `GET /api/v1/metrics/usage` - API usage statistics
- `GET /api/v1/metrics/neural-network` - Neural network data
- `GET /api/v1/metrics/processing-pipeline` - Pipeline metrics
- `GET /api/v1/metrics/dashboard` - Comprehensive dashboard data
- `GET /api/v1/metrics/realtime` - Real-time metrics

### WebSocket Endpoints
- `ws://localhost:8000/ws/chat` - Real-time chat
- `ws://localhost:8000/ws/metrics` - Live metrics
- `ws://localhost:8000/ws/system` - System status

## 🐳 Docker Deployment

### Using Docker Compose
```bash
cd amesie_ai_backend
docker-compose up -d
```

This will start:
- **Amesie AI Backend** (port 8000)
- **PostgreSQL Database** (port 5432)
- **Redis Cache** (port 6379)
- **Prometheus** (port 9090)
- **Grafana** (port 3000)

### Environment Configuration
Edit `amesie_ai_backend/.env`:
```env
# Model Configuration
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
MODEL_QUANTIZATION=4bit
DEVICE=auto

# Security
SECRET_KEY=your-production-secret-key

# Database
DATABASE_URL=postgresql://user:password@localhost/amesie_ai

# Redis
REDIS_URL=redis://localhost:6379
```

## 🧪 Testing

### Backend Testing
```bash
cd amesie_ai_backend
python test_backend.py
```

### Frontend Testing
```bash
npm run test
```

## 📈 Monitoring & Analytics

### Prometheus Metrics
- Request rate and duration
- Model inference performance
- GPU utilization and memory usage
- Error rates and system health
- WebSocket connection counts

### Grafana Dashboards
- Real-time performance monitoring
- System resource utilization
- AI model performance metrics
- Custom alerting and notifications

### Logging
- Structured JSON logging
- Sentry error tracking
- Request/response logging
- Performance monitoring

## 🔒 Security Features

- **JWT Authentication**: Secure token-based auth
- **Rate Limiting**: Per-user and per-IP limits
- **Input Validation**: Comprehensive sanitization
- **CORS Protection**: Cross-origin request security
- **HTTPS Support**: Production-ready SSL/TLS

## 🚀 Production Deployment

### Kubernetes Deployment
```bash
kubectl apply -f k8s/
```

### Environment Variables
```env
# Production Settings
DEBUG=false
LOG_LEVEL=INFO
SENTRY_DSN=your-sentry-dsn
PROMETHEUS_PORT=9090

# Security
SECRET_KEY=your-production-secret-key
CORS_ORIGINS=https://yourdomain.com

# Performance
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
```

### Monitoring Stack
- **Prometheus**: Metrics collection
- **Grafana**: Visualization and alerting
- **Sentry**: Error tracking
- **ELK Stack**: Log aggregation

## 📝 API Documentation

### Chat Request Example
```json
{
  "prompt": "Explain quantum computing",
  "max_length": 2048,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50
}
```

### Response Example
```json
{
  "text": "Quantum computing is a revolutionary...",
  "prompt": "Explain quantum computing",
  "inference_time": 2.34,
  "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
  "parameters": {
    "max_length": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50
  }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

- **Documentation**: Check the `/docs` endpoint
- **Issues**: Create an issue in the repository
- **Contact**: support@amesie.ai

## 🎯 Roadmap

- [ ] Multi-model support
- [ ] Advanced analytics
- [ ] Mobile app
- [ ] Enterprise features
- [ ] API marketplace
- [ ] Advanced security features

---

**Amesie AI** - Building the future of AI, one conversation at a time. 🚀
