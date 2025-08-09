# Amesie AI Backend

A production-ready AI backend powered by Mistral 7B quantized model with comprehensive monitoring, analytics, and real-time capabilities.

## 🚀 Features

- **Mistral 7B Quantized Model**: High-performance language model with optimized inference
- **Real-time Chat API**: WebSocket-based chat with streaming responses
- **Advanced Analytics**: Comprehensive metrics and performance monitoring
- **Multi-model Support**: Easy model switching and A/B testing
- **Production Monitoring**: Prometheus metrics, Sentry error tracking, structured logging
- **Scalable Architecture**: Redis caching, Celery task queue, PostgreSQL database
- **Security**: JWT authentication, rate limiting, input validation
- **Dashboard Integration**: Real-time metrics for frontend dashboard

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   Mistral 7B    │
│   Dashboard     │◄──►│   Backend       │◄──►│   Quantized     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Redis Cache   │    │   PostgreSQL    │    │   Celery Queue  │
│   & Sessions    │    │   Database      │    │   & Tasks       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd amesie_ai_backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize database**
   ```bash
   alembic upgrade head
   ```

6. **Start the services**
   ```bash
   # Start Redis
   redis-server
   
   # Start Celery worker
   celery -A app.celery_app worker --loglevel=info
   
   # Start the API server
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## 🔧 Configuration

### Environment Variables

```env
# Database
DATABASE_URL=postgresql://user:password@localhost/amesie_ai

# Redis
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Model Configuration
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
MODEL_QUANTIZATION=4bit
DEVICE=cuda  # or cpu

# Monitoring
SENTRY_DSN=your-sentry-dsn
PROMETHEUS_PORT=9090

# API Configuration
CORS_ORIGINS=["http://localhost:3000"]
RATE_LIMIT_PER_MINUTE=60
```

## 📊 API Endpoints

### Chat Endpoints
- `POST /api/v1/chat/completion` - Text completion
- `GET /api/v1/chat/stream` - Streaming chat (WebSocket)
- `POST /api/v1/chat/conversation` - Multi-turn conversation

### Analytics Endpoints
- `GET /api/v1/metrics/performance` - Model performance metrics
- `GET /api/v1/metrics/system` - System resource usage
- `GET /api/v1/metrics/usage` - API usage statistics

### Model Management
- `GET /api/v1/models` - List available models
- `POST /api/v1/models/switch` - Switch active model
- `GET /api/v1/models/status` - Model loading status

### Authentication
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/register` - User registration
- `GET /api/v1/auth/me` - Current user info

## 📈 Monitoring & Analytics

### Real-time Metrics
- Model inference latency
- Throughput (requests/second)
- GPU utilization
- Memory usage
- Error rates
- User activity

### Dashboard Integration
The backend provides real-time data for:
- Performance charts
- System status monitoring
- Neural network visualization
- Processing pipeline metrics

## 🔒 Security Features

- JWT-based authentication
- Rate limiting per user/IP
- Input sanitization and validation
- CORS protection
- Request logging and monitoring

## 🚀 Production Deployment

### Docker Deployment
```bash
docker-compose up -d
```

### Kubernetes Deployment
```bash
kubectl apply -f k8s/
```

### Monitoring Stack
- Prometheus for metrics collection
- Grafana for visualization
- Sentry for error tracking
- Structured logging with ELK stack

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For support and questions:
- Create an issue in the repository
- Contact: support@amesie.ai