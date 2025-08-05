# Advanced Neural Network Chat System

A production-ready neural network system with a transformer-based architecture for contextual text processing and real-time chat capabilities. Features a React TypeScript frontend with an AI dashboard and a FastAPI backend with advanced monitoring and caching.

## 🚀 Features

### Neural Network Core
- **Transformer Architecture**: Custom implementation with multi-head attention
- **Mid-sized Language Model**: Optimized for performance and resource efficiency
- **Contextual Responses**: Advanced text processing with conversation memory
- **Production Optimizations**: Caching, batching, and performance monitoring

### Frontend Dashboard
- **Real-time Chat Interface**: WebSocket-powered streaming responses
- **AI Dashboard**: System monitoring and performance metrics
- **Neural Network Visualization**: Interactive model architecture display
- **Performance Charts**: Real-time metrics and analytics
- **Responsive Design**: Modern UI with Tailwind CSS

### Backend API
- **FastAPI Framework**: High-performance async API
- **RESTful Endpoints**: Chat, monitoring, and model management
- **WebSocket Support**: Real-time streaming communication
- **Caching Layer**: Redis-based response caching
- **Monitoring System**: Comprehensive metrics and alerting

### Production Features
- **Docker Containerization**: Multi-stage builds for optimization
- **Database Integration**: PostgreSQL for conversation storage
- **Monitoring Stack**: Prometheus + Grafana dashboards
- **Load Balancing**: Nginx reverse proxy
- **Health Checks**: Comprehensive system monitoring
- **Security**: Authentication, rate limiting, and CORS protection

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  React Frontend │◄──►│   FastAPI Backend │◄──►│ Neural Network  │
│                 │    │                  │    │   (PyTorch)     │
│ - AI Dashboard  │    │ - REST APIs      │    │ - Transformer   │
│ - Chat Interface│    │ - WebSocket      │    │ - Attention     │
│ - Monitoring    │    │ - Caching        │    │ - Generation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌────────▼────────┐             │
         │              │ Supporting Stack │             │
         │              │                 │             │
         └──────────────┤ - PostgreSQL    │◄────────────┘
                        │ - Redis Cache   │
                        │ - Prometheus    │
                        │ - Grafana       │
                        │ - Nginx         │
                        └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Node.js 18+ (for development)
- Python 3.11+ (for development)
- CUDA-compatible GPU (optional, for acceleration)

### Production Deployment

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd neural-network-chat
   ```

2. **Set environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start the complete stack**
   ```bash
   docker-compose up -d
   ```

4. **Access the applications**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Grafana: http://localhost:3001 (admin/admin123)
   - Prometheus: http://localhost:9090

### Development Setup

#### Backend Development
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Development
```bash
npm install
npm run dev
```

## 📊 API Documentation

### Chat Endpoints
- `POST /api/v1/chat/chat` - Send chat message
- `WebSocket /api/v1/chat/chat/stream` - Real-time streaming
- `GET /api/v1/chat/conversations` - Get conversations
- `DELETE /api/v1/chat/conversations/{id}` - Delete conversation

### Monitoring Endpoints
- `GET /api/v1/monitoring/health` - System health check
- `GET /api/v1/monitoring/metrics` - Comprehensive metrics
- `GET /api/v1/monitoring/alerts` - System alerts
- `POST /api/v1/monitoring/cache/clear` - Clear cache

### Example Chat Request
```json
{
  "message": "Explain quantum computing",
  "conversation_id": "optional-uuid",
  "max_length": 512,
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 0.9,
  "repetition_penalty": 1.1
}
```

## 🧠 Neural Network Details

### Model Architecture
- **Type**: Transformer-based language model
- **Parameters**: ~125M (configurable)
- **Layers**: 12 transformer blocks
- **Attention Heads**: 12
- **Embedding Dimension**: 768
- **Vocabulary Size**: 50,000 tokens
- **Context Length**: 2048 tokens

### Key Features
- **Multi-head Self-Attention**: Parallel attention mechanisms
- **Positional Encoding**: Sinusoidal position embeddings
- **Layer Normalization**: Pre-layer norm for stability
- **GELU Activation**: Gaussian Error Linear Units
- **Gradient Checkpointing**: Memory optimization
- **Mixed Precision**: FP16 training support

### Generation Parameters
- **Temperature**: Controls randomness (0.1-2.0)
- **Top-k Sampling**: Limits vocabulary (1-100)
- **Top-p (Nucleus)**: Probability mass threshold (0.1-1.0)
- **Repetition Penalty**: Reduces repetition (1.0-2.0)

## 🔧 Configuration

### Environment Variables
```bash
# Backend Configuration
DEBUG=false
SECRET_KEY=your-secret-key-here
DATABASE_URL=postgresql://user:pass@localhost/neuralnet
REDIS_URL=redis://localhost:6379

# Model Configuration
EMBEDDING_DIM=768
NUM_ATTENTION_HEADS=12
NUM_LAYERS=12
MAX_SEQUENCE_LENGTH=2048
VOCAB_SIZE=50000

# Performance Settings
BATCH_SIZE=32
MIXED_PRECISION=true
DEVICE=cuda  # or cpu
CACHE_SIZE=1000

# Frontend Configuration
VITE_API_BASE_URL=http://localhost:8000
```

### Docker Configuration
The system uses multi-stage Docker builds for optimization:
- **Backend**: Python 3.11 slim with virtual environment
- **Frontend**: Node.js build + Nginx serving
- **Database**: PostgreSQL 15 with persistent storage
- **Cache**: Redis 7 with memory optimization
- **Monitoring**: Prometheus + Grafana stack

## 📈 Monitoring & Metrics

### System Metrics
- **Performance**: Request/response times, throughput
- **Resources**: CPU, memory, GPU utilization
- **Cache**: Hit rates, memory usage
- **Model**: Token generation rates, accuracy metrics
- **Health**: Service status, error rates

### Alerting
- **CPU Usage**: > 80%
- **Memory Usage**: > 85%
- **Error Rate**: > 5%
- **Response Time**: > 5 seconds
- **Cache Hit Rate**: < 70%

### Grafana Dashboards
- **System Overview**: High-level metrics
- **Neural Network Performance**: Model-specific metrics
- **Infrastructure**: Resource utilization
- **Application Logs**: Error tracking and debugging

## 🔒 Security Features

### Backend Security
- **CORS Protection**: Configurable origins
- **Rate Limiting**: Request throttling
- **Input Validation**: Pydantic models
- **SQL Injection Protection**: SQLAlchemy ORM
- **Secure Headers**: Security-focused HTTP headers

### Container Security
- **Non-root Users**: All containers run as non-root
- **Minimal Images**: Alpine-based for smaller attack surface
- **Health Checks**: Automated container monitoring
- **Resource Limits**: Memory and CPU constraints

## 🚀 Performance Optimizations

### Backend Optimizations
- **Async Processing**: FastAPI with async/await
- **Connection Pooling**: Database connection management
- **Response Caching**: Redis-based caching layer
- **Batch Processing**: Request batching for efficiency
- **JIT Compilation**: PyTorch JIT for model optimization

### Frontend Optimizations
- **Code Splitting**: Lazy loading of components
- **Asset Optimization**: Gzip compression, caching headers
- **Virtual Scrolling**: Efficient list rendering
- **Debounced Requests**: Reduced API calls
- **Service Workers**: Offline capability

### Neural Network Optimizations
- **Mixed Precision**: FP16 training and inference
- **Gradient Checkpointing**: Memory-efficient training
- **Model Quantization**: Reduced model size
- **KV-Cache**: Attention key-value caching
- **Parallel Processing**: Multi-GPU support

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v --cov=app
```

### Frontend Tests
```bash
npm test
npm run test:coverage
```

### Integration Tests
```bash
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

## 📝 Development

### Adding New Features
1. **Backend**: Add endpoints in `backend/app/api/v1/`
2. **Frontend**: Add components in `src/components/`
3. **Models**: Extend neural network in `backend/app/models/`
4. **Tests**: Add tests for new functionality

### Code Style
- **Backend**: Black, isort, flake8
- **Frontend**: ESLint, Prettier
- **Commits**: Conventional commits format

### Pre-commit Hooks
```bash
pip install pre-commit
pre-commit install
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Fails**
   - Check CUDA availability
   - Verify memory requirements
   - Review model path configuration

2. **API Connection Issues**
   - Verify backend is running
   - Check CORS configuration
   - Review network connectivity

3. **Performance Issues**
   - Monitor resource usage
   - Check cache hit rates
   - Review batch sizes

4. **Docker Issues**
   - Ensure sufficient disk space
   - Check port conflicts
   - Review container logs

### Logs and Debugging
```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f neural-backend

# Check system metrics
curl http://localhost:8000/api/v1/monitoring/metrics
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting guide

## 🎯 Roadmap

- [ ] Multi-language support
- [ ] Advanced fine-tuning capabilities
- [ ] Distributed training support
- [ ] Mobile application
- [ ] Voice interface integration
- [ ] Advanced analytics dashboard
- [ ] Plugin system for extensions
