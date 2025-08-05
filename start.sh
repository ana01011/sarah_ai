#!/bin/bash

# Neural Network Chat System - Startup Script
# This script starts the complete system with all services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create environment file if it doesn't exist
if [ ! -f .env ]; then
    print_status "Creating .env file with default configuration..."
    cat > .env << EOF
# Neural Network Configuration
SECRET_KEY=your-secret-key-change-in-production-$(openssl rand -hex 32)
DEBUG=false
CUDA_AVAILABLE=false

# Database Configuration
DATABASE_URL=postgresql://neuralnet:neuralnet123@postgres:5432/neuralnet
REDIS_URL=redis://redis:6379

# API Configuration
VITE_API_BASE_URL=http://localhost:8000

# Model Configuration
EMBEDDING_DIM=768
NUM_ATTENTION_HEADS=12
NUM_LAYERS=12
MAX_SEQUENCE_LENGTH=2048
VOCAB_SIZE=50000

# Performance Settings
BATCH_SIZE=32
MIXED_PRECISION=true
DEVICE=cpu
EOF
    print_success "Created .env file with default settings"
else
    print_status ".env file already exists"
fi

# Function to check if CUDA is available
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected. Enabling CUDA support..."
        sed -i 's/CUDA_AVAILABLE=false/CUDA_AVAILABLE=true/' .env
        sed -i 's/DEVICE=cpu/DEVICE=cuda/' .env
    else
        print_warning "No NVIDIA GPU detected. Using CPU mode."
    fi
}

# Function to start services
start_services() {
    print_status "Starting Neural Network Chat System..."
    
    # Pull latest images
    print_status "Pulling Docker images..."
    docker-compose pull
    
    # Build custom images
    print_status "Building application images..."
    docker-compose build
    
    # Start services
    print_status "Starting all services..."
    docker-compose up -d
    
    # Wait for services to be ready
    print_status "Waiting for services to start..."
    sleep 10
    
    # Check service health
    check_services
}

# Function to check service health
check_services() {
    print_status "Checking service health..."
    
    # Check backend
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "✓ Backend API is healthy"
    else
        print_warning "⚠ Backend API is not responding yet"
    fi
    
    # Check frontend
    if curl -f http://localhost:3000/health > /dev/null 2>&1; then
        print_success "✓ Frontend is healthy"
    else
        print_warning "⚠ Frontend is not responding yet"
    fi
    
    # Check database
    if docker-compose exec -T postgres pg_isready -U neuralnet > /dev/null 2>&1; then
        print_success "✓ Database is healthy"
    else
        print_warning "⚠ Database is not ready yet"
    fi
    
    # Check Redis
    if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
        print_success "✓ Redis cache is healthy"
    else
        print_warning "⚠ Redis cache is not ready yet"
    fi
}

# Function to show service URLs
show_urls() {
    echo ""
    print_success "🚀 Neural Network Chat System is starting up!"
    echo ""
    echo "📱 Application URLs:"
    echo "   Frontend Dashboard: http://localhost:3000"
    echo "   Backend API:        http://localhost:8000"
    echo "   API Documentation:  http://localhost:8000/docs"
    echo "   Grafana Monitoring: http://localhost:3001 (admin/admin123)"
    echo "   Prometheus Metrics: http://localhost:9090"
    echo ""
    echo "🔧 Management Commands:"
    echo "   View logs:          docker-compose logs -f"
    echo "   Stop services:      docker-compose down"
    echo "   Restart services:   docker-compose restart"
    echo "   Update system:      docker-compose pull && docker-compose up -d"
    echo ""
    print_status "Services may take a few minutes to fully initialize..."
}

# Function to show system requirements
show_requirements() {
    echo ""
    print_status "System Requirements:"
    echo "   - Docker 20.10+"
    echo "   - Docker Compose 2.0+"
    echo "   - 4GB+ RAM (8GB+ recommended)"
    echo "   - 10GB+ free disk space"
    echo "   - NVIDIA GPU (optional, for acceleration)"
    echo ""
}

# Main execution
main() {
    echo "🧠 Neural Network Chat System - Startup Script"
    echo "=============================================="
    
    # Parse command line arguments
    case "${1:-start}" in
        "start")
            show_requirements
            check_cuda
            start_services
            show_urls
            ;;
        "stop")
            print_status "Stopping all services..."
            docker-compose down
            print_success "All services stopped"
            ;;
        "restart")
            print_status "Restarting all services..."
            docker-compose restart
            print_success "All services restarted"
            ;;
        "logs")
            print_status "Showing service logs..."
            docker-compose logs -f
            ;;
        "status")
            print_status "Service status:"
            docker-compose ps
            check_services
            ;;
        "update")
            print_status "Updating system..."
            docker-compose pull
            docker-compose up -d
            print_success "System updated"
            ;;
        "clean")
            print_warning "This will remove all containers and volumes. Are you sure? (y/N)"
            read -r response
            if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                docker-compose down -v --remove-orphans
                docker system prune -f
                print_success "System cleaned"
            else
                print_status "Clean cancelled"
            fi
            ;;
        "help"|"-h"|"--help")
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  start     Start all services (default)"
            echo "  stop      Stop all services"
            echo "  restart   Restart all services"
            echo "  logs      Show service logs"
            echo "  status    Show service status"
            echo "  update    Update and restart services"
            echo "  clean     Remove all containers and volumes"
            echo "  help      Show this help message"
            ;;
        *)
            print_error "Unknown command: $1"
            print_status "Use '$0 help' for available commands"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"