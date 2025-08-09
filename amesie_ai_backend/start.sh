#!/bin/bash

# Amesie AI Backend Startup Script

echo "🚀 Starting Amesie AI Backend..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚙️ Creating .env file from template..."
    cp .env.example .env
    echo "⚠️ Please edit .env file with your configuration before starting the server"
fi

# Start Redis if not running
echo "🔴 Starting Redis..."
redis-server --daemonize yes 2>/dev/null || echo "Redis already running"

# Start the application
echo "🌟 Starting Amesie AI Backend server..."
echo "📊 API Documentation: http://localhost:8000/docs"
echo "📈 Metrics: http://localhost:8000/metrics"
echo "💚 Health Check: http://localhost:8000/health"

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload