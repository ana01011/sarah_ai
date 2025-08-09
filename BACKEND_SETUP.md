# SARAH AI - Backend Setup Guide

This guide will help you set up and run both the backend and frontend of SARAH AI with full API integration.

## Prerequisites

- Node.js (v16 or higher)
- Python 3.8 or higher
- npm or yarn

## Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd amesie_ai_backend
   ```

2. Create a Python virtual environment:
   ```bash
   python -m venv venv
   
   # On Linux/Mac:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. Install the simplified requirements:
   ```bash
   pip install -r requirements_simple.txt
   ```

4. Run the backend server:
   ```bash
   python app/main_simple.py
   ```

   The backend will start on `http://localhost:8000`

## Frontend Setup

1. In a new terminal, navigate to the project root:
   ```bash
   cd /workspace
   ```

2. Install frontend dependencies (if not already installed):
   ```bash
   npm install
   ```

3. Run the frontend development server:
   ```bash
   npm run dev
   ```

   The frontend will start on `http://localhost:5173`

## Testing the Connection

1. Open your browser and go to `http://localhost:5173`
2. You should see "Backend Connected" in the header if everything is working
3. The dashboard will now fetch real data from the backend API
4. Try sending messages in the chat - they will be processed by the backend

## API Endpoints

The backend provides the following endpoints:

- `GET /health` - Health check
- `GET /api/v1/metrics` - System metrics
- `GET /api/v1/agents` - List all agents
- `GET /api/v1/agents/{agent_id}` - Get specific agent
- `POST /api/v1/chat` - Send chat message
- `GET /api/v1/system/status` - System component status
- `GET /api/v1/dashboard/stats` - Dashboard statistics

## Features with Backend Integration

- **Real-time Metrics**: Dashboard metrics are fetched from the backend every 2 seconds
- **System Status**: Component status updates are retrieved from the API
- **Chat Integration**: Messages are sent to the backend and responses are received
- **Agent Management**: Agent information is loaded from the backend
- **Fallback Mode**: If the backend is offline, the frontend falls back to demo mode

## Troubleshooting

1. **CORS Issues**: Make sure the backend is running on port 8000
2. **Connection Failed**: Check that both servers are running
3. **Port Already in Use**: Change the port in the backend script or kill the process using the port

## Environment Variables

The frontend uses these environment variables (already configured in `.env`):
- `VITE_API_URL=http://localhost:8000/api/v1`
- `VITE_WS_URL=ws://localhost:8000/ws`