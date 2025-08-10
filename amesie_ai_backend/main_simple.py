"""
Simplified Amesie AI Backend for Testing
"""
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import json
import asyncio
from datetime import datetime
import random
from mistralai import Mistral
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Amesie AI Backend",
    description="Simplified backend for testing",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Mistral client if API key is available
mistral_client = None
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if MISTRAL_API_KEY:
    try:
        mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    except Exception as e:
        print(f"Failed to initialize Mistral client: {e}")

# Request/Response models
class ChatRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    role: Optional[str] = "AI Assistant"

class ChatResponse(BaseModel):
    text: str
    prompt: str
    inference_time: float
    model: str  # Changed from model_name to avoid conflict
    parameters: Dict[str, Any]

# Agent configurations
AGENT_CONFIGS = {
    "CEO": {"domain": "company strategy, vision, leadership"},
    "CFO": {"domain": "finance, accounting, projections"},
    "CTO": {"domain": "technology, engineering, software architecture"},
    "COO": {"domain": "operations, process optimization, logistics"},
    "CMO": {"domain": "marketing, branding, campaigns"},
    "AI Assistant": {"domain": "general questions, cross-domain support"},
}

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Amesie AI Backend is running",
        "version": "1.0.0",
        "status": "healthy"
    }

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "operational",
            "mistral": "available" if mistral_client else "not configured",
            "redis": "simulated"
        }
    }

# Chat completion endpoint
@app.post("/api/v1/chat/completion", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    start_time = datetime.now()
    
    try:
        # Use Mistral API if available
        if mistral_client:
            try:
                chat_response = mistral_client.chat.complete(
                    model="mistral-small-latest",
                    messages=[
                        {"role": "system", "content": f"You are {request.role}. Focus on: {AGENT_CONFIGS.get(request.role, {}).get('domain', 'general assistance')}"},
                        {"role": "user", "content": request.prompt}
                    ],
                    temperature=request.temperature,
                    max_tokens=request.max_length
                )
                response_text = chat_response.choices[0].message.content
            except Exception as e:
                print(f"Mistral API error: {e}")
                response_text = f"I am {request.role}. {generate_mock_response(request.prompt, request.role)}"
        else:
            # Mock response if Mistral is not available
            response_text = f"I am {request.role}. {generate_mock_response(request.prompt, request.role)}"
        
        inference_time = (datetime.now() - start_time).total_seconds()
        
        return ChatResponse(
            text=response_text,
            prompt=request.prompt,
            inference_time=inference_time,
            model="mistral-small" if mistral_client else "mock-model",  # Changed from model_name
            parameters={
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "role": request.role
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get available roles
@app.get("/api/v1/chat/roles")
async def get_roles():
    return {
        "roles": list(AGENT_CONFIGS.keys()),
        "configurations": AGENT_CONFIGS
    }

# Metrics endpoints
@app.get("/api/v1/metrics/performance")
async def get_performance_metrics():
    return {
        "total_requests": random.randint(1000, 5000),
        "total_errors": random.randint(0, 50),
        "total_inferences": random.randint(900, 4500),
        "error_rate": round(random.uniform(0, 0.05), 3),
        "avg_response_time": round(random.uniform(0.1, 2.0), 2),
        "requests_per_second": round(random.uniform(1, 10), 1)
    }

@app.get("/api/v1/metrics/system")
async def get_system_metrics():
    return {
        "cpu_usage": round(random.uniform(10, 90), 1),
        "memory_usage": round(random.uniform(20, 80), 1),
        "disk_usage": round(random.uniform(30, 70), 1),
        "gpu_utilization": round(random.uniform(0, 100), 1),
        "gpu_memory_usage": round(random.uniform(0, 100), 1)
    }

@app.get("/api/v1/metrics/dashboard")
async def get_dashboard_metrics():
    return {
        "system": await get_system_metrics(),
        "performance": await get_performance_metrics(),
        "ai": {
            "accuracy": round(random.uniform(85, 99), 1),
            "throughput": round(random.uniform(100, 1000), 0),
            "latency": round(random.uniform(10, 200), 1),
            "gpu_utilization": round(random.uniform(0, 100), 1),
            "memory_usage": round(random.uniform(20, 80), 1),
            "active_models": random.randint(1, 5),
            "requests_per_second": round(random.uniform(1, 10), 1),
            "error_rate": round(random.uniform(0, 0.05), 3),
            "avg_response_time": round(random.uniform(0.1, 2.0), 2)
        }
    }

# WebSocket for real-time chat
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)
            
            # Process the chat request
            prompt = request.get("prompt", "")
            role = request.get("role", "AI Assistant")
            
            if mistral_client:
                try:
                    chat_response = mistral_client.chat.complete(
                        model="mistral-small-latest",
                        messages=[
                            {"role": "system", "content": f"You are {role}."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    response = chat_response.choices[0].message.content
                except:
                    response = generate_mock_response(prompt, role)
            else:
                response = generate_mock_response(prompt, role)
            
            await websocket.send_text(json.dumps({
                "text": response,
                "role": role,
                "timestamp": datetime.now().isoformat()
            }))
    except WebSocketDisconnect:
        pass

def generate_mock_response(prompt: str, role: str) -> str:
    """Generate a mock response for testing"""
    domain = AGENT_CONFIGS.get(role, {}).get("domain", "general assistance")
    responses = [
        f"Based on my expertise in {domain}, I can help you with that.",
        f"As the {role}, I recommend considering the following approach...",
        f"From a {domain} perspective, this is an important consideration.",
        f"Let me analyze this from the {role} viewpoint...",
        f"In my role as {role}, I focus on {domain}. Here's my assessment..."
    ]
    return random.choice(responses) + f" [Mock response to: {prompt[:50]}...]"

if __name__ == "__main__":
    print("Starting Amesie AI Backend (Simplified Version)")
    print(f"Mistral API: {'Configured' if mistral_client else 'Not configured (using mock responses)'}")
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Removed reload=True to avoid warning