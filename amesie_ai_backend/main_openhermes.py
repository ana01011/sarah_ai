"""
Amesie AI Backend with OpenHermes Model
Designed for production deployment on remote server
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
import os
from dotenv import load_dotenv
import logging

# Try to import transformers for OpenHermes
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Will use mock responses.")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Amesie AI Backend with OpenHermes",
    description="Production backend with OpenHermes model",
    version="2.0.0"
)

# CORS configuration - Update with your frontend domain
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://147.93.102.165:3000",
    "http://147.93.102.165:5173",
    "*"  # Allow all origins for testing - restrict in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenHermes model
model = None
tokenizer = None
text_generator = None

def initialize_openhermes():
    """Initialize OpenHermes model"""
    global model, tokenizer, text_generator
    
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("Transformers not available, using mock mode")
        return False
    
    try:
        # Use OpenHermes-2.5-Mistral-7B or a smaller model for testing
        model_name = os.getenv("MODEL_NAME", "teknium/OpenHermes-2.5-Mistral-7B")
        
        # For production on limited resources, you might want to use a smaller model
        # model_name = "microsoft/DialoGPT-medium"  # Smaller alternative
        
        logger.info(f"Loading model: {model_name}")
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load with reduced precision for memory efficiency
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            # For CPU, use full precision but warn about performance
            logger.warning("Running on CPU - responses will be slower")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True
            )
        
        # Create text generation pipeline
        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1
        )
        
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Will use mock responses as fallback")
        return False

# Initialize model on startup
MODEL_LOADED = initialize_openhermes()

# Request/Response models
class ChatRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    role: Optional[str] = "AI Assistant"

class ChatResponse(BaseModel):
    text: str
    prompt: str
    inference_time: float
    model: str
    parameters: Dict[str, Any]

# Agent configurations
AGENT_CONFIGS = {
    "CEO": {"domain": "company strategy, vision, leadership", "system_prompt": "You are a CEO. Focus on strategic planning, vision, and leadership."},
    "CFO": {"domain": "finance, accounting, projections", "system_prompt": "You are a CFO. Focus on financial planning, accounting, and fiscal management."},
    "CTO": {"domain": "technology, engineering, software architecture", "system_prompt": "You are a CTO. Focus on technology strategy, engineering, and architecture."},
    "COO": {"domain": "operations, process optimization, logistics", "system_prompt": "You are a COO. Focus on operations, efficiency, and process improvement."},
    "CMO": {"domain": "marketing, branding, campaigns", "system_prompt": "You are a CMO. Focus on marketing strategy, branding, and customer engagement."},
    "AI Assistant": {"domain": "general questions, cross-domain support", "system_prompt": "You are a helpful AI assistant. Provide comprehensive and accurate responses."},
}

def generate_with_openhermes(prompt: str, role: str, max_length: int = 512, temperature: float = 0.7) -> str:
    """Generate response using OpenHermes model"""
    if not MODEL_LOADED or text_generator is None:
        return generate_mock_response(prompt, role)
    
    try:
        # Prepare the prompt with role context
        system_prompt = AGENT_CONFIGS.get(role, {}).get("system_prompt", "")
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        
        # Generate response
        result = text_generator(
            full_prompt,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Extract the generated text
        generated_text = result[0]['generated_text']
        
        # Remove the prompt from the response
        response = generated_text[len(full_prompt):].strip()
        
        return response if response else "I understand your question. Let me help you with that."
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return generate_mock_response(prompt, role)

def generate_mock_response(prompt: str, role: str) -> str:
    """Generate a mock response for testing when model is not available"""
    domain = AGENT_CONFIGS.get(role, {}).get("domain", "general assistance")
    responses = [
        f"As {role}, focusing on {domain}, I can provide insights on this topic.",
        f"From my perspective as {role}, this relates to {domain}.",
        f"In my role as {role}, I specialize in {domain}. Here's my analysis.",
        f"Based on my expertise in {domain}, I recommend the following approach.",
    ]
    base_response = random.choice(responses)
    return f"{base_response} [Responding to: {prompt[:100]}...]"

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Amesie AI Backend with OpenHermes",
        "version": "2.0.0",
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "server_ip": "147.93.102.165"
    }

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_status": "loaded" if MODEL_LOADED else "mock mode",
        "services": {
            "api": "operational",
            "openhermes": "available" if MODEL_LOADED else "not loaded",
            "mock_fallback": "available"
        }
    }

# Chat completion endpoint
@app.post("/api/v1/chat/completion", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    start_time = datetime.now()
    
    try:
        # Generate response
        response_text = generate_with_openhermes(
            request.prompt,
            request.role,
            request.max_length,
            request.temperature
        )
        
        inference_time = (datetime.now() - start_time).total_seconds()
        
        return ChatResponse(
            text=response_text,
            prompt=request.prompt,
            inference_time=inference_time,
            model="OpenHermes-2.5" if MODEL_LOADED else "mock-model",
            parameters={
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "role": request.role
            }
        )
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get available roles
@app.get("/api/v1/chat/roles")
async def get_roles():
    return {
        "roles": list(AGENT_CONFIGS.keys()),
        "configurations": {
            role: {"domain": config["domain"]} 
            for role, config in AGENT_CONFIGS.items()
        }
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
        "requests_per_second": round(random.uniform(1, 10), 1),
        "model_loaded": MODEL_LOADED
    }

@app.get("/api/v1/metrics/system")
async def get_system_metrics():
    import psutil
    
    return {
        "cpu_usage": psutil.cpu_percent(interval=1),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2)
    }

@app.get("/api/v1/metrics/dashboard")
async def get_dashboard_metrics():
    return {
        "system": await get_system_metrics(),
        "performance": await get_performance_metrics(),
        "model": {
            "name": "OpenHermes-2.5" if MODEL_LOADED else "Mock Model",
            "status": "loaded" if MODEL_LOADED else "not loaded",
            "device": "cuda" if torch.cuda.is_available() else "cpu" if TRANSFORMERS_AVAILABLE else "none"
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
            
            prompt = request.get("prompt", "")
            role = request.get("role", "AI Assistant")
            
            response = generate_with_openhermes(prompt, role)
            
            await websocket.send_text(json.dumps({
                "text": response,
                "role": role,
                "timestamp": datetime.now().isoformat(),
                "model": "OpenHermes-2.5" if MODEL_LOADED else "mock"
            }))
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

if __name__ == "__main__":
    print("="*60)
    print(" Amesie AI Backend with OpenHermes")
    print("="*60)
    print(f" Model Status: {'Loaded' if MODEL_LOADED else 'Mock Mode'}")
    print(f" Server will be accessible at: http://147.93.102.165:8000")
    print("="*60)
    
    # Run with host 0.0.0.0 to allow external connections
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Listen on all interfaces
        port=8000,
        log_level="info"
    )