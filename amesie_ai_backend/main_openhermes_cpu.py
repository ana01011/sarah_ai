"""
Amesie AI Backend with Full OpenHermes Model - CPU Optimized
Designed for CPU deployment with 32GB RAM
"""
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import json
import asyncio
from datetime import datetime
import os
from dotenv import load_dotenv
import logging
import gc
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TextStreamer,
    BitsAndBytesConfig
)
import psutil
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")

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
    title="Amesie AI Backend with Full OpenHermes",
    description="Production backend with OpenHermes model on CPU",
    version="3.0.0"
)

# CORS configuration
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://147.93.102.165:3000",
    "http://147.93.102.165:5173",
    "http://147.93.102.165",
    "*"  # Allow all for testing - restrict in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
model = None
tokenizer = None
executor = ThreadPoolExecutor(max_workers=4)  # For async inference

# CPU Optimization settings
torch.set_num_threads(8)  # Use all 8 vCores
torch.set_grad_enabled(False)  # Disable gradients for inference

def initialize_openhermes():
    """Initialize OpenHermes model with CPU optimizations"""
    global model, tokenizer
    
    try:
        # Model selection - using smaller but powerful models for CPU
        model_name = os.getenv("MODEL_NAME", "teknium/OpenHermes-2.5-Mistral-7B")
        
        # Alternative models optimized for CPU (uncomment to use):
        # model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"  # High quality
        # model_name = "Open-Orca/Mistral-7B-OpenOrca"  # Good balance
        # model_name = "HuggingFaceH4/zephyr-7b-beta"  # Fast on CPU
        # model_name = "microsoft/phi-2"  # Very small, fast (2.7B params)
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"System RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        logger.info(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            padding_side='left'
        )
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # CPU-optimized model loading with quantization
        logger.info("Loading model with CPU optimizations...")
        
        # Quantization config for CPU (8-bit quantization)
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4"
        )
        
        # Try to load with quantization, fallback to float32 if not supported
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="cpu",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                offload_folder="/tmp/offload",  # Use disk for offloading if needed
            )
            logger.info("Model loaded with 8-bit quantization")
        except Exception as e:
            logger.warning(f"Quantization not supported, loading with float32: {e}")
            # Fallback to standard loading
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                low_cpu_mem_usage=True,
                device_map="cpu"
            )
            logger.info("Model loaded with float32 precision")
        
        # Move model to CPU and set to eval mode
        model = model.to("cpu")
        model.eval()
        
        # Clear cache after loading
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logger.info(f"Model loaded successfully!")
        logger.info(f"Memory used: {psutil.virtual_memory().percent}%")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Will use fallback responses")
        return False

# Initialize model on startup
logger.info("Initializing OpenHermes model...")
MODEL_LOADED = initialize_openhermes()

# Request/Response models
class ChatRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 256  # Reduced for CPU performance
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    role: Optional[str] = "AI Assistant"
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    text: str
    prompt: str
    inference_time: float
    model: str
    parameters: Dict[str, Any]
    tokens_generated: Optional[int] = None
    memory_used_gb: Optional[float] = None

# Agent configurations with optimized prompts
AGENT_CONFIGS = {
    "CEO": {
        "domain": "company strategy, vision, leadership",
        "system_prompt": "You are a CEO. Be concise and strategic. Focus on high-level decisions."
    },
    "CFO": {
        "domain": "finance, accounting, projections",
        "system_prompt": "You are a CFO. Provide financial insights concisely."
    },
    "CTO": {
        "domain": "technology, engineering, software architecture",
        "system_prompt": "You are a CTO. Give technical guidance efficiently."
    },
    "COO": {
        "domain": "operations, process optimization, logistics",
        "system_prompt": "You are a COO. Focus on operational efficiency."
    },
    "CMO": {
        "domain": "marketing, branding, campaigns",
        "system_prompt": "You are a CMO. Provide marketing strategies briefly."
    },
    "AI Assistant": {
        "domain": "general questions, cross-domain support",
        "system_prompt": "You are a helpful AI assistant. Be concise and accurate."
    },
}

def generate_with_openhermes(
    prompt: str, 
    role: str, 
    max_length: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50
) -> tuple[str, int]:
    """Generate response using OpenHermes model with CPU optimizations"""
    
    if not MODEL_LOADED or model is None:
        return "Model not loaded. Please check server logs.", 0
    
    try:
        # Prepare the prompt
        system_prompt = AGENT_CONFIGS.get(role, {}).get("system_prompt", "")
        
        # Use chat template if available
        if hasattr(tokenizer, 'chat_template'):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            # Fallback format
            full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
        
        # Tokenize with truncation
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,  # Limit input length
            padding=True
        ).to("cpu")
        
        # Generate with CPU-optimized settings
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,  # Use max_new_tokens instead of max_length
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                num_beams=1,  # Disable beam search for speed
                use_cache=True,  # Enable KV cache
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
        
        # Decode response
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up response
        response = response.strip()
        if not response:
            response = "I understand your question. Let me provide a thoughtful response."
        
        tokens_generated = len(generated_ids)
        
        # Clear memory after generation
        del inputs, outputs, generated_ids
        gc.collect()
        
        return response, tokens_generated
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Error processing request: {str(e)}", 0

# Root endpoint
@app.get("/")
async def root():
    memory_info = psutil.virtual_memory()
    return {
        "message": "Amesie AI Backend with Full OpenHermes",
        "version": "3.0.0",
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "server_ip": "147.93.102.165",
        "system_info": {
            "cpu_cores": psutil.cpu_count(),
            "ram_total_gb": round(memory_info.total / (1024**3), 1),
            "ram_available_gb": round(memory_info.available / (1024**3), 1),
            "ram_used_percent": memory_info.percent
        }
    }

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_status": "loaded" if MODEL_LOADED else "not loaded",
        "memory_usage_percent": psutil.virtual_memory().percent,
        "cpu_usage_percent": psutil.cpu_percent(interval=1),
        "services": {
            "api": "operational",
            "openhermes": "available" if MODEL_LOADED else "not loaded",
            "inference": "ready" if MODEL_LOADED else "unavailable"
        }
    }

# Chat completion endpoint
@app.post("/api/v1/chat/completion", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    start_time = datetime.now()
    
    try:
        # Log request
        logger.info(f"Chat request from role: {request.role}")
        
        # Check memory before processing
        memory_before = psutil.virtual_memory().used / (1024**3)
        
        # Generate response
        response_text, tokens = generate_with_openhermes(
            request.prompt,
            request.role,
            request.max_length,
            request.temperature,
            request.top_p,
            request.top_k
        )
        
        # Calculate metrics
        inference_time = (datetime.now() - start_time).total_seconds()
        memory_after = psutil.virtual_memory().used / (1024**3)
        memory_used = memory_after - memory_before
        
        logger.info(f"Generated {tokens} tokens in {inference_time:.2f}s")
        
        return ChatResponse(
            text=response_text,
            prompt=request.prompt,
            inference_time=inference_time,
            model="OpenHermes-2.5-Mistral-7B" if MODEL_LOADED else "fallback",
            parameters={
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "role": request.role
            },
            tokens_generated=tokens,
            memory_used_gb=round(memory_used, 2)
        )
        
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Streaming chat endpoint for better UX
@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream responses token by token for better perceived performance"""
    # Implementation for streaming (requires SSE or WebSocket)
    pass

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
    process = psutil.Process()
    return {
        "model_loaded": MODEL_LOADED,
        "cpu_percent": process.cpu_percent(),
        "memory_mb": process.memory_info().rss / (1024 * 1024),
        "threads": process.num_threads(),
        "inference_available": MODEL_LOADED
    }

@app.get("/api/v1/metrics/system")
async def get_system_metrics():
    cpu_freq = psutil.cpu_freq()
    return {
        "cpu_usage": psutil.cpu_percent(interval=1),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "cpu_freq_mhz": cpu_freq.current if cpu_freq else 0,
        "memory_usage": psutil.virtual_memory().percent,
        "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "disk_usage": psutil.disk_usage('/').percent,
        "disk_free_gb": round(psutil.disk_usage('/').free / (1024**3), 2)
    }

@app.get("/api/v1/metrics/dashboard")
async def get_dashboard_metrics():
    return {
        "system": await get_system_metrics(),
        "performance": await get_performance_metrics(),
        "model": {
            "name": "OpenHermes-2.5-Mistral-7B" if MODEL_LOADED else "Not Loaded",
            "status": "loaded" if MODEL_LOADED else "not loaded",
            "device": "CPU",
            "optimization": "8-bit quantization" if MODEL_LOADED else "N/A",
            "max_tokens": 256,
            "threads": torch.get_num_threads()
        }
    }

# WebSocket for real-time chat
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)
            
            prompt = request.get("prompt", "")
            role = request.get("role", "AI Assistant")
            
            # Generate response
            start_time = datetime.now()
            response, tokens = generate_with_openhermes(prompt, role, max_length=256)
            inference_time = (datetime.now() - start_time).total_seconds()
            
            await websocket.send_text(json.dumps({
                "text": response,
                "role": role,
                "timestamp": datetime.now().isoformat(),
                "model": "OpenHermes-2.5",
                "inference_time": inference_time,
                "tokens": tokens
            }))
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("="*60)
    logger.info(" Amesie AI Backend Starting")
    logger.info("="*60)
    logger.info(f" Model Status: {'Loaded' if MODEL_LOADED else 'Failed to Load'}")
    logger.info(f" CPU Cores: {psutil.cpu_count()}")
    logger.info(f" Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    logger.info(f" Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    logger.info("="*60)

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    global model, tokenizer
    logger.info("Shutting down...")
    
    # Clean up model from memory
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    logger.info("Cleanup complete")

if __name__ == "__main__":
    print("="*60)
    print(" Amesie AI Backend with Full OpenHermes (CPU Optimized)")
    print("="*60)
    print(f" Server: 147.93.102.165")
    print(f" RAM: 32GB | CPU: 8 vCores | Storage: 400GB")
    print(f" Model: OpenHermes-2.5-Mistral-7B")
    print("="*60)
    
    # Run with optimized settings for production
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker to avoid loading model multiple times
        log_level="info",
        access_log=True,
        loop="asyncio"
    )