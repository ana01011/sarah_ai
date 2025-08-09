"""
Chat API endpoints for text generation and streaming
"""

import json
import time
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Request, Response, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from app.services.ai.model_service import model_service
from app.services.analytics.analytics_service import analytics_service
from app.core.security import sanitize_input, check_rate_limit, get_rate_limit_headers
from app.core.monitoring import record_error
import structlog

logger = structlog.get_logger()

router = APIRouter()


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000, description="Input prompt for text generation")
    max_length: Optional[int] = Field(None, ge=1, le=4096, description="Maximum length of generated text")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: Optional[int] = Field(None, ge=1, le=100, description="Top-k sampling parameter")
    stream: bool = Field(False, description="Whether to stream the response")


class ChatResponse(BaseModel):
    text: str
    prompt: str
    inference_time: float
    model_name: str
    parameters: Dict[str, Any]


class StreamResponse(BaseModel):
    text: str
    is_complete: bool
    tokens_generated: Optional[int] = None
    inference_time: Optional[float] = None
    model_name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


@router.post("/completion", response_model=ChatResponse)
async def generate_completion(
    request: ChatRequest,
    http_request: Request,
    response: Response
):
    """Generate text completion"""
    
    start_time = time.time()
    
    try:
        # Check rate limit
        if not await check_rate_limit(http_request):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Sanitize input
        sanitized_prompt = sanitize_input(request.prompt)
        
        # Generate text
        result = await model_service.generate_text(
            prompt=sanitized_prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stream=False
        )
        
        # Record analytics
        duration = time.time() - start_time
        analytics_service.record_request(duration, 200, "/api/v1/chat/completion")
        analytics_service.record_inference(result["inference_time"], result["model_name"], True)
        
        # Add rate limit headers
        headers = get_rate_limit_headers(http_request)
        for key, value in headers.items():
            response.headers[key] = value
        
        return ChatResponse(**result)
        
    except Exception as e:
        duration = time.time() - start_time
        analytics_service.record_request(duration, 500, "/api/v1/chat/completion")
        analytics_service.record_error("generation_error", "/api/v1/chat/completion", str(e))
        record_error("generation_error", "/api/v1/chat/completion", e)
        
        logger.error("Text generation failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Text generation failed")


@router.post("/stream")
async def generate_stream(
    request: ChatRequest,
    http_request: Request,
    response: Response
):
    """Generate streaming text completion"""
    
    if not request.stream:
        raise HTTPException(status_code=400, detail="Stream must be True for streaming endpoint")
    
    start_time = time.time()
    
    try:
        # Check rate limit
        if not await check_rate_limit(http_request):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Sanitize input
        sanitized_prompt = sanitize_input(request.prompt)
        
        # Add rate limit headers
        headers = get_rate_limit_headers(http_request)
        for key, value in headers.items():
            response.headers[key] = value
        
        async def generate_stream_response():
            try:
                async for chunk in model_service.generate_text(
                    prompt=sanitized_prompt,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    stream=True
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # Record analytics
                duration = time.time() - start_time
                analytics_service.record_request(duration, 200, "/api/v1/chat/stream")
                analytics_service.record_inference(duration, model_service.model_name, True)
                
            except Exception as e:
                duration = time.time() - start_time
                analytics_service.record_request(duration, 500, "/api/v1/chat/stream")
                analytics_service.record_error("stream_error", "/api/v1/chat/stream", str(e))
                record_error("stream_error", "/api/v1/chat/stream", e)
                
                error_response = {
                    "error": "Streaming generation failed",
                    "message": str(e)
                }
                yield f"data: {json.dumps(error_response)}\n\n"
        
        return StreamingResponse(
            generate_stream_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        duration = time.time() - start_time
        analytics_service.record_request(duration, 500, "/api/v1/chat/stream")
        analytics_service.record_error("stream_error", "/api/v1/chat/stream", str(e))
        record_error("stream_error", "/api/v1/chat/stream", e)
        
        logger.error("Streaming generation failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Streaming generation failed")


@router.post("/conversation")
async def generate_conversation(
    request: ChatRequest,
    http_request: Request,
    response: Response
):
    """Generate text in conversation context"""
    
    start_time = time.time()
    
    try:
        # Check rate limit
        if not await check_rate_limit(http_request):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Sanitize input
        sanitized_prompt = sanitize_input(request.prompt)
        
        # Add conversation context
        conversation_prompt = f"""<s>[INST] {sanitized_prompt} [/INST]"""
        
        # Generate text
        result = await model_service.generate_text(
            prompt=conversation_prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stream=False
        )
        
        # Record analytics
        duration = time.time() - start_time
        analytics_service.record_request(duration, 200, "/api/v1/chat/conversation")
        analytics_service.record_inference(result["inference_time"], result["model_name"], True)
        
        # Add rate limit headers
        headers = get_rate_limit_headers(http_request)
        for key, value in headers.items():
            response.headers[key] = value
        
        return ChatResponse(**result)
        
    except Exception as e:
        duration = time.time() - start_time
        analytics_service.record_request(duration, 500, "/api/v1/chat/conversation")
        analytics_service.record_error("conversation_error", "/api/v1/chat/conversation", str(e))
        record_error("conversation_error", "/api/v1/chat/conversation", e)
        
        logger.error("Conversation generation failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Conversation generation failed")


@router.get("/models")
async def get_models():
    """Get available models and their status"""
    try:
        model_info = model_service.get_model_info()
        return {
            "models": [
                {
                    "name": model_info["model_name"],
                    "status": "loaded" if model_info["is_loaded"] else "unloaded",
                    "device": model_info["device"],
                    "quantization": model_info["quantization"],
                    "parameters": model_info["parameters"]
                }
            ],
            "active_model": model_info["model_name"] if model_info["is_loaded"] else None
        }
    except Exception as e:
        logger.error("Failed to get models", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get models")


@router.post("/models/load")
async def load_model():
    """Load the AI model"""
    try:
        success = await model_service.load_model()
        if success:
            return {"message": "Model loaded successfully", "model_name": model_service.model_name}
        else:
            raise HTTPException(status_code=500, detail="Failed to load model")
    except Exception as e:
        logger.error("Failed to load model", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to load model")


@router.post("/models/unload")
async def unload_model():
    """Unload the AI model"""
    try:
        model_service.unload_model()
        return {"message": "Model unloaded successfully"}
    except Exception as e:
        logger.error("Failed to unload model", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to unload model")