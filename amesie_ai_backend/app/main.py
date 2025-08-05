"""
Main FastAPI application for Amesie AI Backend
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.core.monitoring import MetricsMiddleware, get_metrics, get_health_status
from app.core.security import check_rate_limit
from app.services.ai.model_service import model_service
from app.services.analytics.analytics_service import analytics_service

# Import routers
from app.api.v1 import chat, metrics
from app.api.websockets import chat_websocket

import structlog

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Amesie AI Backend", version=settings.APP_VERSION)
    
    # Load the AI model
    try:
        logger.info("Loading Mistral 7B model...")
        success = await model_service.load_model()
        if success:
            logger.info("Model loaded successfully")
        else:
            logger.error("Failed to load model")
    except Exception as e:
        logger.error("Error loading model", error=str(e))
    
    # Start analytics service
    logger.info("Starting analytics service...")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Amesie AI Backend")
    
    # Unload model
    try:
        model_service.unload_model()
        logger.info("Model unloaded successfully")
    except Exception as e:
        logger.error("Error unloading model", error=str(e))


# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.APP_VERSION,
    description="Production-ready AI backend with Mistral 7B quantized model",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(MetricsMiddleware)


# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error("Unhandled exception", error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    # Skip rate limiting for health check and metrics
    if request.url.path in ["/health", "/metrics", "/docs", "/redoc"]:
        response = await call_next(request)
        return response
    
    # Check rate limit
    if not await check_rate_limit(request):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"}
        )
    
    response = await call_next(request)
    return response


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Amesie AI Backend",
        "version": settings.APP_VERSION,
        "status": "running"
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return get_health_status()


# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=get_metrics(), media_type="text/plain")


# Include API routers
app.include_router(
    chat.router,
    prefix=f"{settings.API_V1_STR}/chat",
    tags=["chat"]
)

app.include_router(
    metrics.router,
    prefix=f"{settings.API_V1_STR}/metrics",
    tags=["metrics"]
)


# WebSocket endpoints
@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket):
    """WebSocket endpoint for real-time chat"""
    await chat_websocket(websocket)


@app.websocket("/ws/metrics")
async def websocket_metrics_endpoint(websocket):
    """WebSocket endpoint for real-time metrics"""
    from app.api.websockets.chat_websocket import metrics_websocket
    await metrics_websocket(websocket)


@app.websocket("/ws/system")
async def websocket_system_endpoint(websocket):
    """WebSocket endpoint for system status"""
    from app.api.websockets.chat_websocket import system_status_websocket
    await system_status_websocket(websocket)


# Additional utility endpoints
@app.get("/api/v1/info")
async def get_api_info():
    """Get API information"""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.APP_VERSION,
        "model": {
            "name": model_service.model_name,
            "is_loaded": model_service.is_loaded,
            "device": model_service.device,
            "quantization": model_service.quantization
        },
        "endpoints": {
            "chat": f"{settings.API_V1_STR}/chat",
            "metrics": f"{settings.API_V1_STR}/metrics",
            "websockets": {
                "chat": "/ws/chat",
                "metrics": "/ws/metrics",
                "system": "/ws/system"
            }
        }
    }


@app.get("/api/v1/status")
async def get_status():
    """Get detailed system status"""
    return {
        "system": analytics_service.get_system_metrics(),
        "ai": analytics_service.get_ai_metrics(),
        "model": model_service.get_model_info(),
        "health": get_health_status()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )