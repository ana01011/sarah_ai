"""
Main FastAPI application for Neural Network API
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import structlog
import uvicorn

from .core.config import settings
from .api.v1 import chat, monitoring
from .services.inference_engine import inference_engine


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting Neural Network API", version=settings.APP_VERSION)
    
    try:
        # Initialize and load model
        await inference_engine.load_model()
        logger.info("Model loaded successfully")
        
        # Additional startup tasks can be added here
        
    except Exception as e:
        logger.error("Failed to initialize application", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Neural Network API")
    
    try:
        # Cleanup inference engine
        await inference_engine.cleanup()
        logger.info("Cleanup completed")
        
    except Exception as e:
        logger.error("Error during cleanup", error=str(e))


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Advanced Neural Network API for contextual text processing and generation",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)


# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Custom middleware for request logging and metrics
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log requests and responses"""
    start_time = asyncio.get_event_loop().time()
    
    # Log request
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        client=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent")
    )
    
    # Process request
    try:
        response = await call_next(request)
        
        # Calculate processing time
        process_time = asyncio.get_event_loop().time() - start_time
        
        # Log response
        logger.info(
            "Request completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            process_time=process_time
        )
        
        # Add custom headers
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-API-Version"] = settings.APP_VERSION
        
        return response
        
    except Exception as e:
        process_time = asyncio.get_event_loop().time() - start_time
        
        logger.error(
            "Request failed",
            method=request.method,
            url=str(request.url),
            error=str(e),
            process_time=process_time
        )
        
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error_id": str(id(e))}
        )


# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Endpoint not found",
            "path": str(request.url.path),
            "method": request.method
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 errors"""
    logger.error("Internal server error", error=str(exc), path=str(request.url.path))
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error_id": str(id(exc))
        }
    )


# Include API routers
app.include_router(
    chat.router,
    prefix=f"{settings.API_V1_STR}/chat",
    tags=["Chat"]
)

app.include_router(
    monitoring.router,
    prefix=f"{settings.API_V1_STR}/monitoring",
    tags=["Monitoring"]
)


# Root endpoint
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs_url": "/docs" if settings.DEBUG else None,
        "endpoints": {
            "chat": f"{settings.API_V1_STR}/chat",
            "monitoring": f"{settings.API_V1_STR}/monitoring",
            "health": f"{settings.API_V1_STR}/monitoring/health",
            "metrics": f"{settings.API_V1_STR}/monitoring/metrics"
        }
    }


# Health check endpoint (for load balancers)
@app.get("/health")
async def health():
    """Simple health check endpoint"""
    return {"status": "healthy", "service": settings.APP_NAME}


# API information endpoint
@app.get(f"{settings.API_V1_STR}/info")
async def api_info():
    """Get API information"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "Advanced Neural Network API for contextual text processing",
        "model_info": await inference_engine.get_metrics() if inference_engine.model_loaded else None,
        "features": [
            "Real-time chat completion",
            "WebSocket streaming",
            "Conversation management",
            "Performance monitoring",
            "Caching and optimization",
            "Alert system",
            "Health checks"
        ],
        "endpoints": {
            "chat_completion": f"{settings.API_V1_STR}/chat/chat",
            "chat_stream": f"{settings.API_V1_STR}/chat/chat/stream",
            "conversations": f"{settings.API_V1_STR}/chat/conversations",
            "health": f"{settings.API_V1_STR}/monitoring/health",
            "metrics": f"{settings.API_V1_STR}/monitoring/metrics",
            "alerts": f"{settings.API_V1_STR}/monitoring/alerts"
        }
    }


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )