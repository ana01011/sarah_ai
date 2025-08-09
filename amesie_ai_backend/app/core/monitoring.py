"""
Monitoring and observability for Amesie AI Backend
"""

import time
import logging
import structlog
from typing import Dict, Any, Optional
from contextlib import contextmanager
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest
from fastapi import Request, Response
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from app.core.config import settings

# Configure Sentry
if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        integrations=[
            FastApiIntegration(),
            RedisIntegration(),
        ],
        traces_sample_rate=0.1,
        environment="production" if not settings.DEBUG else "development",
    )

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

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter(
    'amesie_ai_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'amesie_ai_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

MODEL_INFERENCE_DURATION = Histogram(
    'amesie_ai_model_inference_duration_seconds',
    'Model inference duration in seconds',
    ['model_name', 'model_size']
)

MODEL_REQUESTS = Counter(
    'amesie_ai_model_requests_total',
    'Total number of model requests',
    ['model_name', 'status']
)

ACTIVE_CONNECTIONS = Gauge(
    'amesie_ai_active_connections',
    'Number of active WebSocket connections'
)

GPU_UTILIZATION = Gauge(
    'amesie_ai_gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id']
)

MEMORY_USAGE = Gauge(
    'amesie_ai_memory_usage_bytes',
    'Memory usage in bytes',
    ['type']  # gpu, system
)

ERROR_COUNT = Counter(
    'amesie_ai_errors_total',
    'Total number of errors',
    ['error_type', 'endpoint']
)

CACHE_HIT_RATIO = Summary(
    'amesie_ai_cache_hit_ratio',
    'Cache hit ratio'
)

# Performance tracking
class PerformanceTracker:
    """Track performance metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0
    
    def record_request(self, duration: float, status: int):
        """Record a request"""
        self.request_count += 1
        self.total_response_time += duration
        
        if status >= 400:
            self.error_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        uptime = time.time() - self.start_time
        avg_response_time = self.total_response_time / max(self.request_count, 1)
        error_rate = self.error_count / max(self.request_count, 1)
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "avg_response_time": avg_response_time,
            "requests_per_second": self.request_count / max(uptime, 1)
        }


performance_tracker = PerformanceTracker()


@contextmanager
def track_request_metrics(request: Request, response: Response):
    """Track request metrics"""
    start_time = time.time()
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        status = response.status_code
        
        # Update Prometheus metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=status
        ).inc()
        
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        # Update performance tracker
        performance_tracker.record_request(duration, status)
        
        # Log request
        logger.info(
            "Request processed",
            method=request.method,
            path=request.url.path,
            status_code=status,
            duration=duration,
            client_ip=request.client.host,
            user_agent=request.headers.get("user-agent", "")
        )


@contextmanager
def track_model_inference(model_name: str, model_size: str):
    """Track model inference metrics"""
    start_time = time.time()
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        
        MODEL_INFERENCE_DURATION.labels(
            model_name=model_name,
            model_size=model_size
        ).observe(duration)
        
        logger.info(
            "Model inference completed",
            model_name=model_name,
            model_size=model_size,
            duration=duration
        )


def record_model_request(model_name: str, status: str):
    """Record a model request"""
    MODEL_REQUESTS.labels(
        model_name=model_name,
        status=status
    ).inc()


def update_gpu_utilization(gpu_id: str, utilization: float):
    """Update GPU utilization metric"""
    GPU_UTILIZATION.labels(gpu_id=gpu_id).set(utilization)


def update_memory_usage(memory_type: str, usage_bytes: int):
    """Update memory usage metric"""
    MEMORY_USAGE.labels(type=memory_type).set(usage_bytes)


def record_error(error_type: str, endpoint: str, error: Exception):
    """Record an error"""
    ERROR_COUNT.labels(
        error_type=error_type,
        endpoint=endpoint
    ).inc()
    
    logger.error(
        "Error occurred",
        error_type=error_type,
        endpoint=endpoint,
        error=str(error),
        exc_info=True
    )


def update_websocket_connections(count: int):
    """Update WebSocket connection count"""
    ACTIVE_CONNECTIONS.set(count)


def get_metrics() -> str:
    """Get Prometheus metrics"""
    return generate_latest()


def get_health_status() -> Dict[str, Any]:
    """Get health status"""
    stats = performance_tracker.get_stats()
    
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "uptime": stats["uptime_seconds"],
        "performance": stats,
        "memory_usage": {
            "system": MEMORY_USAGE.labels(type="system")._value.get(),
            "gpu": MEMORY_USAGE.labels(type="gpu")._value.get()
        },
        "active_connections": ACTIVE_CONNECTIONS._value.get()
    }


class MetricsMiddleware:
    """FastAPI middleware for metrics collection"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            response = Response()
            
            with track_request_metrics(request, response):
                await self.app(scope, receive, send)
        else:
            await self.app(scope, receive, send)