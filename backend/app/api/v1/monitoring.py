"""
Monitoring API endpoints for system health and performance metrics
"""
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
import structlog
import psutil
import torch

from ...services.inference_engine import inference_engine
from ...core.config import settings

logger = structlog.get_logger(__name__)
router = APIRouter()
security = HTTPBearer(auto_error=False)


# Pydantic models for monitoring
class SystemHealth(BaseModel):
    """System health status"""
    status: str = Field(..., description="Overall system status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    uptime: float = Field(..., description="System uptime in seconds")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device being used")
    health_check_time: Optional[float] = Field(None, description="Health check response time")


class PerformanceMetrics(BaseModel):
    """Performance metrics"""
    request_count: int = Field(..., description="Total number of requests")
    avg_processing_time: float = Field(..., description="Average processing time")
    requests_per_second: float = Field(..., description="Requests per second")
    tokens_per_second: float = Field(..., description="Tokens generated per second")
    error_rate: float = Field(..., description="Error rate (0-1)")
    uptime: float = Field(..., description="System uptime in seconds")


class SystemResources(BaseModel):
    """System resource usage"""
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
    memory_usage_percent: float = Field(..., description="Memory usage percentage")
    memory_available_gb: float = Field(..., description="Available memory in GB")
    disk_usage_percent: float = Field(..., description="Disk usage percentage")
    gpu_memory_allocated_gb: Optional[float] = Field(None, description="GPU memory allocated in GB")
    gpu_memory_reserved_gb: Optional[float] = Field(None, description="GPU memory reserved in GB")
    gpu_utilization_percent: Optional[float] = Field(None, description="GPU utilization percentage")


class CacheMetrics(BaseModel):
    """Cache performance metrics"""
    cache_hits: int = Field(..., description="Number of cache hits")
    cache_misses: int = Field(..., description="Number of cache misses")
    hit_rate: float = Field(..., description="Cache hit rate (0-1)")
    total_requests: int = Field(..., description="Total cache requests")


class ModelInfo(BaseModel):
    """Model information"""
    model_name: str = Field(..., description="Model name")
    vocab_size: int = Field(..., description="Vocabulary size")
    d_model: int = Field(..., description="Model dimension")
    num_layers: int = Field(..., description="Number of layers")
    max_seq_len: int = Field(..., description="Maximum sequence length")
    total_parameters: int = Field(..., description="Total number of parameters")
    device: str = Field(..., description="Device the model is on")


class ComprehensiveMetrics(BaseModel):
    """Comprehensive system metrics"""
    performance: PerformanceMetrics
    resources: SystemResources
    cache: CacheMetrics
    model_info: Optional[ModelInfo]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AlertThresholds(BaseModel):
    """Alert thresholds configuration"""
    cpu_threshold: float = Field(default=80.0, description="CPU usage alert threshold")
    memory_threshold: float = Field(default=85.0, description="Memory usage alert threshold")
    error_rate_threshold: float = Field(default=0.05, description="Error rate alert threshold")
    response_time_threshold: float = Field(default=5.0, description="Response time alert threshold")


class Alert(BaseModel):
    """System alert"""
    id: str = Field(..., description="Alert ID")
    type: str = Field(..., description="Alert type")
    severity: str = Field(..., description="Alert severity (low/medium/high/critical)")
    message: str = Field(..., description="Alert message")
    metric_value: float = Field(..., description="Current metric value")
    threshold: float = Field(..., description="Alert threshold")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    resolved: bool = Field(default=False, description="Whether alert is resolved")


# Global alert storage (replace with database in production)
active_alerts: Dict[str, Alert] = {}
alert_history: List[Alert] = []
alert_thresholds = AlertThresholds()


def check_system_alerts(metrics: ComprehensiveMetrics) -> List[Alert]:
    """Check system metrics against thresholds and generate alerts"""
    alerts = []
    current_time = datetime.utcnow()
    
    # CPU usage alert
    if metrics.resources.cpu_usage_percent > alert_thresholds.cpu_threshold:
        alert_id = "cpu_high"
        alert = Alert(
            id=alert_id,
            type="resource",
            severity="high" if metrics.resources.cpu_usage_percent > 90 else "medium",
            message=f"High CPU usage: {metrics.resources.cpu_usage_percent:.1f}%",
            metric_value=metrics.resources.cpu_usage_percent,
            threshold=alert_thresholds.cpu_threshold,
            timestamp=current_time
        )
        alerts.append(alert)
        active_alerts[alert_id] = alert
    
    # Memory usage alert
    if metrics.resources.memory_usage_percent > alert_thresholds.memory_threshold:
        alert_id = "memory_high"
        alert = Alert(
            id=alert_id,
            type="resource",
            severity="high" if metrics.resources.memory_usage_percent > 95 else "medium",
            message=f"High memory usage: {metrics.resources.memory_usage_percent:.1f}%",
            metric_value=metrics.resources.memory_usage_percent,
            threshold=alert_thresholds.memory_threshold,
            timestamp=current_time
        )
        alerts.append(alert)
        active_alerts[alert_id] = alert
    
    # Error rate alert
    if metrics.performance.error_rate > alert_thresholds.error_rate_threshold:
        alert_id = "error_rate_high"
        alert = Alert(
            id=alert_id,
            type="performance",
            severity="critical" if metrics.performance.error_rate > 0.1 else "high",
            message=f"High error rate: {metrics.performance.error_rate:.2%}",
            metric_value=metrics.performance.error_rate,
            threshold=alert_thresholds.error_rate_threshold,
            timestamp=current_time
        )
        alerts.append(alert)
        active_alerts[alert_id] = alert
    
    # Response time alert
    if metrics.performance.avg_processing_time > alert_thresholds.response_time_threshold:
        alert_id = "response_time_high"
        alert = Alert(
            id=alert_id,
            type="performance",
            severity="medium",
            message=f"High response time: {metrics.performance.avg_processing_time:.2f}s",
            metric_value=metrics.performance.avg_processing_time,
            threshold=alert_thresholds.response_time_threshold,
            timestamp=current_time
        )
        alerts.append(alert)
        active_alerts[alert_id] = alert
    
    # Add to history
    alert_history.extend(alerts)
    
    return alerts


@router.get("/health", response_model=SystemHealth)
async def health_check(token: Optional[str] = Depends(security)):
    """
    Get system health status
    """
    try:
        health_status = await inference_engine.health_check()
        
        return SystemHealth(
            status=health_status["status"],
            uptime=time.time() - inference_engine.performance_monitor.start_time,
            model_loaded=health_status["model_loaded"],
            device=health_status["device"],
            health_check_time=health_status.get("health_check_time")
        )
        
    except Exception as e:
        logger.error("Error in health check", error=str(e))
        return SystemHealth(
            status="unhealthy",
            uptime=0,
            model_loaded=False,
            device="unknown"
        )


@router.get("/metrics", response_model=ComprehensiveMetrics)
async def get_metrics(token: Optional[str] = Depends(security)):
    """
    Get comprehensive system metrics
    """
    try:
        # Get metrics from inference engine
        engine_metrics = await inference_engine.get_metrics()
        
        # Get system resource metrics
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        
        # Build system resources
        resources = SystemResources(
            cpu_usage_percent=psutil.cpu_percent(interval=1),
            memory_usage_percent=memory_info.percent,
            memory_available_gb=memory_info.available / (1024**3),
            disk_usage_percent=disk_info.percent
        )
        
        # Add GPU metrics if available
        if torch.cuda.is_available():
            resources.gpu_memory_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            resources.gpu_memory_reserved_gb = torch.cuda.memory_reserved() / (1024**3)
            if hasattr(torch.cuda, 'utilization'):
                resources.gpu_utilization_percent = torch.cuda.utilization()
        
        # Build performance metrics
        performance = PerformanceMetrics(**engine_metrics["performance"])
        
        # Build cache metrics
        cache = CacheMetrics(**engine_metrics["cache"])
        
        # Build model info
        model_info = None
        if engine_metrics["model_info"]:
            model_info = ModelInfo(**engine_metrics["model_info"])
        
        metrics = ComprehensiveMetrics(
            performance=performance,
            resources=resources,
            cache=cache,
            model_info=model_info
        )
        
        # Check for alerts
        check_system_alerts(metrics)
        
        return metrics
        
    except Exception as e:
        logger.error("Error getting metrics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")


@router.get("/metrics/performance", response_model=PerformanceMetrics)
async def get_performance_metrics(token: Optional[str] = Depends(security)):
    """
    Get performance metrics only
    """
    try:
        engine_metrics = await inference_engine.get_metrics()
        return PerformanceMetrics(**engine_metrics["performance"])
        
    except Exception as e:
        logger.error("Error getting performance metrics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error getting performance metrics: {str(e)}")


@router.get("/metrics/resources", response_model=SystemResources)
async def get_resource_metrics(token: Optional[str] = Depends(security)):
    """
    Get system resource metrics only
    """
    try:
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        
        resources = SystemResources(
            cpu_usage_percent=psutil.cpu_percent(interval=1),
            memory_usage_percent=memory_info.percent,
            memory_available_gb=memory_info.available / (1024**3),
            disk_usage_percent=disk_info.percent
        )
        
        # Add GPU metrics if available
        if torch.cuda.is_available():
            resources.gpu_memory_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            resources.gpu_memory_reserved_gb = torch.cuda.memory_reserved() / (1024**3)
            if hasattr(torch.cuda, 'utilization'):
                resources.gpu_utilization_percent = torch.cuda.utilization()
        
        return resources
        
    except Exception as e:
        logger.error("Error getting resource metrics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error getting resource metrics: {str(e)}")


@router.get("/metrics/cache", response_model=CacheMetrics)
async def get_cache_metrics(token: Optional[str] = Depends(security)):
    """
    Get cache metrics only
    """
    try:
        engine_metrics = await inference_engine.get_metrics()
        return CacheMetrics(**engine_metrics["cache"])
        
    except Exception as e:
        logger.error("Error getting cache metrics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error getting cache metrics: {str(e)}")


@router.get("/model/info", response_model=Optional[ModelInfo])
async def get_model_info(token: Optional[str] = Depends(security)):
    """
    Get model information
    """
    try:
        engine_metrics = await inference_engine.get_metrics()
        
        if engine_metrics["model_info"]:
            return ModelInfo(**engine_metrics["model_info"])
        else:
            return None
            
    except Exception as e:
        logger.error("Error getting model info", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


@router.get("/alerts", response_model=List[Alert])
async def get_alerts(
    active_only: bool = Query(default=True, description="Return only active alerts"),
    severity: Optional[str] = Query(default=None, description="Filter by severity"),
    limit: int = Query(default=50, description="Maximum number of alerts to return"),
    token: Optional[str] = Depends(security)
):
    """
    Get system alerts
    """
    try:
        if active_only:
            alerts = list(active_alerts.values())
        else:
            alerts = alert_history[-limit:]  # Get recent alerts from history
        
        # Filter by severity if specified
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        return alerts[:limit]
        
    except Exception as e:
        logger.error("Error getting alerts", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error getting alerts: {str(e)}")


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    token: Optional[str] = Depends(security)
):
    """
    Resolve an active alert
    """
    try:
        if alert_id not in active_alerts:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert = active_alerts[alert_id]
        alert.resolved = True
        
        # Remove from active alerts
        del active_alerts[alert_id]
        
        logger.info("Alert resolved", alert_id=alert_id)
        
        return {"message": f"Alert {alert_id} resolved successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error resolving alert", error=str(e), alert_id=alert_id)
        raise HTTPException(status_code=500, detail=f"Error resolving alert: {str(e)}")


@router.get("/alerts/thresholds", response_model=AlertThresholds)
async def get_alert_thresholds(token: Optional[str] = Depends(security)):
    """
    Get current alert thresholds
    """
    return alert_thresholds


@router.put("/alerts/thresholds", response_model=AlertThresholds)
async def update_alert_thresholds(
    thresholds: AlertThresholds,
    token: Optional[str] = Depends(security)
):
    """
    Update alert thresholds
    """
    try:
        global alert_thresholds
        alert_thresholds = thresholds
        
        logger.info("Alert thresholds updated", thresholds=thresholds.dict())
        
        return alert_thresholds
        
    except Exception as e:
        logger.error("Error updating alert thresholds", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error updating alert thresholds: {str(e)}")


@router.post("/cache/clear")
async def clear_cache(token: Optional[str] = Depends(security)):
    """
    Clear the inference cache
    """
    try:
        # Clear Redis cache
        cache_client = inference_engine.cache.redis_client
        keys = cache_client.keys("inference_cache:*")
        if keys:
            cache_client.delete(*keys)
        
        logger.info("Cache cleared successfully")
        
        return {"message": "Cache cleared successfully", "keys_deleted": len(keys)}
        
    except Exception as e:
        logger.error("Error clearing cache", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")


@router.get("/status")
async def get_status():
    """
    Simple status endpoint for load balancers
    """
    try:
        if inference_engine.model_loaded:
            return {"status": "ok", "timestamp": datetime.utcnow()}
        else:
            return {"status": "loading", "timestamp": datetime.utcnow()}
            
    except Exception:
        return {"status": "error", "timestamp": datetime.utcnow()}