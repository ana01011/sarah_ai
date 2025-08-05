"""
Metrics API endpoints for dashboard integration
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from app.services.analytics.analytics_service import analytics_service
from app.core.monitoring import get_metrics, get_health_status
import structlog
import time

logger = structlog.get_logger()

router = APIRouter()


@router.get("/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics for dashboard"""
    try:
        return analytics_service.get_performance_metrics()
    except Exception as e:
        logger.error("Failed to get performance metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get performance metrics")


@router.get("/system")
async def get_system_metrics() -> Dict[str, Any]:
    """Get system resource usage metrics"""
    try:
        return {
            "system": analytics_service.get_system_metrics(),
            "ai": analytics_service.get_ai_metrics()
        }
    except Exception as e:
        logger.error("Failed to get system metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get system metrics")


@router.get("/usage")
async def get_usage_metrics() -> Dict[str, Any]:
    """Get API usage statistics"""
    try:
        current_time = time.time()
        
        # Get recent requests (last hour)
        recent_requests = [
            req for req in analytics_service.request_history 
            if current_time - req["timestamp"] <= 3600
        ]
        
        # Get recent errors (last hour)
        recent_errors = [
            error for error in analytics_service.error_history 
            if current_time - error["timestamp"] <= 3600
        ]
        
        # Calculate usage statistics
        total_requests = len(recent_requests)
        total_errors = len(recent_errors)
        error_rate = total_errors / max(total_requests, 1) * 100
        
        # Group by endpoint
        endpoint_stats = {}
        for req in recent_requests:
            endpoint = req["endpoint"]
            if endpoint not in endpoint_stats:
                endpoint_stats[endpoint] = {
                    "requests": 0,
                    "errors": 0,
                    "avg_response_time": 0.0
                }
            endpoint_stats[endpoint]["requests"] += 1
            endpoint_stats[endpoint]["avg_response_time"] += req["duration"]
        
        # Calculate averages
        for endpoint in endpoint_stats:
            requests = endpoint_stats[endpoint]["requests"]
            endpoint_stats[endpoint]["avg_response_time"] /= max(requests, 1)
        
        return {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": error_rate,
            "endpoints": endpoint_stats,
            "time_period": "1 hour"
        }
    except Exception as e:
        logger.error("Failed to get usage metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get usage metrics")


@router.get("/neural-network")
async def get_neural_network_metrics() -> Dict[str, Any]:
    """Get neural network visualization metrics"""
    try:
        return analytics_service.get_neural_network_metrics()
    except Exception as e:
        logger.error("Failed to get neural network metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get neural network metrics")


@router.get("/processing-pipeline")
async def get_processing_pipeline_metrics() -> Dict[str, Any]:
    """Get processing pipeline metrics"""
    try:
        return analytics_service.get_processing_pipeline_metrics()
    except Exception as e:
        logger.error("Failed to get processing pipeline metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get processing pipeline metrics")


@router.get("/health")
async def get_health() -> Dict[str, Any]:
    """Get system health status"""
    try:
        return get_health_status()
    except Exception as e:
        logger.error("Failed to get health status", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get health status")


@router.get("/prometheus")
async def get_prometheus_metrics() -> str:
    """Get Prometheus metrics"""
    try:
        return get_metrics()
    except Exception as e:
        logger.error("Failed to get Prometheus metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get Prometheus metrics")


@router.get("/dashboard")
async def get_dashboard_data() -> Dict[str, Any]:
    """Get comprehensive dashboard data"""
    try:
        # Get all metrics
        performance_metrics = analytics_service.get_performance_metrics()
        neural_network_metrics = analytics_service.get_neural_network_metrics()
        processing_pipeline_metrics = analytics_service.get_processing_pipeline_metrics()
        
        # Combine all metrics
        dashboard_data = {
            "performance": performance_metrics,
            "neural_network": neural_network_metrics,
            "processing_pipeline": processing_pipeline_metrics,
            "timestamp": time.time()
        }
        
        return dashboard_data
    except Exception as e:
        logger.error("Failed to get dashboard data", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get dashboard data")


@router.get("/realtime")
async def get_realtime_metrics() -> Dict[str, Any]:
    """Get real-time metrics for live dashboard updates"""
    try:
        # Get current system metrics
        system_metrics = analytics_service.get_system_metrics()
        ai_metrics = analytics_service.get_ai_metrics()
        
        # Get recent activity
        current_time = time.time()
        recent_requests = [
            req for req in analytics_service.request_history 
            if current_time - req["timestamp"] <= 60  # Last minute
        ]
        
        recent_errors = [
            error for error in analytics_service.error_history 
            if current_time - error["timestamp"] <= 60  # Last minute
        ]
        
        return {
            "system": system_metrics,
            "ai": ai_metrics,
            "activity": {
                "requests_last_minute": len(recent_requests),
                "errors_last_minute": len(recent_errors),
                "requests_per_second": len(recent_requests) / 60.0
            },
            "timestamp": current_time
        }
    except Exception as e:
        logger.error("Failed to get real-time metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get real-time metrics")