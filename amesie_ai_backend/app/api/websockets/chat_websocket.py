"""
WebSocket endpoints for real-time chat communication
"""

import json
import asyncio
from typing import Dict, Any, List
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from app.services.ai.model_service import model_service
from app.services.analytics.analytics_service import analytics_service
from app.core.security import sanitize_input
from app.core.monitoring import update_websocket_connections, record_error
import structlog
import time

logger = structlog.get_logger()


class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_count = 0
    
    async def connect(self, websocket: WebSocket):
        """Connect a new WebSocket client"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_count += 1
        update_websocket_connections(self.connection_count)
        
        logger.info("WebSocket client connected", total_connections=self.connection_count)
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket client"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.connection_count -= 1
            update_websocket_connections(self.connection_count)
            
            logger.info("WebSocket client disconnected", total_connections=self.connection_count)
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific WebSocket client"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error("Failed to send message to WebSocket client", error=str(e))
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected WebSocket clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error("Failed to broadcast message", error=str(e))
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


manager = ConnectionManager()


async def chat_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Validate message format
            if "type" not in message_data:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid message format"
                }, websocket)
                continue
            
            message_type = message_data["type"]
            
            if message_type == "chat":
                await handle_chat_message(message_data, websocket)
            elif message_type == "ping":
                await manager.send_personal_message({
                    "type": "pong",
                    "timestamp": time.time()
                }, websocket)
            elif message_type == "subscribe_metrics":
                await handle_metrics_subscription(websocket)
            else:
                await manager.send_personal_message({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                }, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error("WebSocket error", error=str(e), exc_info=True)
        manager.disconnect(websocket)


async def handle_chat_message(message_data: Dict[str, Any], websocket: WebSocket):
    """Handle chat message from WebSocket client"""
    try:
        # Extract message data
        prompt = message_data.get("prompt", "")
        max_length = message_data.get("max_length", 2048)
        temperature = message_data.get("temperature", 0.7)
        top_p = message_data.get("top_p", 0.9)
        top_k = message_data.get("top_k", 50)
        
        # Validate input
        if not prompt:
            await manager.send_personal_message({
                "type": "error",
                "message": "Prompt cannot be empty"
            }, websocket)
            return
        
        # Sanitize input
        sanitized_prompt = sanitize_input(prompt)
        
        # Send acknowledgment
        await manager.send_personal_message({
            "type": "chat_start",
            "message": "Generating response...",
            "timestamp": time.time()
        }, websocket)
        
        # Generate response
        start_time = time.time()
        
        try:
            async for chunk in model_service.generate_text(
                prompt=sanitized_prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stream=True
            ):
                # Send streaming response
                await manager.send_personal_message({
                    "type": "chat_chunk",
                    "text": chunk["text"],
                    "is_complete": chunk["is_complete"],
                    "tokens_generated": chunk.get("tokens_generated"),
                    "timestamp": time.time()
                }, websocket)
                
                if chunk["is_complete"]:
                    break
        
        except Exception as e:
            # Send error response
            await manager.send_personal_message({
                "type": "chat_error",
                "message": f"Generation failed: {str(e)}",
                "timestamp": time.time()
            }, websocket)
            
            # Record error
            duration = time.time() - start_time
            analytics_service.record_error("websocket_generation_error", "/ws/chat", str(e))
            record_error("websocket_generation_error", "/ws/chat", e)
            return
        
        # Send completion message
        duration = time.time() - start_time
        await manager.send_personal_message({
            "type": "chat_complete",
            "inference_time": duration,
            "timestamp": time.time()
        }, websocket)
        
        # Record analytics
        analytics_service.record_inference(duration, model_service.model_name, True)
        
    except Exception as e:
        logger.error("Failed to handle chat message", error=str(e), exc_info=True)
        await manager.send_personal_message({
            "type": "error",
            "message": f"Internal server error: {str(e)}"
        }, websocket)


async def handle_metrics_subscription(websocket: WebSocket):
    """Handle metrics subscription for real-time dashboard updates"""
    try:
        # Send initial metrics
        metrics = analytics_service.get_performance_metrics()
        await manager.send_personal_message({
            "type": "metrics_update",
            "data": metrics,
            "timestamp": time.time()
        }, websocket)
        
        # Start metrics streaming
        while websocket in manager.active_connections:
            try:
                # Get updated metrics
                metrics = analytics_service.get_performance_metrics()
                
                # Send metrics update
                await manager.send_personal_message({
                    "type": "metrics_update",
                    "data": metrics,
                    "timestamp": time.time()
                }, websocket)
                
                # Wait before next update
                await asyncio.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                logger.error("Failed to send metrics update", error=str(e))
                break
                
    except Exception as e:
        logger.error("Failed to handle metrics subscription", error=str(e), exc_info=True)


async def metrics_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics"""
    await manager.connect(websocket)
    
    try:
        while websocket in manager.active_connections:
            try:
                # Get real-time metrics
                metrics = analytics_service.get_performance_metrics()
                
                # Send metrics
                await manager.send_personal_message({
                    "type": "metrics",
                    "data": metrics,
                    "timestamp": time.time()
                }, websocket)
                
                # Wait before next update
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error("Failed to send metrics", error=str(e))
                break
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error("Metrics WebSocket error", error=str(e), exc_info=True)
        manager.disconnect(websocket)


async def system_status_websocket(websocket: WebSocket):
    """WebSocket endpoint for system status updates"""
    await manager.connect(websocket)
    
    try:
        while websocket in manager.active_connections:
            try:
                # Get system status
                system_metrics = analytics_service.get_system_metrics()
                ai_metrics = analytics_service.get_ai_metrics()
                
                status_data = {
                    "system": system_metrics,
                    "ai": ai_metrics,
                    "connections": manager.connection_count,
                    "timestamp": time.time()
                }
                
                # Send status update
                await manager.send_personal_message({
                    "type": "system_status",
                    "data": status_data
                }, websocket)
                
                # Wait before next update
                await asyncio.sleep(3)  # Update every 3 seconds
                
            except Exception as e:
                logger.error("Failed to send system status", error=str(e))
                break
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error("System status WebSocket error", error=str(e), exc_info=True)
        manager.disconnect(websocket)