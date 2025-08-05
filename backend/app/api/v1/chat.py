"""
Chat API endpoints for neural network interaction
"""
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
import structlog

from ...services.inference_engine import inference_engine, InferenceRequest, InferenceResponse
from ...core.config import settings

logger = structlog.get_logger(__name__)
router = APIRouter()
security = HTTPBearer(auto_error=False)


# Pydantic models for API
class ChatMessage(BaseModel):
    """Chat message model"""
    content: str = Field(..., min_length=1, max_length=4000, description="Message content")
    role: str = Field(default="user", description="Message role (user/assistant)")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for context")
    max_length: int = Field(default=512, ge=10, le=2048, description="Maximum response length")
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(default=50, ge=1, le=100, description="Top-k sampling")
    top_p: float = Field(default=0.9, ge=0.1, le=1.0, description="Top-p (nucleus) sampling")
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0, description="Repetition penalty")
    stream: bool = Field(default=False, description="Enable streaming response")


class ChatResponse(BaseModel):
    """Chat response model"""
    message: str = Field(..., description="AI response message")
    conversation_id: str = Field(..., description="Conversation ID")
    request_id: str = Field(..., description="Request ID")
    processing_time: float = Field(..., description="Processing time in seconds")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationHistory(BaseModel):
    """Conversation history model"""
    conversation_id: str
    messages: List[ChatMessage]
    created_at: datetime
    updated_at: datetime


class StreamingResponse(BaseModel):
    """Streaming response chunk"""
    chunk: str = Field(..., description="Text chunk")
    is_final: bool = Field(default=False, description="Whether this is the final chunk")
    request_id: str = Field(..., description="Request ID")


# In-memory conversation storage (replace with database in production)
conversations: Dict[str, ConversationHistory] = {}


class ConnectionManager:
    """WebSocket connection manager for real-time chat"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket connection established", client=websocket.client)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WebSocket connection closed", client=websocket.client)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove broken connections
                self.active_connections.remove(connection)


manager = ConnectionManager()


async def get_or_create_conversation(conversation_id: Optional[str] = None) -> ConversationHistory:
    """Get existing conversation or create new one"""
    if conversation_id and conversation_id in conversations:
        return conversations[conversation_id]
    
    # Create new conversation
    new_id = conversation_id or str(uuid.uuid4())
    conversation = ConversationHistory(
        conversation_id=new_id,
        messages=[],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    conversations[new_id] = conversation
    return conversation


@router.post("/chat", response_model=ChatResponse)
async def chat_completion(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    token: Optional[str] = Depends(security)
):
    """
    Generate AI response for chat message
    """
    try:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Get or create conversation
        conversation = await get_or_create_conversation(request.conversation_id)
        
        # Add user message to conversation
        user_message = ChatMessage(content=request.message, role="user")
        conversation.messages.append(user_message)
        
        # Prepare context from conversation history
        context = ""
        if len(conversation.messages) > 1:
            # Include last few messages for context
            recent_messages = conversation.messages[-5:]  # Last 5 messages
            context_parts = []
            for msg in recent_messages[:-1]:  # Exclude current message
                role_prefix = "Human: " if msg.role == "user" else "Assistant: "
                context_parts.append(f"{role_prefix}{msg.content}")
            context = "\n".join(context_parts) + "\nHuman: " + request.message + "\nAssistant: "
        else:
            context = f"Human: {request.message}\nAssistant: "
        
        # Create inference request
        inference_request = InferenceRequest(
            request_id=request_id,
            text=context,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty
        )
        
        # Generate response
        inference_response = await inference_engine.generate_response(inference_request)
        
        # Clean up response text
        response_text = inference_response.generated_text.strip()
        
        # Add AI response to conversation
        ai_message = ChatMessage(content=response_text, role="assistant")
        conversation.messages.append(ai_message)
        conversation.updated_at = datetime.utcnow()
        
        # Log interaction
        logger.info(
            "Chat completion generated",
            request_id=request_id,
            conversation_id=conversation.conversation_id,
            processing_time=inference_response.processing_time,
            tokens_generated=inference_response.tokens_generated
        )
        
        return ChatResponse(
            message=response_text,
            conversation_id=conversation.conversation_id,
            request_id=request_id,
            processing_time=inference_response.processing_time,
            tokens_generated=inference_response.tokens_generated,
            model_info=inference_response.model_info
        )
        
    except Exception as e:
        logger.error("Error in chat completion", error=str(e), request_id=request_id)
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@router.websocket("/chat/stream")
async def chat_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming chat
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Validate request
            try:
                request = ChatRequest(**data)
            except Exception as e:
                await websocket.send_json({
                    "error": f"Invalid request format: {str(e)}"
                })
                continue
            
            # Generate unique request ID
            request_id = str(uuid.uuid4())
            
            try:
                # Get or create conversation
                conversation = await get_or_create_conversation(request.conversation_id)
                
                # Add user message
                user_message = ChatMessage(content=request.message, role="user")
                conversation.messages.append(user_message)
                
                # Prepare context
                context = ""
                if len(conversation.messages) > 1:
                    recent_messages = conversation.messages[-5:]
                    context_parts = []
                    for msg in recent_messages[:-1]:
                        role_prefix = "Human: " if msg.role == "user" else "Assistant: "
                        context_parts.append(f"{role_prefix}{msg.content}")
                    context = "\n".join(context_parts) + "\nHuman: " + request.message + "\nAssistant: "
                else:
                    context = f"Human: {request.message}\nAssistant: "
                
                # Create inference request
                inference_request = InferenceRequest(
                    request_id=request_id,
                    text=context,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty
                )
                
                # Send start signal
                await websocket.send_json({
                    "type": "start",
                    "request_id": request_id,
                    "conversation_id": conversation.conversation_id
                })
                
                if request.stream:
                    # TODO: Implement true streaming generation
                    # For now, simulate streaming by sending response in chunks
                    inference_response = await inference_engine.generate_response(inference_request)
                    response_text = inference_response.generated_text.strip()
                    
                    # Simulate streaming by splitting response into words
                    words = response_text.split()
                    current_chunk = ""
                    
                    for i, word in enumerate(words):
                        current_chunk += word + " "
                        
                        # Send chunk every few words or at the end
                        if (i + 1) % 3 == 0 or i == len(words) - 1:
                            await websocket.send_json({
                                "type": "chunk",
                                "chunk": current_chunk.strip(),
                                "is_final": i == len(words) - 1,
                                "request_id": request_id
                            })
                            current_chunk = ""
                            await asyncio.sleep(0.1)  # Simulate processing delay
                else:
                    # Non-streaming response
                    inference_response = await inference_engine.generate_response(inference_request)
                    response_text = inference_response.generated_text.strip()
                    
                    await websocket.send_json({
                        "type": "complete",
                        "message": response_text,
                        "request_id": request_id,
                        "processing_time": inference_response.processing_time,
                        "tokens_generated": inference_response.tokens_generated
                    })
                
                # Add AI response to conversation
                ai_message = ChatMessage(content=response_text, role="assistant")
                conversation.messages.append(ai_message)
                conversation.updated_at = datetime.utcnow()
                
            except Exception as e:
                logger.error("Error in streaming chat", error=str(e), request_id=request_id)
                await websocket.send_json({
                    "type": "error",
                    "error": f"Error generating response: {str(e)}",
                    "request_id": request_id
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        manager.disconnect(websocket)


@router.get("/conversations", response_model=List[ConversationHistory])
async def get_conversations(
    limit: int = 10,
    offset: int = 0,
    token: Optional[str] = Depends(security)
):
    """
    Get list of conversations
    """
    try:
        # Sort conversations by updated_at in descending order
        sorted_conversations = sorted(
            conversations.values(),
            key=lambda x: x.updated_at,
            reverse=True
        )
        
        # Apply pagination
        paginated = sorted_conversations[offset:offset + limit]
        
        return paginated
        
    except Exception as e:
        logger.error("Error fetching conversations", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error fetching conversations: {str(e)}")


@router.get("/conversations/{conversation_id}", response_model=ConversationHistory)
async def get_conversation(
    conversation_id: str,
    token: Optional[str] = Depends(security)
):
    """
    Get specific conversation by ID
    """
    try:
        if conversation_id not in conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return conversations[conversation_id]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching conversation", error=str(e), conversation_id=conversation_id)
        raise HTTPException(status_code=500, detail=f"Error fetching conversation: {str(e)}")


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    token: Optional[str] = Depends(security)
):
    """
    Delete conversation by ID
    """
    try:
        if conversation_id not in conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        del conversations[conversation_id]
        
        logger.info("Conversation deleted", conversation_id=conversation_id)
        
        return {"message": "Conversation deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting conversation", error=str(e), conversation_id=conversation_id)
        raise HTTPException(status_code=500, detail=f"Error deleting conversation: {str(e)}")


@router.post("/conversations/{conversation_id}/clear")
async def clear_conversation(
    conversation_id: str,
    token: Optional[str] = Depends(security)
):
    """
    Clear messages from conversation
    """
    try:
        if conversation_id not in conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        conversation = conversations[conversation_id]
        conversation.messages = []
        conversation.updated_at = datetime.utcnow()
        
        logger.info("Conversation cleared", conversation_id=conversation_id)
        
        return {"message": "Conversation cleared successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error clearing conversation", error=str(e), conversation_id=conversation_id)
        raise HTTPException(status_code=500, detail=f"Error clearing conversation: {str(e)}")