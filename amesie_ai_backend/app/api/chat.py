from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.agents import agent_registry
from app.orchestrator import orchestrator

router = APIRouter()

class ChatRequest(BaseModel):
    role: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    role: str
    response: str

@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    Chat with a specific agent by role, or use orchestrator if role is not provided.
    """
    if not request.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if request.role:
        try:
            agent = agent_registry.get(request.role)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        response = await agent.generate_response(request.message)
        return ChatResponse(role=request.role, response=response)
    else:
        # Use orchestrator
        response = await orchestrator.route(request.message)
        return ChatResponse(role="CEO", response=response)