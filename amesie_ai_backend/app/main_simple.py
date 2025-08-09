"""
Simplified FastAPI backend for SARAH AI
This provides mock endpoints for the frontend without requiring AI models
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import random
import time
from datetime import datetime

app = FastAPI(
    title="SARAH AI Backend",
    description="Backend API for SARAH AI Dashboard",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    agent: Optional[str] = "AI Assistant"
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    agent: str
    timestamp: str
    suggestions: Optional[List[str]] = None

class SystemMetrics(BaseModel):
    accuracy: float
    throughput: int
    latency: float
    gpuUtilization: float
    memoryUsage: float
    activeModels: int
    uptime: float
    deployments: int
    codeQuality: float
    security: float

class AgentInfo(BaseModel):
    id: str
    name: str
    role: str
    department: str
    status: str
    specialties: List[str]

# Mock data
AGENT_RESPONSES = {
    "CEO": [
        "Based on our current metrics, the company is performing well with a 15% increase in overall efficiency.",
        "I recommend focusing on strategic partnerships to expand our market reach.",
        "Our Q4 projections show strong growth potential across all departments."
    ],
    "CTO": [
        "System architecture is optimized for scalability with 99.97% uptime.",
        "I suggest implementing microservices for better modularity.",
        "Our tech stack is aligned with industry best practices."
    ],
    "CFO": [
        "Financial projections indicate a 23% increase in revenue for next quarter.",
        "Budget allocation is optimized across all departments.",
        "Our ROI on technology investments is exceeding expectations."
    ],
    "CMO": [
        "Marketing campaigns are showing 45% higher engagement rates.",
        "Brand sentiment analysis indicates positive market perception.",
        "Digital marketing strategies are yielding excellent results."
    ],
    "COO": [
        "Operational efficiency has improved by 18% this quarter.",
        "Supply chain optimization is reducing costs by 12%.",
        "Process automation is streamlining our workflows effectively."
    ]
}

# Endpoints
@app.get("/")
async def root():
    return {
        "message": "SARAH AI Backend",
        "status": "operational",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "operational",
            "database": "operational",
            "ai_models": "operational"
        }
    }

@app.get("/api/v1/metrics")
async def get_metrics():
    """Get current system metrics"""
    return SystemMetrics(
        accuracy=94.7 + random.uniform(-2, 2),
        throughput=2847 + random.randint(-100, 100),
        latency=12.3 + random.uniform(-2, 2),
        gpuUtilization=78 + random.uniform(-5, 5),
        memoryUsage=65 + random.uniform(-5, 5),
        activeModels=12 + random.randint(-2, 2),
        uptime=99.97 + random.uniform(-0.1, 0.1),
        deployments=247 + random.randint(0, 5),
        codeQuality=94.3 + random.uniform(-1, 1),
        security=98.1 + random.uniform(-0.5, 0.5)
    )

@app.get("/api/v1/agents")
async def get_agents():
    """Get all available agents"""
    agents = [
        AgentInfo(
            id="ceo",
            name="Strategic AI",
            role="CEO",
            department="Executive",
            status="online",
            specialties=["Strategic Planning", "Market Analysis", "Leadership"]
        ),
        AgentInfo(
            id="cto",
            name="Tech Lead AI",
            role="CTO",
            department="Technology",
            status="online",
            specialties=["Architecture", "Innovation", "Technical Strategy"]
        ),
        AgentInfo(
            id="cfo",
            name="Finance AI",
            role="CFO",
            department="Finance",
            status="online",
            specialties=["Financial Planning", "Budget Analysis", "Risk Management"]
        ),
        AgentInfo(
            id="cmo",
            name="Marketing AI",
            role="CMO",
            department="Marketing",
            status="online",
            specialties=["Brand Strategy", "Campaign Management", "Market Research"]
        ),
        AgentInfo(
            id="coo",
            name="Operations AI",
            role="COO",
            department="Operations",
            status="online",
            specialties=["Process Optimization", "Supply Chain", "Efficiency"]
        )
    ]
    return agents

@app.get("/api/v1/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get specific agent information"""
    agents = await get_agents()
    agent = next((a for a in agents if a.id == agent_id), None)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@app.post("/api/v1/chat")
async def chat(request: ChatRequest):
    """Chat with an AI agent"""
    # Simulate processing time
    time.sleep(0.5)
    
    # Get agent-specific responses
    agent_role = request.agent.split()[0] if request.agent else "AI"
    responses = AGENT_RESPONSES.get(agent_role, [
        "I'm analyzing your request and preparing a comprehensive response.",
        "Based on the data available, I can provide several insights.",
        "Let me help you with that request."
    ])
    
    response_text = random.choice(responses)
    
    # Generate suggestions based on agent
    suggestions = []
    if agent_role == "CEO":
        suggestions = ["📊 Show quarterly report", "🎯 Strategic goals", "📈 Growth metrics"]
    elif agent_role == "CTO":
        suggestions = ["🔧 System status", "📊 Tech metrics", "🚀 Innovation roadmap"]
    elif agent_role == "CFO":
        suggestions = ["💰 Financial dashboard", "📊 Budget analysis", "📈 Revenue forecast"]
    elif agent_role == "CMO":
        suggestions = ["📱 Campaign metrics", "🎯 Target audience", "📊 Brand analytics"]
    elif agent_role == "COO":
        suggestions = ["⚙️ Operations status", "📊 Efficiency metrics", "🔄 Process optimization"]
    
    return ChatResponse(
        response=response_text,
        agent=request.agent,
        timestamp=datetime.now().isoformat(),
        suggestions=suggestions
    )

@app.get("/api/v1/system/status")
async def get_system_status():
    """Get detailed system status"""
    return {
        "components": [
            {
                "name": "GPU Cluster A",
                "status": "online",
                "uptime": "99.98%",
                "load": 78 + random.uniform(-10, 10)
            },
            {
                "name": "GPU Cluster B",
                "status": "online",
                "uptime": "99.95%",
                "load": 65 + random.uniform(-10, 10)
            },
            {
                "name": "Data Pipeline",
                "status": random.choice(["online", "warning"]),
                "uptime": "99.87%",
                "load": 92 + random.uniform(-10, 10)
            },
            {
                "name": "Model Registry",
                "status": "online",
                "uptime": "99.99%",
                "load": 45 + random.uniform(-10, 10)
            },
            {
                "name": "API Gateway",
                "status": "online",
                "uptime": "99.96%",
                "load": 67 + random.uniform(-10, 10)
            },
            {
                "name": "Storage Array",
                "status": "online",
                "uptime": "99.94%",
                "load": 34 + random.uniform(-10, 10)
            }
        ],
        "overall_health": "excellent",
        "last_updated": datetime.now().isoformat()
    }

@app.get("/api/v1/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    return {
        "activeUsers": 1247 + random.randint(-50, 50),
        "globalReach": 47,
        "dataProcessed": "2.4TB",
        "uptime": "99.98%",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)