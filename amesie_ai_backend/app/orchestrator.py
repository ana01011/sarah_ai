from typing import Optional
from app.agents import agent_registry, Agent

class Orchestrator:
    """
    Orchestrator agent (CEO) that delegates queries to the appropriate agent based on message content.
    """
    def __init__(self):
        self.ceo = agent_registry.get("CEO")
        self.cfo = agent_registry.get("CFO")
        self.cto = agent_registry.get("CTO")
        # Add more as needed

    async def route(self, user_message: str) -> str:
        """
        Analyze the message and delegate to the appropriate agent.
        CEO summarizes and responds to the user.
        """
        msg_lower = user_message.lower()
        if any(word in msg_lower for word in ["finance", "budget", "revenue", "profit", "loss", "projection", "expense", "cost"]):
            # Delegate to CFO
            cfo_response = await self.cfo.generate_response(user_message)
            summary = await self.ceo.generate_response(f"The CFO said: {cfo_response}. Summarize for the user.")
            return summary
        elif any(word in msg_lower for word in ["tech", "technology", "stack", "deploy", "engineer", "architecture", "software", "platform"]):
            # Delegate to CTO
            cto_response = await self.cto.generate_response(user_message)
            summary = await self.ceo.generate_response(f"The CTO said: {cto_response}. Summarize for the user.")
            return summary
        else:
            # CEO handles directly
            return await self.ceo.generate_response(user_message)

orchestrator = Orchestrator()