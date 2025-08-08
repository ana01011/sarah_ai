from typing import List, Dict, Callable, Any, Optional
from collections import deque
from app.tools import ToolRegistry

class Agent:
    """
    Role-specific agent with independent memory, system prompt, and tool usage.
    """
    def __init__(
        self,
        role: str,
        system_prompt: str,
        tools: Optional[List[str]] = None,
        memory_limit: int = 20,
    ):
        self.role = role
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.memory = deque(maxlen=memory_limit)  # [(user, message), (agent, message)]
        self.tool_registry = ToolRegistry()

    def add_to_memory(self, sender: str, message: str):
        self.memory.append((sender, message))

    def get_memory(self) -> List[Dict[str, str]]:
        return [{"sender": sender, "message": msg} for sender, msg in self.memory]

    def clear_memory(self):
        self.memory.clear()

    def call_tool(self, tool_name: str, *args, **kwargs) -> Any:
        if tool_name not in self.tools:
            raise PermissionError(f"Agent '{self.role}' cannot use tool '{tool_name}'")
        return self.tool_registry.call(tool_name, *args, **kwargs)

    async def generate_response(self, user_message: str) -> str:
        """
        Compose the prompt using system prompt, memory, and user message,
        then call the Mistral model (stubbed here).
        """
        prompt = self.system_prompt + "\n"
        for mem in self.get_memory():
            prompt += f"{mem['sender']}: {mem['message']}\n"
        prompt += f"User: {user_message}\n{self.role}:"
        self.add_to_memory("User", user_message)
        response = await self.call_mistral(prompt)
        self.add_to_memory(self.role, response)
        return response

    async def call_mistral(self, prompt: str) -> str:
        # Replace with your actual Mistral integration
        # Example: return await model_service.generate_text(prompt=prompt, ...)
        return f"[{self.role} would respond here based on prompt: {prompt[:60]}...]"

# Role-specific agent configs
AGENT_CONFIGS = {
    "CEO": {
        "system_prompt": "You are the CEO. You have a strategic, high-level view and can delegate to other executives.",
        "tools": [],
    },
    "CFO": {
        "system_prompt": "You are the CFO. You are an expert in finance, accounting, and projections.",
        "tools": ["financial_projection", "budget_analysis"],
    },
    "CTO": {
        "system_prompt": "You are the CTO. You are an expert in technology, engineering, and product development.",
        "tools": ["tech_stack_advice", "project_timeline"],
    },
    # Add more roles as needed
}

class AgentRegistry:
    """
    Registry for all agents by role name.
    """
    def __init__(self):
        self.agents: Dict[str, Agent] = {
            role: Agent(role, **cfg) for role, cfg in AGENT_CONFIGS.items()
        }

    def get(self, role: str) -> Agent:
        if role not in self.agents:
            raise ValueError(f"Unknown agent role: {role}")
        return self.agents[role]

    def all_roles(self) -> List[str]:
        return list(self.agents.keys())

agent_registry = AgentRegistry()