from typing import List, Dict, Callable, Any, Optional
from app.tools import ToolRegistry
from app.memory import persistent_memory

class Agent:
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
        self.memory_limit = memory_limit
        self.tool_registry = ToolRegistry()

    async def add_to_memory(self, sender: str, message: str):
        await persistent_memory.add(self.role, sender, message)

    async def get_memory(self) -> List[Dict[str, str]]:
        mem = await persistent_memory.get(self.role, self.memory_limit)
        return [{"sender": sender, "message": msg} for sender, msg in mem]

    async def clear_memory(self):
        await persistent_memory.clear(self.role)

    def call_tool(self, tool_name: str, *args, **kwargs) -> Any:
        if tool_name not in self.tools:
            raise PermissionError(f"Agent '{self.role}' cannot use tool '{tool_name}'")
        return self.tool_registry.call(tool_name, *args, **kwargs)

    async def generate_response(self, user_message: str, orchestrator=None) -> str:
        """
        Compose the prompt using system prompt, memory, and user message,
        then call the Mistral model (stubbed here).
        If orchestrator is provided, can call other agents.
        """
        prompt = self.system_prompt + "\n"
        for mem in await self.get_memory():
            prompt += f"{mem['sender']}: {mem['message']}\n"
        prompt += f"User: {user_message}\n{self.role}:"
        await self.add_to_memory("User", user_message)
        # Example: agent-to-agent call
        if orchestrator and "[ask CFO]" in user_message:
            cfo = orchestrator.get_agent("CFO")
            cfo_response = await cfo.generate_response(user_message.replace("[ask CFO]", ""), orchestrator)
            await self.add_to_memory("CFO", cfo_response)
            prompt += f"\nCFO: {cfo_response}\n{self.role}:"
        response = await self.call_mistral(prompt)
        await self.add_to_memory(self.role, response)
        return response

    async def call_mistral(self, prompt: str) -> str:
        # Replace with your actual Mistral integration
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