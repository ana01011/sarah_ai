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
        prompt = self.system_prompt + "\n"
        for mem in await self.get_memory():
            prompt += f"{mem['sender']}: {mem['message']}\n"
        prompt += f"User: {user_message}\n{self.role}:"
        await self.add_to_memory("User", user_message)
        if orchestrator and "[ask CFO]" in user_message:
            cfo = orchestrator.get_agent("CFO")
            cfo_response = await cfo.generate_response(user_message.replace("[ask CFO]", ""), orchestrator)
            await self.add_to_memory("CFO", cfo_response)
            prompt += f"\nCFO: {cfo_response}\n{self.role}:"
        response = await self.call_mistral(prompt)
        await self.add_to_memory(self.role, response)
        return response

    async def call_mistral(self, prompt: str) -> str:
        return f"[{self.role} would respond here based on prompt: {prompt[:60]}...]"

# Role-specific agent configs (all C-suite + AI Assistant)
AGENT_CONFIGS = {
    "CEO": {"system_prompt": "You are the CEO. Strategic, high-level, visionary, and can delegate to other executives.", "tools": []},
    "CFO": {"system_prompt": "You are the CFO. Expert in finance, accounting, projections, and financial strategy.", "tools": ["financial_projection", "budget_analysis"]},
    "CTO": {"system_prompt": "You are the CTO. Expert in technology, engineering, product development, and IT strategy.", "tools": ["tech_stack_advice", "project_timeline"]},
    "COO": {"system_prompt": "You are the COO. Expert in operations, process optimization, logistics, and execution.", "tools": []},
    "CMO": {"system_prompt": "You are the CMO. Expert in marketing, branding, campaigns, and customer acquisition.", "tools": []},
    "CIO": {"system_prompt": "You are the CIO. Expert in information systems, IT infrastructure, and digital transformation.", "tools": []},
    "CHRO": {"system_prompt": "You are the CHRO. Expert in HR, talent, culture, and employee engagement.", "tools": []},
    "CSO": {"system_prompt": "You are the CSO. Expert in security, risk, compliance, and cyber protection.", "tools": []},
    "CDO": {"system_prompt": "You are the CDO. Expert in data, governance, and data quality.", "tools": []},
    "CAO": {"system_prompt": "You are the CAO. Expert in analytics, reporting, KPIs, and business insights.", "tools": []},
    "CLO": {"system_prompt": "You are the CLO. Expert in legal, contracts, regulation, and intellectual property.", "tools": []},
    "CPO": {"system_prompt": "You are the CPO. Expert in product management, design, UX, and releases.", "tools": []},
    "CCO": {"system_prompt": "You are the CCO. Expert in customer service, support, and satisfaction.", "tools": []},
    "CRO": {"system_prompt": "You are the CRO. Expert in revenue, sales, pipeline, and forecasting.", "tools": []},
    "CBO": {"system_prompt": "You are the CBO. Expert in business development, partnerships, and alliances.", "tools": []},
    "CINO": {"system_prompt": "You are the CINO. Expert in innovation, R&D, and disruptive ideas.", "tools": []},
    "CDAO": {"system_prompt": "You are the CDAO. Expert in digital, analytics, AI, and automation.", "tools": []},
    "AI Assistant": {"system_prompt": "You are Sarah, the advanced AI assistant. You help with any general or cross-domain question, and can route to the right executive if needed.", "tools": []},
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