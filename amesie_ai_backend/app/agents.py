from typing import List, Dict, Callable, Any, Optional
from app.tools import ToolRegistry
from app.memory import persistent_memory
from app.intent import intent_detector

AGENT_DOMAINS = {
    "CEO": "company strategy, vision, leadership",
    "CFO": "finance, accounting, projections",
    "CTO": "technology, engineering, software architecture",
    "COO": "operations, process optimization, logistics",
    "CMO": "marketing, branding, campaigns",
    "CIO": "information systems, IT infrastructure",
    "CHRO": "HR, talent, culture, employee engagement",
    "CSO": "security, risk, compliance, cyber protection",
    "CDO": "data, governance, data quality",
    "CAO": "analytics, reporting, KPIs, business insights",
    "CLO": "legal, contracts, regulation, IP",
    "CPO": "product management, design, UX, releases",
    "CCO": "customer service, support, satisfaction",
    "CRO": "revenue, sales, pipeline, forecasting",
    "CBO": "business development, partnerships, alliances",
    "CINO": "innovation, R&D, disruptive ideas",
    "CDAO": "digital, analytics, AI, automation",
    "AI Assistant": "general questions, cross-domain support",
    "Senior Developer": "advanced software engineering, code review, architecture",
    "Junior Developer": "basic coding, debugging, learning to code",
    "Designer": "UI/UX, design, user experience",
    "QA Engineer": "testing, quality assurance, bug reports",
    "Product Manager": "product planning, requirements, user stories, prioritization"
}

AGENT_CONFIGS = {
    "CEO": {"system_prompt": "You are the CEO. Only answer company strategy, vision, and leadership questions. Refuse others.", "tools": []},
    "CFO": {"system_prompt": "You are the CFO. Only answer finance, accounting, and projections. Refuse others.", "tools": ["financial_projection", "budget_analysis"]},
    "CTO": {"system_prompt": "You are the CTO. Only answer technology, engineering, and software architecture. Refuse others.", "tools": ["code_review", "tech_stack_advice", "project_timeline"]},
    "COO": {"system_prompt": "You are the COO. Only answer operations, process optimization, and logistics. Refuse others.", "tools": []},
    "CMO": {"system_prompt": "You are the CMO. Only answer marketing, branding, and campaigns. Refuse others.", "tools": []},
    "CIO": {"system_prompt": "You are the CIO. Only answer information systems and IT infrastructure. Refuse others.", "tools": []},
    "CHRO": {"system_prompt": "You are the CHRO. Only answer HR, talent, and culture. Refuse others.", "tools": []},
    "CSO": {"system_prompt": "You are the CSO. Only answer security, risk, and compliance. Refuse others.", "tools": []},
    "CDO": {"system_prompt": "You are the CDO. Only answer data, governance, and data quality. Refuse others.", "tools": []},
    "CAO": {"system_prompt": "You are the CAO. Only answer analytics, reporting, and KPIs. Refuse others.", "tools": []},
    "CLO": {"system_prompt": "You are the CLO. Only answer legal, contracts, and regulation. Refuse others.", "tools": []},
    "CPO": {"system_prompt": "You are the CPO. Only answer product management, design, and releases. Refuse others.", "tools": []},
    "CCO": {"system_prompt": "You are the CCO. Only answer customer service and satisfaction. Refuse others.", "tools": []},
    "CRO": {"system_prompt": "You are the CRO. Only answer revenue, sales, and forecasting. Refuse others.", "tools": []},
    "CBO": {"system_prompt": "You are the CBO. Only answer business development and partnerships. Refuse others.", "tools": []},
    "CINO": {"system_prompt": "You are the CINO. Only answer innovation and R&D. Refuse others.", "tools": []},
    "CDAO": {"system_prompt": "You are the CDAO. Only answer digital, analytics, and AI. Refuse others.", "tools": []},
    "AI Assistant": {"system_prompt": "You are Sarah, the advanced AI assistant. You help with any general or cross-domain question, and can route to the right executive if needed.", "tools": []},
    "Senior Developer": {"system_prompt": "You are a Senior Developer. Only answer advanced software engineering, code review, and architecture. Refuse others.", "tools": ["code_generation", "code_review", "bug_finder"]},
    "Junior Developer": {"system_prompt": "You are a Junior Developer. Only answer basic coding, debugging, and learning to code. Refuse others.", "tools": ["code_generation", "explain_code"]},
    "Designer": {"system_prompt": "You are a Designer. Only answer UI/UX, design, and user experience. Refuse others.", "tools": ["design_feedback", "color_palette_suggestion"]},
    "QA Engineer": {"system_prompt": "You are a QA Engineer. Only answer testing, quality assurance, and bug reports. Refuse others.", "tools": ["bug_finder", "test_case_generator"]},
    "Product Manager": {"system_prompt": "You are a Product Manager. Only answer product planning, requirements, and prioritization. Refuse others.", "tools": ["user_story_generator", "roadmap_planner"]},
}

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

    async def is_in_domain(self, user_message: str) -> bool:
        # Use intent detection to check if this agent is the best fit
        intents = await intent_detector.detect(user_message)
        if not intents:
            return False
        top_role, confidence = intents[0]
        return top_role == self.role and confidence > 0.5

    async def detect_target_role(self, user_message: str) -> Optional[str]:
        intents = await intent_detector.detect(user_message)
        if not intents:
            return None
        return intents[0][0]

    async def generate_response(self, user_message: str, orchestrator=None, escalation_count=0) -> str:
        prompt = self.system_prompt + "\n"
        for mem in await self.get_memory():
            prompt += f"{mem['sender']}: {mem['message']}\n"
        prompt += f"User: {user_message}\n{self.role}:"
        await self.add_to_memory("User", user_message)
        # Out-of-domain logic
        if not await self.is_in_domain(user_message):
            target_role = await self.detect_target_role(user_message)
            if escalation_count == 0 and target_role and target_role != self.role:
                return f"I am a {self.role} and only answer {AGENT_DOMAINS.get(self.role, 'my domain')} questions. Please ask the {target_role}."
            elif orchestrator and target_role and target_role != self.role:
                # Escalate: fetch answer from correct agent
                target_agent = orchestrator.get_agent(target_role)
                answer = await target_agent.generate_response(user_message, orchestrator, escalation_count=0)
                return f"As requested, here is the {target_role}'s answer: {answer}"
        response = await self.call_mistral(prompt)
        await self.add_to_memory(self.role, response)
        return response

    async def call_mistral(self, prompt: str) -> str:
        return f"[{self.role} would respond here based on prompt: {prompt[:60]}...]"

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