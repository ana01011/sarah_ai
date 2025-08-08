from typing import Any, Dict, Callable

class ToolRegistry:
    """
    Registry for domain-specific tools callable by agents.
    """
    def __init__(self):
        self.tools: Dict[str, Callable] = {
            "financial_projection": self.financial_projection,
            "budget_analysis": self.budget_analysis,
            "tech_stack_advice": self.tech_stack_advice,
            "project_timeline": self.project_timeline,
        }

    def call(self, tool_name: str, *args, **kwargs) -> Any:
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        return self.tools[tool_name](*args, **kwargs)

    # Example tools
    def financial_projection(self, revenue: float, growth_rate: float, years: int) -> Dict[str, Any]:
        projection = [revenue * ((1 + growth_rate) ** y) for y in range(1, years + 1)]
        return {"projection": projection}

    def budget_analysis(self, budget: float, expenses: float) -> Dict[str, Any]:
        return {"remaining": budget - expenses, "status": "healthy" if budget > expenses else "over budget"}

    def tech_stack_advice(self, project_type: str) -> str:
        if "web" in project_type.lower():
            return "Consider using FastAPI, React, and PostgreSQL."
        return "Consider Python and cloud-native tools."

    def project_timeline(self, features: int, team_size: int) -> str:
        months = max(1, features // max(1, team_size))
        return f"Estimated timeline: {months} months."