from typing import List, Tuple
import os

class IntentDetector:
    """
    Pluggable intent detection. Uses LLM if available, falls back to keyword-based.
    """
    def __init__(self):
        self.intent_map = {
            "CEO": ["strategy", "vision", "leadership", "company", "growth", "market", "mission", "board", "investor", "expansion", "overview", "summary"],
            "CFO": ["finance", "budget", "revenue", "profit", "loss", "projection", "expense", "cost", "accounting", "invoice", "payroll", "financial", "cash flow", "balance sheet"],
            "CTO": ["tech", "technology", "stack", "deploy", "engineer", "architecture", "software", "platform", "cloud", "api", "devops", "infrastructure", "system"],
            "COO": ["operations", "process", "efficiency", "logistics", "supply chain", "workflow", "execution"],
            "CMO": ["marketing", "brand", "campaign", "advertising", "promotion", "customer acquisition", "market research"],
            "CIO": ["information", "it", "infrastructure", "systems", "network", "hardware", "software asset"],
            "CHRO": ["hr", "human resources", "recruit", "hiring", "talent", "employee", "benefits", "training", "culture"],
            "CSO": ["security", "risk", "compliance", "cyber", "threat", "incident", "protection"],
            "CDO": ["data", "database", "data warehouse", "data lake", "governance", "data quality"],
            "CAO": ["analytics", "analysis", "insight", "report", "dashboard", "kpi", "metric"],
            "CLO": ["legal", "contract", "regulation", "law", "intellectual property", "litigation"],
            "CPO": ["product", "roadmap", "feature", "design", "ux", "user experience", "release"],
            "CCO": ["customer", "support", "service", "satisfaction", "feedback", "crm"],
            "CRO": ["revenue", "sales", "deal", "pipeline", "quota", "forecast"],
            "CBO": ["business", "partnership", "alliances", "development", "expansion"],
            "CINO": ["innovation", "new idea", "disrupt", "prototype", "r&d", "research"],
            "CDAO": ["digital", "analytics", "transformation", "ai", "machine learning", "automation"],
            "AI Assistant": ["assistant", "help", "general", "ai", "sarah", "question", "info", "support"],
        }
        self.llm_api_key = os.getenv("LLM_INTENT_API_KEY")  # Set this for OpenAI/Mistral

    async def detect_llm(self, message: str) -> List[Tuple[str, float]]:
        """
        Use LLM API for intent detection. Returns [(role, confidence)].
        Stub: Replace with actual OpenAI/Mistral API call.
        """
        # Example prompt for LLM:
        prompt = (
            "Given the following user message, which executive role (CEO, CFO, CTO, COO, CMO, CIO, CHRO, CSO, CDO, CAO, CLO, CPO, CCO, CRO, CBO, CINO, CDAO, AI Assistant) is best suited to answer? "
            "Return a JSON list of (role, confidence) pairs.\n"
            f"Message: {message}"
        )
        # --- Replace this with actual LLM API call ---
        # For now, fallback to keyword-based
        return self.detect_keywords(message)

    def detect_keywords(self, message: str) -> List[Tuple[str, float]]:
        message_lower = message.lower()
        scores = []
        for role, keywords in self.intent_map.items():
            score = sum(1 for kw in keywords if kw in message_lower)
            if score > 0:
                scores.append((role, float(score) / len(keywords)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    async def detect(self, message: str) -> List[Tuple[str, float]]:
        try:
            return await self.detect_llm(message)
        except Exception:
            return self.detect_keywords(message)

intent_detector = IntentDetector()