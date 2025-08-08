from typing import List, Tuple

class IntentDetector:
    """
    Pluggable intent detection. Can use keyword, regex, or LLM/NLU.
    """
    def __init__(self):
        self.intent_map = {
            "CFO": ["finance", "budget", "revenue", "profit", "loss", "projection", "expense", "cost", "accounting", "invoice", "payroll"],
            "CTO": ["tech", "technology", "stack", "deploy", "engineer", "architecture", "software", "platform", "cloud", "api", "devops"],
            "CEO": ["strategy", "vision", "leadership", "company", "growth", "market", "mission", "board", "investor", "expansion"],
        }

    def detect(self, message: str) -> List[Tuple[str, float]]:
        """
        Returns a list of (role, confidence) sorted by confidence.
        """
        message_lower = message.lower()
        scores = []
        for role, keywords in self.intent_map.items():
            score = sum(1 for kw in keywords if kw in message_lower)
            if score > 0:
                scores.append((role, float(score) / len(keywords)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

intent_detector = IntentDetector()