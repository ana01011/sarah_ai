from app.agents import agent_registry, Agent
from app.intent import intent_detector

class Orchestrator:
    def __init__(self):
        self.agent_registry = agent_registry

    def get_agent(self, role: str) -> Agent:
        return self.agent_registry.get(role)

    async def route(self, user_message: str) -> str:
        """
        Use intent detection to select agent(s), support agent-to-agent chat.
        """
        intents = intent_detector.detect(user_message)
        if not intents:
            # Default to CEO
            agent = self.get_agent("CEO")
            return await agent.generate_response(user_message, orchestrator=self)
        # If high confidence, route to that agent
        top_role, confidence = intents[0]
        if confidence > 0.5:
            agent = self.get_agent(top_role)
            return await agent.generate_response(user_message, orchestrator=self)
        # If ambiguous, CEO delegates and summarizes
        ceo = self.get_agent("CEO")
        responses = []
        for role, conf in intents:
            if conf > 0.2:
                agent = self.get_agent(role)
                resp = await agent.generate_response(user_message, orchestrator=self)
                responses.append(f"{role}: {resp}")
        summary = await ceo.generate_response(
            f"Multiple agents responded: {' | '.join(responses)}. Summarize for the user.",
            orchestrator=self
        )
        return summary

orchestrator = Orchestrator()