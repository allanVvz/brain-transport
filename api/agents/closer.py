import os
from agents.base import BaseAgent


class CloserAgent(BaseAgent):
    name = "Closer"
    model = "claude-sonnet-4-6"

    def __init__(self):
        super().__init__(os.environ.get("CLOSER_AGENT_URL", "http://localhost:8002/run"))
