import os
from agents.base import BaseAgent


class SDRAgent(BaseAgent):
    name = "SDR"
    model = "claude-haiku-4-5-20251001"

    def __init__(self):
        super().__init__(os.environ.get("SDR_AGENT_URL") or "")
