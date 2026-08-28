import os
from workers.base_worker import BaseWorker
from services import sre_logger


class FlowValidatorWorker(BaseWorker):
    name = "FlowValidatorWorker"
    interval = int(os.environ.get("FLOW_VALIDATOR_INTERVAL", 900))

    def _run_cycle(self):
        from agents.flow_validator.orchestrator import run
        run()
        sre_logger.info(self.name, "cycle complete")

