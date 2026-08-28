from __future__ import annotations

import asyncio
import os
import socket
import time

from services import sre_logger, supabase_client, wa_validator_service
from workers.base_worker import BaseWorker


class WaValidatorWorker(BaseWorker):
    """Own long validator conversations outside the API request loop."""

    name = "WaValidatorWorker"
    interval = 2

    def __init__(self) -> None:
        super().__init__()
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}:wa-validator"
        self._last_retention: float | None = None

    def _run_cycle(self) -> None:
        claimed = supabase_client.claim_next_wa_validator_session(self.worker_id)
        if claimed.get("claimed"):
            session = claimed.get("session") or {}
            session_id = str(session.get("id") or "")
            try:
                asyncio.run(
                    wa_validator_service.run_session_direct(
                        session_id, claimed_session=session,
                    )
                )
            except Exception as exc:
                wa_validator_service.mark_session_execution_error(session_id, exc)
                raise

        now = time.monotonic()
        if self._last_retention is not None and now - self._last_retention < 3600:
            return
        self._last_retention = now
        enabled = os.environ.get(
            "WA_VALIDATOR_RETENTION_ENABLED", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        result = wa_validator_service.cleanup_expired_artifacts(
            hours=12, dry_run=not enabled,
        )
        sre_logger.info(
            self.name,
            f"retention {'applied' if enabled else 'dry-run'}: "
            f"{result.get('lead_count', 0)} leads, "
            f"{result.get('session_count', 0)} sessions",
        )

