"""Conservative recovery detector for uncommitted canonical inbounds.

Disabled and dry-run by default. It never creates proactive outbound messages
and never resumes a paused lead.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from services import sre_logger, supabase_client
from workers.base_worker import BaseWorker


class InactivityRecoveryWorker(BaseWorker):
    name = "InactivityRecoveryWorker"
    interval = int(os.environ.get("INACTIVITY_RECOVERY_INTERVAL", "60"))

    def _run_cycle(self) -> None:
        if os.environ.get("INACTIVITY_RECOVERY_ENABLED", "false").lower() != "true":
            return
        dry_run = os.environ.get("INACTIVITY_RECOVERY_DRY_RUN", "true").lower() != "false"
        enabled_from_raw = os.environ.get("INACTIVITY_RECOVERY_ENABLED_FROM", "").strip()
        if not enabled_from_raw:
            sre_logger.warn(self.name, "enabled without enabled_from; refusing scan")
            return
        enabled_from = datetime.fromisoformat(enabled_from_raw.replace("Z", "+00:00"))
        if enabled_from.tzinfo is None:
            enabled_from = enabled_from.replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=max(60, int(os.environ.get("INACTIVITY_RECOVERY_AFTER_SECONDS", "300")))
        )
        rows = supabase_client.list_inactivity_recovery_candidates(
            enabled_from=enabled_from.isoformat(), cutoff=cutoff.isoformat(), limit=100
        )
        for row in rows:
            # Candidate query excludes commits/proofs/outbound. Revalidation and
            # claim happen atomically in the RPC when apply is explicitly enabled.
            if dry_run:
                sre_logger.info(self.name, f"dry-run pending inbound={row.get('id')}")
                continue
            result = supabase_client.claim_inactivity_recovery_candidate(
                inbound_id=str(row["id"])
            )
            if result.get("state") == "paused":
                sre_logger.info(self.name, f"pending_response_detected inbound={row.get('id')}")
            elif result.get("state") == "requeued":
                sre_logger.info(self.name, f"canonical inbound requeued={row.get('id')}")

