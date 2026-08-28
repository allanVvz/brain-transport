"""
Fire-and-forget event recorder. Never raises — swallows all exceptions
so it never blocks the main request/worker flow.
"""
from __future__ import annotations
from typing import Optional


def emit(
    event_type: str,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    persona_id: Optional[str] = None,
    payload: Optional[dict] = None,
    level: str = "info",
    source: Optional[str] = None,
) -> None:
    try:
        from services import supabase_client
        supabase_client.insert_event(
            {
                "event_type": event_type,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "persona_id": persona_id,
                "payload": payload or {},
            },
            level=level,
            source=source,
        )
        _update_pipeline_service(event_type)
    except Exception:
        pass


_EVENT_TO_SERVICE: dict[str, str] = {
    "vault_sync_started":    "vault_sync",
    "vault_sync_completed":  "vault_sync",
    "vault_sync_failed":     "vault_sync",
    "item_approved":         "knowledge_validation",
    "item_rejected":         "knowledge_validation",
    "item_promoted_to_kb":   "knowledge_validation",
    "kb_synced":             "knowledge_intake",
    "upload_received":       "knowledge_intake",
    "flow_validator_ran":    "flow_validator",
    "health_checked":        "supabase",
}


def _update_pipeline_service(event_type: str) -> None:
    service = _EVENT_TO_SERVICE.get(event_type)
    if not service:
        return
    from datetime import datetime, timezone
    from services import supabase_client
    status = "error" if "failed" in event_type or "error" in event_type else "online"
    supabase_client.update_pipeline_status(service, {
        "status": status,
        "last_activity": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
