from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from schemas.agent_harness import ToolGrant
from services import supabase_client


_PHONE_RE = re.compile(r"(?<!\d)\+?\d[\d\s().-]{6,}\d(?!\d)")
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.IGNORECASE)
_SENSITIVE_KEYS = {
    "phone", "telefone", "lead_id", "email", "contacts", "recipients",
    "raw_text", "input_payload", "parsed_data",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def database_uuid(value: Any) -> str | None:
    try:
        return str(uuid.UUID(str(value))) if value else None
    except (ValueError, TypeError, AttributeError):
        return None


def redact(value: Any, *, key: str | None = None) -> Any:
    """Remove PII before data reaches system_events, logs, or public responses."""
    if key and key.lower() in _SENSITIVE_KEYS:
        if isinstance(value, list):
            return {"redacted": True, "count": len(value)}
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): redact(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        if _UUID_RE.fullmatch(value):
            return value

        def redact_phone(match: re.Match[str]) -> str:
            digits = re.sub(r"\D", "", match.group(0))
            return "[REDACTED_PHONE]" if 10 <= len(digits) <= 15 else match.group(0)

        return _EMAIL_RE.sub("[REDACTED_EMAIL]", _PHONE_RE.sub(redact_phone, value))
    return value


def _rows(query: Any) -> list[dict[str, Any]]:
    result = query.execute()
    data = getattr(result, "data", None) or []
    return data if isinstance(data, list) else [data]


class AgentHarnessRepository:
    def __init__(self, client: Any | None = None) -> None:
        self.client = client or supabase_client.get_client()

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        rows = _rows(self.client.table("agent_sessions").insert(payload))
        if not rows:
            raise RuntimeError("agent session was not persisted")
        return rows[0]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        rows = _rows(
            self.client.table("agent_sessions").select("*").eq("id", session_id).limit(1)
        )
        return rows[0] if rows else None

    def update_session(
        self,
        session_id: str,
        *,
        expected_revision: int,
        changes: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {**changes, "revision": expected_revision + 1, "updated_at": now_iso()}
        rows = _rows(
            self.client.table("agent_sessions").update(payload)
            .eq("id", session_id).eq("revision", expected_revision)
        )
        if not rows:
            raise HTTPException(409, "A sessao mudou; recarregue antes de continuar.")
        return rows[0]

    def create_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        existing = _rows(
            self.client.table("agent_runs").select("*")
            .eq("session_id", payload["session_id"])
            .eq("idempotency_key", payload["idempotency_key"]).limit(1)
        )
        if existing:
            return existing[0]
        rows = _rows(self.client.table("agent_runs").insert(payload))
        if not rows:
            raise RuntimeError("agent run was not persisted")
        return rows[0]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        rows = _rows(self.client.table("agent_runs").select("*").eq("id", run_id).limit(1))
        return rows[0] if rows else None

    def list_runs(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        return _rows(
            self.client.table("agent_runs").select("*").eq("session_id", session_id)
            .order("created_at", desc=True).limit(limit)
        )

    def update_run(
        self,
        run_id: str,
        *,
        expected_revision: int,
        changes: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {**changes, "revision": expected_revision + 1, "updated_at": now_iso()}
        rows = _rows(
            self.client.table("agent_runs").update(payload)
            .eq("id", run_id).eq("revision", expected_revision)
        )
        if not rows:
            raise HTTPException(409, "O run mudou; recarregue antes de continuar.")
        return rows[0]

    def create_step(self, payload: dict[str, Any]) -> dict[str, Any]:
        existing = _rows(
            self.client.table("agent_run_steps").select("*")
            .eq("run_id", payload["run_id"])
            .eq("idempotency_key", payload["idempotency_key"]).limit(1)
        )
        if existing:
            return existing[0]
        rows = _rows(self.client.table("agent_run_steps").insert(payload))
        if not rows:
            raise RuntimeError("agent run step was not persisted")
        return rows[0]

    def update_step(self, step_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        rows = _rows(
            self.client.table("agent_run_steps").update({**changes, "updated_at": now_iso()})
            .eq("id", step_id)
        )
        if not rows:
            raise RuntimeError("agent run step update failed")
        return rows[0]

    def list_steps(self, run_id: str) -> list[dict[str, Any]]:
        return _rows(
            self.client.table("agent_run_steps").select("*").eq("run_id", run_id)
            .order("step_order").limit(12)
        )

    def grant_state(self, user_id: str, persona_id: str) -> dict[str, Any] | None:
        rows = _rows(
            self.client.table("user_persona_access")
            .select("id,user_id,persona_id,agent_tool_grants,agent_tool_grants_revision")
            .eq("user_id", user_id).eq("persona_id", persona_id).limit(1)
        )
        return rows[0] if rows else None

    def grants(self, user_id: str, persona_id: str) -> list[ToolGrant]:
        state = self.grant_state(user_id, persona_id) or {}
        grants: list[ToolGrant] = []
        for raw in state.get("agent_tool_grants") or []:
            try:
                grants.append(ToolGrant.model_validate(raw))
            except Exception:
                continue
        return grants

    def put_grants(
        self,
        *,
        user_id: str,
        persona_id: str,
        expected_revision: int,
        grants: list[ToolGrant],
    ) -> dict[str, Any]:
        rows = _rows(
            self.client.table("user_persona_access").update({
                "agent_tool_grants": [grant.model_dump(mode="json") for grant in grants],
                "agent_tool_grants_revision": expected_revision + 1,
            }).eq("user_id", user_id).eq("persona_id", persona_id)
            .eq("agent_tool_grants_revision", expected_revision)
        )
        if not rows:
            raise HTTPException(409, "Os grants mudaram; recarregue antes de continuar.")
        return rows[0]

    def audit(self, event_type: str, *, entity_type: str, entity_id: str, persona_id: str | None, payload: dict[str, Any], level: str = "info") -> None:
        supabase_client.insert_event(
            {
                "event_type": event_type,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "persona_id": persona_id,
                "payload": redact(payload),
            },
            level=level,
            source="agent_harness",
        )
