import base64
import json
import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from services import (
    agents_service,
    auth_service,
    event_emitter,
    lead_qualification,
    supabase_client,
    whatsapp_outbox,
)

router = APIRouter(prefix="/messages", tags=["messages"])
logger = logging.getLogger("messages")


class SendMessageBody(BaseModel):
    lead_ref: int
    client_message_id: UUID
    agent_id: str | None = None
    texto: str
    sender_id: str | None = None
    nome: str | None = None


def _decode_message_cursor(value: str | None) -> tuple[str | None, int | None]:
    if not value:
        return None, None
    try:
        padding = "=" * (-len(value) % 4)
        payload = json.loads(base64.urlsafe_b64decode(value + padding).decode("utf-8"))
        return str(payload["created_at"]), int(payload["id"])
    except Exception as exc:
        raise HTTPException(400, detail="Cursor de mensagens inválido.") from exc


def _encode_message_cursor(row: dict | None) -> str | None:
    if not row or row.get("id") is None or not row.get("created_at"):
        return None
    raw = json.dumps(
        {"created_at": row["created_at"], "id": row["id"]}, separators=(",", ":")
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _message_page(
    lead_ref: int, *, limit: int, after: str | None, before: str | None,
) -> dict:
    if after and before:
        raise HTTPException(400, detail="Use apenas um cursor: before ou after.")
    after_created_at, after_id = _decode_message_cursor(after)
    before_created_at, before_id = _decode_message_cursor(before)
    rows = supabase_client.get_messages_page(
        lead_ref,
        limit=limit,
        after_created_at=after_created_at,
        after_id=after_id,
        before_created_at=before_created_at,
        before_id=before_id,
    )
    has_more = len(rows) > limit
    if has_more:
        rows = rows[:limit] if after else rows[-limit:]
    return {
        "items": rows,
        "before_cursor": _encode_message_cursor(rows[0] if rows else None) or before,
        "after_cursor": _encode_message_cursor(rows[-1] if rows else None) or after,
        # Kept for one release so older dashboard bundles continue polling.
        "next_cursor": _encode_message_cursor(rows[-1] if rows else None) or after,
        "has_more": has_more,
    }


@router.post("/send")
def send_message(body: SendMessageBody, request: Request) -> dict:
    """Atomically enqueue one operator message for one explicit binding."""
    text = (body.texto or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="texto vazio")

    lead = supabase_client.get_lead_by_ref(body.lead_ref) or {}
    if not lead:
        raise HTTPException(404, detail="Lead nao encontrado")
    persona_id = lead.get("persona_id")
    if not persona_id:
        raise HTTPException(409, detail="Lead sem persona.")
    auth_service.assert_persona_access(request, persona_id=persona_id)

    agent: dict | None = None
    if body.agent_id:
        agent = agents_service.get_agent(body.agent_id)
        if not agent or agent.get("persona_id") != persona_id:
            raise HTTPException(403, detail="Agente nao pertence a persona do lead.")
    else:
        stage = lead.get("stage") or lead.get("funnel_stage") or "novo"
        try:
            agent, _role = agents_service.resolve_for_stage(persona_id, stage)
        except Exception as exc:
            logger.warning("resolve_for_stage failed in send: %s", exc)

    client_message_id = str(body.client_message_id)
    message_id = f"manual:{client_message_id}"
    try:
        result = whatsapp_outbox.enqueue_outbound(
            lead=lead,
            text=text,
            sender_type="human",
            message_id=message_id,
            correlation_id=message_id,
            idempotency_key=message_id,
            metadata={
                "agent_id": agent.get("id") if agent else None,
                "bot_name": agent.get("bot_name") if agent else None,
                "sender_id": body.sender_id,
                "nome": body.nome or body.sender_id or "Operador",
                "client_message_id": client_message_id,
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("outbox enqueue failed: %s", exc)
        raise HTTPException(500, detail="Falha ao enfileirar mensagem") from exc

    if not result.get("deduplicated"):
        event_emitter.emit(
            "message.new",
            entity_type="message",
            entity_id=message_id,
            persona_id=persona_id,
            payload={
                "lead_ref": body.lead_ref,
                "sender_type": "human",
                "buffer_id": result["buffer_id"],
            },
            source="messages.send",
        )
    return {
        "ok": True,
        "message_id": result.get("message_id") or message_id,
        "status": result.get("status") or "pending_send",
        "buffer_id": result["buffer_id"],
        "deduplicated": bool(result.get("deduplicated")),
    }


def _resolve_scope_lead_refs(
    request: Request,
    *,
    persona_id: str | None = None,
    persona_slug: str | None = None,
    audience_id: str | None = None,
    audience_slug: str | None = None,
) -> tuple[str | None, list[int] | None]:
    resolved_persona_id = persona_id
    if not resolved_persona_id and persona_slug:
        persona = supabase_client.get_persona(persona_slug)
        resolved_persona_id = persona.get("id") if persona else None
    if resolved_persona_id:
        auth_service.assert_persona_access(
            request,
            persona_id=resolved_persona_id,
            persona_slug=persona_slug,
        )
        if audience_id or audience_slug:
            lead_refs = supabase_client.get_lead_refs_for_audience_scope(
                persona_id=resolved_persona_id,
                audience_id=audience_id,
                audience_slug=audience_slug,
            )
            return resolved_persona_id, lead_refs
    return resolved_persona_id, None


def _decorate_conversations(
    rows: list[dict],
    validation_scope: str = "exclude",
) -> list[dict]:
    decorated: list[dict] = []
    leads_by_ref = supabase_client.get_leads_by_refs([
        int(row["lead_ref"])
        for row in rows
        if row.get("lead_ref") is not None
    ])
    for row in rows:
        lead_ref = row.get("lead_ref")
        lead = leads_by_ref.get(int(lead_ref), {}) if lead_ref is not None else {}
        extra = lead_qualification.decorate_lead(lead)
        is_validation = bool(extra.get("validation", {}).get("is_validation"))
        if validation_scope == "only" and not is_validation:
            continue
        if validation_scope == "exclude" and is_validation:
            continue
        decorated.append({
            **row,
            "qualification": extra.get("qualification") or {},
            "qualification_score": extra.get("qualification_score") or 0,
            "qualification_signals": extra.get("qualification_signals") or [],
            "validation": extra.get("validation") or {},
        })
    return decorated


@router.get("/conversations")
def get_conversations(
    request: Request,
    hours: int = Query(168, le=720),
    persona_id: str | None = Query(None),
    persona_slug: str | None = Query(None),
    audience_id: str | None = Query(None),
    audience_slug: str | None = Query(None),
    validation_scope: str = Query("exclude", pattern="^(exclude|only|all)$"),
):
    """Return one row per conversation, ordered by the latest message."""
    try:
        user = auth_service.current_user(request)
        if (user.get("account_type") or "internal") == "client":
            validation_scope = "exclude"
        resolved_persona_id, lead_refs = _resolve_scope_lead_refs(
            request,
            persona_id=persona_id,
            persona_slug=persona_slug,
            audience_id=audience_id,
            audience_slug=audience_slug,
        )
        if resolved_persona_id and lead_refs is not None:
            return _decorate_conversations(
                supabase_client.get_conversations(
                    hours=hours,
                    persona_id=resolved_persona_id,
                    lead_refs=lead_refs,
                ),
                validation_scope,
            )
        if persona_id:
            auth_service.assert_persona_access(request, persona_id=persona_id)
            return _decorate_conversations(
                supabase_client.get_conversations(
                    hours=hours,
                    persona_id=persona_id,
                ),
                validation_scope,
            )
        if auth_service.is_admin(auth_service.current_user(request)):
            return _decorate_conversations(
                supabase_client.get_conversations(hours=hours, persona_id=None),
                validation_scope,
            )
        rows: list[dict] = []
        for pid in auth_service.allowed_persona_ids(request):
            rows.extend(
                supabase_client.get_conversations(hours=hours, persona_id=pid)
            )
        rows.sort(
            key=lambda item: item.get("last_at") or item.get("created_at") or "",
            reverse=True,
        )
        return _decorate_conversations(rows, validation_scope)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("get_conversations failed: %s", exc)
        try:
            from services import sre_logger
            sre_logger.error("messages.conversations", f"failed: {exc}", exc)
        except Exception:
            pass
        raise HTTPException(
            status_code=500,
            detail="Falha ao carregar conversas da persona selecionada.",
        ) from exc


@router.get("/by-ref/{lead_ref}")
def get_messages_by_ref(
    lead_ref: int,
    request: Request,
    limit: int = Query(50, ge=1, le=100),
    after: str | None = Query(None),
    before: str | None = Query(None),
    persona_id: str | None = Query(None),
    persona_slug: str | None = Query(None),
    audience_id: str | None = Query(None),
    audience_slug: str | None = Query(None),
    validation_scope: str = Query("exclude", pattern="^(exclude|only|all)$"),
):
    """Fetch messages by the canonical integer lead reference."""
    lead = supabase_client.get_lead_by_ref(lead_ref)
    user = auth_service.current_user(request)
    if (user.get("account_type") or "internal") == "client":
        validation_scope = "exclude"
    if lead:
        is_validation = lead_qualification.is_validation_lead(lead)
        if (
            (validation_scope == "only" and not is_validation)
            or (validation_scope == "exclude" and is_validation)
        ):
            raise HTTPException(status_code=404, detail="Conversa nao encontrada.")
    resolved_persona_id, lead_refs = _resolve_scope_lead_refs(
        request,
        persona_id=persona_id,
        persona_slug=persona_slug,
        audience_id=audience_id,
        audience_slug=audience_slug,
    )
    if lead_refs is not None and lead_ref not in lead_refs:
        raise HTTPException(status_code=403, detail="Lead fora da audiencia atual.")
    if (
        resolved_persona_id
        and supabase_client.lead_has_membership(
            lead_ref,
            resolved_persona_id,
            audience_id,
        )
    ):
        return _message_page(lead_ref, limit=limit, after=after, before=before)
    if lead and lead.get("persona_id"):
        auth_service.assert_persona_access(
            request,
            persona_id=lead.get("persona_id"),
        )
    return _message_page(lead_ref, limit=limit, after=after, before=before)


@router.get("/{lead_id}")
def get_messages(
    lead_id: str,
    request: Request,
    limit: int = Query(200, le=500),
):
    if lead_id.isdigit():
        lead = supabase_client.get_lead_by_ref(int(lead_id))
        if lead and lead.get("persona_id"):
            auth_service.assert_persona_access(
                request,
                persona_id=lead.get("persona_id"),
            )
    return supabase_client.get_messages(lead_id, limit=limit)


@router.get("")
def recent_messages(
    request: Request,
    hours: int = Query(24, le=168),
    persona_id: str | None = Query(None),
    persona_slug: str | None = Query(None),
    audience_id: str | None = Query(None),
    audience_slug: str | None = Query(None),
):
    """Return recent messages without status filtering."""
    resolved_persona_id, lead_refs = _resolve_scope_lead_refs(
        request,
        persona_id=persona_id,
        persona_slug=persona_slug,
        audience_id=audience_id,
        audience_slug=audience_slug,
    )
    if resolved_persona_id and lead_refs is not None:
        return supabase_client.get_recent_messages(
            hours=hours,
            limit=500,
            persona_id=resolved_persona_id,
            lead_refs=lead_refs,
        )
    if persona_id:
        auth_service.assert_persona_access(request, persona_id=persona_id)
        return supabase_client.get_recent_messages(
            hours=hours,
            limit=500,
            persona_id=persona_id,
        )
    if auth_service.is_admin(auth_service.current_user(request)):
        return supabase_client.get_recent_messages(
            hours=hours,
            limit=500,
            persona_id=None,
        )
    rows: list[dict] = []
    for pid in auth_service.allowed_persona_ids(request):
        rows.extend(
            supabase_client.get_recent_messages(
                hours=hours,
                limit=500,
                persona_id=pid,
            )
        )
    rows.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return rows[:500]
