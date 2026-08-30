import base64
import json
import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel
from brain_contracts import CanonicalInboundEnvelope

from services import (
    auth_service,
    event_emitter,
    internal_auth,
    operator_messaging,
    runtime_client,
    supabase_client,
    validator_media,
    whatsapp_outbox,
)

router = APIRouter(prefix="/messages", tags=["messages"])
internal_router = APIRouter(prefix="/internal/v1/transport/messages", tags=["messages"])
logger = logging.getLogger("messages")


class SendMessageBody(BaseModel):
    lead_ref: int
    client_message_id: UUID
    agent_id: str | None = None
    texto: str
    sender_id: str | None = None
    nome: str | None = None


class InternalPortalMessageBody(BaseModel):
    persona_id: str
    lead_ref: int
    client_message_id: UUID
    text: str = ""
    media_base64: str | None = None
    media_mime: str | None = None
    media_filename: str | None = None


class InternalOutboundBody(BaseModel):
    lead: dict[str, Any]
    text: str
    sender_type: str = "agent"
    message_id: str
    correlation_id: str
    idempotency_key: str
    initial_status: str = "pending_send"
    metadata: dict[str, Any] | None = None
    media: dict[str, Any] | None = None
    template: dict[str, Any] | None = None
    campaign_scope: dict[str, Any] | None = None


class InternalCampaignOutboundBody(InternalOutboundBody):
    sender_type: str = "campaign"
    campaign_scope: dict[str, Any]


class InternalValidatorMediaBody(BaseModel):
    session_id: str
    persona_id: str
    lead_ref: int
    channel_binding_id: str | None = None
    filename: str
    mime: str
    content_base64: str
    idempotency_key: str


class InternalTechnicalFailureBody(BaseModel):
    lead_ref: int
    error: str


def _validator_inbound_key(inbound_id: str) -> str:
    return f"inbound:wa-validator:{inbound_id}"


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
        agent = operator_messaging.agent_metadata(body.agent_id, persona_id)

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
                "agent_id": agent.get("agent_id") if agent else None,
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


@internal_router.post("/send")
def send_portal_message_internal(
    body: InternalPortalMessageBody,
    x_webhook_token: str | None = Header(None, alias="X-Webhook-Token"),
    x_brain_actor_id: str | None = Header(None, alias="X-Brain-Actor-Id"),
) -> dict:
    internal_auth.authorize_webhook_token(x_webhook_token)
    media = None
    if body.media_base64:
        try:
            data = base64.b64decode(body.media_base64, validate=True)
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, "Midia base64 invalida") from exc
        if len(data) > 25 * 1024 * 1024:
            raise HTTPException(413, "Arquivo excede 25 MiB")
        media = {
            "data": data,
            "mime": body.media_mime or "application/octet-stream",
            "filename": body.media_filename or "arquivo",
        }
    return operator_messaging.enqueue(
        lead_ref=body.lead_ref,
        persona_id=body.persona_id,
        client_message_id=str(body.client_message_id),
        text=body.text,
        media=media,
        metadata={"actor_user_id": x_brain_actor_id},
    )


@internal_router.post("/campaign-outbound")
def enqueue_campaign_outbound_internal(
    body: InternalCampaignOutboundBody,
    x_webhook_token: str | None = Header(None, alias="X-Webhook-Token"),
) -> dict:
    """Accept one idempotent campaign command from the control plane.

    Provider selection, the delivery queue and delivery status remain owned by
    transport; campaign policy and recipient selection never move here.
    """
    internal_auth.authorize_webhook_token(x_webhook_token)
    return whatsapp_outbox.enqueue_outbound(**body.model_dump())


@internal_router.post("/prepare-outbound")
def prepare_outbound_internal(
    body: InternalOutboundBody,
    x_webhook_token: str | None = Header(None, alias="X-Webhook-Token"),
) -> dict:
    """Build a provider-safe envelope for the runtime's atomic proof commit."""
    internal_auth.authorize_webhook_token(x_webhook_token)
    return whatsapp_outbox.prepare_outbound_envelope(**body.model_dump())


@internal_router.post("/outbound")
def enqueue_outbound_internal(
    body: InternalOutboundBody,
    x_webhook_token: str | None = Header(None, alias="X-Webhook-Token"),
) -> dict:
    """Persist one idempotent runtime outbound under transport ownership."""
    internal_auth.authorize_webhook_token(x_webhook_token)
    return whatsapp_outbox.enqueue_outbound(**body.model_dump())


@internal_router.post("/validator-media")
def store_validator_media_internal(
    body: InternalValidatorMediaBody,
    x_webhook_token: str | None = Header(None, alias="X-Webhook-Token"),
) -> dict:
    internal_auth.authorize_webhook_token(x_webhook_token)
    try:
        content = base64.b64decode(body.content_base64, validate=True)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, "Midia base64 invalida") from exc
    try:
        return validator_media.store(
            **body.model_dump(exclude={"content_base64"}), content=content,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@internal_router.post("/validator-inbound")
def enqueue_validator_inbound_internal(
    body: CanonicalInboundEnvelope,
    x_webhook_token: str | None = Header(None, alias="X-Webhook-Token"),
) -> dict:
    """Persist one inert synthetic inbound under transport ownership."""
    internal_auth.authorize_webhook_token(x_webhook_token)
    if body.provider != "internal_validator":
        raise HTTPException(422, "Provider invalido para validacao interna")
    text = str(body.content.get("text") or "").strip()
    if not text or body.message_type != "text":
        raise HTTPException(422, "Inbound de validacao deve conter texto")
    try:
        lead_ref = int(body.lead_ref)
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, "lead_ref invalido") from exc
    if lead_ref <= 0:
        raise HTTPException(422, "lead_ref invalido")

    inbound_id = body.inbound_id
    return supabase_client.enqueue_whatsapp_envelope(
        buffer={
            "persona_id": str(body.persona_id),
            "lead_ref": lead_ref,
            "channel_binding_id": str(body.channel_binding_id),
            "whatsapp_phone_number_id": None,
            "external_message_id": inbound_id,
            "direction": "inbound",
            "payload": {"text": text, "sender": "wa-validator"},
            # A direct validation turn is consumed synchronously by runtime.
            "status": "waiting_human",
            "batch_key": f"{body.persona_id}:{lead_ref}",
            "idempotency_key": _validator_inbound_key(inbound_id),
            "correlation_id": body.correlation_id,
        },
        message={
            "lead_id": lead_ref,
            "role": "user",
            "content": text,
            "direction": "inbound",
            "status": "buffered",
            "channel": "whatsapp",
            "sender_id": "wa-validator",
            "external_message_id": inbound_id,
            "channel_binding_id": str(body.channel_binding_id),
            "correlation_id": body.correlation_id,
            "metadata": {
                "provider": "wa-validator",
                "contract_version": body.contract_version,
                "persona_slug": body.persona_slug,
            },
            "created_at": body.received_at.isoformat(),
        },
    )


@internal_router.post("/validator-inbound/{session_id}/{turn}/complete")
def complete_validator_inbound_internal(
    session_id: UUID,
    turn: int,
    x_webhook_token: str | None = Header(None, alias="X-Webhook-Token"),
) -> dict:
    """Terminalize only the exact synthetic inbound created by the validator."""
    internal_auth.authorize_webhook_token(x_webhook_token)
    if turn < 0:
        raise HTTPException(422, "Turno invalido")
    inbound_id = f"validator:{session_id}:{turn}"
    row = supabase_client.get_whatsapp_buffer_by_idempotency(
        _validator_inbound_key(inbound_id)
    ) or {}
    payload = row.get("payload") or {}
    if (
        not row.get("id")
        or row.get("direction") != "inbound"
        or row.get("external_message_id") != inbound_id
        or payload.get("sender") != "wa-validator"
    ):
        raise HTTPException(404, "Inbound de validacao nao encontrado")
    supabase_client.complete_whatsapp_buffer(str(row["id"]), "sent")
    return {"ok": True, "buffer_id": str(row["id"]), "status": "sent"}


@internal_router.post("/inbound/{buffer_id}/technical-failure")
def quarantine_inbound_technical_failure_internal(
    buffer_id: UUID,
    body: InternalTechnicalFailureBody,
    x_webhook_token: str | None = Header(None, alias="X-Webhook-Token"),
) -> dict:
    """Terminalize one failed inbound under transport data ownership."""
    internal_auth.authorize_webhook_token(x_webhook_token)
    row = supabase_client.get_whatsapp_buffer(str(buffer_id)) or {}
    if (
        not row.get("id")
        or row.get("direction") != "inbound"
        or int(row.get("lead_ref") or 0) != body.lead_ref
    ):
        raise HTTPException(404, "Inbound nao encontrado")
    error = body.error.strip()
    if not error:
        raise HTTPException(422, "Motivo tecnico vazio")
    supabase_client.complete_whatsapp_buffer(
        str(buffer_id), "dead_letter", error=error[:1000]
    )
    return {"ok": True, "buffer_id": str(buffer_id), "status": "dead_letter"}


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
    decorated_leads = runtime_client.decorate_leads(list(leads_by_ref.values()))
    qualification_by_ref = {
        int(lead["id"]): lead
        for lead in decorated_leads
        if lead.get("id") is not None
    }
    for row in rows:
        lead_ref = row.get("lead_ref")
        extra = qualification_by_ref.get(int(lead_ref), {}) if lead_ref is not None else {}
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
        decorated = runtime_client.decorate_leads([lead])
        is_validation = bool((decorated[0] if decorated else {}).get("validation", {}).get("is_validation"))
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
