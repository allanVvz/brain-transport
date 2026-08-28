"""Canonical Meta WhatsApp ingress and delivery callbacks.

n8n is intentionally only a transport: these routes persist the source event
before any downstream work and never execute a commercial prompt.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
import re
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field
from starlette.responses import PlainTextResponse

from services import event_emitter, media_ingest, supabase_client
from services.whatsapp_providers.meta import MetaWhatsAppProvider

router = APIRouter(prefix="/webhooks/whatsapp", tags=["whatsapp"])
internal_router = APIRouter(prefix="/internal/whatsapp", tags=["whatsapp"])
logger = logging.getLogger("whatsapp")
# Evolution's webhook has always had this guard (EVOLUTION_WEBHOOK_MAX_BYTES);
# Meta's had none at all. Same env var name/default so both providers share
# one documented limit.
MAX_WEBHOOK_BYTES = int(os.environ.get("EVOLUTION_WEBHOOK_MAX_BYTES", str(2 * 1024 * 1024)))


def _mask(phone: str | None) -> str | None:
    if not phone:
        return None
    return f"{phone[:3]}******{phone[-4:]}" if len(phone) > 7 else "***"


def _normalize_phone(phone: str | None) -> str:
    return re.sub(r"\D", "", phone or "")


def _verify_signature(raw: bytes, signature: str | None) -> None:
    secret = (os.getenv("META_WHATSAPP_APP_SECRET") or "").strip()
    if not secret:
        allow_unsigned = (
            os.getenv("META_WHATSAPP_ALLOW_UNSIGNED_LOCAL") or ""
        ).strip().lower() in {"1", "true", "yes"}
        environment = (os.getenv("ENVIRONMENT") or "").strip().lower()
        if allow_unsigned and environment in {"local", "test", "qa"}:
            return
        raise HTTPException(503, "Meta app secret is not configured")
    expected = "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    if not signature or not hmac.compare_digest(expected, signature):
        raise HTTPException(401, "invalid Meta signature")


def _binding(phone_number_id: str) -> dict[str, Any] | None:
    return supabase_client.get_active_workflow_binding_by_phone_number_id(phone_number_id)


def _allowed(binding: dict[str, Any], sender: str | None) -> bool:
    metadata = binding.get("metadata") or {}
    # "active" is the default for bindings that never set mode explicitly
    # (every provider except meta_cloud today); meta_cloud always sets it
    # via PUT /meta/whatsapp/personas/{slug}/binding, so this fallback is
    # never actually exercised there.
    mode = metadata.get("mode", "active")
    if mode == "active":
        return True
    if mode != "test_allowlist":
        return False
    allowed = {_normalize_phone(str(x)) for x in metadata.get("allowlist", [])}
    return _normalize_phone(sender) in allowed


def _text(message: dict[str, Any]) -> str:
    kind = message.get("type") or "text"
    if kind == "text":
        return ((message.get("text") or {}).get("body") or "").strip()
    if kind == "button":
        return ((message.get("button") or {}).get("text") or "").strip()
    if kind == "interactive":
        interactive = message.get("interactive") or {}
        return ((interactive.get("button_reply") or interactive.get("list_reply") or {}).get("title") or "").strip()
    return ""


class OutboundResult(BaseModel):
    buffer_id: str
    channel_binding_id: str
    correlation_id: str
    wamid: str | None = None
    success: bool
    execution_id: str | None = None
    error: str | None = None


async def _process_inbound(payload: dict[str, Any]) -> dict:
    accepted = duplicate = ignored = 0
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value") or {}
            metadata = value.get("metadata") or {}
            phone_number_id = metadata.get("phone_number_id")
            if not phone_number_id:
                continue
            binding = _binding(phone_number_id)
            if not binding or binding.get("provider") != "meta_cloud":
                ignored += len(value.get("messages") or [])
                continue
            contacts = {c.get("wa_id"): c.get("profile", {}).get("name") for c in value.get("contacts") or []}
            for message in value.get("messages") or []:
                sender = message.get("from")
                external_id = message.get("id")
                if not external_id:
                    ignored += 1
                    continue
                if not sender:
                    ignored += 1
                    continue
                if not _allowed(binding, sender):
                    ignored += 1
                    supabase_client.insert_event({"event_type": "whatsapp.inbound_ignored", "entity_type": "whatsapp", "entity_id": external_id or "unknown", "persona_id": binding["persona_id"], "payload": {"sender": _mask(sender), "phone_number_id": phone_number_id}}, source="whatsapp.inbound")
                    continue
                lead = supabase_client.ensure_channel_lead(
                    persona_id=binding["persona_id"],
                    channel_binding_id=binding["id"],
                    external_contact_id=sender,
                    display_name=contacts.get(sender),
                    metadata={
                        "identities": {"meta_wa_id": sender},
                        "portal": {"reply_state": "ready"},
                    },
                )
                if not lead.get("id"):
                    raise RuntimeError("failed to resolve lead for Meta inbound")
                correlation_id = f"meta:{binding['id']}:{external_id}"
                media = MetaWhatsAppProvider.describe_media(message)
                # An attachment holds dispatch until the ingest worker has
                # read it; `text` carries an honest placeholder meanwhile, so
                # a reading failure degrades the answer instead of producing
                # an empty user turn. A captioned image keeps its caption.
                payload_text = _text(message) or (media or {}).get("caption") or ""
                if media:
                    payload_text = payload_text or media_ingest.placeholder_text(media)
                result = supabase_client.enqueue_whatsapp_envelope(
                    buffer={
                        "persona_id": binding["persona_id"],
                        "lead_ref": lead["id"],
                        "channel_binding_id": binding["id"],
                        "whatsapp_phone_number_id": phone_number_id,
                        "external_message_id": external_id,
                        "direction": "inbound",
                        "payload": {
                            "type": message.get("type"),
                            "text": payload_text,
                            "sender": sender,
                            "raw": message,
                            **({"media": media} if media else {}),
                        },
                        "status": "buffered",
                        "batch_key": f"{binding['persona_id']}:{lead['id']}",
                        "available_at": supabase_client.debounce_available_at(
                            media_ingest.MEDIA_HOLD_SECONDS if media else 3
                        ),
                        "idempotency_key": f"inbound:meta_cloud:{binding['id']}:{external_id}",
                        "correlation_id": correlation_id,
                    },
                    message={
                        "lead_id": lead["id"],
                        "role": "user",
                        "content": payload_text,
                        "direction": "inbound",
                        "status": "buffered",
                        "channel": "whatsapp",
                        "sender_id": external_id,
                        "whatsapp_phone_number_id": phone_number_id,
                        "external_message_id": external_id,
                        "channel_binding_id": binding["id"],
                        "correlation_id": correlation_id,
                        "metadata": {
                            "type": message.get("type"),
                            "provider": "meta_cloud",
                            **({"media": media} if media else {}),
                        },
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                if result.get("deduplicated"):
                    duplicate += 1
                    continue
                if media:
                    media_ingest.register_inbound_media(
                        persona_id=binding["persona_id"],
                        lead=lead,
                        descriptor=media,
                        buffer_id=result.get("buffer_id"),
                        message_row_id=result.get("message_row_id"),
                        binding_id=binding["id"],
                    )
                accepted += 1
                event_emitter.emit("whatsapp.inbound_buffered", entity_type="lead", entity_id=str(lead.get("id") or ""), persona_id=binding["persona_id"], payload={"correlation_id": correlation_id, "sender": _mask(sender)}, source="whatsapp.inbound")
    return {"accepted": accepted, "duplicate": duplicate, "ignored": ignored}


def _process_status(payload: dict[str, Any]) -> dict:
    updated = 0
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value") or {}
            phone_number_id = (value.get("metadata") or {}).get("phone_number_id")
            binding = _binding(phone_number_id) if phone_number_id else None
            for item in (value.get("statuses") or []):
                wamid, state = item.get("id"), item.get("status")
                if wamid and state and binding:
                    supabase_client.update_whatsapp_delivery_by_binding(
                        binding["id"],
                        wamid,
                        state,
                    )
                    updated += 1
    return {"updated": updated}


async def _signed_payload(request: Request, signature: str | None) -> dict[str, Any]:
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_WEBHOOK_BYTES:
        raise HTTPException(413, "Payload too large.")
    raw = await request.body()
    if len(raw) > MAX_WEBHOOK_BYTES:
        raise HTTPException(413, "Payload too large.")
    try:
        _verify_signature(raw, signature)
    except HTTPException:
        logger.warning("whatsapp_callback_signature_rejected bytes=%d", len(raw))
        raise
    logger.info("whatsapp_callback_signature_verified bytes=%d", len(raw))
    try:
        return await request.json()
    except Exception as exc:
        raise HTTPException(400, "invalid JSON") from exc


@router.get("")
async def meta_verify(
    hub_mode: str | None = Query(None, alias="hub.mode"),
    hub_verify_token: str | None = Query(None, alias="hub.verify_token"),
    hub_challenge: str | None = Query(None, alias="hub.challenge"),
) -> PlainTextResponse:
    """Meta's public callback verification endpoint."""
    expected = (os.getenv("META_WHATSAPP_VERIFY_TOKEN") or "").strip()
    if hub_mode != "subscribe" or not expected or not hmac.compare_digest(hub_verify_token or "", expected):
        raise HTTPException(403, "invalid Meta verify token")
    return PlainTextResponse(hub_challenge or "")


@router.post("")
async def meta_callback(request: Request, x_hub_signature_256: str | None = Header(None)) -> dict:
    """Canonical Meta callback: one URL receives inbound messages and statuses."""
    payload = await _signed_payload(request, x_hub_signature_256)
    inbound = await _process_inbound(payload) if any(
        (change.get("value") or {}).get("messages")
        for entry in payload.get("entry", []) for change in entry.get("changes", [])
    ) else {"accepted": 0, "duplicate": 0, "ignored": 0}
    delivery = _process_status(payload)
    return {**inbound, **delivery}


@router.post("/inbound")
async def inbound(request: Request, x_hub_signature_256: str | None = Header(None)) -> dict:
    return await _process_inbound(await _signed_payload(request, x_hub_signature_256))


@router.post("/status")
async def status(request: Request, x_hub_signature_256: str | None = Header(None)) -> dict:
    return _process_status(await _signed_payload(request, x_hub_signature_256))


@internal_router.post("/outbound-result")
def outbound_result(body: OutboundResult, x_webhook_token: str | None = Header(None, alias="X-Webhook-Token")) -> dict:
    expected = (os.getenv("AI_BRAIN_WEBHOOK_TOKEN") or "").strip()
    if not expected:
        raise HTTPException(503, "internal webhook token is not configured")
    if not hmac.compare_digest((x_webhook_token or "").encode(), expected.encode()):
        raise HTTPException(401, "invalid webhook token")
    result = supabase_client.complete_whatsapp_outbound(
        body.buffer_id,
        binding_id=body.channel_binding_id,
        correlation_id=body.correlation_id,
        wamid=body.wamid,
        success=body.success,
        error=body.error,
        execution_id=body.execution_id,
    )
    return {
        "ok": bool(result.get("ok", True)),
        "message_id": result.get("message_id"),
        "buffer_id": result.get("buffer_id") or body.buffer_id,
        "status": result.get("status"),
        "deduplicated": bool(result.get("deduplicated")),
    }

