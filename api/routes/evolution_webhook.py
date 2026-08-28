from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from services import auth_service, media_ingest, supabase_client
from services.whatsapp_providers import get_provider

router = APIRouter(prefix="/webhooks/evolution", tags=["evolution-webhook"])
logger = logging.getLogger(__name__)
MAX_WEBHOOK_BYTES = int(os.environ.get("EVOLUTION_WEBHOOK_MAX_BYTES", str(2 * 1024 * 1024)))
IGNORED_SUFFIXES = ("@g.us", "@broadcast", "@newsletter")


def _mask(contact: str | None) -> str | None:
    if not contact:
        return None
    return f"{contact[:3]}******{contact[-4:]}" if len(contact) > 7 else "***"


def _normalize_phone(value: str | None) -> str:
    return re.sub(r"\D", "", value or "")


def _allowed(binding: dict, contact: str | None) -> bool:
    metadata = binding.get("metadata") or {}
    # Mirrors api/routes/whatsapp.py::_allowed. "active" is the default
    # since no current Evolution binding-creation path (portal.py,
    # configure_whatsapp_hotfix_bindings.py) sets mode explicitly.
    mode = metadata.get("mode", "active")
    if mode == "active":
        return True
    if mode != "test_allowlist":
        return False
    allowed = {_normalize_phone(str(x)) for x in metadata.get("allowlist", [])}
    return _normalize_phone(contact) in allowed


def _callback_token(binding_id: str) -> str:
    secret = (
        os.environ.get("EVOLUTION_WEBHOOK_HMAC_SECRET")
        or os.environ.get("AI_BRAIN_AUTH_SECRET")
        or ""
    )
    if not secret:
        raise HTTPException(503, "Webhook secret not configured.")
    return auth_service._b64encode(
        hmac.new(secret.encode(), binding_id.encode(), hashlib.sha256).digest()
    )


def _safe_event_id(binding_id: str, event: dict) -> str:
    stable = json.dumps({
        "binding": binding_id,
        "event": event.get("event_type"),
        "message": event.get("external_message_id"),
        "contact": event.get("external_contact_id"),
    }, sort_keys=True)
    return hashlib.sha256(stable.encode()).hexdigest()


@router.post("/{binding_id}")
async def evolution_webhook(binding_id: str, request: Request):
    callback_token = request.headers.get("x-brain-webhook-token") or ""
    expected = _callback_token(binding_id)
    if not hmac.compare_digest(callback_token.encode(), expected.encode()):
        raise HTTPException(401, "Invalid webhook token.")
    length = request.headers.get("content-length")
    if length and int(length) > MAX_WEBHOOK_BYTES:
        raise HTTPException(413, "Payload too large.")
    raw = await request.body()
    if len(raw) > MAX_WEBHOOK_BYTES:
        raise HTTPException(413, "Payload too large.")
    try:
        payload = json.loads(raw or b"{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "Invalid JSON.") from exc

    binding = supabase_client.get_workflow_binding_by_id(binding_id)
    if not binding or binding.get("provider") != "evolution_baileys" or not binding.get("active"):
        raise HTTPException(404, "Binding not found.")
    provider = get_provider("evolution_baileys")
    events = provider.normalize_webhook(payload)
    accepted = 0
    duplicate = 0
    ignored = 0
    for event in events:
        instance = str(event.get("instance") or "")
        if instance and instance != binding.get("provider_instance_key"):
            raise HTTPException(403, "Instance mismatch.")
        event_type = event.get("event_type")
        if event_type in {"CONNECTION_UPDATE", "QRCODE_UPDATED"}:
            event_raw = event.get("raw") or {}
            raw_state = str(event_raw.get("state") or event_raw.get("status") or "")
            normalized = {
                "open": "connected", "connected": "connected",
                "close": "disconnected", "disconnected": "disconnected",
                "connecting": "connecting",
            }.get(raw_state.lower(), "qr_ready" if event_type == "QRCODE_UPDATED" else "connecting")
            update = {
                "last_connection_at": (
                    datetime.now(timezone.utc).isoformat()
                    if normalized == "connected"
                    else binding.get("last_connection_at")
                ),
            }
            if not (binding.get("metadata") or {}).get("safety_paused"):
                update["connection_status"] = normalized
            supabase_client.update_workflow_binding(binding_id, update)
            if event_type == "CONNECTION_UPDATE":
                status_code = event_raw.get("statusCode") or event_raw.get("status_code")
                reason = event_raw.get("reason") or event_raw.get("disconnectReason") or event_raw.get("disconnect_reason")
                logger.warning(
                    "Evolution connection update binding=%s state=%s status_code=%s reason=%s",
                    binding_id,
                    raw_state[:40],
                    str(status_code)[:40],
                    str(reason)[:160],
                )
                # Confirmed live 2026-08-06/07: a channel dropped silently
                # with no visible trace in Settings > Logs > Auditoria,
                # because this only ever reached the raw function logger.
                # Mirroring the reason/status code into system_events (same
                # shape as the whatsapp.inbound_ignored event below) makes
                # disconnects investigable from the product itself.
                supabase_client.insert_event({
                    "event_type": "whatsapp.connection_update",
                    "entity_type": "whatsapp",
                    "entity_id": binding_id,
                    "persona_id": binding["persona_id"],
                    "payload": {
                        "state": normalized,
                        "status_code": str(status_code)[:40] if status_code is not None else None,
                        "reason": str(reason)[:160] if reason is not None else None,
                    },
                    "level": "warning" if normalized == "disconnected" else "info",
                }, source="whatsapp.connection")
            ignored += 1
            continue
        if event_type not in {"MESSAGES_UPSERT", "MESSAGES_UPDATE", "SEND_MESSAGE"}:
            ignored += 1
            continue
        if event_type in {"MESSAGES_UPDATE", "SEND_MESSAGE"}:
            if event.get("external_message_id"):
                supabase_client.update_whatsapp_delivery_by_binding(
                    binding_id,
                    str(event["external_message_id"]),
                    str(event.get("status") or "sent"),
                )
            ignored += 1
            continue
        contact = str(event.get("external_contact_id") or "")
        if not contact or contact.endswith(IGNORED_SUFFIXES):
            ignored += 1
            continue
        if event.get("from_me"):
            # Sent by hand from the linked phone, bypassing the platform.
            # Record it for history (never re-dispatch â€” it already went
            # out â€” and never trigger an AI reply to the business's own
            # message).
            external_id = str(event.get("external_message_id") or "")
            if external_id:
                lead = supabase_client.ensure_channel_lead(
                    persona_id=binding["persona_id"],
                    channel_binding_id=binding_id,
                    external_contact_id=contact,
                    display_name=None,
                )
                supabase_client.insert_message({
                    "lead_id": lead["id"],
                    "direction": "outbound",
                    "sender_type": "human",
                    "content": event.get("text") or "",
                    "channel": "whatsapp",
                    "external_message_id": external_id,
                    "channel_binding_id": binding_id,
                    "correlation_id": f"evolution_phone:{binding_id}:{external_id}",
                    "metadata": {"provider": "evolution_baileys", "source": "phone_manual"},
                })
            ignored += 1
            continue
        if not _allowed(binding, contact):
            ignored += 1
            supabase_client.insert_event({
                "event_type": "whatsapp.inbound_ignored",
                "entity_type": "whatsapp",
                "entity_id": str(event.get("external_message_id") or "unknown"),
                "persona_id": binding["persona_id"],
                "payload": {"sender": _mask(contact), "binding_id": binding_id},
            }, source="whatsapp.inbound")
            continue
        external_id = str(event.get("external_message_id") or "")
        if not external_id:
            external_id = _safe_event_id(binding_id, event)
        lead = supabase_client.ensure_channel_lead(
            persona_id=binding["persona_id"],
            channel_binding_id=binding_id,
            external_contact_id=contact,
            display_name=None,
            metadata={
                "identities": {
                    "remote_jid": event.get("remote_jid"),
                    "remote_jid_alt": event.get("remote_jid_alt"),
                },
                "portal": {
                    "reply_state": "ready" if event.get("remote_jid_alt") or "@s.whatsapp.net" in contact else "awaiting_identification"
                },
            },
        )
        correlation_id = f"evolution:{binding_id}:{external_id}"
        media = event.get("media")
        # A media message is enqueued with a longer hold so the ingest worker
        # has time to download and read the file before the agent answers.
        # `text` starts as an honest placeholder â€” if the worker never lands,
        # the hold expires and the conversation continues with it rather than
        # with an empty string.
        payload_text = event.get("text") or ""
        if media:
            payload_text = payload_text or media_ingest.placeholder_text(media)
        result = supabase_client.enqueue_whatsapp_envelope(
            buffer={
                "persona_id": binding["persona_id"],
                "lead_ref": lead["id"],
                "channel_binding_id": binding_id,
                "whatsapp_phone_number_id": None,
                "external_message_id": external_id,
                "direction": "inbound",
                "payload": {
                    "text": payload_text,
                    "sender": contact,
                    **({"media": media} if media else {}),
                },
                "status": "buffered",
                "batch_key": f"{binding['persona_id']}:{lead['id']}",
                "available_at": supabase_client.debounce_available_at(
                    media_ingest.MEDIA_HOLD_SECONDS if media else 3
                ),
                "idempotency_key": (
                    f"inbound:evolution_baileys:{binding_id}:{external_id}"
                ),
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
                "external_message_id": external_id,
                "channel_binding_id": binding_id,
                "correlation_id": correlation_id,
                "metadata": {
                    "provider": "evolution_baileys",
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
                binding_id=binding_id,
            )
        accepted += 1
    return JSONResponse(
        {
            "ok": True,
            "accepted": accepted,
            "duplicate": duplicate,
            "ignored": ignored,
        },
        status_code=202,
    )

