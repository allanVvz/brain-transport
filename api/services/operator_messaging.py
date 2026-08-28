"""Exactly-once operator outbound owned by transport."""

from __future__ import annotations

import uuid

from fastapi import HTTPException

from services import event_emitter, supabase_client, whatsapp_outbox


def agent_metadata(agent_id: str, persona_id: str) -> dict:
    result = (
        supabase_client.get_client()
        .table("agents")
        .select("id,persona_id,bot_name")
        .eq("id", agent_id)
        .maybe_single()
        .execute()
    )
    agent = getattr(result, "data", None) or {}
    if not agent or str(agent.get("persona_id")) != str(persona_id):
        raise HTTPException(403, "Agente nao pertence a persona do lead")
    return {"agent_id": agent.get("id"), "bot_name": agent.get("bot_name")}


def enqueue(
    *,
    lead_ref: int,
    persona_id: str,
    client_message_id: str,
    text: str,
    media: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    lead = supabase_client.get_lead_by_ref(lead_ref) or {}
    if not lead:
        raise HTTPException(404, "Lead nao encontrado")
    if str(lead.get("persona_id") or "") != str(persona_id):
        raise HTTPException(403, "Lead fora da persona autorizada")
    if not text.strip() and not media:
        raise HTTPException(400, "Mensagem vazia")

    binding = whatsapp_outbox.resolve_lead_binding(lead)
    message_id = f"manual:{client_message_id}"
    existing = supabase_client.get_whatsapp_buffer_by_idempotency(message_id)
    if existing:
        if (
            existing.get("lead_ref") != lead["id"]
            or existing.get("channel_binding_id") != binding["id"]
        ):
            raise HTTPException(409, "client_message_id ja pertence a outra mensagem")
        return {
            "ok": True,
            "message_id": message_id,
            "status": existing.get("status") or "pending_send",
            "buffer_id": existing["id"],
            "deduplicated": True,
        }

    outbox_media = None
    media_metadata = None
    if media:
        data = media.get("data") or b""
        mime = str(media.get("mime") or "application/octet-stream")
        filename = str(media.get("filename") or "arquivo")
        bucket = "message-media"
        if not supabase_client.ensure_bucket(bucket, public=False):
            raise HTTPException(503, "Storage de mensagens indisponivel")
        path = f"{persona_id}/{lead_ref}/{uuid.uuid4().hex}-{filename}"
        supabase_client.upload_to_storage(bucket, path, data, mime)
        media_metadata = {
            "bucket": bucket,
            "path": path,
            "mime": mime,
            "filename": filename,
        }
        outbox_media = dict(media_metadata)

    result = whatsapp_outbox.enqueue_outbound(
        lead=lead,
        text=text.strip(),
        sender_type="human",
        message_id=message_id,
        correlation_id=message_id,
        idempotency_key=message_id,
        metadata={
            **(metadata or {}),
            "source": "portal",
            "media": media_metadata,
            "client_message_id": client_message_id,
        },
        media=outbox_media,
    )
    if not result.get("deduplicated"):
        event_emitter.emit(
            "message.new",
            entity_type="message",
            entity_id=message_id,
            persona_id=persona_id,
            payload={
                "lead_ref": lead_ref,
                "sender_type": "human",
                "buffer_id": result["buffer_id"],
            },
            source="operator_messaging.enqueue",
        )
    return {
        "ok": True,
        "message_id": result.get("message_id") or message_id,
        "status": result.get("status") or "pending_send",
        "buffer_id": result["buffer_id"],
        "deduplicated": bool(result.get("deduplicated")),
    }
