"""Persistence boundary for WA Validator synthetic inbound media."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from services import control_plane_client, media_ingest, supabase_client


def store(
    *,
    session_id: str,
    persona_id: str,
    lead_ref: int,
    channel_binding_id: str | None,
    filename: str,
    mime: str,
    content: bytes,
    idempotency_key: str,
) -> dict[str, Any]:
    safe_name = Path(str(filename or "arquivo").replace("\\", "/")).name[:180]
    mime = str(mime or "application/octet-stream").split(";", 1)[0].lower()
    if not content or len(content) > 20 * 1024 * 1024:
        raise ValueError("O arquivo deve ter entre 1 byte e 20 MB.")
    if mime == "image/png" and not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("O conteudo nao corresponde a um PNG valido.")
    if mime == "image/jpeg" and not content.startswith(b"\xff\xd8\xff"):
        raise ValueError("O conteudo nao corresponde a um JPEG valido.")
    if mime == "image/webp" and not (content.startswith(b"RIFF") and content[8:12] == b"WEBP"):
        raise ValueError("O conteudo nao corresponde a um WebP valido.")
    if mime == "application/pdf" and not content.startswith(b"%PDF-"):
        raise ValueError("O conteudo nao corresponde a um PDF valido.")
    if mime not in {"image/png", "image/jpeg", "image/webp", "application/pdf"}:
        raise ValueError("Formato de teste nao suportado. Use PNG, JPEG, WebP ou PDF.")
    if not re.fullmatch(r"[A-Za-z0-9._:-]{8,160}", idempotency_key or ""):
        raise ValueError("Chave de idempotencia invalida.")
    checksum = hashlib.sha256(content).hexdigest()
    client = supabase_client.get_client()
    existing_rows = (
        client.table("assets")
        .select("*")
        .eq("persona_id", persona_id)
        .eq("lead_id", int(lead_ref))
        .eq("upload_context", "whatsapp_inbound")
        .contains("metadata", {"validator_media": {"idempotency_key": idempotency_key}})
        .limit(1)
        .execute()
    ).data or []
    asset = existing_rows[0] if existing_rows else None
    if asset:
        recorded = ((asset.get("metadata") or {}).get("validator_media") or {}).get("sha256")
        if recorded and recorded != checksum:
            raise ValueError("A chave de idempotencia ja foi usada por outro arquivo.")

    kind = "image" if mime.startswith("image/") else "document"
    descriptor = {
        "kind": kind,
        "mime": mime,
        "filename": safe_name,
        "size": len(content),
        "reading_status": "completed",
        "validator_direct": True,
    }
    attribution = media_ingest.resolve_campaign_attribution(persona_id, int(lead_ref))
    if not asset:
        asset = supabase_client.insert_asset({
            "persona_id": persona_id,
            "lead_id": int(lead_ref),
            "campaign_id": attribution.get("campaign_id"),
            "campaign_recipient_id": attribution.get("campaign_recipient_id"),
            "type": "image" if kind == "image" else "pdf",
            "name": safe_name,
            "source": "whatsapp",
            "upload_context": "whatsapp_inbound",
            "status": "reading",
            "mime_type": mime,
            "file_size": len(content),
            "original_filename": safe_name,
            "metadata": {
                "media": descriptor,
                "direction": "inbound",
                "reading_status": "completed",
                "validation_status": "not_applicable",
                "upload_context": "whatsapp_inbound",
                "rag_eligible": False,
                "validator_media": {
                    "session_id": session_id,
                    "idempotency_key": idempotency_key,
                    "sha256": checksum,
                },
            },
        })

    asset_id = str(asset.get("id") or "")
    if not asset_id:
        raise RuntimeError("Nao foi possivel registrar o asset de validacao.")
    extension = ".pdf" if mime == "application/pdf" else {
        "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp",
    }[mime]
    storage_path = f"{persona_id}/{lead_ref}/{asset_id}-validator{extension}"
    supabase_client.upload_to_storage(
        supabase_client.WHATSAPP_MEDIA_BUCKET, storage_path, content, mime,
    )
    asset = supabase_client.update_asset(asset_id, {
        "status": "ready",
        "storage_bucket": supabase_client.WHATSAPP_MEDIA_BUCKET,
        "storage_path": storage_path,
        "file_size": len(content),
    }) or asset

    external_id = f"validator-media:{session_id}:{idempotency_key}"
    text = f"[imagem de teste recebida: {safe_name}]" if kind == "image" else f"[documento: {safe_name}]"
    supabase_client.insert_message({
        "lead_id": int(lead_ref),
        "role": "user",
        "content": text,
        "direction": "inbound",
        "status": "delivered",
        "channel": "whatsapp",
        "sender_id": external_id,
        "external_message_id": external_id,
        "channel_binding_id": channel_binding_id,
        "correlation_id": external_id,
        "metadata": {
            "asset_id": asset_id,
            "media": descriptor,
            "validation": {"is_validation": True, "session_id": session_id},
        },
    })
    message_rows = (
        client.table("messages")
        .select("id,external_message_id,direction,created_at")
        .eq("lead_id", int(lead_ref))
        .eq("external_message_id", external_id)
        .limit(1)
        .execute()
    ).data or []
    message = message_rows[0] if message_rows else {}
    if message.get("id") and not asset.get("message_id"):
        asset = supabase_client.update_asset(asset_id, {"message_id": message["id"]}) or asset

    try:
        graph_attachment = control_plane_client.attach_inbound_asset(asset_id)
    except Exception as exc:
        graph_attachment = {
            "attached": False,
            "status": "error",
            "reason": type(exc).__name__,
        }

    return {
        "session_id": session_id,
        "lead_ref": int(lead_ref),
        "asset": {
            "id": asset_id,
            "filename": safe_name,
            "mime_type": mime,
            "file_size": len(content),
            "sha256": checksum,
            "status": "ready",
            "media_url": f"/assets/{asset_id}/media",
        },
        "message": message,
        "graph_attachment": graph_attachment,
        "idempotent": bool(existing_rows),
        "outbound_enqueued": False,
    }
