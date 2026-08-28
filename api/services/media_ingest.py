"""Inbound WhatsApp media: registration, attribution and text descriptors.

The design constraint that shapes this whole module: ``payload["text"]`` is the
*only* channel into both conversation runtimes
(``workers/whatsapp_dispatch_worker.py`` reads nothing else from the buffer).
So instead of teaching the runtime, n8n and the proof checker about
attachments, the extracted text is folded back into ``payload.text`` at ingest
time. Everything downstream keeps working unchanged.

That trade has one cost — reading a file takes seconds, and a webhook must
answer immediately — which is paid with a dispatch hold:

    webhook   → asset(status='reading') + buffer(available_at = now + HOLD)
    worker    → download → read → resolve_media_buffer(text) → available_at = now + 3s
    (failure) → hold expires, the conversation continues with the placeholder

A failed transcription therefore degrades the answer. It never blocks the
conversation and never produces an empty user turn.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from services import supabase_client

logger = logging.getLogger("services.media_ingest")

# How long dispatch waits for the file to be read. Sized for the worst
# realistic case: a ~60s voice note transcribed by faster-whisper `small` on
# CPU runs near real time, plus download and transcode.
MEDIA_HOLD_SECONDS = int(os.environ.get("WHATSAPP_MEDIA_HOLD_SECONDS", "45"))

# Placeholder shown while the file is being read, and kept permanently if the
# reading fails. Deliberately states only what is certain — that something was
# received — instead of inventing content.
_PLACEHOLDER = {
    "audio": "[o cliente enviou um audio]",
    "image": "[o cliente enviou uma imagem]",
    "video": "[o cliente enviou um video]",
    "document": "[o cliente enviou um documento]",
}
_VOICE_PLACEHOLDER = "[o cliente enviou um audio]"


def placeholder_text(descriptor: dict) -> str:
    """Honest stand-in used until (or instead of) a successful reading."""
    kind = str((descriptor or {}).get("kind") or "document")
    if kind == "audio" and (descriptor or {}).get("voice_note"):
        return _VOICE_PLACEHOLDER
    return _PLACEHOLDER.get(kind, _PLACEHOLDER["document"])


def describe(descriptor: dict, reading: Optional[dict]) -> tuple[str, str]:
    """Turn a finished reading into ``(text, reading_status)``.

    The text is what the operator sees in the thread and what the agent
    receives as the user's turn, so each kind is labelled explicitly — the
    agent must never mistake a transcription for something the customer typed.
    """
    descriptor = descriptor or {}
    kind = str(descriptor.get("kind") or "document")
    caption = str(descriptor.get("caption") or "").strip()
    reading = reading or {}
    transcript = str(reading.get("transcript") or "").strip()
    extracted = str(reading.get("extracted_text") or "").strip()
    summary = str(reading.get("visual_summary") or "").strip()

    if kind == "audio":
        if not transcript:
            return placeholder_text(descriptor), "failed"
        return f"[audio do cliente]: {transcript}", "completed"

    if kind == "image":
        detail = summary or extracted
        if not detail:
            # The file is stored and visible; only the machine reading failed.
            return (caption or placeholder_text(descriptor)), "failed"
        body = f"[imagem enviada pelo cliente: {detail}]"
        return (f"{caption}\n{body}" if caption else body), "completed"

    if kind == "document":
        filename = descriptor.get("filename") or "documento"
        if not extracted:
            return (caption or f"[documento: {filename}]"), "failed"
        excerpt = "\n".join(extracted.splitlines()[:20]).strip()
        body = f"[documento: {filename}]\n{excerpt}"
        return (f"{caption}\n{body}" if caption else body), "completed"

    # Video has no real reader yet (asset_pipeline.video_mock), so it stays at
    # the placeholder rather than pretending to have been watched.
    return (caption or placeholder_text(descriptor)), "partial"


def resolve_campaign_attribution(persona_id: str, lead_id: int) -> dict[str, Any]:
    """Find the campaign whose outreach opened this conversation.

    ``messages.campaign_id`` cannot carry this for inbound traffic — the
    constraint from migration 087 requires ``direction='outbound'`` — so the
    campaign is resolved through ``campaign_recipients``, which is also where
    the reply-attribution machinery already lives.

    Returns ``{}`` for an organic contact with no campaign behind it.
    """
    if not persona_id or not lead_id:
        return {}
    try:
        rows = (
            supabase_client.get_client().table("campaign_recipients")
            .select("id,campaign_id,campaign_revision_id,campaign_revision")
            .eq("persona_id", persona_id)
            .eq("lead_id", int(lead_id))
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        ).data or []
    except Exception as exc:
        logger.warning("campaign attribution lookup failed lead=%s: %s", lead_id, exc)
        return {}
    if not rows:
        return {}
    recipient = rows[0]
    return {
        "campaign_id": recipient.get("campaign_id"),
        "campaign_recipient_id": recipient.get("id"),
        "campaign_revision_id": recipient.get("campaign_revision_id"),
    }


def register_inbound_media(
    *,
    persona_id: str,
    lead: dict,
    descriptor: dict,
    buffer_id: Optional[str],
    message_row_id: Optional[Any],
    binding_id: Optional[str] = None,
) -> Optional[dict]:
    """Create the ``assets`` row for a just-received attachment.

    Runs inside the webhook, so it must stay cheap and must never raise: a
    failure here costs the file, but dropping the whole webhook would cost the
    message itself. Bytes are fetched later by the media ingest worker.
    """
    lead_id = lead.get("id")
    if not lead_id:
        return None
    attribution = resolve_campaign_attribution(persona_id, int(lead_id))
    enriched = {
        **descriptor,
        "buffer_id": buffer_id,
        "binding_id": binding_id,
        **attribution,
    }
    try:
        asset = supabase_client.insert_inbound_media_asset(
            persona_id=persona_id,
            lead_id=int(lead_id),
            message_id=int(message_row_id) if message_row_id is not None else None,
            descriptor=enriched,
            campaign_id=attribution.get("campaign_id"),
            campaign_recipient_id=attribution.get("campaign_recipient_id"),
        )
    except Exception as exc:
        logger.error(
            "inbound media registration failed lead=%s kind=%s: %s",
            lead_id, descriptor.get("kind"), exc,
        )
        return None
    if asset is None:
        # Provider retry for a message we already registered.
        logger.info("inbound media already registered message_row_id=%s", message_row_id)
    elif message_row_id is not None and asset.get("id"):
        try:
            supabase_client.link_inbound_media_asset_to_message(
                int(message_row_id), str(asset["id"]),
            )
        except Exception as exc:
            # The asset row remains the canonical relationship through
            # assets.message_id. The messages read path hydrates this link too,
            # so a transient projection failure must not lose the attachment.
            logger.warning(
                "inbound media message link failed message_row_id=%s asset=%s: %s",
                message_row_id, asset.get("id"), exc,
            )
    return asset
