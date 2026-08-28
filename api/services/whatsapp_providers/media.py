"""Canonical inbound-media descriptor shared by every WhatsApp provider.

Evolution (Baileys) and Meta Cloud describe an attachment with completely
different payload shapes, and each needs a different call to fetch the bytes
later. Both normalizers converge on the dict built here so that the webhook
routes, ``lead_buffer.payload.media`` and the media ingest worker only ever
speak one vocabulary.

The descriptor deliberately carries no bytes: a webhook must persist and return
fast, so the download happens later in the worker using ``fetch_ref``.
"""
from __future__ import annotations

from typing import Any, Optional

# WhatsApp's own message-type names, which both providers ultimately derive
# from. ``sticker`` is folded into ``image`` because it behaves like one
# everywhere downstream (thumbnail, vision read, gallery).
MEDIA_KINDS = ("image", "audio", "video", "document")

_MIME_PREFIX_KIND = (
    ("image/", "image"),
    ("audio/", "audio"),
    ("video/", "video"),
)

# WhatsApp voice notes arrive as audio/ogg with the opus codec; some Android
# builds send audio/mp4 or audio/amr instead. Anything not matched by prefix
# falls back to ``document``, which is the safe bucket: it is stored and shown
# but never fed to the transcriber.
_DEFAULT_KIND = "document"


def kind_for_mime(mime: Optional[str], fallback: Optional[str] = None) -> str:
    """Map a MIME type to one of ``MEDIA_KINDS``.

    ``fallback`` is the provider's own type label, used when the MIME type is
    missing or unhelpful (Meta omits it on some forwarded messages).
    """
    value = (mime or "").strip().lower()
    for prefix, kind in _MIME_PREFIX_KIND:
        if value.startswith(prefix):
            return kind
    if value == "application/pdf":
        return _DEFAULT_KIND
    hint = (fallback or "").strip().lower()
    if hint == "sticker":
        return "image"
    if hint in MEDIA_KINDS:
        return hint
    return _DEFAULT_KIND


def build_descriptor(
    *,
    provider: str,
    kind: str,
    mime: Optional[str] = None,
    filename: Optional[str] = None,
    caption: Optional[str] = None,
    size: Optional[int] = None,
    voice_note: bool = False,
    duration_seconds: Optional[int] = None,
    fetch_ref: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build the canonical descriptor stored in ``lead_buffer.payload.media``.

    ``fetch_ref`` is the provider-specific handle the ingest worker needs to
    retrieve the bytes â€” a Meta ``media_id``, or the Evolution message key.
    """
    return {
        "provider": provider,
        "kind": kind if kind in MEDIA_KINDS else _DEFAULT_KIND,
        "mime": (mime or "").strip() or None,
        "filename": (filename or "").strip() or None,
        "caption": (caption or "").strip() or None,
        "size": int(size) if isinstance(size, (int, float)) and size else None,
        # A voice note (PTT) is worth distinguishing from an attached audio
        # file: it is the case that most needs transcription, and the UI
        # labels it differently.
        "voice_note": bool(voice_note),
        "duration_seconds": int(duration_seconds) if duration_seconds else None,
        "fetch_ref": fetch_ref or {},
        "reading_status": "pending",
    }

