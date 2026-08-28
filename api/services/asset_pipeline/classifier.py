# -*- coding: utf-8 -*-
"""Local image/file classifier — no external API calls.

Decides kind, whether OCR is needed, and a confidence score. Uses Pillow when
available; falls back to mime + extension heuristics otherwise.
"""
from __future__ import annotations

import io
import os
from typing import Optional

from .schemas import AssetKind, ClassificationResult

_IMAGE_MIMES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/heic", "image/heif", "image/gif", "image/svg+xml"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif", ".gif", ".svg"}
_VIDEO_MIMES = {"video/mp4", "video/quicktime", "video/webm"}
_VIDEO_EXTS = {".mp4", ".mov", ".webm"}
# WhatsApp voice notes arrive as audio/ogg with the opus codec; some Android
# builds send mp4/amr instead. The mime often carries codec parameters
# ("audio/ogg; codecs=opus"), so matching is done on the type prefix.
_AUDIO_MIMES = {"audio/ogg", "audio/opus", "audio/mpeg", "audio/mp4", "audio/amr", "audio/wav", "audio/aac", "audio/x-wav"}
_AUDIO_EXTS = {".ogg", ".opus", ".mp3", ".m4a", ".amr", ".wav", ".aac"}
_PDF_MIMES = {"application/pdf"}
_PDF_EXTS = {".pdf"}
_TEXT_MIMES = {"text/plain"}
_MD_EXTS = {".md", ".markdown"}
_TXT_EXTS = {".txt"}


def _ext_of(filename: str) -> str:
    if not filename or "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def _basic_kind(mime: str, ext: str) -> AssetKind:
    if mime in _IMAGE_MIMES or ext in _IMAGE_EXTS:
        return "image_other"
    if mime in _AUDIO_MIMES or ext in _AUDIO_EXTS or mime.startswith("audio/"):
        return "audio"
    if mime in _VIDEO_MIMES or ext in _VIDEO_EXTS:
        return "video"
    if mime in _PDF_MIMES or ext in _PDF_EXTS:
        return "pdf"
    if ext in _MD_EXTS:
        return "markdown"
    if mime in _TEXT_MIMES or ext in _TXT_EXTS:
        return "text"
    return "unknown"


def _refine_image_kind(file_bytes: bytes, filename: str) -> tuple[AssetKind, Optional[int], Optional[int], float]:
    """Try Pillow-based refinement; fall back to filename heuristics."""
    width = height = None
    confidence = 0.5
    name_lc = (filename or "").lower()

    try:
        from PIL import Image
    except Exception:  # Pillow optional
        return ("image_other", width, height, confidence)

    try:
        with Image.open(io.BytesIO(file_bytes)) as im:
            width, height = im.size
            ratio = (width / height) if height else 1.0
            # Screenshot heuristic: typical phone/desktop captures have specific ratios + height >= width-ish
            if any(token in name_lc for token in ("screenshot", "screen", "captura", "tela", "print")):
                return ("image_screenshot", width, height, 0.85)
            # Document scan heuristic: white-ish background with high aspect ratio
            if any(token in name_lc for token in ("doc", "pdf", "scan", "fatura", "boleto", "nota")):
                return ("image_document", width, height, 0.75)
            # Social heuristic: square or 9:16
            if any(token in name_lc for token in ("ig", "insta", "story", "stories", "post", "tiktok", "reel")):
                return ("image_social", width, height, 0.7)
            # Product heuristic: very tall (1:1.4+) or "produto"/"kit" in name
            if any(token in name_lc for token in ("produto", "product", "kit", "pack")):
                return ("image_product", width, height, 0.7)
            # Aspect-ratio fallbacks
            if 0.95 <= ratio <= 1.05:
                return ("image_social", width, height, 0.55)
            if ratio < 0.65:
                return ("image_product", width, height, 0.55)
            if ratio > 1.6:
                return ("image_screenshot", width, height, 0.55)
            return ("image_other", width, height, 0.5)
    except Exception:
        return ("image_other", width, height, 0.4)


def detect(file_bytes: bytes, mime: Optional[str], filename: Optional[str]) -> ClassificationResult:
    """Return a ClassificationResult for the given file."""
    mime = (mime or "").lower().strip()
    ext = _ext_of(filename or "")
    base = _basic_kind(mime, ext)

    needs_ocr = False
    has_text_estimate = False
    width = height = None
    confidence = 0.6

    if base.startswith("image"):
        kind, width, height, confidence = _refine_image_kind(file_bytes, filename or "")
        # Most images we receive in this pipeline carry text worth extracting.
        needs_ocr = kind in {"image_screenshot", "image_document", "image_social"} or kind == "image_other"
        has_text_estimate = needs_ocr
    elif base == "pdf":
        kind = "pdf"
        needs_ocr = False  # pdf_text handles it; OCR only if pypdf returns empty
        has_text_estimate = True
        confidence = 0.9
    elif base == "audio":
        kind = "audio"
        needs_ocr = False
        # The text is in the speech, recovered by transcription rather than OCR.
        has_text_estimate = True
        confidence = 0.95
    elif base == "video":
        kind = "video"
        needs_ocr = False
        has_text_estimate = False
        confidence = 0.95
    elif base in ("text", "markdown"):
        kind = base  # type: ignore[assignment]
        needs_ocr = False
        has_text_estimate = True
        confidence = 0.95
    else:
        kind = "unknown"

    return ClassificationResult(
        kind=kind,
        needs_ocr=needs_ocr,
        has_text_estimate=has_text_estimate,
        confidence=confidence,
        width=width,
        height=height,
        extra={"mime": mime, "ext": ext, "size": len(file_bytes or b"")},
    )
