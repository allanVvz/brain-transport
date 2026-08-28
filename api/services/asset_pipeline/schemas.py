# -*- coding: utf-8 -*-
"""Dataclasses + enums for the asset reading pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

ReadingType = Literal[
    "classification", "ocr", "ai_fallback", "pdf_text", "video_mock", "rename",
    "transcription",
]
ReadingStatus = Literal["pending", "completed", "partial", "mocked", "failed"]
AssetKind = Literal[
    "image_screenshot",
    "image_product",
    "image_document",
    "image_social",
    "image_other",
    "pdf",
    "text",
    "markdown",
    "video",
    "audio",
    "unknown",
]
UploadContext = Literal[
    "sofia_chat", "create_sidebar", "asset_card", "imported", "whatsapp_inbound"
]


@dataclass
class AssetPipelineContext:
    persona_id: Optional[str]
    persona_slug: Optional[str] = None
    session_id: Optional[str] = None
    upload_context: UploadContext = "sofia_chat"
    original_filename: str = "upload"
    mime: Optional[str] = None
    branch_hint: Optional[str] = None  # parent slug
    branch_label: Optional[str] = None
    asset_function: Optional[str] = None
    openai_api_key: Optional[str] = None


@dataclass
class ClassificationResult:
    kind: AssetKind
    needs_ocr: bool
    has_text_estimate: bool
    confidence: float
    width: Optional[int] = None
    height: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class OcrResult:
    engine: str
    extracted_text: str
    confidence: float
    needs_ai_fallback: bool
    error: Optional[str] = None


@dataclass
class PdfTextResult:
    page_count: int
    extracted_text: str
    error: Optional[str] = None


@dataclass
class AiFallbackResult:
    extracted_text: str
    visual_summary: str
    model_used: str
    error: Optional[str] = None


@dataclass
class TranscriptionResult:
    transcript: str
    language: Optional[str]
    duration_seconds: Optional[float]
    model_used: str
    error: Optional[str] = None


@dataclass
class RenameResult:
    filename: str
    title: str
    slug: str
    asset_function: str
    tags: list[str]
    suggested_parent_slug: Optional[str]
    used_model: bool = False
    model_used: Optional[str] = None


@dataclass
class AssetReadingBundle:
    classification: ClassificationResult
    ocr: Optional[OcrResult]
    ai_fallback: Optional[AiFallbackResult]
    pdf_text: Optional[PdfTextResult]
    video_mock: Optional[dict[str, Any]]
    rename: RenameResult
    reading_status: ReadingStatus
    transcription: Optional[TranscriptionResult] = None
    extracted_text: str = ""
    visual_summary: str = ""
    rows_to_persist: list[dict[str, Any]] = field(default_factory=list)

    def to_summary(self) -> dict[str, Any]:
        """Compact dict for chat-context and frontend consumption."""
        return {
            "reading_status": self.reading_status,
            "kind": self.classification.kind,
            "needs_ocr": self.classification.needs_ocr,
            "ocr_engine": self.ocr.engine if self.ocr else None,
            "ocr_confidence": self.ocr.confidence if self.ocr else None,
            "needs_ai_fallback": self.ocr.needs_ai_fallback if self.ocr else False,
            "ai_fallback_used": self.ai_fallback is not None and not self.ai_fallback.error,
            "ai_fallback_model": self.ai_fallback.model_used if self.ai_fallback else None,
            "extracted_text": self.extracted_text or "",
            "visual_summary": self.visual_summary or "",
            "pdf_pages": self.pdf_text.page_count if self.pdf_text else None,
            "video_reading_mocked": bool(self.video_mock),
            "transcript": self.transcription.transcript if self.transcription else None,
            "transcript_language": self.transcription.language if self.transcription else None,
            "rename": {
                "filename": self.rename.filename,
                "title": self.rename.title,
                "slug": self.rename.slug,
                "asset_function": self.rename.asset_function,
                "tags": self.rename.tags,
                "suggested_parent_slug": self.rename.suggested_parent_slug,
            },
        }
