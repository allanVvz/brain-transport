# -*- coding: utf-8 -*-
"""PDF text extraction via pypdf."""
from __future__ import annotations

import io
import logging

from .schemas import PdfTextResult

logger = logging.getLogger("asset_pipeline.pdf")


def run(file_bytes: bytes) -> PdfTextResult:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        return PdfTextResult(page_count=0, extracted_text="", error=f"pypdf unavailable: {exc}")

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = list(reader.pages or [])
        parts: list[str] = []
        for page in pages:
            try:
                parts.append((page.extract_text() or "").strip())
            except Exception:
                continue
        text = "\n\n".join(p for p in parts if p).strip()
        return PdfTextResult(page_count=len(pages), extracted_text=text)
    except Exception as exc:
        logger.warning("pypdf failed: %s", exc)
        return PdfTextResult(page_count=0, extracted_text="", error=str(exc))

