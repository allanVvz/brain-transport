# -*- coding: utf-8 -*-
"""Local OCR adapters â€” paddle, easy, tesseract, mock.

Selection order via env `ASSET_OCR_BACKEND` (comma list). The first adapter
whose `available()` returns True is used. Falls back to mock so the pipeline
always returns a structured result.
"""
from __future__ import annotations

import io
import logging
import os
from typing import Callable, Optional

from .schemas import OcrResult

logger = logging.getLogger("asset_pipeline.ocr")

_AI_FALLBACK_TEXT_THRESHOLD = 8
_AI_FALLBACK_CONFIDENCE_THRESHOLD = 0.45

_MOCK_TEXT = "(OCR ainda nao habilitado â€” fallback IA quando necessario)"


# â”€â”€ Adapters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _try_paddle(file_bytes: bytes) -> Optional[tuple[str, float]]:
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception:
        return None
    try:
        ocr = PaddleOCR(use_angle_cls=False, lang="pt", show_log=False)
        # PaddleOCR needs a numpy array or path; lazy-import numpy + Pillow.
        from PIL import Image
        import numpy as np
        img = np.array(Image.open(io.BytesIO(file_bytes)).convert("RGB"))
        result = ocr.ocr(img, cls=False)
        lines: list[str] = []
        confidences: list[float] = []
        for page in result or []:
            for row in page or []:
                if not row or len(row) < 2:
                    continue
                txt = (row[1][0] or "").strip()
                conf = float(row[1][1] or 0.0)
                if txt:
                    lines.append(txt)
                    confidences.append(conf)
        text = "\n".join(lines)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return (text, avg_conf)
    except Exception as exc:
        logger.warning("paddle OCR failed: %s", exc)
        return None


def _try_easyocr(file_bytes: bytes) -> Optional[tuple[str, float]]:
    try:
        import easyocr  # type: ignore
    except Exception:
        return None
    try:
        reader = easyocr.Reader(["pt", "en"], gpu=False, verbose=False)
        result = reader.readtext(file_bytes, detail=1, paragraph=False)
        lines = [row[1] for row in result if row and len(row) > 1]
        confidences = [float(row[2]) for row in result if row and len(row) > 2]
        text = "\n".join(lines)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return (text, avg_conf)
    except Exception as exc:
        logger.warning("easyocr failed: %s", exc)
        return None


def _try_tesseract(file_bytes: bytes) -> Optional[tuple[str, float]]:
    try:
        import pytesseract  # type: ignore
        from PIL import Image
    except Exception:
        return None
    try:
        img = Image.open(io.BytesIO(file_bytes))
        text = (pytesseract.image_to_string(img, lang="por+eng") or "").strip()
        # Tesseract doesn't expose a numerical confidence cheaply; use word-density heuristic.
        conf = 0.7 if len(text) >= 24 else (0.4 if text else 0.0)
        return (text, conf)
    except Exception as exc:
        logger.warning("tesseract failed: %s", exc)
        return None


def _try_mock(file_bytes: bytes) -> Optional[tuple[str, float]]:
    return (_MOCK_TEXT, 0.0)


_ADAPTERS: dict[str, Callable[[bytes], Optional[tuple[str, float]]]] = {
    "paddle":    _try_paddle,
    "easy":      _try_easyocr,
    "tesseract": _try_tesseract,
    "mock":      _try_mock,
}


def _backend_order() -> list[str]:
    raw = os.environ.get("ASSET_OCR_BACKEND", "paddle,easy,tesseract,mock")
    names = [n.strip().lower() for n in raw.split(",") if n.strip()]
    if "mock" not in names:
        names.append("mock")
    return [n for n in names if n in _ADAPTERS]


def run(file_bytes: bytes) -> OcrResult:
    """Run the configured OCR cascade. Always returns a result (mock is last)."""
    last_err: Optional[str] = None
    for name in _backend_order():
        adapter = _ADAPTERS[name]
        try:
            out = adapter(file_bytes)
        except Exception as exc:
            last_err = f"{name}: {exc}"
            continue
        if out is None:
            continue
        text, confidence = out
        text = (text or "").strip()
        needs_ai = (
            confidence < _AI_FALLBACK_CONFIDENCE_THRESHOLD
            or len(text) < _AI_FALLBACK_TEXT_THRESHOLD
            or name == "mock"
        )
        return OcrResult(
            engine=name,
            extracted_text=text,
            confidence=confidence,
            needs_ai_fallback=needs_ai,
        )
    return OcrResult(
        engine="mock",
        extracted_text=_MOCK_TEXT,
        confidence=0.0,
        needs_ai_fallback=True,
        error=last_err,
    )

