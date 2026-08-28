# -*- coding: utf-8 -*-
"""AI vision fallback — runs only when local OCR is empty/low-confidence."""
from __future__ import annotations

import logging
from typing import Optional

from .schemas import AiFallbackResult

logger = logging.getLogger("asset_pipeline.ai_fallback")


def run(file_bytes: bytes, mime: Optional[str], prior_text: str = "", openai_api_key: Optional[str] = None) -> Optional[AiFallbackResult]:
    """Try to extract text + visual summary using a vision-capable model.

    Returns None when no vision provider is configured (offline / CI).
    """
    try:
        from services.model_router import ModelRouter, get_router
    except Exception as exc:
        logger.warning("model_router import failed: %s", exc)
        return None

    prompt = (
        "Extraia o texto presente nesta imagem (transcricao fiel, sem inventar). "
        "Depois descreva o conteudo visual em UMA linha em portugues. "
        "Formato: bloco 1 = transcricao; bloco 2 (separado por linha em branco) = descricao visual."
    )
    if prior_text and len(prior_text.strip()) > 0:
        prompt += (
            f"\n\nDicas: o OCR local detectou '{prior_text[:160]}'. "
            "Refine, complete ou corrija."
        )

    try:
        router = ModelRouter(openai_api_key=openai_api_key) if openai_api_key else get_router()
        out = router.vision_extract(file_bytes, mime or "image/png", prompt=prompt)
    except Exception as exc:
        logger.warning("vision_extract raised: %s", exc)
        return AiFallbackResult(extracted_text="", visual_summary="", model_used="", error=str(exc))

    if not out:
        return None

    return AiFallbackResult(
        extracted_text=(out.get("extracted_text") or "").strip(),
        visual_summary=(out.get("visual_summary") or "").strip(),
        model_used=str(out.get("model_used") or ""),
    )
