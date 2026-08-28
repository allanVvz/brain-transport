# -*- coding: utf-8 -*-
"""Filename / title renamer.

Heuristic first (no model call). When the heuristic produces less than 3
significant tokens, falls back to a cheap text model via model_router.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

from .schemas import AssetKind, RenameResult

logger = logging.getLogger("asset_pipeline.renamer")

_STOPWORDS = {
    "de", "da", "do", "das", "dos", "e", "a", "o", "para", "com", "em", "um",
    "uma", "the", "and", "or", "of", "to", "for", "by", "this", "that",
    "img", "image", "imagem", "foto", "photo", "screen", "screenshot",
}

_ASSET_FUNCTION_BY_KIND: dict[str, str] = {
    "image_screenshot": "visual_reference",
    "image_product":    "product_reference",
    "image_document":   "text_reference",
    "image_social":     "campaign_reference",
    "image_other":      "visual_reference",
    "pdf":              "text_reference",
    "text":             "text_reference",
    "markdown":         "text_reference",
    "video":            "campaign_reference",
    "unknown":          "visual_reference",
}


def _slugify(value: str, max_len: int = 60) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[áàâãä]", "a", value)
    value = re.sub(r"[éèêë]", "e", value)
    value = re.sub(r"[íìîï]", "i", value)
    value = re.sub(r"[óòôõö]", "o", value)
    value = re.sub(r"[úùûü]", "u", value)
    value = re.sub(r"[ç]", "c", value)
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    if len(value) > max_len:
        value = value[:max_len].rstrip("-")
    return value or "asset"


def _significant_tokens(*parts: str) -> list[str]:
    tokens: list[str] = []
    for part in parts:
        for tok in re.split(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9]+", part or ""):
            t = tok.strip().lower()
            if not t or t in _STOPWORDS or len(t) < 3:
                continue
            if t not in tokens:
                tokens.append(t)
    return tokens


def _heuristic(
    persona_slug: Optional[str],
    branch_label: Optional[str],
    kind: AssetKind,
    extracted_text: str,
    visual_summary: str,
    original_filename: str,
) -> RenameResult:
    persona_token = _slugify(persona_slug or "", max_len=24) if persona_slug else ""
    branch_token = _slugify(branch_label or "", max_len=24) if branch_label else ""
    main_tokens = _significant_tokens(visual_summary, extracted_text, original_filename)[:5]
    suffix = "-".join(main_tokens) if main_tokens else _slugify(kind, max_len=24)

    parts = [p for p in (persona_token, branch_token, suffix) if p]
    slug = _slugify("-".join(parts), max_len=80)

    ext = ""
    if original_filename and "." in original_filename:
        ext = "." + original_filename.rsplit(".", 1)[-1].lower()

    title_seed = visual_summary.strip() or extracted_text.strip().splitlines()[0] if extracted_text.strip() else ""
    if not title_seed:
        title_seed = original_filename.rsplit(".", 1)[0] if original_filename else "Asset"
    title = title_seed[:80].strip().rstrip(".") or "Asset"

    return RenameResult(
        filename=f"{slug}{ext}",
        title=title,
        slug=slug,
        asset_function=_ASSET_FUNCTION_BY_KIND.get(kind, "visual_reference"),
        tags=main_tokens[:6],
        suggested_parent_slug=branch_token or None,
        used_model=False,
    )


def _model_refine(
    initial: RenameResult,
    persona_slug: Optional[str],
    branch_label: Optional[str],
    extracted_text: str,
    visual_summary: str,
    openai_api_key: Optional[str] = None,
) -> RenameResult:
    """Call a cheap text model to refine title/slug/tags. Best effort."""
    try:
        from services.model_router import ModelRouter, get_router
        router = ModelRouter(openai_api_key=openai_api_key) if openai_api_key else get_router()
        prompt = (
            "Voce e um renomeador de assets. Receba o contexto e devolva JSON estrito.\n"
            f"persona_slug: {persona_slug or '-'}\n"
            f"branch: {branch_label or '-'}\n"
            f"extracted_text: {extracted_text[:400]}\n"
            f"visual_summary: {visual_summary[:200]}\n"
            f"filename_atual: {initial.filename}\n"
            "Responda em JSON com: title (curto pt-BR), slug (kebab-case), tags (lista ate 5).\n"
            "Apenas o JSON, sem prefixo."
        )
        raw = router.cheap_text(prompt, max_tokens=200) or ""
    except Exception as exc:
        logger.warning("renamer model refine failed: %s", exc)
        return initial

    try:
        import json
        # Strip code fences if present
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.strip("`")
            if "\n" in clean:
                clean = clean.split("\n", 1)[1]
        data = json.loads(clean)
    except Exception:
        return initial

    title = (data.get("title") or initial.title).strip()[:80]
    slug = _slugify((data.get("slug") or initial.slug), max_len=80)
    tags = data.get("tags") or initial.tags
    if isinstance(tags, list):
        tags = [str(t).strip().lower() for t in tags if str(t).strip()][:6]
    else:
        tags = initial.tags

    ext = ""
    if initial.filename and "." in initial.filename:
        ext = "." + initial.filename.rsplit(".", 1)[-1].lower()

    return RenameResult(
        filename=f"{slug}{ext}",
        title=title,
        slug=slug,
        asset_function=initial.asset_function,
        tags=tags,
        suggested_parent_slug=initial.suggested_parent_slug,
        used_model=True,
        model_used=os.environ.get("ASSET_RENAME_MODEL", "gpt-4o-mini"),
    )


def run(
    persona_slug: Optional[str],
    branch_label: Optional[str],
    kind: AssetKind,
    extracted_text: str,
    visual_summary: str,
    original_filename: str,
    *,
    allow_model_refine: bool = True,
    openai_api_key: Optional[str] = None,
) -> RenameResult:
    initial = _heuristic(persona_slug, branch_label, kind, extracted_text, visual_summary, original_filename)

    # Skip model refine if heuristic already has enough signal or env disables it.
    if not allow_model_refine or os.environ.get("ASSET_RENAME_DISABLE_MODEL") == "1":
        return initial
    if len(initial.tags) >= 3 and len(initial.title) >= 12:
        return initial

    return _model_refine(initial, persona_slug, branch_label, extracted_text, visual_summary, openai_api_key=openai_api_key)
