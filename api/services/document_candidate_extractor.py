"""Extract product/knowledge candidates from a pasted or uploaded document.

Complements catalog_crawler.py's URL-based extraction with a document-text
path, producing the exact same "product_candidates" envelope shape
(catalog_crawler.crawl_catalog_url_tool's return dict) so both sources plug
into kb_intake_service.attach_crawler_capture / build_full_tree_plan_from_
session unchanged -- no new deterministic-tree-building code needed.

Two extraction strategies, tried in order:
1. Structured markdown pattern: "### Title" headings followed by
   "- **Field:** value" bullet lines (description/tamanho/preco/copy/...) --
   the shape used by the real Tock Fatal catalog document this session, and
   a common shape for operator-authored catalog documents. Deterministic, no
   LLM call, free, and directly validated against a real production case.
2. LLM fallback for anything less structured: asks the model to emit a JSON
   array of candidates from the raw text, never inventing a price the
   document doesn't state.
"""
from __future__ import annotations

import json
import re
from typing import Any

from services.model_router import ModelRouter, ModelRouterError

_HEADING_RE = re.compile(r"^#{2,4}\s+(.+)$", re.M)
_FIELD_RE = re.compile(r"^\s*-\s*\*\*([^:*]+):\*\*\s*(.+)$", re.M)
_PRICE_RE = re.compile(r"R\$\s*([\d.]+,\d{2})")
_TITLE_PRICE_SUFFIX_RE = re.compile(r"\s*[â€”-]\s*R\$\s*[\d.,]+.*$")


def _parse_price(text: str | None) -> float | None:
    match = _PRICE_RE.search(text or "")
    if not match:
        return None
    raw = match.group(1).replace(".", "").replace(",", ".")
    try:
        return round(float(raw), 2)
    except ValueError:
        return None


def _structured_markdown_candidates(text: str) -> list[dict[str, Any]]:
    headings = list(_HEADING_RE.finditer(text))
    if len(headings) < 2:
        return []
    candidates: list[dict[str, Any]] = []
    for index, match in enumerate(headings):
        title = match.group(1).strip()
        start = match.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        block = text[start:end]
        fields = {
            field.strip().lower(): value.strip()
            for field, value in _FIELD_RE.findall(block)
        }
        description = fields.get("descricao") or fields.get("descriÃ§Ã£o") or fields.get("copy")
        price = _parse_price(fields.get("preco") or fields.get("preÃ§o") or title)
        colors_raw = fields.get("cores") or ""
        candidates.append({
            "title": _TITLE_PRICE_SUFFIX_RE.sub("", title).strip() or title,
            "description": description,
            "prices": [price] if price is not None else [],
            "colors": [c.strip() for c in colors_raw.split(",") if c.strip()],
            "product_type": None,
            "tags": [],
            "source": "document_structured",
            "raw_fields": fields,
        })
    return candidates


def _llm_candidates(text: str, *, model: str) -> list[dict[str, Any]]:
    router = ModelRouter()
    prompt = (
        "Extraia uma lista de itens de catalogo (produtos, planos, pacotes ou "
        "servicos) deste documento. Responda SOMENTE um array JSON, sem texto "
        "fora do array, cada item no formato: "
        '{"title": str, "description": str|null, "prices": [number], '
        '"colors": [str], "product_type": str|null, "tags": [str]}. '
        "Nao invente preco: se o documento nao informa preco para o item, "
        "prices deve ser []. Documento:\n\n" + text[:12000]
    )
    try:
        raw = router.messages_create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            system="Voce extrai dados estruturados de documentos comerciais. Responda so com JSON valido.",
            max_tokens=4000,
        )
    except ModelRouterError:
        return []
    match = re.search(r"\[.*\]", raw if isinstance(raw, str) else "", re.S)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return []
    candidates: list[dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not str(item.get("title") or "").strip():
            continue
        candidates.append({
            "title": str(item.get("title")).strip(),
            "description": item.get("description"),
            "prices": [p for p in (item.get("prices") or []) if isinstance(p, (int, float))],
            "colors": [str(c) for c in (item.get("colors") or [])],
            "product_type": item.get("product_type"),
            "tags": [str(t) for t in (item.get("tags") or [])],
            "source": "document_llm",
        })
    return candidates


def extract_candidates_from_document(
    text: str, *, model: str = "gpt-4o-mini", limit: int = 200,
) -> dict[str, Any]:
    """Return a catalog_crawler-shaped envelope from pasted/uploaded text."""
    text = (text or "").strip()
    if not text:
        return {
            "url": None, "final_url": None, "status_code": None,
            "confidence": 0.0, "confidence_label": "baixa",
            "raw_text_preview": "", "text_blocks": [],
            "product_candidates": [], "warnings": ["Documento vazio."],
            "stages": [], "validation_policy": "raw_capture_only_human_validation_required",
            "source_note": "document_ingest", "absolute_source": None,
        }
    warnings: list[str] = []
    candidates = _structured_markdown_candidates(text)
    strategy = "document_structured"
    if not candidates:
        candidates = _llm_candidates(text, model=model)
        strategy = "document_llm"
        if not candidates:
            warnings.append(
                "Nao foi possivel extrair itens estruturados nem via LLM. "
                "Revise o documento manualmente pelo chat."
            )
    confidence = 0.8 if strategy == "document_structured" and candidates else (
        0.5 if candidates else 0.0
    )
    label = "alta" if confidence >= 0.72 else "media" if confidence >= 0.45 else "baixa"
    return {
        "url": None,
        "final_url": None,
        "status_code": None,
        "confidence": confidence,
        "confidence_label": label,
        "raw_text_preview": text[:5000],
        "text_blocks": [],
        "product_candidates": candidates[:limit],
        "warnings": warnings,
        "stages": [{"key": "extract", "label": strategy, "status": "done" if candidates else "warning"}],
        "validation_policy": "raw_capture_only_human_validation_required",
        "source_note": f"Documento colado/enviado pelo operador, estrategia: {strategy}.",
        "absolute_source": None,
    }

