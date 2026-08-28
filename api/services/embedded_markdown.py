"""Embedded node Markdown body.

The Embedded node is the semantic destination for approved/published FAQs. On
top of the graph edges, it also keeps a human-readable Markdown body that lists
every FAQ currently connected to it. This module builds that body and keeps it
in sync whenever a FAQ is published to, or disconnected from, Embedded.

`build_embedded_markdown` is pure (unit-testable). `rebuild_embedded_markdown`
reads the live graph and persists the body into the Embedded node's
`metadata.markdown` (which graph-data already surfaces via `_metadata_markdown`).
"""
from __future__ import annotations

import re
from typing import Any, Optional

from services import supabase_client


def _answer_from_content(content: str) -> str:
    text = (content or "").strip()
    if not text:
        return ""
    match = re.search(r"resposta\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def faq_payload_from_node(node: dict) -> dict:
    """Resolve {question, answer, node_label, source_url, parent_type} for a FAQ node."""
    meta = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    item = None
    if node.get("source_table") == "knowledge_items" and node.get("source_id"):
        item = supabase_client.get_knowledge_item(str(node.get("source_id")))
    item_meta = (item or {}).get("metadata") if isinstance((item or {}).get("metadata"), dict) else {}
    question = (node.get("title") or (item or {}).get("title") or "").strip()
    answer = str(
        (item_meta or {}).get("answer")
        or (meta or {}).get("answer")
        or _answer_from_content((item or {}).get("content") or (meta or {}).get("markdown") or node.get("summary") or "")
    ).strip()
    source_ctx = (item_meta or {}).get("source_context") or (meta or {}).get("source_context") or {}
    source_url = (
        (item_meta or {}).get("source_url")
        or (meta or {}).get("source_url")
        or (source_ctx.get("source_url") if isinstance(source_ctx, dict) else None)
    )
    parent_type = (item_meta or {}).get("parent_node_type") or (meta or {}).get("parent_node_type") or "faq"
    return {
        "question": question,
        "answer": answer,
        "node_label": question,
        "source_url": source_url,
        "parent_type": parent_type,
    }


def build_embedded_markdown(persona_label: str, faqs: list[dict]) -> str:
    """Render the Embedded body listing every connected FAQ (pure)."""
    label = (persona_label or "Persona").strip()
    lines: list[str] = [f"# Embedded â€” {label}", "", "## FAQs conectadas", ""]
    if not faqs:
        lines.append("_Nenhuma FAQ conectada ao Embedded._")
        return "\n".join(lines).rstrip() + "\n"
    for faq in faqs:
        question = str(faq.get("question") or "").strip() or "(sem pergunta)"
        answer = str(faq.get("answer") or "").strip()
        lines.append(f"### {question}")
        lines.append("")
        lines.append(f"Resposta: {answer}" if answer else "Resposta: (sem resposta)")
        lines.append("")
        lines.append("Origem:")
        lines.append("")
        lines.append(f"* Node: {faq.get('node_label') or question}")
        if faq.get("source_url"):
            lines.append(f"* Source URL: {faq.get('source_url')}")
        lines.append(f"* Parent type: {faq.get('parent_type') or 'faq'}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def collect_connected_faqs(persona_id: str) -> list[dict]:
    """FAQ payloads for every active FAQ -> Embedded edge of the persona."""
    if not persona_id:
        return []
    embedded = supabase_client.ensure_embedded_node(persona_id)
    if not embedded or not embedded.get("id"):
        return []
    embedded_id = embedded["id"]
    try:
        edges = supabase_client.list_edges_for_nodes([embedded_id]) or []
    except Exception:
        edges = []
    faqs: list[dict] = []
    seen: set[str] = set()
    for edge in edges:
        if edge.get("target_node_id") != embedded_id:
            continue  # only incoming FAQ -> Embedded edges
        src_id = edge.get("source_node_id")
        if not src_id or src_id in seen:
            continue
        node = supabase_client.get_knowledge_node(src_id)
        if not node or str(node.get("node_type") or "").lower() != "faq":
            continue
        seen.add(src_id)
        faqs.append(faq_payload_from_node(node))
    return faqs


def rebuild_embedded_markdown(persona_id: Optional[str]) -> Optional[str]:
    """Recompute and persist the Embedded body from the live connected FAQs.

    Best-effort: returns the markdown it stored, or None when unavailable. Never
    raises so it can be called from edge create/delete/revert paths safely.
    """
    if not persona_id:
        return None
    try:
        embedded = supabase_client.ensure_embedded_node(persona_id)
        if not embedded or not embedded.get("id"):
            return None
        persona = supabase_client.get_persona_by_id(persona_id) or {}
        label = persona.get("name") or persona.get("slug") or "Persona"
        faqs = collect_connected_faqs(persona_id)
        markdown = build_embedded_markdown(label, faqs)
        supabase_client.update_knowledge_node(
            embedded["id"],
            {
                "metadata": {
                    **(embedded.get("metadata") or {}),
                    "markdown": markdown,
                    "connected_faq_count": len(faqs),
                }
            },
            mark_related_faqs=False,
        )
        return markdown
    except Exception:
        return None

