# -*- coding: utf-8 -*-
"""
Backfill legacy knowledge sources into the RAG-ready knowledge_rag_* tables.

Sources covered:
- configured vault source, local or GitHub API (optional, read-only scan)
- knowledge_items
- kb_entries
- knowledge_nodes

The routine is deterministic and idempotent. It reuses the classification
logic from knowledge_rag_intake, writes chunks, mirrors entries back into the
semantic graph, and converts existing graph edges into knowledge_rag_links
when both endpoints have a RAG entry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from services import knowledge_graph, supabase_client, vault_sync
from services.knowledge_rag_intake import ALLOWED_CONTENT_TYPES, classify_intake


_CONTENT_TYPE_MAP = {
    "brand": "brand",
    "marca": "brand",
    "campaign": "campaign",
    "campanha": "campaign",
    "product": "product",
    "produto": "product",
    "faq": "faq",
    "copy": "copy",
    "asset": "asset",
    "briefing": "briefing",
    "tone": "tone",
    "tom": "tone",
    "rule": "rule",
    "regra": "rule",
    "prompt": "rule",
    "entity": "entity",
    "competitor": "entity",
    "kb_entry": "general_note",
    "knowledge_item": "general_note",
    "other": "general_note",
    "maker_material": "asset",
}

_SKIP_NODE_TYPES = {"persona", "tag", "mention"}

@dataclass
class LegacySource:
    source_table: str
    source_id: str
    persona_id: str
    title: str
    content: str
    content_type: Optional[str] = None
    tags: Optional[list[str]] = None
    metadata: Optional[dict] = None
    status: Optional[str] = None
    artifact_id: Optional[str] = None

def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]

def _normalize_content_type(value: Optional[str]) -> str:
    key = str(value or "").strip().lower()
    return _CONTENT_TYPE_MAP.get(key, key if key in ALLOWED_CONTENT_TYPES else "general_note")

def _entry_status(source_status: Optional[str]) -> str:
    status = str(source_status or "").lower()
    if status in {"approved", "embedded", "ativo", "active", "validated"}:
        return "validated"
    if status in {"rejected", "duplicate"}:
        return status
    return "pending_validation"

def _chunk_text(content: str, max_chars: int = 1600) -> list[str]:
    text = (content or "").strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs or [text]:
        if not current:
            current = paragraph
        elif len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}"
        else:
            chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    out: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            out.append(chunk)
            continue
        for start in range(0, len(chunk), max_chars):
            out.append(chunk[start:start + max_chars])
    return out

def _source_metadata(source: LegacySource) -> dict:
    meta = dict(source.metadata or {})
    legacy_sources = list(meta.get("legacy_sources") or [])
    legacy_marker = {
        "source_table": source.source_table,
        "source_id": source.source_id,
    }
    if legacy_marker not in legacy_sources:
        legacy_sources.append(legacy_marker)
    meta["legacy_sources"] = legacy_sources
    meta["legacy_source_table"] = source.source_table
    meta["legacy_source_id"] = source.source_id
    if source.artifact_id:
        meta["artifact_id"] = source.artifact_id
    return meta

def _upsert_source(source: LegacySource) -> dict:
    content_type = _normalize_content_type(source.content_type)
    meta = _source_metadata(source)
    classified = classify_intake(
        text=source.content,
        persona_id=source.persona_id,
        title=source.title,
        content_type=content_type,
        tags=source.tags or [],
        metadata=meta,
    )
    status = _entry_status(source.status)
    entry = supabase_client.upsert_knowledge_rag_entry({
        "persona_id": source.persona_id,
        "artifact_id": source.artifact_id,
        "content_type": classified["content_type"],
        "semantic_level": classified["semantic_level"],
        "title": classified["title"],
        "question": classified["question"],
        "answer": classified["answer"],
        "content": classified["content"],
        "summary": classified["summary"],
        "canonical_key": classified["canonical_key"],
        "slug": classified["slug"],
        "status": status,
        "tags": classified["tags"],
        "entities": classified["entities"],
        "products": classified["products"],
        "campaigns": classified["campaigns"],
        "metadata": classified["metadata"],
        "confidence": classified["confidence"],
        "importance": classified["importance"],
    })

    chunks = _chunk_text(classified["content"]) or [classified["content"]]
    supabase_client.replace_knowledge_rag_chunks(
        entry["id"],
        source.persona_id,
        [
            {
                "chunk_index": idx,
                "chunk_text": chunk,
                "chunk_summary": classified["summary"] if idx == 0 else chunk[:300],
                "metadata": {
                    "content_type": classified["content_type"],
                    "canonical_key": classified["canonical_key"],
                    "legacy_source_table": source.source_table,
                    "legacy_source_id": source.source_id,
                    "embedding_status": "pending",
                },
            }
            for idx, chunk in enumerate(chunks)
        ],
    )

    mirror = knowledge_graph.bootstrap_from_item(
        {
            "id": entry.get("id"),
            "persona_id": source.persona_id,
            "content_type": classified["content_type"],
            "title": classified["title"],
            "content": classified["content"],
            "tags": classified["tags"],
            "metadata": classified["metadata"],
            "status": status,
            "artifact_id": source.artifact_id,
        },
        frontmatter={
            **classified["metadata"],
            "slug": classified["slug"],
            "product": classified["products"],
            "campaigns": classified["campaigns"],
            "tags": classified["tags"],
        },
        body=classified["content"],
        persona_id=source.persona_id,
        source_table="knowledge_rag_entries",
    )
    return {"source": source, "classification": classified, "entry": entry, "graph_node": mirror}

def _sources_from_knowledge_items(persona_id: Optional[str], limit: int) -> list[LegacySource]:
    rows = supabase_client.get_knowledge_items(persona_id=persona_id, limit=limit, offset=0) or []
    sources: list[LegacySource] = []
    for row in rows:
        pid = row.get("persona_id")
        if not pid:
            continue
        sources.append(LegacySource(
            source_table="knowledge_items",
            source_id=str(row.get("id")),
            persona_id=pid,
            title=row.get("title") or row.get("file_path") or str(row.get("id")),
            content=row.get("content") or "",
            content_type=row.get("content_type"),
            tags=knowledge_graph._normalize_tags(row.get("tags")),
            metadata={
                **(row.get("metadata") or {}),
                "file_path": row.get("file_path"),
                "source": row.get("source"),
            },
            status=row.get("status"),
            artifact_id=row.get("artifact_id"),
        ))
    return sources

def _sources_from_kb_entries(persona_id: Optional[str]) -> list[LegacySource]:
    rows = supabase_client.get_kb_entries(persona_id=persona_id, status="") or []
    sources: list[LegacySource] = []
    for row in rows:
        pid = row.get("persona_id")
        if not pid:
            continue
        node_type = knowledge_graph._tipo_to_node_type(row.get("tipo") or row.get("categoria") or "")
        tags = knowledge_graph._normalize_tags(row.get("tags"))
        if row.get("produto"):
            tags.append(str(row.get("produto")))
        metadata = {
            "produto": row.get("produto"),
            "intencao": row.get("intencao"),
            "link": row.get("link"),
            "source": row.get("source"),
            "priority": row.get("prioridade"),
            "agent_visibility": row.get("agent_visibility") or [],
        }
        sources.append(LegacySource(
            source_table="kb_entries",
            source_id=str(row.get("id")),
            persona_id=pid,
            title=row.get("titulo") or str(row.get("id")),
            content=row.get("conteudo") or "",
            content_type=node_type,
            tags=tags,
            metadata={k: v for k, v in metadata.items() if v is not None},
            status=row.get("status") or "ATIVO",
        ))
    return sources

def _sources_from_knowledge_nodes(persona_id: Optional[str], limit: int) -> tuple[list[LegacySource], list[dict], list[dict]]:
    nodes, edges = supabase_client.list_all_knowledge_graph(persona_id=persona_id, limit_nodes=limit)
    sources: list[LegacySource] = []
    for node in nodes or []:
        node_type = node.get("node_type")
        pid = node.get("persona_id")
        if not pid or node_type in _SKIP_NODE_TYPES:
            continue
        content_type = _normalize_content_type(node_type)
        content = node.get("summary") or node.get("title") or node.get("slug") or ""
        sources.append(LegacySource(
            source_table="knowledge_nodes",
            source_id=str(node.get("id")),
            persona_id=pid,
            title=node.get("title") or node.get("slug") or str(node.get("id")),
            content=content,
            content_type=content_type,
            tags=knowledge_graph._normalize_tags(node.get("tags")),
            metadata={
                **(node.get("metadata") or {}),
                "slug": node.get("slug"),
                "source_table": node.get("source_table"),
                "source_id": node.get("source_id"),
                "node_type": node_type,
            },
            status=node.get("status"),
            artifact_id=node.get("artifact_id"),
        ))
    return sources, nodes or [], edges or []

def _sources_from_vault(vault_path: Optional[str], persona_slug: Optional[str]) -> list[LegacySource]:
    """Read the configured vault source without assuming a local repository."""
    sources: list[LegacySource] = []
    persona_cache: dict[str, Optional[dict]] = {}

    for fp, raw_content, root in vault_sync._yield_vault_files(vault_path):
        fm, body = vault_sync._frontmatter_and_body(fp, raw_content)
        slug = vault_sync._detect_persona(fp, fm)

        if persona_slug and slug != persona_slug:
            continue
        if not slug:
            continue

        if slug not in persona_cache:
            persona_cache[slug] = supabase_client.get_persona(slug)
        persona = persona_cache.get(slug)
        if not persona:
            continue

        rel_path = str(fp.relative_to(root))
        sources.append(LegacySource(
            source_table="obsidian_vault",
            source_id=rel_path,
            persona_id=persona["id"],
            title=vault_sync._file_title(fp, fm),
            content=body,
            content_type=vault_sync._detect_content_type(fp, fm),
            tags=knowledge_graph._normalize_tags(fm.get("tags")),
            metadata={**fm, "file_path": rel_path, "file_type": fp.suffix.lstrip(".")},
            status="pending",
        ))
    return sources

def _link_entries(created: list[dict], graph_nodes: list[dict], graph_edges: list[dict]) -> dict:
    by_source = {
        (r["source"].source_table, r["source"].source_id): r["entry"]
        for r in created
        if r.get("entry") and r.get("source")
    }
    by_slug = {
        (r["entry"].get("persona_id"), r["entry"].get("content_type"), r["entry"].get("slug")): r["entry"]
        for r in created
        if r.get("entry")
    }
    node_to_entry: dict[str, dict] = {}
    for node in graph_nodes:
        entry = by_source.get(("knowledge_nodes", str(node.get("id"))))
        if not entry:
            entry = by_slug.get((
                node.get("persona_id"),
                _normalize_content_type(node.get("node_type")),
                knowledge_graph._slugify(node.get("slug") or node.get("title") or ""),
            ))
        if entry:
            node_to_entry[str(node.get("id"))] = entry

    counts = {"links_created": 0, "links_skipped": 0}
    for edge in graph_edges:
        source_entry = node_to_entry.get(str(edge.get("source_node_id")))
        target_entry = node_to_entry.get(str(edge.get("target_node_id")))
        if not source_entry or not target_entry or source_entry.get("id") == target_entry.get("id"):
            counts["links_skipped"] += 1
            continue
        supabase_client.upsert_knowledge_rag_link({
            "persona_id": source_entry.get("persona_id") or target_entry.get("persona_id"),
            "source_entry_id": source_entry["id"],
            "target_entry_id": target_entry["id"],
            "relation_type": edge.get("relation_type") or "same_topic_as",
            "weight": edge.get("weight") or 1,
            "confidence": edge.get("confidence") or 0.6,
            "created_by": "rag_backfill",
            "metadata": {
                "legacy_edge_id": edge.get("id"),
                "source": "knowledge_edges",
            },
        })
        counts["links_created"] += 1

    product_entries = {
        (entry.get("persona_id"), entry.get("slug")): entry
        for entry in (r.get("entry") for r in created)
        if entry and entry.get("content_type") == "product"
    }
    for row in created:
        entry = row.get("entry") or {}
        for product_slug in entry.get("products") or []:
            target = product_entries.get((entry.get("persona_id"), product_slug))
            if not target or target.get("id") == entry.get("id"):
                continue
            is_faq = entry.get("content_type") == "faq"
            rel = "answers_question" if is_faq else "about_product"
            supabase_client.upsert_knowledge_rag_link({
                "persona_id": entry.get("persona_id"),
                "source_entry_id": target["id"] if is_faq else entry["id"],
                "target_entry_id": entry["id"] if is_faq else target["id"],
                "relation_type": rel,
                "weight": 1,
                "confidence": 0.7,
                "created_by": "rag_backfill",
                "metadata": {"source": "rag_product_detection"},
            })
            counts["links_created"] += 1
    return counts

def backfill_knowledge_rag(
    *,
    persona_id: Optional[str] = None,
    persona_slug: Optional[str] = None,
    include_vault: bool = True,
    vault_path: Optional[str] = None,
    limit_items: int = 5000,
    limit_nodes: int = 5000,
) -> dict:
    """Reprocess legacy knowledge into knowledge_rag_entries/chunks/links."""
    resolved_persona_id = persona_id
    if persona_slug:
        persona = supabase_client.get_persona(persona_slug)
        if not persona:
            raise ValueError(f"Persona not found: {persona_slug}")
        resolved_persona_id = persona.get("id")

    sources: list[LegacySource] = []
    sources.extend(_sources_from_knowledge_items(resolved_persona_id, limit_items))
    sources.extend(_sources_from_kb_entries(resolved_persona_id))
    node_sources, graph_nodes, graph_edges = _sources_from_knowledge_nodes(resolved_persona_id, limit_nodes)
    sources.extend(node_sources)
    if include_vault:
        sources.extend(_sources_from_vault(vault_path, persona_slug))

    counts = {
        "sources_seen": len(sources),
        "entries_upserted": 0,
        "chunks_replaced": 0,
        "links_created": 0,
        "links_skipped": 0,
        "errors": [],
        "by_source": {},
        "by_type": {},
    }
    created: list[dict] = []
    for source in sources:
        counts["by_source"][source.source_table] = counts["by_source"].get(source.source_table, 0) + 1
        try:
            result = _upsert_source(source)
            created.append(result)
            entry = result["entry"]
            counts["entries_upserted"] += 1
            counts["chunks_replaced"] += max(1, len(_chunk_text(result["classification"]["content"])))
            ctype = entry.get("content_type")
            counts["by_type"][ctype] = counts["by_type"].get(ctype, 0) + 1
        except Exception as exc:
            counts["errors"].append({
                "source_table": source.source_table,
                "source_id": source.source_id,
                "error": str(exc)[:300],
            })

    link_counts = _link_entries(created, graph_nodes, graph_edges)
    counts.update(link_counts)
    counts["errors"] = counts["errors"][:50]
    return counts
