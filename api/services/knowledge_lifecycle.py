from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from services import knowledge_graph, knowledge_rag_intake, supabase_client
from services import vault_sync


def _resolve_persona(*, persona_id: Optional[str] = None, persona_slug: Optional[str] = None) -> dict:
    if persona_slug:
        persona = supabase_client.get_persona(persona_slug)
        if not persona:
            raise ValueError(f"Persona not found: {persona_slug}")
        return persona
    if persona_id:
        resolved = supabase_client._resolve_persona_id(persona_id)
        if not resolved:
            raise ValueError(f"Persona not found: {persona_id}")
        for persona in supabase_client.get_personas():
            if persona.get("id") == resolved:
                return persona
        return {"id": resolved, "slug": persona_id, "name": persona_id}
    raise ValueError("persona_id or persona_slug is required")


def _slug_from_file_path(file_path: Optional[str], title: str) -> str:
    if file_path:
        return knowledge_graph._slugify(Path(file_path).stem)
    return knowledge_graph._slugify(title)


def _content_type_to_tipo(content_type: str) -> str:
    return {
        "faq": "faq",
        "brand": "brand",
        "briefing": "briefing",
        "product": "produto",
        "copy": "copy",
        "prompt": "prompt",
        "rule": "regra",
        "tone": "tom",
        "competitor": "concorrente",
        "audience": "audiencia",
        "campaign": "campanha",
        "maker_material": "maker",
        "asset": "asset",
        "other": "geral",
    }.get((content_type or "").lower(), "geral")


def _normalized_file_path(file_path: Optional[str]) -> Optional[str]:
    return supabase_client.normalize_file_path(file_path)


def _confirmed_graph_node_for_item(item: dict) -> Optional[dict]:
    if not item or not item.get("id"):
        return None
    persona_id = item.get("persona_id")
    node = supabase_client.get_knowledge_node_for_source(
        "knowledge_items",
        str(item["id"]),
        persona_id=persona_id,
    )
    if node:
        return node
    node_id = ((item.get("metadata") or {}).get("knowledge_node_id"))
    if node_id:
        return supabase_client.get_knowledge_node(node_id)
    return None


def persist_pending_knowledge_item(
    *,
    persona_slug: str,
    title: str,
    content: str,
    content_type: str,
    file_path: Optional[str],
    file_type: str = "md",
    metadata: Optional[dict] = None,
    tags: Optional[list[str]] = None,
    source_ref: Optional[str] = None,
    agent_visibility: Optional[list[str]] = None,
) -> dict:
    def _log_persist(level: str, message: str, exc: Optional[Exception] = None) -> None:
        try:
            from services import sre_logger
            payload = (
                f"{message} | persona_slug={persona_slug} content_type={content_type} file_path={file_path}"
            )
            if level == "error":
                sre_logger.error("knowledge_lifecycle", payload, exc)
            else:
                sre_logger.info("knowledge_lifecycle", payload)
        except Exception:
            pass

    normalized_file_path = _normalized_file_path(file_path)
    persona = _resolve_persona(persona_slug=persona_slug)
    source = supabase_client.get_or_create_manual_source()
    existing = supabase_client.get_knowledge_item_by_path(normalized_file_path) if normalized_file_path else None
    merged_metadata = {
        **(existing.get("metadata") or {} if existing else {}),
        **(metadata or {}),
        "source_ref": source_ref,
        "created_via": "kb_intake_sofia",
    }
    merged_classification = dict(merged_metadata.get("classification") or {})
    if merged_classification:
        merged_classification["content_type"] = content_type
        merged_metadata["classification"] = merged_classification
    requested_status = "pending"
    if existing and existing.get("content") == content:
        requested_status = existing.get("status") or "pending"
    payload = {
        "persona_id": persona["id"],
        "source_id": source["id"],
        "status": requested_status,
        "content_type": content_type,
        "title": title,
        "content": content[:8000],
        "metadata": merged_metadata,
        "file_path": normalized_file_path,
        "file_type": file_type,
        "tags": tags or (existing.get("tags") if existing else []) or [],
        "agent_visibility": agent_visibility or (existing.get("agent_visibility") if existing else ["SDR", "Closer", "Classifier"]),
    }
    contract_errors = supabase_client.validate_knowledge_item_payload(payload)
    if contract_errors:
        _log_persist("error", f"contract violation: {contract_errors}")
        raise ValueError(f"contract: {'; '.join(contract_errors)}")

    if existing:
        supabase_client.update_knowledge_item(existing["id"], payload)
        item = supabase_client.get_knowledge_item(existing["id"]) or {**existing, **payload}
    else:
        try:
            item = supabase_client.insert_knowledge_item(payload)
        except Exception as exc:
            _log_persist("error", f"knowledge_item insert raised: {exc}", exc)
            raise ValueError(
                f"insert failed for file_path={normalized_file_path or '<none>'}: {exc}"
            ) from exc
        if not item or not item.get("id"):
            confirmed = supabase_client.get_knowledge_item_by_path(normalized_file_path) if normalized_file_path else None
            if confirmed and confirmed.get("id"):
                _log_persist("info", "knowledge_item insert returned empty payload; recovered by file_path lookup")
                item = confirmed
            else:
                _log_persist("error", "knowledge_item insert returned empty payload and lookup failed")
                raise ValueError(
                    f"knowledge_item insert returned no row for file_path={normalized_file_path or '<none>'}"
                )
    if not item or not item.get("id"):
        _log_persist("error", "knowledge_item persistence resolved without id")
        raise ValueError(
            f"knowledge_item persistence resolved without id for file_path={normalized_file_path or '<none>'}"
        )
    mirror = knowledge_graph.bootstrap_from_item(
        item,
        frontmatter=item.get("metadata") or {},
        body=item.get("content") or "",
        persona_id=persona["id"],
        source_table="knowledge_items",
    )
    if mirror and mirror.get("id"):
        supabase_client.update_knowledge_item(item["id"], {
            "metadata": {
                **(item.get("metadata") or {}),
                "knowledge_node_id": mirror.get("id"),
            },
        })
        item = supabase_client.get_knowledge_item(item["id"]) or item
    confirmed_node = _confirmed_graph_node_for_item(item)
    if not confirmed_node or not confirmed_node.get("id"):
        _log_persist("error", "knowledge_item persisted but graph node was not confirmed")
        raise ValueError(
            f"knowledge_item graph node not confirmed for file_path={normalized_file_path or '<none>'}"
        )
    if (item.get("metadata") or {}).get("knowledge_node_id") != confirmed_node.get("id"):
        supabase_client.update_knowledge_item(item["id"], {
            "metadata": {
                **(item.get("metadata") or {}),
                "knowledge_node_id": confirmed_node.get("id"),
            },
        })
        item = supabase_client.get_knowledge_item(item["id"]) or item
    return item


def promote_knowledge_item(
    item_id: str,
    *,
    promote_to_kb: bool,
    agent_visibility: Optional[list[str]] = None,
    approval_mode: str = "manual_validation",
    edge_metadata: Optional[dict] = None,
) -> dict:
    item = supabase_client.get_knowledge_item(item_id)
    if not item:
        raise ValueError("Item not found")
    if not item.get("persona_id"):
        raise ValueError("Item must have a persona assigned")

    now_iso = datetime.now(timezone.utc).isoformat()
    update_data = {
        "status": "approved",
        "curation_status": "approved",
        "approved_at": now_iso,
        "agent_visibility": agent_visibility or item.get("agent_visibility") or ["SDR", "Closer", "Classifier"],
    }
    supabase_client.update_knowledge_item(item_id, update_data)
    item = supabase_client.get_knowledge_item(item_id) or {**item, **update_data}

    evidence = {
        "knowledge_item_id": item_id,
        "kb_entry_id": None,
        "knowledge_rag_entry_id": None,
        "knowledge_rag_chunk_ids": [],
        "knowledge_node_id": None,
        "main_tree_edge_id": None,
        "embedded_edge_id": None,
        "final_status": "approved",
    }
    if promote_to_kb:
        raise ValueError("Publication to the Golden Dataset now happens only by connecting an approved FAQ node to Embedded in the graph")
    if not promote_to_kb:
        mirror = knowledge_graph.bootstrap_from_item(
            item,
            frontmatter=item.get("metadata") or {},
            body=item.get("content") or "",
            persona_id=item["persona_id"],
            source_table="knowledge_items",
        )
        confirmed_node = _confirmed_graph_node_for_item(item) or mirror or {}
        if not confirmed_node.get("id"):
            raise ValueError("Approved item is missing a confirmed knowledge_node")
        evidence["knowledge_node_id"] = confirmed_node.get("id")
        return {"item": item, "evidence": evidence}


def delete_knowledge_item_cascade(item_id: str) -> dict:
    item = supabase_client.get_knowledge_item(item_id)
    if not item:
        raise ValueError("Item not found")

    metadata = item.get("metadata") or {}
    evidence = {
        "knowledge_item_deleted": False,
        "knowledge_node_deleted_or_detached": False,
        "vault_file_deleted": False,
        "references_cleaned": False,
        "knowledge_item_id": item_id,
        "knowledge_node_id": None,
        "kb_entry_id": metadata.get("kb_entry_id"),
        "knowledge_rag_entry_id": metadata.get("knowledge_rag_entry_id"),
        "file_path": item.get("file_path"),
    }

    node = _confirmed_graph_node_for_item(item)
    if node and node.get("id"):
        evidence["knowledge_node_id"] = node.get("id")
        supabase_client.delete_knowledge_node(node["id"])
        evidence["knowledge_node_deleted_or_detached"] = True

    rag_entry_id = metadata.get("knowledge_rag_entry_id")
    if rag_entry_id:
        supabase_client.delete_knowledge_rag_entry(rag_entry_id)

    kb_entry_id = metadata.get("kb_entry_id")
    if kb_entry_id:
        supabase_client.delete_kb_entry(kb_entry_id)

    evidence["vault_file_deleted"] = vault_sync.delete_vault_relative_path(item.get("file_path"))
    evidence["knowledge_item_deleted"] = supabase_client.delete_knowledge_item(item_id)
    evidence["references_cleaned"] = (
        evidence["knowledge_item_deleted"]
        and (evidence["knowledge_node_deleted_or_detached"] or not node)
    )
    return evidence
