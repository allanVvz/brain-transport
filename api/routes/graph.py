# -*- coding: utf-8 -*-
"""Graph data endpoint â€” returns ReactFlow-compatible nodes and edges.

The payload is decorated with semantic level/importance/tier/weight pulled
from knowledge_node_type_registry + knowledge_relation_type_registry
(migration 009) so the frontend can render a layered, hierarchical graph
without re-deriving the ontology in JS.
"""

import logging
import json
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from typing import Optional

from services import auth_service, supabase_client, knowledge_graph, knowledge_lifecycle, approved_knowledge_snapshots
from services import embedded_markdown, graph_document_publisher, graph_json_v2_store
from services.event_emitter import emit
from services.audit_helpers import current_actor

router = APIRouter(prefix="/knowledge", tags=["graph"])
logger = logging.getLogger("ai_brain.graph")


class GraphEdgeCreateBody(BaseModel):
    source_node_id: str
    target_node_id: str
    relation_type: str = "manual"
    persona_id: Optional[str] = None
    weight: float = 1.0
    metadata: dict = {}


class GraphPreflightBody(BaseModel):
    graph: Optional[dict] = None
    tree_edge_ids: list[str] = []


def _contract_node(node_id: str, node_type: str, approved: bool = False) -> dict:
    return {"id": node_id, "type": node_type, "approved": approved}


def _contract_edge(edge_id: str, src: str, tgt: str, edge_type: str = "main", active: bool = True) -> dict:
    return {"id": edge_id, "source": src, "target": tgt, "edge_type": edge_type, "active": active}


def _build_vzlupas_contract_graph() -> tuple[dict, list[str]]:
    fixture = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "vzlupas-products-9.json"
    products = json.loads(fixture.read_text(encoding="utf-8"))

    nodes = [
        _contract_node("persona-vzlupas", "persona"),
        _contract_node("brand-vzlupas", "brand"),
        _contract_node("briefing-vzlupas", "briefing"),
        _contract_node("campaign-vzlupas", "campaign"),
        _contract_node("audience-vzlupas", "audience"),
        _contract_node("pg-clipon", "product_group"),
        _contract_node("pg-grau", "product_group"),
        _contract_node("pg-sol", "product_group"),
        _contract_node("embed-vzlupas", "embed"),
    ]
    edges = [
        _contract_edge("e1", "persona-vzlupas", "brand-vzlupas"),
        _contract_edge("e2", "brand-vzlupas", "briefing-vzlupas"),
        _contract_edge("e3", "briefing-vzlupas", "campaign-vzlupas"),
        _contract_edge("e4", "campaign-vzlupas", "audience-vzlupas"),
        _contract_edge("e5", "audience-vzlupas", "pg-clipon"),
        _contract_edge("e6", "audience-vzlupas", "pg-grau"),
        _contract_edge("e7", "audience-vzlupas", "pg-sol"),
    ]
    group_for_idx = {0: "pg-clipon", 1: "pg-clipon", 2: "pg-clipon", 3: "pg-grau", 4: "pg-grau", 5: "pg-grau", 6: "pg-sol", 7: "pg-sol", 8: "pg-sol"}
    tree_edge_ids = [e["id"] for e in edges]

    for i, prod in enumerate(products):
        product_id = f"product-{prod['slug']}"
        faq1 = f"faq-{prod['slug']}-1"
        faq2 = f"faq-{prod['slug']}-2"
        nodes.extend([
            _contract_node(product_id, "product"),
            _contract_node(faq1, "faq", approved=True),
            _contract_node(faq2, "faq", approved=True),
        ])
        edges.extend([
            _contract_edge(f"ep-{i}", group_for_idx[i], product_id),
            _contract_edge(f"ef1-{i}", product_id, faq1),
            _contract_edge(f"ef2-{i}", product_id, faq2),
            _contract_edge(f"ee1-{i}", faq1, "embed-vzlupas"),
            _contract_edge(f"ee2-{i}", faq2, "embed-vzlupas", edge_type="reference"),
        ])
        tree_edge_ids.extend([f"ep-{i}", f"ef1-{i}", f"ef2-{i}", f"ee1-{i}"])

    edges.append(_contract_edge("ref-brand-campaign", "brand-vzlupas", "campaign-vzlupas", edge_type="reference"))
    return {"nodes": nodes, "edges": edges}, tree_edge_ids


def _run_vzlupas_preflight(graph: dict, tree_edge_ids: list[str]) -> tuple[bool, list[dict], list[str]]:
    nodes = {n.get("id"): n for n in graph.get("nodes", [])}
    edges = graph.get("edges", [])
    main_edges = [e for e in edges if e.get("edge_type", "main") == "main"]
    active_edges = [e for e in edges if e.get("active", True)]
    level = {
        "persona": 0, "brand": 1, "briefing": 2, "campaign": 3, "audience": 4,
        "product_group": 5, "product": 6, "offer": 7, "copy": 8, "faq": 9, "embed": 10,
    }
    main_next = {
        "persona": {"brand"}, "brand": {"briefing"}, "briefing": {"campaign"}, "campaign": {"audience"},
        "audience": {"product_group"}, "product_group": {"product"}, "product": {"offer", "faq", "copy"},
        "offer": {"copy", "faq"}, "copy": {"faq"}, "faq": {"embed"},
    }
    checks: list[dict] = []
    errors: list[str] = []

    invalid_main = []
    for e in main_edges:
        src_type = str((nodes.get(e.get("source")) or {}).get("type") or "")
        tgt_type = str((nodes.get(e.get("target")) or {}).get("type") or "")
        if tgt_type not in main_next.get(src_type, set()):
            invalid_main.append(f"{src_type}->{tgt_type}:{e.get('id')}")
    checks.append({"name": "main_hierarchy", "ok": len(invalid_main) == 0, "errors": invalid_main})
    errors.extend(invalid_main)

    forbidden = []
    for e in active_edges:
        src = nodes.get(e.get("source")) or {}
        tgt = nodes.get(e.get("target")) or {}
        src_type = str(src.get("type") or "")
        tgt_type = str(tgt.get("type") or "")
        if tgt_type == "embed" and src_type != "faq":
            forbidden.append(f"EMBED_SOURCE_NOT_FAQ:{e.get('id')}")
        if src_type == "faq" and tgt_type == "embed" and not bool(src.get("approved", False)):
            forbidden.append(f"FAQ_NOT_APPROVED_FOR_EMBED:{e.get('id')}")
    checks.append({"name": "embed_gate", "ok": len(forbidden) == 0, "errors": forbidden})
    errors.extend(forbidden)

    tree_errors = []
    edge_by_id = {e.get("id"): e for e in edges}
    for edge_id in tree_edge_ids:
        edge = edge_by_id.get(edge_id)
        if not edge:
            tree_errors.append(f"missing_tree_edge:{edge_id}")
            continue
        if edge.get("edge_type", "main") != "main":
            tree_errors.append(f"tree_not_main:{edge_id}")
    checks.append({"name": "tree_main_only", "ok": len(tree_errors) == 0, "errors": tree_errors})
    errors.extend(tree_errors)

    depth_errors = []
    for e in main_edges:
        src_type = str((nodes.get(e.get("source")) or {}).get("type") or "")
        tgt_type = str((nodes.get(e.get("target")) or {}).get("type") or "")
        if level.get(tgt_type, 999) <= level.get(src_type, -1):
            depth_errors.append(f"non_increasing_depth:{e.get('id')}")
    checks.append({"name": "layout_depth", "ok": len(depth_errors) == 0, "errors": depth_errors})
    errors.extend(depth_errors)

    return len(errors) == 0, checks, errors


def _raw_graph_node_id(value: str) -> str:
    raw = (value or "").strip()
    if raw.startswith("gn:"):
        return raw[3:]
    return raw


def _resolve_graph_node_ref(value: str, persona_id: Optional[str] = None) -> Optional[dict]:
    raw = (value or "").strip()
    if raw.startswith("ki:"):
        item = supabase_client.get_knowledge_item(raw.split(":", 1)[1])
        if not item:
            return None
        return {
            "id": f"ki:{item['id']}",
            "node_type": "knowledge_item",
            "persona_id": item.get("persona_id") or persona_id,
            "source_table": "knowledge_items",
            "source_id": item.get("id"),
            "title": item.get("title"),
            "metadata": item.get("metadata") or {},
        }
    if raw.startswith("gn:"):
        return supabase_client.get_knowledge_node(raw[3:])
    if raw.startswith("persona:"):
        return supabase_client.ensure_persona_knowledge_node(raw.split(":", 1)[1])
    if raw and len(raw) == 36 and raw.count("-") == 4:
        return supabase_client.get_knowledge_node(raw)
    if raw == "gallery" and persona_id:
        return supabase_client.ensure_gallery_node(persona_id)
    if raw.startswith("embedded:"):
        return supabase_client.ensure_embedded_node(raw.split(":", 1)[1])
    if raw == "embedded" and persona_id:
        return supabase_client.ensure_embedded_node(persona_id)
    return None


def _content_type_to_tipo(content_type: str) -> str:
    return {
        "faq": "faq",
        "brand": "brand",
        "briefing": "briefing",
        "product": "produto",
        "offer": "oferta",
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


def _knowledge_item_for_graph_node(node: dict) -> Optional[dict]:
    if not node:
        return None
    if node.get("source_table") == "knowledge_items" and node.get("source_id"):
        return supabase_client.get_knowledge_item(str(node.get("source_id")))
    return None


def _require_graph_promotion_evidence(evidence: dict) -> None:
    required = ["knowledge_item_id", "approved_snapshot_id", "knowledge_node_id", "rag_entry_id", "embedded_edge_id"]
    missing = [key for key in required if not evidence.get(key)]
    if not evidence.get("rag_chunk_ids"):
        missing.append("rag_chunk_ids")
    if missing:
        raise HTTPException(502, f"Golden Dataset publication incomplete: missing {', '.join(missing)}")


def _prepare_faq_for_embedded(source_node: dict) -> tuple[dict, dict]:
    """Make a FAQ graph node publishable without losing its graph provenance.

    Legacy/imported FAQ nodes can exist before a matching ``knowledge_item``
    exists.  Connecting one to Embedded is an explicit approval action, so we
    first materialize the canonical item and its complete structural path.  Do
    this *before* the publisher creates the terminal edge: an invalid branch
    must never leave an FAQ -> Embedded edge behind.
    """
    if str(source_node.get("node_type") or "").lower() != "faq":
        raise HTTPException(400, "Only FAQ nodes can be published to the Golden Dataset")
    persona_id = source_node.get("persona_id")
    if not persona_id:
        raise HTTPException(422, "FAQ must belong to a persona before publication")

    chain, path_edges = approved_knowledge_snapshots._build_hierarchy(source_node)
    path_types = [str(node.get("node_type") or "").lower() for node in chain]
    if (
        len(chain) < 2
        or not path_types
        or path_types[0] != "persona"
        or chain[-1].get("id") != source_node.get("id")
    ):
        raise HTTPException(422, "FAQ needs a complete active path from Persona before publication")

    question, answer = approved_knowledge_snapshots._faq_parts(source_node, None)
    if not question or not answer:
        raise HTTPException(422, "FAQ needs a question and answer before publication")
    if not approved_knowledge_snapshots._answer_is_useful(question, answer):
        raise HTTPException(422, "FAQ answer is not sufficient for Golden Dataset publication")

    persona = supabase_client.get_persona_by_id(persona_id) or {"id": persona_id, "slug": persona_id}
    branch_context = approved_knowledge_snapshots._branch_context(chain, path_edges, persona)
    branch_path = branch_context.get("path") or []
    if not branch_path or branch_path[-1].get("node_id") != source_node.get("id"):
        raise HTTPException(422, "FAQ branch path could not be reconstructed")

    node_meta = source_node.get("metadata") or {}
    parent = chain[-2]
    next_meta = {
        **node_meta,
        "question": question,
        "answer": answer,
        "source": node_meta.get("source") or "pending_source",
        "source_node_id": parent.get("id"),
        "source_node_type": parent.get("node_type"),
        "branch_path": branch_path,
        "branch_context": branch_context,
        "approved_from": "faq_embedded_connection",
    }
    source_item = _knowledge_item_for_graph_node(source_node)
    body = f"Pergunta: {question}\nResposta: {answer}"
    if source_item:
        supabase_client.update_knowledge_item(source_item["id"], {
            "status": "approved",
            "title": source_node.get("title") or question,
            "content": body,
            "metadata": {**(source_item.get("metadata") or {}), **next_meta},
        })
    else:
        slug = str(source_node.get("slug") or source_node.get("id"))
        source_item = supabase_client.insert_knowledge_item({
            "persona_id": persona_id,
            "source_id": source_node["id"],
            "status": "approved",
            "content_type": "faq",
            "title": source_node.get("title") or question,
            "content": body,
            "metadata": next_meta,
            "file_path": f"generated/{persona.get('slug') or persona_id}/faq/{slug}.md",
            "file_type": "md",
            "tags": list(dict.fromkeys([*(source_node.get("tags") or []), "faq", "golden-dataset"])),
            "agent_visibility": ["SDR", "Closer", "Classifier"],
            "canonical_key": f"{persona.get('slug') or persona_id}:faq:{slug}",
        })
    if not source_item or not source_item.get("id"):
        raise HTTPException(502, "Could not materialize FAQ knowledge item for publication")

    updated = supabase_client.update_knowledge_node(source_node["id"], {
        "source_table": "knowledge_items",
        "source_id": source_item["id"],
        "status": "approved",
        "metadata": {**next_meta, "knowledge_item_id": source_item["id"]},
    }, mark_related_faqs=False)
    return (updated or {**source_node, "source_table": "knowledge_items", "source_id": source_item["id"], "metadata": next_meta}), source_item


@router.post("/graph-edges")
def create_graph_edge(body: GraphEdgeCreateBody, request: Request):
    raw_source = (body.source_node_id or "").strip()
    raw_target = (body.target_node_id or "").strip()
    edge_persona_id = body.persona_id
    metadata = {
        **(body.metadata or {}),
        "created_from": (body.metadata or {}).get("created_from") or "graph_ui",
        "direction": (body.metadata or {}).get("direction") or "source_to_target",
    }
    if (raw_source.startswith("ki:") or raw_target.startswith("ki:")) and not (raw_target.startswith("embedded:") or raw_target == "embedded"):
        raise HTTPException(400, "Pending knowledge items cannot be linked from the graph before approval")
    if raw_source.startswith("ki:") and (raw_target.startswith("embedded:") or raw_target == "embedded"):
        raise HTTPException(400, "Connect the semantic FAQ node to Embedded; a pending knowledge item cannot be published directly.")

    source_node = _resolve_graph_node_ref(body.source_node_id, body.persona_id)
    target_node = _resolve_graph_node_ref(body.target_node_id, body.persona_id)
    if not source_node or not target_node:
        raise HTTPException(400, "source_node_id and target_node_id are required")
    relation_type = (body.relation_type or "manual").strip() or "manual"
    source_type = str(source_node.get("node_type") or "").lower()
    target_type = str(target_node.get("node_type") or "").lower()
    if source_type == "faq" and target_type != "embedded":
        raise HTTPException(400, "FAQ is a terminal node. It can only publish to Embedded; connect product, offer or copy into FAQ instead.")
    if target_type == "faq" and relation_type == "manual":
        relation_type = "answers_question"
    if target_node.get("node_type") == "gallery":
        relation_type = "gallery_asset"
    if relation_type == "gallery_asset" and "gallery" not in {source_node.get("node_type"), target_node.get("node_type")}:
        raise HTTPException(400, "gallery_asset edges must involve a Gallery node")
    if relation_type == "gallery_asset":
        asset_node = target_node if source_node.get("node_type") == "gallery" else source_node
        if asset_node.get("node_type") != "asset":
            raise HTTPException(400, "Gallery can only receive approved asset nodes")
        if knowledge_graph._validation_state(asset_node) != "validated":
            raise HTTPException(400, "Approve the asset before adding it to Gallery")
        # Canonical direction is asset -> Gallery. Accept reverse drag gestures,
        # but never persist gallery -> asset.
        if source_type == "gallery" and target_type == "asset":
            source_node, target_node = target_node, source_node
            source_type, target_type = target_type, source_type
            metadata["direction_normalized"] = "asset_to_gallery"
    source_id = source_node["id"]
    target_id = target_node["id"]
    if source_id == target_id:
        raise HTTPException(400, "Self connections are not allowed")
    edge_persona_id = edge_persona_id or source_node.get("persona_id") or target_node.get("persona_id")
    if edge_persona_id:
        auth_service.assert_persona_access(request, persona_id=edge_persona_id)
    source_item = _knowledge_item_for_graph_node(source_node)
    if target_node.get("node_type") == "embedded":
        if source_node.get("node_type") != "faq":
            raise HTTPException(400, "Only FAQ nodes can be published to the Golden Dataset")
    if "primary_tree" not in metadata:
        metadata["primary_tree"] = relation_type == "manual"
    if "embedded" in {source_node.get("node_type"), target_node.get("node_type")} or "gallery" in {source_node.get("node_type"), target_node.get("node_type")}:
        metadata["primary_tree"] = False
    try:
        # FAQ -> Embedded is a publication command, not a plain edge insert.
        # The publisher creates the terminal edge only after the FAQ has a
        # materialized source, an approved status, and a traceable hierarchy.
        if target_node.get("node_type") == "embedded":
            source_node, source_item = _prepare_faq_for_embedded(source_node)
            source_id = source_node["id"]
            publication = approved_knowledge_snapshots.publish_approved_node(
                source_id,
                approved_by=(auth_service.current_user(request) or {}).get("id"),
                require_rag_for_faq=True,
            )
            embedded_edge_id = publication.get("embedded_edge_id")
            edge = supabase_client.get_knowledge_edge(embedded_edge_id) if embedded_edge_id else None
            if not edge:
                raise RuntimeError("Golden Dataset publication did not return an active FAQ -> Embedded edge")
            embedded_markdown.rebuild_embedded_markdown(target_node.get("persona_id") or edge_persona_id)
        else:
            edge = supabase_client.upsert_knowledge_edge(
                source_node_id=source_id,
                target_node_id=target_id,
                relation_type=relation_type,
                persona_id=edge_persona_id,
                weight=body.weight,
                metadata=metadata,
            )
        gallery_asset = None
        if edge and relation_type == "gallery_asset":
            gallery_content_node = target_node if source_node.get("node_type") == "gallery" else source_node
            gallery_asset = supabase_client.sync_gallery_asset_node(gallery_content_node, edge)
        if edge:
            evidence = {}
            if publication:
                evidence = {
                    "knowledge_item_id": (source_item or {}).get("id"),
                    "approved_snapshot_id": publication.get("approved_snapshot_id"),
                    "rag_entry_id": publication.get("rag_entry_id"),
                    "rag_chunk_ids": publication.get("rag_chunk_ids") or [],
                    "knowledge_rag_entry_id": publication.get("rag_entry_id"),
                    "knowledge_rag_chunk_ids": publication.get("rag_chunk_ids") or [],
                    "knowledge_node_id": source_node.get("id"),
                    "main_tree_edge_id": None,
                    "embedded_edge_id": edge.get("id"),
                    "final_status": publication.get("status") or "active",
                }
                _require_graph_promotion_evidence(evidence)
            supabase_client.insert_event(
                {
                    "event_type": "knowledge_path_linked",
                    "entity_type": "knowledge_edge",
                    "entity_id": edge.get("id"),
                    "persona_id": edge_persona_id,
                    "payload": {
                        "source_node_id": source_id,
                        "target_node_id": target_id,
                        "relation_type": relation_type,
                        "gallery_asset_id": (gallery_asset or {}).get("id"),
                        "golden_dataset_publication": bool(publication),
                        "evidence": evidence,
                    },
                },
                source="graph.create_edge",
            )
    except Exception as exc:
        emit(
            "graph_edge_create_failed",
            entity_type="knowledge_edge",
            persona_id=edge_persona_id,
            payload={
                "actor": current_actor(request),
                "context": {
                    "source_node_id": source_id,
                    "target_node_id": target_id,
                    "relation_type": relation_type,
                },
                "reason": str(exc)[:400],
            },
            level="error",
            source="routes.graph",
        )
        raise HTTPException(502, f"Could not create graph edge: {exc}") from exc
    if not edge:
        emit(
            "graph_edge_create_failed",
            entity_type="knowledge_edge",
            persona_id=edge_persona_id,
            payload={
                "actor": current_actor(request),
                "context": {
                    "source_node_id": source_id,
                    "target_node_id": target_id,
                    "relation_type": relation_type,
                },
                "reason": "upsert returned no row",
            },
            level="warn",
            source="routes.graph",
        )
        raise HTTPException(400, "Graph edge was not created")
    if publication:
        return {
            "ok": True,
            "success": True,
            "edge": edge,
            "publication": publication,
            "evidence": {
                "knowledge_item_id": (source_item or {}).get("id"),
                "approved_snapshot_id": publication.get("approved_snapshot_id"),
                "rag_entry_id": publication.get("rag_entry_id"),
                "rag_chunk_ids": publication.get("rag_chunk_ids") or [],
                "knowledge_rag_entry_id": publication.get("rag_entry_id"),
                "knowledge_rag_chunk_ids": publication.get("rag_chunk_ids") or [],
                "knowledge_node_id": source_node.get("id"),
                "main_tree_edge_id": None,
                "embedded_edge_id": edge.get("id"),
                "final_status": publication.get("status") or "active",
            },
            **publication,
        }
    return {"ok": True, "edge": edge}


@router.delete("/graph-edges/{edge_id}")
def delete_graph_edge(edge_id: str, request: Request):
    raw_edge_id = edge_id[3:] if edge_id.startswith("ge:") else edge_id
    edge = supabase_client.get_knowledge_edge(raw_edge_id)
    if edge and edge.get("persona_id"):
        auth_service.assert_persona_access(request, persona_id=edge.get("persona_id"))
    # When the edge being removed published a FAQ to Embedded, the Embedded body
    # must drop that FAQ's section after the soft-delete.
    target_node_before = supabase_client.get_knowledge_node((edge or {}).get("target_node_id")) if edge else None
    disconnects_embedded = bool(target_node_before and str(target_node_before.get("node_type") or "").lower() == "embedded")
    try:
        ok = supabase_client.delete_knowledge_edge(raw_edge_id)
    except Exception as exc:
        logger.exception("Could not delete graph edge", extra={"edge_id": edge_id, "raw_edge_id": raw_edge_id})
        raise HTTPException(502, f"Could not delete graph edge: {exc}") from exc
    if not ok:
        logger.warning("Graph edge not found for delete", extra={"edge_id": edge_id, "raw_edge_id": raw_edge_id})
        emit(
            "graph_edge_delete_failed",
            entity_type="knowledge_edge",
            entity_id=raw_edge_id,
            persona_id=(edge or {}).get("persona_id"),
            payload={"requested_edge_id": edge_id, "reason": "not_found"},
            level="warn",
            source="routes.graph",
        )
        raise HTTPException(404, "Graph edge not found")
    logger.info("Graph edge soft-deleted", extra={"edge_id": edge_id, "raw_edge_id": raw_edge_id})
    emit(
        "graph_edge_deleted",
        entity_type="knowledge_edge",
        entity_id=raw_edge_id,
        persona_id=(edge or {}).get("persona_id"),
        payload={
            "requested_edge_id": edge_id,
            "source_node_id": (edge or {}).get("source_node_id"),
            "target_node_id": (edge or {}).get("target_node_id"),
            "relation_type": (edge or {}).get("relation_type"),
            "metadata": (edge or {}).get("metadata") or {},
        },
        level="warn",
        source="routes.graph",
    )
    if disconnects_embedded:
        embedded_markdown.rebuild_embedded_markdown((edge or {}).get("persona_id") or (target_node_before or {}).get("persona_id"))
    return {"ok": True, "edge_id": raw_edge_id}


@router.delete("/graph-nodes/{node_id}")
def delete_graph_node(node_id: str, request: Request):
    raw_node_id = _raw_graph_node_id(node_id)
    node = supabase_client.get_knowledge_node(raw_node_id)
    if node and node.get("persona_id"):
        auth_service.assert_persona_access(request, persona_id=node.get("persona_id"))
    persona_id = (node or {}).get("persona_id")
    before_snapshot = {
        "node_type": (node or {}).get("node_type"),
        "slug": (node or {}).get("slug"),
        "title": (node or {}).get("title"),
        "status": (node or {}).get("status"),
        "source_table": (node or {}).get("source_table"),
        "source_id": (node or {}).get("source_id"),
    }
    actor = current_actor(request)
    if node and node.get("source_table") == "knowledge_items" and node.get("source_id"):
        try:
            evidence = knowledge_lifecycle.delete_knowledge_item_cascade(str(node.get("source_id")))
        except ValueError as exc:
            emit(
                "graph_node_delete_failed",
                entity_type="knowledge_node",
                entity_id=raw_node_id,
                persona_id=persona_id,
                payload={"actor": actor, "before": before_snapshot, "reason": str(exc)[:400], "stage": "cascade"},
                level="warn",
                source="routes.graph",
            )
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            emit(
                "graph_node_delete_failed",
                entity_type="knowledge_node",
                entity_id=raw_node_id,
                persona_id=persona_id,
                payload={"actor": actor, "before": before_snapshot, "reason": str(exc)[:400], "stage": "cascade"},
                level="error",
                source="routes.graph",
            )
            raise HTTPException(502, f"Could not delete knowledge item graph node: {exc}") from exc
        emit(
            "graph_node_deleted",
            entity_type="knowledge_node",
            entity_id=raw_node_id,
            persona_id=persona_id,
            payload={
                "actor": actor,
                "before": before_snapshot,
                "context": {"strategy": "knowledge_item_cascade", "evidence": evidence},
            },
            level="warn",
            source="routes.graph",
        )
        return {"ok": True, "node_id": raw_node_id, "evidence": evidence}
    try:
        ok = supabase_client.delete_knowledge_node(raw_node_id)
    except Exception as exc:
        emit(
            "graph_node_delete_failed",
            entity_type="knowledge_node",
            entity_id=raw_node_id,
            persona_id=persona_id,
            payload={"actor": actor, "before": before_snapshot, "reason": str(exc)[:400], "stage": "direct"},
            level="error",
            source="routes.graph",
        )
        raise HTTPException(502, f"Could not delete graph node: {exc}") from exc
    if not ok:
        emit(
            "graph_node_delete_failed",
            entity_type="knowledge_node",
            entity_id=raw_node_id,
            persona_id=persona_id,
            payload={"actor": actor, "before": before_snapshot, "reason": "not_found", "stage": "direct"},
            level="warn",
            source="routes.graph",
        )
        raise HTTPException(404, "Graph node not found")
    emit(
        "graph_node_deleted",
        entity_type="knowledge_node",
        entity_id=raw_node_id,
        persona_id=persona_id,
        payload={
            "actor": actor,
            "before": before_snapshot,
            "context": {"strategy": "direct"},
        },
        level="warn",
        source="routes.graph",
    )
    return {"ok": True, "node_id": raw_node_id}


class GraphNodeUpdateBody(BaseModel):
    title: Optional[str] = None
    markdown: Optional[str] = None
    summary: Optional[str] = None
    tags: Optional[list] = None
    status: Optional[str] = None


_PROTECTED_NODE_TYPES = {"persona", "embedded", "gallery"}
_FAQ_REVERT_STATUSES = {"approved", "embedded"}


@router.patch("/graph-nodes/{node_id}")
def update_graph_node(node_id: str, body: GraphNodeUpdateBody, request: Request):
    """Edit a node's contextual Markdown body (and title/summary/tags).

    The Markdown body is the node's real context: it is stored in
    `knowledge_nodes.metadata.markdown` (surfaced by graph-data) and, when the
    node mirrors a knowledge_item, mirrored into the item's content. Editing an
    approved/embedded FAQ reverts it to draft and withdraws it from Embedded
    (its section is removed from the Embedded body)."""
    raw_node_id = _raw_graph_node_id(node_id)
    node = supabase_client.get_knowledge_node(raw_node_id)
    if not node:
        raise HTTPException(404, "Graph node not found")
    persona_id = node.get("persona_id")
    if persona_id:
        auth_service.assert_persona_access(request, persona_id=persona_id)
    node_type = str(node.get("node_type") or "").lower()
    if node_type in _PROTECTED_NODE_TYPES:
        raise HTTPException(400, f"Node type '{node_type}' has an auto-managed body and cannot be edited directly")

    data = body.model_dump(exclude_none=True)
    if not data:
        raise HTTPException(400, "Nothing to update")

    item = _knowledge_item_for_graph_node(node)
    item_status = str((item or {}).get("status") or "").lower()
    markdown_changed = "markdown" in data
    reverted = bool(
        node_type == "faq"
        and markdown_changed
        and "status" not in data
        and item_status in _FAQ_REVERT_STATUSES
    )

    node_payload: dict = {}
    if "title" in data:
        node_payload["title"] = data["title"]
    if "summary" in data:
        node_payload["summary"] = data["summary"]
    if "tags" in data:
        node_payload["tags"] = data["tags"]
    if markdown_changed:
        node_payload["metadata"] = {**(node.get("metadata") or {}), "markdown": data["markdown"]}
    if "status" in data:
        node_payload["status"] = data["status"]
    elif reverted:
        node_payload["status"] = "pending_validation"

    updated = supabase_client.update_knowledge_node(raw_node_id, node_payload, mark_related_faqs=False)

    if item and item.get("id"):
        item_payload: dict = {}
        if "title" in data:
            item_payload["title"] = data["title"]
        if markdown_changed:
            item_payload["content"] = data["markdown"]
        if "tags" in data:
            item_payload["tags"] = data["tags"]
        if reverted:
            item_payload["status"] = "pending"
            item_payload["curation_status"] = "draft"
        if item_payload:
            supabase_client.update_knowledge_item(item["id"], item_payload)

    if reverted and item and item.get("id"):
        try:
            supabase_client.withdraw_faq_from_embedded(item["id"])
        except Exception:
            pass
        embedded_markdown.rebuild_embedded_markdown(persona_id)

    emit(
        "graph_node_markdown_updated",
        entity_type="knowledge_node",
        entity_id=raw_node_id,
        persona_id=persona_id,
        payload={"actor": current_actor(request), "fields": list(data.keys()), "reverted_to_draft": reverted},
        source="routes.graph",
    )
    return {"ok": True, "node": updated, "reverted_to_draft": reverted}


_APPOINTMENT_POLICY_TEXT_KEYS = (
    "atendimento_humano",
    "encaminhamento_excepcional",
    "esclarecimento_duvida",
    "encaminhamento_duvida_persistente",
    "complemento_confirmacao",
    "cabecalho_servicos",
    "saudacao_abertura",
    "sem_comparar_concorrentes",
    "mensagem_ausencia",
    "preco_humano",
    "avaliacao_presencial",
    "lacuna_conhecimento",
)


class AppointmentPolicyTextsBody(BaseModel):
    atendimento_humano: Optional[str] = None
    encaminhamento_excepcional: Optional[str] = None
    esclarecimento_duvida: Optional[str] = None
    encaminhamento_duvida_persistente: Optional[str] = None
    complemento_confirmacao: Optional[str] = None
    cabecalho_servicos: Optional[str] = None
    saudacao_abertura: Optional[str] = None
    sem_comparar_concorrentes: Optional[str] = None
    mensagem_ausencia: Optional[str] = None
    preco_humano: Optional[str] = None
    avaliacao_presencial: Optional[str] = None
    lacuna_conhecimento: Optional[str] = None


def _published_persona_node(persona_slug: str):
    """Load the persona node from the *published* graph (graph_json_v2_store).

    This is a different store from the live knowledge_nodes authoring table
    the rest of this file edits â€” appointment_policy only exists here, and
    it's what conversation_runtime._appointment_policy() actually reads at
    runtime, so reads/writes of it must go through this store too.
    """
    loaded = graph_json_v2_store.load_current(persona_slug)
    if not loaded:
        raise HTTPException(404, "Graph publicado nao encontrado para a persona.")
    version, graph = loaded
    persona_node = next((n for n in graph.nodes if n.node_type == "persona"), None)
    if not persona_node:
        raise HTTPException(404, "Node de persona nao encontrado no grafo publicado.")
    return version, graph, persona_node


@router.get("/personas/{persona_slug}/appointment-policy")
def get_persona_appointment_policy(persona_slug: str, request: Request):
    """Read a persona's handoff-message templates for the card editor UI."""
    auth_service.assert_persona_access(request, persona_slug=persona_slug)
    _version, _graph, persona_node = _published_persona_node(persona_slug)
    policy = (persona_node.data or {}).get("appointment_policy") or {}
    texts = {key: policy.get("texts", {}).get(key) for key in _APPOINTMENT_POLICY_TEXT_KEYS}
    return {"ok": True, "texts": texts}


@router.patch("/personas/{persona_slug}/appointment-policy")
def update_persona_appointment_policy(persona_slug: str, body: AppointmentPolicyTextsBody, request: Request):
    """Edit a persona's handoff-message templates (`data.appointment_policy.texts`).

    These are the fallback texts DeterministicAppointment sends when it
    hands a conversation off to a human (e.g. "vou chamar a equipe") â€” see
    docs/sdr/aurora/rules/no-auto-confirm.md. They live on the persona node
    of the *published* graph, so this goes through the same
    apply_operations + commit path agent_harness_tools.publish_patch uses
    for Sofia's own card edits, instead of a parallel write mechanism.
    """
    auth_service.assert_persona_capability(request, "edit", persona_slug=persona_slug)
    changes = {
        key: value
        for key, value in body.model_dump(exclude_none=True).items()
        if key in _APPOINTMENT_POLICY_TEXT_KEYS
    }
    if not changes:
        raise HTTPException(400, "Nothing to update")

    version, graph, persona_node = _published_persona_node(persona_slug)

    operation = {
        "op": "update_node",
        "node_id": persona_node.id,
        "patch": {
            f"data.appointment_policy.texts.{key}": value
            for key, value in changes.items()
        },
    }
    next_graph = graph_document_publisher.apply_operations(graph, [operation])
    actor = current_actor(request)
    result = graph_document_publisher.commit(
        graph=next_graph,
        persona_slug=persona_slug,
        brand_slug=graph.brand_slug,
        source="dashboard.persona_appointment_policy",
        reason="Atualizacao das mensagens de handoff via card da persona",
        published_by=actor.get("user_id"),
        expected_version=version,
        idempotency_key=f"appointment-policy:{persona_slug}:{uuid.uuid4()}",
    )
    persona_row = supabase_client.get_persona(persona_slug) or {}
    emit(
        "persona_appointment_policy_updated",
        entity_type="knowledge_node",
        entity_id=persona_node.id,
        persona_id=persona_row.get("id"),
        payload={"actor": actor, "fields": list(changes.keys())},
        source="routes.graph",
    )
    updated_persona = next(
        (n for n in next_graph.nodes if n.node_type == "persona"), None
    )
    return {
        "ok": True,
        "graph_version": result.get("graph_version"),
        "appointment_policy": (updated_persona.data or {}).get("appointment_policy") if updated_persona else None,
    }


# knowledge_items statuses â†’ nodeClass
_KI_STATUS: dict[str, str] = {
    "approved":       "validated",
    "embedded":       "validated",
    "pending":        "pending",
    "needs_persona":  "pending",
    "needs_category": "pending",
    "rejected":       "rejected",
}

# kb_entries statuses â†’ nodeClass
_KB_STATUS: dict[str, str] = {
    "ATIVO":   "validated",
    "INATIVO": "rejected",
}

# Auxiliary node_types: hidden by default in the UI, but data is preserved.
_AUXILIARY_NODE_TYPES: set[str] = {"tag", "mention"}
_TECHNICAL_NODE_TYPES: set[str] = {"knowledge_item", "kb_entry"}


def _is_import_audience_node(node: dict) -> bool:
    """The `import` leads bucket is operational, not a semantic audience. If an
    older mirror exists in knowledge_nodes we hide it from the tree (data is
    preserved in DB; this is a visibility filter only, never a delete)."""
    if (node.get("node_type") or "").strip().lower() != "audience":
        return False
    slug = str(node.get("slug") or "").strip().lower()
    meta = node.get("metadata") or {}
    source_type = str(meta.get("source_type") or "").strip().lower()
    return slug == "import" or source_type == "import"


def _canonicalize_semantic_node_types(sem_nodes: list[dict], nt_by_type: dict[str, dict]) -> list[dict]:
    """Repair legacy semantic mirrors in-memory for graph-data.

    Migration 031 backfills persisted offer nodes, but graph-data should not
    render a stale knowledge_item card in the primary tree while a database
    still has pre-migration rows.
    """
    repaired: list[dict] = []
    item_cache: dict[str, Optional[dict]] = {}
    for node in sem_nodes:
        next_node = dict(node)
        source_id = next_node.get("source_id")
        if next_node.get("source_table") == "knowledge_items" and source_id:
            meta = dict(next_node.get("metadata") or {})
            content_type = str(meta.get("content_type") or "").strip().lower()
            if not content_type:
                sid = str(source_id)
                if sid not in item_cache:
                    try:
                        item_cache[sid] = supabase_client.get_knowledge_item(sid)
                    except Exception:
                        item_cache[sid] = None
                content_type = str((item_cache.get(sid) or {}).get("content_type") or "").strip().lower()
            expected = knowledge_graph._CONTENT_TYPE_TO_NODE.get(content_type)
            current = str(next_node.get("node_type") or "").strip().lower()
            if expected and expected != current:
                meta.setdefault("runtime_repaired_node_type_from", current or "unknown")
                meta["content_type"] = content_type
                registry = nt_by_type.get(expected, {})
                next_node["node_type"] = expected
                next_node["metadata"] = meta
                next_node["level"] = registry.get("default_level", next_node.get("level"))
                next_node["importance"] = registry.get("default_importance", next_node.get("importance"))
        repaired.append(next_node)
    return repaired


def _demote_auxiliary_primary_edges(sem_edges: list[dict], nodes_by_id: dict[str, dict]) -> list[dict]:
    out: list[dict] = []
    for edge in sem_edges:
        src_type = str((nodes_by_id.get(edge.get("source_node_id")) or {}).get("node_type") or "").lower()
        tgt_type = str((nodes_by_id.get(edge.get("target_node_id")) or {}).get("node_type") or "").lower()
        if (
            src_type in _AUXILIARY_NODE_TYPES
            or tgt_type in _AUXILIARY_NODE_TYPES
            or src_type in _TECHNICAL_NODE_TYPES
            or tgt_type in _TECHNICAL_NODE_TYPES
        ):
            next_edge = dict(edge)
            meta = dict(next_edge.get("metadata") or {})
            meta["primary_tree"] = False
            meta["graph_layer"] = "semantic_tags" if edge.get("relation_type") == "has_tag" else "auxiliary"
            meta["visual_hidden"] = True
            next_edge["metadata"] = meta
            out.append(next_edge)
            continue
        out.append(edge)
    return out


def _is_quarantined_metadata(metadata: Optional[dict]) -> bool:
    meta = metadata or {}
    state = str(meta.get("quarantine_state") or "").strip().lower()
    return state in {"legacy_orphaned", "structural"}

def _metadata_asset_url(metadata: Optional[dict]) -> str:
    meta = metadata or {}
    return str(
        meta.get("url")
        or meta.get("public_url")
        or meta.get("asset_url")
        or meta.get("file_url")
        or ""
    )


def _metadata_file_type(metadata: Optional[dict]) -> str:
    meta = metadata or {}
    explicit = str(meta.get("file_type") or meta.get("asset_type") or "").lower()
    if explicit in {"image", "video"}:
        path = str(meta.get("file_path") or meta.get("storage_path") or meta.get("original_filename") or "")
    else:
        path = str(
            meta.get("file_path")
            or meta.get("storage_path")
            or meta.get("original_filename")
            or _metadata_asset_url(meta)
            or ""
        )
    ext = path.rsplit("?", 1)[0].rsplit(".", 1)[-1].lower() if "." in path else ""
    if ext in {"png", "jpg", "jpeg", "webp", "svg", "gif"}:
        return ext
    if ext in {"mp4", "mov", "webm"}:
        return ext
    return explicit


def _metadata_markdown(metadata: Optional[dict], fallback: str = "") -> str:
    meta = metadata or {}
    return str(meta.get("markdown") or meta.get("body") or meta.get("content") or fallback or "")

# Edge tier overrides â€” relations whose tier is fixed regardless of weight.
_STRUCTURAL_RELATIONS: set[str] = {
    "belongs_to_persona",
    "contains",
    "part_of_campaign",
    "targets_audience",
    "campaign_has_audience",
    "audience_has_product_group",
    "product_group_has_product",
    "about_product",
    "offers_product",
    "briefed_by",
    "answers_question",
    "product_has_copy",
    "product_has_faq",
    "copy_has_faq",
    "supports_copy",
    "uses_asset",
    "manual",
    "gallery_asset",
}
_AUXILIARY_RELATIONS: set[str] = {"has_tag", "mentions", "same_topic_as", "visible_to_agent"}
_CURATION_RELATIONS: set[str] = {"duplicate_of"}


def _classify_relation_tier(relation_type: str, weight: float) -> str:
    """Map relation_type + default_weight â†’ tier {strong|structural|auxiliary|curation}."""
    rt = (relation_type or "").lower()
    if rt in _CURATION_RELATIONS:
        return "curation"
    if rt in _STRUCTURAL_RELATIONS:
        return "structural"
    if rt in _AUXILIARY_RELATIONS:
        return "auxiliary"
    if weight is not None and weight >= 0.70:
        return "strong"
    return "auxiliary"


def _semantic_chain_to_persona(
    node_id: str,
    parent_by_child: dict[str, str],
    persona_node_ids: set[str],
) -> tuple[list[str], bool]:
    chain: list[str] = []
    current = node_id
    seen: set[str] = set()
    while current and current not in seen and len(chain) < 64:
        seen.add(current)
        chain.append(current)
        current = parent_by_child.get(current)
    return chain, bool(chain and chain[-1] in persona_node_ids)


def _resolve_focus(focus: str, persona_id: Optional[str]) -> Optional[dict]:
    """Resolve a `focus` query param into the matching knowledge_node row.

    Accepts either:
      - "<node_type>:<slug>" (preferred â€” matches what _link_target generates)
      - a bare UUID (knowledge_nodes.id)
    """
    if not focus:
        return None
    client = supabase_client.get_client()
    try:
        if ":" in focus:
            node_type, slug = focus.split(":", 1)
            q = client.table("knowledge_nodes").select("*").eq("node_type", node_type.strip()).eq("slug", slug.strip())
            if persona_id:
                q = q.eq("persona_id", persona_id)
            rows = q.limit(1).execute().data or []
            return rows[0] if rows else None
        # Bare UUID lookup
        q = client.table("knowledge_nodes").select("*").eq("id", focus).limit(1)
        rows = q.execute().data or []
        return rows[0] if rows else None
    except Exception:
        return None


def _bfs_focus_subgraph(
    focus_node_id: str,
    all_nodes: list[dict],
    all_edges: list[dict],
    max_depth: int,
) -> tuple[set[str], dict[str, int], dict[str, tuple[str, str, str]]]:
    """BFS from focus_node_id through the directed graph (treated as undirected
    for navigation), returning:
      - reachable: set of node_ids within max_depth hops
      - distance: node_id -> hops from focus
      - predecessor: node_id -> (parent_id, relation_type, direction "in"|"out")
    """
    node_ids = {n["id"] for n in all_nodes if n.get("id")}
    if focus_node_id not in node_ids:
        return set(), {}, {}

    adj: dict[str, list[tuple[str, str, str]]] = {}
    for e in all_edges:
        src, tgt = e.get("source_node_id"), e.get("target_node_id")
        rel = e.get("relation_type") or "related"
        if not src or not tgt:
            continue
        adj.setdefault(src, []).append((tgt, rel, "out"))
        adj.setdefault(tgt, []).append((src, rel, "in"))

    distance: dict[str, int] = {focus_node_id: 0}
    predecessor: dict[str, tuple[str, str, str]] = {}
    queue: deque[str] = deque([focus_node_id])
    while queue:
        nid = queue.popleft()
        d = distance[nid]
        if d >= max_depth:
            continue
        for (nb, rel, direction) in adj.get(nid, []):
            if nb in distance or nb not in node_ids:
                continue
            distance[nb] = d + 1
            predecessor[nb] = (nid, rel, direction)
            queue.append(nb)
    return set(distance.keys()), distance, predecessor


def _build_focus_path(
    focus_node_id: str,
    persona_node_id: Optional[str],
    nodes_by_id: dict[str, dict],
    predecessor: dict[str, tuple[str, str, str]],
) -> list[dict]:
    """Build a breadcrumb persona â†’ ... â†’ focus.

    The BFS is rooted at focus, so `predecessor[X] = (Y, rel, dir)` tells us
    that during the search we discovered X from Y. Walking predecessors from
    persona therefore gives [persona, ..., focus]. The relation/direction
    on each step describes the edge from the *next* node to the previous in
    the BFS direction; we re-orient it so the breadcrumb reads forward
    persona â†’ focus.
    """
    if not persona_node_id or persona_node_id == focus_node_id:
        # Trivial path: only the focus node itself.
        node = nodes_by_id.get(focus_node_id, {})
        return [{
            "node_id": focus_node_id,
            "slug": node.get("slug"),
            "title": node.get("title"),
            "node_type": node.get("node_type"),
            "relation_type": None,
            "direction": None,
        }]
    if persona_node_id not in predecessor:
        # Persona wasn't reached by BFS â€” return just the focus node.
        node = nodes_by_id.get(focus_node_id, {})
        return [{
            "node_id": focus_node_id,
            "slug": node.get("slug"),
            "title": node.get("title"),
            "node_type": node.get("node_type"),
            "relation_type": None,
            "direction": None,
        }]

    # Walk persona â†’ ... â†’ focus via the BFS predecessor chain.
    raw: list[tuple[str, Optional[str], Optional[str]]] = []
    cur: Optional[str] = persona_node_id
    guard = 0
    seen: set[str] = set()
    while cur is not None and guard < 32 and cur not in seen:
        seen.add(cur)
        rel: Optional[str] = None
        direction: Optional[str] = None
        if cur in predecessor:
            _, rel, direction = predecessor[cur]
        raw.append((cur, rel, direction))
        if cur == focus_node_id:
            break
        nxt = predecessor.get(cur, (None,))[0]
        cur = nxt
        guard += 1

    chain: list[dict] = []
    for nid, rel, direction in raw:
        node = nodes_by_id.get(nid, {})
        chain.append({
            "node_id": nid,
            "slug": node.get("slug"),
            "title": node.get("title"),
            "node_type": node.get("node_type"),
            "relation_type": rel,
            "direction": direction,
        })
    return chain


def _compute_primary_depths(nodes: list[dict], edges: list[dict]) -> tuple[dict[str, int], dict[str, str]]:
    """Return node depth and parent map from the current structural graph."""
    node_ids = {n["id"] for n in nodes if n.get("id")}
    persona_ids = {n["id"] for n in nodes if n.get("node_type") == "persona"}
    structural = _STRUCTURAL_RELATIONS | {"manual", "contains", "parent_of", "belongs_to", "part_of"}
    parent_by_child: dict[str, str] = {}
    parent_priority_by_child: dict[str, int] = {}
    for edge in edges:
        src = edge.get("source_node_id")
        tgt = edge.get("target_node_id")
        if not src or not tgt or src not in node_ids or tgt not in node_ids:
            continue
        relation = (edge.get("relation_type") or "").lower()
        meta = edge.get("metadata") or {}
        src_node = next((n for n in nodes if n.get("id") == src), {}) or {}
        tgt_node = next((n for n in nodes if n.get("id") == tgt), {}) or {}
        if (src_node.get("node_type") or "").lower() in _AUXILIARY_NODE_TYPES:
            continue
        if (tgt_node.get("node_type") or "").lower() in {"persona", "mention", "tag"}:
            continue
        if meta.get("primary_tree") is not True:
            continue
        priority = 0
        priority += 10
        if relation == "manual":
            priority += 100
        if tgt not in parent_by_child or priority >= parent_priority_by_child.get(tgt, -1):
            parent_by_child[tgt] = src
            parent_priority_by_child[tgt] = priority

    children_by_parent: dict[str, list[str]] = {}
    for child_id, parent_id in parent_by_child.items():
        children_by_parent.setdefault(parent_id, []).append(child_id)

    depth: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque((node_id, 0) for node_id in persona_ids)
    while queue:
        node_id, d = queue.popleft()
        if node_id in depth and depth[node_id] <= d:
            continue
        depth[node_id] = d
        for child_id in children_by_parent.get(node_id, []):
            queue.append((child_id, d + 1))

    # Any disconnected semantic node gets a depth after the deepest connected node.
    fallback_depth = (max(depth.values()) + 1) if depth else 1
    for node_id in node_ids:
        depth.setdefault(node_id, fallback_depth)
    return depth, parent_by_child


def _focus_path_from_parents(
    focus_node_id: str,
    parent_by_child: dict[str, str],
    nodes_by_id: dict[str, dict],
) -> list[dict]:
    chain_ids: list[str] = []
    seen: set[str] = set()
    current: Optional[str] = focus_node_id
    while current and current not in seen and len(chain_ids) < 64:
        seen.add(current)
        chain_ids.append(current)
        current = parent_by_child.get(current)
    chain_ids.reverse()
    out: list[dict] = []
    for node_id in chain_ids:
        node = nodes_by_id.get(node_id, {})
        out.append({
            "node_id": node_id,
            "slug": node.get("slug"),
            "title": node.get("title"),
            "node_type": node.get("node_type"),
            "relation_type": None,
            "direction": None,
        })
    return out


@router.get("/graph-data")
def get_graph_data(
    request: Request,
    persona_slug: Optional[str] = Query(None),
    focus: Optional[str] = Query(None, description="<node_type>:<slug> or node_id"),
    max_depth: int = Query(3, ge=1, le=6, description="BFS depth from focus (or persona root)"),
    include_tags: bool = Query(False),
    include_mentions: bool = Query(False),
    include_technical: bool = Query(False, description="Include knowledge_items + kb_entries duplicate sources"),
    include_embedded: bool = Query(True, description="Include synthetic RAG embedded node and validated links"),
    mode: str = Query("layered", description="layered|semantic_tree|graph (frontend hint only)"),
):
    personas = supabase_client.get_personas()
    user = auth_service.current_user(request)
    access = auth_service.allowed_access(request)
    personas = auth_service.filter_personas_for_user(user, personas, access)
    if persona_slug:
        personas = [p for p in personas if p["slug"] == persona_slug]
        if not personas:
            raise HTTPException(403, "Acesso negado para esta persona.")

    persona_map = {p["id"]: p for p in personas}
    persona_id_set = set(persona_map.keys())

    single_persona_id: Optional[str] = None
    if persona_slug and len(personas) == 1:
        single_persona_id = personas[0]["id"]
    for p in personas:
        try:
            supabase_client.ensure_gallery_node(p["id"])
            supabase_client.ensure_embedded_node(p["id"])
        except Exception:
            logger.exception("Could not ensure protected graph nodes", extra={"persona_id": p.get("id")})

    # â”€â”€ Registries (cached) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    node_type_registry = supabase_client.get_node_type_registry()
    relation_type_registry = supabase_client.get_relation_type_registry()
    nt_by_type = {r["node_type"]: r for r in node_type_registry}
    rt_by_type = {r["relation_type"]: r for r in relation_type_registry}

    # â”€â”€ Source: semantic graph (knowledge_nodes/edges) is the truth â”€â”€â”€â”€â”€
    try:
        sem_nodes, sem_edges = supabase_client.list_all_knowledge_graph(
            persona_id=single_persona_id, limit_nodes=1500,
        )
    except Exception:
        sem_nodes, sem_edges = [], []

    sem_nodes = _canonicalize_semantic_node_types(sem_nodes, nt_by_type)
    sem_nodes_by_raw_id = {n.get("id"): n for n in sem_nodes if n.get("id")}
    sem_edges = _demote_auxiliary_primary_edges(sem_edges, sem_nodes_by_raw_id)

    # Apply visibility filters early (still keeps data in DB).
    if not include_tags:
        sem_nodes = [n for n in sem_nodes if n.get("node_type") != "tag"]
    if not include_mentions:
        sem_nodes = [n for n in sem_nodes if n.get("node_type") != "mention"]
    if not include_technical:
        sem_nodes = [n for n in sem_nodes if (n.get("node_type") or "").lower() not in _TECHNICAL_NODE_TYPES]
    sem_nodes = [n for n in sem_nodes if n.get("status") != "archived"]
    sem_nodes = [n for n in sem_nodes if not _is_quarantined_metadata(n.get("metadata"))]
    sem_nodes = [n for n in sem_nodes if not _is_import_audience_node(n)]

    surviving_ids = {n["id"] for n in sem_nodes}
    sem_edges = [
        e for e in sem_edges
        if e.get("source_node_id") in surviving_ids and e.get("target_node_id") in surviving_ids
    ]
    sem_edges = [
        e for e in sem_edges
        if (e.get("metadata") or {}).get("active") is not False
        and not (e.get("metadata") or {}).get("visual_hidden")
        and not (
            (e.get("relation_type") or "").lower() == "belongs_to_persona"
            and (e.get("metadata") or {}).get("primary_tree") is not True
        )
    ]
    if (mode or "layered") != "graph":
        visible_tree_edges = []
        primary_relation_seen: set[tuple[str, str, str]] = set()
        relation_rank = {"offers_product": 0, "supports_copy": 1, "answers_question": 2, "contains": 3}
        for edge in sorted(sem_edges, key=lambda e: relation_rank.get((e.get("relation_type") or "").lower(), 9)):
            meta = edge.get("metadata") or {}
            src_node = next((n for n in sem_nodes if n.get("id") == edge.get("source_node_id")), {}) or {}
            tgt_node = next((n for n in sem_nodes if n.get("id") == edge.get("target_node_id")), {}) or {}
            src_type = (src_node.get("node_type") or "").lower()
            tgt_type = (tgt_node.get("node_type") or "").lower()
            rt = (edge.get("relation_type") or "").lower()
            terminal_edge = (src_type == "faq" and tgt_type == "embedded") or rt == "gallery_asset"
            if meta.get("primary_tree") is True:
                relation_key = (
                    str(edge.get("source_node_id") or ""),
                    str(edge.get("target_node_id") or ""),
                    rt,
                )
                if relation_key in primary_relation_seen:
                    continue
                primary_relation_seen.add(relation_key)
                visible_tree_edges.append(edge)
            elif terminal_edge:
                visible_tree_edges.append(edge)
        sem_edges = visible_tree_edges
    graph_depths, parent_by_child = _compute_primary_depths(sem_nodes, sem_edges)

    # â”€â”€ Optional: focus subgraph (BFS) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    focus_node = _resolve_focus(focus, single_persona_id) if focus else None
    focus_path: list[dict] = []
    if focus_node:
        nodes_by_id_raw = {n["id"]: n for n in sem_nodes}
        if focus_node["id"] not in nodes_by_id_raw:
            # Add it back if it was filtered out by include_*
            sem_nodes.append(focus_node)
            nodes_by_id_raw[focus_node["id"]] = focus_node
        structural_focus_path = _focus_path_from_parents(focus_node["id"], parent_by_child, nodes_by_id_raw)
        reachable, distance_map, predecessor = _bfs_focus_subgraph(
            focus_node["id"], sem_nodes, sem_edges, max_depth=max_depth,
        )
        reachable.update(step["node_id"] for step in structural_focus_path if step.get("node_id"))
        sem_nodes = [n for n in sem_nodes if n["id"] in reachable]
        sem_edges = [
            e for e in sem_edges
            if e.get("source_node_id") in reachable and e.get("target_node_id") in reachable
        ]
        focus_path = structural_focus_path
        if len(focus_path) <= 1:
            persona_root_id: Optional[str] = None
            for n in sem_nodes:
                if n.get("node_type") == "persona" and (n.get("persona_id") in persona_id_set or n.get("slug") == "self"):
                    persona_root_id = n["id"]
                    break
            nodes_by_id = {n["id"]: n for n in sem_nodes}
            focus_path = _build_focus_path(focus_node["id"], persona_root_id, nodes_by_id, predecessor)
    else:
        distance_map = {}

    current_depths, _ = _compute_primary_depths(sem_nodes, sem_edges)
    max_current_depth = max((d for d in current_depths.values() if d > 0), default=1)
    sem_nodes_by_id = {n["id"]: n for n in sem_nodes if n.get("id")}
    try:
        snapshot_by_node = supabase_client.list_approved_snapshots_for_nodes(list(sem_nodes_by_id.keys()))
        rag_chunk_counts = supabase_client.count_knowledge_rag_chunks_by_entry_ids(
            [
                snapshot.get("rag_entry_id")
                for snapshot in snapshot_by_node.values()
                if snapshot.get("rag_entry_id")
            ]
        )
    except Exception:
        snapshot_by_node = {}
        rag_chunk_counts = {}
    semantic_persona_node_ids = {n["id"] for n in sem_nodes if n.get("node_type") == "persona" and n.get("id")}
    validated_semantic_node_ids = {
        n["id"]
        for n in sem_nodes
        if n.get("id") and (
            (n.get("node_type") == "persona")
            or knowledge_graph._validation_state(n) == "validated"
        )
    }
    path_connected_semantic_node_ids: set[str] = set()
    branch_complete_semantic_node_ids: set[str] = set()
    branch_complete_edge_ids: set[str] = set()
    terminal_connected_semantic_node_ids: set[str] = set()
    primary_edge_by_pair: dict[tuple[str, str], str] = {}
    for edge in sem_edges:
        src = edge.get("source_node_id")
        tgt = edge.get("target_node_id")
        if not src or not tgt:
            continue
        relation = (edge.get("relation_type") or "").lower()
        meta = edge.get("metadata") or {}
        if meta.get("primary_tree") or relation in _STRUCTURAL_RELATIONS:
            if parent_by_child.get(tgt) == src:
                primary_edge_by_pair[(tgt, src)] = edge.get("id")

    for node in sem_nodes:
        node_id = node.get("id")
        ntype = (node.get("node_type") or "").lower()
        if not node_id or ntype in {"persona", "embedded", "gallery"}:
            continue
        chain, reaches_persona = _semantic_chain_to_persona(node_id, parent_by_child, semantic_persona_node_ids)
        if reaches_persona:
            path_connected_semantic_node_ids.update(chain)

    for edge in sem_edges:
        src_id = edge.get("source_node_id")
        tgt_id = edge.get("target_node_id")
        if not src_id or not tgt_id:
            continue
        src_node = sem_nodes_by_id.get(src_id) or {}
        tgt_node = sem_nodes_by_id.get(tgt_id) or {}
        src_type = (src_node.get("node_type") or "").lower()
        tgt_type = (tgt_node.get("node_type") or "").lower()
        rt = (edge.get("relation_type") or "").lower()
        terminal_node_id: Optional[str] = None
        if src_type == "faq" and tgt_type == "embedded":
            terminal_node_id = src_id
        elif rt == "gallery_asset":
            if src_type == "gallery" and tgt_type != "gallery":
                terminal_node_id = tgt_id
            elif tgt_type == "gallery" and src_type != "gallery":
                terminal_node_id = src_id
        if not terminal_node_id:
            continue
        terminal_connected_semantic_node_ids.add(terminal_node_id)
        chain, reaches_persona = _semantic_chain_to_persona(terminal_node_id, parent_by_child, semantic_persona_node_ids)
        if not reaches_persona:
            continue
        if not all(node_ref in validated_semantic_node_ids for node_ref in chain):
            continue
        branch_complete_semantic_node_ids.update(chain)
        branch_complete_edge_ids.add(edge.get("id"))
        for child_id, parent_id in zip(chain, chain[1:]):
            primary_edge_id = primary_edge_by_pair.get((child_id, parent_id))
            if primary_edge_id:
                branch_complete_edge_ids.add(primary_edge_id)

    def current_level_for(node_id: str, node_type: str) -> int:
        if node_type == "persona":
            return 0
        if node_type == "gallery":
            return 112
        if node_type == "embedded":
            return 120
        depth = max(1, current_depths.get(node_id, max_current_depth))
        step = 99 / max(1, max_current_depth)
        return max(1, int(round(100 - depth * step)))

    # â”€â”€ Build response payload â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    nodes: list[dict] = []
    edges: list[dict] = []

    # Persona root nodes â€” only when not focus-scoped, or when focus reached
    # the persona. (If focus excluded the persona, we still want to show it.)
    persona_node_ids_emitted: set[str] = set()
    sem_persona_ids = {n.get("persona_id") for n in sem_nodes if n.get("node_type") == "persona"}
    for p in personas:
        # If we're scoped to focus and persona didn't survive BFS, skip.
        if focus_node and p["id"] not in {n.get("persona_id") for n in sem_nodes}:
            continue
        nodes.append({
            "id": f"persona:{p['id']}",
            "type": "personaNode",
            "position": {"x": 0, "y": 0},
            "data": {
                "label": p.get("name", p["slug"]),
                "slug": p["slug"],
                "description": p.get("description", ""),
                "nodeClass": "persona",
                "node_type": "persona",
                "persona_id": p["id"],
                "persona_slug": p.get("slug"),
                "level": 0,
                "importance": 1.0,
                "color": nt_by_type.get("persona", {}).get("color", "#7c6fff"),
                "icon": nt_by_type.get("persona", {}).get("icon", "user"),
                "is_auxiliary": False,
                "validated": True,
                "path_connected_to_persona": True,
                "terminal_connected": False,
                "branch_complete_validated": bool(branch_complete_semantic_node_ids),
                "in_focus_path": any(
                    step.get("node_type") == "persona" and step.get("node_id") in {
                        n.get("id") for n in sem_nodes if n.get("node_type") == "persona" and n.get("persona_id") == p["id"]
                    }
                    for step in focus_path
                ),
            },
        })
        persona_node_ids_emitted.add(p["id"])

    mirrored_knowledge_item_ids = {
        str(n.get("source_id"))
        for n in sem_nodes
        if n.get("source_table") == "knowledge_items" and n.get("source_id")
    }

    # â”€â”€ Optional technical sources (ki:/kb:) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ki_items: list[dict] = []
    kb_entries: list[dict] = []
    try:
        ki_items = supabase_client.get_knowledge_items(persona_id=single_persona_id, limit=500, offset=0) or []
    except Exception:
        ki_items = []
    pending_graph_items = [
        item for item in ki_items
        if include_technical
        and str(item.get("status") or "").lower() in {"pending", "needs_persona", "needs_category", "draft", "approved"}
        and str(item.get("id")) not in mirrored_knowledge_item_ids
        and not _is_quarantined_metadata((item.get("metadata") or {}))
    ]
    pending_graph_item_ids = {item.get("id") for item in pending_graph_items}
    for item in pending_graph_items:
        nid = f"ki:{item['id']}"
        persona_id = item.get("persona_id")
        node_class = _KI_STATUS.get(item.get("status", ""), "pending")
        content_type = item.get("content_type", "other")
        file_type = (item.get("file_type") or "").lower()
        nodes.append({
            "id": nid, "type": "knowledgeNode", "position": {"x": 0, "y": 0},
            "data": {
                "label": item.get("title") or content_type,
                "status": item.get("status", "pending"),
                "content_type": content_type,
                "file_type": file_type,
                "file_path": item.get("file_path"),
                "content_preview": (item.get("content") or "")[:200],
                "nodeClass": node_class,
                "item_id": item["id"],
                "source": "queue",
                "tags": item.get("tags") or [],
                "node_type": "knowledge_item",
                "level": nt_by_type.get("knowledge_item", {}).get("default_level", 95),
                "importance": nt_by_type.get("knowledge_item", {}).get("default_importance", 0.40),
                "color": nt_by_type.get("knowledge_item", {}).get("color", "#94a3b8"),
                "icon": nt_by_type.get("knowledge_item", {}).get("icon", "inbox"),
                "is_auxiliary": False,
                "validated": node_class == "validated",
                "approval_state": item.get("status", "pending"),
                "publication_state": "published" if str(item.get("status") or "").lower() == "embedded" else "unpublished",
                "path_connected_to_persona": bool(persona_id and persona_id in persona_node_ids_emitted),
                "terminal_connected": False,
                "branch_complete_validated": False,
            },
        })
        if persona_id and persona_id in persona_node_ids_emitted:
            edges.append({
                "id": f"e:ki-{persona_id}-{item['id']}",
                "source": f"persona:{persona_id}", "target": nid, "type": "smoothstep",
                "data": {
                    "relation_type": "draft_member",
                    "tier": "auxiliary",
                    "weight": 0.45,
                    "directional": True,
                    "draft_edge": True,
                    "primary_tree": False,
                },
            })

    if include_technical:
        try:
            kb_entries = supabase_client.get_kb_entries(persona_id=single_persona_id, status="") or []
        except Exception:
            kb_entries = []

        for item in ki_items:
            if item.get("id") in pending_graph_item_ids:
                continue
            if str(item.get("id")) in mirrored_knowledge_item_ids:
                continue
            nid = f"ki:{item['id']}"
            persona_id = item.get("persona_id")
            node_class = _KI_STATUS.get(item.get("status", ""), "pending")
            content_type = item.get("content_type", "other")
            file_type = (item.get("file_type") or "").lower()
            nodes.append({
                "id": nid, "type": "knowledgeNode", "position": {"x": 0, "y": 0},
                "data": {
                    "label": item.get("title") or content_type,
                    "status": item.get("status", "pending"),
                    "content_type": content_type,
                    "file_type": file_type,
                    "file_path": item.get("file_path"),
                    "content_preview": (item.get("content") or "")[:200],
                    "nodeClass": node_class,
                    "item_id": item["id"],
                    "source": "queue",
                    "tags": item.get("tags") or [],
                    "node_type": "knowledge_item",
                    "level": nt_by_type.get("knowledge_item", {}).get("default_level", 95),
                    "importance": nt_by_type.get("knowledge_item", {}).get("default_importance", 0.40),
                    "color": nt_by_type.get("knowledge_item", {}).get("color", "#94a3b8"),
                    "icon": nt_by_type.get("knowledge_item", {}).get("icon", "inbox"),
                    "is_auxiliary": True,
                    "validated": node_class == "validated",
                    "approval_state": item.get("status", "pending"),
                    "publication_state": "published" if str(item.get("status") or "").lower() == "embedded" else "unpublished",
                    "path_connected_to_persona": bool(persona_id and persona_id in persona_node_ids_emitted),
                    "terminal_connected": False,
                    "branch_complete_validated": False,
                },
            })
            if persona_id and persona_id in persona_node_ids_emitted:
                edges.append({
                    "id": f"e:ki-{persona_id}-{item['id']}",
                    "source": f"persona:{persona_id}", "target": nid, "type": "smoothstep",
                    "data": {"relation_type": "belongs_to_persona", "tier": "structural", "weight": 1.0, "directional": True},
                })

        for entry in kb_entries:
            nid = f"kb:{entry['id']}"
            persona_id = entry.get("persona_id")
            if persona_slug and (not persona_id or persona_id not in persona_id_set):
                continue
            node_class = _KB_STATUS.get(entry.get("status", ""), "pending")
            tipo = entry.get("tipo") or "kb"
            title = entry.get("titulo") or entry.get("produto") or entry.get("intencao") or tipo
            nodes.append({
                "id": nid, "type": "knowledgeNode", "position": {"x": 0, "y": 0},
                "data": {
                    "label": title,
                    "status": entry.get("status", "ATIVO"),
                    "content_type": tipo,
                    "file_type": "",
                    "file_path": entry.get("link"),
                    "content_preview": (entry.get("conteudo") or "")[:200],
                    "nodeClass": node_class,
                    "item_id": entry["id"],
                    "source": "vault",
                    "tags": entry.get("tags") or [],
                    "node_type": "kb_entry",
                    "level": nt_by_type.get("kb_entry", {}).get("default_level", 95),
                    "importance": nt_by_type.get("kb_entry", {}).get("default_importance", 0.50),
                    "color": nt_by_type.get("kb_entry", {}).get("color", "#94a3b8"),
                    "icon": nt_by_type.get("kb_entry", {}).get("icon", "database"),
                    "is_auxiliary": True,
                    "validated": node_class == "validated",
                    "approval_state": entry.get("status", "ATIVO"),
                    "publication_state": "published",
                    "path_connected_to_persona": bool(persona_id and persona_id in persona_node_ids_emitted),
                    "terminal_connected": False,
                    "branch_complete_validated": False,
                },
            })
            if persona_id and persona_id in persona_node_ids_emitted:
                edges.append({
                    "id": f"e:kb-{persona_id}-{entry['id']}",
                    "source": f"persona:{persona_id}", "target": nid, "type": "smoothstep",
                    "data": {"relation_type": "belongs_to_persona", "tier": "structural", "weight": 1.0, "directional": True},
                })

    # â”€â”€ Semantic graph (the actual knowledge nodes/edges) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    semantic_nodes_count = 0
    semantic_edges_count = 0
    sem_node_ids = {n["id"]: n for n in sem_nodes}
    semantic_persona_aliases: dict[str, str] = {}
    semantic_embedded_aliases: dict[str, str] = {}
    semantic_gallery_aliases: dict[str, str] = {}

    for n in sem_nodes:
        ntype = (n.get("node_type") or "").lower()
        if ntype == "embedded":
            pid = n.get("persona_id") or single_persona_id
            if pid and pid in persona_node_ids_emitted:
                semantic_embedded_aliases[n["id"]] = f"embedded:{pid}"
            continue
        if ntype == "gallery":
            pid = n.get("persona_id") or single_persona_id
            if pid and pid in persona_node_ids_emitted:
                semantic_gallery_aliases[n["id"]] = f"gallery:{pid}"
            continue
        if ntype != "persona":
            continue
        pid = n.get("persona_id") or single_persona_id
        if pid and pid in persona_node_ids_emitted:
            semantic_persona_aliases[n["id"]] = f"persona:{pid}"

    for n in sem_nodes:
        ntype = (n.get("node_type") or "kb").lower()
        if ntype == "persona":
            # The UI already emits a stable persona root card with the real
            # filtered entity name. Skip semantic duplicates like "Persona".
            continue
        if ntype in {"embedded", "gallery"}:
            # The UI emits a stable Embedded destination card per persona.
            # Semantic Embedded/Gallery nodes are aliases for persisted edges only.
            continue
        meta = n.get("metadata") or {}
        tags = n.get("tags") or []
        nid = f"gn:{n['id']}"

        registry_row = nt_by_type.get(ntype, {})
        level = current_level_for(n["id"], ntype)
        importance = max(0.01, min(1.0, level / 99))
        confidence = n.get("confidence")
        is_auxiliary = ntype in _AUXILIARY_NODE_TYPES or (level or 0) >= 90

        # Map semantic node_type â†’ ReactFlow nodeClass for legacy color/shape.
        semantic_state = knowledge_graph._validation_state(n)
        source_status = str(meta.get("source_status") or n.get("status") or "").lower()
        asset_validation_status = str(meta.get("validation_status") or meta.get("approval_status") or "").lower()
        if ntype == "asset" and asset_validation_status:
            if asset_validation_status in {"approved", "embedded", "validated"}:
                semantic_state = "validated"
                source_status = asset_validation_status
            else:
                semantic_state = "pending"
                source_status = asset_validation_status
        approved_snapshot = snapshot_by_node.get(n.get("id")) or {}
        approved_snapshot_id = approved_snapshot.get("id") or meta.get("approved_snapshot_id")
        rag_entry_id = approved_snapshot.get("rag_entry_id") or meta.get("knowledge_rag_entry_id")
        rag_chunk_count = rag_chunk_counts.get(str(rag_entry_id or ""), 0)
        snapshot_status = approved_snapshot.get("status") or meta.get("snapshot_status")
        publication_state = (
            "active"
            if snapshot_status == "active" and rag_entry_id and rag_chunk_count > 0
            else "published"
            if source_status == "embedded" or n.get("source_table") == "kb_entries"
            else "approved"
            if approved_snapshot_id
            else "unpublished"
        )
        node_class = {
            "persona": "persona",
            "gallery": "validated",
            "embedded": "validated",
            "mention": "pending",
            "tag": "pending",
        }.get(ntype, "validated" if semantic_state == "validated" else "pending")

        # focus highlights
        is_focus = bool(focus_node and n.get("id") == focus_node.get("id"))
        in_focus_path = any(step.get("node_id") == n.get("id") for step in focus_path)
        graph_distance = distance_map.get(n.get("id")) if focus_node else None

        nodes.append({
            "id": nid, "type": "knowledgeNode", "position": {"x": 0, "y": 0},
            "data": {
                "label": n.get("title") or n.get("slug") or ntype,
                "status": source_status or n.get("status") or "active",
                "content_type": ntype,
                "file_type": _metadata_file_type(meta),
                "file_path": meta.get("file_path") or meta.get("storage_path"),
                "asset_url": _metadata_asset_url(meta),
                "content_preview": (_metadata_markdown(meta, n.get("summary") or ""))[:2000],
                "markdown": _metadata_markdown(meta, n.get("summary") or ""),
                "nodeClass": node_class,
                "item_id": n.get("source_id") or n["id"],
                "source": "graph",
                "source_table": n.get("source_table"),
                "source_id": n.get("source_id"),
                "persona_id": n.get("persona_id"),
                "tags": tags,
                # Semantic decoration â”€â”€â”€â”€â”€â”€â”€â”€
                "node_type": ntype,
                "slug": n.get("slug"),
                "level": level,
                "importance": importance,
                "confidence": confidence,
                "color": registry_row.get("color"),
                "icon": registry_row.get("icon"),
                "is_auxiliary": is_auxiliary,
                "validated": semantic_state == "validated",
                "approval_state": source_status or semantic_state,
                "publication_state": publication_state,
                "approved_snapshot_id": approved_snapshot_id,
                "rag_entry_id": rag_entry_id,
                "rag_chunk_count": rag_chunk_count,
                "snapshot_status": snapshot_status,
                "path_connected_to_persona": n.get("id") in path_connected_semantic_node_ids,
                "terminal_connected": n.get("id") in terminal_connected_semantic_node_ids,
                "branch_complete_validated": n.get("id") in branch_complete_semantic_node_ids,
                "is_focus": is_focus,
                "in_focus_path": in_focus_path,
                "graph_distance": graph_distance,
                "protected": bool(meta.get("protected") or ntype in {"persona", "embedded", "gallery"}),
                "metadata": meta,
                # Pass through legacy fields for backwards-compat with sidebar
                "asset_type": meta.get("asset_type"),
                "asset_function": meta.get("asset_function"),
                "campaign": (n.get("slug") if ntype == "campaign" else None),
                "product":  (n.get("slug") if ntype == "product"  else None),
            },
        })
        semantic_nodes_count += 1

    # Wire semantic edges between semantic nodes.
    focus_path_pairs: set[tuple[str, str]] = set()
    if len(focus_path) >= 2:
        for prev, nxt in zip(focus_path, focus_path[1:]):
            a, b = prev.get("node_id"), nxt.get("node_id")
            if a and b:
                focus_path_pairs.add((a, b))
                focus_path_pairs.add((b, a))

    for e in sem_edges:
        src = sem_node_ids.get(e.get("source_node_id"))
        tgt = sem_node_ids.get(e.get("target_node_id"))
        if not src or not tgt:
            continue
        source_id = (
            semantic_persona_aliases.get(e.get("source_node_id"))
            or semantic_embedded_aliases.get(e.get("source_node_id"))
            or semantic_gallery_aliases.get(e.get("source_node_id"))
            or f"gn:{e['source_node_id']}"
        )
        target_id = (
            semantic_persona_aliases.get(e.get("target_node_id"))
            or semantic_embedded_aliases.get(e.get("target_node_id"))
            or semantic_gallery_aliases.get(e.get("target_node_id"))
            or f"gn:{e['target_node_id']}"
        )
        if source_id == target_id:
            continue
        src_type = (src.get("node_type") or "").lower()
        tgt_type = (tgt.get("node_type") or "").lower()
        rt = (e.get("relation_type") or "related").lower()
        if "gallery" in {src_type, tgt_type} and rt != "gallery_asset":
            continue
        if "embedded" in {src_type, tgt_type} and "faq" not in {src_type, tgt_type}:
            continue
        registry_rel = rt_by_type.get(rt, {})
        weight = e.get("weight")
        if weight is None:
            weight = registry_rel.get("default_weight", 0.50)
        directional = registry_rel.get("directional", True)
        tier = _classify_relation_tier(rt, weight)
        in_path = (e.get("source_node_id"), e.get("target_node_id")) in focus_path_pairs

        edges.append({
            "id": f"ge:{e['id']}",
            "source": source_id,
            "target": target_id,
            "type": "smoothstep",
            "data": {
                "relation_type": rt,
                "tier": tier,
                "weight": weight,
                "directional": directional,
                "in_focus_path": in_path,
                "label": registry_rel.get("label"),
                "original_edge_id": f"ge:{e['id']}",
                "deletable": True,
                "metadata": e.get("metadata") or {},
                "primary_tree": bool((e.get("metadata") or {}).get("primary_tree")),
                "gallery_edge": rt == "gallery_asset",
                "embedded_edge": "embedded" in {src_type, tgt_type},
                "terminal_connected": (
                    (src_type == "faq" and tgt_type == "embedded")
                    or (rt == "gallery_asset")
                ),
                "branch_complete_validated": e.get("id") in branch_complete_edge_ids,
            },
        })
        semantic_edges_count += 1

    gallery_persona_ids = [
        p["id"]
        for p in personas
        if p.get("id") in persona_node_ids_emitted
    ]
    for persona_id in gallery_persona_ids:
        gallery_id = f"gallery:{persona_id}"
        nodes.append({
            "id": gallery_id,
            "type": "knowledgeNode",
            "position": {"x": 0, "y": 0},
            "data": {
                "label": "Gallery",
                "status": "active",
                "content_type": "asset-gallery",
                "content_preview": "Destino protegido para assets e referencias visuais ligadas a esta persona.",
                "nodeClass": "validated",
                "source": "synthetic",
                "tags": ["gallery", "assets", "visual", "default"],
                "node_type": "gallery",
                "slug": "gallery-default",
                "level": 112,
                "importance": 0.76,
                "confidence": 1,
                "color": "#f0abfc",
                "icon": "images",
                "is_auxiliary": False,
                "validated": True,
                "approval_state": "system",
                "publication_state": "destination",
                "persona_id": persona_id,
                "path_connected_to_persona": True,
                "terminal_connected": False,
                "branch_complete_validated": bool(branch_complete_semantic_node_ids),
            },
        })

    draft_terminal_edge_ids: set[str] = set()
    connected_gallery_assets: set[str] = set()
    connected_embedded_faqs: set[str] = set()
    for edge in sem_edges:
        src_id = edge.get("source_node_id")
        tgt_id = edge.get("target_node_id")
        if not src_id or not tgt_id:
            continue
        src_node = sem_nodes_by_id.get(src_id) or {}
        tgt_node = sem_nodes_by_id.get(tgt_id) or {}
        src_type = (src_node.get("node_type") or "").lower()
        tgt_type = (tgt_node.get("node_type") or "").lower()
        if src_type == "faq" and tgt_type == "embedded":
            connected_embedded_faqs.add(src_id)
        if (edge.get("relation_type") or "").lower() == "gallery_asset":
            if src_type == "gallery" and tgt_type == "asset":
                connected_gallery_assets.add(tgt_id)
            elif tgt_type == "gallery" and src_type == "asset":
                connected_gallery_assets.add(src_id)

    for node in sem_nodes:
        node_id = node.get("id")
        node_type = (node.get("node_type") or "").lower()
        persona_id = node.get("persona_id")
        if not node_id or not persona_id:
            continue
        if node_type == "faq" and persona_id in persona_node_ids_emitted and node_id not in connected_embedded_faqs and include_embedded:
            edge_id = f"draft:embedded:{node_id}"
            if edge_id not in draft_terminal_edge_ids:
                edges.append({
                    "id": edge_id,
                    "source": f"gn:{node_id}",
                    "target": f"embedded:{persona_id}",
                    "type": "smoothstep",
                    "data": {
                        "relation_type": "embedded_pending",
                        "tier": "auxiliary",
                        "weight": 0.55,
                        "directional": True,
                        "embedded_edge": True,
                        "draft_terminal_edge": True,
                        "terminal_connected": False,
                        "branch_complete_validated": False,
                        "deletable": False,
                    },
                })
                draft_terminal_edge_ids.add(edge_id)
        if node_type == "asset" and persona_id in persona_node_ids_emitted and node_id not in connected_gallery_assets:
            edge_id = f"draft:gallery:{node_id}"
            if edge_id not in draft_terminal_edge_ids:
                edges.append({
                    "id": edge_id,
                    "source": f"gn:{node_id}",
                    "target": f"gallery:{persona_id}",
                    "type": "smoothstep",
                    "data": {
                        "relation_type": "gallery_pending",
                        "tier": "auxiliary",
                        "weight": 0.55,
                        "directional": True,
                        "gallery_edge": True,
                        "draft_terminal_edge": True,
                        "terminal_connected": False,
                        "branch_complete_validated": False,
                        "deletable": False,
                    },
                })
                draft_terminal_edge_ids.add(edge_id)

    if include_embedded:
        embedded_persona_ids = [
            p["id"]
            for p in personas
            if p.get("id") in persona_node_ids_emitted
        ]
        for persona_id in embedded_persona_ids:
            embedded_id = f"embedded:{persona_id}"
            nodes.append({
                "id": embedded_id,
                "type": "knowledgeNode",
                "position": {"x": 0, "y": 0},
                "data": {
                    "label": "Embedded",
                    "status": "active",
                    "content_type": "rag",
                    "content_preview": "Destino protegido para FAQs publicados no Golden Dataset e enviados ao RAG do modelo.",
                    "nodeClass": "validated",
                    "source": "synthetic",
                    "tags": ["rag", "embedded", "golden-dataset", "default"],
                    "node_type": "embedded",
                    "slug": "embedded-default",
                    "level": 120,
                    "importance": 0.78,
                    "confidence": 1,
                    "color": "#ffffff",
                    "icon": "database",
                    "is_auxiliary": False,
                    "validated": True,
                    "approval_state": "system",
                    "publication_state": "destination",
                    "persona_id": persona_id,
                    "rag_index": "default",
                },
            })

    # â”€â”€ Orphan root for ki:/kb: items without persona â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    has_orphans = any(e.get("source") == "orphan" for e in edges)
    if has_orphans:
        nodes.insert(0, {
            "id": "orphan", "type": "personaNode", "position": {"x": 0, "y": 0},
            "data": {
                "label": "Sem Persona", "slug": "_orphan",
                "description": "Itens ainda nÃ£o atribuÃ­dos",
                "nodeClass": "orphan",
                "node_type": "persona", "level": 0, "importance": 0.5,
                "color": "#475569", "icon": "user", "is_auxiliary": False,
                "validated": False,
            },
        })

    deduped_nodes: dict[str, dict] = {}
    for node in nodes:
        node_id = node.get("id")
        if node_id and node_id not in deduped_nodes:
            deduped_nodes[node_id] = node
    nodes = list(deduped_nodes.values())

    deduped_edges: dict[str, dict] = {}
    primary_seen: set[tuple[str, str, str]] = set()
    for edge in edges:
        edge_id = edge.get("id")
        if edge_id and edge_id in deduped_edges:
            continue
        data = edge.get("data") or {}
        meta = data.get("metadata") or {}
        relation_type = str(data.get("relation_type") or "").lower()
        if data.get("primary_tree") is True or meta.get("primary_tree") is True:
            primary_key = (str(edge.get("source") or ""), str(edge.get("target") or ""), relation_type)
            if primary_key in primary_seen:
                continue
            primary_seen.add(primary_key)
        if edge_id:
            deduped_edges[edge_id] = edge
    edges = list(deduped_edges.values())

    total_items = len(ki_items) + len(kb_entries)

    return {
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "total_personas": len(personas),
            "total_items": total_items,
            "ki_items": len(ki_items),
            "kb_entries": len(kb_entries),
            "semantic_nodes": semantic_nodes_count,
            "semantic_edges": semantic_edges_count,
            "focus": (
                {
                    "node_id": focus_node["id"],
                    "node_type": focus_node.get("node_type"),
                    "slug": focus_node.get("slug"),
                    "title": focus_node.get("title"),
                }
                if focus_node else None
            ),
            "focus_path": focus_path,
            "applied_filters": {
                "max_depth": max_depth,
                "include_tags": include_tags,
                "include_mentions": include_mentions,
                "include_technical": include_technical,
                "include_embedded": include_embedded,
                "mode": mode,
                "persona_slug": persona_slug,
            },
            "registry": {
                "node_types": [
                    {
                        "node_type": r["node_type"],
                        "label": r.get("label"),
                        "level": r.get("default_level"),
                        "importance": r.get("default_importance"),
                        "color": r.get("color"),
                        "icon": r.get("icon"),
                        "sort_order": r.get("sort_order"),
                    }
                    for r in node_type_registry
                ],
                "relations": [
                    {
                        "relation_type": r["relation_type"],
                        "label": r.get("label"),
                        "inverse_label": r.get("inverse_label"),
                        "weight": r.get("default_weight"),
                        "directional": r.get("directional"),
                        "tier": _classify_relation_tier(r["relation_type"], r.get("default_weight") or 0),
                        "sort_order": r.get("sort_order"),
                    }
                    for r in relation_type_registry
                ],
            },
        },
    }


@router.post("/dev/preflight/vzlupas")
def graph_contract_preflight_vzlupas(body: GraphPreflightBody):
    graph, default_tree = _build_vzlupas_contract_graph()
    if body.graph:
        graph = body.graph
    tree_edge_ids = body.tree_edge_ids or default_tree
    ok, checks, errors = _run_vzlupas_preflight(graph, tree_edge_ids)
    return {
        "ok": ok,
        "contract": {
            "name": "vzlupas_catalog_to_hierarchical_graph",
            "version": "2026-05-25",
            "rules": [
                "invalid graph edges are rejected before persistence",
                "direct product->embed is impossible",
                "unapproved faq->embed is impossible",
                "tree view includes only main edges",
            ],
        },
        "graph_summary": {
            "node_count": len(graph.get("nodes", [])),
            "edge_count": len(graph.get("edges", [])),
            "tree_edge_count": len(tree_edge_ids),
        },
        "checks": checks,
        "errors": errors,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

