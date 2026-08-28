from __future__ import annotations

from typing import Optional

from fastapi import HTTPException

from services import knowledge_graph, supabase_client


def _asset_graph_ref(asset: dict, key: str) -> Optional[str]:
    metadata = asset.get("metadata") or {}
    graph_meta = metadata.get("graph") or {}
    return asset.get(key) or metadata.get(key) or graph_meta.get(key)


def ensure_conversation_asset_graph(
    *,
    asset_id: str,
    asset_row: dict,
    persona_id: str,
    parent_node: dict,
    storage_bucket: Optional[str],
    storage_path: Optional[str],
    title: str,
    summary: str,
    slug_seed: str,
    asset_type: Optional[str],
    tags: Optional[list[str]] = None,
    knowledge_item_id: Optional[str] = None,
    created_from: str = "whatsapp_media_ingest",
) -> dict:
    """Attach inbound media to its conversation and the persona Gallery.

    Customer media never receives a landing slot or asset_function. Authoring
    uploads remain owned by the control-plane assets route.
    """
    if not asset_id:
        raise HTTPException(500, "Asset sem id nao pode ser conectado ao grafo.")
    if not persona_id:
        raise HTTPException(422, "Asset precisa de persona para entrar no grafo.")
    if not parent_node or not parent_node.get("id"):
        raise HTTPException(422, "Asset precisa de um braco do grafo como parent.")
    if (parent_node.get("node_type") or "").lower() in {"gallery", "embedded", "asset"}:
        raise HTTPException(422, "Asset precisa de um parent conversacional valido.")

    existing_node = None
    existing_node_id = _asset_graph_ref(asset_row or {}, "knowledge_node_id")
    if existing_node_id:
        existing_node = supabase_client.get_knowledge_node(existing_node_id)
    if not existing_node:
        existing_node = supabase_client.get_knowledge_node_for_source(
            "assets", asset_id, persona_id=persona_id
        )

    base_slug = knowledge_graph._slugify(slug_seed or title or asset_id)[:60] or "asset"
    node_payload = {
        "persona_id": persona_id,
        "source_table": "assets",
        "source_id": asset_id,
        "node_type": "asset",
        "slug": f"{base_slug}-{asset_id[:8]}",
        "title": title or "Midia recebida",
        "summary": (summary or "")[:400] or None,
        "tags": tags or ["whatsapp", "recebido", "midia"],
        "metadata": {
            **((asset_row or {}).get("metadata") or {}),
            "asset_id": asset_id,
            "knowledge_item_id": knowledge_item_id,
            "storage_bucket": storage_bucket,
            "storage_path": storage_path,
            "file_path": (
                f"{storage_bucket}:{storage_path}"
                if storage_bucket and storage_path
                else None
            ),
            "asset_type": asset_type,
            "parent_node_id": parent_node.get("id"),
            "parent_slug": parent_node.get("slug"),
            "parent_type": parent_node.get("node_type"),
        },
        "status": "active",
        "level": 108,
        "importance": 0.64,
        "confidence": 1.0,
    }
    node_payload["metadata"] = {
        key: value
        for key, value in node_payload["metadata"].items()
        if value is not None
    }
    asset_node = supabase_client.upsert_knowledge_node(node_payload)
    if not asset_node or not asset_node.get("id"):
        raise HTTPException(502, "Falha ao criar node de asset no Graph.")

    parent_edge = supabase_client.upsert_knowledge_edge(
        parent_node["id"],
        asset_node["id"],
        "uses_asset",
        persona_id=persona_id,
        weight=0.85,
        metadata={
            "created_from": created_from,
            "direction": "branch_to_asset",
            "primary_tree": True,
            "parent_slug": parent_node.get("slug"),
            "parent_type": parent_node.get("node_type"),
        },
    )
    if not parent_edge or not parent_edge.get("id"):
        raise HTTPException(502, "Falha ao conectar conversation -> asset no Graph.")

    gallery_node = supabase_client.ensure_gallery_node(persona_id)
    if not gallery_node or not gallery_node.get("id"):
        raise HTTPException(502, "Falha ao criar node Gallery.")

    gallery_edge = supabase_client.upsert_knowledge_edge(
        asset_node["id"],
        gallery_node["id"],
        "gallery_asset",
        persona_id=persona_id,
        weight=0.9,
        metadata={
            "graph_layer": "auxiliary",
            "primary_tree": False,
            "visual_hidden": False,
            "relation_role": "asset_gallery",
            "created_from": created_from,
            "direction": "asset_to_gallery",
        },
    )
    if not gallery_edge or not gallery_edge.get("id"):
        raise HTTPException(502, "Falha ao conectar asset -> Gallery no Graph.")

    updated_asset = supabase_client.update_asset_graph_refs(
        asset_id,
        knowledge_node_id=asset_node["id"],
        gallery_edge_id=gallery_edge["id"],
        parent_node_id=parent_node["id"],
        parent_edge_id=parent_edge["id"],
    )
    return {
        "asset_node": asset_node,
        "parent_edge": parent_edge,
        "landing_edge": None,
        "gallery_node": gallery_node,
        "gallery_edge": gallery_edge,
        "asset": updated_asset or asset_row,
    }

