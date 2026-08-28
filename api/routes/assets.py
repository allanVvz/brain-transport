# -*- coding: utf-8 -*-
"""Asset card upload & connect routes.

Two distinct flows are served by this module:

1. POST /assets/upload â€” Card ASSET. Multipart upload that:
   - stores the file in Supabase Storage (assets-raw bucket)
   - inserts public.assets row (upload_context='asset_card')
   - runs the asset_pipeline and writes public.asset_readings rows
   - creates knowledge_items (content_type='asset', status='pending')
   - bootstraps a knowledge_node node_type='asset' and edges:
       parent â†’ asset      (relation_type='uses_asset')
       asset â†’ gallery     (relation_type='gallery_asset')
   - never auto-creates knowledge_rag_entries (gated by is_rag_eligible)

2. The Sofia/CRIAR upload remains in /kb-intake/upload â€” it persists in
   public.assets only when the file can be tied to a persona and Graph evidence.
"""
from __future__ import annotations

import logging
import io
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, File, Form, Header, HTTPException, Query, Request, Response, UploadFile
from pydantic import BaseModel
from schemas.graph_json_v2 import Edge, GraphJson, Node

from core.landing_slots import (
    LandingSlot,
    edge_metadata_for_slot,
    slot_config,
    slot_for_key,
    slot_for_metadata,
    slots_for_parent_type,
)
from services import auth_service, graph_document_publisher, graph_json_v2_store, knowledge_graph, supabase_client
from services.asset_pipeline import AssetPipelineContext, compose_markdown, run_pipeline
from services.event_emitter import emit
from services.audit_helpers import current_actor, summarize_diff

logger = logging.getLogger("routes.assets")

router = APIRouter(prefix="/assets", tags=["assets"])


def _publish_asset_path_to_canonical_graph(
    *,
    asset: dict,
    parent: dict,
    status: str,
    actor_id: Optional[str],
    source: str,
) -> dict:
    persona = supabase_client.get_persona_by_id(asset.get("persona_id")) or {}
    persona_slug = persona.get("slug")
    current = graph_json_v2_store.load_current(persona_slug) if persona_slug else None
    if current is None:
        raise RuntimeError("Asset mutation requires a published canonical Graph JSON")
    version, graph = current
    next_graph = GraphJson.model_validate(graph.model_dump())

    def find_doc_node(knowledge_node: dict, node_type: str) -> Optional[Node]:
        return next(
            (
                node
                for node in next_graph.nodes
                if node.node_type == node_type
                and (
                    node.slug == knowledge_node.get("slug")
                    or str((node.data or {}).get("knowledge_node_id") or "") == str(knowledge_node.get("id") or "")
                )
            ),
            None,
        )

    parent_doc = find_doc_node(parent, str(parent.get("node_type") or ""))
    gallery_doc = next((node for node in next_graph.nodes if node.node_type == "gallery"), None)
    if parent_doc is None or gallery_doc is None:
        raise RuntimeError("Canonical commercial parent or Gallery was not found")
    metadata = dict(asset.get("metadata") or {})
    asset_slug = str(metadata.get("slug") or asset.get("slug") or f"asset-{asset.get('id')}").strip().lower()
    asset_node_ref = {
        "id": _asset_graph_ref(asset, "knowledge_node_id"),
        "slug": asset_slug,
    }
    asset_doc = find_doc_node(asset_node_ref, "asset")
    if asset_doc is None:
        asset_doc = Node(
            id=f"node:asset:{asset_slug}",
            node_type="asset",
            slug=asset_slug,
            label=str(asset.get("name") or asset.get("filename") or metadata.get("original_filename") or asset_slug),
            parent_id=parent_doc.id,
            data={},
        )
        next_graph.nodes.append(asset_doc)
    asset_doc.parent_id = parent_doc.id
    asset_doc.data = {
        **(asset_doc.data or {}),
        **metadata,
        "status": status,
        "source": metadata.get("source") or "asset_upload",
        "asset_id": asset.get("id"),
        "knowledge_node_id": _asset_graph_ref(asset, "knowledge_node_id"),
        "storage_path": asset.get("storage_path") or metadata.get("storage_path"),
        "asset_type": asset.get("type") or metadata.get("asset_type") or "image",
    }
    next_graph.edges = [
        edge
        for edge in next_graph.edges
        if not (
            edge.target == asset_doc.id and edge.primary_tree
            or edge.source == asset_doc.id and edge.target == gallery_doc.id
        )
    ]
    next_graph.edges.extend(
        [
            Edge(
                id=f"edge:asset-parent:{asset_doc.slug}",
                source=parent_doc.id,
                target=asset_doc.id,
                relation="uses_asset",
                primary_tree=True,
                metadata={"active": True, "created_from": source},
            ),
            Edge(
                id=f"edge:asset-gallery:{asset_doc.slug}",
                source=asset_doc.id,
                target=gallery_doc.id,
                relation="gallery_asset",
                primary_tree=False,
                metadata={"active": True, "created_from": source},
            ),
        ]
    )
    return graph_document_publisher.publish(
        graph=next_graph,
        persona_slug=persona_slug,
        brand_slug=next_graph.brand_slug,
        source=source,
        published_by=actor_id,
        expected_version=version,
        idempotency_key=f"{source}:{asset.get('id')}:{version}:{status}",
    )


def _log_asset_flow(
    event_type: str,
    *,
    asset_id: Optional[str] = None,
    persona_id: Optional[str] = None,
    payload: Optional[dict] = None,
    level: str = "info",
) -> None:
    emit(
        event_type,
        entity_type="asset",
        entity_id=asset_id,
        persona_id=persona_id,
        payload=payload or {},
        level=level,
        source="routes.assets",
    )

# â”€â”€ Type guards (Gallery only accepts asset connections) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_GALLERY_ONLY_FROM = {"asset"}


def _bundle_to_asset_type(kind: str) -> str:
    if kind.startswith("image"):
        return "image"
    if kind == "pdf":
        return "pdf"
    if kind == "video":
        return "video"
    if kind in ("text", "markdown"):
        return "text"
    return "image"


def _is_heic_file(filename: str, mime: str) -> bool:
    value = f"{filename or ''} {mime or ''}".lower()
    return ".heic" in value or ".heif" in value or "image/heic" in value or "image/heif" in value


def _jpeg_preview_from_heic(content: bytes) -> Optional[bytes]:
    try:
        import pillow_heif  # type: ignore
        from PIL import Image
        pillow_heif.register_heif_opener()
        with Image.open(io.BytesIO(content)) as image:
            if image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            out = io.BytesIO()
            image.save(out, format="JPEG", quality=92, optimize=True)
            return out.getvalue()
    except Exception as exc:
        logger.warning("HEIC preview conversion failed: %s", exc)
        return None


def _preview_path_for(filename: str, persona_id: str) -> str:
    stem = (filename or "asset").rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0]
    safe = knowledge_graph._slugify(stem)[:80] or "asset"
    return f"{persona_id}/previews/{safe}.jpg"


def _safe_storage_filename(filename: Optional[str]) -> str:
    """Slugify a filename so Supabase Storage accepts it as an object key.

    Storage rejects spaces, accents and most punctuation with `400 InvalidKey`.
    We keep the extension as-is (lowercased) and slugify the stem. The original
    filename is still preserved in `assets.original_filename` for display.
    """
    name = (filename or "upload").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if "." in name:
        stem, _, ext = name.rpartition(".")
        ext = ext.lower()
    else:
        stem, ext = name, ""
    safe_stem = knowledge_graph._slugify(stem)[:120] or "asset"
    safe_ext = knowledge_graph._slugify(ext) if ext else ""
    return f"{safe_stem}.{safe_ext}" if safe_ext else safe_stem


def _ensure_heic_preview(asset: dict) -> dict:
    if not asset:
        return asset
    metadata = asset.get("metadata") or {}
    if metadata.get("preview_path") or metadata.get("preview_url"):
        return asset
    filename = asset.get("original_filename") or metadata.get("original_filename") or asset.get("name") or ""
    mime = asset.get("mime_type") or metadata.get("mime") or ""
    if not _is_heic_file(filename, mime):
        return asset
    bucket = asset.get("storage_bucket") or metadata.get("storage_bucket")
    path = asset.get("storage_path") or metadata.get("storage_path")
    if not bucket or not path:
        return asset
    try:
        raw = supabase_client.download_from_storage(bucket, path)
        preview_bytes = _jpeg_preview_from_heic(raw)
        if not preview_bytes:
            return asset
        preview_bucket = "assets-derived"
        preview_path = _preview_path_for(filename, asset.get("persona_id") or "unassigned")
        preview_url = supabase_client.upload_to_storage(preview_bucket, preview_path, preview_bytes, "image/jpeg")
        updated_meta = {
            **metadata,
            "original_url": metadata.get("original_url") or asset.get("url"),
            "preview_bucket": preview_bucket,
            "preview_path": preview_path,
            "preview_url": preview_url,
        }
        updated = supabase_client.update_asset(asset["id"], {"url": preview_url, "metadata": updated_meta})
        return updated or {**asset, "url": preview_url, "metadata": updated_meta}
    except Exception as exc:
        logger.warning("HEIC lazy preview failed asset=%s: %s", asset.get("id"), exc)
        return asset


def _strip_gn_prefix(node_id: Optional[str]) -> Optional[str]:
    if not node_id:
        return None
    return node_id[3:] if node_id.startswith("gn:") else node_id


def _resolve_persona(persona_id: Optional[str], persona_slug: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if persona_id and persona_slug:
        return persona_id, persona_slug
    if persona_id and not persona_slug:
        persona = supabase_client.get_persona_by_id(persona_id) or {}
        return persona_id, persona.get("slug")
    if persona_slug and not persona_id:
        persona = supabase_client.get_persona(persona_slug) or {}
        return persona.get("id"), persona_slug
    return None, None


def _find_parent_node(persona_id: Optional[str], branch_hint: Optional[str]) -> Optional[dict]:
    """Look up a node by slug (or id) in the persona scope."""
    if not branch_hint:
        return None
    raw = _strip_gn_prefix(branch_hint) or ""
    # Try as id first.
    try:
        node = supabase_client.get_knowledge_node(raw)
        if node:
            return node
    except Exception:
        pass
    # Then by slug.
    try:
        return supabase_client.get_knowledge_node_by_slug(branch_hint, persona_id=persona_id)
    except Exception:
        return None


def _asset_graph_ref(asset: dict, key: str) -> Optional[str]:
    metadata = asset.get("metadata") or {}
    graph_meta = metadata.get("graph") or {}
    return asset.get(key) or metadata.get(key) or graph_meta.get(key)


def _asset_title(asset: dict, fallback: str = "Asset") -> str:
    return (
        asset.get("name")
        or asset.get("original_filename")
        or (asset.get("metadata") or {}).get("original_filename")
        or fallback
    )


def _slot_for_asset_function(asset_function: Optional[str]) -> Optional[LandingSlot]:
    return {
        "brand_logo": LandingSlot.BRAND_LOGO,
        "brand_secondary": LandingSlot.BRAND_SECONDARY,
        "brand_cover": LandingSlot.BRAND_COVER,
        "campaign_hero": LandingSlot.HERO,
        "campaign_footer": LandingSlot.CAMPAIGN_FOOTER,
        "category_cover": LandingSlot.PRODUCT_GROUP_COVER,
        "product_image": LandingSlot.PRODUCT_IMAGE,
        # Legacy upload labels: keep accepting them, but normalize to the real
        # landing/cardapio functions that menu.py consumes.
        "campaign_reference": LandingSlot.HERO,
        "product_reference": LandingSlot.PRODUCT_IMAGE,
        "visual_reference": None,
        "text_reference": None,
    }.get((asset_function or "").strip())


def _asset_function_for_slot(slot: Optional[LandingSlot]) -> Optional[str]:
    return slot_config(slot)["asset_function"] if slot else None


def _effective_asset_function(asset_function: Optional[str], parent_node: dict) -> Optional[str]:
    slot = _slot_for_asset_function(asset_function)
    if slot:
        return _asset_function_for_slot(slot)
    parent_type = (parent_node.get("node_type") or "").lower()
    return {
        "brand": "brand_logo",
        "campaign": "campaign_hero",
        "product_collection": "campaign_hero",
        "category": "category_cover",
        "product": "product_image",
    }.get(parent_type)


def _bind_upload_to_landing_slot(
    *,
    persona_id: str,
    parent_node: dict,
    asset_node_id: str,
    asset_function: Optional[str],
) -> Optional[dict]:
    slot = _slot_for_asset_function(asset_function)
    if not slot:
        return None
    cfg = slot_config(slot)
    parent_type = (parent_node.get("node_type") or "").lower()
    if cfg["parent_node_type"] == "brand" and parent_type != "brand":
        return None
    if cfg["parent_node_type"] == "category" and parent_type != "category":
        return None
    if cfg["parent_node_type"] == "product" and parent_type != "product":
        return None
    if cfg["parent_node_type"] == "campaign":
        collection_slug = (parent_node.get("metadata") or {}).get("collection_slug") or parent_node.get("slug")
        if not collection_slug:
            return None
        parent_node = _ensure_collection_campaign_parent(
            persona_id,
            collection_slug,
            label=parent_node.get("title") or parent_node.get("slug"),
        )

    removed_previous = _remove_existing_slot_edges(
        parent=parent_node,
        asset_node_id=asset_node_id,
        slot=slot,
        relation_type=cfg["relation_type"],
    )
    slot_metadata = edge_metadata_for_slot(slot, label=parent_node.get("title") or parent_node.get("slug"))
    if slot == LandingSlot.PRODUCT_IMAGE:
        binding = dict(slot_metadata.get("page_binding") or {})
        if parent_node.get("slug"):
            binding["slot_key"] = f"{slot.value}:{parent_node['slug']}"
            binding["target_slug"] = parent_node["slug"]
        slot_metadata["page_binding"] = binding
    edge = supabase_client.upsert_knowledge_edge(
        parent_node["id"],
        asset_node_id,
        cfg["relation_type"],
        persona_id=persona_id,
        weight=0.85,
        metadata={
            "created_from": "asset_upload",
            "primary_tree": True,
            "direction": "branch_to_asset",
            "parent_slug": parent_node.get("slug"),
            "parent_type": parent_node.get("node_type"),
            **slot_metadata,
        },
    )
    if not edge:
        return None
    edge["removed_previous_edge_ids"] = removed_previous
    return edge


def _ensure_asset_graph_contract(
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
    asset_function: Optional[str],
    tags: Optional[list[str]] = None,
    knowledge_item_id: Optional[str] = None,
    created_from: str = "asset_upload",
) -> dict:
    """Ensure every asset has branch -> asset and asset -> Gallery edges.

    This is intentionally strict for asset-card uploads: a file visible in
    marketing assets cannot be a detached database row.
    """
    if not asset_id:
        raise HTTPException(500, "Asset sem id nao pode ser conectado ao grafo.")
    if not persona_id:
        raise HTTPException(422, "Asset precisa de persona para entrar no grafo.")
    if not parent_node or not parent_node.get("id"):
        raise HTTPException(422, "Asset precisa de um braco do grafo como parent.")
    if (parent_node.get("node_type") or "").lower() in {"gallery", "embedded", "asset"}:
        raise HTTPException(422, "Asset deve ser conectado a campanha, produto, oferta, copy, briefing ou audiencia; nao a Gallery/Embedded/Asset.")

    existing_node = None
    existing_node_id = _asset_graph_ref(asset_row or {}, "knowledge_node_id")
    if existing_node_id:
        existing_node = supabase_client.get_knowledge_node(existing_node_id)
    if not existing_node:
        existing_node = supabase_client.get_knowledge_node_for_source("assets", asset_id, persona_id=persona_id)

    base_slug = knowledge_graph._slugify(slug_seed or title or asset_id)[:60] or "asset"
    node_payload = {
        "persona_id": persona_id,
        "source_table": "assets",
        "source_id": asset_id,
        "node_type": "asset",
        "slug": f"{base_slug}-{asset_id[:8]}",
        "title": title or _asset_title(asset_row),
        "summary": (summary or "")[:400] or None,
        "tags": tags or ["asset"],
        "metadata": {
            **((asset_row or {}).get("metadata") or {}),
            "asset_id": asset_id,
            "knowledge_item_id": knowledge_item_id,
            "storage_bucket": storage_bucket,
            "storage_path": storage_path,
            "file_path": f"{storage_bucket}:{storage_path}" if storage_bucket and storage_path else None,
            "asset_type": asset_type,
            "asset_function": asset_function,
            "parent_node_id": parent_node.get("id"),
            "parent_slug": parent_node.get("slug"),
            "parent_type": parent_node.get("node_type"),
            "open_url": "/marketing/assets",
        },
        "status": "active",
        "level": 108,
        "importance": 0.64,
        "confidence": 1.0,
    }
    node_payload["metadata"] = {k: v for k, v in node_payload["metadata"].items() if v is not None}
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
        raise HTTPException(502, "Falha ao conectar branch -> asset no Graph.")

    landing_edge = _bind_upload_to_landing_slot(
        persona_id=persona_id,
        parent_node=parent_node,
        asset_node_id=asset_node["id"],
        asset_function=asset_function,
    )

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
        parent_edge_id=(landing_edge or parent_edge)["id"],
    )

    return {
        "asset_node": asset_node,
        "parent_edge": parent_edge,
        "landing_edge": landing_edge,
        "gallery_node": gallery_node,
        "gallery_edge": gallery_edge,
        "asset": updated_asset or asset_row,
    }


def _repair_asset_graph_if_possible(asset: dict) -> dict:
    """Best-effort repair for older asset rows returned by /assets."""
    if not asset:
        return asset
    existing_node_id = _asset_graph_ref(asset, "knowledge_node_id")
    existing_gallery_edge_id = _asset_graph_ref(asset, "gallery_edge_id")
    if existing_node_id and existing_gallery_edge_id:
        return asset
    if existing_node_id and not existing_gallery_edge_id and asset.get("persona_id"):
        try:
            gallery = supabase_client.ensure_gallery_node(asset["persona_id"])
            if gallery:
                gallery_edge = supabase_client.upsert_knowledge_edge(
                    existing_node_id,
                    gallery["id"],
                    "gallery_asset",
                    persona_id=asset["persona_id"],
                    weight=0.9,
                    metadata={
                        "graph_layer": "auxiliary",
                        "primary_tree": False,
                        "visual_hidden": False,
                        "relation_role": "asset_gallery",
                        "created_from": "asset_list_repair",
                        "direction": "asset_to_gallery",
                    },
                )
                return supabase_client.update_asset_graph_refs(
                    asset["id"],
                    knowledge_node_id=existing_node_id,
                    gallery_edge_id=(gallery_edge or {}).get("id") if isinstance(gallery_edge, dict) else None,
                ) or asset
        except Exception as exc:
            logger.warning("asset gallery repair skipped asset=%s: %s", asset.get("id"), exc)
            return {**asset, "graph_status": "repair_failed"}
    persona_id = asset.get("persona_id")
    metadata = asset.get("metadata") or {}
    branch_hint = metadata.get("parent_node_id") or metadata.get("branch_hint") or metadata.get("parent_slug")
    if not persona_id or not branch_hint:
        return {**asset, "graph_status": "missing_branch"}
    parent_node = _find_parent_node(persona_id, branch_hint)
    if not parent_node:
        return {**asset, "graph_status": "missing_branch"}
    try:
        result = _ensure_asset_graph_contract(
            asset_id=asset["id"],
            asset_row=asset,
            persona_id=persona_id,
            parent_node=parent_node,
            storage_bucket=asset.get("storage_bucket") or metadata.get("storage_bucket"),
            storage_path=asset.get("storage_path") or metadata.get("storage_path"),
            title=_asset_title(asset),
            summary=metadata.get("visual_summary") or metadata.get("extracted_text") or "",
            slug_seed=asset.get("original_filename") or asset.get("name") or asset["id"],
            asset_type=asset.get("type") or metadata.get("kind"),
            asset_function=metadata.get("asset_function"),
            tags=metadata.get("tags") if isinstance(metadata.get("tags"), list) else ["asset"],
            knowledge_item_id=metadata.get("knowledge_item_id"),
            created_from="asset_list_repair",
        )
        return result.get("asset") or asset
    except Exception as exc:
        logger.warning("asset graph repair skipped asset=%s: %s", asset.get("id"), exc)
        return {**asset, "graph_status": "repair_failed"}


# â”€â”€ POST /assets/upload â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.post("/upload")
async def upload_asset(
    request: Request,
    file: UploadFile = File(...),
    persona_id: str = Form(...),
    branch_hint: Optional[str] = Form(None),
    asset_function: Optional[str] = Form(None),
    persona_slug: Optional[str] = Form(None),
):
    auth_service.assert_persona_access(request, persona_id=persona_id)
    persona_id, persona_slug = _resolve_persona(persona_id, persona_slug)
    if not persona_id:
        raise HTTPException(400, "persona invalida")
    if not branch_hint:
        # Asset cannot live without a branch â€” UI must pick one.
        raise HTTPException(
            422,
            {"error": "branch_required", "needs_parent": True, "message": "Escolha o galho do grafo onde o asset sera conectado."},
        )

    parent_node = _find_parent_node(persona_id, branch_hint)
    if not parent_node:
        raise HTTPException(404, f"Parent node nao encontrado para slug/id={branch_hint}")

    try:
        return await _upload_asset_impl(file, persona_id, persona_slug, parent_node, asset_function)
    except HTTPException:
        raise
    except Exception as exc:
        import traceback as _tb
        logger.error("/assets/upload failed for persona=%s parent=%s: %s\n%s",
                     persona_id, (parent_node or {}).get("slug"), exc, _tb.format_exc())
        raise HTTPException(
            502,
            {
                "error": "asset_upload_failed",
                "exception_type": type(exc).__name__,
                "message": str(exc)[:400] or "Falha inesperada ao processar o asset.",
                "filename": file.filename,
                "parent_slug": (parent_node or {}).get("slug"),
            },
        ) from exc


def _append_image_ref_to_parent_card(parent_node: dict, image_slug: str) -> bool:
    """Append `![[image_slug]]` (Obsidian embed) to the parent card's .md.

    The parent card is a knowledge_node of type brand / briefing / product /
    copy / faq whose long-form markdown lives in a linked knowledge_item.
    Returns True when the embed was actually inserted (False when the parent
    has no editable knowledge_item or the reference is already present).
    """
    if not parent_node or not image_slug:
        return False
    parent_id = parent_node.get("id")
    if not parent_id:
        return False
    item_id = None
    if parent_node.get("source_table") == "knowledge_items" and parent_node.get("source_id"):
        item_id = parent_node["source_id"]
    else:
        item_id = (parent_node.get("metadata") or {}).get("knowledge_item_id")
    if not item_id:
        return False
    try:
        item = supabase_client.get_knowledge_item(item_id)
    except Exception:
        return False
    if not item:
        return False
    current = str(item.get("content") or "")
    ref = f"![[{image_slug}]]"
    if ref in current:
        return False
    section_header = "## Imagens"
    if section_header in current:
        new_content = current.rstrip() + f"\n{ref}\n"
    else:
        sep = "\n\n" if current.strip() else ""
        new_content = f"{current.rstrip()}{sep}{section_header}\n\n{ref}\n"
    try:
        supabase_client.update_knowledge_item(item_id, {"content": new_content})
        return True
    except Exception:
        return False


async def _upload_asset_impl(
    file: UploadFile,
    persona_id: str,
    persona_slug: Optional[str],
    parent_node: dict,
    asset_function: Optional[str],
):
    content = await file.read()
    fname = file.filename or "upload"
    mime = file.content_type or ""
    effective_asset_function = _effective_asset_function(asset_function, parent_node)

    # 1) Upload to Supabase Storage.
    # Supabase Storage rejects keys with spaces or non-ASCII characters
    # (400 InvalidKey). Keep the original filename in the assets row
    # (original_filename / metadata) but slug the storage path so the
    # PUT URL is always valid.
    storage_bucket = "assets-raw"
    storage_path = f"{persona_id}/{_safe_storage_filename(fname)}"
    try:
        file_url = supabase_client.upload_to_storage(storage_bucket, storage_path, content, mime or "application/octet-stream")
    except Exception as exc:
        message = str(exc)
        if "Bucket not found" in message or "bucket_not_found" in message.lower():
            # Try to self-heal once before failing: maybe migration 033 didn't
            # land or the project was bootstrapped without storage seeding.
            healed = supabase_client.ensure_bucket(storage_bucket, public=False)
            if healed:
                file_url = supabase_client.upload_to_storage(storage_bucket, storage_path, content, mime or "application/octet-stream")
            else:
                logger.error("/assets/upload bucket missing and could not be created: %s", storage_bucket)
                raise HTTPException(
                    503,
                    {
                        "error": "storage_bucket_missing",
                        "bucket": storage_bucket,
                        "message": (
                            f"O bucket '{storage_bucket}' nao existe no Supabase desta API. "
                            "Aplicar migration 033_asset_upload_pipeline.sql ou rodar "
                            "scripts/ensure_qa_buckets.py com a SERVICE_KEY do projeto."
                        ),
                    },
                ) from exc
        elif "InvalidKey" in message or "Invalid key" in message:
            logger.error("/assets/upload InvalidKey for path=%s: %s", storage_path, message)
            raise HTTPException(
                422,
                {
                    "error": "invalid_storage_key",
                    "storage_path": storage_path,
                    "original_filename": fname,
                    "message": (
                        "Supabase Storage rejeitou o nome do arquivo. Use letras, numeros, "
                        "hifens, pontos e sublinhados (sem espacos nem acentos)."
                    ),
                },
            ) from exc
        else:
            logger.warning("assets-raw upload failed, falling back to 'knowledge': %s", exc)
            storage_bucket = "knowledge"
            storage_path = f"assets/{persona_id}/{_safe_storage_filename(fname)}"
            file_url = supabase_client.upload_to_storage(storage_bucket, storage_path, content, mime or "application/octet-stream")

    preview_bucket = None
    preview_path = None
    preview_url = None
    if _is_heic_file(fname, mime):
        preview_bytes = _jpeg_preview_from_heic(content)
        if preview_bytes:
            preview_bucket = "assets-derived"
            preview_path = _preview_path_for(fname, persona_id)
            try:
                preview_url = supabase_client.upload_to_storage(preview_bucket, preview_path, preview_bytes, "image/jpeg")
            except Exception as exc:
                logger.warning("HEIC preview upload failed filename=%s path=%s: %s", fname, preview_path, exc)
                preview_bucket = None
                preview_path = None
                preview_url = None

    # 2) Run the pipeline.
    ctx = AssetPipelineContext(
        persona_id=persona_id,
        persona_slug=persona_slug,
        session_id=None,
        upload_context="asset_card",
        original_filename=fname,
        mime=mime,
        branch_hint=parent_node.get("slug"),
        branch_label=parent_node.get("title") or parent_node.get("slug"),
        asset_function=effective_asset_function,
    )
    bundle = run_pipeline(content, ctx)

    # 3) Persist public.assets.
    asset_type = _bundle_to_asset_type(bundle.classification.kind)
    asset_row = supabase_client.insert_asset({
        "persona_id": persona_id,
        "type": asset_type,
        "name": bundle.rename.title or fname,
        "url": preview_url or file_url,
        "metadata": {
            "session_id": None,
            "persona_slug": persona_slug,
            "original_filename": fname,
            "original_url": file_url,
            "preview_bucket": preview_bucket,
            "preview_path": preview_path,
            "preview_url": preview_url,
            "reading_status": bundle.reading_status,
            "extracted_text": (bundle.extracted_text or "")[:8000],
            "visual_summary": bundle.visual_summary or "",
            "video_reading_mocked": bool(bundle.video_mock),
            "upload_context": "asset_card",
            "validation_status": "pending_validation",
            "kind": bundle.classification.kind,
            "asset_function": effective_asset_function or bundle.rename.asset_function,
            "branch_hint": parent_node.get("slug"),
        },
        "source": "upload",
        "storage_bucket": storage_bucket,
        "storage_path": storage_path,
        "mime_type": mime,
        "file_size": len(content),
        "original_filename": fname,
        "status": "ready" if bundle.reading_status in ("completed", "mocked") else "reading",
        "upload_context": "asset_card",
    })
    asset_id = (asset_row or {}).get("id")
    if not asset_id:
        raise HTTPException(500, "Falha ao registrar asset.")

    for row in bundle.rows_to_persist:
        payload = dict(row)
        payload["asset_id"] = asset_id
        try:
            supabase_client.insert_asset_reading(payload)
        except Exception as exc:
            logger.warning("asset_reading insert failed: %s", exc)

    # 4) Create knowledge_item content_type='asset' pending.
    file_type = ""
    if "." in fname:
        file_type = fname.rsplit(".", 1)[-1].lower()
    source = supabase_client.get_or_create_manual_source()
    markdown = compose_markdown(bundle, persona_slug, parent_node.get("title") or parent_node.get("slug"), storage_path)
    item = supabase_client.insert_knowledge_item({
        "persona_id": persona_id,
        "source_id": source["id"],
        "status": "pending",
        "content_type": "asset",
        "title": bundle.rename.title or fname,
        "content": markdown[:12000],
        "file_type": file_type,
        "file_path": f"{storage_bucket}:{storage_path}",
        "asset_type": asset_type,
        "asset_function": effective_asset_function or bundle.rename.asset_function,
        "agent_visibility": ["private"],
        "tags": bundle.rename.tags,
        "metadata": {
            "asset_id": asset_id,
            "storage_bucket": storage_bucket,
            "storage_path": storage_path,
            "original_url": file_url,
            "preview_bucket": preview_bucket,
            "preview_path": preview_path,
            "preview_url": preview_url,
            "asset_kind": bundle.classification.kind,
            "asset_type": asset_type,
            "asset_function": effective_asset_function or bundle.rename.asset_function,
            "validation_status": "pending_validation",
            "upload_context": "asset_card",
            "parent_slug": parent_node.get("slug"),
            "session_id": None,
            "created_via": "asset_upload",
            "engine": (bundle.ocr.engine if bundle.ocr else None),
        },
    })

    # 5) Create/verify graph node + branch / gallery edges. This is a hard
    # contract: an asset-card upload is not successful while detached from Graph.
    graph = _ensure_asset_graph_contract(
        asset_id=asset_id,
        asset_row=asset_row,
        persona_id=persona_id,
        parent_node=parent_node,
        storage_bucket=storage_bucket,
        storage_path=storage_path,
        title=bundle.rename.title or fname,
        summary=bundle.visual_summary or bundle.extracted_text or markdown,
        slug_seed=bundle.rename.slug or fname,
        asset_type=asset_type,
        asset_function=effective_asset_function or bundle.rename.asset_function,
        tags=bundle.rename.tags,
        knowledge_item_id=(item or {}).get("id"),
        created_from="asset_upload",
    )
    mirror = graph["asset_node"]
    parent_edge = graph["parent_edge"]
    landing_edge = graph.get("landing_edge")
    gallery_edge = graph["gallery_edge"]
    asset_row = graph.get("asset") or asset_row

    # 6) Insert `![[slug]]` (Obsidian-style embed) into the parent card's
    # markdown so the image is anchored inside the card's .md. The parent
    # card is brand / briefing / product / copy / faq â€” whichever was
    # selected as branch_hint at upload time.
    parent_card_md_appended = False
    image_slug = (mirror or {}).get("slug") or fname
    try:
        parent_card_md_appended = _append_image_ref_to_parent_card(parent_node, image_slug)
    except Exception as exc:
        logger.warning("parent card .md append skipped parent=%s asset=%s: %s",
                       parent_node.get("id"), asset_id, exc)

    return {
        "success": True,
        "asset_id": asset_id,
        "asset": asset_row,
        "evidence": {
            "asset_id": asset_id,
            "knowledge_item_id": (item or {}).get("id"),
            "knowledge_node_id": (mirror or {}).get("id"),
            "parent_edge_id": (parent_edge or {}).get("id") if isinstance(parent_edge, dict) else None,
            "landing_edge_id": (landing_edge or {}).get("id") if isinstance(landing_edge, dict) else None,
            "gallery_edge_id": (gallery_edge or {}).get("id") if isinstance(gallery_edge, dict) else None,
            "parent_card_node_id": (parent_node or {}).get("id"),
            "parent_card_md_appended": parent_card_md_appended,
            "image_ref": f"![[{image_slug}]]",
            "reading_status": bundle.reading_status,
        },
        "asset_reading": bundle.to_summary(),
        "file_url": preview_url or file_url,
    }


# â”€â”€ GET /assets â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.get("")
def list_assets_route(
    request: Request,
    persona_id: Optional[str] = Query(None),
    upload_context: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    ensure_graph: bool = Query(True),
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
):
    if persona_id:
        auth_service.assert_persona_access(request, persona_id=persona_id)
    rows = supabase_client.list_assets(
        persona_id=persona_id,
        upload_context=upload_context,
        status=status,
        limit=limit,
        offset=offset,
    )
    if not ensure_graph:
        return [_asset_list_payload(row) for row in rows]
    repaired = [_ensure_heic_preview(_repair_asset_graph_if_possible(row)) for row in rows]
    return [
        {
            **_asset_list_payload(row),
            "landing_path": _asset_primary_landing_path(row),
        }
        for row in repaired
    ]


def _asset_list_payload(row: dict) -> dict:
    display_url = supabase_client.asset_display_url(row)
    return {
        **row,
        "url": display_url,
        "display_url": display_url,
        "approval_status": (row.get("metadata") or {}).get("validation_status") or row.get("approval_status"),
        "workflow_status": row.get("status"),
    }


# â”€â”€ GET /assets/{id} â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _kind_label(kind: str) -> str:
    return {
        "image_screenshot": "Imagem (screenshot)",
        "image_product":    "Imagem (produto)",
        "image_document":   "Imagem (documento)",
        "image_social":     "Imagem (social)",
        "image_other":      "Imagem",
        "pdf":              "PDF",
        "text":             "Texto",
        "markdown":         "Markdown",
        "video":            "Video",
        "unknown":          "Desconhecido",
    }.get(kind or "", kind or "Asset")


def _compose_markdown_from_asset(asset: dict, readings: list[dict]) -> str:
    """Re-build the canonical markdown for an asset directly from the DB rows.

    Used by the detail view and by ensure-gallery so the same string is shown
    regardless of whether the linked knowledge_item is reachable.
    """
    metadata = asset.get("metadata") or {}
    title = _asset_title(asset, fallback="Asset")
    kind = metadata.get("kind") or asset.get("type") or "unknown"
    storage_path = (
        asset.get("storage_path")
        or metadata.get("storage_path")
        or asset.get("file_path")
    )
    bucket = asset.get("storage_bucket") or metadata.get("storage_bucket")
    file_location = f"{bucket}:{storage_path}" if bucket and storage_path else storage_path or "-"
    extracted = (metadata.get("extracted_text") or "").strip()
    visual = (metadata.get("visual_summary") or "").strip()
    asset_function = metadata.get("asset_function") or "(definir)"
    branch_hint = metadata.get("branch_hint") or metadata.get("parent_slug") or "(definir)"
    status = metadata.get("validation_status") or asset.get("status") or "ready"

    # Prefer richer readings when present (full OCR text instead of the 8k slice
    # we keep in metadata).
    full_ocr = ""
    full_visual = ""
    for r in readings or []:
        rtype = r.get("reading_type") or ""
        payload = r.get("payload") or {}
        if rtype == "ocr" and not full_ocr:
            full_ocr = (payload.get("text") or payload.get("extracted_text") or "").strip()
        if rtype == "ai_fallback" and not full_visual:
            full_visual = (payload.get("visual_summary") or payload.get("description") or "").strip()
        if rtype == "pdf_text" and not full_ocr:
            full_ocr = (payload.get("text") or "").strip()
    if full_ocr:
        extracted = full_ocr
    if full_visual:
        visual = full_visual

    lines = [
        f"# Asset â€” {title}",
        "",
        "## Tipo",
        _kind_label(str(kind)),
        "",
        "## Origem",
        str(metadata.get("upload_context") or asset.get("upload_context") or "asset_card"),
        "",
        "## Arquivo",
        str(file_location),
        "",
        "## Texto extraido",
        extracted or "(sem texto)",
        "",
        "## Leitura visual",
        visual or "(sem leitura visual)",
        "",
        "## Uso recomendado",
        str(asset_function),
        "",
        "## Conexao sugerida",
        str(branch_hint),
        "",
        "## Status",
        str(status),
    ]
    return "\n".join(lines)


def _graph_state(asset: dict) -> dict:
    """Compact view of where the asset lives in the graph. Drives the
    'EstÃ¡ no grafo de conhecimento' badge in the dashboard detail modal."""
    knowledge_node_id = _asset_graph_ref(asset, "knowledge_node_id")
    gallery_edge_id = _asset_graph_ref(asset, "gallery_edge_id")
    parent_node_id = _asset_graph_ref(asset, "parent_node_id")
    parent_edge_id = _asset_graph_ref(asset, "parent_edge_id")
    in_graph = bool(knowledge_node_id and gallery_edge_id)
    if in_graph:
        reason = "ok"
    elif not knowledge_node_id:
        reason = "missing_node"
    else:
        reason = "missing_gallery_edge"
    return {
        "in_graph": in_graph,
        "reason": reason,
        "knowledge_node_id": knowledge_node_id,
        "gallery_edge_id": gallery_edge_id,
        "parent_node_id": parent_node_id,
        "parent_edge_id": parent_edge_id,
    }


@router.get("/{asset_id}")
def get_asset_route(asset_id: str, request: Request):
    asset = supabase_client.get_asset(asset_id)
    if not asset:
        raise HTTPException(404, "Asset nao encontrado")
    asset = _ensure_heic_preview(asset)
    if asset.get("persona_id"):
        auth_service.assert_persona_access(request, persona_id=asset["persona_id"])
    readings = supabase_client.list_asset_readings(asset_id)

    # The /assets modal shows the PARENT CARD's markdown (brand/briefing/
    # product/copy/faq). The image was inserted into it as `![[slug]]` at
    # upload time. Falls back to the asset's own description when there is
    # no linked parent card with content.
    metadata = asset.get("metadata") or {}
    parent_card: Optional[dict] = None
    parent_card_markdown: Optional[str] = None
    parent_node_id = metadata.get("parent_node_id") or _asset_graph_ref(asset, "parent_node_id")
    if parent_node_id:
        try:
            parent_card = supabase_client.get_knowledge_node(parent_node_id)
        except Exception as exc:
            logger.warning("get_parent_card failed for asset=%s parent=%s: %s", asset_id, parent_node_id, exc)
        if parent_card:
            parent_item_id = (parent_card.get("metadata") or {}).get("knowledge_item_id") or parent_card.get("source_id")
            if parent_item_id and parent_card.get("source_table") == "knowledge_items":
                try:
                    pitem = supabase_client.get_knowledge_item(parent_item_id)
                    if pitem and pitem.get("content"):
                        parent_card_markdown = str(pitem["content"])
                except Exception as exc:
                    logger.warning("get_parent_card_item failed parent=%s: %s", parent_node_id, exc)

    markdown: Optional[str] = parent_card_markdown
    item_id = metadata.get("knowledge_item_id")
    if not markdown and item_id:
        try:
            item = supabase_client.get_knowledge_item(item_id)
            if item and item.get("content"):
                markdown = str(item["content"])
        except Exception as exc:
            logger.warning("get_knowledge_item failed for asset=%s item=%s: %s", asset_id, item_id, exc)
    if not markdown:
        markdown = _compose_markdown_from_asset(asset, readings)

    return {
        "asset": asset,
        "readings": readings,
        "markdown": markdown,
        "parent_card": parent_card,
        "graph": _graph_state(asset),
    }


# â”€â”€ GET /assets/{id}/media â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _parse_range(header: Optional[str], size: int) -> Optional[tuple[int, int]]:
    """Parse a single ``bytes=start-end`` range. Multi-range is not supported."""
    if not header or not header.startswith("bytes=") or size <= 0:
        return None
    spec = header.removeprefix("bytes=").split(",")[0].strip()
    start_raw, _, end_raw = spec.partition("-")
    try:
        if not start_raw:  # suffix form: bytes=-500 (last 500 bytes)
            length = int(end_raw)
            if length <= 0:
                return None
            return max(size - length, 0), size - 1
        start = int(start_raw)
        end = int(end_raw) if end_raw else size - 1
    except ValueError:
        return None
    end = min(end, size - 1)
    if start > end or start < 0:
        return None
    return start, end


@router.get("/{asset_id}/media")
def get_asset_media(asset_id: str, request: Request, range: Optional[str] = Header(None)):
    """Stream an asset's bytes, scoped to the caller's persona.

    ``GET /knowledge/file`` cannot serve this: it requires a session but does
    no persona scoping, so any authenticated user could read any object. That
    is tolerable for marketing material in a public bucket; it is not for a
    customer's photo or voice note, which live in the private
    ``whatsapp-media`` bucket.

    Range support is what makes ``<audio>`` seekable â€” without a 206 the
    browser can only play a voice note straight through.
    """
    asset = supabase_client.get_asset(asset_id)
    if not asset:
        raise HTTPException(404, "Asset nao encontrado")
    persona_id = asset.get("persona_id")
    if not persona_id:
        raise HTTPException(422, "Asset sem persona nao pode ser servido.")
    auth_service.assert_persona_access(request, persona_id=persona_id)

    metadata = asset.get("metadata") or {}
    bucket = asset.get("storage_bucket") or metadata.get("storage_bucket")
    path = asset.get("storage_path") or metadata.get("storage_path")
    if not bucket or not path:
        raise HTTPException(
            409,
            {"error": "media_not_ready", "status": asset.get("status"),
             "message": "O arquivo ainda nao foi baixado do WhatsApp."},
        )
    try:
        data = supabase_client.download_from_storage(bucket, path)
    except Exception as exc:
        logger.warning("asset media download failed asset=%s: %s", asset_id, exc)
        raise HTTPException(404, "Arquivo nao encontrado no storage.") from exc

    media_type = asset.get("mime_type") or metadata.get("mime") or "application/octet-stream"
    size = len(data)
    # private: this is customer content, so it must never land in a shared
    # cache between personas.
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "private, max-age=300"}

    window = _parse_range(range, size)
    if window:
        start, end = window
        return Response(
            content=data[start:end + 1],
            status_code=206,
            media_type=media_type,
            headers={**headers, "Content-Range": f"bytes {start}-{end}/{size}"},
        )
    return Response(content=data, media_type=media_type, headers=headers)


# â”€â”€ POST /assets/{id}/ensure-gallery â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.post("/{asset_id}/ensure-gallery")
def ensure_gallery_for_asset(asset_id: str, request: Request):
    """Repair flow used by the Assets detail modal: when an asset is not in the
    graph (no `knowledge_node_id` or no `gallery_edge_id`), this endpoint
    creates the md node and ensures `asset -> Gallery`.

    The asset-card contract still requires a parent for new uploads; this
    endpoint is the salvage path for older rows or for Sofia attachments that
    never had a branch assigned. The created node carries the same canonical
    markdown the detail modal shows, so Gallery curation stays in sync."""
    asset = supabase_client.get_asset(asset_id)
    if not asset:
        raise HTTPException(404, "Asset nao encontrado")
    persona_id = asset.get("persona_id")
    if persona_id:
        auth_service.assert_persona_access(request, persona_id=persona_id)
    if not persona_id:
        raise HTTPException(
            422,
            {"error": "missing_persona", "message": "Asset sem persona nao pode entrar no grafo."},
        )

    readings = supabase_client.list_asset_readings(asset_id)
    markdown = _compose_markdown_from_asset(asset, readings)
    metadata = asset.get("metadata") or {}

    knowledge_node_id = _asset_graph_ref(asset, "knowledge_node_id")
    knowledge_node: Optional[dict] = None
    if knowledge_node_id:
        try:
            knowledge_node = supabase_client.get_knowledge_node(knowledge_node_id)
        except Exception:
            knowledge_node = None
    if not knowledge_node:
        knowledge_node = supabase_client.get_knowledge_node_for_source(
            "assets", asset_id, persona_id=persona_id
        )

    if not knowledge_node:
        base_slug = knowledge_graph._slugify(
            asset.get("original_filename") or asset.get("name") or asset_id
        )[:60] or "asset"
        node_payload = {
            "persona_id": persona_id,
            "source_table": "assets",
            "source_id": asset_id,
            "node_type": "asset",
            "slug": f"{base_slug}-{asset_id[:8]}",
            "title": _asset_title(asset),
            "summary": (markdown or "")[:400] or None,
            "tags": metadata.get("tags") if isinstance(metadata.get("tags"), list) else ["asset"],
            "metadata": {
                **metadata,
                "asset_id": asset_id,
                "storage_bucket": asset.get("storage_bucket") or metadata.get("storage_bucket"),
                "storage_path": asset.get("storage_path") or metadata.get("storage_path"),
                "asset_type": asset.get("type"),
                "asset_function": metadata.get("asset_function"),
                "created_via": "ensure_gallery",
                "open_url": "/marketing/assets",
                "markdown_excerpt": markdown[:600],
            },
            "status": "active",
            "level": 108,
            "importance": 0.64,
            "confidence": 1.0,
        }
        node_payload["metadata"] = {k: v for k, v in node_payload["metadata"].items() if v is not None}
        knowledge_node = supabase_client.upsert_knowledge_node(node_payload)
        if not knowledge_node or not knowledge_node.get("id"):
            raise HTTPException(502, "Falha ao criar node do asset no Graph.")

    gallery_node = supabase_client.ensure_gallery_node(persona_id)
    if not gallery_node or not gallery_node.get("id"):
        raise HTTPException(502, "Falha ao garantir node Gallery.")

    gallery_edge = supabase_client.upsert_knowledge_edge(
        knowledge_node["id"],
        gallery_node["id"],
        "gallery_asset",
        persona_id=persona_id,
        weight=0.9,
        metadata={
            "graph_layer": "auxiliary",
            "primary_tree": False,
            "visual_hidden": False,
            "relation_role": "asset_gallery",
            "created_from": "ensure_gallery",
            "direction": "asset_to_gallery",
        },
    )
    if not gallery_edge or not gallery_edge.get("id"):
        raise HTTPException(502, "Falha ao conectar asset -> Gallery no Graph.")

    updated = supabase_client.update_asset_graph_refs(
        asset_id,
        knowledge_node_id=knowledge_node["id"],
        gallery_edge_id=gallery_edge["id"],
    )
    asset_after = updated or {**asset, "knowledge_node_id": knowledge_node["id"], "gallery_edge_id": gallery_edge["id"]}

    return {
        "success": True,
        "asset": asset_after,
        "knowledge_node": knowledge_node,
        "gallery_node": gallery_node,
        "gallery_edge": gallery_edge,
        "graph": _graph_state(asset_after),
        "markdown": markdown,
    }


# â”€â”€ POST /assets/{id}/connect â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ConnectBody(BaseModel):
    parent_node_id: str
    relation_type: str = "uses_asset"


@router.post("/{asset_id}/connect", deprecated=True)
def connect_asset(asset_id: str, body: ConnectBody, request: Request):
    asset = supabase_client.get_asset(asset_id)
    if not asset:
        raise HTTPException(404, "Asset nao encontrado")
    if asset.get("persona_id"):
        auth_service.assert_persona_access(request, persona_id=asset["persona_id"])
    _log_asset_flow(
        "asset_legacy_route_used",
        asset_id=asset_id,
        persona_id=asset.get("persona_id"),
        payload={"route": "connect", "replacement": "rebind-path"},
        level="warn",
    )

    knowledge_node_id = _asset_graph_ref(asset, "knowledge_node_id")
    if not knowledge_node_id:
        raise HTTPException(409, "Asset ainda nao foi promovido para o grafo.")

    parent_raw = _strip_gn_prefix(body.parent_node_id)
    parent = _find_parent_node(asset.get("persona_id"), parent_raw)
    if not parent:
        raise HTTPException(404, "Parent node nao encontrado.")

    previous_parent_node_id = _asset_graph_ref(asset, "parent_node_id")
    previous_parent_edge_id = _asset_graph_ref(asset, "parent_edge_id")
    relation_type = body.relation_type or "uses_asset"
    actor = current_actor(request)

    # Guard: never let a non-asset source land directly on Gallery via this endpoint.
    if (parent.get("node_type") or "") == "gallery":
        _log_asset_flow(
            "asset_connect_rejected_gallery",
            asset_id=asset_id,
            persona_id=asset.get("persona_id"),
            payload={
                "actor": actor,
                "context": {
                    "knowledge_node_id": knowledge_node_id,
                    "attempted_parent_node_id": parent.get("id"),
                    "attempted_parent_slug": parent.get("slug"),
                    "attempted_relation_type": relation_type,
                },
                "reason": "Gallery so aceita asset -> gallery.",
            },
            level="warn",
        )
        raise HTTPException(
            422,
            {"error": "gallery_invalid_target", "message": "Gallery so aceita asset -> gallery."},
        )

    edge = supabase_client.upsert_knowledge_edge(
        parent["id"], knowledge_node_id, relation_type,
        persona_id=asset.get("persona_id"),
        weight=0.8,
        metadata={"created_from": "asset_connect", "primary_tree": True},
    )
    gallery = supabase_client.ensure_gallery_node(asset.get("persona_id"))
    gallery_edge = None
    if gallery:
        gallery_edge = supabase_client.upsert_knowledge_edge(
            knowledge_node_id,
            gallery["id"],
            "gallery_asset",
            persona_id=asset.get("persona_id"),
            weight=0.9,
            metadata={
                "graph_layer": "auxiliary",
                "primary_tree": False,
                "visual_hidden": False,
                "relation_role": "asset_gallery",
                "created_from": "asset_connect",
                "direction": "asset_to_gallery",
            },
        )
    supabase_client.update_asset_graph_refs(
        asset_id,
        knowledge_node_id=knowledge_node_id,
        gallery_edge_id=(gallery_edge or {}).get("id") if isinstance(gallery_edge, dict) else None,
        parent_node_id=parent["id"],
        parent_edge_id=(edge or {}).get("id") if isinstance(edge, dict) else None,
    )

    before = {"parent_node_id": previous_parent_node_id, "parent_edge_id": previous_parent_edge_id}
    after = {
        "parent_node_id": parent.get("id"),
        "parent_edge_id": (edge or {}).get("id") if isinstance(edge, dict) else None,
    }
    _log_asset_flow(
        "asset_connected",
        asset_id=asset_id,
        persona_id=asset.get("persona_id"),
        payload={
            "actor": actor,
            "before": before,
            "after": after,
            "diff": summarize_diff(before, after),
            "context": {
                "knowledge_node_id": knowledge_node_id,
                "parent_node_type": parent.get("node_type"),
                "parent_slug": parent.get("slug"),
                "relation_type": relation_type,
                "gallery_edge_id": (gallery_edge or {}).get("id") if isinstance(gallery_edge, dict) else None,
            },
        },
    )
    return {"success": True, "edge": edge, "gallery_edge": gallery_edge}


# â”€â”€ Landing-page slot bindings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# A landing page has fixed surfaces (hero, product_group_cover,
# campaign_footer). Binding an asset to a slot is just an upsert of a
# knowledge_edge with metadata.page_binding.slot_key set.


class BindSlotBody(BaseModel):
    slot: str
    persona_slug: Optional[str] = None
    target_slug: Optional[str] = None      # required for product_group_cover (category slug)
    collection_slug: Optional[str] = None  # required for hero / campaign_footer
    position: int = 0
    label: Optional[str] = None


class RebindPathBody(BindSlotBody):
    remove_existing: bool = True


class AssetUpdateBody(BaseModel):
    asset_type: Optional[str] = None
    asset_function: Optional[str] = None


def _validate_asset_path(asset: dict) -> dict:
    knowledge_node_id = _asset_graph_ref(asset, "knowledge_node_id")
    gallery_edge_id = _asset_graph_ref(asset, "gallery_edge_id")
    errors: list[str] = []
    landing_connections: list[dict] = []
    if not knowledge_node_id:
        errors.append("Asset sem node no grafo.")
    if not gallery_edge_id:
        errors.append("Asset sem edge para Gallery.")
    if not knowledge_node_id:
        return {"ok": False, "errors": errors, "connections": []}

    edges = supabase_client.list_edges_for_nodes([knowledge_node_id], limit=500)
    parent_ids = {
        edge.get("source_node_id")
        for edge in edges
        if edge.get("target_node_id") == knowledge_node_id
        and edge.get("relation_type") not in {"gallery_asset", "belongs_to_persona"}
    }
    parents = {row["id"]: row for row in supabase_client.list_knowledge_nodes_by_ids(list(parent_ids))}
    for edge in edges:
        if edge.get("target_node_id") != knowledge_node_id:
            continue
        if edge.get("relation_type") in {"gallery_asset", "belongs_to_persona"}:
            continue
        slot = slot_for_metadata(edge.get("metadata") or {})
        if not slot:
            continue
        parent = parents.get(edge.get("source_node_id"))
        if not parent:
            continue
        expected_parent_type = slot_config(slot)["parent_node_type"]
        parent_type = (parent.get("node_type") or "").lower()
        if parent_type == expected_parent_type:
            landing_connections.append(_asset_connection_row(edge, parent, slot))

    if not landing_connections:
        errors.append("Asset sem caminho valido de landing/cardapio.")
    return {"ok": not errors, "errors": errors, "connections": landing_connections}


def _ensure_campaign_node_for_slot(
    persona_id: str,
    slot: LandingSlot,
    collection_slug: str,
    *,
    label: Optional[str],
) -> dict:
    """Find or create a per-collection campaign node that owns hero/footer slots.

    A single campaign node per (persona, collection_slug, slot) keeps the graph
    flat: hero edges live on one campaign, footer edges on another. The node
    carries page_binding so menu.py can serialize the binding back out.
    """
    cfg = slot_config(slot)
    slug = f"landing-{slot.value}-{collection_slug}"[:60]
    existing = supabase_client.get_knowledge_node_by_slug(
        slug,
        persona_id=persona_id,
        node_type="campaign",
    )
    if existing:
        return existing
    payload = {
        "persona_id": persona_id,
        "node_type": "campaign",
        "slug": slug,
        "title": label or f"{cfg['label']} Â· {collection_slug}",
        "summary": f"Landing slot '{slot.value}' for collection '{collection_slug}'.",
        "tags": ["landing", "campaign", slot.value],
        "metadata": {
            "collection_slug": collection_slug,
            "page_binding": {
                "slot_key": slot.value,
                "section": cfg["page_section"],
                "label": label or cfg["label"],
            },
            "landing_slot": slot.value,
            "created_via": "bind_slot",
        },
        "status": "active",
        "level": 60,
        "importance": 0.7,
        "confidence": 1.0,
    }
    node = supabase_client.upsert_knowledge_node(payload)
    if not node or not node.get("id"):
        raise HTTPException(502, "Falha ao criar node de campanha para o slot.")
    return node


def _ensure_collection_campaign_parent(
    persona_id: str,
    collection_slug: str,
    *,
    label: Optional[str],
) -> dict:
    """Return the canonical campaign that owns landing slots for a collection.

    `hero` and `campaign_footer` are landing slots, not graph nodes. Older
    flows created synthetic `landing-hero-*` campaign nodes; new bindings go
    directly under the campaign that represents the catalog/landing campaign.
    """
    campaign = supabase_client.get_knowledge_node_by_slug(
        collection_slug,
        persona_id=persona_id,
        node_type="campaign",
    )
    if campaign:
        return campaign

    collection = supabase_client.get_knowledge_node_by_slug(
        collection_slug,
        persona_id=persona_id,
        node_type="product_collection",
    )
    if not collection:
        raise HTTPException(404, f"collection nao encontrada: {collection_slug}")

    metadata = collection.get("metadata") or {}
    campaign_label = collection.get("title") or metadata.get("display_name") or collection_slug
    campaign_payload = {
        "persona_id": persona_id,
        "node_type": "campaign",
        "slug": collection_slug,
        "title": campaign_label,
        "summary": collection.get("summary") or f"Campanha principal da landing/catalogo {collection_slug}.",
        "tags": sorted(set((collection.get("tags") or []) + ["landing", "catalog", "main_campaign"])),
        "metadata": {
            "collection_slug": collection_slug,
            "product_collection_node_id": collection.get("id"),
            "campaign_role": "primary_catalog",
            "page_binding": {
                "section": "menu.collection",
                "component": "cardapio",
                "label": campaign_label,
            },
            "created_via": "bind_slot_campaign_parent",
        },
        "status": collection.get("status") or "active",
        "level": 30,
        "importance": 0.85,
        "confidence": 1.0,
    }
    campaign = supabase_client.upsert_knowledge_node(campaign_payload)
    if not campaign or not campaign.get("id"):
        raise HTTPException(502, "Falha ao criar campanha principal da colecao.")
    return campaign


def _resolve_brand_parent(persona_id: str, target_slug: Optional[str]) -> dict:
    """Brand slots target a brand node. If the caller did not pass target_slug
    we accept the persona's single brand (most personas have exactly one). When
    there is more than one brand on the persona we require the slug to be
    explicit, since auto-picking would be ambiguous."""
    if target_slug:
        node = supabase_client.get_knowledge_node_by_slug(
            target_slug, persona_id=persona_id, node_type="brand",
        )
        if not node:
            raise HTTPException(404, f"brand nao encontrado: {target_slug}")
        return node
    try:
        rows = (
            supabase_client.get_client()
            .table("knowledge_nodes")
            .select("*")
            .eq("persona_id", persona_id)
            .eq("node_type", "brand")
            .neq("status", "archived")
            .limit(10)
            .execute()
            .data or []
        )
    except Exception as exc:
        raise HTTPException(502, f"Falha ao localizar brand da persona: {exc}") from exc
    if not rows:
        raise HTTPException(
            404,
            "Persona sem node 'brand'. Crie um brand antes de bindar slots brand_*.",
        )
    if len(rows) > 1:
        raise HTTPException(
            409,
            "Persona tem mais de um brand. Informe target_slug do brand desejado.",
        )
    return rows[0]


def _resolve_slot_parent(
    persona_id: str,
    slot: LandingSlot,
    *,
    target_slug: Optional[str],
    collection_slug: Optional[str],
    label: Optional[str],
) -> dict:
    cfg = slot_config(slot)
    parent_type = cfg["parent_node_type"]
    if parent_type == "brand":
        return _resolve_brand_parent(persona_id, target_slug)
    if parent_type in {"category", "product"}:
        if not target_slug:
            raise HTTPException(422, f"{slot.value} precisa de target_slug ({parent_type} slug).")
        node = supabase_client.get_knowledge_node_by_slug(
            target_slug, persona_id=persona_id, node_type=parent_type,
        )
        if not node:
            raise HTTPException(404, f"{parent_type} nao encontrado: {target_slug}")
        return node
    # hero / footer â†’ canonical campaign for the collection. The slot lives in
    # edge metadata; it must not materialise as a separate "hero" graph node.
    if not collection_slug:
        raise HTTPException(422, f"{slot.value} precisa de collection_slug.")
    return _ensure_collection_campaign_parent(persona_id, collection_slug, label=label)


def _asset_connection_row(edge: dict, parent_node: dict, slot: Optional[LandingSlot]) -> dict:
    meta = edge.get("metadata") or {}
    binding = meta.get("page_binding") or {}
    parent_type = parent_node.get("node_type")
    return {
        "edge_id": edge.get("id"),
        "relation_type": edge.get("relation_type"),
        "slot_key": binding.get("slot_key") or (slot.value if slot else binding.get("section")),
        "page_section": binding.get("section") or meta.get("page_section"),
        "label": binding.get("label"),
        "position": binding.get("position"),
        "role": meta.get("role"),
        "parent_node": {
            "id": parent_node.get("id"),
            "slug": parent_node.get("slug"),
            "node_type": parent_type,
            "title": parent_node.get("title"),
            "collection_slug": (parent_node.get("metadata") or {}).get("collection_slug"),
        },
        "slot_options": slots_for_parent_type(parent_type),
    }


def _asset_primary_landing_path(asset: dict) -> Optional[dict]:
    knowledge_node_id = _asset_graph_ref(asset, "knowledge_node_id")
    if not knowledge_node_id:
        return None
    try:
        edges = supabase_client.list_edges_for_nodes([knowledge_node_id], limit=500)
    except Exception as exc:
        logger.warning("asset landing path lookup failed asset=%s: %s", asset.get("id"), exc)
        return None
    parent_ids = [
        edge.get("source_node_id")
        for edge in edges
        if edge.get("target_node_id") == knowledge_node_id
        and edge.get("relation_type") not in {"gallery_asset", "belongs_to_persona"}
    ]
    parents = {row["id"]: row for row in supabase_client.list_knowledge_nodes_by_ids(parent_ids)} if parent_ids else {}
    for edge in edges:
        if edge.get("target_node_id") != knowledge_node_id:
            continue
        if edge.get("relation_type") in {"gallery_asset", "belongs_to_persona"}:
            continue
        parent = parents.get(edge.get("source_node_id"))
        if not parent:
            continue
        slot = slot_for_metadata(edge.get("metadata") or {})
        row = _asset_connection_row(edge, parent, slot)
        cfg = slot_config(slot) if slot else {}
        return {
            "edge_id": row.get("edge_id"),
            "slot_key": row.get("slot_key"),
            "slot_label": cfg.get("label") or row.get("label"),
            "asset_function": cfg.get("asset_function"),
            "parent_node_type": row.get("parent_node", {}).get("node_type"),
            "target_slug": row.get("parent_node", {}).get("slug"),
            "collection_slug": row.get("parent_node", {}).get("collection_slug"),
            "label": row.get("parent_node", {}).get("title") or row.get("parent_node", {}).get("slug"),
        }
    return None


def _landing_target_row(slot: LandingSlot, node: dict, *, collection_slug: Optional[str] = None) -> dict:
    cfg = slot_config(slot)
    metadata = node.get("metadata") or {}
    resolved_collection = collection_slug or metadata.get("collection_slug")
    return {
        "slot_key": slot.value,
        "slot_label": cfg["label"],
        "parent_node_type": cfg["parent_node_type"],
        "target_slug": node.get("slug"),
        "collection_slug": resolved_collection,
        "label": node.get("title") or node.get("slug") or node.get("id"),
        "node_id": node.get("id"),
    }


def _list_landing_targets(persona_id: str) -> list[dict]:
    """Return concrete targets the asset can be rebound to from the preview UI."""
    targets: list[dict] = []
    brands = supabase_client.list_product_collection_nodes(persona_id=persona_id, node_type="brand", limit=100)
    collections = supabase_client.list_product_collection_nodes(persona_id=persona_id, node_type="product_collection", limit=200)
    categories = supabase_client.list_product_collection_nodes(persona_id=persona_id, node_type="category", limit=500)
    products = supabase_client.list_product_nodes(persona_id=persona_id, limit=1000)

    for brand in brands:
        for slot in (LandingSlot.BRAND_LOGO, LandingSlot.BRAND_COVER, LandingSlot.BRAND_SECONDARY):
            targets.append(_landing_target_row(slot, brand))
    for collection in collections:
        collection_slug = collection.get("slug") or (collection.get("metadata") or {}).get("collection_slug")
        if not collection_slug:
            continue
        targets.append(_landing_target_row(LandingSlot.HERO, collection, collection_slug=collection_slug))
        targets.append(_landing_target_row(LandingSlot.CAMPAIGN_FOOTER, collection, collection_slug=collection_slug))
    for category in categories:
        targets.append(_landing_target_row(LandingSlot.PRODUCT_GROUP_COVER, category))
    for product in products:
        targets.append(_landing_target_row(LandingSlot.PRODUCT_IMAGE, product))
    return targets


def _delete_asset_parent_edges(asset_id: str, knowledge_node_id: str) -> list[str]:
    removed_ids: list[str] = []
    edges = supabase_client.list_edges_for_nodes([knowledge_node_id], limit=1000)
    for edge in edges:
        if edge.get("target_node_id") != knowledge_node_id:
            continue
        if edge.get("relation_type") in {"gallery_asset", "belongs_to_persona"}:
            continue
        if supabase_client.delete_knowledge_edge(edge.get("id")):
            removed_ids.append(edge["id"])

    asset = supabase_client.get_asset(asset_id) or {}
    metadata = dict(asset.get("metadata") or {})
    graph_meta = dict(metadata.get("graph") or {})
    metadata.pop("parent_node_id", None)
    metadata.pop("parent_edge_id", None)
    graph_meta.pop("parent_node_id", None)
    graph_meta.pop("parent_edge_id", None)
    if graph_meta:
        metadata["graph"] = graph_meta
    else:
        metadata.pop("graph", None)
    try:
        supabase_client.update_asset(asset_id, {"metadata": metadata})
    except Exception as exc:
        logger.warning("clearing asset parent refs failed asset=%s: %s", asset_id, exc)
    return removed_ids


def _remove_existing_slot_edges(
    *,
    parent: dict,
    asset_node_id: str,
    slot: LandingSlot,
    relation_type: str,
) -> list[str]:
    """Keep landing slots singleton per parent.

    A product card must have one current image. Binding a new asset to the same
    product/slot removes previous active slot edges so the menu payload cannot
    return stale images before the new one.
    """
    parent_id = parent.get("id")
    if not parent_id or not asset_node_id:
        return []
    removed: list[str] = []
    edges = supabase_client.list_edges_for_nodes(
        [parent_id],
        relation_types=[relation_type],
        limit=500,
    )
    for edge in edges:
        if edge.get("source_node_id") != parent_id:
            continue
        if edge.get("target_node_id") == asset_node_id:
            continue
        edge_slot = slot_for_metadata(edge.get("metadata") or {})
        if edge_slot != slot and not (slot == LandingSlot.PRODUCT_IMAGE and edge.get("relation_type") == "product_has_asset"):
            continue
        if supabase_client.delete_knowledge_edge(edge["id"]):
            removed.append(edge["id"])
    return removed


@router.post("/{asset_id}/bind-slot", deprecated=True)
def bind_asset_to_slot(asset_id: str, body: BindSlotBody, request: Request):
    """Upsert a knowledge_edge that binds the asset to a landing-page slot."""
    slot = slot_for_key(body.slot)
    if not slot:
        raise HTTPException(422, f"Slot invalido: {body.slot}")

    asset = supabase_client.get_asset(asset_id)
    if not asset:
        raise HTTPException(404, "Asset nao encontrado")
    persona_id = asset.get("persona_id")
    if not persona_id:
        raise HTTPException(422, "Asset sem persona nao pode ser conectado a um slot.")
    auth_service.assert_persona_access(request, persona_id=persona_id)
    _log_asset_flow(
        "asset_legacy_route_used",
        asset_id=asset_id,
        persona_id=persona_id,
        payload={"route": "bind-slot", "replacement": "rebind-path"},
        level="warn",
    )

    knowledge_node_id = _asset_graph_ref(asset, "knowledge_node_id")
    if not knowledge_node_id:
        raise HTTPException(409, "Asset ainda nao foi promovido para o grafo. Use ensure-gallery primeiro.")

    parent = _resolve_slot_parent(
        persona_id,
        slot,
        target_slug=body.target_slug,
        collection_slug=body.collection_slug,
        label=body.label,
    )

    cfg = slot_config(slot)
    removed_previous = _remove_existing_slot_edges(
        parent=parent,
        asset_node_id=knowledge_node_id,
        slot=slot,
        relation_type=cfg["relation_type"],
    )
    if slot == LandingSlot.PRODUCT_IMAGE:
        removed_previous.extend(_remove_existing_slot_edges(
            parent=parent,
            asset_node_id=knowledge_node_id,
            slot=slot,
            relation_type="product_has_asset",
        ))
    slot_metadata = edge_metadata_for_slot(slot, position=body.position, label=body.label)
    if slot == LandingSlot.PRODUCT_IMAGE:
        binding = dict(slot_metadata.get("page_binding") or {})
        target = parent.get("slug") or body.target_slug
        if target:
            binding["slot_key"] = f"{slot.value}:{target}"
            binding["target_slug"] = target
        slot_metadata["page_binding"] = binding
    edge = supabase_client.upsert_knowledge_edge(
        parent["id"],
        knowledge_node_id,
        cfg["relation_type"],
        persona_id=persona_id,
        weight=0.85,
        metadata={
            "created_from": "bind_slot",
            "primary_tree": True,
            "direction": "branch_to_asset",
            "parent_slug": parent.get("slug"),
            "parent_type": parent.get("node_type"),
            **slot_metadata,
        },
    )
    if not edge or not edge.get("id"):
        _log_asset_flow(
            "asset_slot_bind_failed",
            asset_id=asset_id,
            persona_id=persona_id,
            payload={
                "slot": slot.value,
                "target_slug": body.target_slug,
                "collection_slug": body.collection_slug,
                "reason": "edge_upsert_empty",
            },
            level="error",
        )
        raise HTTPException(502, "Falha ao criar edge do slot.")

    asset_metadata = {**(asset.get("metadata") or {}), "asset_function": cfg["asset_function"]}
    try:
        supabase_client.update_asset(asset_id, {"metadata": asset_metadata})
        supabase_client.update_asset_graph_refs(
            asset_id,
            knowledge_node_id=knowledge_node_id,
            parent_node_id=parent["id"],
            parent_edge_id=edge["id"],
        )
    except Exception as exc:
        logger.warning("update_asset metadata failed asset=%s: %s", asset_id, exc)
        _log_asset_flow(
            "asset_slot_metadata_update_failed",
            asset_id=asset_id,
            persona_id=persona_id,
            payload={"edge_id": edge.get("id"), "error": str(exc)},
            level="warn",
        )

    _log_asset_flow(
        "asset_slot_bound",
        asset_id=asset_id,
        persona_id=persona_id,
        payload={
            "edge_id": edge.get("id"),
            "slot": slot.value,
            "slot_instance_key": (edge.get("metadata") or {}).get("page_binding", {}).get("slot_key"),
            "target_slug": parent.get("slug") or body.target_slug,
            "parent_node_id": parent.get("id"),
            "parent_node_type": parent.get("node_type"),
            "removed_previous_edge_ids": removed_previous,
            "relation_type": cfg["relation_type"],
        },
    )

    return {
        "success": True,
        "edge": edge,
        "slot": slot.value,
        "removed_previous_edge_ids": removed_previous,
        "connection": _asset_connection_row(edge, parent, slot),
    }


@router.post("/{asset_id}/rebind-path")
def rebind_asset_path(asset_id: str, body: RebindPathBody, request: Request):
    """Replace the asset's canonical catalog path with one concrete slot target."""
    slot = slot_for_key(body.slot)
    if not slot:
        raise HTTPException(422, f"Slot invalido: {body.slot}")

    asset = supabase_client.get_asset(asset_id)
    if not asset:
        raise HTTPException(404, "Asset nao encontrado")
    persona_id = asset.get("persona_id")
    if not persona_id:
        raise HTTPException(422, "Asset sem persona nao pode ser conectado a um caminho.")
    auth_service.assert_persona_access(request, persona_id=persona_id)

    knowledge_node_id = _asset_graph_ref(asset, "knowledge_node_id")
    if not knowledge_node_id:
        raise HTTPException(409, "Asset ainda nao foi promovido para o grafo. Use ensure-gallery primeiro.")

    parent = _resolve_slot_parent(
        persona_id,
        slot,
        target_slug=body.target_slug,
        collection_slug=body.collection_slug,
        label=body.label,
    )
    cfg = slot_config(slot)
    removed_existing = _delete_asset_parent_edges(asset_id, knowledge_node_id) if body.remove_existing else []
    removed_previous = _remove_existing_slot_edges(
        parent=parent,
        asset_node_id=knowledge_node_id,
        slot=slot,
        relation_type=cfg["relation_type"],
    )
    if slot == LandingSlot.PRODUCT_IMAGE:
        removed_previous.extend(_remove_existing_slot_edges(
            parent=parent,
            asset_node_id=knowledge_node_id,
            slot=slot,
            relation_type="product_has_asset",
        ))
    slot_metadata = edge_metadata_for_slot(slot, position=body.position, label=body.label)
    binding = dict(slot_metadata.get("page_binding") or {})
    target = parent.get("slug") or body.target_slug
    if slot == LandingSlot.PRODUCT_IMAGE and target:
        binding["slot_key"] = f"{slot.value}:{target}"
        binding["target_slug"] = target
    if (parent.get("metadata") or {}).get("collection_slug"):
        binding["collection_slug"] = (parent.get("metadata") or {}).get("collection_slug")
    slot_metadata["page_binding"] = binding

    edge = supabase_client.upsert_knowledge_edge(
        parent["id"],
        knowledge_node_id,
        cfg["relation_type"],
        persona_id=persona_id,
        weight=0.85,
        metadata={
            "created_from": "rebind_path",
            "primary_tree": True,
            "direction": "branch_to_asset",
            "parent_slug": parent.get("slug"),
            "parent_type": parent.get("node_type"),
            "status": "active",
            **slot_metadata,
        },
    )
    if not edge or not edge.get("id"):
        _log_asset_flow(
            "asset_path_rebind_failed",
            asset_id=asset_id,
            persona_id=persona_id,
            payload={
                "slot": slot.value,
                "target_slug": body.target_slug,
                "collection_slug": body.collection_slug,
                "reason": "edge_upsert_empty",
            },
            level="error",
        )
        raise HTTPException(502, "Falha ao criar edge do caminho.")

    asset_metadata = {
        **(asset.get("metadata") or {}),
        "asset_function": cfg["asset_function"],
        "asset_type": asset.get("type") or (asset.get("metadata") or {}).get("asset_type") or "image",
    }
    updated_asset = supabase_client.update_asset(asset_id, {
        "type": asset_metadata["asset_type"],
        "metadata": {k: v for k, v in asset_metadata.items() if v is not None},
    })
    supabase_client.update_asset_graph_refs(
        asset_id,
        knowledge_node_id=knowledge_node_id,
        parent_node_id=parent["id"],
        parent_edge_id=edge["id"],
    )
    try:
        node = supabase_client.get_knowledge_node(knowledge_node_id) or {}
        supabase_client.update_knowledge_node(
            knowledge_node_id,
            {"metadata": {**(node.get("metadata") or {}), **asset_metadata}},
        )
    except Exception as exc:
        logger.warning("rebind asset node metadata failed asset=%s node=%s: %s", asset_id, knowledge_node_id, exc)
        _log_asset_flow(
            "asset_node_metadata_update_failed",
            asset_id=asset_id,
            persona_id=persona_id,
            payload={"edge_id": edge.get("id"), "knowledge_node_id": knowledge_node_id, "error": str(exc)},
            level="warn",
        )

    connection = _asset_connection_row(edge, parent, slot)
    _log_asset_flow(
        "asset_path_rebound",
        asset_id=asset_id,
        persona_id=persona_id,
        payload={
            "edge_id": edge.get("id"),
            "slot": slot.value,
            "slot_instance_key": (edge.get("metadata") or {}).get("page_binding", {}).get("slot_key"),
            "target_slug": parent.get("slug") or body.target_slug,
            "parent_node_id": parent.get("id"),
            "parent_node_type": parent.get("node_type"),
            "removed_existing_edge_ids": removed_existing,
            "removed_previous_edge_ids": removed_previous,
            "relation_type": cfg["relation_type"],
        },
    )
    canonical_publication = _publish_asset_path_to_canonical_graph(
        asset=updated_asset or supabase_client.get_asset(asset_id) or asset,
        parent=parent,
        status="pending_validation",
        actor_id=(auth_service.current_user(request) or {}).get("id"),
        source="assets.rebind-path",
    )
    return {
        "success": True,
        "asset": updated_asset,
        "edge": edge,
        "slot": slot.value,
        "connection": connection,
        "landing_path": _asset_primary_landing_path(updated_asset or supabase_client.get_asset(asset_id) or asset),
        "removed_existing_edge_ids": removed_existing,
        "removed_previous_edge_ids": removed_previous,
        "canonical_publication": canonical_publication,
    }


@router.delete("/{asset_id}/bind-slot/{slot_key}")
def unbind_asset_slot(asset_id: str, slot_key: str, request: Request, target_slug: Optional[str] = Query(None)):
    """Soft-delete every active edge that ties this asset to the given slot.

    For product_group_cover the same asset can be on multiple categories, so
    `target_slug` narrows the unbind to a single category when provided.
    """
    slot = slot_for_key(slot_key)
    if not slot:
        raise HTTPException(422, f"Slot invalido: {slot_key}")

    asset = supabase_client.get_asset(asset_id)
    if not asset:
        raise HTTPException(404, "Asset nao encontrado")
    persona_id = asset.get("persona_id")
    if persona_id:
        auth_service.assert_persona_access(request, persona_id=persona_id)

    knowledge_node_id = _asset_graph_ref(asset, "knowledge_node_id")
    if not knowledge_node_id:
        return {"success": True, "removed": 0}

    edges = supabase_client.list_edges_for_nodes([knowledge_node_id], limit=200)
    removed_ids: list[str] = []
    for edge in edges:
        if edge.get("target_node_id") != knowledge_node_id:
            continue
        edge_slot = slot_for_metadata(edge.get("metadata") or {})
        relation_type = edge.get("relation_type")
        if edge_slot != slot and not (slot == LandingSlot.PRODUCT_IMAGE and relation_type == "product_has_asset"):
            continue
        if target_slug:
            parent_node = supabase_client.get_knowledge_node(edge.get("source_node_id"))
            if not parent_node or parent_node.get("slug") != target_slug:
                continue
        if supabase_client.delete_knowledge_edge(edge["id"]):
            removed_ids.append(edge["id"])

    _log_asset_flow(
        "asset_slot_unbound",
        asset_id=asset_id,
        persona_id=persona_id,
        payload={
            "slot": slot.value,
            "requested_slot_key": slot_key,
            "target_slug": target_slug,
            "removed_edge_ids": removed_ids,
            "removed_count": len(removed_ids),
        },
    )
    return {"success": True, "removed": len(removed_ids), "edge_ids": removed_ids}


@router.get("/{asset_id}/connections")
def list_asset_connections(asset_id: str, request: Request):
    """List every active landing-relevant edge incident on this asset.

    Excludes gallery_asset (already reflected by graph_state.in_graph).
    """
    asset = supabase_client.get_asset(asset_id)
    if not asset:
        raise HTTPException(404, "Asset nao encontrado")
    if asset.get("persona_id"):
        auth_service.assert_persona_access(request, persona_id=asset["persona_id"])

    knowledge_node_id = _asset_graph_ref(asset, "knowledge_node_id")
    if not knowledge_node_id:
        return {"asset_id": asset_id, "knowledge_node_id": None, "connections": []}

    edges = supabase_client.list_edges_for_nodes([knowledge_node_id], limit=500)
    parent_ids = {
        edge.get("source_node_id")
        for edge in edges
        if edge.get("target_node_id") == knowledge_node_id
        and edge.get("relation_type") not in {"gallery_asset", "belongs_to_persona"}
    }
    parent_ids.discard(None)
    parents = {row["id"]: row for row in supabase_client.list_knowledge_nodes_by_ids(list(parent_ids))}

    connections: list[dict] = []
    for edge in edges:
        if edge.get("target_node_id") != knowledge_node_id:
            continue
        if edge.get("relation_type") in {"gallery_asset", "belongs_to_persona"}:
            continue
        parent_node = parents.get(edge.get("source_node_id"))
        if not parent_node:
            continue
        slot = slot_for_metadata(edge.get("metadata") or {})
        connections.append(_asset_connection_row(edge, parent_node, slot))

    return {
        "asset_id": asset_id,
        "knowledge_node_id": knowledge_node_id,
        "connections": connections,
    }


@router.get("/{asset_id}/landing-targets")
def list_asset_landing_targets(asset_id: str, request: Request):
    asset = supabase_client.get_asset(asset_id)
    if not asset:
        raise HTTPException(404, "Asset nao encontrado")
    persona_id = asset.get("persona_id")
    if not persona_id:
        return {"asset_id": asset_id, "targets": []}
    auth_service.assert_persona_access(request, persona_id=persona_id)
    return {"asset_id": asset_id, "targets": _list_landing_targets(persona_id)}


@router.post("/{asset_id}/validate-path")
def validate_asset_path_route(asset_id: str, request: Request):
    asset = supabase_client.get_asset(asset_id)
    if not asset:
        raise HTTPException(404, "Asset nao encontrado")
    persona_id = asset.get("persona_id")
    if persona_id:
        auth_service.assert_persona_access(request, persona_id=persona_id)
    validation = _validate_asset_path(asset)
    _log_asset_flow(
        "asset_path_validated",
        asset_id=asset_id,
        persona_id=persona_id,
        payload={
            "ok": validation.get("ok"),
            "errors": validation.get("errors") or [],
            "connection_count": len(validation.get("connections") or []),
        },
        level="info" if validation.get("ok") else "warn",
    )
    return {"asset_id": asset_id, **validation}


def _validate_asset_approval(asset: dict) -> dict:
    validation = _validate_asset_path(asset)
    metadata = asset.get("metadata") or {}
    asset_type = asset.get("type") or metadata.get("asset_type") or metadata.get("kind")
    errors = list(validation.get("errors") or [])
    if not asset_type:
        errors.append("Asset sem tipo definido.")
    storage_ref = (
        asset.get("storage_path")
        or asset.get("url")
        or metadata.get("storage_path")
        or metadata.get("file_path")
    )
    if not storage_ref:
        errors.append("Asset sem arquivo ou referencia de storage.")
    item_id = metadata.get("knowledge_item_id")
    sidecar = supabase_client.get_knowledge_item(item_id) if item_id else None
    sidecar_path = (sidecar or {}).get("file_path") or metadata.get("sidecar_markdown_path")
    if not sidecar or not sidecar_path or not str(sidecar_path).lower().endswith(".md"):
        errors.append("Asset sem sidecar Markdown materializado.")
    return {**validation, "ok": not errors, "errors": errors}


@router.patch("/{asset_id}")
def update_asset_route(asset_id: str, body: AssetUpdateBody, request: Request):
    asset = supabase_client.get_asset(asset_id)
    if not asset:
        raise HTTPException(404, "Asset nao encontrado")
    persona_id = asset.get("persona_id")
    if persona_id:
        auth_service.assert_persona_access(request, persona_id=persona_id)

    before_metadata = dict(asset.get("metadata") or {})
    before_view = {
        "type": asset.get("type"),
        "asset_type": before_metadata.get("asset_type"),
        "asset_function": before_metadata.get("asset_function"),
        "kind": before_metadata.get("kind"),
    }

    metadata = dict(before_metadata)
    patch: dict = {}
    if body.asset_type is not None:
        asset_type = body.asset_type.strip() if isinstance(body.asset_type, str) else body.asset_type
        patch["type"] = asset_type or None
        metadata["asset_type"] = asset_type or None
        metadata["kind"] = asset_type or metadata.get("kind")
    if body.asset_function is not None:
        asset_function = body.asset_function.strip() if isinstance(body.asset_function, str) else body.asset_function
        metadata["asset_function"] = asset_function or None
    patch["metadata"] = {k: v for k, v in metadata.items() if v is not None}
    updated = supabase_client.update_asset(asset_id, patch)

    node_id = _asset_graph_ref(updated or asset, "knowledge_node_id")
    if node_id:
        try:
            node = supabase_client.get_knowledge_node(node_id) or {}
            supabase_client.update_knowledge_node(
                node_id,
                {
                    "metadata": {**(node.get("metadata") or {}), **patch["metadata"]},
                    "node_type": "asset",
                },
            )
        except Exception as exc:
            logger.warning("update asset node metadata failed asset=%s node=%s: %s", asset_id, node_id, exc)

    item_id = metadata.get("knowledge_item_id")
    if item_id:
        try:
            item_patch = {"metadata": patch["metadata"]}
            if body.asset_type is not None:
                item_patch["asset_type"] = patch.get("type")
            if body.asset_function is not None:
                item_patch["asset_function"] = patch["metadata"].get("asset_function")
            supabase_client.update_knowledge_item(item_id, item_patch)
        except Exception as exc:
            logger.warning("update asset knowledge_item metadata failed asset=%s item=%s: %s", asset_id, item_id, exc)

    after_metadata = (updated or {**asset, **patch}).get("metadata") or metadata
    after_view = {
        "type": (updated or {**asset, **patch}).get("type"),
        "asset_type": after_metadata.get("asset_type"),
        "asset_function": after_metadata.get("asset_function"),
        "kind": after_metadata.get("kind"),
    }
    _log_asset_flow(
        "asset_updated",
        asset_id=asset_id,
        persona_id=persona_id,
        payload={
            "actor": current_actor(request),
            "before": before_view,
            "after": after_view,
            "diff": summarize_diff(before_view, after_view),
            "context": {
                "knowledge_node_id": node_id,
                "knowledge_item_id": item_id,
            },
        },
    )

    return {"success": True, "asset": updated or {**asset, **patch}}


@router.post("/{asset_id}/approve")
def approve_asset_route(asset_id: str, request: Request):
    asset = supabase_client.get_asset(asset_id)
    if not asset:
        raise HTTPException(404, "Asset nao encontrado")
    persona_id = asset.get("persona_id")
    if persona_id:
        auth_service.assert_persona_access(request, persona_id=persona_id)
    validation = _validate_asset_approval(asset)
    if not validation["ok"]:
        raise HTTPException(422, {"error": "invalid_asset_path", "errors": validation["errors"]})

    user = auth_service.current_user(request)
    metadata = {
        **(asset.get("metadata") or {}),
        "validation_status": "approved",
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "approved_by": user.get("id"),
    }
    patch = {"metadata": metadata}
    # Live DB status CHECK may not include "approved"; keep the durable approval
    # status in metadata and leave status as ready for older schemas.
    if (asset.get("status") or "") in {"pending", "reading", "failed", ""}:
        patch["status"] = "ready"
    updated = supabase_client.update_asset(asset_id, patch)

    item_id = metadata.get("knowledge_item_id")
    if item_id:
        try:
            supabase_client.update_knowledge_item(item_id, {"status": "approved", "metadata": metadata})
        except Exception as exc:
            logger.warning("approve asset knowledge_item update failed asset=%s item=%s: %s", asset_id, item_id, exc)

    node_id = _asset_graph_ref(updated or asset, "knowledge_node_id")
    if node_id:
        try:
            supabase_client.update_knowledge_node(node_id, {"status": "active", "metadata": metadata})
        except Exception as exc:
            logger.warning("approve asset node update failed asset=%s node=%s: %s", asset_id, node_id, exc)

    parent_info = ((validation.get("connections") or [{}])[0].get("parent_node") or {})
    parent = supabase_client.get_knowledge_node(parent_info.get("id")) if parent_info.get("id") else None
    try:
        if not parent:
            raise RuntimeError("Asset approval has no canonical commercial parent")
        canonical_publication = _publish_asset_path_to_canonical_graph(
            asset=updated or {**asset, "metadata": metadata},
            parent=parent,
            status="approved",
            actor_id=user.get("id"),
            source="assets.approve",
        )
    except Exception as exc:
        try:
            supabase_client.update_asset(
                asset_id,
                {"status": asset.get("status"), "metadata": asset.get("metadata") or {}},
            )
            if item_id:
                supabase_client.update_knowledge_item(item_id, {"status": "pending"})
            if node_id:
                supabase_client.update_knowledge_node(
                    node_id,
                    {"status": "pending"},
                    mark_related_faqs=False,
                )
        except Exception:
            pass
        raise HTTPException(
            502,
            {"code": "ASSET_CANONICAL_APPROVAL_FAILED", "error": str(exc)},
        ) from exc

    _log_asset_flow(
        "asset_approved",
        asset_id=asset_id,
        persona_id=persona_id,
        payload={
            "knowledge_node_id": node_id,
            "previous_workflow_status": asset.get("status"),
            "workflow_status": (updated or asset).get("status"),
            "approval_status": "approved",
            "knowledge_item_id": item_id,
        },
    )

    return {
        "success": True,
        "asset": updated or {**asset, "metadata": metadata},
        "validation": validation,
        "canonical_publication": canonical_publication,
    }


@router.post("/{asset_id}/reject")
def reject_asset_route(asset_id: str, request: Request):
    asset = supabase_client.get_asset(asset_id)
    if not asset:
        raise HTTPException(404, "Asset nao encontrado")
    persona_id = asset.get("persona_id")
    if persona_id:
        auth_service.assert_persona_access(request, persona_id=persona_id)

    user = auth_service.current_user(request)
    metadata = {
        **(asset.get("metadata") or {}),
        "validation_status": "rejected",
        "rejected_at": datetime.now(timezone.utc).isoformat(),
        "rejected_by": user.get("id"),
    }
    updated = supabase_client.update_asset(asset_id, {"status": "archived", "metadata": metadata})

    item_id = metadata.get("knowledge_item_id")
    if item_id:
        try:
            supabase_client.update_knowledge_item(item_id, {"status": "rejected", "metadata": metadata})
        except Exception as exc:
            logger.warning("reject asset knowledge_item update failed asset=%s item=%s: %s", asset_id, item_id, exc)

    node_id = _asset_graph_ref(updated or asset, "knowledge_node_id")
    if node_id:
        try:
            supabase_client.update_knowledge_node(node_id, {"status": "archived", "metadata": metadata})
        except Exception as exc:
            logger.warning("reject asset node update failed asset=%s node=%s: %s", asset_id, node_id, exc)

    _log_asset_flow(
        "asset_rejected",
        asset_id=asset_id,
        persona_id=persona_id,
        payload={
            "knowledge_node_id": node_id,
            "previous_workflow_status": asset.get("status"),
            "workflow_status": (updated or asset).get("status"),
            "approval_status": "rejected",
            "knowledge_item_id": item_id,
        },
    )

    return {"success": True, "asset": updated or {**asset, "status": "archived", "metadata": metadata}}


@router.delete("/{asset_id}")
def delete_asset_route(asset_id: str, request: Request):
    asset = supabase_client.get_asset(asset_id)
    if not asset:
        raise HTTPException(404, "Asset nao encontrado")
    persona_id = asset.get("persona_id")
    if persona_id:
        auth_service.assert_persona_access(request, persona_id=persona_id)

    client = supabase_client.get_client()
    knowledge_node_id = _asset_graph_ref(asset, "knowledge_node_id")
    removed_edges: list[str] = []
    if knowledge_node_id:
        for edge in supabase_client.list_edges_for_nodes([knowledge_node_id], limit=1000):
            if supabase_client.delete_knowledge_edge(edge.get("id")):
                removed_edges.append(edge["id"])
        supabase_client.delete_knowledge_node(knowledge_node_id)
        _log_asset_flow(
            "asset_deleted_graph_edges_removed",
            asset_id=asset_id,
            persona_id=persona_id,
            payload={
                "knowledge_node_id": knowledge_node_id,
                "removed_edge_ids": removed_edges,
                "removed_count": len(removed_edges),
            },
            level="warn",
        )

    metadata = asset.get("metadata") or {}
    knowledge_item_id = metadata.get("knowledge_item_id")
    if knowledge_item_id:
        try:
            supabase_client.delete_knowledge_item(knowledge_item_id)
        except Exception as exc:
            logger.warning("delete asset knowledge_item failed asset=%s item=%s: %s", asset_id, knowledge_item_id, exc)

    try:
        supabase_client._execute_with_retry(client.table("asset_readings").delete().eq("asset_id", asset_id))
    except Exception as exc:
        logger.warning("delete asset_readings failed asset=%s: %s", asset_id, exc)

    try:
        supabase_client._execute_with_retry(client.table("assets").delete().eq("id", asset_id))
    except Exception as exc:
        raise HTTPException(502, f"Falha ao excluir asset: {exc}") from exc

    bucket = asset.get("storage_bucket") or metadata.get("storage_bucket")
    path = asset.get("storage_path") or metadata.get("storage_path")
    storage_deleted = False
    if bucket and path:
        try:
            client.storage.from_(bucket).remove([path])
            storage_deleted = True
        except Exception as exc:
            logger.warning("delete asset storage failed asset=%s bucket=%s path=%s: %s", asset_id, bucket, path, exc)

    return {
        "success": True,
        "asset_id": asset_id,
        "knowledge_node_id": knowledge_node_id,
        "removed_edge_ids": removed_edges,
        "storage_deleted": storage_deleted,
    }

