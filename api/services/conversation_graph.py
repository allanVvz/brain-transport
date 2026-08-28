"""Place inbound WhatsApp media in the knowledge graph.

The shape is::

    campaign -> audience -> conversation(lead) -> asset -> Gallery

A `conversation` node stands for one lead's thread. It hangs under the
audience the lead belongs to, which hangs under the campaign that originated
the outreach — so a photo a customer sends is traceable back to the campaign
that prompted it, which is exactly the attribution the CRM needs.

When no campaign can be attributed (an organic contact who messaged on their
own), the conversation node is still created but left out of the primary tree
rather than invented under a fabricated campaign. The asset still reaches
Gallery, so it remains visible in Assets either way.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from services import knowledge_graph, supabase_client

logger = logging.getLogger("services.conversation_graph")


def _audience_node_for_recipient(persona_id: str, campaign_recipient_id: Optional[str]) -> Optional[dict]:
    """Resolve the audience node behind a campaign recipient.

    ``campaign_recipients`` points at the revision, and the revision names the
    audience. Returns None for an organic contact.
    """
    if not campaign_recipient_id:
        return None
    client = supabase_client.get_client()
    try:
        recipients = (
            client.table("campaign_recipients")
            .select("campaign_revision_id")
            .eq("id", campaign_recipient_id).limit(1).execute()
        ).data or []
        if not recipients:
            return None
        revisions = (
            client.table("campaign_revisions")
            .select("audience_id")
            .eq("id", recipients[0].get("campaign_revision_id")).limit(1).execute()
        ).data or []
        audience_id = (revisions[0] if revisions else {}).get("audience_id")
        if not audience_id:
            return None
    except Exception as exc:
        logger.warning("audience lookup failed recipient=%s: %s", campaign_recipient_id, exc)
        return None

    node = supabase_client.get_knowledge_node_for_source(
        "audiences", str(audience_id), persona_id=persona_id
    )
    if node:
        return node
    # The audience exists but was never mirrored into the graph; mirror it now
    # so the chain is complete rather than dangling.
    audience = supabase_client.get_audience(str(audience_id))
    return supabase_client.sync_audience_node(audience) if audience else None


def ensure_conversation_node(
    *,
    persona_id: str,
    lead: dict,
    audience_node: Optional[dict] = None,
) -> Optional[dict]:
    """Create or update the `conversation` node for one lead."""
    lead_id = lead.get("id")
    if not persona_id or not lead_id:
        return None
    display = (
        lead.get("nome")
        or lead.get("name")
        or lead.get("external_contact_id")
        or f"lead {lead_id}"
    )
    node = supabase_client.upsert_knowledge_node({
        "persona_id": persona_id,
        "source_table": "leads",
        "node_type": "conversation",
        "slug": f"conversa-{lead_id}",
        "title": f"Conversa — {display}",
        "summary": f"Thread de WhatsApp com {display}.",
        "tags": ["conversa", "whatsapp"],
        "metadata": {
            "lead_id": lead_id,
            "audience_node_id": (audience_node or {}).get("id"),
            "open_url": "/messages",
            # Conversation content is customer data: it documents the thread
            # in the graph but is never a source of commercial truth.
            "rag_eligible": False,
        },
        "status": "active",
        "level": 106,
        "importance": 0.5,
        "confidence": 1.0,
    })
    if not node or not node.get("id"):
        return None

    if audience_node and audience_node.get("id"):
        supabase_client.upsert_knowledge_edge(
            audience_node["id"],
            node["id"],
            "contains",
            persona_id=persona_id,
            weight=0.7,
            metadata={
                "created_from": "whatsapp_media_ingest",
                "primary_tree": True,
                "direction": "audience_to_conversation",
            },
        )
    return node


def attach_inbound_asset(asset_id: str) -> dict[str, Any]:
    """Hang a received file under its lead's conversation node.

    Called by the media ingest worker after the bytes are stored, and only
    then — the graph should describe files that actually exist.
    """
    asset = supabase_client.get_asset(asset_id)
    if not asset:
        return {"attached": False, "reason": "asset_not_found"}
    persona_id = asset.get("persona_id")
    lead_id = asset.get("lead_id")
    if not persona_id or not lead_id:
        return {"attached": False, "reason": "missing_persona_or_lead"}

    lead = supabase_client.get_lead(str(lead_id)) or {"id": lead_id}
    audience_node = _audience_node_for_recipient(
        str(persona_id), asset.get("campaign_recipient_id")
    )
    conversation = ensure_conversation_node(
        persona_id=str(persona_id), lead=lead, audience_node=audience_node
    )
    if not conversation:
        return {"attached": False, "reason": "conversation_node_failed"}

    metadata = asset.get("metadata") or {}
    descriptor = metadata.get("media") or {}
    # Imported here rather than at module scope: the graph contract lives with
    # the assets routes, and importing routes at import time would make
    # services depend on the web layer.
    from routes.assets import _ensure_asset_graph_contract

    result = _ensure_asset_graph_contract(
        asset_id=str(asset_id),
        asset_row=asset,
        persona_id=str(persona_id),
        parent_node=conversation,
        storage_bucket=asset.get("storage_bucket"),
        storage_path=asset.get("storage_path"),
        title=asset.get("name") or asset.get("original_filename") or "Midia recebida",
        summary=metadata.get("descriptor_text") or metadata.get("visual_summary") or "",
        slug_seed=asset.get("original_filename") or str(asset_id),
        asset_type=asset.get("type"),
        # Deliberately None: a customer's file must never take a landing slot
        # or reach the public site.
        asset_function=None,
        tags=["whatsapp", "recebido", str(descriptor.get("kind") or "midia")],
        created_from="whatsapp_media_ingest",
    )
    return {
        "attached": True,
        "status": "attached",
        "conversation_node_id": conversation.get("id"),
        "audience_node_id": (audience_node or {}).get("id"),
        "asset_node_id": (result.get("asset_node") or {}).get("id"),
        "conversation_asset_edge_id": (result.get("parent_edge") or {}).get("id"),
        "gallery_node_id": (result.get("gallery_node") or {}).get("id"),
        "asset_gallery_edge_id": (result.get("gallery_edge") or {}).get("id"),
    }
