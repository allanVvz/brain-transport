"""In-memory adapter from published Graph JSON 2.0 to the 2.1 write model."""
from __future__ import annotations

from typing import Any

from schemas.graph_json_v2 import GraphJson


_RELATION_ALIASES = {
    "answers_question": "answers",
    "gallery_asset": "publishes_to",
    "gallery_has_asset": "publishes_to",
    "faq_has_embed": "publishes_to",
    "visible_to_agent": "publishes_to",
    "product_has_asset": "uses_asset",
    "product_image": "represents",
    "illustrates": "represents",
    "supports_copy": "supports",
    "briefed_by": "applies_to",
}


def _action(node: dict[str, Any], persona_slug: str) -> dict[str, Any]:
    node_type = node["node_type"]
    slug = node.get("slug") or node_type
    if node_type == "gallery":
        return {
            "destination_id": f"site:{persona_slug}",
            "destination_type": "public_site",
            "consumer": {"kind": "site", "ref": persona_slug},
            "accepted_node_types": [
                "brand", "campaign", "product_group", "product", "offer",
                "copy", "faq", "asset",
            ],
            "projection": {"kind": "public_site_json", "format_ref": "landing_page"},
            "policy": {"requires_approved_content": True, "auto_connect": []},
        }
    return {
        "destination_id": f"dataset:{slug}",
        "destination_type": "rag_dataset",
        "consumer": {"kind": "agent", "ref": slug},
        "accepted_node_types": [
            "brand", "briefing", "campaign", "audience", "product_group",
            "product", "service", "offer", "copy", "faq", "rule", "tone", "asset",
        ],
        "projection": {
            "kind": "rag",
            "embedding_profile_ref": "default-1536",
            "chunking_profile_ref": "markdown-semantic-v1",
        },
        "policy": {
            "requires_approved_content": True,
            "asset_requires_textual_description": True,
        },
    }


def upgrade_to_v21(graph: GraphJson) -> GraphJson:
    if graph.schema_version == "2.1":
        return GraphJson.model_validate(graph.model_dump(mode="json"))
    raw = graph.model_dump(mode="json")
    raw["schema_version"] = "2.1"
    raw["status"] = "draft"
    raw["graph_version"] = None
    raw["content_checksum"] = None
    nodes_by_id: dict[str, dict[str, Any]] = {}
    for node in raw.get("nodes") or []:
        node_type = node.get("node_type")
        data = node.get("data") or {}
        node["title"] = node.get("title") or node.get("label") or node.get("slug")
        node["node_class"] = "action" if node_type in {"gallery", "embedded"} else "knowledge"
        if node["node_class"] == "action":
            node["action"] = node.get("action") or _action(node, graph.persona_slug)
            node["lifecycle"] = {"status": "approved", "revision": 1}
        else:
            status = str(data.get("validation_status") or data.get("status") or "pending_validation").lower()
            if status in {"validated", "active", "ativo", "embedded"}:
                status = "approved"
            elif status == "pending":
                status = "pending_validation"
            node["lifecycle"] = {"status": status, "revision": 1}
        node["provenance"] = node.get("provenance") or {
            "source": data.get("source") or "graph_json_v20_adapter"
        }
        node["spec"] = node.get("spec") or {
            key: value for key, value in data.items()
            if key not in {"status", "validation_status", "source", "markdown"}
        }
        node["markdown"] = None
        nodes_by_id[node["id"]] = node

    upgraded_edges: list[dict[str, Any]] = []
    for edge in raw.get("edges") or []:
        source = nodes_by_id.get(edge.get("source"))
        target = nodes_by_id.get(edge.get("target"))
        if not source or not target:
            continue
        if target.get("node_class") == "action" and source.get("node_type") == "persona":
            # v2.0 used Persona -> terminal as a layout anchor.  Action nodes
            # have no hierarchy in 2.1 and an anchor must never become a grant.
            continue
        relation = edge.get("relation_type") or edge.get("relation") or "contains"
        if edge.get("primary_tree") is True and target.get("node_class") == "knowledge":
            relation = "contains"
        relation = _RELATION_ALIASES.get(relation, relation)
        if target.get("node_class") == "action":
            relation = "publishes_to"
        if relation == "same_topic_as":
            relation = "references"
        if relation not in {
            "contains", "targets", "represents", "uses_asset", "supports",
            "answers", "applies_to", "derived_from", "references", "publishes_to",
        }:
            relation = "references"
        edge["relation"] = relation
        edge["relation_type"] = relation
        edge["relation_class"] = (
            "hierarchy" if relation == "contains" else
            "publication" if relation == "publishes_to" else
            "provenance" if relation == "derived_from" else "semantic"
        )
        edge["primary_tree"] = relation == "contains"
        edge["lifecycle"] = {"status": "active"}
        if relation == "publishes_to":
            edge["grant"] = {
                "mode": "manual",
                "reason": "Migrated from Graph JSON v2.0 publication edge",
            }
            # Legacy graphs could publish pending content.  Preserve the node,
            # but do not manufacture authorization for it.
            if source.get("lifecycle", {}).get("status") != "approved":
                edge["lifecycle"] = {"status": "revoked"}
        upgraded_edges.append(edge)
    raw["edges"] = upgraded_edges
    return GraphJson.model_validate(raw)
