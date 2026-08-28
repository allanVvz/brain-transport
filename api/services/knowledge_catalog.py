"""Read-only catalog projection of canonical published Graph JSON documents."""
from __future__ import annotations

from collections import Counter
from typing import Any

from schemas.graph_json_v2 import GraphJson, Node
from services import graph_json_v2_store


CATEGORY_ORDER = (
    ("faqs", "FAQs", {"faq"}),
    ("rules_tone", "Regras e Tom", {"rule", "tone"}),
    ("products", "Produtos e Grupos", {"product", "product_group"}),
    ("campaigns", "Campanhas e PÃºblicos", {"campaign", "audience"}),
    ("brand_briefing", "Marca e Briefing", {"persona", "brand", "briefing"}),
    ("copies", "Copies", {"copy"}),
    ("assets", "Assets", {"asset"}),
)


def _status(node: Node) -> str:
    return str(
        (node.data or {}).get("validation_status")
        or (node.data or {}).get("status")
        or "pending_validation"
    )


def _path(graph: GraphJson, node: Node) -> list[dict[str, str]]:
    by_id = {item.id: item for item in graph.nodes}
    result: list[dict[str, str]] = []
    current: Node | None = node
    visited: set[str] = set()
    while current and current.id not in visited:
        visited.add(current.id)
        result.insert(0, {
            "id": current.id, "type": current.node_type,
            "slug": current.slug, "title": current.label,
        })
        current = by_id.get(current.parent_id or "")
    return result


def project_graph(
    graph: GraphJson,
    *,
    version: int,
    checksum: str,
    persona_id: str | None = None,
    persona_name: str | None = None,
) -> dict[str, Any]:
    embedded = next((n for n in graph.nodes if n.node_type == "embedded"), None)
    node_by_id = {node.id: node for node in graph.nodes}
    embedded_faq_ids = {
        edge.source for edge in graph.edges
        if embedded and edge.target == embedded.id and not edge.primary_tree
        and node_by_id.get(edge.source)
        and node_by_id[edge.source].node_type == "faq"
    }
    documents: list[dict[str, Any]] = []
    for node in graph.nodes:
        if node.node_type in {"embedded", "gallery"}:
            continue
        data = node.data or {}
        path = _path(graph, node)
        documents.append({
            "id": node.id,
            "node_type": node.node_type,
            "slug": node.slug,
            "title": node.label,
            "markdown": str(
                data.get("markdown") or data.get("content")
                or data.get("summary") or ""
            ),
            "status": _status(node),
            "source": data.get("source") or "pending_source",
            "path": path,
            "path_label": " â€º ".join(item["title"] for item in path),
            "faq_count": int(data.get("question_count") or 1)
            if node.node_type == "faq" else 0,
            "embedded": node.id in embedded_faq_ids,
            "metadata": {
                "graph_node_id": node.id,
                "markdown_document": bool(data.get("markdown_document")),
            },
        })
    categories = []
    for key, label, node_types in CATEGORY_ORDER:
        rows = [row for row in documents if row["node_type"] in node_types]
        categories.append({"key": key, "label": label, "count": len(rows), "items": rows})
    return {
        "persona": {
            "id": persona_id, "slug": graph.persona_slug,
            "name": persona_name or graph.persona_slug,
        },
        "graph": {
            "id": graph.graph_id, "version": version, "checksum": checksum,
            "status": graph.status, "node_count": len(graph.nodes),
            "edge_count": len(graph.edges), "document_count": len(documents),
        },
        "categories": categories,
        "documents": documents,
        "status_counts": dict(Counter(row["status"] for row in documents)),
        "embedded": {
            "node_id": embedded.id if embedded else None,
            "status": _status(embedded) if embedded else "missing",
            "faq_count": len(embedded_faq_ids),
            "faq_node_ids": sorted(embedded_faq_ids),
        },
    }


def load_catalog(
    *, persona_slug: str, persona_id: str | None = None,
    persona_name: str | None = None,
) -> dict[str, Any] | None:
    current = graph_json_v2_store.load_current(persona_slug)
    if not current:
        return None
    version, graph = current
    event = graph_json_v2_store.latest_event(persona_slug) or {}
    checksum = str(
        ((event.get("payload") or {}).get("checksum"))
        or graph_json_v2_store.checksum_graph(graph)
    )
    return project_graph(
        graph, version=version, checksum=checksum,
        persona_id=persona_id, persona_name=persona_name,
    )

