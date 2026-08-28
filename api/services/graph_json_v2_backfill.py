"""Build and publish canonical Graph JSON v2 documents from derived graph rows.

This is an explicit migration/backfill path.  The dashboard never falls back
to ``knowledge_nodes``/``knowledge_edges`` at read time; instead, legacy rows
are normalized once and published as the canonical v2 document.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from schemas.graph_json_v2 import Edge, GraphJson, Node
from services import (
    graph_json_importer,
    graph_json_v2_store,
    graph_json_v2_validator,
    supabase_client,
)


_TYPE_ORDER = [
    "brand",
    "briefing",
    "campaign",
    "audience",
    "product_group",
    "product",
    "offer",
    "copy",
    "rule",
    "faq",
    "gallery",
    "embedded",
    "asset",
]
_SUPPORTED_TYPES = {"persona", *_TYPE_ORDER}
_PREFERRED_PARENT_TYPES: dict[str, tuple[str, ...]] = {
    "brand": ("persona",),
    "briefing": ("brand",),
    "campaign": ("briefing", "brand"),
    "audience": ("campaign", "briefing"),
    "product_group": ("audience",),
    "product": ("product_group", "audience"),
    "offer": ("product", "product_group"),
    "copy": ("product", "offer", "product_group"),
    "rule": ("campaign", "briefing", "brand", "persona"),
    "faq": ("copy", "product", "product_group", "audience", "briefing", "campaign", "brand", "persona", "rule"),
    "gallery": ("persona",),
    "embedded": ("faq",),
    "asset": ("product", "product_group", "campaign", "brand"),
}
_SYNTHETIC_LABELS = {
    "brand": "Marca",
    "briefing": "Briefing principal",
    "campaign": "Campanha principal",
    "audience": "Público principal",
    "product_group": "Produtos",
    "gallery": "Galeria",
}


def _active(edge: dict) -> bool:
    metadata = edge.get("metadata") if isinstance(edge.get("metadata"), dict) else {}
    return metadata.get("active", True) is not False


def _node_id(node_type: str, slug: str) -> str:
    return f"node:{node_type}:{slug}"


def _node_data(row: dict) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    metadata.setdefault("source", row.get("source_table") or metadata.get("source") or "derived_graph_backfill")
    metadata.setdefault("status", row.get("status") or "pending_validation")
    if row.get("summary") and not metadata.get("summary"):
        metadata["summary"] = row["summary"]
    if row.get("tags") and not metadata.get("tags"):
        metadata["tags"] = row["tags"]
    return metadata


def _branch_path(node_id: str, nodes_by_id: dict[str, Node]) -> list[dict[str, str]]:
    path: list[dict[str, str]] = []
    current_id = node_id
    seen: set[str] = set()
    while current_id and current_id not in seen:
        seen.add(current_id)
        node = nodes_by_id.get(current_id)
        if not node:
            break
        path.insert(0, {"node_type": node.node_type, "slug": node.slug, "label": node.label})
        current_id = node.parent_id or ""
    return path


def build_from_derived_graph(persona_slug: str, *, tenant: str = "production") -> tuple[GraphJson, dict]:
    persona = supabase_client.get_persona(persona_slug)
    if not persona:
        raise ValueError(f"Persona not found: {persona_slug}")
    persona_id = persona["id"]
    rows, legacy_edges = supabase_client.list_all_knowledge_graph(persona_id=persona_id, limit_nodes=5000)

    root = Node(
        id=_node_id("persona", persona_slug),
        node_type="persona",
        slug=persona_slug,
        label=str(persona.get("name") or persona_slug),
        parent_id=None,
        data={"source": "personas", "status": "validated", "persona_id": persona_id},
    )
    nodes: list[Node] = [root]
    nodes_by_id: dict[str, Node] = {root.id: root}
    nodes_by_type: dict[str, list[Node]] = defaultdict(list)
    nodes_by_type["persona"].append(root)
    legacy_to_doc_id: dict[str, str] = {}
    skipped: list[dict[str, str]] = []

    # Deduplicate legacy rows by canonical (type, slug).  Legacy persona rows
    # are deliberately replaced by the one persona root above.
    rows_by_type: dict[str, list[dict]] = defaultdict(list)
    seen_keys: set[tuple[str, str]] = set()
    for row in rows:
        node_type = str(row.get("node_type") or "").strip().lower()
        slug = str(row.get("slug") or "").strip().lower()
        if str(row.get("status") or "").strip().lower() == "archived":
            skipped.append({"id": str(row.get("id") or ""), "reason": "archived node"})
            continue
        if node_type == "persona":
            if row.get("id"):
                legacy_to_doc_id[str(row["id"])] = root.id
            continue
        if node_type not in _SUPPORTED_TYPES or not slug:
            skipped.append({"id": str(row.get("id") or ""), "reason": f"unsupported node_type={node_type or 'empty'}"})
            continue
        key = (node_type, slug)
        if key in seen_keys:
            skipped.append({"id": str(row.get("id") or ""), "reason": f"duplicate {node_type}:{slug}"})
            continue
        seen_keys.add(key)
        rows_by_type[node_type].append(row)

    incoming: dict[str, list[dict]] = defaultdict(list)
    for edge in legacy_edges:
        if _active(edge) and edge.get("target_node_id"):
            incoming[str(edge["target_node_id"])].append(edge)

    def first_node(node_type: str) -> Node | None:
        values = nodes_by_type.get(node_type) or []
        return values[0] if values else None

    def ensure_synthetic(node_type: str) -> Node:
        existing = first_node(node_type)
        if existing:
            return existing
        if node_type == "brand":
            parent = root
        elif node_type == "briefing":
            parent = ensure_synthetic("brand")
        elif node_type == "campaign":
            parent = ensure_synthetic("briefing")
        elif node_type == "audience":
            parent = ensure_synthetic("campaign")
        elif node_type == "product_group":
            parent = ensure_synthetic("audience")
        elif node_type == "gallery":
            parent = root
        else:
            raise ValueError(f"Cannot synthesize parent node_type={node_type}")
        slug = f"bootstrap-{node_type}"
        node = Node(
            id=_node_id(node_type, slug),
            node_type=node_type,
            slug=slug,
            label=_SYNTHETIC_LABELS.get(node_type, node_type.replace("_", " ").title()),
            parent_id=parent.id,
            data={
                "source": "derived_graph_backfill",
                "status": "pending_validation",
                "synthetic_backfill": True,
            },
        )
        nodes.append(node)
        nodes_by_id[node.id] = node
        nodes_by_type[node_type].append(node)
        return node

    def choose_parent(row: dict, node_type: str) -> Node | None:
        allowed_types = _PREFERRED_PARENT_TYPES[node_type]
        # Imports persist an explicit parent reference in metadata.  Prefer it
        # for every branch node because historical edge vocabularies may be
        # absent or no longer primary-tree compatible.
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        parent_legacy_id = str(meta.get("parent_node_id") or meta.get("source_node_id") or "")
        parent_doc_id = legacy_to_doc_id.get(parent_legacy_id)
        parent = nodes_by_id.get(parent_doc_id or "")
        if parent and parent.node_type in allowed_types:
            return parent
        parent_slug = str(meta.get("parent_slug") or "").strip().lower()
        parent_type = str(meta.get("parent_type") or meta.get("source_node_type") or "").strip().lower()
        if parent_slug:
            for candidate in nodes_by_type.get(parent_type, []) if parent_type else []:
                if candidate.slug == parent_slug and candidate.node_type in allowed_types:
                    return candidate
        # Product imports carry their canonical group in metadata. Prefer that
        # explicit source over whichever group happens to be encountered first
        # when legacy edges are incomplete or use retired relation names.
        if node_type == "product":
            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            group_slug = str(
                meta.get("product_group_slug")
                or meta.get("category_slug")
                or meta.get("parent_group")
                or ""
            ).strip().lower()
            if group_slug:
                for candidate in nodes_by_type.get("product_group", []):
                    if candidate.slug == group_slug:
                        return candidate
        for edge in incoming.get(str(row.get("id") or ""), []):
            source_doc_id = legacy_to_doc_id.get(str(edge.get("source_node_id") or ""))
            source = nodes_by_id.get(source_doc_id or "")
            if source and source.node_type in allowed_types:
                return source
        for parent_type in allowed_types:
            parent = first_node(parent_type)
            if parent:
                if node_type == "embedded":
                    status = str((parent.data or {}).get("status") or "").lower()
                    if status not in graph_json_v2_validator.FAQ_APPROVED_STATUSES:
                        continue
                return parent
        if node_type == "brand":
            return root
        if node_type == "briefing":
            return ensure_synthetic("brand")
        if node_type == "campaign":
            return ensure_synthetic("briefing")
        if node_type == "audience":
            return ensure_synthetic("campaign")
        if node_type == "product_group":
            return ensure_synthetic("audience")
        if node_type == "product":
            return ensure_synthetic("product_group")
        if node_type == "rule":
            return ensure_synthetic("brand")
        if node_type == "gallery":
            return root
        if node_type == "embedded":
            # Layout anchor only; do not create a prohibited Persona ->
            # Embedded primary edge. Approved FAQ edges remain semantic inputs.
            return root
        if node_type == "asset":
            return None
        return None

    for node_type in _TYPE_ORDER:
        for row in rows_by_type.get(node_type, []):
            parent = choose_parent(row, node_type)
            if not parent:
                skipped.append({"id": str(row.get("id") or ""), "reason": f"no valid parent for {node_type}"})
                continue
            slug = str(row["slug"]).strip().lower()
            data = _node_data(row)
            if node_type == "faq":
                data["source_node_id"] = parent.id
                data["source_node_type"] = parent.node_type
                data["branch_path"] = _branch_path(parent.id, nodes_by_id)
            node = Node(
                id=_node_id(node_type, slug),
                node_type=node_type,
                slug=slug,
                label=str(row.get("title") or slug),
                parent_id=parent.id,
                data=data,
            )
            nodes.append(node)
            nodes_by_id[node.id] = node
            nodes_by_type[node_type].append(node)
            if row.get("id"):
                legacy_to_doc_id[str(row["id"])] = node.id

    edges: list[Edge] = []
    primary_pairs: set[tuple[str, str]] = set()
    for node in nodes:
        if node.node_type == "persona" or not node.parent_id:
            continue
        parent = nodes_by_id[node.parent_id]
        if node.node_type == "embedded" and parent.node_type == "persona":
            continue
        relation = graph_json_importer.RELATION_BY_PAIR.get((parent.node_type, node.node_type), "contains")
        edges.append(
            Edge(
                id=f"edge:primary:{len(edges) + 1}",
                source=parent.id,
                target=node.id,
                relation=relation,
                primary_tree=True,
                metadata={"created_from": "derived_graph_backfill"},
            )
        )
        primary_pairs.add((parent.id, node.id))

    # Gallery is the terminal curation output.  Assets keep their commercial
    # parent in the tree and gain a non-primary asset -> Gallery edge.
    gallery = first_node("gallery")
    if gallery:
        for asset in nodes_by_type.get("asset", []):
            edges.append(
                Edge(
                    id=f"edge:gallery:{asset.id}",
                    source=asset.id,
                    target=gallery.id,
                    relation="gallery_asset",
                    primary_tree=False,
                    metadata={"created_from": "derived_graph_backfill", "active": True},
                )
            )

    # Keep valid active semantic relations as secondary edges.  Invalid legacy
    # primary-parent shapes remain in the audit DB but are not canonicalized.
    seen_secondary: set[tuple[str, str, str]] = set()
    for row in legacy_edges:
        if not _active(row):
            continue
        source = legacy_to_doc_id.get(str(row.get("source_node_id") or ""))
        target = legacy_to_doc_id.get(str(row.get("target_node_id") or ""))
        relation = str(row.get("relation_type") or "contains")
        if not source or not target or source == target or (source, target) in primary_pairs:
            continue
        source_node = nodes_by_id.get(source)
        target_node = nodes_by_id.get(target)
        if not source_node or not target_node:
            continue
        if target_node.node_type == "embedded" and source_node.node_type != "faq":
            skipped.append({"id": str(row.get("id") or ""), "reason": "invalid non-FAQ edge into embedded"})
            continue
        key = (source, target, relation)
        if key in seen_secondary:
            continue
        seen_secondary.add(key)
        metadata = dict(row.get("metadata") or {})
        metadata["primary_tree"] = False
        metadata["created_from"] = "derived_graph_backfill"
        edges.append(
            Edge(
                id=f"edge:legacy:{row.get('id') or len(edges) + 1}",
                source=source,
                target=target,
                relation=relation,
                primary_tree=False,
                metadata=metadata,
            )
        )

    graph = GraphJson(
        schema_version="2.0",
        graph_id=f"{persona_slug}-main",
        tenant=tenant,
        persona_slug=persona_slug,
        status="published",
        nodes=nodes,
        edges=edges,
    )
    valid, errors = graph_json_v2_validator.validate_graph_json(graph)
    report = {
        "persona_slug": persona_slug,
        "legacy_nodes": len(rows),
        "legacy_edges": len(legacy_edges),
        "canonical_nodes": len(nodes),
        "canonical_edges": len(edges),
        "skipped": skipped,
        "valid": valid,
        "validation_errors": errors,
    }
    return graph, report


def publish_backfill(
    persona_slug: str,
    *,
    force: bool = False,
    materialize: bool = True,
    tenant: str = "production",
) -> dict:
    current = graph_json_v2_store.load_current(persona_slug)
    if current and not force:
        return {
            "ok": True,
            "persona_slug": persona_slug,
            "skipped_existing": True,
            "version": current[0],
        }

    graph, report = build_from_derived_graph(persona_slug, tenant=tenant)
    if not report["valid"]:
        return {**report, "ok": False, "error_code": "GRAPH_VALIDATION_FAILED"}

    materialized: dict[str, Any] = {"ok": True}
    if materialize:
        materialized = graph_json_importer.import_graph_json(
            graph_json=graph,
            source="graph_json_v2_backfill",
        )
        if materialized.get("ok") is False:
            return {
                **report,
                "ok": False,
                "error_code": "GRAPH_MATERIALIZATION_FAILED",
                "materialization": materialized,
            }

    next_version = 1 if not current else current[0] + 1
    checksum = graph_json_v2_store.save_version(
        persona_slug,
        next_version,
        graph,
        source="graph_json_v2_backfill",
        note="Canonical Graph JSON v2 bootstrap from current derived graph",
    )
    return {
        **report,
        "ok": True,
        "skipped_existing": False,
        "version": next_version,
        "checksum": checksum,
        "materialization": materialized,
    }
