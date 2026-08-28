from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from schemas.graph_json_v2 import Edge, GraphJson, Node
from services import graph_conversation_contract, graph_json_v2_validator, supabase_client
from services.vault_sync import VAULT_PATH as CONFIGURED_VAULT_PATH


VAULT_PATH = Path(CONFIGURED_VAULT_PATH)


RELATION_BY_PAIR: dict[tuple[str, str], str] = {
    ("persona", "brand"): "belongs_to_persona",
    ("brand", "briefing"): "contains",
    ("brand", "campaign"): "part_of_campaign",
    ("campaign", "briefing"): "contains",
    ("briefing", "campaign"): "part_of_campaign",
    ("campaign", "audience"): "targets_audience",
    ("briefing", "audience"): "targets_audience",
    ("brand", "tone"): "contains",
    ("campaign", "tone"): "contains",
    ("briefing", "tone"): "contains",
    ("audience", "product_group"): "audience_has_product_group",
    ("product_group", "product"): "product_group_has_product",
    ("product_group", "offer"): "contains",
    ("product", "offer"): "contains",
    ("product_group", "copy"): "supports_copy",
    ("product", "copy"): "supports_copy",
    ("offer", "copy"): "supports_copy",
    ("campaign", "rule"): "contains",
    ("briefing", "rule"): "contains",
    ("brand", "rule"): "contains",
    ("rule", "faq"): "answers_question",
    ("copy", "faq"): "answers_question",
    ("product", "faq"): "answers_question",
    ("product_group", "faq"): "answers_question",
    ("faq", "embedded"): "visible_to_agent",
    ("brand", "asset"): "uses_asset",
    ("campaign", "asset"): "uses_asset",
    ("product_group", "asset"): "uses_asset",
    ("product", "asset"): "uses_asset",
    ("asset", "gallery"): "gallery_asset",
}


def _slug(value: str) -> str:
    text = (value or "").strip().lower()
    out: list[str] = []
    last_dash = False
    for ch in text:
        if ch.isalnum():
            out.append(ch)
            last_dash = False
        elif not last_dash:
            out.append("-")
            last_dash = True
    return "".join(out).strip("-") or "node"


def _content_hash(content: str) -> str:
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()


def _node_markdown(node: Node) -> str:
    data = node.data or {}
    raw_content = data.get("content")
    if isinstance(raw_content, dict) and node.node_type == "faq":
        question = str(raw_content.get("question") or data.get("question") or node.label or "").strip()
        answer = str(raw_content.get("answer") or data.get("answer") or "").strip()
        markdown = f"# {question}" + (f"\n\n{answer}" if answer else "")
    else:
        markdown = str(data.get("markdown") or raw_content or "").strip()
    if markdown:
        return markdown
    title = node.label or node.slug
    summary = str(data.get("summary") or "").strip()
    return f"# {title}\n\n{summary}".strip()


_FOLDER_BY_TYPE: dict[str, str] = {
    "persona": "00_PERSONA",
    "brand": "01_BRAND",
    "briefing": "02_BRIEFING",
    "campaign": "03_CAMPAIGNS",
    "audience": "04_AUDIENCES",
    "product_group": "05_PRODUCT_GROUPS",
    "product": "06_PRODUCTS",
    "offer": "07_OFFERS",
    "copy": "08_COPY",
    "faq": "09_FAQ",
    "rule": "10_RULES",
    "tone": "10_TONE",
    "asset": "11_ASSETS",
    "embedded": "12_EMBEDDED",
    "gallery": "13_GALLERY",
}


def _persona_folder(persona_slug: str) -> str:
    return (persona_slug or "UNKNOWN").strip().upper().replace("-", "_")


def _file_path(graph: GraphJson, node: Node) -> str:
    supplied = str((node.data or {}).get("file_path") or "").replace("\\", "/")
    if supplied:
        candidate = Path(supplied)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"unsafe file_path on node {node.id}")
    folder = _FOLDER_BY_TYPE.get(node.node_type, "99_OTHER")
    return (
        f"AI-BRAIN/05_ENTITIES/CLIENTS/{_persona_folder(graph.persona_slug)}"
        f"/{folder}/{node.slug}.md"
    )


def _write_vault_file(relative_path: str, content: str) -> Path:
    root = VAULT_PATH.resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("file_path escapes the configured vault root") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _branch_path(graph: GraphJson, node: Node) -> list[str]:
    by_id = {item.id: item for item in graph.nodes}
    chain: list[str] = []
    current: Node | None = node
    visited: set[str] = set()
    while current and current.id not in visited:
        visited.add(current.id)
        chain.insert(0, f"{current.node_type}:{current.slug}")
        current = by_id.get(current.parent_id or "")
    return chain


def _relation_manifest(graph: GraphJson, node_id: str) -> list[dict[str, Any]]:
    return [
        {
            "id": edge.id,
            "source": edge.source,
            "target": edge.target,
            "relation": edge.relation,
            "primary_tree": edge.primary_tree,
        }
        for edge in graph.edges
        if edge.source == node_id or edge.target == node_id
    ]


def _projected_body(graph: GraphJson, node: Node) -> str:
    if node.node_type == "embedded":
        faq_ids = {
            edge.source
            for edge in graph.edges
            if edge.target == node.id and edge.relation in {"visible_to_agent", "contains"}
        }
        faqs = [item for item in graph.nodes if item.id in faq_ids and item.node_type == "faq"]
        lines = [f"# {node.label}", "", "## FAQs publicadas"]
        lines.extend(f"- [{faq.label}](../09_FAQ/{faq.slug}.md)" for faq in faqs)
        return "\n".join(lines).strip()
    if node.node_type == "gallery":
        asset_ids = {
            edge.source
            for edge in graph.edges
            if edge.target == node.id and edge.relation == "gallery_asset"
        }
        assets = [item for item in graph.nodes if item.id in asset_ids and item.node_type == "asset"]
        lines = [f"# {node.label}", "", "## Assets aprovados"]
        lines.extend(f"- [{asset.label}](../11_ASSETS/{asset.slug}.md)" for asset in assets)
        return "\n".join(lines).strip()
    return _node_markdown(node)


def _markdown_document(
    graph: GraphJson,
    node: Node,
    *,
    version: int | None,
    graph_checksum: str | None,
) -> str:
    body = _projected_body(graph, node)
    relative_path = _file_path(graph, node)
    content_checksum = _content_hash(body)
    frontmatter = {
        "graph_id": graph.graph_id,
        "graph_version": version,
        "graph_checksum": graph_checksum,
        "node_id": node.id,
        "node_type": node.node_type,
        "slug": node.slug,
        "status": (node.data or {}).get("status") or "pending_validation",
        "source": (node.data or {}).get("source") or "pending_source",
        "parent_id": node.parent_id,
        "branch_path": _branch_path(graph, node),
        "file_path": relative_path,
        "checksum": content_checksum,
        "relations": _relation_manifest(graph, node.id),
    }
    # JSON is valid YAML and avoids unsafe interpolation in frontmatter.
    return f"---\n{json.dumps(frontmatter, ensure_ascii=False, indent=2)}\n---\n\n{body}\n"


def _node_status(node: Node) -> str:
    status = str((node.data or {}).get("validation_status") or (node.data or {}).get("status") or "pending").lower()
    # Gallery and Embedded are protected terminal nodes.  They are operational
    # infrastructure, not content awaiting curation, so keep the DB state that
    # the public projection uses to discover them.
    if node.node_type in {"gallery", "embedded", "marketing_workspace"} and status in {"validated", "approved", "active", "ativo"}:
        return "active"
    # Markdown uses `validated` as the canonical approval state. The legacy
    # FAQ -> Embedded database trigger still requires the historical
    # `approved` label, so translate only this operational projection.
    if node.node_type == "faq" and status in {"validated", "approved", "active", "ativo"}:
        return "approved"
    if status in {"validated", "approved", "active", "ativo"}:
        return "validated"
    return "pending"


def _item_status(node: Node) -> str:
    status = str((node.data or {}).get("validation_status") or (node.data or {}).get("status") or "pending").lower()
    if status in {"validated", "approved", "active", "ativo"}:
        # knowledge_items has its own legacy enum and does not accept the
        # graph-node status ``validated``.  Keep the graph node validated, but
        # materialize the source item as ``approved`` (the closest allowed
        # canonical state).
        return "approved"
    return "pending"


def _node_metadata(
    graph: GraphJson,
    node: Node,
    relative_path: str,
    content: str,
    session_id: str | None,
    *,
    version: int | None = None,
    graph_checksum: str | None = None,
) -> dict[str, Any]:
    data = dict(node.data or {})
    data.update(
        {
            "graph_json_id": graph.graph_id,
            "graph_json_node_id": node.id,
            "graph_json_import": True,
            "schema_version": graph.schema_version,
            "slug": node.slug,
            "file_path": relative_path,
            "content_hash": _content_hash(content),
            "graph_version": version,
            "graph_checksum": graph_checksum,
            "active": True,
        }
    )
    if session_id:
        data["session_id"] = session_id
    if node.parent_id:
        data["graph_json_parent_id"] = node.parent_id
    return data


def _default_relation(parent_type: str, child_type: str) -> str:
    return RELATION_BY_PAIR.get((parent_type, child_type), "contains")


def _rag_status(node: Node) -> bool:
    return str(
        (node.data or {}).get("validation_status")
        or (node.data or {}).get("status")
        or ""
    ).lower() in {"validated", "approved", "active", "ativo"}


def _publication_destinations(graph: GraphJson) -> dict[str, list[Node]]:
    """Active Embedded grants keyed by source document node id."""
    by_id = {node.id: node for node in graph.nodes}
    destinations: dict[str, list[Node]] = {}
    for edge in graph.edges:
        relation = edge.relation_type or edge.relation
        if relation != "publishes_to" or edge.lifecycle.status != "active":
            continue
        source = by_id.get(edge.source)
        target = by_id.get(edge.target)
        if not source or not target or target.node_type != "embedded" or not target.action:
            continue
        if not _rag_status(source):
            continue
        destinations.setdefault(source.id, []).append(target)
    return destinations


def _project_rag_document(
    *,
    graph: GraphJson,
    node: Node,
    graph_node: dict[str, Any],
    persona_id: str,
    version: int | None,
    graph_checksum: str | None,
    action_node: Node | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Idempotently project one approved Markdown node to one RAG chunk."""
    body = _projected_body(graph, node)
    hierarchy_path = _branch_path(graph, node)
    coordinate = graph_conversation_contract.coordinate_for_node(
        graph, node.id, graph_version=version
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    action_node_id = action_node.id if action_node else None
    destination_id = action_node.action.destination_id if action_node and action_node.action else None
    embedding_profile = (
        action_node.action.projection.embedding_profile_ref
        if action_node and action_node.action
        else None
    ) or "default-1536"
    metadata = {
        **(node.data or {}),
        "source": "graph_json_markdown",
        "source_node_id": node.id,
        "knowledge_node_id": graph_node.get("id"),
        "graph_json_node_id": node.id,
        "graph_id": graph.graph_id,
        "graph_version": version,
        "graph_checksum": graph_checksum,
        "hierarchy_path": hierarchy_path,
        "rag_index": "default",
        "action_node_id": action_node_id,
        "destination_id": destination_id,
        "active": True,
        **coordinate,
    }
    entry = supabase_client.upsert_knowledge_rag_entry(
        {
            "persona_id": persona_id,
            "content_type": node.node_type,
            "semantic_level": 50,
            "title": node.label,
            "content": body,
            "summary": str((node.data or {}).get("summary") or body)[:500],
            "canonical_key": (
                f"{graph.persona_slug}:{action_node_id}:v{version}:{node.id}"
                if action_node_id
                else f"{graph.persona_slug}:{node.node_type}:{node.slug}"
            ),
            "slug": node.slug,
            "language": str((node.data or {}).get("language") or "pt-BR"),
            # Destination-scoped rows stay invisible until the activation RPC
            # atomically publishes this version and withdraws the previous one.
            "status": "building" if action_node_id else "active",
            "tags": (node.data or {}).get("tags") or [node.node_type],
            "entities": [],
            "products": [node.slug] if node.node_type == "product" else [],
            "campaigns": [node.slug] if node.node_type == "campaign" else [],
            "metadata": metadata,
            "embedding_model": embedding_profile,
            "confidence": 1.0,
            "importance": 0.75,
            "validated_at": now_iso,
            "source_node_id": graph_node.get("id"),
            **({
                "action_node_id": action_node_id,
                "destination_id": destination_id,
                "graph_version": version,
                "graph_checksum": graph_checksum,
                "projection_status": "building",
            } if action_node_id else {}),
        }
    )
    if not entry or not entry.get("id"):
        raise RuntimeError(f"RAG entry import returned no id for {node.id}")
    chunks = supabase_client.replace_knowledge_rag_chunks(
        entry["id"],
        persona_id,
        [
            {
                "chunk_index": 0,
                "chunk_text": body,
                "chunk_summary": str((node.data or {}).get("summary") or node.label)[:280],
                "embedding_model": embedding_profile,
                "metadata": {
                    "source": "graph_json_markdown",
                    "source_node_id": node.id,
                    "knowledge_node_id": graph_node.get("id"),
                    "graph_json_node_id": node.id,
                    "graph_version": version,
                    "graph_checksum": graph_checksum,
                    "hierarchy_path": hierarchy_path,
                    "chunk_status": "pending_embedding",
                    "active": True,
                    **coordinate,
                },
                **({
                    "action_node_id": action_node_id,
                    "destination_id": destination_id,
                    "graph_version": version,
                    "graph_checksum": graph_checksum,
                    "source_node_id": graph_node.get("id"),
                    "projection_status": "building",
                } if action_node_id else {}),
            }
        ],
    )
    if not chunks:
        raise RuntimeError(f"RAG chunk import returned no rows for {node.id}")
    return entry, chunks


# content_type (Sofia Criar plan) -> canonical graph node_type. Non-canonical
# types (tone/entity/competitor/prompt/maker_material/other) are NOT part of the
# graph law chain and are dropped from the canonical tree.
_PLAN_TYPE_ALIASES: dict[str, str] = {
    "product_collection": "product_group",
    "publico": "audience",
    "rules": "rule",
}
_CANONICAL_NODE_TYPES: set[str] = {
    "persona", "brand", "briefing", "campaign", "audience", "tone",
    "product_group", "product", "offer", "copy", "rule", "faq",
    "embedded", "asset", "gallery",
}
_PERSONA_PARENTS: set[str] = {"", "self", "persona", "root", "global"}


def _canon_plan_type(content_type: str | None) -> str | None:
    t = (content_type or "").strip().lower()
    t = _PLAN_TYPE_ALIASES.get(t, t)
    return t if t in _CANONICAL_NODE_TYPES else None


def _parent_slug_of(entry: dict, links_by_target: dict[str, str]) -> str:
    meta = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    raw = str(meta.get("parent_slug") or "").strip().lower()
    if raw:
        return raw
    slug = str(entry.get("slug") or "").strip().lower()
    return links_by_target.get(slug, "")


def normalized_plan_to_graph_json(
    plan: dict[str, Any],
    session: dict[str, Any] | None = None,
    *,
    tenant: str = "qa",
    graph_id: str | None = None,
) -> GraphJson:
    """Pure conversion: Sofia Criar `normalized_plan` -> canonical GraphJson.

    Does NOT persist. The single canonical document the importer/validator and the
    front-end all consume. Non-canonical content_types are dropped from the tree.
    The result is meant to be validated by `graph_json_v2_validator.validate_graph_json`
    before any persistence (the importer calls the validator itself).
    """
    plan = plan or {}
    session = session or {}
    persona_slug = str(
        plan.get("persona_slug") or session.get("persona_slug") or ""
    ).strip().lower()
    persona_id = f"node:persona:{persona_slug}"
    persona_label = str(
        session.get("persona_name") or plan.get("persona_name") or persona_slug
    ).strip() or persona_slug

    nodes: list[Node] = [
        Node(
            id=persona_id,
            node_type="persona",
            slug=persona_slug,
            label=persona_label,
            parent_id=None,
            data={"source": "sofia_criar", "status": "validated"},
        )
    ]

    entries = [e for e in (plan.get("entries") or []) if isinstance(e, dict)]
    links_by_target: dict[str, str] = {}
    for link in plan.get("links") or []:
        if not isinstance(link, dict):
            continue
        tgt = str(link.get("target_slug") or "").strip().lower()
        src = str(link.get("source_slug") or "").strip().lower()
        if tgt and src and tgt not in links_by_target:
            links_by_target[tgt] = src

    # Pass 1: assign canonical node ids (dedupe by node_type+slug).
    resolved: list[dict[str, Any]] = []
    slug_to_id: dict[str, str] = {}
    seen_ids: set[str] = set()
    for entry in entries:
        node_type = _canon_plan_type(entry.get("content_type"))
        if not node_type or node_type == "persona":
            continue
        slug = _slug(str(entry.get("slug") or entry.get("title") or node_type))
        node_id = f"node:{node_type}:{slug}"
        if node_id in seen_ids:
            continue
        seen_ids.add(node_id)
        slug_to_id[slug] = node_id
        orig_slug = str(entry.get("slug") or "").strip().lower()
        if orig_slug:
            slug_to_id[orig_slug] = node_id
        resolved.append({"entry": entry, "node_type": node_type, "slug": slug, "id": node_id})

    # Pass 2: parent resolution.
    parent_id_by_node: dict[str, str] = {}
    for item in resolved:
        parent_slug = _parent_slug_of(item["entry"], links_by_target)
        if parent_slug in _PERSONA_PARENTS:
            parent_id_by_node[item["id"]] = persona_id
        else:
            parent_id_by_node[item["id"]] = slug_to_id.get(parent_slug, persona_id)

    nodes_by_id: dict[str, Node] = {}

    def _branch_path(node_id: str) -> list[dict[str, str]]:
        chain: list[dict[str, str]] = []
        cur = node_id
        guard = 0
        while cur and guard < 64:
            guard += 1
            node = nodes_by_id.get(cur)
            if node is None:
                break
            chain.insert(0, {"node_type": node.node_type, "slug": node.slug, "label": node.label})
            cur = node.parent_id or ""
        return chain

    # Pass 3: build nodes (persona first so branch_path can walk it).
    nodes_by_id[persona_id] = nodes[0]
    for item in resolved:
        entry = item["entry"]
        node_type = item["node_type"]
        meta = dict(entry.get("metadata") or {})
        content = str(entry.get("content") or "").strip()
        data: dict[str, Any] = {
            **meta,
            "source": meta.get("source") or entry.get("source") or "sofia_criar",
            "status": str(entry.get("status") or meta.get("status") or "pending_validation"),
            "summary": (content[:400] if content else meta.get("summary")),
            "tags": entry.get("tags") or meta.get("tags") or [node_type],
        }
        if content:
            data["markdown"] = content
        node = Node(
            id=item["id"],
            node_type=node_type,
            slug=item["slug"],
            label=str(entry.get("title") or item["slug"]),
            parent_id=parent_id_by_node[item["id"]],
            data=data,
        )
        nodes_by_id[item["id"]] = node
        nodes.append(node)

    # Pass 4: enrich FAQ nodes with the inherited-context fields the validator requires.
    for item in resolved:
        if item["node_type"] != "faq":
            continue
        node = nodes_by_id[item["id"]]
        parent_id = node.parent_id or persona_id
        parent = nodes_by_id.get(parent_id)
        node.data.setdefault("source_node_id", parent_id)
        node.data.setdefault("source_node_type", parent.node_type if parent else "persona")
        node.data.setdefault("branch_path", _branch_path(parent_id))
        meta = item["entry"].get("metadata") or {}
        if meta.get("markdown_document") is True or meta.get("generate_via") == "branch":
            node.data["markdown_document"] = True
            node.data.setdefault("markdown", str(item["entry"].get("content") or node.label))
            node.data["question_count"] = int(meta.get("question_count") or 1) or 1

    # Pass 5: primary edges mirror parent_id.
    edges: list[Edge] = []
    for idx, item in enumerate(resolved):
        node = nodes_by_id[item["id"]]
        parent_id = node.parent_id or persona_id
        parent = nodes_by_id.get(parent_id)
        relation = _default_relation(parent.node_type if parent else "persona", node.node_type)
        edges.append(
            Edge(
                id=f"edge:{idx + 1}",
                source=parent_id,
                target=node.id,
                relation=relation,
                primary_tree=True,
                metadata={"created_from": "normalized_plan_to_graph_json"},
            )
        )

    return GraphJson(
        schema_version="2.0",
        graph_id=graph_id or f"{persona_slug}-criar",
        tenant=tenant,
        persona_slug=persona_slug,
        status="draft",
        nodes=nodes,
        edges=edges,
    )


def import_graph_json(
    *,
    graph_json: GraphJson,
    source: str = "graph_json.import",
    session_id: str | None = None,
    version: int | None = None,
    graph_checksum: str | None = None,
) -> dict[str, Any]:
    is_valid, errors = graph_json_v2_validator.validate_graph_json(graph_json)
    if not is_valid:
        return {"ok": False, "error_code": "GRAPH_JSON_INVALID", "errors": errors}

    persona = supabase_client.get_persona(graph_json.persona_slug)
    if not persona:
        return {"ok": False, "error_code": "PERSONA_NOT_FOUND", "errors": [f"Persona not found: {graph_json.persona_slug}"]}
    persona_id = persona["id"]
    source_row = supabase_client.get_or_create_manual_source()

    graph_nodes_by_doc_id: dict[str, dict] = {}
    item_ids: list[str] = []
    written_files: list[str] = []
    projected_rows = supabase_client.list_graph_json_projection_nodes(persona_id)
    projections_by_doc_id: dict[str, list[dict[str, Any]]] = {}
    for projected in projected_rows:
        stable_id = str((projected.get("metadata") or {}).get("graph_json_node_id") or "")
        if stable_id:
            projections_by_doc_id.setdefault(stable_id, []).append(projected)

    for node in graph_json.nodes:
        metadata = dict(node.data or {})
        content = _markdown_document(
            graph_json,
            node,
            version=version,
            graph_checksum=graph_checksum,
        )
        relative_path = _file_path(graph_json, node)
        _write_vault_file(relative_path, content)
        written_files.append(relative_path)
        node_meta = _node_metadata(
            graph_json,
            node,
            relative_path,
            content,
            session_id,
            version=version,
            graph_checksum=graph_checksum,
        )
        if node.node_type in {"persona", "embedded", "gallery", "marketing_workspace"}:
            database_node_type = "embed" if node.node_type == "embedded" else node.node_type
            node_payload = {
                    "persona_id": persona_id,
                    "node_type": database_node_type,
                    "slug": node.slug,
                    "title": node.label,
                    "summary": metadata.get("summary") or _projected_body(graph_json, node)[:400],
                    "tags": metadata.get("tags") or [node.node_type],
                    "metadata": node_meta,
                    "status": "validated" if node.node_type == "persona" else _node_status(node),
                }
            matches = projections_by_doc_id.get(node.id) or []
            preferred = next(
                (row for row in matches if row.get("node_type") == database_node_type),
                matches[0] if matches else None,
            )
            if preferred:
                graph_node = supabase_client.update_knowledge_node(
                    str(preferred["id"]),
                    {
                        "title": node_payload["title"],
                        "summary": node_payload["summary"],
                        "tags": node_payload["tags"],
                        "metadata": {**(preferred.get("metadata") or {}), **node_meta},
                        "status": node_payload["status"],
                    },
                    mark_related_faqs=False,
                )
                graph_node = {**preferred, **(graph_node or {})}
                for duplicate in matches:
                    if duplicate.get("id") == preferred.get("id"):
                        continue
                    supabase_client.update_knowledge_node(
                        str(duplicate["id"]),
                        {"metadata": {
                            **(duplicate.get("metadata") or {}),
                            "active": False,
                            "projection_duplicate_of": preferred["id"],
                            "projection_removed_in_version": version,
                        }},
                        mark_related_faqs=False,
                    )
                    for duplicate_edge in supabase_client.list_edges_for_nodes([str(duplicate["id"])]):
                        supabase_client.delete_knowledge_edge(str(duplicate_edge["id"]))
            else:
                graph_node = supabase_client.upsert_knowledge_node(node_payload)
            if not graph_node or not graph_node.get("id"):
                raise RuntimeError(f"knowledge_node import returned no id for {node.id}")
            graph_nodes_by_doc_id[node.id] = graph_node
            continue

        item_payload = {
            "persona_id": persona_id,
            "source_id": source_row.get("id"),
            "status": _item_status(node),
            "content_type": node.node_type,
            "title": node.label,
            "content": content[:8000],
            "metadata": node_meta,
            "file_path": supabase_client.normalize_file_path(relative_path),
            "file_type": "md",
            "tags": metadata.get("tags") or [node.node_type],
            # Golden Dataset visibility is persona-scoped, never role-selected.
            "agent_visibility": [],
            "content_hash": _content_hash(content),
            "canonical_key": f"{graph_json.persona_slug}:{node.node_type}:{node.slug}",
            "canonical_hash": _content_hash(f"{graph_json.persona_slug}:{node.node_type}:{node.slug}:{content}"),
        }
        existing = supabase_client.get_knowledge_item_by_path(item_payload["file_path"])
        if existing and existing.get("id"):
            supabase_client.update_knowledge_item(existing["id"], item_payload)
            item = {**existing, **item_payload}
        else:
            item = supabase_client.insert_knowledge_item(item_payload)
        if not item or not item.get("id"):
            raise RuntimeError(f"knowledge_item import returned no id for {relative_path}")
        item_ids.append(item["id"])
        graph_node = supabase_client.upsert_knowledge_node(
            {
                "persona_id": persona_id,
                "source_table": "knowledge_items",
                "source_id": item["id"],
                "node_type": node.node_type,
                "slug": node.slug,
                "title": node.label,
                "summary": content[:400],
                "tags": item_payload["tags"],
                "metadata": {**item_payload["metadata"], "knowledge_item_id": item["id"]},
                "status": _node_status(node),
            }
        )
        if not graph_node or not graph_node.get("id"):
            raise RuntimeError(f"knowledge_node import returned no id for {node.id}")
        graph_nodes_by_doc_id[node.id] = graph_node
        supabase_client.update_knowledge_item(item["id"], {"metadata": {**item_payload["metadata"], "knowledge_node_id": graph_node["id"]}})

    # Updating a later product/copy intentionally marks existing FAQs as
    # pending_regeneration. During a canonical all-at-once publication those
    # later nodes belong to the same validated document, so restore the
    # document's approved FAQ status only after every node has been projected
    # and immediately before FAQ -> Embedded edges are enforced.
    for node in graph_json.nodes:
        if node.node_type != "faq" or _node_status(node) != "approved":
            continue
        projected = graph_nodes_by_doc_id.get(node.id)
        if projected and projected.get("id"):
            refreshed = supabase_client.update_knowledge_node(
                projected["id"],
                {"status": "approved"},
                mark_related_faqs=False,
            )
            if refreshed:
                graph_nodes_by_doc_id[node.id] = {**projected, **refreshed}

    edge_ids: list[str] = []
    for edge in graph_json.edges:
        if edge.lifecycle.status == "revoked":
            continue
        source_node = graph_nodes_by_doc_id.get(edge.source)
        target_node = graph_nodes_by_doc_id.get(edge.target)
        if not source_node or not target_node:
            raise RuntimeError(f"edge {edge.id} references non-imported node")
        relation = edge.relation if edge.relation and edge.relation != "main" else _default_relation(source_node.get("node_type"), target_node.get("node_type"))
        imported = supabase_client.upsert_knowledge_edge(
            source_node["id"],
            target_node["id"],
            relation,
            persona_id=persona_id,
            metadata={
                **(edge.metadata or {}),
                # v2.1 hierarchy is validated from the immutable document.
                # The legacy DB trigger has a narrower hard-coded chain, so
                # its projection must remain a reference edge during cutover.
                "primary_tree": edge.primary_tree is True and graph_json.schema_version == "2.0",
                "active": True,
                "graph_json_id": graph_json.graph_id,
                "graph_json_edge_id": edge.id,
                "created_from": source,
            },
        )
        if not imported or not imported.get("id"):
            raise RuntimeError(f"knowledge_edge import returned no id for {edge.id}")
        edge_ids.append(imported["id"])

    rag_entries_by_doc_id: dict[str, dict[str, Any]] = {}
    rag_entries_by_destination: dict[tuple[str, str], dict[str, Any]] = {}
    rag_chunk_ids: list[str] = []
    v21_destinations = _publication_destinations(graph_json) if graph_json.schema_version == "2.1" else {}
    for node in graph_json.nodes:
        if node.node_type in {"persona", "embedded", "gallery", "marketing_workspace"}:
            continue
        if not _rag_status(node):
            continue
        action_nodes: list[Node | None]
        if graph_json.schema_version == "2.1":
            action_nodes = list(v21_destinations.get(node.id) or [])
            if not action_nodes:
                continue
        else:
            if node.node_type == "faq":
                continue
            action_nodes = [None]
        node_entry_ids: list[str] = []
        node_chunk_ids: list[str] = []
        for action_node in action_nodes:
            entry, chunks = _project_rag_document(
                graph=graph_json,
                node=node,
                graph_node=graph_nodes_by_doc_id[node.id],
                persona_id=persona_id,
                version=version,
                graph_checksum=graph_checksum,
                action_node=action_node,
            )
            rag_entries_by_doc_id.setdefault(node.id, entry)
            if action_node:
                rag_entries_by_destination[(node.id, action_node.id)] = entry
            node_entry_ids.append(str(entry["id"]))
            node_chunk_ids.extend(str(chunk["id"]) for chunk in chunks if chunk.get("id"))
        rag_chunk_ids.extend(node_chunk_ids)
        imported_node = graph_nodes_by_doc_id[node.id]
        supabase_client.update_knowledge_node(
            imported_node["id"],
            {
                "metadata": {
                    **(imported_node.get("metadata") or {}),
                    "knowledge_rag_entry_id": node_entry_ids[0] if node_entry_ids else None,
                    "knowledge_rag_entry_ids": node_entry_ids,
                    "knowledge_rag_chunk_ids": node_chunk_ids,
                }
            },
            mark_related_faqs=False,
        )

    active_doc_node_ids = set(graph_nodes_by_doc_id)
    active_projection_edge_ids = set(edge_ids)
    nodes_deactivated = 0
    edges_deactivated = 0
    stale_nodes = supabase_client.list_graph_json_projection_nodes(persona_id)
    for stale in stale_nodes:
        stale_meta = stale.get("metadata") or {}
        doc_node_id = stale_meta.get("graph_json_node_id")
        if not doc_node_id or doc_node_id in active_doc_node_ids:
            continue
        next_meta = {
            **stale_meta,
            "active": False,
            "projection_removed_in_version": version,
            "projection_removed_from": source,
        }
        supabase_client.update_knowledge_node(
            stale["id"],
            {"metadata": next_meta},
            mark_related_faqs=False,
        )
        if stale.get("node_type") == "faq" and stale.get("source_id"):
            supabase_client.withdraw_faq_from_embedded(str(stale["source_id"]))
        nodes_deactivated += 1

    stale_edges = supabase_client.list_graph_json_projection_edges(persona_id)
    for stale in stale_edges:
        stale_meta = stale.get("metadata") or {}
        doc_edge_id = stale_meta.get("graph_json_edge_id")
        if not doc_edge_id or stale.get("id") in active_projection_edge_ids:
            continue
        if supabase_client.delete_knowledge_edge(stale.get("id")):
            edges_deactivated += 1

    faq_publications: list[dict[str, Any]] = []
    approved_statuses = {"approved", "validated", "embedded", "active", "ativo"}
    for edge in graph_json.edges:
        source_doc = next((item for item in graph_json.nodes if item.id == edge.source), None)
        target_doc = next((item for item in graph_json.nodes if item.id == edge.target), None)
        if not source_doc or not target_doc:
            continue
        if graph_json.schema_version == "2.1":
            continue
        if source_doc.node_type != "faq" or target_doc.node_type != "embedded":
            continue
        status = str(
            (source_doc.data or {}).get("validation_status")
            or (source_doc.data or {}).get("status")
            or ""
        ).lower()
        if status not in approved_statuses:
            raise RuntimeError(f"pending FAQ cannot be materialized into Embedded: {source_doc.id}")
        from services import approved_knowledge_snapshots

        publication = approved_knowledge_snapshots.publish_approved_node(
            graph_nodes_by_doc_id[source_doc.id]["id"],
            require_rag_for_faq=True,
        )
        faq_publications.append(publication)
        entry_ids = publication.get("rag_entry_ids") or []
        if entry_ids:
            rag_entries_by_doc_id[source_doc.id] = {"id": entry_ids[0]}
        rag_chunk_ids.extend(
            str(chunk_id) for chunk_id in (publication.get("rag_chunk_ids") or [])
        )

    rag_link_ids: list[str] = []
    graph_coordinates = graph_conversation_contract.coordinates(
        graph_json, graph_version=version
    )
    for edge in graph_json.edges:
        relation = edge.relation_type or edge.relation
        if relation == "publishes_to" or edge.lifecycle.status == "revoked":
            continue
        if graph_json.schema_version == "2.1":
            shared_actions = {
                action_id for node_id, action_id in rag_entries_by_destination if node_id == edge.source
            } & {
                action_id for node_id, action_id in rag_entries_by_destination if node_id == edge.target
            }
            pairs = [
                (
                    rag_entries_by_destination[(edge.source, action_id)],
                    rag_entries_by_destination[(edge.target, action_id)],
                    action_id,
                )
                for action_id in shared_actions
            ]
        else:
            source_entry = rag_entries_by_doc_id.get(edge.source)
            target_entry = rag_entries_by_doc_id.get(edge.target)
            pairs = [(source_entry, target_entry, None)] if source_entry and target_entry else []
        for source_entry, target_entry, action_id in pairs:
            action_node = next((item for item in graph_json.nodes if item.id == action_id), None)
            destination_id = action_node.action.destination_id if action_node and action_node.action else None
            link = supabase_client.upsert_knowledge_rag_link(
                {
                    "persona_id": persona_id,
                    "source_entry_id": source_entry["id"],
                    "target_entry_id": target_entry["id"],
                    "relation_type": relation,
                    "weight": edge.weight or 1.0,
                    "confidence": float((edge.metadata or {}).get("confidence") or 1.0),
                    "created_by": source,
                    "metadata": {
                        "graph_json_edge_id": edge.id,
                        "graph_version": version,
                        "graph_checksum": graph_checksum,
                        "action_node_id": action_id,
                        "destination_id": destination_id,
                        "primary_tree": edge.primary_tree,
                        "active": True,
                        "source_coordinate": graph_coordinates.get(edge.source),
                        "target_coordinate": graph_coordinates.get(edge.target),
                        "path_checksum": "sha256:" + _content_hash(
                            json.dumps(
                                {
                                    "graph_version": version,
                                    "edge_id": edge.id,
                                    "source": edge.source,
                                    "target": edge.target,
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                        ),
                    },
                    **({
                        "action_node_id": action_id,
                        "destination_id": destination_id,
                        "graph_version": version,
                        "graph_checksum": graph_checksum,
                    } if action_id else {}),
                }
            )
            if link.get("id"):
                rag_link_ids.append(str(link["id"]))

    action_projections: list[dict[str, Any]] = []
    if graph_json.schema_version == "2.1":
        for action_node in [node for node in graph_json.nodes if node.node_class == "action" and node.action]:
            grant_edges = [
                edge for edge in graph_json.edges
                if edge.relation_type == "publishes_to"
                and edge.target == action_node.id
                and edge.lifecycle.status == "active"
            ]
            source_ids = sorted(edge.source for edge in grant_edges)
            projection_checksum = "sha256:" + _content_hash(
                json.dumps(
                    {
                        "graph_checksum": graph_checksum,
                        "action_node_id": action_node.id,
                        "destination_id": action_node.action.destination_id,
                        "source_node_ids": source_ids,
                        "source_edge_ids": sorted(edge.id for edge in grant_edges),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            projection = {
                "projection_id": f"projection:{action_node.action.destination_id}:v{version}",
                "projection_type": action_node.action.projection.kind,
                "persona_slug": graph_json.persona_slug,
                "action_node_id": action_node.id,
                "destination_id": action_node.action.destination_id,
                "graph_version": version,
                "graph_checksum": graph_checksum,
                "projection_checksum": projection_checksum,
                "status": "published",
                "source_node_ids": source_ids,
                "source_edge_ids": sorted(edge.id for edge in grant_edges),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            action_projections.append(projection)
            supabase_client.record_graph_projection_event_v2(
                persona_slug=graph_json.persona_slug,
                projection=projection,
                source=source,
            )

    supabase_client.insert_event(
        {
            "event_type": "graph_json_imported",
            "entity_type": "graph_document",
            "entity_id": graph_json.graph_id,
            "persona_id": persona_id,
            "level": "info",
            "source": source,
            "payload": {
                "persona_slug": graph_json.persona_slug,
                "graph_id": graph_json.graph_id,
                "node_count": len(graph_json.nodes),
                "edge_count": len(graph_json.edges),
                "knowledge_item_ids": item_ids,
                "knowledge_edge_ids": edge_ids,
                "nodes_deactivated": nodes_deactivated,
                "edges_deactivated": edges_deactivated,
                "faq_publications": len(faq_publications),
                "rag_entries": len(rag_entries_by_doc_id),
                "rag_chunks": len(rag_chunk_ids),
                "rag_links": len(rag_link_ids),
                "graph_version": version,
                "graph_checksum": graph_checksum,
                "source_files": sorted(
                    {
                        str((node.data or {}).get("source_file"))
                        for node in graph_json.nodes
                        if (node.data or {}).get("source_file")
                    }
                ),
            },
        },
        source=source,
    )

    return {
        "ok": True,
        "graph_id": graph_json.graph_id,
        "persona_slug": graph_json.persona_slug,
        "nodes_imported": len(graph_nodes_by_doc_id),
        "edges_imported": len(edge_ids),
        "knowledge_item_ids": item_ids,
        "knowledge_node_ids": [node["id"] for node in graph_nodes_by_doc_id.values() if node.get("id")],
        "knowledge_edge_ids": edge_ids,
        "written_files": written_files,
        "nodes_deactivated": nodes_deactivated,
        "edges_deactivated": edges_deactivated,
        "faq_publications": faq_publications,
        "rag_entries_imported": len(rag_entries_by_doc_id),
        "rag_chunks_imported": len(rag_chunk_ids),
        "rag_links_imported": len(rag_link_ids),
        "rag_chunk_ids": rag_chunk_ids,
        "source_files": sorted(
            {
                str((node.data or {}).get("source_file"))
                for node in graph_json.nodes
                if (node.data or {}).get("source_file")
            }
        ),
        "action_projections": action_projections,
    }

