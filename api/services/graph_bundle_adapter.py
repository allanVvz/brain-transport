"""Adapt a Sofia (kb_intake) normalized knowledge plan into a GraphBundle.

Sofia's plan entries and services.graph_bundle's node/edge contract are
structurally close (same content_type/node_type vocabulary, same
knowledge_taxonomy.PRIMARY_CHAIN hierarchy Sofia's own tool-calling path
already enforces) but not identical: plan entry status is Portuguese, entries
don't distinguish a "contains" tree edge from other relation types, and
nothing in the plan format guarantees a single persona root reachable via
"contains" the way graph_bundle.normalize_bundle requires.

This module is the one-way bridge from a Sofia plan to that bundle shape.
It performs no I/O and never publishes anything itself -- callers still run
the result through services.graph_bundle.build_publication_plan and stop for
human approval before services.graph_bundle_publisher touches any store.

Only a plan entry the operator has explicitly confirmed ("confirmado") is
eligible to publish. Everything else (Portuguese "pendente_validacao",
"inferido", or a child whose parent isn't itself confirmed yet) is reported
back as held_back instead of silently entering the bundle -- the same
held-back pattern used by hand for the Tock Fatal catalog this session.
"""
from __future__ import annotations

from typing import Any

from services import graph_bundle, graph_compiler_v3

_STATUS_MAP = {
    "confirmado": "validated",
    "confirmed": "validated",
    "validated": "validated",
    "pendente_validacao": "pending_validation",
    "pending_validation": "pending_validation",
    "inferido": "pending_validation",
    "inferred": "pending_validation",
}

_DEFAULT_EMBEDDING_PROFILE = {
    "embedding_provider": "local",
    "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "embedding_dimension": graph_compiler_v3.EMBEDDING_DIMENSION,
}


def _mapped_status(raw: Any) -> str:
    return _STATUS_MAP.get(str(raw or "").strip().lower(), "pending_validation")


def normalized_plan_to_graph_bundle(
    plan: dict[str, Any],
    session: dict[str, Any],
    *,
    base_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build `{"bundle": <GraphBundle dict>, "held_back": [...]}` from a plan.

    `base_bundle`, when given, is treated as the persona's current graph
    (e.g. compiled from `supabase_client.list_all_knowledge_graph` the same
    way this session's manual work did) so the result is additive over what
    is already live, matching `graph_bundle.build_publication_plan`'s
    `current_document` diff semantics.

    Mirrors `kb_intake_service.normalized_plan_to_graph_json`'s persona/entry
    walk, but emits nodes/edges in the shape `services.graph_bundle.
    normalize_bundle` expects instead of `schemas.graph_json_v2.GraphJson`.
    """
    # Local import: kb_intake_service imports this module, so importing it
    # back at module scope would create a cycle.
    from services.kb_intake_service import _entry_parent_slug, _entry_type

    persona_slug = str(
        plan.get("persona_slug")
        or (session.get("classification") or {}).get("persona_slug")
        or session.get("persona_slug")
        or ""
    ).strip()
    if not persona_slug:
        raise ValueError("persona_slug is required to build a GraphBundle")
    persona_id = str(session.get("persona_id") or plan.get("persona_id") or "").strip()
    if not persona_id:
        raise ValueError("persona_id is required to build a GraphBundle")

    entries = [entry for entry in (plan.get("entries") or []) if isinstance(entry, dict)]
    session_id = str(session.get("id") or "unknown")
    default_source = str(plan.get("source") or f"kb_intake.sofia:{session_id}")

    base = base_bundle or {}
    base_nodes = [node for node in (base.get("nodes") or []) if isinstance(node, dict)]
    base_edges = [edge for edge in (base.get("edges") or []) if isinstance(edge, dict)]

    persona_node_id = f"persona:{persona_slug}"
    slug_to_node_id: dict[str, str] = {persona_slug: persona_node_id, "self": persona_node_id}
    included_ids = {str(node.get("id")) for node in base_nodes}
    for node in base_nodes:
        slug = str(node.get("slug") or "")
        if slug:
            slug_to_node_id.setdefault(slug, str(node.get("id")))

    nodes: list[dict[str, Any]] = list(base_nodes)
    if persona_node_id not in included_ids:
        nodes.append({
            "id": persona_node_id,
            "node_type": "persona",
            "slug": persona_slug,
            "title": persona_slug,
            "summary": f"Persona {persona_slug}.",
            "tags": ["persona"],
            "status": "validated",
            "data": {"source": default_source, "status": "validated"},
        })
        included_ids.add(persona_node_id)

    held_back: list[dict[str, str]] = []
    candidate_nodes: dict[str, dict[str, Any]] = {}
    for entry in entries:
        ctype = _entry_type(entry)
        slug = str(entry.get("slug") or "").strip()
        if not ctype or ctype == "persona" or not slug:
            continue
        node_id = f"{ctype}:{slug}"
        slug_to_node_id[slug] = node_id
        metadata = dict(entry.get("metadata") or {})
        content = str(entry.get("content") or metadata.get("markdown") or "").strip()
        status = _mapped_status(entry.get("status") or metadata.get("validation_status"))
        node = {
            "id": node_id,
            "node_type": ctype,
            "slug": slug,
            "title": str(entry.get("title") or slug),
            "summary": content or str(entry.get("title") or slug),
            "tags": list(entry.get("tags") or []),
            "status": status,
            "data": {
                **{k: v for k, v in metadata.items() if k != "parent_slug"},
                "content": content,
                "source": str(metadata.get("source") or default_source),
                "status": status,
                "_parent_slug": str(_entry_parent_slug(entry) or "self"),
            },
        }
        if status != "validated":
            held_back.append({
                "id": node_id, "title": node["title"],
                "reason": "aguardando confirmacao do operador (status != confirmado)",
            })
            continue
        candidate_nodes[node_id] = node

    # A confirmed child whose parent isn't itself confirmed (yet) can't be
    # safely attached to the "contains" tree -- hold it back too rather than
    # silently re-parent it onto the persona root.
    changed = True
    while changed:
        changed = False
        for node_id, node in list(candidate_nodes.items()):
            parent_slug = node["data"]["_parent_slug"]
            parent_id = slug_to_node_id.get(parent_slug) or persona_node_id
            if parent_id in included_ids or parent_id in candidate_nodes:
                continue
            held_back.append({
                "id": node_id, "title": node["title"],
                "reason": f"pai ainda nao confirmado ({parent_slug})",
            })
            del candidate_nodes[node_id]
            changed = True

    new_nodes = []
    edges: list[dict[str, Any]] = list(base_edges)
    seen_edges = {
        (str(e.get("source")), str(e.get("target")), str(e.get("relation_type")))
        for e in base_edges
    }
    for node_id, node in candidate_nodes.items():
        parent_slug = node["data"].pop("_parent_slug")
        parent_id = slug_to_node_id.get(parent_slug) or persona_node_id
        new_nodes.append(node)
        included_ids.add(node_id)
        key = (parent_id, node_id, "contains")
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edges.append({
            "id": f"edge:{parent_id}->{node_id}",
            "source": parent_id,
            "target": node_id,
            "relation_type": "contains",
            "weight": 1.0,
            "metadata": {"source": "kb_intake.normalized_plan"},
        })

    # Semantic (non-"contains") links the operator/Sofia added via
    # sofia_tools.tool_connect_nodes (plan["links"]) were previously
    # silently dropped here -- this is what let a hand-authored
    # `include_in_branch` edge (or any other semantic edge) vanish before
    # ever reaching the compiled GraphBundle. Both endpoints must already be
    # in the publishable set (not held back) or the link is skipped, same
    # as a "contains" edge to a held-back parent would be.
    for link in plan.get("links") or []:
        if not isinstance(link, dict):
            continue
        source_slug = str(link.get("source_slug") or "").strip()
        target_slug = str(link.get("target_slug") or "").strip()
        relation_type = str(link.get("relation_type") or "contains").strip() or "contains"
        source_id = slug_to_node_id.get(source_slug) or (persona_node_id if source_slug in ("self", persona_slug) else None)
        target_id = slug_to_node_id.get(target_slug) or (persona_node_id if target_slug in ("self", persona_slug) else None)
        if not source_id or not target_id or source_id not in included_ids or target_id not in included_ids:
            continue
        key = (source_id, target_id, relation_type)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edges.append({
            "id": f"edge:{source_id}->{target_id}:{relation_type}",
            "source": source_id,
            "target": target_id,
            "relation_type": relation_type,
            "weight": 1.0,
            "metadata": dict(link.get("metadata") or {}, source="kb_intake.sofia_link"),
        })

    bundle: dict[str, Any] = {
        "bundle_version": "1.0",
        "persona": {"id": persona_id, "slug": persona_slug},
        "metadata": dict(base.get("metadata") or {}) or {
            "purpose": f"kb_intake_sofia_{persona_slug}",
            "source": default_source,
            "publication_allowed": True,
            "embedding_profile": dict(_DEFAULT_EMBEDDING_PROFILE),
        },
        "nodes": nodes + new_nodes,
        "edges": edges,
    }
    return {"bundle": bundle, "held_back": held_back}


_DETACHED_TERMINAL_TYPES = {"embed", "embedded", "gallery"}


def ensure_branch_reachability(bundle: dict[str, Any]) -> dict[str, Any]:
    """Auto-repair pass: any node not reachable from ANY audience branch's
    closure (primary "contains" tree, `global_context` capability, or an
    existing semantic edge) gets a persona-rooted `visible_to_agent` edge
    with `metadata.include_in_branch: true` added automatically -- the same
    mechanism validated by hand for the Tock Fatal catalog this session,
    now applied without anyone having to pick an edge type.

    This is what "the agent on the number receives all persona knowledge,
    branch-bound or not" actually means mechanically: closure membership is
    what makes a node embeddable/searchable at all (services.graph_compiler_v3
    only chunks nodes inside SOME branch's closure) -- a node reachable from
    nothing is invisible in real conversation even though it compiles and
    counts fine in an audit, which is exactly the incident this fixes.

    Returns a NEW bundle dict (never mutates the input). If the bundle
    doesn't compile yet (still being built, or genuinely invalid) or has no
    branch anchors at all, returns the input unchanged -- real validation
    errors are still the caller's job to surface, this function only adds
    edges to an otherwise-valid bundle."""
    try:
        normalized = graph_bundle.normalize_bundle(bundle)
    except graph_bundle.GraphBundleError:
        return bundle
    try:
        document = graph_bundle.compile_bundle(bundle)
    except graph_compiler_v3.GraphCompilationError:
        return bundle

    memberships = document.get("branch_memberships") or {}
    if not memberships:
        return bundle

    reachable = {node_id for members in memberships.values() for node_id in members}
    persona_id = next(
        (node["id"] for node in normalized["nodes"] if node["node_type"] == "persona"), None,
    )
    if not persona_id:
        return bundle

    missing = [
        node["id"] for node in normalized["nodes"]
        if node["id"] != persona_id
        and node["node_type"] not in _DETACHED_TERMINAL_TYPES
        and node["id"] not in reachable
    ]
    if not missing:
        return bundle

    edges = list(bundle.get("edges") or [])
    existing_keys = {(e.get("source"), e.get("target"), e.get("relation_type")) for e in edges}
    for node_id in missing:
        key = (persona_id, node_id, "visible_to_agent")
        if key in existing_keys:
            continue
        edges.append({
            "id": f"edge:auto-visible:{node_id}",
            "source": persona_id,
            "target": node_id,
            "relation_type": "visible_to_agent",
            "weight": 1.0,
            "metadata": {"include_in_branch": True, "source": "auto_branch_reachability_repair"},
        })

    repaired = dict(bundle)
    repaired["edges"] = edges
    return repaired
