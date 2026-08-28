"""Gated GraphBundle materialization, staging and activation.

The source graph is materialized first, then recompiled from the canonical
knowledge tables. The staged runtime checksum must equal the reviewed plan.
Activation is a separate function with both approved checksums as CAS gates.
"""
from __future__ import annotations

from typing import Any, Callable

from services import graph_bundle, graph_compiler_v3, supabase_client


class GraphBundlePublishError(RuntimeError):
    pass


def _active_edge(row: dict[str, Any]) -> bool:
    return (row.get("metadata") or {}).get("active", True) is not False


def _preflight_source_scope(
    normalized: dict[str, Any],
    node_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
) -> None:
    desired_nodes = {
        (node["node_type"], node["slug"]) for node in normalized["nodes"]
    }
    has_authored_persona = any(
        node_type == "persona" and slug != "self"
        for node_type, slug in desired_nodes
    )
    existing_nodes = {
        (str(row.get("node_type") or ""), str(row.get("slug") or ""))
        for row in node_rows
        if str(row.get("status") or "").lower() in graph_compiler_v3.PUBLISHED_STATUSES
        and not (
            has_authored_persona
            and str(row.get("node_type") or "").lower() == "persona"
            and str(row.get("slug") or "").lower() == "self"
        )
    }
    unexpected_nodes = sorted(existing_nodes - desired_nodes)
    if unexpected_nodes:
        raise GraphBundlePublishError(
            "source_graph_has_unplanned_nodes:" + ",".join(
                f"{node_type}:{slug}" for node_type, slug in unexpected_nodes
            )
        )

    stable_by_projection = {
        str(row.get("id")): graph_compiler_v3._stable_node_id(row)
        for row in node_rows
    }
    desired_edges = {
        (edge["source"], edge["target"], edge["relation_type"])
        for edge in normalized["edges"]
    }
    existing_edges = {
        (
            stable_by_projection.get(str(row.get("source_node_id")), ""),
            stable_by_projection.get(str(row.get("target_node_id")), ""),
            str(row.get("relation_type") or ""),
        )
        for row in edge_rows if _active_edge(row)
    }
    unexpected_edges = sorted(existing_edges - desired_edges)
    if unexpected_edges:
        raise GraphBundlePublishError(
            "source_graph_has_unplanned_edges:" + ",".join(
                ":".join(edge) for edge in unexpected_edges
            )
        )


def stage_bundle(
    bundle: dict[str, Any],
    *,
    approved_draft_checksum: str,
    actor: str,
    embedder: Callable[[list[str]], list[list[float]]] | None = None,
) -> dict[str, Any]:
    plan = graph_bundle.build_publication_plan(bundle)
    if plan.get("validation_errors"):
        raise GraphBundlePublishError(
            "plan_blocked:" + ",".join(plan["validation_errors"])
        )
    if plan.get("publication_allowed") is not True:
        raise GraphBundlePublishError("publication_not_allowed_by_bundle")
    if plan.get("draft_checksum") != approved_draft_checksum:
        raise GraphBundlePublishError(
            f"approved_draft_checksum_mismatch:{approved_draft_checksum}:"
            f"{plan.get('draft_checksum')}"
        )

    normalized = graph_bundle.normalize_bundle(bundle)
    persona_id = normalized["persona"]["id"]
    persona_slug = normalized["persona"]["slug"]
    persona = supabase_client.get_persona(persona_slug)
    if not persona or str(persona.get("id") or "") != persona_id:
        raise GraphBundlePublishError("persona_scope_mismatch")

    current_nodes, current_edges = supabase_client.list_all_knowledge_graph(
        persona_id=persona_id, limit_nodes=10000
    )
    _preflight_source_scope(normalized, current_nodes, current_edges)

    projection_by_stable: dict[str, str] = {}
    for node in normalized["nodes"]:
        row = supabase_client.upsert_knowledge_node({
            "id": node["projection_node_id"],
            "persona_id": persona_id,
            "node_type": node["node_type"],
            "slug": node["slug"],
            "title": node["title"],
            "summary": node["summary"],
            "tags": node["tags"],
            "status": node["status"],
            "source_table": "graph_bundle",
            "metadata": {
                "graph_json_node_id": node["id"],
                "graph_bundle_draft_checksum": approved_draft_checksum,
                **node["data"],
            },
        })
        if not row or not row.get("id"):
            raise GraphBundlePublishError(f"node_materialization_failed:{node['id']}")
        if str(row["id"]) != node["projection_node_id"]:
            raise GraphBundlePublishError(
                f"projection_node_id_drift:{node['id']}:{row['id']}:"
                f"{node['projection_node_id']}"
            )
        row = supabase_client.update_knowledge_node(
            str(row["id"]),
            {
                "title": node["title"],
                "summary": node["summary"],
                "tags": node["tags"],
                "status": node["status"],
                "source_table": "graph_bundle",
                "metadata": {
                    "graph_json_node_id": node["id"],
                    "graph_bundle_draft_checksum": approved_draft_checksum,
                    **node["data"],
                },
            },
            mark_related_faqs=False,
        )
        if not row:
            raise GraphBundlePublishError(f"node_projection_replace_failed:{node['id']}")
        projection_by_stable[node["id"]] = str(row["id"])

    for edge in normalized["edges"]:
        row = supabase_client.upsert_knowledge_edge(
            projection_by_stable[edge["source"]],
            projection_by_stable[edge["target"]],
            edge["relation_type"],
            persona_id=persona_id,
            weight=edge["weight"],
            metadata={
                "active": True,
                "primary_tree": edge["relation_type"] == "contains",
                "graph_json_edge_id": edge["id"],
                "graph_bundle_draft_checksum": approved_draft_checksum,
                **edge["metadata"],
            },
        )
        if not row:
            raise GraphBundlePublishError(f"edge_materialization_failed:{edge['id']}")
        row = supabase_client.update_knowledge_edge(
            str(row["id"]),
            {
                "persona_id": persona_id,
                "weight": edge["weight"],
                "metadata": {
                    "active": True,
                    "primary_tree": edge["relation_type"] == "contains",
                    "graph_json_edge_id": edge["id"],
                    "graph_bundle_draft_checksum": approved_draft_checksum,
                    **edge["metadata"],
                },
            },
        )
        if not row:
            raise GraphBundlePublishError(f"edge_projection_replace_failed:{edge['id']}")

    node_rows, edge_rows = supabase_client.list_all_knowledge_graph(
        persona_id=persona_id, limit_nodes=10000
    )
    document = graph_compiler_v3.compile_graph(
        persona=persona,
        node_rows=node_rows,
        edge_rows=edge_rows,
        embedding_profile=normalized["metadata"]["embedding_profile"],
    )
    if document["checksum"] != plan["runtime_checksum"]:
        raise GraphBundlePublishError(
            f"materialized_runtime_checksum_mismatch:{document['checksum']}:"
            f"{plan['runtime_checksum']}"
        )

    staged = graph_compiler_v3.compile_persona_publication(
        persona_slug,
        activate=False,
        embedder=embedder,
        embedding_profile=normalized["metadata"]["embedding_profile"],
    )
    publication = staged.get("publication") or {}
    if publication.get("checksum") != plan["runtime_checksum"]:
        raise GraphBundlePublishError("staged_publication_checksum_mismatch")
    supabase_client.insert_event({
        "event_type": "graph_bundle_publication_staged",
        "entity_type": "graph_publication",
        "entity_id": str(publication.get("id") or ""),
        "persona_id": persona_id,
        "payload": {
            "persona_slug": persona_slug,
            "draft_checksum": plan["draft_checksum"],
            "runtime_checksum": plan["runtime_checksum"],
            "publication_id": publication.get("id"),
            "version": publication.get("version"),
            "actor": actor,
        },
    }, source="services.graph_bundle_publisher")
    return {"plan": plan, **staged}


def activate_staged_bundle(
    bundle: dict[str, Any],
    *,
    publication_id: str,
    approved_draft_checksum: str,
    approved_runtime_checksum: str,
    actor: str,
) -> dict[str, Any]:
    plan = graph_bundle.build_publication_plan(bundle)
    if plan.get("draft_checksum") != approved_draft_checksum:
        raise GraphBundlePublishError("activation_draft_checksum_mismatch")
    if plan.get("runtime_checksum") != approved_runtime_checksum:
        raise GraphBundlePublishError("activation_runtime_checksum_mismatch")

    # Activation must be scoped just as tightly as staging.  The publication
    # id comes from a previous stage, but it is still an external identifier at
    # this boundary; never let a bundle for persona A activate a staged
    # publication belonging to persona B.
    normalized = graph_bundle.normalize_bundle(bundle)
    persona_id = normalized["persona"]["id"]
    persona_slug = normalized["persona"]["slug"]
    persona = supabase_client.get_persona(persona_slug)
    if not persona or str(persona.get("id") or "") != persona_id:
        raise GraphBundlePublishError("persona_scope_mismatch")

    client = supabase_client.get_client()
    publication = (
        client.table("graph_publications").select("*")
        .eq("id", publication_id).maybe_single().execute().data
    )
    if not publication:
        raise GraphBundlePublishError("staged_publication_not_found")
    if str(publication.get("persona_id") or "") != persona_id:
        raise GraphBundlePublishError("staged_publication_persona_scope_mismatch")
    if publication.get("status") not in {"compiled", "active"}:
        raise GraphBundlePublishError(
            f"staged_publication_not_activatable:{publication.get('status')}"
        )
    if publication.get("checksum") != approved_runtime_checksum:
        raise GraphBundlePublishError("publication_checksum_changed_before_activation")
    activation = client.rpc(
        "activate_graph_publication_v3", {"p_publication_id": publication_id}
    ).execute().data
    supabase_client.insert_event({
        "event_type": "graph_bundle_publication_activated",
        "entity_type": "graph_publication",
        "entity_id": publication_id,
        "persona_id": publication.get("persona_id"),
        "payload": {
            "draft_checksum": approved_draft_checksum,
            "runtime_checksum": approved_runtime_checksum,
            "publication_id": publication_id,
            "version": publication.get("version"),
            "actor": actor,
        },
    }, source="services.graph_bundle_publisher")
    return {"publication": publication, "activation": activation}
