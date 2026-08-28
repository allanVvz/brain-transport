"""Atomic-ish canonical Graph JSON publication over existing system_events.

The published event is deliberately written *after* mandatory projections
complete.  Projection writes are idempotent, so a failed attempt can safely be
retried without advancing the canonical version.
"""
from __future__ import annotations

import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from schemas.graph_json_v2 import Edge, GraphJson, Node
from services import (
    graph_json_importer,
    graph_json_v2_store,
    graph_json_v2_validator,
    graph_markdown,
    graph_action_policy,
    graph_conversation_contract,
    supabase_client,
)


class VersionConflict(RuntimeError):
    def __init__(self, *, expected: int, current: int):
        super().__init__(f"Graph version conflict: expected {expected}, current {current}")
        self.expected = expected
        self.current = current


class GraphValidationError(RuntimeError):
    def __init__(self, errors: list[str]):
        super().__init__("Canonical graph validation failed")
        self.errors = errors


class ProjectionFailed(RuntimeError):
    def __init__(self, *, operation_id: str, version: int, error: str):
        super().__init__(error)
        self.operation_id = operation_id
        self.version = version


def _set_dotted(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [part for part in dotted_key.split(".") if part]
    if not parts:
        raise ValueError("patch path cannot be empty")
    cursor = target
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def apply_operations(graph: GraphJson, operations: list[dict[str, Any]]) -> GraphJson:
    """Apply mutation conveniences to a complete document in memory."""
    payload = deepcopy(graph.model_dump(mode="json"))
    nodes = {str(node["id"]): node for node in payload.get("nodes") or []}
    edges = {str(edge["id"]): edge for edge in payload.get("edges") or []}
    for operation in operations:
        op = str(operation.get("op") or "")
        node_id = str(operation.get("node_id") or operation.get("id") or "")
        edge_id = str(operation.get("edge_id") or operation.get("id") or "")
        if op == "add_node":
            node = deepcopy(operation.get("node") or {})
            if not node.get("id") or node["id"] in nodes:
                raise ValueError("add_node requires a new stable node id")
            nodes[node["id"]] = node
        elif op == "update_node":
            if node_id not in nodes:
                raise ValueError(f"node not found: {node_id}")
            for key, value in (operation.get("patch") or {}).items():
                _set_dotted(nodes[node_id], key, value)
            lifecycle = nodes[node_id].setdefault("lifecycle", {})
            lifecycle["revision"] = int(lifecycle.get("revision") or 1) + 1
            # A material edit invalidates the prior approval.
            lifecycle["status"] = "pending_validation"
            lifecycle.pop("approved_revision_checksum", None)
            nodes[node_id].pop("markdown", None)
        elif op == "approve_node":
            if node_id not in nodes:
                raise ValueError(f"node not found: {node_id}")
            lifecycle = nodes[node_id].setdefault("lifecycle", {})
            lifecycle["status"] = "approved"
            lifecycle["approved_at"] = datetime.now(timezone.utc).isoformat()
            if operation.get("approved_by"):
                lifecycle["approved_by"] = str(operation["approved_by"])
            if operation.get("reason"):
                nodes[node_id].setdefault("provenance", {})["reason"] = str(operation["reason"])
        elif op in {"archive_node", "remove_node"}:
            if node_id not in nodes:
                raise ValueError(f"node not found: {node_id}")
            nodes[node_id].setdefault("lifecycle", {})["status"] = "archived"
            for edge in edges.values():
                if node_id in {edge.get("source"), edge.get("target")}:
                    edge.setdefault("lifecycle", {})["status"] = "revoked"
        elif op == "add_edge":
            edge = deepcopy(operation.get("edge") or {})
            if not edge.get("id") or edge["id"] in edges:
                raise ValueError("add_edge requires a new stable edge id")
            edges[edge["id"]] = edge
        elif op in {"revoke_edge", "remove_edge"}:
            if edge_id not in edges:
                raise ValueError(f"edge not found: {edge_id}")
            edges[edge_id].setdefault("lifecycle", {})["status"] = "revoked"
        elif op == "set_status":
            if node_id not in nodes:
                raise ValueError(f"node not found: {node_id}")
            nodes[node_id].setdefault("lifecycle", {})["status"] = operation.get("value")
        else:
            raise ValueError(f"unsupported graph operation: {op}")
    payload["nodes"] = list(nodes.values())
    payload["edges"] = list(edges.values())
    payload["schema_version"] = "2.1"
    payload["status"] = "draft"
    payload["graph_version"] = None
    payload["content_checksum"] = None
    return GraphJson.model_validate(payload)


_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _scope_key(persona_slug: str, brand_slug: str | None) -> str:
    return f"{persona_slug}:{brand_slug or 'default'}"


def _lock_for(scope: str) -> threading.RLock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(scope, threading.RLock())


def _audit(
    event_type: str,
    *,
    persona_slug: str,
    brand_slug: str | None,
    version: int,
    source: str,
    idempotency_key: str | None,
    payload: dict[str, Any] | None = None,
    level: str = "info",
) -> None:
    try:
        persona = supabase_client.get_persona(persona_slug) or {}
        supabase_client.insert_event(
            {
                "event_type": event_type,
                "entity_type": "graph_document",
                "entity_id": f"{_scope_key(persona_slug, brand_slug)}:v{version}",
                "persona_id": persona.get("id"),
                "payload": {
                    "persona_slug": persona_slug,
                    "brand_slug": brand_slug,
                    "version": version,
                    "idempotency_key": idempotency_key,
                    "occurred_at": datetime.now(timezone.utc).isoformat(),
                    **(payload or {}),
                },
                "level": level,
                "source": source,
            },
            source=source,
        )
    except Exception:
        # The final graph_document_published event is mandatory and is written
        # by save_version. Auxiliary attempt/failure telemetry must not mask the
        # original projection error.
        return


def _load_current(persona_slug: str, brand_slug: str | None):
    try:
        return graph_json_v2_store.load_current(persona_slug, brand_slug)
    except TypeError:  # compatibility with older test doubles/callers
        return graph_json_v2_store.load_current(persona_slug)


def _idempotent_result(
    persona_slug: str,
    brand_slug: str | None,
    idempotency_key: str | None,
) -> dict[str, Any] | None:
    if not idempotency_key:
        return None
    for row in supabase_client.list_system_events(
        entity_type="graph_document",
        event_types=["graph_document_published", "graph_version_activated"],
        limit=500,
    ):
        payload = row.get("payload") or {}
        if payload.get("persona_slug") != persona_slug:
            continue
        if (payload.get("brand_slug") or None) != (brand_slug or None):
            continue
        if payload.get("idempotency_key") != idempotency_key:
            continue
        projections = payload.get("projections") or {}
        return {
            **projections,
            "ok": True,
            "idempotent_replay": True,
            "id": row.get("entity_id"),
            "persona_slug": persona_slug,
            "brand_slug": brand_slug,
            "version": int(payload.get("version") or 0),
            "checksum": payload.get("checksum"),
            "projections": projections,
        }
    return None


def commit(
    *,
    graph: GraphJson,
    persona_slug: str,
    brand_slug: str | None,
    source: str,
    reason: str,
    published_by: str | None,
    expected_version: int,
    idempotency_key: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Commit an immutable 2.1 version, then build and atomically activate it.

    Version allocation/idempotency happen under a Postgres advisory transaction
    lock.  A failed projector leaves the committed version inactive, so readers
    continue to use the previous activation.
    """
    if graph.schema_version != "2.1":
        raise GraphValidationError(["commit requires schema_version 2.1"])
    # Production deploys intentionally publish the same canonical source on
    # every release. Once its idempotency key has an activation event, replay
    # must be a read: re-importing and re-activating the same version rewrites
    # every RAG row and can outlive PostgREST's request timeout. A committed but
    # not yet activated attempt has no activation event and therefore still
    # resumes through the projector below.
    replay = _idempotent_result(persona_slug, brand_slug, idempotency_key)
    if replay:
        return replay
    graph = GraphJson.model_validate(graph.model_dump(mode="json"))
    graph = graph_conversation_contract.materialize_qualification_questions(graph)
    graph.graph_version = expected_version + 1
    graph.status = "committed"
    graph.provenance.base_version = expected_version
    graph.provenance.reason = reason
    graph.provenance.source = source
    graph.provenance.authored_by = published_by
    graph, policy_changes = graph_action_policy.apply(graph)
    try:
        graph = graph_markdown.canonicalize_graph(graph)
    except graph_markdown.GraphMarkdownError as exc:
        raise GraphValidationError(exc.errors) from exc
    is_valid, errors = graph_json_v2_validator.validate_graph_json(graph)
    if not is_valid:
        raise GraphValidationError(errors)
    graph.content_checksum = graph_json_v2_store.checksum_graph(graph)

    try:
        committed = graph_json_v2_store.commit_version(
            persona_slug=persona_slug,
            brand_slug=brand_slug,
            expected_version=expected_version,
            idempotency_key=idempotency_key,
            reason=reason,
            graph=graph,
            source=source,
            authored_by=published_by,
        )
    except Exception as exc:
        message = str(exc)
        if "GRAPH_VERSION_CONFLICT" in message:
            current = expected_version
            try:
                loaded = _load_current(persona_slug, brand_slug)
                current = int(loaded[0]) if loaded else 0
            except Exception:
                pass
            raise VersionConflict(expected=expected_version, current=current) from exc
        raise

    version = int(committed["graph_version"])
    operation_id = str(committed["operation_id"])
    checksum = str(committed.get("checksum") or graph.content_checksum)
    graph.graph_version = version
    graph.content_checksum = checksum
    try:
        projections = graph_json_importer.import_graph_json(
            graph_json=graph,
            source=source,
            session_id=session_id,
            version=version,
            graph_checksum=checksum,
        )
        if not projections or projections.get("ok") is not True:
            raise RuntimeError(str((projections or {}).get("errors") or "projection failed"))
    except Exception as exc:
        _audit(
            "graph_projection_failed",
            persona_slug=persona_slug,
            brand_slug=brand_slug,
            version=version,
            source=source,
            idempotency_key=idempotency_key,
            payload={
                "operation_id": operation_id,
                "checksum": checksum,
                "error": str(exc)[:1000],
            },
            level="error",
        )
        raise ProjectionFailed(operation_id=operation_id, version=version, error=str(exc)) from exc

    activation = graph_json_v2_store.activate_version(
        persona_slug=persona_slug,
        brand_slug=brand_slug,
        version=version,
        checksum=checksum,
        operation_id=operation_id,
        projections=projections,
        source=source,
    )
    return {
        "ok": True,
        "operation_id": operation_id,
        "graph_version": version,
        "version": version,
        "checksum": checksum,
        "status": activation.get("status") or "published",
        "idempotent_replay": bool(committed.get("idempotent_replay")),
        "projection_status_url": f"/graph-projections/operations/{operation_id}",
        "projections": projections,
        "policy_changes": policy_changes,
    }


def publish(
    *,
    graph: GraphJson,
    persona_slug: str,
    brand_slug: str | None,
    source: str,
    note: str | None = None,
    published_by: str | None = None,
    expected_version: int | None = None,
    idempotency_key: str | None = None,
    session_id: str | None = None,
    current_version_override: int | None = None,
) -> dict[str, Any]:
    graph = graph_conversation_contract.materialize_qualification_questions(graph)
    try:
        graph = graph_markdown.canonicalize_graph(graph)
    except graph_markdown.GraphMarkdownError as exc:
        raise GraphValidationError(exc.errors) from exc
    scope = _scope_key(persona_slug, brand_slug)
    with _lock_for(scope):
        replay = _idempotent_result(persona_slug, brand_slug, idempotency_key)
        if replay:
            return replay

        current = None if current_version_override is not None else _load_current(persona_slug, brand_slug)
        current_version = (
            int(current_version_override)
            if current_version_override is not None
            else (int(current[0]) if current else 0)
        )
        if expected_version is not None and int(expected_version) != current_version:
            raise VersionConflict(expected=int(expected_version), current=current_version)

        next_version = current_version + 1
        checksum = graph_json_v2_store.checksum_graph(graph)
        _audit(
            "graph_document_publish_attempted",
            persona_slug=persona_slug,
            brand_slug=brand_slug,
            version=next_version,
            source=source,
            idempotency_key=idempotency_key,
            payload={"checksum": checksum},
        )
        try:
            projections = graph_json_importer.import_graph_json(
                graph_json=graph,
                source=source,
                session_id=session_id,
                version=next_version,
                graph_checksum=checksum,
            )
            if not projections or projections.get("ok") is not True:
                raise RuntimeError(
                    str((projections or {}).get("errors") or (projections or {}).get("reindex_error") or "projection failed")
                )
            persisted_checksum = graph_json_v2_store.save_version(
                persona_slug,
                next_version,
                graph,
                brand_slug=brand_slug,
                source=source,
                note=note,
                published_by=published_by,
                idempotency_key=idempotency_key,
                projections=projections,
            )
        except Exception as exc:
            try:
                _audit(
                    "graph_document_publish_failed",
                    persona_slug=persona_slug,
                    brand_slug=brand_slug,
                    version=next_version,
                    source=source,
                    idempotency_key=idempotency_key,
                    payload={"checksum": checksum, "error": str(exc)[:1000]},
                    level="error",
                )
            finally:
                raise

        return {
            "ok": True,
            "idempotent_replay": False,
            "id": f"{scope}:v{next_version}",
            "persona_slug": persona_slug,
            "brand_slug": brand_slug,
            "version": next_version,
            "checksum": persisted_checksum,
            "projections": projections,
            "reindex_ok": True,
            "nodes_imported": projections.get("nodes_imported"),
            "edges_imported": projections.get("edges_imported"),
            "reindex_error": None,
        }


def sync(
    *,
    persona_slug: str,
    brand_slug: str | None,
    source: str,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    current = _load_current(persona_slug, brand_slug)
    if current is None:
        raise LookupError("No published graph document available")
    version, graph = current
    is_valid, errors = graph_json_v2_validator.validate_graph_json(graph)
    if not is_valid:
        raise GraphValidationError(errors)
    try:
        event = graph_json_v2_store.latest_event(persona_slug, brand_slug) or {}
    except Exception:
        event = {}
    checksum = ((event.get("payload") or {}).get("checksum")) or graph_json_v2_store.checksum_graph(graph)
    projections = graph_json_importer.import_graph_json(
        graph_json=graph,
        source=source,
        version=version,
        graph_checksum=checksum,
    )
    if not projections or projections.get("ok") is not True:
        raise RuntimeError(str((projections or {}).get("errors") or "projection sync failed"))
    _audit(
        "graph_document_synced",
        persona_slug=persona_slug,
        brand_slug=brand_slug,
        version=version,
        source=source,
        idempotency_key=idempotency_key,
        payload={"checksum": checksum, "projections": projections},
    )
    return {
        "ok": True,
        "persona_slug": persona_slug,
        "brand_slug": brand_slug,
        "version": version,
        "checksum": checksum,
        "projections": projections,
    }


def apply_sofia_patch(graph: GraphJson, patch: dict[str, Any]) -> GraphJson:
    """Translate Sofia's compatibility patch into a complete canonical document."""
    next_graph = GraphJson.model_validate(graph.model_dump())
    nodes = {node.id: node for node in next_graph.nodes}

    def refs() -> dict[str, str]:
        out: dict[str, str] = {}
        for node in nodes.values():
            out[f"id:{node.id}"] = node.id
            out[f"slug:{node.slug.lower()}"] = node.id
            if node.node_type == "persona":
                out["persona:self"] = node.id
                out["slug:self"] = node.id
            knowledge_id = str((node.data or {}).get("knowledge_node_id") or "")
            if knowledge_id:
                out[f"id:{knowledge_id}"] = node.id
        return out

    ref_map = refs()
    for raw in patch.get("nodes_upsert") or []:
        node_type = str(raw.get("node_type") or "").strip().lower()
        slug = str(raw.get("slug") or "").strip().lower()
        if not node_type or not slug:
            raise ValueError("Sofia nodes_upsert requires node_type and slug")
        existing_id = next(
            (
                node.id
                for node in nodes.values()
                if node.node_type == node_type and node.slug.lower() == slug
            ),
            None,
        )
        node_id = existing_id or f"node:{node_type}:{slug}"
        existing = nodes.get(node_id)
        nodes[node_id] = Node(
            id=node_id,
            node_type=node_type,
            slug=slug,
            label=str(raw.get("title") or (existing.label if existing else slug)),
            parent_id=existing.parent_id if existing else None,
            data={
                **((existing.data if existing else {}) or {}),
                **(raw.get("metadata") or {}),
                "summary": raw.get("summary") or ((existing.data if existing else {}) or {}).get("summary"),
                "status": raw.get("status") or ((existing.data if existing else {}) or {}).get("status") or "pending_validation",
                "tags": raw.get("tags") or ((existing.data if existing else {}) or {}).get("tags") or [node_type],
                "source": (raw.get("metadata") or {}).get("source") or "sofia_graph",
            },
        )
        ref_map = refs()

    remove_ids: set[str] = set()
    for raw in patch.get("nodes_delete") or []:
        value = raw if isinstance(raw, str) else raw.get("id") or raw.get("node_id") or raw.get("slug")
        key = str(value or "")
        resolved = ref_map.get(key) or ref_map.get(f"id:{key}") or ref_map.get(f"slug:{key.lower()}")
        if resolved:
            node = nodes.get(resolved)
            if node and node.node_type in {"persona", "embedded", "gallery"}:
                raise ValueError(f"Protected node cannot be removed: {node.node_type}")
            remove_ids.add(resolved)
    for node_id in remove_ids:
        nodes.pop(node_id, None)

    edges = {
        edge.id: edge
        for edge in next_graph.edges
        if edge.source in nodes and edge.target in nodes
    }
    for raw in patch.get("edges_delete") or []:
        value = raw if isinstance(raw, str) else raw.get("id") or raw.get("edge_id")
        if value:
            edges.pop(str(value), None)

    ref_map = refs()
    for raw in patch.get("edges_upsert") or []:
        source_ref = str(raw.get("source_ref") or raw.get("source") or "")
        target_ref = str(raw.get("target_ref") or raw.get("target") or "")
        source = ref_map.get(source_ref) or ref_map.get(f"id:{source_ref}") or ref_map.get(f"slug:{source_ref.lower()}")
        target = ref_map.get(target_ref) or ref_map.get(f"id:{target_ref}") or ref_map.get(f"slug:{target_ref.lower()}")
        if not source or not target:
            raise ValueError(f"Sofia edge reference not found: {source_ref} -> {target_ref}")
        source_relation = str(raw.get("relation_type") or "contains").strip().lower()
        primary = (raw.get("metadata") or {}).get("primary_tree") is not False
        relation_aliases = {
            "belongs_to_persona": "contains",
            "persona_has_brand": "contains",
            "brand_has_briefing": "contains",
            "brand_has_campaign": "contains",
            "briefing_has_campaign": "contains",
            "campaign_has_audience": "contains",
            "audience_has_product_group": "contains",
            "product_group_has_product": "contains",
            "supports_copy": "supports",
            "answers_question": "answers",
            "published_to_rag": "publishes_to",
            "gallery_asset": "uses_asset",
        }
        relation = relation_aliases.get(source_relation, source_relation)
        existing_id = next(
            (
                edge.id
                for edge in edges.values()
                if edge.source == source and edge.target == target and edge.relation == relation
            ),
            None,
        )
        edge_id = existing_id or f"edge:sofia:{uuid.uuid4().hex}"
        edges[edge_id] = Edge(
            id=edge_id,
            source=source,
            target=target,
            relation=relation,
            primary_tree=primary,
            metadata={
                **(raw.get("metadata") or {}),
                "created_from": "sofia_graph",
                "semantic_relation": source_relation,
            },
        )
        if primary and target in nodes:
            nodes[target].parent_id = source

    next_graph.nodes = list(nodes.values())
    next_graph.edges = list(edges.values())
    return next_graph

