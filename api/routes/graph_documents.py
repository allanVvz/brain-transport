from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from schemas.graph_json_v2 import GraphJson
from services import auth_service, graph_document_publisher, graph_json_importer, graph_json_v2_store
from services import graph_json_v21_adapter
from services import graph_json_v2_validator, supabase_client

router = APIRouter(prefix="/graph-documents", tags=["graph-documents"])


class _MutationBody(BaseModel):
    persona_slug: str = Field(..., min_length=1)
    graph_json: dict
    source: str
    note: Optional[str] = None
    expected_version: Optional[int] = Field(None, ge=0)
    idempotency_key: Optional[str] = Field(None, min_length=1, max_length=200)


class PublishGraphDocumentBody(_MutationBody):
    brand_slug: Optional[str] = None
    source: str = "import_v1_to_v2"


class ApplyPatchBody(_MutationBody):
    source: str = "apply_patch"


class ImportGraphDocumentBody(_MutationBody):
    source: str = "graph_documents.import_json"
    session_id: Optional[str] = None


class RollbackBody(BaseModel):
    persona_slug: str = Field(..., min_length=1)
    version: int = Field(..., ge=1)
    note: Optional[str] = None
    expected_version: Optional[int] = Field(None, ge=0)
    idempotency_key: Optional[str] = Field(None, min_length=1, max_length=200)


class SyncBody(BaseModel):
    persona_slug: str = Field(..., min_length=1)
    brand_slug: Optional[str] = None
    idempotency_key: Optional[str] = Field(None, min_length=1, max_length=200)


class ReindexBody(SyncBody):
    note: Optional[str] = None


class ValidateGraphDocumentBody(BaseModel):
    persona_slug: str = Field(..., min_length=1)
    graph_json: dict


class CommitGraphDocumentBody(BaseModel):
    persona_slug: str = Field(..., min_length=1)
    brand_slug: Optional[str] = None
    expected_version: int = Field(..., ge=0)
    idempotency_key: str = Field(..., min_length=1, max_length=200)
    reason: str = Field(..., min_length=1, max_length=1000)
    operations: list[dict[str, Any]] = Field(default_factory=list)
    graph_json: Optional[dict] = None
    source: str = "graph_documents.commit"


def _latest_event(persona_slug: str, brand_slug: Optional[str]) -> Optional[dict]:
    return graph_json_v2_store.latest_event(persona_slug, brand_slug)


def _validate_graph_json_or_422(graph_json: dict) -> GraphJson:
    try:
        graph = GraphJson.model_validate(graph_json)
    except Exception as exc:
        raise HTTPException(422, f"Invalid graph_json payload: {exc}") from exc
    is_valid, errors = graph_json_v2_validator.validate_graph_json(graph)
    if not is_valid:
        raise HTTPException(422, {"code": "GRAPH_VALIDATION_FAILED", "errors": errors})
    return graph


def _assert_persona_view(request: Request, persona_slug: str) -> None:
    auth_service.assert_persona_access(request, persona_slug=persona_slug)


def _assert_persona_edit(request: Request, persona_slug: str) -> None:
    user = auth_service.current_user(request)
    if auth_service.is_admin(user):
        return
    for row in auth_service.allowed_access(request):
        if row.get("persona_slug") == persona_slug and row.get("can_edit"):
            return
    raise HTTPException(status_code=403, detail="Acesso de edicao negado para esta persona.")


def _assert_graph_persona_matches(body_persona_slug: str, graph: GraphJson) -> None:
    if graph.persona_slug != body_persona_slug:
        raise HTTPException(
            422,
            {
                "code": "GRAPH_PERSONA_MISMATCH",
                "errors": [
                    f"body persona_slug={body_persona_slug} differs from graph_json persona_slug={graph.persona_slug}"
                ],
            },
        )


def _publish_or_http(**kwargs):
    try:
        return graph_document_publisher.publish(**kwargs)
    except graph_document_publisher.VersionConflict as exc:
        raise HTTPException(
            409,
            {
                "code": "GRAPH_VERSION_CONFLICT",
                "expected_version": exc.expected,
                "current_version": exc.current,
            },
        ) from exc
    except graph_document_publisher.GraphValidationError as exc:
        raise HTTPException(
            422, {"code": "GRAPH_VALIDATION_FAILED", "errors": exc.errors}
        ) from exc
    except Exception as exc:
        raise HTTPException(502, str(exc)) from exc


def _commit_or_http(**kwargs):
    try:
        return graph_document_publisher.commit(**kwargs)
    except graph_document_publisher.VersionConflict as exc:
        raise HTTPException(
            409,
            {
                "code": "GRAPH_VERSION_CONFLICT",
                "expected_version": exc.expected,
                "current_version": exc.current,
            },
        ) from exc
    except graph_document_publisher.GraphValidationError as exc:
        raise HTTPException(422, {"code": "GRAPH_VALIDATION_FAILED", "errors": exc.errors}) from exc
    except graph_document_publisher.ProjectionFailed as exc:
        raise HTTPException(
            502,
            {
                "code": "GRAPH_PROJECTION_FAILED",
                "operation_id": exc.operation_id,
                "graph_version": exc.version,
                "error": str(exc),
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, str(exc)) from exc


@router.get("/current")
def graph_document_current(
    request: Request,
    persona_slug: str = Query(...),
    brand_slug: Optional[str] = Query(None),
):
    _assert_persona_view(request, persona_slug)
    evt = _latest_event(persona_slug, brand_slug)
    if not evt:
        return {
            "id": None,
            "persona_slug": persona_slug,
            "brand_slug": brand_slug,
            "version": 0,
            "checksum": None,
            "graph_json": {},
            "published_at": None,
            "source": None,
            "note": None,
            "projections": {},
        }
    payload = evt.get("payload") or {}
    return {
        "id": evt.get("entity_id"),
        "persona_slug": payload.get("persona_slug"),
        "brand_slug": payload.get("brand_slug"),
        "version": payload.get("version", 1),
        "checksum": payload.get("checksum"),
        "graph_json": payload.get("graph_json") or {},
        "published_at": evt.get("created_at"),
        "source": payload.get("source"),
        "note": payload.get("note"),
        "projections": payload.get("projections") or {},
    }


@router.get("/versions")
def graph_document_versions(
    request: Request,
    persona_slug: str = Query(...),
    brand_slug: Optional[str] = Query(None),
):
    _assert_persona_view(request, persona_slug)
    rows = supabase_client.list_system_events(
        entity_type="graph_document",
        event_types=["graph_version_committed", "graph_version_activated", "graph_document_published"],
        payload_equals={
            "persona_slug": persona_slug,
            **({"brand_slug": brand_slug} if brand_slug is not None else {}),
        },
        limit=500,
    )
    by_version: dict[int, dict] = {}
    for row in rows:
        payload = row.get("payload") or {}
        if payload.get("persona_slug") != persona_slug:
            continue
        if brand_slug is not None and (payload.get("brand_slug") or None) != brand_slug:
            continue
        version = int(payload.get("version") or 0)
        candidate = {
                "id": row.get("entity_id"),
                "version": version,
                "checksum": payload.get("checksum"),
                "published_at": row.get("created_at"),
                "source": payload.get("source"),
                "note": payload.get("note"),
                "status": "published" if row.get("event_type") in {"graph_version_activated", "graph_document_published"} else "committed",
            }
        current = by_version.get(version)
        if current is None or (candidate["status"] == "published" and current["status"] != "published"):
            by_version[version] = candidate
    out = sorted(by_version.values(), key=lambda item: item.get("version") or 0, reverse=True)
    return {"persona_slug": persona_slug, "brand_slug": brand_slug, "versions": out}


@router.get("/{version}/diff")
def graph_document_diff(
    version: int,
    request: Request,
    persona_slug: str = Query(...),
    against: int = Query(..., ge=1),
    brand_slug: Optional[str] = Query(None),
):
    _assert_persona_view(request, persona_slug)
    left = graph_json_v2_store.load_version(persona_slug, against, brand_slug)
    right = graph_json_v2_store.load_version(persona_slug, version, brand_slug)
    if left is None or right is None:
        raise HTTPException(404, "Graph version not found")
    left_nodes = {node.id: node.model_dump(mode="json") for node in left.nodes}
    right_nodes = {node.id: node.model_dump(mode="json") for node in right.nodes}
    left_edges = {edge.id: edge.model_dump(mode="json") for edge in left.edges}
    right_edges = {edge.id: edge.model_dump(mode="json") for edge in right.edges}
    return {
        "against": against,
        "version": version,
        "nodes": {
            "added": sorted(right_nodes.keys() - left_nodes.keys()),
            "removed": sorted(left_nodes.keys() - right_nodes.keys()),
            "changed": sorted(key for key in left_nodes.keys() & right_nodes.keys() if left_nodes[key] != right_nodes[key]),
        },
        "edges": {
            "added": sorted(right_edges.keys() - left_edges.keys()),
            "removed": sorted(left_edges.keys() - right_edges.keys()),
            "changed": sorted(key for key in left_edges.keys() & right_edges.keys() if left_edges[key] != right_edges[key]),
        },
    }


@router.post("/validate")
def graph_document_validate(body: ValidateGraphDocumentBody, request: Request):
    _assert_persona_edit(request, body.persona_slug)
    graph = _validate_graph_json_or_422(body.graph_json)
    _assert_graph_persona_matches(body.persona_slug, graph)
    try:
        canonical = graph_document_publisher.graph_markdown.canonicalize_graph(graph)
    except Exception as exc:
        errors = getattr(exc, "errors", [str(exc)])
        raise HTTPException(422, {"code": "GRAPH_MARKDOWN_INVALID", "errors": errors}) from exc
    policy_preview = graph_document_publisher.graph_action_policy.evaluate(canonical) if canonical.schema_version == "2.1" else []
    return {
        "ok": True,
        "schema_version": canonical.schema_version,
        "checksum": graph_json_v2_store.checksum_graph(canonical),
        "graph_json": canonical.model_dump(mode="json"),
        "policy_preview": policy_preview,
    }


@router.post("/commit")
def graph_document_commit(body: CommitGraphDocumentBody, request: Request):
    _assert_persona_edit(request, body.persona_slug)
    if body.graph_json is not None:
        try:
            graph = GraphJson.model_validate(body.graph_json)
        except Exception as exc:
            raise HTTPException(422, f"Invalid graph_json payload: {exc}") from exc
        if graph.schema_version == "2.0":
            graph = graph_json_v21_adapter.upgrade_to_v21(graph)
    else:
        current = graph_json_v2_store.load_current(body.persona_slug, body.brand_slug)
        if current is None:
            raise HTTPException(409, "First v2.1 commit requires graph_json")
        graph = graph_json_v21_adapter.upgrade_to_v21(current[1])
    _assert_graph_persona_matches(body.persona_slug, graph)
    if body.operations:
        try:
            graph = graph_document_publisher.apply_operations(graph, body.operations)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    is_valid, errors = graph_json_v2_validator.validate_graph_json(graph)
    if not is_valid:
        raise HTTPException(422, {"code": "GRAPH_VALIDATION_FAILED", "errors": errors})
    return _commit_or_http(
        graph=graph,
        persona_slug=body.persona_slug,
        brand_slug=body.brand_slug if body.brand_slug is not None else graph.brand_slug,
        source=body.source,
        reason=body.reason,
        published_by=(auth_service.current_user(request) or {}).get("id"),
        expected_version=body.expected_version,
        idempotency_key=body.idempotency_key,
    )


@router.get("/current/actions")
def graph_document_actions(
    request: Request,
    persona_slug: str = Query(...),
    brand_slug: Optional[str] = Query(None),
):
    _assert_persona_view(request, persona_slug)
    current = graph_json_v2_store.load_current(persona_slug, brand_slug)
    if current is None:
        return {"persona_slug": persona_slug, "version": 0, "actions": []}
    version, graph = current
    upgraded = graph_json_v21_adapter.upgrade_to_v21(graph)
    return {
        "persona_slug": persona_slug,
        "version": version,
        "actions": [
            node.model_dump(mode="json") for node in upgraded.nodes if node.node_class == "action"
        ],
    }


@router.get("/{version}/subgraph")
def graph_document_subgraph(
    version: int,
    request: Request,
    persona_slug: str = Query(...),
    action_node_id: str = Query(...),
    brand_slug: Optional[str] = Query(None),
):
    _assert_persona_view(request, persona_slug)
    loaded = graph_json_v2_store.load_version(persona_slug, version, brand_slug)
    if loaded is None:
        raise HTTPException(404, "Graph version not found")
    graph = graph_json_v21_adapter.upgrade_to_v21(loaded)
    action = next((node for node in graph.nodes if node.id == action_node_id and node.node_class == "action"), None)
    if action is None:
        raise HTTPException(404, "Action node not found")
    grants = [
        edge for edge in graph.edges
        if edge.target == action_node_id
        and edge.relation_type == "publishes_to"
        and edge.lifecycle.status == "active"
    ]
    source_ids = {edge.source for edge in grants}
    return {
        "persona_slug": persona_slug,
        "graph_version": version,
        "action": action.model_dump(mode="json"),
        "nodes": [node.model_dump(mode="json") for node in graph.nodes if node.id in source_ids],
        "edges": [edge.model_dump(mode="json") for edge in grants],
    }


@router.post("/publish")
def graph_document_publish(body: PublishGraphDocumentBody, request: Request):
    _assert_persona_edit(request, body.persona_slug)
    graph = _validate_graph_json_or_422(body.graph_json)
    _assert_graph_persona_matches(body.persona_slug, graph)
    document_brand = body.brand_slug if body.brand_slug is not None else graph.brand_slug
    current_event = _latest_event(body.persona_slug, document_brand)
    current_version = int(((current_event or {}).get("payload") or {}).get("version") or 0)
    return _publish_or_http(
        graph=graph,
        persona_slug=body.persona_slug,
        brand_slug=document_brand,
        source=f"graph_documents.publish:{body.source}",
        note=body.note,
        published_by=(auth_service.current_user(request) or {}).get("id"),
        expected_version=body.expected_version,
        idempotency_key=body.idempotency_key,
        current_version_override=current_version,
    )


@router.post("/apply-patch")
def graph_document_apply_patch(body: ApplyPatchBody, request: Request):
    _assert_persona_edit(request, body.persona_slug)
    graph = _validate_graph_json_or_422(body.graph_json)
    _assert_graph_persona_matches(body.persona_slug, graph)
    return _publish_or_http(
        graph=graph,
        persona_slug=body.persona_slug,
        brand_slug=graph.brand_slug,
        source=f"graph_documents.apply_patch:{body.source}",
        note=body.note,
        published_by=(auth_service.current_user(request) or {}).get("id"),
        expected_version=body.expected_version,
        idempotency_key=body.idempotency_key,
    )


@router.post("/import-json")
def graph_document_import_json(body: ImportGraphDocumentBody, request: Request):
    _assert_persona_edit(request, body.persona_slug)
    graph = _validate_graph_json_or_422(body.graph_json)
    _assert_graph_persona_matches(body.persona_slug, graph)
    return _publish_or_http(
        graph=graph,
        persona_slug=body.persona_slug,
        brand_slug=graph.brand_slug,
        source=body.source,
        note=body.note,
        published_by=(auth_service.current_user(request) or {}).get("id"),
        expected_version=body.expected_version,
        idempotency_key=body.idempotency_key,
        session_id=body.session_id,
    )


@router.post("/rollback")
def graph_document_rollback(body: RollbackBody, request: Request):
    _assert_persona_edit(request, body.persona_slug)
    target = graph_json_v2_store.load_version(body.persona_slug, body.version)
    if target is None:
        raise HTTPException(404, "Requested rollback version not found")
    result = _publish_or_http(
        graph=target,
        persona_slug=body.persona_slug,
        brand_slug=target.brand_slug,
        source=f"graph_documents.rollback:{body.persona_slug}:v{body.version}",
        note=body.note,
        published_by=(auth_service.current_user(request) or {}).get("id"),
        expected_version=body.expected_version,
        idempotency_key=body.idempotency_key,
    )
    return {**result, "rolled_back_from": body.version, "new_version": result["version"]}


@router.post("/sync")
def graph_document_sync(body: SyncBody, request: Request):
    _assert_persona_edit(request, body.persona_slug)
    try:
        return graph_document_publisher.sync(
            persona_slug=body.persona_slug,
            brand_slug=body.brand_slug,
            source=f"graph_documents.sync:{body.persona_slug}",
            idempotency_key=body.idempotency_key,
        )
    except graph_document_publisher.GraphValidationError as exc:
        raise HTTPException(422, {"code": "GRAPH_VALIDATION_FAILED", "errors": exc.errors}) from exc
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, str(exc)) from exc


@router.post("/reindex", deprecated=True)
def graph_document_reindex(body: ReindexBody, request: Request):
    result = graph_document_sync(
        SyncBody(
            persona_slug=body.persona_slug,
            brand_slug=body.brand_slug,
            idempotency_key=body.idempotency_key,
        ),
        request,
    )
    projections = result.get("projections") or {}
    return {
        **result,
        "indexed_nodes": projections.get("nodes_imported"),
        "indexed_edges": projections.get("edges_imported"),
        "reindex_ok": True,
        "deprecation": "Use POST /graph-documents/sync.",
        "note": body.note,
    }


@router.get("/events")
def graph_document_events(
    request: Request,
    persona_slug: str = Query(...),
    brand_slug: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
):
    _assert_persona_view(request, persona_slug)
    rows = supabase_client.list_system_events(
        entity_type="graph_document",
        event_types=[
            "graph_document_publish_attempted",
            "graph_document_published",
            "graph_document_publish_failed",
            "graph_document_synced",
        ],
        payload_equals={
            "persona_slug": persona_slug,
            **({"brand_slug": brand_slug} if brand_slug is not None else {}),
        },
        limit=limit,
    )
    out: list[dict] = []
    for row in rows:
        payload = row.get("payload") or {}
        if payload.get("persona_slug") != persona_slug:
            continue
        if brand_slug is not None and (payload.get("brand_slug") or None) != brand_slug:
            continue
        out.append(
            {
                "id": row.get("id"),
                "event_type": row.get("event_type"),
                "entity_id": row.get("entity_id"),
                "payload": payload,
                "created_at": row.get("created_at"),
            }
        )
    out.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return {"persona_slug": persona_slug, "brand_slug": brand_slug, "events": out}


# Keep the single-segment dynamic route last; Starlette routes in declaration
# order and otherwise `/publish`, `/commit`, etc. would be parsed as versions.
@router.get("/{version}")
def graph_document_version(
    version: int,
    request: Request,
    persona_slug: str = Query(...),
    brand_slug: Optional[str] = Query(None),
):
    _assert_persona_view(request, persona_slug)
    graph = graph_json_v2_store.load_version(persona_slug, version, brand_slug)
    if graph is None:
        raise HTTPException(404, "Graph version not found")
    return {
        "persona_slug": persona_slug,
        "brand_slug": brand_slug,
        "version": version,
        "checksum": graph.content_checksum or graph_json_v2_store.checksum_graph(graph),
        "graph_json": graph.model_dump(mode="json"),
    }

