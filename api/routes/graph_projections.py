from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from services import auth_service, graph_document_publisher, graph_json_v2_store, supabase_client


router = APIRouter(prefix="/graph-projections", tags=["graph-projections"])


def _rows(limit: int = 500) -> list[dict]:
    return supabase_client.list_system_events(
        entity_type="graph_projection",
        event_types=["graph_projection_published", "graph_projection_failed", "graph_projection_withdrawn"],
        limit=limit,
    )


@router.get("")
def list_graph_projections(
    request: Request,
    persona_slug: str = Query(...),
    graph_version: int | None = Query(None),
):
    auth_service.assert_persona_access(request, persona_slug=persona_slug)
    projections = []
    for row in _rows():
        payload = row.get("payload") or {}
        if payload.get("persona_slug") != persona_slug:
            continue
        if graph_version is not None and int(payload.get("graph_version") or 0) != graph_version:
            continue
        projections.append({**payload, "event_id": row.get("id")})
    return {"persona_slug": persona_slug, "projections": projections}


@router.get("/operations/{operation_id}")
def graph_projection_operation(operation_id: str, request: Request):
    rows = supabase_client.list_system_events(
        entity_type="graph_document",
        event_types=["graph_version_committed", "graph_version_activated", "graph_projection_failed"],
        limit=500,
    )
    matching = [row for row in rows if str((row.get("payload") or {}).get("operation_id") or row.get("id")) == operation_id]
    if not matching:
        raise HTTPException(404, "Projection operation not found")
    payload = matching[0].get("payload") or {}
    auth_service.assert_persona_access(request, persona_slug=payload.get("persona_slug"))
    status = "projection_failed" if any(row.get("event_type") == "graph_projection_failed" for row in matching) else (
        "published" if any(row.get("event_type") == "graph_version_activated" for row in matching) else "committed"
    )
    return {"operation_id": operation_id, "status": status, "events": matching}


class RetryProjectionBody(BaseModel):
    persona_slug: str
    brand_slug: str | None = None


@router.post("/{projection_id}/retry")
def retry_graph_projection(projection_id: str, body: RetryProjectionBody, request: Request):
    user = auth_service.current_user(request)
    if not auth_service.is_admin(user):
        raise HTTPException(403, "Admin access required")
    try:
        result = graph_document_publisher.sync(
            persona_slug=body.persona_slug,
            brand_slug=body.brand_slug,
            source=f"graph_projection.retry:{projection_id}",
            idempotency_key=f"projection-retry:{projection_id}",
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {**result, "projection_id": projection_id}


@router.get("/{projection_id}")
def get_graph_projection(projection_id: str, request: Request):
    row = next((item for item in _rows() if item.get("entity_id") == projection_id), None)
    if not row:
        raise HTTPException(404, "Projection not found")
    payload = row.get("payload") or {}
    auth_service.assert_persona_access(request, persona_slug=payload.get("persona_slug"))
    return {**payload, "event_id": row.get("id")}

