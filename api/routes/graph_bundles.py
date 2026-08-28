from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request

from services import auth_service, graph_bundle_view


router = APIRouter(prefix="/graph-bundles", tags=["graph-bundles"])


def _assert_persona_view(request: Request, persona_slug: str) -> None:
    auth_service.assert_persona_access(request, persona_slug=persona_slug)


@router.get("/versions")
def graph_bundle_versions(
    request: Request,
    persona_slug: str = Query(..., min_length=1),
):
    _assert_persona_view(request, persona_slug)
    try:
        return graph_bundle_view.list_versions(persona_slug)
    except graph_bundle_view.GraphBundleViewNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/view")
def graph_bundle_view_get(
    request: Request,
    persona_slug: str = Query(..., min_length=1),
    source: Literal["draft", "publication"] = Query(...),
    ref: str = Query(..., min_length=1),
):
    _assert_persona_view(request, persona_slug)
    try:
        return graph_bundle_view.get_view(persona_slug, source=source, ref=ref)
    except graph_bundle_view.GraphBundleViewNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

