import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from services import auth_service, supabase_client

router = APIRouter(prefix="/audiences", tags=["audiences"])
logger = logging.getLogger("audiences")


class AudienceCreateBody(BaseModel):
    persona_id: str
    name: str
    slug: str | None = None
    description: str | None = None
    source_type: str = "manual"
    metadata: dict = Field(default_factory=dict)


class AudienceUpdateBody(BaseModel):
    name: str | None = None
    slug: str | None = None
    description: str | None = None
    metadata: dict | None = None


@router.get("")
def list_audiences(request: Request, persona_id: str = Query(...)):
    """Lista as audiences operacionais de uma persona para a aba Leads.

    Retorna as audiences reais da persona (reconciliadas com as audiences
    criadas no Graph/Sofia) menos o bucket interno `import`, que e operacional
    e nao deve aparecer como audience semantica/filtro. Sync de graph nodes e
    best-effort e nunca bloqueia a resposta; `sync_audience_node` ja ignora
    `import` e audiences de origem `graph` para evitar duplicacao no grafo."""
    auth_service.assert_persona_access(request, persona_id=persona_id)
    try:
        user = auth_service.current_user(request)
        supabase_client.ensure_system_audiences_for_persona(
            persona_id, created_by_user_id=user.get("id") if user else None
        )
    except Exception as exc:
        logger.warning("ensure_system_audiences_for_persona failed: %s", exc)
    try:
        rows = supabase_client.list_persona_audiences(persona_id) or []
    except Exception as exc:
        logger.exception("list_persona_audiences failed: %s", exc)
        return []
    for row in rows:
        try:
            supabase_client.sync_audience_node(row)
        except Exception as exc:
            logger.warning("sync_audience_node failed for %s: %s", row.get("id"), exc)
    return rows


@router.post("")
def create_audience(body: AudienceCreateBody, request: Request):
    auth_service.assert_persona_capability(request, "edit", persona_id=body.persona_id)
    if body.source_type not in {"manual", "crm", "shared"}:
        raise HTTPException(422, "A API publica cria apenas grupos semanticos.")
    if body.metadata.get("kind") not in {None, "semantic_group"}:
        raise HTTPException(422, "O tipo da audience nao pode ser alterado pela API.")
    user = auth_service.current_user(request)
    audience = supabase_client.create_audience({
        **body.model_dump(),
        "metadata": {**body.metadata, "kind": "semantic_group"},
        "created_by_user_id": user.get("id"),
    })
    if not audience:
        raise HTTPException(502, "Nao foi possivel criar a audiencia.")
    audience = supabase_client.update_audience(
        audience["id"],
        {"updated_at": datetime.now(timezone.utc).isoformat()},
    ) or audience
    node = supabase_client.sync_audience_node(audience)
    supabase_client.insert_event(
        {
            "event_type": "audience_created",
            "entity_type": "audience",
            "entity_id": audience["id"],
            "persona_id": audience["persona_id"],
            "payload": {
                "audience_id": audience["id"],
                "slug": audience.get("slug"),
                "name": audience.get("name"),
                "graph_node_id": (node or {}).get("id"),
                "created_by_user_id": user.get("id"),
            },
        },
        source="audiences.create",
    )
    return {"ok": True, "audience": audience, "graph_node": node}


@router.patch("/{audience_id}")
def rename_audience(audience_id: str, body: AudienceUpdateBody, request: Request):
    audience = supabase_client.get_audience(audience_id)
    if not audience:
        raise HTTPException(404, "Audience not found")
    auth_service.assert_persona_capability(request, "edit", persona_id=audience.get("persona_id"))
    if body.metadata is not None:
        current_kind = str((audience.get("metadata") or {}).get("kind") or "semantic_group")
        requested_kind = str(body.metadata.get("kind") or current_kind)
        if requested_kind != current_kind:
            raise HTTPException(422, "O tipo da audience e imutavel.")
    changes = body.model_dump(exclude_none=True)
    if body.metadata is not None:
        changes["metadata"] = {**(audience.get("metadata") or {}), **body.metadata}
    updated = supabase_client.update_audience(
        audience_id,
        {
            **changes,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    ) or {**audience, **changes}
    node = supabase_client.sync_audience_node(updated)
    supabase_client.insert_event(
        {
            "event_type": "audience_updated",
            "entity_type": "audience",
            "entity_id": audience_id,
            "persona_id": updated.get("persona_id"),
            "payload": {
                "audience_id": audience_id,
                "slug": updated.get("slug"),
                "name": updated.get("name"),
                "graph_node_id": (node or {}).get("id"),
            },
        },
        source="audiences.update",
    )
    return {"ok": True, "audience": updated, "graph_node": node}


@router.get("/{audience_id}/leads")
def audience_leads(audience_id: str, request: Request, limit: int = Query(1000, le=2000), offset: int = 0):
    audience = supabase_client.get_audience(audience_id)
    if not audience:
        raise HTTPException(404, "Audience not found")
    auth_service.assert_persona_access(request, persona_id=audience.get("persona_id"))
    rows = supabase_client.get_leads_for_audience_scope(
        persona_id=audience["persona_id"],
        audience_id=audience["id"],
        limit=limit,
        offset=offset,
    )
    return {"audience": audience, "leads": rows}

