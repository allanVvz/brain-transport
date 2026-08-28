from fastapi import APIRouter, HTTPException, Query, Request
from services import auth_service, supabase_client

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.get("/status")
def pipeline_status(request: Request):
    if not auth_service.is_admin(auth_service.current_user(request)):
        raise HTTPException(403, "Apenas admin pode ver status global do pipeline.")
    return supabase_client.get_pipeline_statuses()


@router.get("/metrics")
def pipeline_metrics(request: Request, persona_id: str = Query(None)):
    if persona_id:
        auth_service.assert_persona_access(request, persona_id=persona_id)
        return supabase_client.get_pipeline_metrics(persona_id=persona_id)
    if not auth_service.is_admin(auth_service.current_user(request)):
        totals = {
            "pending_attention": 0,
            "approved_today": 0,
            "kb_entries": 0,
            "assets_pending": 0,
            "errors_24h": 0,
        }
        for pid in auth_service.allowed_persona_ids(request):
            metrics = supabase_client.get_pipeline_metrics(persona_id=pid)
            for key in totals:
                totals[key] += int(metrics.get(key) or 0)
        return totals
    return supabase_client.get_pipeline_metrics(persona_id=persona_id)


@router.get("/events")
def pipeline_events(
    request: Request,
    limit: int = 50,
    event_type: str = Query(None),
    persona_id: str = Query(None),
):
    if persona_id:
        auth_service.assert_persona_access(request, persona_id=persona_id)
        return supabase_client.get_events(limit=limit, event_type=event_type, persona_id=persona_id)
    if not auth_service.is_admin(auth_service.current_user(request)):
        rows = []
        for pid in auth_service.allowed_persona_ids(request):
            rows.extend(supabase_client.get_events(limit=limit, event_type=event_type, persona_id=pid))
        return sorted(rows, key=lambda row: str(row.get("created_at") or ""), reverse=True)[:limit]
    return supabase_client.get_events(limit=limit, event_type=event_type, persona_id=persona_id)

