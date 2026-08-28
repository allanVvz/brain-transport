import asyncio
import logging
import traceback
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request
from schemas.insight import InsightCreate, InsightUpdate
from services import auth_service, supabase_client

logger = logging.getLogger("insights")

router = APIRouter(prefix="/insights", tags=["insights"])


@router.get("")
def list_insights(request: Request, status: str = Query(None), limit: int = 50, persona_id: str = Query(None)):
    try:
        if persona_id:
            auth_service.assert_persona_access(request, persona_id=persona_id)
            return supabase_client.get_insights(status=status, limit=limit, persona_id=persona_id)
        if auth_service.is_admin(auth_service.current_user(request)):
            return supabase_client.get_insights(status=status, limit=limit)
        rows = []
        for pid in auth_service.allowed_persona_ids(request):
            rows.extend(supabase_client.get_insights(status=status, limit=limit, persona_id=pid))
        return sorted(rows, key=lambda row: str(row.get("created_at") or ""), reverse=True)[:limit]
    except Exception as exc:
        logger.error("list_insights failed (status=%r): %s\n%s", status, exc, traceback.format_exc())
        return []          # degrade gracefully â€” dashboard shows empty state instead of crashing


@router.patch("/{insight_id}")
def update_insight(insight_id: str, body: InsightUpdate, request: Request):
    try:
        if not auth_service.is_admin(auth_service.current_user(request)):
            raise HTTPException(403, "Apenas admin pode atualizar insights globais.")
        data: dict = {"status": body.status}
        if body.status == "resolved":
            data["resolved_at"] = datetime.now(timezone.utc).isoformat()
        supabase_client.update_insight(insight_id, data)
        return {"ok": True}
    except Exception as exc:
        logger.error("update_insight failed (id=%r): %s", insight_id, exc)
        raise HTTPException(500, str(exc))


@router.post("/run-validator")
async def trigger_validator(request: Request):
    try:
        if not auth_service.is_admin(auth_service.current_user(request)):
            raise HTTPException(403, "Apenas admin pode executar validador global.")
        from agents.flow_validator.orchestrator import run as run_validator
        result = await asyncio.to_thread(run_validator)
        return result
    except Exception as exc:
        logger.error("trigger_validator failed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(500, str(exc))

