from fastapi import APIRouter, Query, Request
from services import auth_service, supabase_client
from services.knowledge_service import sync_from_sheets
import asyncio

router = APIRouter(prefix="/kb", tags=["knowledge-base"])


@router.get("")
def list_kb(request: Request, persona_id: str = Query(None), status: str = "ATIVO"):
    if persona_id:
        auth_service.assert_persona_access(request, persona_id=persona_id)
        return supabase_client.get_kb_entries(persona_id=persona_id, status=status)
    if not auth_service.is_admin(auth_service.current_user(request)):
        return supabase_client.get_kb_entries_for_persona_ids(auth_service.allowed_persona_ids(request), status=status)
    return supabase_client.get_kb_entries(persona_id=persona_id, status=status)


@router.post("/sync")
async def sync_kb(persona_id: str, request: Request, spreadsheet_id: str = Query(None)):
    auth_service.assert_persona_access(request, persona_id=persona_id)
    count = await asyncio.to_thread(sync_from_sheets, persona_id, spreadsheet_id)
    return {"synced": count}

