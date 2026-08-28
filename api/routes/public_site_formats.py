from fastapi import APIRouter, Request

from services import auth_service, supabase_client

router = APIRouter(prefix="/api/public-site-formats", tags=["public-site"])


@router.get("")
def list_public_site_formats(request: Request):
    auth_service.current_user(request)
    return supabase_client.list_public_site_formats(enabled_only=True)

