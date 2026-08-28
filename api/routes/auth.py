from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from services import auth_service, supabase_client

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginBody(BaseModel):
    identifier: str = Field(min_length=1, max_length=320)
    password: str = Field(min_length=1, max_length=1024)
    remember: bool = False


class ChangePasswordBody(BaseModel):
    current_password: str = Field(min_length=1, max_length=1024)
    new_password: str = Field(min_length=12, max_length=1024)


@router.post("/login")
def login(body: LoginBody, response: Response):
    try:
        user = auth_service.authenticate(body.identifier, body.password)
        session_payload = auth_service.build_session_response(user)
        token, ttl = auth_service.create_session_token(user, remember=body.remember)
        auth_service.set_session_cookie(response, token, ttl, remember=body.remember)
    except HTTPException:
        raise
    except Exception as exc:
        try:
            from services import sre_logger
            sre_logger.error("auth.login", f"unexpected login failure: {exc}", exc)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="Unexpected login failure.")
    try:
        supabase_client.get_client().table("app_users").update({"last_login_at": datetime.now(timezone.utc).isoformat()}).eq("id", user["id"]).execute()
    except Exception:
        pass
    return session_payload


@router.get("/me")
def me(request: Request):
    user = auth_service.current_user(request)
    return auth_service.build_session_response(user)


@router.post("/logout")
def logout(response: Response):
    auth_service.clear_session_cookie(response)
    return {"ok": True}


@router.post("/change-password")
def change_password(body: ChangePasswordBody, request: Request):
    user = auth_service.current_user(request)
    auth_service.change_password(user["id"], body.current_password, body.new_password)
    return {"ok": True, "must_change_password": False}

