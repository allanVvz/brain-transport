from __future__ import annotations

import secrets
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from services import auth_service, supabase_client

router = APIRouter(prefix="/access", tags=["access"])


class MemberCreateBody(BaseModel):
    email: str
    name: str | None = None
    role: str = "user"
    can_view: bool = True
    can_edit: bool = False
    can_manage: bool = False


class MemberPatchBody(BaseModel):
    can_view: bool | None = None
    can_edit: bool | None = None
    can_manage: bool | None = None


def _persona(slug: str) -> dict[str, Any]:
    persona = supabase_client.get_persona(slug)
    if not persona:
        raise HTTPException(404, "Persona nao encontrada.")
    return persona


def _assert_manage_members(request: Request, persona: dict[str, Any]) -> None:
    access = auth_service.assert_persona_capability(
        request, "manage", persona_id=persona["id"], persona_slug=persona["slug"]
    )
    user = auth_service.current_user(request)
    if not (auth_service.is_admin(user) or access.get("capabilities", {}).get("manage_members")):
        raise HTTPException(403, "Acesso negado.")


def _normalized_flags(view: bool, edit: bool, manage: bool) -> tuple[bool, bool, bool]:
    if manage:
        edit = True
    if edit:
        view = True
    return view, edit, manage


@router.get("/personas/{slug}/members")
def list_members(slug: str, request: Request):
    persona = _persona(slug)
    _assert_manage_members(request, persona)
    rows = (
        supabase_client.get_client()
        .table("user_persona_access")
        .select("id,user_id,can_view,can_edit,can_manage,app_users!inner(id,email,name,role,account_type,is_active)")
        .eq("persona_id", persona["id"])
        .execute()
        .data
        or []
    )
    return rows


@router.post("/personas/{slug}/members", status_code=201)
def create_member(slug: str, body: MemberCreateBody, request: Request):
    persona = _persona(slug)
    _assert_manage_members(request, persona)
    if body.role not in {"user", "operator", "viewer"}:
        raise HTTPException(400, "Papel invalido.")
    email = str(body.email).strip().lower()
    if "@" not in email or email.startswith("@") or email.endswith("@"):
        raise HTTPException(400, "Email invalido.")
    client = supabase_client.get_client()
    existing_result = (
        client.table("app_users")
        .select("*")
        .eq("email", email)
        .maybe_single()
        .execute()
    )
    existing = existing_result.data if existing_result else None
    temporary_password: str | None = None
    if existing:
        if (existing.get("account_type") or "internal") != "client":
            raise HTTPException(409, "Este email pertence a uma conta que nao pode ser vinculada.")
        user = existing
    else:
        temporary_password = secrets.token_urlsafe(18)
        created = client.table("app_users").insert({
            "email": email,
            "username": None,
            "name": body.name,
            "role": body.role,
            "account_type": "client",
            "must_change_password": True,
            "password_hash": auth_service.hash_password(temporary_password),
            "is_active": True,
        }).execute().data or []
        user = created[0] if created else None
        if not user:
            raise HTTPException(500, "Falha ao criar usuario.")

    can_view, can_edit, can_manage = _normalized_flags(
        body.can_view, body.can_edit, body.can_manage
    )
    association = client.table("user_persona_access").upsert({
        "user_id": user["id"],
        "client_id": persona["slug"],
        "persona_id": persona["id"],
        "persona_slug": persona["slug"],
        "can_view": can_view,
        "can_edit": can_edit,
        "can_manage": can_manage,
    }, on_conflict="user_id,persona_id").execute().data or []
    response = {
        "user": auth_service._safe_user(user),
        "access": association[0] if association else {},
        "portal_url": f"/clientes/{persona['slug']}/mensagens",
    }
    if temporary_password:
        response["temporary_password"] = temporary_password
    return response


@router.patch("/personas/{slug}/members/{user_id}")
def update_member(slug: str, user_id: str, body: MemberPatchBody, request: Request):
    persona = _persona(slug)
    _assert_manage_members(request, persona)
    client = supabase_client.get_client()
    current = (
        client.table("user_persona_access")
        .select("*")
        .eq("persona_id", persona["id"])
        .eq("user_id", user_id)
        .maybe_single().execute().data
    )
    if not current:
        raise HTTPException(404, "Membro nao encontrado.")
    view, edit, manage = _normalized_flags(
        current["can_view"] if body.can_view is None else body.can_view,
        current["can_edit"] if body.can_edit is None else body.can_edit,
        current["can_manage"] if body.can_manage is None else body.can_manage,
    )
    rows = client.table("user_persona_access").update({
        "can_view": view, "can_edit": edit, "can_manage": manage,
    }).eq("id", current["id"]).execute().data or []
    return rows[0] if rows else {}


@router.delete("/personas/{slug}/members/{user_id}")
def revoke_member(slug: str, user_id: str, request: Request):
    persona = _persona(slug)
    _assert_manage_members(request, persona)
    rows = (
        supabase_client.get_client().table("user_persona_access")
        .update({"can_view": False, "can_edit": False, "can_manage": False})
        .eq("persona_id", persona["id"]).eq("user_id", user_id)
        .execute().data or []
    )
    if not rows:
        raise HTTPException(404, "Membro nao encontrado.")
    return {"ok": True}

