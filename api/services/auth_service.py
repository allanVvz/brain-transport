import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any, Optional

from fastapi import HTTPException, Request, Response

from services import supabase_client
from utils.env import get_auth_secret, is_production_runtime, is_strong_auth_secret

SESSION_COOKIE = "ai_brain_session"
HASH_ALGORITHM = "pbkdf2_sha256"
HASH_ITERATIONS = 390000
SESSION_TTL_SECONDS = 12 * 60 * 60
REMEMBER_TTL_SECONDS = 30 * 24 * 60 * 60


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def _auth_secret() -> bytes:
    secret = get_auth_secret()
    if not secret:
        if is_production_runtime():
            raise RuntimeError("AI_BRAIN_AUTH_SECRET is required in production.")
        secret = "dev-only-ai-brain-auth-secret-change-me"
    if is_production_runtime() and not is_strong_auth_secret(secret):
        raise RuntimeError("AI_BRAIN_AUTH_SECRET must contain at least 32 random characters.")
    return secret.encode("utf-8")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, HASH_ITERATIONS)
    return f"{HASH_ALGORITHM}${HASH_ITERATIONS}${_b64encode(salt)}${_b64encode(digest)}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iterations_raw, salt_raw, digest_raw = password_hash.split("$", 3)
        if algorithm != HASH_ALGORITHM:
            return False
        iterations = int(iterations_raw)
        salt = _b64decode(salt_raw)
        expected = _b64decode(digest_raw)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def _safe_user(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "email": row.get("email"),
        "username": row.get("username"),
        "name": row.get("name"),
        "role": row.get("role") or "user",
        "account_type": row.get("account_type") or "internal",
        "must_change_password": bool(row.get("must_change_password", False)),
        "is_active": bool(row.get("is_active", True)),
    }


def _execute_auth_query(query, *, operation: str):
    try:
        return supabase_client._execute_with_retry(query)
    except Exception as exc:
        try:
            from services import sre_logger
            sre_logger.error("auth_service", f"{operation} failed: {exc}", exc)
        except Exception:
            pass
        raise HTTPException(status_code=503, detail="Auth backend unavailable.")


def get_user_by_identifier(identifier: str) -> Optional[dict[str, Any]]:
    ident = (identifier or "").strip().lower()
    if not ident:
        return None
    client = supabase_client.get_client()
    fields = "id,email,username,password_hash,name,role,account_type,must_change_password,is_active"
    result = _execute_auth_query(
        client.table("app_users").select(fields).eq("email", ident).limit(1),
        operation="lookup app_users by email",
    )
    rows = result.data or []
    if rows:
        return rows[0]
    result = _execute_auth_query(
        client.table("app_users").select(fields).eq("username", ident).limit(1),
        operation="lookup app_users by username",
    )
    rows = result.data or []
    return rows[0] if rows else None


def get_user_by_id(user_id: str) -> Optional[dict[str, Any]]:
    result = _execute_auth_query(
        supabase_client.get_client()
        .table("app_users")
        .select("id,email,username,name,role,account_type,must_change_password,is_active")
        .eq("id", user_id)
        .maybe_single(),
        operation="lookup app_users by id",
    )
    return result.data


def get_user_access(user_id: str) -> list[dict[str, Any]]:
    result = _execute_auth_query(
        supabase_client.get_client()
        .table("user_persona_access")
        .select("id,user_id,client_id,persona_id,persona_slug,can_view,can_edit,can_manage")
        .eq("user_id", user_id)
        .eq("can_view", True),
        operation="lookup user_persona_access",
    )
    return result.data or []


def get_auth_personas() -> list[dict[str, Any]]:
    try:
        return [
            {
                "id": row.get("id"),
                "slug": row.get("slug"),
                "name": row.get("name"),
                "active": bool(row.get("active", True)),
            }
            for row in (supabase_client.get_personas() or [])
        ]
    except Exception as exc:
        try:
            from services import sre_logger
            sre_logger.error("auth_service", f"lookup personas during login failed: {exc}", exc)
        except Exception:
            pass
        raise HTTPException(status_code=503, detail="Auth persona catalog unavailable.")


def get_session_payload(token: str) -> Optional[dict[str, Any]]:
    try:
        payload_raw, signature = token.split(".", 1)
        expected = _b64encode(hmac.new(_auth_secret(), payload_raw.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            return None
        payload = json.loads(_b64decode(payload_raw).decode("utf-8"))
        if int(payload.get("exp") or 0) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


def create_session_token(user: dict[str, Any], remember: bool = False) -> tuple[str, int]:
    now = int(time.time())
    ttl = REMEMBER_TTL_SECONDS if remember else SESSION_TTL_SECONDS
    payload = {
        "sub": user["id"],
        "email": user.get("email"),
        "role": user.get("role") or "user",
        "iat": now,
        "exp": now + ttl,
    }
    payload_raw = _b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = _b64encode(hmac.new(_auth_secret(), payload_raw.encode("ascii"), hashlib.sha256).digest())
    return f"{payload_raw}.{signature}", ttl


def _cookie_secure() -> bool:
    secure = (os.environ.get("AI_BRAIN_COOKIE_SECURE") or "").strip().lower() in {"1", "true", "yes"}
    return secure or is_production_runtime()


def set_session_cookie(
    response: Response,
    token: str,
    ttl: int,
    *,
    remember: bool = False,
) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        # Without "remember me", keep a browser-session cookie. The signed
        # token still expires server-side after SESSION_TTL_SECONDS.
        max_age=ttl if remember else None,
        httponly=True,
        secure=_cookie_secure(),
        samesite="lax",
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        SESSION_COOKIE,
        path="/",
        httponly=True,
        secure=_cookie_secure(),
        samesite="lax",
    )


def authenticate(identifier: str, password: str) -> dict[str, Any]:
    row = get_user_by_identifier(identifier)
    if not row or not verify_password(password, row.get("password_hash") or ""):
        raise HTTPException(status_code=401, detail="Email/usuario ou senha invalidos.")
    if not row.get("is_active", True):
        raise HTTPException(status_code=403, detail="Usuario inativo. Fale com um administrador.")
    return _safe_user(row)


def is_admin(user: Optional[dict[str, Any]]) -> bool:
    return bool(
        user
        and user.get("role") == "admin"
        and (user.get("account_type") or "internal") == "internal"
    )


def current_user(request: Request) -> dict[str, Any]:
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Sessao obrigatoria.")
    return user


def allowed_access(request: Request) -> list[dict[str, Any]]:
    return list(getattr(request.state, "persona_access", []) or [])


def allowed_persona_ids(request: Request) -> list[str]:
    if is_admin(getattr(request.state, "user", None)):
        return []
    return [row["persona_id"] for row in allowed_access(request) if row.get("persona_id")]


def assert_persona_access(request: Request, persona_id: Optional[str] = None, persona_slug: Optional[str] = None) -> None:
    assert_persona_capability(
        request,
        "view",
        persona_id=persona_id,
        persona_slug=persona_slug,
    )


_CAPABILITY_RANK = {"view": 1, "edit": 2, "manage": 3}
_ROLE_CEILING = {"viewer": 1, "operator": 2, "user": 3, "admin": 3}


def _effective_capabilities(user: dict[str, Any], access: dict[str, Any]) -> dict[str, bool]:
    ceiling = _ROLE_CEILING.get(str(user.get("role") or "viewer"), 1)
    view = bool(access.get("can_view")) and ceiling >= 1
    edit = view and bool(access.get("can_edit")) and ceiling >= 2
    manage = edit and bool(access.get("can_manage")) and ceiling >= 3
    return {
        "view": view,
        "edit": edit,
        "manage": manage,
        "manage_members": bool(
            is_admin(user)
            or (
                (user.get("account_type") or "internal") == "agency"
                and manage
            )
        ),
    }


def assert_persona_capability(
    request: Request,
    capability: str,
    *,
    persona_id: Optional[str] = None,
    persona_slug: Optional[str] = None,
) -> dict[str, Any]:
    if capability not in _CAPABILITY_RANK:
        raise ValueError(f"Unknown persona capability: {capability}")
    user = current_user(request)
    if is_admin(user):
        return {
            "persona_id": persona_id,
            "persona_slug": persona_slug,
            "capabilities": {
                "view": True, "edit": True, "manage": True, "manage_members": True,
            },
        }
    if not persona_id and not persona_slug:
        raise HTTPException(status_code=400, detail="Selecione uma persona.")
    access = allowed_access(request)
    if not access:
        raise HTTPException(status_code=403, detail="Nenhuma persona foi atribuida a este usuario.")
    for row in access:
        matches_id = bool(persona_id and row.get("persona_id") == persona_id)
        matches_slug = bool(persona_slug and row.get("persona_slug") == persona_slug)
        if not (matches_id or matches_slug):
            continue
        capabilities = _effective_capabilities(user, row)
        if capabilities.get(capability):
            return {**row, "capabilities": capabilities}
    raise HTTPException(status_code=403, detail="Acesso negado para esta persona.")


def filter_personas_for_user(user: dict[str, Any], personas: list[dict[str, Any]], access: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if is_admin(user):
        return personas
    allowed = {row.get("persona_id") for row in access}
    allowed_slugs = {row.get("persona_slug") for row in access if row.get("persona_slug")}
    return [p for p in personas if p.get("id") in allowed or p.get("slug") in allowed_slugs]


def build_session_response(user: dict[str, Any]) -> dict[str, Any]:
    personas = get_auth_personas()
    access = [] if is_admin(user) else get_user_access(user["id"])
    visible_personas = filter_personas_for_user(user, personas, access)
    if not is_admin(user) and not visible_personas:
        raise HTTPException(status_code=403, detail="Nenhuma persona foi atribuida a este usuario.")
    projected_personas = []
    strongest = 3 if is_admin(user) else 0
    for persona in visible_personas:
        row = next(
            (
                item for item in access
                if item.get("persona_id") == persona.get("id")
                or item.get("persona_slug") == persona.get("slug")
            ),
            {},
        )
        capabilities = (
            {"view": True, "edit": True, "manage": True, "manage_members": True}
            if is_admin(user)
            else _effective_capabilities(user, row)
        )
        if capabilities["manage"]:
            strongest = max(strongest, 3)
        elif capabilities["edit"]:
            strongest = max(strongest, 2)
        elif capabilities["view"]:
            strongest = max(strongest, 1)
        projected_personas.append({**persona, "capabilities": capabilities})

    account_type = user.get("account_type") or "internal"
    if is_admin(user):
        access_profile = "brain_admin"
    else:
        suffix = "manager" if strongest >= 3 else "operator" if strongest >= 2 else "viewer"
        prefix = "agency" if account_type == "agency" else "client" if account_type == "client" else "brain"
        access_profile = f"{prefix}_{suffix}"

    first_persona_slug = next(
        (str(persona.get("slug")) for persona in projected_personas if persona.get("slug")),
        "",
    )
    if account_type == "client":
        surface = "client_portal"
        home_url = (
            f"/clientes/{first_persona_slug}/mensagens"
            if first_persona_slug
            else "/login"
        )
    elif account_type == "agency":
        surface = "agency"
        home_url = "/"
    elif is_admin(user):
        surface = "brain_admin"
        home_url = "/"
    else:
        surface = "internal"
        home_url = "/"

    return {
        "user": user,
        "account_type": account_type,
        "access_profile": access_profile,
        "navigation": {
            "surface": surface,
            "home_url": home_url,
        },
        "personas": projected_personas,
        "permissions": {
            "role": user.get("role"),
            "persona_access": access,
        },
    }


def change_password(user_id: str, current_password: str, new_password: str) -> None:
    if len(new_password or "") < 12:
        raise HTTPException(status_code=400, detail="A nova senha deve ter pelo menos 12 caracteres.")
    result = _execute_auth_query(
        supabase_client.get_client()
        .table("app_users")
        .select("id,password_hash")
        .eq("id", user_id)
        .maybe_single(),
        operation="lookup app_users for password change",
    )
    row = result.data or {}
    if not row or not verify_password(current_password, row.get("password_hash") or ""):
        raise HTTPException(status_code=400, detail="Senha atual invalida.")
    if verify_password(new_password, row.get("password_hash") or ""):
        raise HTTPException(status_code=400, detail="A nova senha deve ser diferente da atual.")
    _execute_auth_query(
        supabase_client.get_client()
        .table("app_users")
        .update({
            "password_hash": hash_password(new_password),
            "must_change_password": False,
        })
        .eq("id", user_id),
        operation="update app_users password",
    )

