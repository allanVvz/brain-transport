import os
import re
import base64
import hashlib
import hmac
import json
import time

from fastapi import Request
from fastapi.responses import JSONResponse

from services import auth_service

PUBLIC_EXACT_PATHS = {
    "/health",
    "/health/live",
    "/health/ready",
    "/auth/login",
    "/auth/logout",
    "/process",
    "/webhooks/whatsapp",
    "/webhooks/whatsapp/inbound",
    "/webhooks/whatsapp/status",
    "/internal/whatsapp/outbound-result",
    "/internal/conversations/context",
    "/internal/conversations/decide",
    "/internal/conversations/commit",
    "/internal/conversations/fail-safe-handoff",
    "/internal/conversations/technical-failure",
    # Integration-authenticated equivalent of the operator conversion route.
    # The handler performs constant-time X-Webhook-Token validation.
    "/internal/agents/leads/{lead_ref}/purchase-completed",
    "/internal/agents/leads/{lead_ref}/journey-events",
}

ADMIN_TOKEN_HEADER = "x-ai-brain-admin-token"
AUTHORIZATION_HEADER = "authorization"
ADMIN_TOKEN_ENV_NAMES = ("QA", "qa", "preview", "PREVIEW", "test", "TEST")
INTERNAL_PRINCIPAL_HEADER = "x-brain-principal"
INTERNAL_SIGNATURE_HEADER = "x-brain-principal-signature"


def _internal_principal_user(request: Request) -> tuple[dict, list] | None:
    """Verify the gateway-issued short-lived principal.

    The edge gateway must remove both headers from the client request before
    issuing its own values. Services still enforce persona scope in-route/DB.
    """
    encoded = (request.headers.get(INTERNAL_PRINCIPAL_HEADER) or "").strip()
    signature = (request.headers.get(INTERNAL_SIGNATURE_HEADER) or "").strip()
    secret = (os.environ.get("BRAIN_INTERNAL_AUTH_SECRET") or "").encode("utf-8")
    if not encoded or not signature:
        return None
    if len(secret) < 32:
        return None
    expected = hmac.new(secret, encoded.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        claims = json.loads(raw)
        now = int(time.time())
        if claims.get("iss") != "brain-gateway" or int(claims.get("exp") or 0) <= now:
            return None
        if int(claims.get("iat") or 0) > now + 5:
            return None
        role = str(claims.get("role") or "")
        if role not in {"admin", "user", "operator", "viewer", "service"}:
            return None
        subject = str(claims.get("sub") or "")
        if not subject:
            return None
        persona_ids = [str(value) for value in claims.get("persona_ids") or []]
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    return ({
        "id": subject,
        "email": claims.get("email"),
        "username": claims.get("username"),
        "role": role,
        "is_active": True,
        "auth_method": "internal_principal",
    }, [{"persona_id": value} for value in persona_ids])


def _disable_auth_response_cache(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    return response


def _auth_error(detail: str, status_code: int) -> JSONResponse:
    return _disable_auth_response_cache(
        JSONResponse({"detail": detail}, status_code=status_code)
    )


def is_public_path(path: str) -> bool:
    if path.startswith("/webhooks/evolution/"):
        return True
    if path in PUBLIC_EXACT_PATHS:
        return True
    if path.startswith("/internal/agents/leads/"):
        for suffix in ("/purchase-completed", "/journey-events"):
            if path.endswith(suffix):
                return path.removeprefix("/internal/agents/leads/").removesuffix(
                    suffix
                ).strip("/").isdigit()
    # Only the public site contract is anonymous. Nested admin endpoints under
    # the same prefix must still pass through session/persona authorization.
    if path.startswith("/api/menu/"):
        remainder = path.removeprefix("/api/menu/").strip("/")
        return bool(remainder) and "/" not in remainder
    return False


def is_client_path_allowed(method: str, path: str) -> bool:
    """Client portal allowlist; asset media remains persona-scoped in-route."""
    if path in {"/auth/me", "/auth/logout", "/auth/change-password"}:
        return True
    if path.startswith("/portal/"):
        return True
    return bool(
        str(method or "").upper() == "GET"
        and re.fullmatch(r"/assets/[^/]+/media", path or "")
    )


def _admin_test_token_user(request: Request) -> dict | None:
    """When ENVIRONMENT is qa/preview, allow a shared admin token to act as
    the admin user. Production never accepts this path.

    The token must come from the env var AI_BRAIN_ADMIN_TEST_TOKEN and is
    compared in constant time. Accepted QA auth headers:
      - X-AI-BRAIN-ADMIN-TOKEN: <token>
      - Authorization: Bearer <token>  (compatibility alias)
    The token value itself is never logged.
    """
    env_name = (os.environ.get("ENVIRONMENT") or "").strip()
    if env_name not in ADMIN_TOKEN_ENV_NAMES:
        return None
    expected = (os.environ.get("AI_BRAIN_ADMIN_TEST_TOKEN") or "").strip()
    if not expected:
        return None
    presented = (request.headers.get(ADMIN_TOKEN_HEADER) or "").strip()
    if not presented:
        authz = (request.headers.get(AUTHORIZATION_HEADER) or "").strip()
        prefix = "bearer "
        if authz.lower().startswith(prefix):
            presented = authz[len(prefix):].strip()
    if not presented:
        return None
    # Constant-time compare to avoid timing leaks.
    import hmac

    if not hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8")):
        return None
    return {
        "id": "qa-admin-token",
        "email": "qa-admin@token.local",
        "username": "qa-admin",
        "name": "QA Admin (token)",
        "role": "admin",
        "is_active": True,
        "auth_method": "admin_test_token",
    }


async def auth_middleware(request: Request, call_next):
    path = request.url.path
    is_auth_path = path.startswith("/auth/")
    if request.method == "OPTIONS" or is_public_path(path):
        response = await call_next(request)
        return _disable_auth_response_cache(response) if is_auth_path else response

    internal = _internal_principal_user(request)
    if internal:
        request.state.user, request.state.persona_access = internal
        return await call_next(request)

    token_user = _admin_test_token_user(request)
    if token_user:
        request.state.user = token_user
        request.state.persona_access = []
        return await call_next(request)

    token = request.cookies.get(auth_service.SESSION_COOKIE)
    payload = auth_service.get_session_payload(token or "")
    if not payload:
        return _auth_error("Sessao obrigatoria.", 401)

    try:
        user = auth_service.get_user_by_id(payload.get("sub") or "")
    except Exception as exc:
        try:
            from services import sre_logger
            sre_logger.warn(
                "auth_middleware",
                f"session revalidation unavailable: {exc}",
                exc,
            )
        except Exception:
            pass
        return _auth_error("Auth backend unavailable.", 503)

    if not user or not user.get("is_active", True):
        return _auth_error("Sessao invalida.", 401)

    request.state.user = user
    if auth_service.is_admin(user):
        request.state.persona_access = []
    else:
        try:
            request.state.persona_access = auth_service.get_user_access(user["id"])
        except Exception as exc:
            try:
                from services import sre_logger
                sre_logger.warn(
                    "auth_middleware",
                    f"persona access unavailable, using empty scope: {exc}",
                    exc,
                )
            except Exception:
                pass
            request.state.persona_access = []

    account_type = user.get("account_type") or "internal"
    if account_type == "client":
        if not is_client_path_allowed(request.method, path):
            return JSONResponse({"detail": "Acesso negado."}, status_code=403)
    response = await call_next(request)
    return _disable_auth_response_cache(response) if is_auth_path else response
