from fastapi import FastAPI, Response
from fastapi.testclient import TestClient

from middleware.auth import auth_middleware
from routes import auth
from services import auth_service
from utils.env import validate_backend_env


def test_session_cookie_is_not_persistent_without_remember(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "qa")
    monkeypatch.setenv("AI_BRAIN_COOKIE_SECURE", "false")
    response = Response()

    auth_service.set_session_cookie(response, "signed-token", 3600, remember=False)

    cookie = response.headers["set-cookie"]
    assert "ai_brain_session=signed-token" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    assert "Max-Age" not in cookie


def test_remembered_session_cookie_is_persistent(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "qa")
    monkeypatch.setenv("AI_BRAIN_COOKIE_SECURE", "false")
    response = Response()

    auth_service.set_session_cookie(response, "signed-token", 3600, remember=True)

    cookie = response.headers["set-cookie"]
    assert "Max-Age=3600" in cookie


def test_production_cookie_is_always_secure(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("AI_BRAIN_COOKIE_SECURE", "false")
    response = Response()

    auth_service.set_session_cookie(response, "signed-token", 3600)

    assert "Secure" in response.headers["set-cookie"]


def test_strict_environment_requires_strong_auth_secret(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://db.example.test")
    monkeypatch.setenv("BRAIN_TRANSPORT_DB_KEY", "service-key")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://dashboard.example.test")
    monkeypatch.setenv("AI_BRAIN_AUTH_SECRET", "short")

    assert "AI_BRAIN_AUTH_SECRET (minimum 32 random characters)" in validate_backend_env(strict=True)

    monkeypatch.setenv("AI_BRAIN_AUTH_SECRET", "local-dev-auth-secret-change-me")
    assert "AI_BRAIN_AUTH_SECRET (minimum 32 random characters)" in validate_backend_env(strict=True)

    monkeypatch.setenv("AI_BRAIN_AUTH_SECRET", "a-strong-random-auth-secret-with-40-bytes")
    assert "AI_BRAIN_AUTH_SECRET (minimum 32 random characters)" not in validate_backend_env(strict=True)


def test_login_respects_remember(monkeypatch):
    user = {
        "id": "user-1",
        "email": "admin@example.test",
        "role": "admin",
        "account_type": "internal",
    }
    captured = {}
    monkeypatch.setattr(auth.auth_service, "authenticate", lambda *_args: user)
    monkeypatch.setattr(auth.auth_service, "build_session_response", lambda _user: {"user": user})
    monkeypatch.setattr(
        auth.auth_service,
        "create_session_token",
        lambda _user, remember: ("token", 3600),
    )
    monkeypatch.setattr(
        auth.auth_service,
        "set_session_cookie",
        lambda _response, _token, _ttl, *, remember: captured.update(remember=remember),
    )
    monkeypatch.setattr(auth.supabase_client, "get_client", lambda: (_ for _ in ()).throw(RuntimeError()))
    response = Response()

    result = auth.login(
        auth.LoginBody(identifier="admin@example.test", password="secret", remember=True),
        response,
    )

    assert result == {"user": user}
    assert captured == {"remember": True}


def test_auth_middleware_disables_cache_for_unauthorized_and_validation_responses():
    app = FastAPI()
    app.middleware("http")(auth_middleware)
    app.include_router(auth.router)
    client = TestClient(app)

    unauthorized = client.get("/auth/me")
    invalid = client.post("/auth/login", content="not-json", headers={"Content-Type": "application/json"})

    assert unauthorized.status_code == 401
    assert invalid.status_code == 422
    for response in (unauthorized, invalid):
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["pragma"] == "no-cache"

