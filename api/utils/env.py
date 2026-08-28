from __future__ import annotations

import os
from typing import Any


_INSECURE_AUTH_SECRETS = {
    "dev-only-ai-brain-auth-secret-change-me",
    "local-dev-auth-secret-change-me",
    "replace-with-url-safe-random-value",
}


def _bool_env(name: str) -> bool:
    value = (os.environ.get(name) or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def is_production_runtime() -> bool:
    return (
        bool((os.environ.get("K_SERVICE") or "").strip())
        or bool((os.environ.get("CLOUD_RUN_JOB") or "").strip())
        or (os.environ.get("ENV", "").strip().lower() == "production")
        or (os.environ.get("PYTHON_ENV", "").strip().lower() == "production")
        or (os.environ.get("ENVIRONMENT", "").strip().lower() == "production")
    )


def get_auth_secret() -> str:
    primary = (os.environ.get("AI_BRAIN_AUTH_SECRET") or "").strip()
    fallback = (os.environ.get("NEXTAUTH_SECRET") or "").strip()
    return primary or fallback


def is_strong_auth_secret(secret: str) -> bool:
    normalized = (secret or "").strip()
    return (
        len(normalized.encode("utf-8")) >= 32
        and normalized.lower() not in _INSECURE_AUTH_SECRETS
        and not normalized.lower().startswith("replace_with_")
    )


def get_backend_env() -> dict[str, Any]:
    default_dev_origins = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    configured_origins = [
        origin.strip()
        for origin in os.environ.get("ALLOWED_ORIGINS", ",".join(default_dev_origins)).split(",")
        if origin.strip()
    ]
    if not is_production_runtime():
        for origin in default_dev_origins:
            if origin not in configured_origins:
                configured_origins.append(origin)
    allowed_origin_regex = (os.environ.get("ALLOWED_ORIGIN_REGEX") or "").strip() or None
    return {
        "allowed_origins": configured_origins,
        "allowed_origin_regex": allowed_origin_regex,
        "supabase_url": (os.environ.get("SUPABASE_URL") or "").strip(),
        "brain_db_jwt": (os.environ.get("BRAIN_DB_JWT") or "").strip(),
        "is_production": is_production_runtime(),
        "run_embedded_workers": _bool_env("RUN_EMBEDDED_WORKERS"),
    }


def validate_backend_env(strict: bool | None = None) -> list[str]:
    env = get_backend_env()
    if strict is None:
        strict = bool(env["is_production"])
    missing: list[str] = []
    if strict:
        if not env["supabase_url"]:
            missing.append("SUPABASE_URL")
        if not env["brain_db_jwt"]:
            missing.append("BRAIN_DB_JWT")
        if not env["allowed_origins"]:
            missing.append("ALLOWED_ORIGINS")
        if not is_strong_auth_secret(get_auth_secret()):
            missing.append("AI_BRAIN_AUTH_SECRET (minimum 32 random characters)")
    return missing
