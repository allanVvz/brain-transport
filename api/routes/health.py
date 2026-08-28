import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from services import supabase_client

router = APIRouter(tags=["health"])
SERVICE_NAME = "brain-transport"
DEFAULT_REQUIRED_SCHEMA_VERSION = 130


def _build_metadata() -> dict:
    return {
        "source_sha": (os.environ.get("SOURCE_SHA") or "0" * 40).strip(),
        "build_digest": (os.environ.get("BUILD_DIGEST") or "unknown").strip(),
        "contracts_version": (os.environ.get("BRAIN_CONTRACTS_VERSION") or "1.0.0").strip(),
        "slot": (os.environ.get("BRAIN_SLOT") or "unknown").strip(),
    }


@router.get("/")
def root():
    return {"name": "Brain AI", "version": "1.0.0", "status": "ok", "docs": "/docs"}


@router.get("/health")
def health():
    return health_live()


@router.get("/health/live")
def health_live():
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        **_build_metadata(),
        "workers_embedded": (os.environ.get("RUN_EMBEDDED_WORKERS") or "").strip().lower() in {"1", "true", "yes", "on"},
    }


@router.get("/health/ready")
def health_ready():
    ok, detail = supabase_client.ping_supabase()
    schema_version = int(os.environ.get("CURRENT_SCHEMA_VERSION") or "0")
    required_schema_version = int(
        os.environ.get("REQUIRED_SCHEMA_VERSION") or DEFAULT_REQUIRED_SCHEMA_VERSION
    )
    schema_ok = schema_version >= required_schema_version
    payload = {
        "status": "ready" if ok and schema_ok else "not_ready",
        "service": SERVICE_NAME,
        **_build_metadata(),
        "schema_version": schema_version,
        "required_schema_version": required_schema_version,
        "checks": {
            "supabase": {
                "ok": ok,
                "detail": detail,
            },
            "schema": {"ok": schema_ok},
        },
    }
    if ok and schema_ok:
        return payload
    return JSONResponse(payload, status_code=503)


@router.get("/health/score")
def health_score():
    history = supabase_client.get_health_history(limit=1)
    latest = history[-1] if history else None
    return latest or {"score_total": 0, "message": "no snapshot yet"}


@router.get("/health/storage")
def health_storage():
    """Diagnostic for /assets/upload: which Supabase project the API actually
    talks to, and whether the required buckets are visible to it.

    Useful when an upload returns "Bucket not found" — this endpoint pins down
    if the API is hitting a different project than expected (env mix-up) or if
    the bucket truly is missing from the right project.
    """
    url = os.environ.get("SUPABASE_URL") or ""
    project_ref = url.replace("https://", "").split(".", 1)[0] if url else None
    bucket_names: list[str] = []
    bucket_error = None
    try:
        client = supabase_client.get_client()
        rows = client.storage.list_buckets() or []
        for b in rows:
            name = b.get("name") if isinstance(b, dict) else getattr(b, "name", None)
            if name:
                bucket_names.append(name)
    except Exception as exc:
        bucket_error = f"{type(exc).__name__}: {str(exc)[:200]}"
    required = ["assets-raw", "assets-derived"]
    missing = [b for b in required if b not in bucket_names]
    payload = {
        "supabase_url": url,
        "project_ref": project_ref,
        "buckets_visible": bucket_names,
        "required": required,
        "missing": missing,
        "ok": not missing and not bucket_error,
        "bucket_error": bucket_error,
    }
    status_code = 200 if payload["ok"] else 503
    return JSONResponse(payload, status_code=status_code)
