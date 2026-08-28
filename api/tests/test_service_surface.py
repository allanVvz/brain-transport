from __future__ import annotations

import base64
import json
import os

import pytest

os.environ.setdefault("SUPABASE_OFFLINE", "true")
os.environ.setdefault("KNOWLEDGE_TAXONOMY_OFFLINE", "true")
os.environ.setdefault("CURRENT_SCHEMA_VERSION", "130")

import main
from repositories import transport as transport_repository
from services import supabase_client
from workers.runner import WORKERS


FORBIDDEN_PREFIXES = (
    "/agents",
    "/agent-harness",
    "/wa-validator",
    "/personas",
    "/knowledge",
    "/auth",
    "/leads",
    "/process",
)


def test_service_identity_and_readiness_surface():
    assert main.app.title == "Brain Transport"
    paths = set(main.app.openapi()["paths"])
    assert "/health" in paths
    assert "/health/ready" in paths
    assert "/internal/v1/transport/whatsapp/outbound-result" in paths
    assert "/internal/v1/transport/messages/send" in paths
    assert "/internal/whatsapp/outbound-result" not in paths


def test_worker_group_is_domain_scoped():
    assert set(WORKERS) == {
        "health_check",
        "media_ingest",
        "whatsapp_dispatch",
    }


def test_transport_dispatch_uses_runtime_service_boundary():
    from workers import whatsapp_dispatch_worker

    assert not hasattr(whatsapp_dispatch_worker, "conversation_runtime")
    assert hasattr(whatsapp_dispatch_worker, "runtime_client")


def test_legacy_database_module_is_transport_repository_alias():
    assert supabase_client is transport_repository


def test_public_surface_excludes_other_domains():
    paths = set(main.app.openapi()["paths"])
    offenders = sorted(
        path
        for path in paths
        for prefix in FORBIDDEN_PREFIXES
        if path == prefix or path.startswith(prefix + "/")
    )
    assert offenders == []


def _jwt_for_role(role: str) -> str:
    payload = base64.urlsafe_b64encode(
        json.dumps({"role": role}).encode()
    ).decode().rstrip("=")
    return f"header.{payload}.signature"


def test_database_jwt_is_restricted_to_manifest_role(monkeypatch):
    expected = _jwt_for_role("brain_transport")
    monkeypatch.setenv("BRAIN_DB_JWT", expected)
    assert supabase_client._validated_db_jwt() == expected

    monkeypatch.setenv("BRAIN_DB_JWT", _jwt_for_role("service_role"))
    with pytest.raises(RuntimeError, match="brain_transport"):
        supabase_client._validated_db_jwt()


def test_readiness_uses_image_schema_requirement_and_build_metadata(monkeypatch):
    from routes import health

    monkeypatch.setenv("CURRENT_SCHEMA_VERSION", "131")
    monkeypatch.setenv("REQUIRED_SCHEMA_VERSION", "131")
    monkeypatch.setenv("SOURCE_SHA", "a" * 40)
    monkeypatch.setenv("BUILD_DIGEST", "sha256:image")
    monkeypatch.setattr(health.supabase_client, "ping_supabase", lambda: (True, "ok"))

    result = health.health_ready()

    assert result["status"] == "ready"
    assert result["required_schema_version"] == 131
    assert result["source_sha"] == "a" * 40
    assert result["build_digest"] == "sha256:image"
