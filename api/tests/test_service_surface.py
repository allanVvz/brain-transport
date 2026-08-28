from __future__ import annotations

import base64
import ast
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("SUPABASE_OFFLINE", "true")
os.environ.setdefault("KNOWLEDGE_TAXONOMY_OFFLINE", "true")
os.environ.setdefault("CURRENT_SCHEMA_VERSION", "131")

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
ROOT = Path(__file__).resolve().parents[2]


def test_service_identity_and_readiness_surface():
    assert main.app.title == "Brain Transport"
    paths = set(main.app.openapi()["paths"])
    assert "/health" in paths
    assert "/health/ready" in paths
    assert "/internal/v1/transport/messages/campaign-outbound" in paths
    assert "/internal/v1/transport/whatsapp/evolution/provision" in paths
    assert "/internal/v1/transport/whatsapp/evolution/action" in paths
    assert not any(path.startswith("/messaging") for path in paths)
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


def test_transport_repository_contains_only_the_reviewed_production_surface():
    path = ROOT / "api/repositories/transport.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert len(functions) == 72
    assert {
        "claim_conversation_commit",
        "enqueue_wa_validator_session",
        "upsert_sofia_plan_session",
        "delete_knowledge_item",
        "create_audience",
    }.isdisjoint(functions)


def test_public_surface_excludes_other_domains():
    paths = set(main.app.openapi()["paths"])
    offenders = sorted(
        path
        for path in paths
        for prefix in FORBIDDEN_PREFIXES
        if path == prefix or path.startswith(prefix + "/")
    )
    assert offenders == []


def test_production_image_excludes_authoring_runtime_and_test_tools():
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    dockerfile = (ROOT / "api/Dockerfile").read_text(encoding="utf-8")
    for excluded in ("api/scripts/", "api/tests/", "tests/", "docs/"):
        assert excluded in dockerignore
    assert "docs/sdr" not in dockerfile
    assert "/data/vault" not in dockerfile


def test_other_domain_modules_are_not_shipped_in_transport_source():
    forbidden_directories = (
        "api/agents",
        "api/core",
    )
    forbidden_files = (
        "api/routes/messaging_campaigns.py",
        "api/services/campaigns_service.py",
        "api/services/asset_graph_contract.py",
        "api/services/conversation_graph.py",
        "api/services/graph_agent_runtime_v3.py",
        "api/services/graph_compiler_v3.py",
        "api/services/graph_json_v2_store.py",
        "api/services/graph_proof_checker_v3.py",
        "api/services/knowledge_graph.py",
        "api/services/knowledge_rag_intake.py",
        "api/services/lead_qualification.py",
        "api/services/conversation_runtime.py",
        "api/services/graph_bundle.py",
        "api/services/sofia_orchestrator.py",
        "api/services/wa_validator_service.py",
    )
    assert [
        path for path in forbidden_directories
        if any((ROOT / path).rglob("*.py"))
    ] == []
    assert [path for path in forbidden_files if (ROOT / path).exists()] == []


def test_internal_campaign_command_authenticates_and_uses_transport_outbox(monkeypatch):
    from routes import messages

    calls = []
    monkeypatch.setattr(messages.internal_auth, "authorize_webhook_token", calls.append)
    monkeypatch.setattr(
        messages.whatsapp_outbox,
        "enqueue_outbound",
        lambda **payload: {"buffer_id": "buffer-1", **payload},
    )
    body = messages.InternalCampaignOutboundBody(
        lead={"id": 42, "persona_id": "00000000-0000-0000-0000-000000000001"},
        text="mensagem",
        message_id="campaign:c:r:1",
        correlation_id="campaign:c:1:r:1",
        idempotency_key="campaign-send:c:1:r:1",
        campaign_scope={"campaign_id": "c", "campaign_recipient_id": "r"},
    )

    result = messages.enqueue_campaign_outbound_internal(body, "internal-token")

    assert calls == ["internal-token"]
    assert result["buffer_id"] == "buffer-1"
    assert result["campaign_scope"]["campaign_id"] == "c"


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
