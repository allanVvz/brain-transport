from __future__ import annotations

import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from services import graph_agent_runtime_v3, kb_intake_service


def _v3_binding() -> dict:
    return {"active": True, "metadata": {"runtime_version": graph_agent_runtime_v3.RUNTIME_VERSION}}


def _legacy_binding() -> dict:
    return {"active": True, "metadata": {"pipeline_contract": "conversation_v1"}}


def test_persona_uses_graph_bundle_pipeline_true_for_v3_binding(monkeypatch):
    monkeypatch.setattr(
        kb_intake_service.supabase_client,
        "get_workflow_bindings",
        lambda persona_id: [_v3_binding()],
    )
    assert kb_intake_service._persona_uses_graph_bundle_pipeline("any-persona-id") is True


def test_persona_uses_graph_bundle_pipeline_false_for_legacy_binding(monkeypatch):
    monkeypatch.setattr(
        kb_intake_service.supabase_client,
        "get_workflow_bindings",
        lambda persona_id: [_legacy_binding()],
    )
    assert kb_intake_service._persona_uses_graph_bundle_pipeline("any-persona-id") is False


def test_persona_uses_graph_bundle_pipeline_false_when_no_active_binding(monkeypatch):
    monkeypatch.setattr(
        kb_intake_service.supabase_client,
        "get_workflow_bindings",
        lambda persona_id: [{**_v3_binding(), "active": False}],
    )
    assert kb_intake_service._persona_uses_graph_bundle_pipeline("any-persona-id") is False


def test_save_routes_v3_persona_through_graph_bundle_branch(monkeypatch):
    """save() must dispatch to _save_via_graph_bundle for a v3-bound
    persona, and must NOT touch graph_document_publisher (the v2 store) at
    all on that path -- that's the whole point of this redesign."""
    def _fail_if_called(*_a, **_k):
        raise AssertionError("graph_document_publisher.publish must not run on the v3 path")

    monkeypatch.setattr(kb_intake_service.graph_document_publisher, "publish", _fail_if_called)

    session = {
        "id": "sess-v3", "mode": "criar", "persona_id": "pid-1", "persona_slug": "some-persona",
        "messages": [],
        "classification": {"persona_slug": "some-persona", "content_type": "brand", "title": "Teste"},
        "knowledge_plan": {"entries": [{
            "content_type": "brand", "slug": "minha-marca", "title": "Minha Marca",
            "status": "confirmado", "content": "Marca teste.", "metadata": {"parent_slug": "self"},
        }]},
        "current_block_counts": {},
    }
    monkeypatch.setattr(kb_intake_service, "_get_session", lambda _sid: session)
    monkeypatch.setattr(kb_intake_service, "_save_session", lambda _s: None)

    # First call only normalizes the plan and asks the operator to confirm
    # it (real save() behaviour, unrelated to this redesign) -- mirror that
    # confirm step, then flip on the v3 routing for the real save.
    first = kb_intake_service.save("sess-v3", "", None)
    assert first.get("error_code") == "PLAN_CONFIRMATION_REQUIRED"
    session["confirmed_plan_hash"] = first["plan_hash"]

    calls: list[str] = []
    monkeypatch.setattr(
        kb_intake_service, "_persona_uses_graph_bundle_pipeline", lambda _pid: True
    )
    monkeypatch.setattr(
        kb_intake_service,
        "_save_via_graph_bundle",
        lambda *a, **k: calls.append("graph_bundle") or {"ok": True, "status": "pending_approval"},
    )

    result = kb_intake_service.save("sess-v3", "", None)

    assert calls == ["graph_bundle"]
    assert result == {"ok": True, "status": "pending_approval"}


def test_approve_publication_requires_matching_checksum(monkeypatch):
    from routes import kb_intake as kb_intake_routes

    session = {
        "id": "sess-1", "user_id": "user-1", "stage": "awaiting_publication_approval",
        "pending_graph_bundle": {"bundle_version": "1.0"},
        "pending_publication_plan": {
            "draft_checksum": "sha256:aaa", "runtime_checksum": "sha256:bbb",
            "breaking_contract_changes": [],
        },
    }
    monkeypatch.setattr(kb_intake_routes, "get_session", lambda _sid: session)
    monkeypatch.setattr(
        kb_intake_routes.auth_service, "current_user", lambda _req: {"id": "user-1", "email": "op@test"}
    )

    def _reject_stale(*_a, **k):
        raise kb_intake_routes.graph_bundle_publisher.GraphBundlePublishError(
            ["approved_draft_checksum_mismatch"]
        )

    monkeypatch.setattr(kb_intake_routes.graph_bundle_publisher, "stage_bundle", _reject_stale)

    from fastapi import HTTPException

    body = kb_intake_routes.ApprovePublicationBody(
        approved_draft_checksum="sha256:stale", approved_runtime_checksum="sha256:bbb",
    )
    try:
        kb_intake_routes.approve_publication("sess-1", body, request=_FakeRequest())
        raise AssertionError("expected HTTPException for stale checksum")
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "GRAPH_BUNDLE_STAGE_FAILED" in str(exc.detail)


def test_approve_publication_blocks_breaking_changes_without_ack(monkeypatch):
    from routes import kb_intake as kb_intake_routes
    from fastapi import HTTPException

    session = {
        "id": "sess-2", "user_id": "user-1", "stage": "awaiting_publication_approval",
        "pending_graph_bundle": {"bundle_version": "1.0"},
        "pending_publication_plan": {
            "draft_checksum": "sha256:aaa", "runtime_checksum": "sha256:bbb",
            "breaking_contract_changes": ["branch_structure_changed:audience:x"],
        },
    }
    monkeypatch.setattr(kb_intake_routes, "get_session", lambda _sid: session)
    monkeypatch.setattr(
        kb_intake_routes.auth_service, "current_user", lambda _req: {"id": "user-1", "email": "op@test"}
    )
    stage_calls: list[dict] = []
    monkeypatch.setattr(
        kb_intake_routes.graph_bundle_publisher, "stage_bundle",
        lambda *_a, **k: stage_calls.append(k) or {"publication": {"id": "p1", "version": 2, "checksum": "sha256:bbb"}},
    )

    body = kb_intake_routes.ApprovePublicationBody(
        approved_draft_checksum="sha256:aaa", approved_runtime_checksum="sha256:bbb",
        acknowledge_breaking_changes=False,
    )
    try:
        kb_intake_routes.approve_publication("sess-2", body, request=_FakeRequest())
        raise AssertionError("expected HTTPException for unacknowledged breaking change")
    except HTTPException as exc:
        assert exc.status_code == 400
        assert stage_calls == []  # must reject BEFORE staging, no writes attempted


class _FakeRequest:
    headers: dict = {}

