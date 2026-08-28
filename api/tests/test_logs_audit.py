"""GET /logs/audit used to 404 on every call: the frontend
(dashboard/app/logs/page.tsx, api.auditLogs in dashboard/lib/api.ts) has
called this exact route/param shape since it was built, but routes/logs.py
never defined it â€” every event in system_events (conversation.fail_safe_handoff,
whatsapp.safety_violation, ...) was invisible in the Logs > Auditoria tab.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from routes import logs


def request_for(user: dict, access: list[dict] | None = None):
    return SimpleNamespace(state=SimpleNamespace(user=user, persona_access=access or []))


def test_admin_audit_logs_passes_filters_through(monkeypatch):
    captured = {}

    def fake_list_system_events(**kwargs):
        captured.update(kwargs)
        return [{"id": "e1", "event_type": "conversation.fail_safe_handoff", "level": "error"}]

    monkeypatch.setattr(logs.supabase_client, "list_system_events", fake_list_system_events)

    request = request_for({"id": "u1", "role": "admin", "account_type": "internal"})
    result = logs.audit_logs(
        request,
        entity_type=None,
        event_type="conversation.fail_safe_handoff,whatsapp.safety_violation",
        persona_id=None,
        entity_id=None,
        since=None,
        search=None,
        level="error",
        limit=200,
    )

    assert result == [{"id": "e1", "event_type": "conversation.fail_safe_handoff", "level": "error"}]
    assert captured["event_types"] == ["conversation.fail_safe_handoff", "whatsapp.safety_violation"]
    assert captured["level"] == "error"
    assert captured["limit"] == 200


def test_non_admin_scoped_to_their_personas_only(monkeypatch):
    calls = []

    def fake_list_system_events(**kwargs):
        calls.append(kwargs)
        return [
            {
                "id": f"e-{kwargs['persona_id']}",
                "persona_id": kwargs["persona_id"],
                "created_at": "2026-08-02T00:00:00+00:00",
            }
        ]

    monkeypatch.setattr(logs.supabase_client, "list_system_events", fake_list_system_events)

    request = request_for(
        {"id": "u2", "role": "user", "account_type": "client"},
        [
            {"persona_id": "p1", "can_view": True, "can_edit": False, "can_manage": False},
            {"persona_id": "p2", "can_view": True, "can_edit": False, "can_manage": False},
        ],
    )
    result = logs.audit_logs(
        request,
        entity_type=None, event_type=None, persona_id=None, entity_id=None,
        since=None, search=None, level=None, limit=200,
    )

    assert {row["persona_id"] for row in result} == {"p1", "p2"}
    assert len(calls) == 2


def test_persona_scoped_request_checks_access(monkeypatch):
    monkeypatch.setattr(
        logs.auth_service, "assert_persona_access",
        lambda request, persona_id=None, persona_slug=None: (_ for _ in ()).throw(
            AssertionError("access checked")
        ),
    )
    request = request_for({"id": "u3", "role": "user", "account_type": "client"})
    try:
        logs.audit_logs(
            request,
            entity_type=None, event_type=None, persona_id="p1", entity_id=None,
            since=None, search=None, level=None, limit=200,
        )
        raised = False
    except AssertionError:
        raised = True
    assert raised

