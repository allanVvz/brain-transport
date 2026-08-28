"""Regression coverage for persona-scoped WhatsApp binding provisioning."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))


def _admin_request():
    return SimpleNamespace(state=SimpleNamespace(user={"id": "admin-1", "role": "admin"}))


def test_new_persona_cannot_claim_another_personas_phone_number(monkeypatch):
    from routes import integrations

    persona = {"id": "persona-new", "slug": "new-persona"}
    monkeypatch.setattr(integrations.auth_service, "is_admin", lambda _user: True)
    monkeypatch.setattr(integrations.auth_service, "assert_persona_access", lambda *_a, **_k: None)
    monkeypatch.setattr(integrations.supabase_client, "get_persona", lambda _slug: persona)
    monkeypatch.setattr(
        integrations.supabase_client,
        "get_persona_integration_connection",
        lambda *_args: {},
    )
    monkeypatch.setattr(
        integrations.supabase_client,
        "get_workflow_bindings_by_phone_number_id",
        lambda _phone_id: [{"id": "binding-existing", "persona_id": "persona-existing"}],
    )
    monkeypatch.setattr(
        integrations.supabase_client,
        "upsert_workflow_binding",
        lambda _payload: pytest.fail("foreign phone must be rejected before upsert"),
    )

    with pytest.raises(HTTPException) as error:
        integrations.put_whatsapp_binding(
            "new-persona",
            integrations.WhatsAppBindingBody(
                phone_number_id="phone-existing",
                mode="disabled",
            ),
            _admin_request(),
        )

    assert error.value.status_code == 409
    assert "reassociacao auditada" in str(error.value.detail)

