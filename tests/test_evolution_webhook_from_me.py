"""Evolution webhook must record messages sent by hand from the linked
phone (fromMe=true) instead of silently dropping them.

Regression test for the 2026-08-01 finding: any message typed directly in
WhatsApp on the phone paired to an Evolution/Baileys instance never showed
up anywhere in the platform, because evolution_webhook.py unconditionally
ignored every event with from_me=true. Real-time fromMe events ARE
delivered by Baileys/Evolution (this is not a provider limitation) — our
own handler just threw them away.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "api"
for path in (API_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture
def app_client(monkeypatch):
    from routes import evolution_webhook as mod

    monkeypatch.setenv("EVOLUTION_WEBHOOK_HMAC_SECRET", "test-secret")

    app = FastAPI()
    app.include_router(mod.router)
    return app, mod, TestClient(app)


def _token(binding_id: str) -> str:
    from services import auth_service

    return auth_service._b64encode(
        hmac.new(b"test-secret", binding_id.encode(), hashlib.sha256).digest()
    )


def test_from_me_message_is_recorded_not_dropped(monkeypatch, app_client):
    app, mod, client = app_client
    binding_id = "binding-1"
    binding = {
        "id": binding_id,
        "persona_id": "persona-1",
        "provider": "evolution_baileys",
        "provider_instance_key": "brain-aurora-test",
        "active": True,
        "metadata": {"mode": "active"},
    }
    monkeypatch.setattr(mod.supabase_client, "get_workflow_binding_by_id", lambda _id: binding)

    ensured_lead = {"id": 42}
    ensure_calls = []
    monkeypatch.setattr(
        mod.supabase_client,
        "ensure_channel_lead",
        lambda **kwargs: ensure_calls.append(kwargs) or ensured_lead,
    )

    inserted = []
    monkeypatch.setattr(
        mod.supabase_client,
        "insert_message",
        lambda data: inserted.append(data),
    )

    def _boom_enqueue(**_kwargs):
        raise AssertionError("from_me messages must never be enqueued for dispatch/AI reply")

    monkeypatch.setattr(mod.supabase_client, "enqueue_whatsapp_envelope", _boom_enqueue)

    payload = {
        "event": "messages.upsert",
        "instance": "brain-aurora-test",
        "data": {
            "key": {
                "id": "WAMID123",
                "remoteJid": "555199999999@s.whatsapp.net",
                "fromMe": True,
            },
            "message": {"conversation": "oi, escrevi isso direto no celular"},
        },
    }
    resp = client.post(
        f"/webhooks/evolution/{binding_id}",
        json=payload,
        headers={"x-brain-webhook-token": _token(binding_id)},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["accepted"] == 0
    assert body["ignored"] == 1

    assert len(inserted) == 1
    recorded = inserted[0]
    assert recorded["lead_id"] == 42
    assert recorded["direction"] == "outbound"
    assert recorded["sender_type"] == "human"
    assert recorded["content"] == "oi, escrevi isso direto no celular"
    assert recorded["external_message_id"] == "WAMID123"
    assert recorded["metadata"]["source"] == "phone_manual"

    assert ensure_calls[0]["external_contact_id"] == "555199999999@s.whatsapp.net"


def test_connection_update_is_recorded_as_a_system_event(monkeypatch, app_client):
    """Regression test for the 2026-08-07 finding.

    A channel disconnected live with no trace anywhere in the product —
    evolution_webhook.py only ever logged the disconnect reason via
    logger.warning (raw function logs), never inserted it into
    system_events, so Settings > Logs > Auditoria never showed why.
    """
    app, mod, client = app_client
    binding_id = "binding-3"
    binding = {
        "id": binding_id,
        "persona_id": "persona-1",
        "provider": "evolution_baileys",
        "provider_instance_key": "brain-aurora-test",
        "active": True,
        "metadata": {"mode": "active"},
        "last_connection_at": "2026-08-01T00:00:00+00:00",
    }
    monkeypatch.setattr(mod.supabase_client, "get_workflow_binding_by_id", lambda _id: binding)
    monkeypatch.setattr(mod.supabase_client, "update_workflow_binding", lambda *_a, **_k: None)

    events = []
    monkeypatch.setattr(
        mod.supabase_client,
        "insert_event",
        lambda data, **kwargs: events.append({**data, **kwargs}),
    )

    payload = {
        "event": "connection.update",
        "instance": "brain-aurora-test",
        "data": {"state": "close", "statusCode": 428, "reason": "connectionClosed"},
    }
    resp = client.post(
        f"/webhooks/evolution/{binding_id}",
        json=payload,
        headers={"x-brain-webhook-token": _token(binding_id)},
    )
    assert resp.status_code == 202
    assert resp.json()["ignored"] == 1

    assert len(events) == 1
    recorded = events[0]
    assert recorded["event_type"] == "whatsapp.connection_update"
    assert recorded["entity_type"] == "whatsapp"
    assert recorded["entity_id"] == binding_id
    assert recorded["persona_id"] == "persona-1"
    assert recorded["payload"]["state"] == "disconnected"
    assert recorded["payload"]["status_code"] == "428"
    assert recorded["payload"]["reason"] == "connectionClosed"
    assert recorded["level"] == "warning"
    assert recorded["source"] == "whatsapp.connection"


def test_from_me_message_without_id_is_ignored_without_crashing(monkeypatch, app_client):
    app, mod, client = app_client
    binding_id = "binding-2"
    binding = {
        "id": binding_id,
        "persona_id": "persona-1",
        "provider": "evolution_baileys",
        "provider_instance_key": "brain-aurora-test",
        "active": True,
        "metadata": {"mode": "active"},
    }
    monkeypatch.setattr(mod.supabase_client, "get_workflow_binding_by_id", lambda _id: binding)

    def _boom(*_a, **_k):
        raise AssertionError("must not touch the DB when there is no message id")

    monkeypatch.setattr(mod.supabase_client, "ensure_channel_lead", _boom)
    monkeypatch.setattr(mod.supabase_client, "insert_message", _boom)

    payload = {
        "event": "messages.upsert",
        "instance": "brain-aurora-test",
        "data": {
            "key": {"remoteJid": "555199999999@s.whatsapp.net", "fromMe": True},
            "message": {"conversation": "sem id"},
        },
    }
    resp = client.post(
        f"/webhooks/evolution/{binding_id}",
        json=payload,
        headers={"x-brain-webhook-token": _token(binding_id)},
    )
    assert resp.status_code == 202
    assert resp.json()["ignored"] == 1
