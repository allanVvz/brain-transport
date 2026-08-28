from __future__ import annotations

from workers.whatsapp_dispatch_worker import WhatsAppDispatchWorker


def test_deterministic_inbound_delegates_without_burning_attempt(monkeypatch):
    row = {
        "id": "buffer-1",
        "persona_id": "persona-1",
        "lead_ref": 7,
        "payload": {"text": "Quero agendar"},
        "channel_binding_id": "binding-1",
        "whatsapp_phone_number_id": "phone-1",
        "external_message_id": "wamid-1",
        "correlation_id": "correlation-1",
    }
    worker = WhatsAppDispatchWorker()
    calls: list[dict] = []

    monkeypatch.setattr(
        "workers.whatsapp_dispatch_worker.supabase_client.get_persona_by_id",
        lambda _persona_id: {"id": "persona-1", "slug": "persona"},
    )
    monkeypatch.setattr(
        "workers.whatsapp_dispatch_worker.supabase_client.get_lead_by_ref",
        lambda _lead_ref: {"id": 7, "ai_paused": False},
    )
    monkeypatch.setattr(
        "workers.whatsapp_dispatch_worker.supabase_client.get_messages",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "workers.whatsapp_dispatch_worker.supabase_client.get_workflow_binding_by_id",
        lambda _binding_id: {
            "id": "binding-1",
            "active": True,
            "persona_id": "persona-1",
            "metadata": {"decision_owner": "deterministic"},
        },
    )
    monkeypatch.setattr(
        "workers.whatsapp_dispatch_worker.supabase_client.mark_whatsapp_attempt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("transport must not own deterministic decision attempts")
        ),
    )
    monkeypatch.setattr(
        "workers.whatsapp_dispatch_worker.runtime_client.execute_inbound",
        lambda payload: calls.append(payload) or {"ok": True, "handoff": False},
    )
    monkeypatch.setattr(
        "workers.whatsapp_dispatch_worker.supabase_client.complete_whatsapp_buffer",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "workers.whatsapp_dispatch_worker.event_emitter.emit",
        lambda *_args, **_kwargs: None,
    )

    worker._dispatch_inbound(row)

    assert calls == [{
        "persona_slug": "persona",
        "lead_ref": 7,
        "message": "Quero agendar",
        "message_id": "wamid-1",
        "correlation_id": "correlation-1",
        "phone_number_id": "phone-1",
        "channel_binding_id": "binding-1",
        "inbound_buffer_id": "buffer-1",
    }]


def test_unavailable_runtime_releases_deterministic_inbound_for_retry(monkeypatch):
    worker = WhatsAppDispatchWorker()
    row = {
        "id": "buffer-2",
        "direction": "inbound",
        "attempt_count": 1,
        "max_attempts": 5,
    }
    releases: list[tuple] = []
    monkeypatch.setattr(
        "workers.whatsapp_dispatch_worker.supabase_client.release_whatsapp_buffer",
        lambda *args, **kwargs: releases.append((args, kwargs)),
    )

    worker._retry_or_dead_letter(row, RuntimeError("runtime unavailable"))

    assert releases
    assert releases[0][0][1] == "retry"
