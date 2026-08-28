from services import operator_messaging


def test_duplicate_client_message_is_not_enqueued_twice(monkeypatch):
    lead = {"id": 42, "persona_id": "persona-1"}
    monkeypatch.setattr(operator_messaging.supabase_client, "get_lead_by_ref", lambda _ref: lead)
    monkeypatch.setattr(
        operator_messaging.whatsapp_outbox,
        "resolve_lead_binding",
        lambda _lead: {"id": "binding-1"},
    )
    rows = []
    monkeypatch.setattr(
        operator_messaging.supabase_client,
        "get_whatsapp_buffer_by_idempotency",
        lambda _key: rows[0] if rows else None,
    )

    def enqueue_outbound(**kwargs):
        rows.append({
            "id": "buffer-1",
            "lead_ref": 42,
            "channel_binding_id": "binding-1",
            "status": "pending_send",
        })
        return {"buffer_id": "buffer-1", "status": "pending_send", "deduplicated": False}

    monkeypatch.setattr(operator_messaging.whatsapp_outbox, "enqueue_outbound", enqueue_outbound)
    monkeypatch.setattr(operator_messaging.event_emitter, "emit", lambda *_args, **_kwargs: None)

    first = operator_messaging.enqueue(
        lead_ref=42,
        persona_id="persona-1",
        client_message_id="same-id",
        text="Oi",
    )
    second = operator_messaging.enqueue(
        lead_ref=42,
        persona_id="persona-1",
        client_message_id="same-id",
        text="Oi",
    )

    assert first["deduplicated"] is False
    assert second["deduplicated"] is True
    assert len(rows) == 1
