from routes import conversations


def test_fail_safe_persists_structured_n8n_node_diagnostic(monkeypatch):
    events = []
    handoffs = []
    monkeypatch.setenv("AI_BRAIN_WEBHOOK_TOKEN", "token")
    monkeypatch.setattr(
        conversations.conversation_runtime.supabase_client,
        "get_lead_by_ref",
        lambda _lead_ref: {"id": 41, "persona_id": "persona-1"},
    )
    monkeypatch.setattr(
        conversations.conversation_runtime.supabase_client,
        "handoff_whatsapp_lead",
        handoffs.append,
    )
    monkeypatch.setattr(
        conversations.conversation_runtime.supabase_client,
        "insert_event",
        lambda data, **kwargs: events.append((data, kwargs)),
    )

    result = conversations.fail_safe_handoff(
        conversations.FailSafeHandoffRequest(
            lead_ref=41,
            correlation_id="meta:test",
            reason="workflow_step_failed:DeepSeek agentic reply:invalid syntax",
            diagnostic={
                "failed_node": "DeepSeek agentic reply",
                "message": "invalid syntax",
                "http_code": 400,
                "workflow_template": "graph_agentic_v1",
            },
        ),
        x_webhook_token="token",
    )

    assert result == {"ok": True, "handoff": True, "ai_paused": True}
    assert handoffs == [41]
    assert [event[0]["event_type"] for event in events] == [
        "n8n.workflow_step_failed",
        "conversation.fail_safe_handoff",
    ]
    assert events[0][0]["payload"]["failed_node"] == "DeepSeek agentic reply"
    assert events[0][0]["payload"]["http_code"] == 400

