from workers import n8n_mirror_worker


def test_mirror_requests_execution_data_and_persists_node_errors(monkeypatch):
    captured = {}
    rows = []

    monkeypatch.setattr(
        n8n_mirror_worker.integration_service,
        "system_service_has_runtime_credentials",
        lambda _service: True,
    )

    def get_executions(*, limit, include_data):
        captured.update({"limit": limit, "include_data": include_data})
        return [
            {
                "id": "128",
                "status": "success",
                "startedAt": "2026-08-04T20:57:02Z",
                "stoppedAt": "2026-08-04T20:57:03Z",
                "workflowData": {"name": "Brain â€” Persona â€” ConversaÃ§Ã£o"},
                "data": {
                    "resultData": {
                        "runData": {
                            "DeepSeek agentic reply": [
                                {
                                    "error": {
                                        "message": "invalid syntax",
                                        "httpCode": 400,
                                    },
                                    "data": {
                                        "main": [[{"json": {"persona_id": "p1", "lead_ref": 41}}]]
                                    },
                                }
                            ]
                        }
                    }
                },
            }
        ]

    monkeypatch.setattr(n8n_mirror_worker.n8n_client, "get_executions", get_executions)
    monkeypatch.setattr(n8n_mirror_worker.supabase_client, "upsert_n8n_execution", rows.append)
    monkeypatch.setattr(n8n_mirror_worker.sre_logger, "info", lambda *a, **k: None)

    n8n_mirror_worker.N8nMirrorWorker()._run_cycle()

    assert captured == {"limit": 50, "include_data": True}
    assert rows[0]["status"] == "success"
    assert rows[0]["node_errors"] == [
        {
            "node": "DeepSeek agentic reply",
            "error": {"message": "invalid syntax", "httpCode": 400},
        }
    ]
    assert rows[0]["persona_id"] == "p1"
    assert rows[0]["lead_id"] == "41"

