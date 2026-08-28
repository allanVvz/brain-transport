import json

import pytest

from services import deepseek_n8n_service


MODEL_BINDING = {
    "model": "fixture-model",
    "endpoint": "https://models.example.test/chat/completions",
    "reply_source": "fixture-model",
}


def _silence_events(monkeypatch):
    monkeypatch.setattr(deepseek_n8n_service.event_emitter, "emit", lambda *a, **k: None)


def test_every_persona_uses_the_same_graph_agentic_template():
    commerce = deepseek_n8n_service._workflow_for_persona(
        {
            "id": "p-commerce",
            "slug": "commerce",
            "name": "Commerce",
            "config": {"portal": {"business_model": "commerce"}},
        },
        credential_id="cred-commerce",
        credential_name="DeepSeek commerce",
        model_binding=MODEL_BINDING,
    )
    appointment = deepseek_n8n_service._workflow_for_persona(
        {
            "id": "p-appointment",
            "slug": "appointment",
            "name": "Appointment",
            "config": {"portal": {"business_model": "appointment"}},
        },
        credential_id="cred-appointment",
        credential_name="DeepSeek appointment",
        model_binding=MODEL_BINDING,
    )

    assert commerce["meta"]["template"] == "graph_agentic_v3"
    assert appointment["meta"]["template"] == "graph_agentic_v3"
    assert [node["id"] for node in commerce["nodes"]] == [
        node["id"] for node in appointment["nodes"]
    ]
    assert commerce["connections"] == appointment["connections"]
    assert "baita" not in str(commerce).lower()
    assert "aurora" not in str(appointment).lower()


def test_canonical_template_connections_resolve_to_published_node_names():
    template = json.loads(
        deepseek_n8n_service._TEMPLATE.read_text(encoding="utf-8")
    )
    deepseek_n8n_service._validate_workflow_topology(template)


def test_workflow_topology_rejects_dangling_connection():
    with pytest.raises(ValueError, match="connection target is missing"):
        deepseek_n8n_service._validate_workflow_topology({
            "nodes": [{"name": "Inbound"}],
            "connections": {
                "Inbound": {
                    "main": [[{"node": "Missing", "type": "main", "index": 0}]],
                },
            },
        })


def test_model_request_is_built_in_code_and_http_body_is_simple():
    workflow = _live_workflow("cred-1")
    request_node = next(node for node in workflow["nodes"] if node["id"] == "model_request")
    deepseek_node = next(node for node in workflow["nodes"] if node["id"] == "deepseek")
    fail_safe = next(node for node in workflow["nodes"] if node["id"] == "failsafe")

    assert "context_cards" in request_node["parameters"]["jsCode"]
    assert "rendered_content" not in request_node["parameters"]["jsCode"]
    assert "prompt_budget_exceeded" in request_node["parameters"]["jsCode"]
    assert deepseek_node["parameters"]["body"] == "={{JSON.stringify($json.request_body)}}"
    assert "buffer_id" in fail_safe["parameters"]["body"]
    assert "correlation_id" in fail_safe["parameters"]["body"]
    assert "workflow_template" in fail_safe["parameters"]["body"]


def test_model_proposal_is_proved_and_repaired_against_graph_before_commit():
    workflow = _live_workflow("cred-1")
    node_ids = [node["id"] for node in workflow["nodes"]]
    reconcile = next(node for node in workflow["nodes"] if node["id"] == "reconcile")
    policy = next(node for node in workflow["nodes"] if node["id"] == "policy")
    model_response = next(
        node for node in workflow["nodes"] if node["id"] == "model_response"
    )
    final_response = next(
        node for node in workflow["nodes"] if node["id"] == "final_response"
    )
    commit = next(node for node in workflow["nodes"] if node["id"] == "commit")
    repair_gate = next(node for node in workflow["nodes"] if node["id"] == "repair_gate")
    repair_reconcile = next(node for node in workflow["nodes"] if node["id"] == "repair_reconcile")

    assert node_ids.index("model_response") < node_ids.index("reconcile")
    assert node_ids.index("reconcile") < node_ids.index("repair_gate")
    assert node_ids.index("repair_gate") < node_ids.index("repair_reconcile")
    assert node_ids.index("repair_reconcile") < node_ids.index("final_response")
    assert node_ids.index("final_response") < node_ids.index("commit")
    assert "model_observation" in reconcile["parameters"]["body"]
    assert "contract_probe" in policy["parameters"]["body"]
    assert "extracted_facts" in model_response["parameters"]["jsCode"]
    assert "repair_required" in str(repair_gate["parameters"])
    assert "repair_attempt" in repair_reconcile["parameters"]["body"] or "model_observation" in repair_reconcile["parameters"]["body"]
    assert "graph proof" in final_response["parameters"]["jsCode"]
    assert "response: $json.response" in commit["parameters"]["body"]


def test_canonical_template_has_no_persona_or_business_hardcode():
    workflow = _live_workflow("cred-1")
    serialized = str(workflow).lower()

    for forbidden in ("aurora", "sofia", "allan", "onix", "higienizacao"):
        assert forbidden not in serialized


def test_provision_keeps_key_only_in_n8n_credential(monkeypatch):
    _silence_events(monkeypatch)
    calls = {}

    def create_credential(**payload):
        calls["credential"] = payload
        return {"id": "credential-new"}

    def update_workflow(workflow_id, workflow):
        calls["workflow"] = workflow
        return {"id": workflow_id}

    deleted = []
    monkeypatch.setattr(deepseek_n8n_service.n8n_client, "create_credential", create_credential)
    monkeypatch.setattr(deepseek_n8n_service.n8n_client, "update_workflow", update_workflow)
    monkeypatch.setattr(deepseek_n8n_service.n8n_client, "activate_workflow", lambda workflow_id: {"id": workflow_id})
    monkeypatch.setattr(deepseek_n8n_service.n8n_client, "delete_credential", deleted.append)

    key = "sk-test-deepseek-secret"
    result = deepseek_n8n_service.provision(
        persona={
            "id": "persona-id",
            "slug": "baita-conveniencia",
            "name": "Baita",
            "config": {"agent_slug": "vitoria"},
        },
        api_key=key,
        previous_config={
            "n8n_workflow_id": "workflow-existing",
            "n8n_credential_id": "credential-old",
        },
        model_binding=MODEL_BINDING,
    )

    assert calls["credential"]["data"]["value"] == f"Bearer {key}"
    assert key not in str(calls["workflow"])
    deepseek_node = next(node for node in calls["workflow"]["nodes"] if node["id"] == "deepseek")
    assert deepseek_node["credentials"]["httpHeaderAuth"]["id"] == "credential-new"
    assert calls["workflow"]["settings"]["saveDataSuccessExecution"] == "all"
    assert calls["workflow"]["meta"]["template"] == "graph_agentic_v3"
    assert calls["workflow"]["active"] is False
    assert result["n8n_workflow_id"] == "workflow-existing"
    assert result["n8n_credential_id"] == "credential-new"
    assert key not in str(result)
    assert deleted == ["credential-old"]


def test_provision_rolls_back_only_new_credential_when_workflow_fails(monkeypatch):
    _silence_events(monkeypatch)
    deleted = []
    monkeypatch.setattr(
        deepseek_n8n_service.n8n_client,
        "create_credential",
        lambda **_payload: {"id": "credential-new"},
    )
    monkeypatch.setattr(
        deepseek_n8n_service.n8n_client,
        "update_workflow",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("n8n failed")),
    )
    monkeypatch.setattr(
        deepseek_n8n_service.n8n_client,
        "delete_credential",
        deleted.append,
    )

    try:
        deepseek_n8n_service.provision(
            persona={"slug": "baita-conveniencia", "name": "Baita"},
            api_key="sk-test-deepseek-secret",
            previous_config={
                "n8n_workflow_id": "workflow-existing",
                "n8n_credential_id": "credential-old",
            },
            model_binding=MODEL_BINDING,
        )
        raise AssertionError("provision should fail")
    except RuntimeError as exc:
        assert str(exc) == "n8n failed"

    assert deleted == ["credential-new"]


def test_resync_workflow_reuses_existing_credential_and_reactivates(monkeypatch):
    """Regression test: this replaces the manual SSH ritual (rebuild
    workflow from the template on disk, update_workflow, activate_workflow)
    that was run by hand for every persona-level engine/config change this
    session â€” the settings UI must be able to trigger the same steps."""
    calls = {}
    _silence_events(monkeypatch)

    def update_workflow(workflow_id, workflow):
        calls["workflow_id"] = workflow_id
        calls["workflow"] = workflow
        return {"id": workflow_id}

    monkeypatch.setattr(deepseek_n8n_service.n8n_client, "update_workflow", update_workflow)
    monkeypatch.setattr(
        deepseek_n8n_service.n8n_client,
        "activate_workflow",
        lambda workflow_id: calls.setdefault("activated", workflow_id),
    )

    result = deepseek_n8n_service.resync_workflow_for_persona(
        {"id": "persona-id", "slug": "baita-conveniencia", "name": "Baita", "config": {}},
        {
            "n8n_credential_id": "credential-existing",
            "n8n_workflow_id": "workflow-existing",
            **MODEL_BINDING,
        },
    )

    assert result["n8n_workflow_id"] == "workflow-existing"
    assert result["conversation_webhook_path"] == "baita-conveniencia/conversation"
    assert calls["workflow_id"] == "workflow-existing"
    assert calls["activated"] == "workflow-existing"
    deepseek_node = next(node for node in calls["workflow"]["nodes"] if node["id"] == "deepseek")
    assert deepseek_node["credentials"]["httpHeaderAuth"]["id"] == "credential-existing"


def test_resync_can_update_same_template_and_keep_workflow_inactive(monkeypatch):
    calls = {}
    _silence_events(monkeypatch)
    monkeypatch.setattr(
        deepseek_n8n_service.n8n_client, "update_workflow",
        lambda workflow_id, workflow: calls.update(
            {"workflow_id": workflow_id, "workflow": workflow}
        ) or {"id": workflow_id},
    )
    monkeypatch.setattr(
        deepseek_n8n_service.n8n_client, "activate_workflow",
        lambda _workflow_id: (_ for _ in ()).throw(
            AssertionError("inactive resync must not activate")
        ),
    )
    monkeypatch.setattr(
        deepseek_n8n_service.n8n_client, "deactivate_workflow",
        lambda workflow_id: calls.setdefault("deactivated", workflow_id),
    )

    result = deepseek_n8n_service.resync_workflow_for_persona(
        {"id": "persona-off", "slug": "persona-off", "name": "Off", "config": {}},
        {
            "n8n_credential_id": "credential-existing",
            "n8n_workflow_id": "workflow-existing",
            **MODEL_BINDING,
        },
        activate_workflow=False,
    )

    assert calls["deactivated"] == "workflow-existing"
    assert calls["workflow"]["meta"]["template"] == "graph_agentic_v3"
    assert result["workflow_active"] is False


def test_resync_workflow_creates_it_when_missing_reusing_the_credential(monkeypatch):
    """Regression test for the exact gap found live: a persona
    (baita-conveniencia) already had a DeepSeek credential provisioned but
    its workflow reference was missing, so switching to n8n_agents errored
    out instead of just working. The raw API key isn't recoverable once
    saved (it only lives inside the n8n credential from then on), so the
    fix must build a new workflow from the credential that's already
    there, never ask for the key again."""
    calls = {}
    _silence_events(monkeypatch)

    def create_workflow(workflow):
        calls["created"] = workflow
        return {"id": "workflow-new"}

    monkeypatch.setattr(deepseek_n8n_service.n8n_client, "create_workflow", create_workflow)
    monkeypatch.setattr(
        deepseek_n8n_service.n8n_client,
        "update_workflow",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must create, not update, when no workflow id exists")),
    )
    monkeypatch.setattr(
        deepseek_n8n_service.n8n_client,
        "activate_workflow",
        lambda workflow_id: calls.setdefault("activated", workflow_id),
    )

    result = deepseek_n8n_service.resync_workflow_for_persona(
        {"id": "persona-id", "slug": "baita-conveniencia", "name": "Baita", "config": {}},
        {"n8n_credential_id": "credential-existing", **MODEL_BINDING},
    )

    assert result["n8n_workflow_id"] == "workflow-new"
    assert result["conversation_webhook_path"] == "baita-conveniencia/conversation"
    assert calls["activated"] == "workflow-new"
    deepseek_node = next(node for node in calls["created"]["nodes"] if node["id"] == "deepseek")
    assert deepseek_node["credentials"]["httpHeaderAuth"]["id"] == "credential-existing"


def test_resync_workflow_requires_prior_provisioning(monkeypatch):
    _silence_events(monkeypatch)
    try:
        deepseek_n8n_service.resync_workflow_for_persona(
            {"slug": "baita-conveniencia", "name": "Baita", "config": {}},
            {},
        )
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "provisionado" in str(exc)


def _live_workflow(credential_id: str) -> dict:
    workflow = deepseek_n8n_service._workflow_for_persona(
        {"id": "persona-1", "slug": "baita-conveniencia", "name": "Baita"},
        credential_id=credential_id,
        credential_name="Brain DeepSeek",
        model_binding=MODEL_BINDING,
    )
    workflow["id"] = "wf-1"
    workflow["active"] = True
    return workflow


def test_check_workflow_wiring_ok_when_workflow_active_and_credential_matches(monkeypatch):
    monkeypatch.setattr(
        deepseek_n8n_service.n8n_client, "get_workflow",
        lambda workflow_id: _live_workflow("cred-1"),
    )
    result = deepseek_n8n_service.check_workflow_wiring(
        {
            "n8n_workflow_id": "wf-1",
            "n8n_credential_id": "cred-1",
            "conversation_webhook_path": "baita-conveniencia/conversation",
        }
    )
    assert result["ok"] is True
    assert result["reason"] is None
    assert result["diagnostics"]["checks"]["required_nodes"] is True


def test_check_workflow_wiring_fails_when_workflow_deleted(monkeypatch):
    monkeypatch.setattr(deepseek_n8n_service.n8n_client, "get_workflow", lambda workflow_id: None)
    result = deepseek_n8n_service.check_workflow_wiring(
        {"n8n_workflow_id": "wf-1", "n8n_credential_id": "cred-1"}
    )
    assert result["ok"] is False
    assert "nao existe mais" in result["reason"]


def test_check_workflow_wiring_fails_when_credential_reference_drifted(monkeypatch):
    monkeypatch.setattr(
        deepseek_n8n_service.n8n_client, "get_workflow",
        lambda workflow_id: _live_workflow("cred-DIFFERENT"),
    )
    result = deepseek_n8n_service.check_workflow_wiring(
        {"n8n_workflow_id": "wf-1", "n8n_credential_id": "cred-1"}
    )
    assert result["ok"] is False
    assert "credencial diferente" in result["reason"]


def test_check_workflow_wiring_requires_config_present(monkeypatch):
    result = deepseek_n8n_service.check_workflow_wiring({})
    assert result["ok"] is False
    assert "nao provisionado" in result["reason"]

