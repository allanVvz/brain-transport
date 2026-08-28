"""Provision a persona-scoped DeepSeek credential and canonical n8n workflow."""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from services import event_emitter, n8n_client


_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "n8n-workflows"
    / "persona-conversation-template.json"
)
_TEMPLATE_VERSION = "graph_agentic_v3"
_REQUIRED_NODE_IDS = {
    "inbound", "binding", "context", "policy", "model_request",
    "deepseek", "model_response", "reconcile", "repair_gate",
    "repair_request", "deepseek_repair", "repair_response",
    "repair_reconcile", "final_response", "commit", "failsafe", "respond",
}


def _validate_workflow_topology(workflow: dict[str, Any]) -> None:
    """Reject dangling n8n connections before a workflow can be published."""
    nodes = workflow.get("nodes") or []
    names = [str(node.get("name") or "").strip() for node in nodes]
    if not names or any(not name for name in names):
        raise ValueError("conversation workflow contains an unnamed node")
    if len(names) != len(set(names)):
        raise ValueError("conversation workflow node names must be unique")
    known = set(names)
    for source_name, outputs in (workflow.get("connections") or {}).items():
        if source_name not in known:
            raise ValueError(
                f"conversation workflow connection source is missing: {source_name}"
            )
        for channel_outputs in (outputs or {}).values():
            for branch in channel_outputs or []:
                for connection in branch or []:
                    target_name = str(connection.get("node") or "").strip()
                    if target_name not in known:
                        raise ValueError(
                            "conversation workflow connection target is missing: "
                            f"{source_name} -> {target_name or '<empty>'}"
                        )


def _workflow_checksum(workflow: dict[str, Any]) -> str:
    payload = {
        key: workflow.get(key)
        for key in ("name", "nodes", "connections", "settings")
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _emit(
    persona: dict[str, Any],
    event_type: str,
    *,
    payload: dict[str, Any] | None = None,
    level: str = "info",
) -> None:
    persona_id = str(persona.get("id") or "") or None
    event_emitter.emit(
        event_type,
        entity_type="persona",
        entity_id=persona_id or str(persona.get("slug") or ""),
        persona_id=persona_id,
        payload={
            "persona_slug": str(persona.get("slug") or ""),
            "template": _TEMPLATE_VERSION,
            **(payload or {}),
        },
        level=level,
        source="services.deepseek_n8n_service",
    )


def _workflow_for_persona(
    persona: dict[str, Any],
    *,
    credential_id: str,
    credential_name: str,
    model_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    slug = str(persona.get("slug") or "").strip()
    if not slug:
        raise ValueError("persona slug is required")
    config = persona.get("config") or {}
    agent_slug = str(
        config.get("agent_slug")
        or (config.get("automation") or {}).get("agent_slug")
        or "assistant"
    )
    configured_model = dict(model_binding or {})
    model = str(configured_model.get("model") or "").strip()
    endpoint = str(configured_model.get("endpoint") or "").strip()
    reply_source = str(configured_model.get("reply_source") or model).strip()
    if not model or not endpoint.startswith("https://"):
        raise ValueError("conversation model binding requires model and HTTPS endpoint")
    template = json.loads(_TEMPLATE.read_text(encoding="utf-8"))
    webhook_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"brain-ai:{slug}:conversation"))
    serialized = json.dumps(template, ensure_ascii=False)
    for placeholder, value in {
        "__PERSONA_SLUG__": slug,
        "__PERSONA_NAME__": str(persona.get("name") or slug),
        "__AGENT_SLUG__": agent_slug,
        "__WEBHOOK_ID__": webhook_id,
        "__MODEL__": model,
        "__MODEL_ENDPOINT__": endpoint,
        "__REPLY_SOURCE__": reply_source,
    }.items():
        serialized = serialized.replace(placeholder, value)
    workflow = json.loads(serialized)
    _validate_workflow_topology(workflow)
    workflow["active"] = False
    for node in workflow.get("nodes") or []:
        if node.get("id") in {"deepseek", "deepseek_repair"}:
            node["credentials"] = {
                "httpHeaderAuth": {
                    "id": credential_id,
                    "name": credential_name,
                }
            }
    workflow.setdefault("settings", {})
    workflow.setdefault("meta", {})
    workflow["meta"]["binding"] = {
        **(workflow["meta"].get("binding") or {}),
        "persona_slug": slug,
        "agent_slug": agent_slug,
        "decision_owner": "n8n_agents",
        "pipeline_contract": "conversation_v3",
        "runtime_version": "graph_agent_runtime_v3",
        "reply_source": reply_source,
        "model_required": True,
        "model": model,
        "endpoint": endpoint,
    }
    return workflow


def provision(
    *,
    persona: dict[str, Any],
    api_key: str,
    previous_config: dict[str, Any] | None = None,
    model_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    previous = dict(previous_config or {})
    slug = str(persona.get("slug") or "")
    credential_name = f"Brain DeepSeek â€” {slug}"
    step = "create_credential"
    credential_id = ""
    _emit(persona, "n8n.workflow_provision.started")
    try:
        created = n8n_client.create_credential(
            name=credential_name,
            credential_type="httpHeaderAuth",
            data={"name": "Authorization", "value": f"Bearer {api_key}"},
        )
        credential_id = str(created.get("id") or "")
        if not credential_id:
            raise RuntimeError("n8n did not return a credential id")
        _emit(
            persona,
            "n8n.workflow_provision.credential_created",
            payload={"credential_id": credential_id},
        )
        step = "render_template"
        effective_model_binding = {
            **{
                key: value
                for key, value in previous.items()
                if key in {"model", "endpoint", "reply_source"} and value
            },
            **{
                key: value
                for key, value in dict(model_binding or {}).items()
                if key in {"model", "endpoint", "reply_source"} and value
            },
        }
        workflow = _workflow_for_persona(
            persona,
            credential_id=credential_id,
            credential_name=credential_name,
            model_binding=effective_model_binding,
        )
        workflow_checksum = _workflow_checksum(workflow)
        existing_workflow_id = str(previous.get("n8n_workflow_id") or "")
        step = "save_workflow"
        if existing_workflow_id:
            saved = n8n_client.update_workflow(existing_workflow_id, workflow)
        else:
            match = next(
                (
                    row for row in n8n_client.get_workflows()
                    if row.get("name") == workflow["name"]
                ),
                None,
            )
            saved = (
                n8n_client.update_workflow(str(match["id"]), workflow)
                if match
                else n8n_client.create_workflow(workflow)
            )
        workflow_id = str(saved.get("id") or existing_workflow_id)
        if not workflow_id:
            raise RuntimeError("n8n did not return a workflow id")
        # Provisioning creates an auditable inactive workflow. Selecting the
        # persona's n8n mode later performs validation and explicit activation.
    except Exception as exc:
        if credential_id:
            n8n_client.delete_credential(credential_id)
        _emit(
            persona,
            "n8n.workflow_provision.failed",
            payload={"step": step, "error": str(exc)[:2000]},
            level="error",
        )
        raise

    old_credential_id = str(previous.get("n8n_credential_id") or "")
    if old_credential_id and old_credential_id != credential_id:
        n8n_client.delete_credential(old_credential_id)
    result = {
        "n8n_credential_id": credential_id,
        "n8n_workflow_id": workflow_id,
        "conversation_webhook_path": f"{slug}/conversation",
        "model": workflow["meta"]["binding"]["model"],
        "endpoint": workflow["meta"]["binding"]["endpoint"],
        "reply_source": workflow["meta"]["binding"]["reply_source"],
        "fingerprint": hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12],
        "persona_id": str(persona.get("id") or ""),
        "persona_slug": slug,
        "workflow_template": _TEMPLATE_VERSION,
        "workflow_checksum": workflow_checksum,
        "runtime_version": "graph_agent_runtime_v3",
        "pipeline_contract": "conversation_v3",
    }
    _emit(
        persona,
        "n8n.workflow_provision.succeeded",
        payload={
            "workflow_id": workflow_id,
            "workflow_checksum": workflow_checksum,
            "webhook_path": result["conversation_webhook_path"],
        },
    )
    return result


def resync_workflow_for_persona(
    persona: dict[str, Any],
    deepseek_config: dict[str, Any],
    *,
    activate_workflow: bool = True,
) -> dict[str, Any]:
    """Rebuild this persona's n8n workflow from the current template/graph
    and publish it, reusing the DeepSeek credential already provisioned â€”
    creating the workflow if it doesn't exist yet.

    Called whenever the operator switches (or re-confirms) n8n_agents mode
    from the settings UI, so the live n8n workflow always matches what's on
    disk without a manual SSH resync â€” the same steps that were previously
    run by hand for every persona-level change.

    The credential is only ever reused, never recreated: once a DeepSeek
    key is saved, its raw value isn't retrievable from our own storage
    again (the only place it still lives is inside the n8n credential
    object itself), so a genuinely first-time setup â€” no credential at all
    â€” still requires the operator to (re)enter the key in Ferramentas. But
    a persona that already has a credential and simply never got (or lost)
    its workflow â€” the exact gap that made switching to n8n error out
    instead of just working â€” gets one built here from the template, no
    key re-entry needed.

    Returns the config dict with n8n_workflow_id (and conversation_webhook_
    path) filled in, so the caller can persist it back onto the persona's
    integration record.
    """
    credential_id = str(deepseek_config.get("n8n_credential_id") or "")
    if not credential_id:
        raise RuntimeError("DeepSeek nao provisionado para esta persona")
    credential_name = f"Brain DeepSeek â€” {persona.get('slug') or ''}"
    step = "render_template"
    workflow_id = str(deepseek_config.get("n8n_workflow_id") or "")
    _emit(
        persona,
        "n8n.workflow_resync.started",
        payload={"workflow_id": workflow_id or None},
    )
    try:
        workflow = _workflow_for_persona(
            persona,
            credential_id=credential_id,
            credential_name=credential_name,
            model_binding={
                "model": deepseek_config.get("model"),
                "endpoint": deepseek_config.get("endpoint"),
                "reply_source": deepseek_config.get("reply_source"),
            },
        )
        workflow_checksum = _workflow_checksum(workflow)
        step = "save_workflow"
        if workflow_id:
            n8n_client.update_workflow(workflow_id, workflow)
        else:
            created = n8n_client.create_workflow(workflow)
            workflow_id = str(created.get("id") or "")
            if not workflow_id:
                raise RuntimeError("n8n nao retornou um workflow id")
        step = "activate_workflow" if activate_workflow else "deactivate_workflow"
        if activate_workflow:
            n8n_client.activate_workflow(workflow_id)
        else:
            n8n_client.deactivate_workflow(workflow_id)
    except Exception as exc:
        _emit(
            persona,
            "n8n.workflow_resync.failed",
            payload={
                "workflow_id": workflow_id or None,
                "step": step,
                "error": str(exc)[:2000],
            },
            level="error",
        )
        raise
    slug = str(persona.get("slug") or "")
    result = {
        **deepseek_config,
        "n8n_workflow_id": workflow_id,
        "conversation_webhook_path": f"{slug}/conversation",
        "persona_id": str(persona.get("id") or ""),
        "persona_slug": slug,
        "workflow_template": _TEMPLATE_VERSION,
        "workflow_checksum": workflow_checksum,
        "runtime_version": "graph_agent_runtime_v3",
        "pipeline_contract": "conversation_v3",
        "workflow_active": activate_workflow,
    }
    _emit(
        persona,
        "n8n.workflow_resync.succeeded",
        payload={
            "workflow_id": workflow_id,
            "workflow_checksum": workflow_checksum,
            "webhook_path": result["conversation_webhook_path"],
            "active": activate_workflow,
        },
    )
    return result


def check_workflow_wiring(deepseek_config: dict[str, Any]) -> dict[str, Any]:
    """Ask n8n whether the persona's workflow still exists and still points
    at the credential id we have on file.

    This is the strongest check available without either exposing the raw
    DeepSeek key or triggering a live execution: n8n's public API has no
    read endpoint for credentials themselves (GET by id and list both
    return 405 â€” deliberately, credentials are write-only over the API).
    So a workflow node can keep referencing a credential id that was
    deleted elsewhere in n8n and this check will still report "ok" â€” it
    only catches the workflow being deleted or the node's own credential
    reference having drifted from what we have stored. Confirmed live
    2026-08-02: this exact class of failure (credential silently gone,
    node reference unchanged) only surfaced by actually triggering the
    workflow and reading n8n's execution error â€” there is no safe way to
    detect it proactively without that side effect.
    """
    workflow_id = str(deepseek_config.get("n8n_workflow_id") or "")
    credential_id = str(deepseek_config.get("n8n_credential_id") or "")
    diagnostics: dict[str, Any] = {
        "workflow_id": workflow_id or None,
        "template": deepseek_config.get("workflow_template") or None,
        "checks": {},
    }
    if not workflow_id or not credential_id:
        return {
            "ok": False,
            "reason": "DeepSeek nao provisionado (sem workflow ou credential id).",
            "diagnostics": diagnostics,
        }
    workflow = n8n_client.get_workflow(workflow_id)
    if workflow is None:
        return {
            "ok": False,
            "reason": f"Workflow n8n {workflow_id} nao existe mais.",
            "diagnostics": diagnostics,
        }
    diagnostics["checks"]["workflow_exists"] = True
    if not workflow.get("active"):
        diagnostics["checks"]["active"] = False
        return {
            "ok": False,
            "reason": "Workflow n8n existe mas esta inativo.",
            "diagnostics": diagnostics,
        }
    diagnostics["checks"]["active"] = True
    nodes = workflow.get("nodes") or []
    node_ids = {str(node.get("id") or "") for node in nodes}
    missing_nodes = sorted(_REQUIRED_NODE_IDS - node_ids)
    diagnostics["checks"]["required_nodes"] = not missing_nodes
    diagnostics["missing_nodes"] = missing_nodes
    if missing_nodes:
        return {
            "ok": False,
            "reason": "Workflow n8n nao corresponde ao template agentic canonico.",
            "diagnostics": diagnostics,
        }
    inbound = next((node for node in nodes if node.get("id") == "inbound"), {})
    live_path = str((inbound.get("parameters") or {}).get("path") or "").strip("/")
    expected_path = str(
        deepseek_config.get("conversation_webhook_path") or ""
    ).strip("/")
    diagnostics["checks"]["webhook_path"] = not expected_path or live_path == expected_path
    diagnostics["live_webhook_path"] = live_path
    diagnostics["expected_webhook_path"] = expected_path or None
    if expected_path and live_path != expected_path:
        return {
            "ok": False,
            "reason": "Webhook do workflow n8n nao corresponde a persona configurada.",
            "diagnostics": diagnostics,
        }
    referenced_ids = {
        str((node.get("credentials") or {}).get("httpHeaderAuth", {}).get("id") or "")
        for node in nodes
        if node.get("id") in {"deepseek", "deepseek_repair"}
    }
    diagnostics["checks"]["credential_reference"] = credential_id in referenced_ids
    if credential_id not in referenced_ids:
        return {
            "ok": False,
            "reason": "O node DeepSeek do workflow n8n referencia uma credencial diferente da configurada.",
            "diagnostics": diagnostics,
        }
    expected_checksum = str(deepseek_config.get("workflow_checksum") or "")
    live_checksum = _workflow_checksum(workflow)
    diagnostics["live_workflow_checksum"] = live_checksum
    diagnostics["expected_workflow_checksum"] = expected_checksum or None
    diagnostics["checks"]["workflow_checksum"] = (
        not expected_checksum or live_checksum == expected_checksum
    )
    if expected_checksum and live_checksum != expected_checksum:
        return {
            "ok": False,
            "reason": "Workflow n8n ativo divergiu do template publicado.",
            "diagnostics": diagnostics,
        }
    return {"ok": True, "reason": None, "diagnostics": diagnostics}


def check_binding(
    *,
    persona: dict[str, Any],
    binding: dict[str, Any],
    deepseek_config: dict[str, Any],
    n8n_base_url: str,
) -> dict[str, Any]:
    """Validate the persisted transport binding against the provisioned workflow."""
    metadata = binding.get("metadata") or {}
    expected_workflow_id = str(deepseek_config.get("n8n_workflow_id") or "")
    expected_path = str(
        deepseek_config.get("conversation_webhook_path") or ""
    ).strip("/")
    expected_url = f"{n8n_base_url.rstrip('/')}/webhook/{expected_path}"
    checks = {
        "active": bool(binding.get("active")),
        "persona_id": str(binding.get("persona_id") or "")
        == str(persona.get("id") or ""),
        "decision_owner": metadata.get("decision_owner") == "n8n_agents",
        "pipeline_contract": metadata.get("pipeline_contract")
        == str(deepseek_config.get("pipeline_contract") or "conversation_v3"),
        "runtime_version": metadata.get("runtime_version")
        == str(deepseek_config.get("runtime_version") or "graph_agent_runtime_v3"),
        "workflow_id": str(binding.get("n8n_workflow_id") or "")
        == expected_workflow_id,
        "webhook_url": str(metadata.get("conversation_webhook_url") or "")
        == expected_url,
    }
    failed = [name for name, ok in checks.items() if not ok]
    diagnostics = {
        "binding_id": binding.get("id"),
        "workflow_id": expected_workflow_id or None,
        "expected_webhook_url": expected_url,
        "checks": checks,
        "failed_checks": failed,
    }
    if failed:
        return {
            "ok": False,
            "reason": "Binding n8n invalido: " + ", ".join(failed),
            "diagnostics": diagnostics,
        }
    return {"ok": True, "reason": None, "diagnostics": diagnostics}


def revoke(config: dict[str, Any] | None) -> None:
    credential_id = str((config or {}).get("n8n_credential_id") or "")
    if credential_id:
        n8n_client.delete_credential(credential_id)

