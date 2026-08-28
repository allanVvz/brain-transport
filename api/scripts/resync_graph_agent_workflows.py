"""Resync personas from the one canonical n8n conversation template.

Inactive personas receive the exact same structural workflow but remain
deactivated. Only slugs passed with ``--active-persona`` may update their
binding to Graph Agent Runtime v3.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_ROOT))

from services import deepseek_n8n_service, n8n_client, supabase_client


def _binding(persona: dict[str, Any], *, require_active: bool) -> dict[str, Any]:
    bindings = supabase_client.get_workflow_bindings(str(persona["id"]))
    active = [row for row in bindings if row.get("active")]
    if require_active and len(active) != 1:
        raise RuntimeError(
            f"expected one active binding for {persona['slug']}, found {len(active)}"
        )
    if active:
        return active[0]
    # A human-only persona may intentionally have no active transport while
    # retaining an inactive audit workflow. Rendering that workflow needs only
    # its integration config; it must not create or activate a binding.
    return next(
        (row for row in bindings if row.get("n8n_workflow_id")),
        bindings[0] if bindings else {},
    )


def _resolved_config(
    persona: dict[str, Any], connection: dict[str, Any], binding: dict[str, Any]
) -> dict[str, Any]:
    config = dict(connection.get("config_json") or {})
    metadata = dict(binding.get("metadata") or {})
    config.setdefault("n8n_workflow_id", binding.get("n8n_workflow_id"))
    config.setdefault("model", metadata.get("model") or metadata.get("model_name"))
    config.setdefault(
        "endpoint", metadata.get("model_endpoint") or metadata.get("endpoint")
    )
    config.setdefault("reply_source", metadata.get("reply_source") or config.get("model"))
    if not config.get("endpoint") and config.get("n8n_workflow_id"):
        workflow = n8n_client.get_workflow(str(config["n8n_workflow_id"])) or {}
        workflow_binding = (workflow.get("meta") or {}).get("binding") or {}
        config.setdefault("endpoint", workflow_binding.get("endpoint"))
        for node in workflow.get("nodes") or []:
            if node.get("id") not in {"deepseek", "deepseek_repair"}:
                continue
            live_url = str((node.get("parameters") or {}).get("url") or "").strip()
            if live_url.startswith("https://"):
                config["endpoint"] = live_url
                break
    missing = [
        key for key in ("n8n_credential_id", "n8n_workflow_id", "model", "endpoint")
        if not config.get(key)
    ]
    if missing:
        raise RuntimeError(
            f"audited model binding incomplete for {persona['slug']}: {','.join(missing)}"
        )
    return config


def _binding_update(
    persona: dict[str, Any], binding: dict[str, Any], result: dict[str, Any]
) -> str:
    metadata = dict(binding.get("metadata") or {})
    metadata.update({
        "decision_owner": "n8n_agents",
        "conversation_mode": "n8n_agents",
        "runtime_version": "graph_agent_runtime_v3",
        "pipeline_contract": "conversation_v3",
        "transport_mode": "provider_direct",
        "conversation_webhook_url": (
            f"{os.environ['N8N_BASE_URL'].rstrip('/')}/webhook/"
            f"{result['conversation_webhook_path'].strip('/')}"
        ),
        "model": result.get("model"),
        "model_endpoint": result.get("endpoint"),
        "reply_source": result.get("reply_source"),
    })
    supabase_client.get_client().table("workflow_bindings").update({
        "metadata": metadata,
        "n8n_workflow_id": result["n8n_workflow_id"],
    }).eq("id", binding["id"]).execute()
    return str(binding["id"])


def run(slugs: list[str], *, active_personas: set[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    prepared: list[
        tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], bool]
    ] = []
    for slug in slugs:
        persona = supabase_client.get_persona(slug)
        if not persona:
            raise RuntimeError(f"persona not found: {slug}")
        active = slug in active_personas
        if active and not supabase_client.get_active_graph_publication(str(persona["id"])):
            raise RuntimeError(f"active GraphRAG publication required for {slug}")
        connection = supabase_client.get_persona_integration_connection(
            str(persona["id"]), "deepseek"
        ) or {}
        binding = _binding(persona, require_active=active)
        config = _resolved_config(persona, connection, binding)
        prepared.append((persona, connection, binding, config, active))

    for persona, connection, binding, config, active in prepared:
        slug = str(persona["slug"])
        synced = deepseek_n8n_service.resync_workflow_for_persona(
            persona, config, activate_workflow=active
        )
        supabase_client.save_persona_integration_connection({
            **connection,
            "persona_id": persona["id"],
            "service": "deepseek",
            "config_json": synced,
        })
        binding_id = _binding_update(persona, binding, synced) if active else None
        supabase_client.insert_event({
            "event_type": "graph_v3.n8n_template_resynced",
            "entity_type": "persona", "entity_id": str(persona["id"]),
            "persona_id": persona["id"],
            "payload": {
                "workflow_id": synced["n8n_workflow_id"],
                "workflow_checksum": synced["workflow_checksum"],
                "workflow_active": active,
                "binding_id": binding_id,
            },
        }, source="scripts.resync_graph_agent_workflows")
        results.append({
            "persona_slug": slug,
            "workflow_id": synced["n8n_workflow_id"],
            "workflow_checksum": synced["workflow_checksum"],
            "workflow_template": synced["workflow_template"],
            "workflow_active": active,
            "binding_id": binding_id,
        })
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("personas", nargs="+", help="Persona slugs to resync")
    parser.add_argument("--active-persona", action="append", default=[])
    args = parser.parse_args()
    unknown = set(args.active_persona) - set(args.personas)
    if unknown:
        parser.error(f"active personas not listed for resync: {sorted(unknown)}")
    print(json.dumps(
        run(args.personas, active_personas=set(args.active_persona)),
        ensure_ascii=False, indent=2, default=str,
    ))

