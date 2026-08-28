"""Switch a persona's live conversation engine (mirrors PATCH /personas/{slug}/routing).

Runs the exact same n8n_agents activation path as
routes.personas.update_routing: validates the persona's DeepSeek
credential, resyncs (creates or updates) the n8n workflow via
deepseek_n8n_service.resync_workflow_for_persona(), then points every
active workflow binding at it. Exists so this switch can be applied from
the VPS without going through an authenticated HTTP session â€” the
same reason configure_whatsapp_hotfix_bindings.py exists as a script
rather than a curl call.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


API_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = API_DIR.parent
for path in (API_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from services import deepseek_n8n_service, supabase_client

_KNOWN_MODES = {"deterministic", "n8n_agents", "orquestrador"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("persona_slug")
    parser.add_argument("--mode", choices=sorted(_KNOWN_MODES), required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    persona = supabase_client.get_persona(args.persona_slug)
    if not persona:
        raise SystemExit(f"persona not found: {args.persona_slug}")

    deepseek_config: dict = {}
    if args.mode == "n8n_agents":
        integration = (
            supabase_client.get_persona_integration_connection(
                persona["id"], "deepseek"
            )
            or {}
        )
        deepseek_config = dict(integration.get("config_json") or {})
        if not integration.get("enabled") or not deepseek_config.get("n8n_credential_id"):
            raise SystemExit(
                "DeepSeek nao configurada para esta persona "
                "(enabled=false ou sem n8n_credential_id)."
            )
        if args.apply:
            deepseek_config = deepseek_n8n_service.resync_workflow_for_persona(
                persona, deepseek_config,
            )
            supabase_client.save_persona_integration_connection(
                {
                    "persona_id": persona["id"],
                    "service": "deepseek",
                    "config_json": deepseek_config,
                }
            )

    bindings = [
        b
        for b in supabase_client.get_workflow_bindings(persona["id"])
        if b.get("id") and b.get("active")
    ]
    if not bindings:
        raise SystemExit(f"no active workflow binding for {args.persona_slug}")

    planned = []
    for binding in bindings:
        metadata = {
            **(binding.get("metadata") or {}),
            "conversation_mode": args.mode,
            "decision_owner": args.mode,
            "pipeline_contract": "conversation_v1",
            "transport_mode": "provider_direct",
        }
        binding_update: dict = {"metadata": metadata}
        if args.mode == "n8n_agents":
            n8n_base = str(os.environ.get("N8N_BASE_URL") or "").rstrip("/")
            webhook_path = str(
                deepseek_config.get("conversation_webhook_path") or ""
            ).strip("/")
            if args.apply and (not n8n_base or not webhook_path):
                raise SystemExit("N8N_BASE_URL ou conversation_webhook_path ausente.")
            metadata["conversation_webhook_url"] = (
                f"{n8n_base}/webhook/{webhook_path}" if n8n_base and webhook_path else None
            )
            binding_update["n8n_workflow_id"] = deepseek_config.get("n8n_workflow_id")
        else:
            metadata.pop("conversation_webhook_url", None)
            binding_update["n8n_workflow_id"] = None
        planned.append({"binding_id": binding["id"], **binding_update})
        if args.apply:
            supabase_client.update_workflow_binding(str(binding["id"]), binding_update)

    if args.apply:
        supabase_client.update_persona_routing(
            args.persona_slug,
            {"process_mode": "n8n" if args.mode == "n8n_agents" else "internal"},
        )
        supabase_client.insert_event(
            {
                "event_type": "conversation.mode_updated",
                "entity_type": "persona",
                "entity_id": str(persona.get("id") or args.persona_slug),
                "persona_id": persona["id"],
                "payload": {
                    "persona_slug": args.persona_slug,
                    "conversation_mode": args.mode,
                    "pipeline_contract": "conversation_v1",
                },
            },
            source="scripts.switch_persona_conversation_mode",
        )

    print(
        json.dumps(
            {
                "ok": True,
                "applied": args.apply,
                "persona_slug": args.persona_slug,
                "mode": args.mode,
                "n8n_workflow_id": deepseek_config.get("n8n_workflow_id"),
                "conversation_webhook_path": deepseek_config.get("conversation_webhook_path"),
                "bindings": planned,
            },
            default=str,
        )
    )


if __name__ == "__main__":
    main()

