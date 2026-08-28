"""Switch an existing WhatsApp binding's (meta_cloud or evolution_baileys)
conversation routing to the deterministic pipeline (persona_slug-driven,
reads the persona's own Graph JSON) instead of an n8n workflow webhook.

Use this after move_whatsapp_binding.py when the moved binding still
carries a decision_owner=n8n_agents / conversation_webhook_url pointing at
another persona's n8n workflow. Only touches metadata + the n8n_workflow_id
column; credential, instance/phone identity and connection_status are
untouched.

This is a FULL reset, not a partial one: it also drops graph_agent_runtime_v3
routing keys (model, model_endpoint, runtime_version, reply_source,
provider_version) and any stale workflow_version tag from a previous owner,
so nothing "dead" lingers in metadata to confuse the next person reading it.

Dry-run by default; pass --apply to actually write.

Usage (on the VPS, inside the api container):
  docker compose --env-file .env.compose exec -T api \
    python scripts/set_binding_deterministic.py --persona-slug tock-fatal --apply

If the persona has more than one active WhatsApp binding, pass
--provider {meta_cloud|evolution_baileys} to disambiguate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = API_DIR.parent
for path in (API_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from services import supabase_client

_STALE_ROUTING_KEYS = (
    "conversation_webhook_url", "n8n_outbound_webhook_url",
    "model", "model_endpoint", "runtime_version", "reply_source",
    "provider_version", "workflow_version",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona-slug", required=True)
    parser.add_argument("--provider", choices=["meta_cloud", "evolution_baileys"], default=None)
    parser.add_argument("--apply", action="store_true", help="Without this flag, only prints the plan.")
    args = parser.parse_args()

    persona = supabase_client.get_persona(args.persona_slug)
    if not persona:
        raise SystemExit(f"persona not found: {args.persona_slug}")

    candidates = [
        b for b in supabase_client.get_workflow_bindings(persona["id"])
        if b.get("channel") == "whatsapp" and b.get("active")
    ]
    if args.provider:
        candidates = [b for b in candidates if b.get("provider") == args.provider]
    if len(candidates) > 1:
        raise SystemExit(
            f"persona has {len(candidates)} active WhatsApp bindings; pass --provider"
        )
    binding = candidates[0] if candidates else None
    if not binding:
        raise SystemExit(f"no active WhatsApp binding found for {args.persona_slug}")

    old_metadata = dict(binding.get("metadata") or {})
    new_metadata = {
        **old_metadata,
        "decision_owner": "deterministic",
        "conversation_mode": "deterministic",
        "transport_mode": "provider_direct",
        "pipeline_contract": "conversation_v1",
    }
    for key in _STALE_ROUTING_KEYS:
        new_metadata.pop(key, None)

    plan = {
        "persona_slug": args.persona_slug,
        "binding_id": binding["id"],
        "old_decision_owner": old_metadata.get("decision_owner"),
        "old_conversation_webhook_url": old_metadata.get("conversation_webhook_url"),
        "old_n8n_workflow_id": binding.get("n8n_workflow_id"),
        "new_decision_owner": "deterministic",
    }
    if not args.apply:
        print(json.dumps({**plan, "dry_run": True}, ensure_ascii=False, indent=2))
        return

    client = supabase_client.get_client()
    result = (
        client.table("workflow_bindings")
        .update({"metadata": new_metadata, "n8n_workflow_id": None})
        .eq("id", binding["id"])
        .execute()
    )
    supabase_client.update_persona_routing(args.persona_slug, {"process_mode": "internal"})
    supabase_client.insert_event(
        {
            "event_type": "whatsapp.binding_conversation_mode_changed",
            "entity_type": "workflow_binding",
            "entity_id": binding["id"],
            "persona_id": persona["id"],
            "payload": {
                "from": old_metadata.get("decision_owner"),
                "to": "deterministic",
            },
        },
        source="scripts.set_binding_deterministic",
    )
    print(json.dumps({
        **plan,
        "ok": True,
        "applied": True,
        "updated_binding": result.data[0] if result.data else None,
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

