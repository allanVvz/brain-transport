"""Point an existing binding's (meta_cloud or evolution_baileys) n8n_agents
routing at a specific webhook/workflow, without touching
credential/provider/phone or instance identity.

Unlike configure_persona_conversation.py (which upserts on
workflow_name+persona_id and could create a duplicate row, or silently drop
columns it doesn't set, depending on upsert semantics), this does a plain
column-scoped UPDATE on the known binding row -- the same safe pattern as
set_binding_deterministic.py.

This performs no existence/reachability check on the webhook URL or
workflow id you pass -- it will happily persist a value that later fails
whatsapp_outbox.validate_direct_binding's stricter runtime check (the URL
must start with f"{N8N_BASE_URL}/webhook/"). Verify those yourself before
--apply, e.g. by confirming the target persona already has a working n8n
workflow at that path (graph_agent_runtime_v3 personas: check another of
their bindings, or api/n8n-workflows/persona-conversation-template.json's
__WEBHOOK_ID__ derivation) -- don't just guess the URL shape.

If the binding you're routing needs graph_agent_runtime_v3 fields (model,
model_endpoint, runtime_version, pipeline_contract=conversation_v3,
reply_source) rather than the basic v1 contract this script writes, don't
reconstruct them field-by-field here -- clone the full metadata dict from
another binding of the same target persona that already has them.

Dry-run by default; pass --apply to actually write.

Usage (on the VPS, inside the api container):
  docker compose --env-file .env.compose exec -T api \
    python scripts/set_binding_n8n_webhook.py --persona-slug vz-lupas \
    --webhook-url http://n8n:5678/webhook/vz-lupas/conversation \
    --n8n-workflow-id <new-workflow-id-from-n8n> --apply

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona-slug", required=True)
    parser.add_argument("--webhook-url", required=True)
    parser.add_argument("--n8n-workflow-id", required=True)
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
        "decision_owner": "n8n_agents",
        "conversation_mode": "n8n_agents",
        "transport_mode": "provider_direct",
        "pipeline_contract": "conversation_v1",
        "conversation_webhook_url": args.webhook_url,
    }
    new_metadata.pop("n8n_outbound_webhook_url", None)

    plan = {
        "persona_slug": args.persona_slug,
        "binding_id": binding["id"],
        "old_decision_owner": old_metadata.get("decision_owner"),
        "old_conversation_webhook_url": old_metadata.get("conversation_webhook_url"),
        "old_n8n_workflow_id": binding.get("n8n_workflow_id"),
        "new_conversation_webhook_url": args.webhook_url,
        "new_n8n_workflow_id": args.n8n_workflow_id,
    }
    if not args.apply:
        print(json.dumps({**plan, "dry_run": True}, ensure_ascii=False, indent=2))
        return

    client = supabase_client.get_client()
    result = (
        client.table("workflow_bindings")
        .update({"metadata": new_metadata, "n8n_workflow_id": args.n8n_workflow_id})
        .eq("id", binding["id"])
        .execute()
    )
    supabase_client.update_persona_routing(args.persona_slug, {"process_mode": "n8n"})
    supabase_client.insert_event(
        {
            "event_type": "whatsapp.binding_conversation_mode_changed",
            "entity_type": "workflow_binding",
            "entity_id": binding["id"],
            "persona_id": persona["id"],
            "payload": {
                "from": old_metadata.get("decision_owner"),
                "to": "n8n_agents",
                "webhook_url": args.webhook_url,
            },
        },
        source="scripts.set_binding_n8n_webhook",
    )
    print(json.dumps({
        **plan,
        "ok": True,
        "applied": True,
        "updated_binding": result.data[0] if result.data else None,
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
