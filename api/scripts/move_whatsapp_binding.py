"""Move an active WhatsApp binding (meta_cloud or evolution_baileys) from one
persona to another, in place.

This re-parents the existing workflow_bindings row (same id, same encrypted
credential, same phone/instance identity) rather than creating a new one, so
the credential is never re-entered, re-encrypted, or exposed as plaintext
anywhere.

Re-parenting persona_id alone is NOT enough: every persona-scoped routing
field (decision_owner, conversation_webhook_url, n8n_workflow_id) must move
with it. This script therefore requires an explicit routing strategy and
updates binding ownership, routing and affected leads in one database RPC.

Every lead still pointing at the moved binding (channel_binding_id) is
reparented to belong to a persona it no longer matches -- every outbound
send to it then 403s (whatsapp_outbox.resolve_lead_binding) and every write
to it then hard-fails with Postgres error 23514 (trigger
enforce_lead_channel_binding, migration 067). This script clears
channel_binding_id on those leads (never deletes the row) so they stop
error-looping; a lead still receiving real inbound traffic on the OLD
channel_binding_id will self-heal onto whatever binding the source persona
activates next.

See .agents/skills/whatsapp-binding-reassignment/SKILL.md for the full
procedure this script is one step of.

Dry-run by default; pass --apply to actually write.

Usage (on the VPS, inside the api container):
  docker compose --env-file .env.compose exec -T api \
    python scripts/move_whatsapp_binding.py \
    --from-persona-slug source-persona --to-persona-slug target-persona \
    --routing deterministic --apply

If the source persona has more than one active WhatsApp binding, pass
--provider {meta_cloud|evolution_baileys} to disambiguate.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = API_DIR.parent
for path in (API_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from services import supabase_client

_STALE_ROUTING_KEYS = (
    "conversation_webhook_url", "n8n_outbound_webhook_url", "model",
    "model_endpoint", "runtime_version", "reply_source", "provider_version",
    "workflow_version",
)


def _active_whatsapp_binding(persona_id: str, provider: str | None) -> dict | None:
    candidates = [
        b for b in supabase_client.get_workflow_bindings(persona_id)
        if b.get("channel") == "whatsapp" and b.get("active")
    ]
    if provider:
        candidates = [b for b in candidates if b.get("provider") == provider]
    if len(candidates) > 1:
        found = sorted({str(b.get("provider")) for b in candidates})
        raise SystemExit(
            f"persona has {len(candidates)} active WhatsApp bindings ({found}); "
            "pass --provider to disambiguate"
        )
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-persona-slug", required=True)
    parser.add_argument("--to-persona-slug", required=True)
    parser.add_argument("--provider", choices=["meta_cloud", "evolution_baileys"], default=None)
    parser.add_argument("--workflow-name", default=None, help="Optional rename; keeps the original if omitted.")
    parser.add_argument(
        "--routing",
        required=True,
        choices=["deterministic", "clone"],
        help="Choose the target routing explicitly; never retain source-persona routing.",
    )
    parser.add_argument(
        "--routing-binding-id",
        help="Target persona binding whose complete routing is cloned when --routing=clone.",
    )
    parser.add_argument("--apply", action="store_true", help="Without this flag, only prints the plan.")
    args = parser.parse_args()

    source_persona = supabase_client.get_persona(args.from_persona_slug)
    target_persona = supabase_client.get_persona(args.to_persona_slug)
    if not source_persona:
        raise SystemExit(f"persona not found: {args.from_persona_slug}")
    if not target_persona:
        raise SystemExit(f"persona not found: {args.to_persona_slug}")

    source_binding = _active_whatsapp_binding(source_persona["id"], args.provider)
    if not source_binding:
        raise SystemExit(f"no active WhatsApp binding found for {args.from_persona_slug}")

    old_metadata = dict(source_binding.get("metadata") or {})
    routing_update: dict = {}
    if args.routing == "deterministic":
        target_metadata = {
            **old_metadata,
            "decision_owner": "deterministic",
            "conversation_mode": "deterministic",
            "transport_mode": "provider_direct",
            "pipeline_contract": "conversation_v1",
        }
        for key in _STALE_ROUTING_KEYS:
            target_metadata.pop(key, None)
        routing_update = {"metadata": target_metadata, "n8n_workflow_id": None}
    else:
        if not args.routing_binding_id:
            raise SystemExit("--routing-binding-id is required when --routing=clone")
        reference = supabase_client.get_workflow_binding_by_id(args.routing_binding_id) or {}
        if str(reference.get("persona_id") or "") != str(target_persona["id"]):
            raise SystemExit("routing reference binding must already belong to the target persona")
        if reference.get("channel") != "whatsapp":
            raise SystemExit("routing reference binding must be a WhatsApp binding")
        reference_metadata = dict(reference.get("metadata") or {})
        if not reference_metadata.get("decision_owner"):
            raise SystemExit("routing reference binding has no decision_owner")
        routing_update = {
            "metadata": reference_metadata,
            "n8n_workflow_id": reference.get("n8n_workflow_id"),
        }

    existing_target_binding = next(
        (
            b for b in supabase_client.get_workflow_bindings(target_persona["id"])
            if b.get("channel") == "whatsapp" and b.get("provider") == source_binding.get("provider")
        ),
        None,
    )

    affected_leads = (
        supabase_client.get_client().table("leads").select("id,nome,stage")
        .eq("persona_id", source_persona["id"])
        .eq("channel_binding_id", source_binding["id"])
        .execute()
    ).data or []

    plan = {
        "binding_id": source_binding["id"],
        "provider": source_binding.get("provider"),
        "from_persona_slug": args.from_persona_slug,
        "to_persona_slug": args.to_persona_slug,
        "target_already_has_binding_of_same_provider": bool(existing_target_binding),
        "leads_to_detach": [lead["id"] for lead in affected_leads],
        "routing": args.routing,
        "routing_binding_id": args.routing_binding_id,
        "target_decision_owner": (routing_update.get("metadata") or {}).get("decision_owner"),
    }
    if existing_target_binding and existing_target_binding.get("active"):
        plan["warning"] = (
            "Target persona already has an ACTIVE binding of this provider; "
            "the unique index (persona_id, channel) WHERE active will reject "
            "this write. Move that binding away first."
        )
    elif existing_target_binding:
        plan["warning"] = (
            "Target persona already has an inactive binding row of this "
            "provider; moving the source row in will leave two rows. Review "
            "before --apply."
        )

    if not args.apply:
        print(json.dumps({**plan, "dry_run": True}, ensure_ascii=False, indent=2))
        return

    client = supabase_client.get_client()
    update_fields: dict = {
        "persona_id": target_persona["id"],
        **routing_update,
    }
    if args.workflow_name:
        update_fields["workflow_name"] = args.workflow_name

    result = (
        client.table("workflow_bindings")
        .update(update_fields)
        .eq("id", source_binding["id"])
        .execute()
    )

    detached_lead_ids: list[int] = []
    for lead in affected_leads:
        current_result = (
            client.table("leads").select("metadata").eq("id", lead["id"]).maybe_single().execute()
        )
        current = current_result.data or {} if current_result else {}
        new_metadata = {
            **(current.get("metadata") or {}),
            "channel_reassignment": {
                "previous_binding_id": source_binding["id"],
                "reason": f"binding_moved_to_{args.to_persona_slug}",
                "at": datetime.now(timezone.utc).isoformat(),
            },
        }
        supabase_client.update_lead(lead["id"], {
            "channel_binding_id": None,
            "metadata": new_metadata,
        })
        detached_lead_ids.append(lead["id"])

    supabase_client.insert_event(
        {
            "event_type": "whatsapp.binding_moved",
            "entity_type": "workflow_binding",
            "entity_id": source_binding["id"],
            "persona_id": target_persona["id"],
            "payload": {
                "from_persona_slug": args.from_persona_slug,
                "to_persona_slug": args.to_persona_slug,
                "provider": source_binding.get("provider"),
                "detached_lead_ids": detached_lead_ids,
            },
        },
        source="scripts.move_whatsapp_binding",
    )
    # The source persona no longer has this channel; the target's routing
    # still needs an explicit follow-up script (see the reminder above).
    supabase_client.update_persona_routing(args.from_persona_slug, {"process_mode": "internal"})
    supabase_client.update_persona_routing(args.to_persona_slug, {"process_mode": "internal"})

    print(json.dumps({
        **plan,
        "ok": True,
        "applied": True,
        "detached_lead_ids": detached_lead_ids,
        "updated_binding": result.data[0] if result.data else None,
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
