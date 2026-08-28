"""Configure one persona's canonical conversation binding without secrets."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


API_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = API_DIR.parent
for path in (API_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from services import supabase_client


def phone(value: str) -> str:
    normalized = re.sub(r"\D", "", value or "")
    if not normalized:
        raise argparse.ArgumentTypeError("phone must contain digits")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("persona_slug")
    parser.add_argument("workflow_name")
    parser.add_argument("--n8n-workflow-id")
    parser.add_argument("--phone-number-id", required=True)
    parser.add_argument("--conversation-webhook-url")
    parser.add_argument("--outbound-webhook-url")
    parser.add_argument("--allow", action="append", type=phone, default=[])
    parser.add_argument(
        "--mode",
        choices=("active", "test_allowlist", "disabled"),
        default="active",
    )
    parser.add_argument(
        "--conversation-mode",
        choices=("deterministic", "n8n_agents"),
        default="deterministic",
    )
    parser.add_argument(
        "--transport-mode",
        choices=("provider_direct", "n8n_adapter"),
        default="provider_direct",
    )
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()

    persona = supabase_client.get_persona(args.persona_slug)
    if not persona:
        raise SystemExit(f"persona not found: {args.persona_slug}")
    if args.mode == "test_allowlist" and not args.allow:
        raise SystemExit("test_allowlist requires at least one --allow")
    if args.conversation_mode == "n8n_agents" and not args.conversation_webhook_url:
        raise SystemExit("n8n_agents requires --conversation-webhook-url")
    if args.transport_mode == "n8n_adapter" and not args.outbound_webhook_url:
        raise SystemExit("n8n_adapter requires --outbound-webhook-url")
    binding = supabase_client.upsert_workflow_binding(
        {
            "persona_id": persona["id"],
            "workflow_name": args.workflow_name,
            "n8n_workflow_id": args.n8n_workflow_id,
            "whatsapp_phone_number_id": args.phone_number_id,
            "active": args.activate,
            "metadata": {
                "decision_owner": args.conversation_mode,
                "conversation_mode": args.conversation_mode,
                "transport_mode": args.transport_mode,
                "pipeline_contract": "conversation_v1",
                "mode": args.mode,
                "allowlist": args.allow,
                "conversation_webhook_url": args.conversation_webhook_url,
                "n8n_outbound_webhook_url": args.outbound_webhook_url,
            },
        }
    )
    supabase_client.update_persona_routing(
        args.persona_slug,
        {
            "process_mode": (
                "n8n"
                if args.conversation_mode == "n8n_agents"
                else "internal"
            )
        },
    )
    supabase_client.insert_event(
        {
            "event_type": "conversation.binding_configured",
            "entity_type": "workflow_binding",
            "entity_id": str(binding.get("id") or args.workflow_name),
            "persona_id": persona["id"],
            "payload": {
                "workflow_name": args.workflow_name,
                "active": args.activate,
                "decision_owner": args.conversation_mode,
                "transport_mode": args.transport_mode,
                "pipeline_contract": "conversation_v1",
                "mode": args.mode,
                "allowlist_count": len(args.allow),
            },
        },
        source="scripts.configure_persona_conversation",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "persona_slug": args.persona_slug,
                "binding_id": binding.get("id"),
                "active": args.activate,
                "conversation_mode": args.conversation_mode,
                "transport_mode": args.transport_mode,
                "mode": args.mode,
                "allowlist_count": len(args.allow),
            }
        )
    )


if __name__ == "__main__":
    main()

