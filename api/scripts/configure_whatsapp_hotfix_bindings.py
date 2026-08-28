"""Apply the production-safe Meta/Evolution WhatsApp binding contract.

This script never prints or rotates credentials.  The existing Meta token is
copied from the process environment into its binding using the configured
secret-store key.  Evolution keeps its existing encrypted credential.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


API_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = API_DIR.parent
for path in (API_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from services import secret_store, supabase_client


def _binding(persona_slug: str, provider: str) -> tuple[dict, dict]:
    persona = supabase_client.get_persona(persona_slug) or {}
    if not persona.get("id"):
        raise RuntimeError(f"persona not found: {persona_slug}")
    matches = [
        row
        for row in supabase_client.get_workflow_bindings(persona["id"])
        if row.get("provider") == provider and row.get("active")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{persona_slug} requires exactly one active {provider} binding"
        )
    return persona, matches[0]


def _canonical_metadata(binding: dict) -> dict:
    metadata = {
        **(binding.get("metadata") or {}),
        "decision_owner": "deterministic",
        "conversation_mode": "deterministic",
        "transport_mode": "provider_direct",
        "pipeline_contract": "conversation_v1",
    }
    metadata.pop("outbound_webhook_url", None)
    metadata.pop("n8n_outbound_webhook_url", None)
    metadata.pop("conversation_webhook_url", None)
    return metadata


def _is_complete_n8n_agents_binding(binding: dict) -> bool:
    """True only for a binding already fully, validly configured for the
    agentic engine — never for a half-configured one (e.g. decision_owner
    flipped to n8n_agents without a workflow id or webhook), which stays
    exactly as dangerous as before and must still be reset to the safe
    deterministic baseline."""
    metadata = binding.get("metadata") or {}
    return (
        metadata.get("decision_owner") == "n8n_agents"
        and bool(binding.get("n8n_workflow_id"))
        and bool(str(metadata.get("conversation_webhook_url") or "").strip())
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-persona", required=True)
    parser.add_argument("--evolution-persona", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    meta_persona, meta_binding = _binding(args.meta_persona, "meta_cloud")
    evolution_persona, evolution_binding = _binding(
        args.evolution_persona,
        "evolution_baileys",
    )

    meta_token = (os.environ.get("META_WHATSAPP_ACCESS_TOKEN") or "").strip()
    if not meta_token:
        raise RuntimeError("META_WHATSAPP_ACCESS_TOKEN is not configured")
    meta_ciphertext = meta_binding.get("provider_secret_ciphertext")
    if meta_ciphertext:
        if secret_store.decrypt_secret(meta_ciphertext) != meta_token:
            raise RuntimeError(
                "Meta binding credential differs from the preserved token"
            )
    else:
        meta_ciphertext = secret_store.encrypt_secret(meta_token)

    evolution_ciphertext = evolution_binding.get("provider_secret_ciphertext")
    if (
        not evolution_ciphertext
        or not secret_store.decrypt_secret(evolution_ciphertext)
    ):
        raise RuntimeError("Evolution binding credential is unavailable")

    meta_preserved = _is_complete_n8n_agents_binding(meta_binding)
    evolution_preserved = _is_complete_n8n_agents_binding(evolution_binding)

    if args.apply:
        # Either binding only gets reset if it is NOT already a complete,
        # valid n8n_agents configuration — a half-configured one (missing
        # workflow id or webhook) is exactly as dangerous as before and
        # still gets reset. Confirmed live 2026-08-01: this reset used to
        # unconditionally revert Aurora's decision_owner on every deploy,
        # silently undoing an intentional activation of the agentic flow.
        # Confirmed live again 2026-08-02: Meta/Baita had the same
        # unconditional reset with no exception at all — "Meta never uses
        # the agentic engine" stopped being true the moment the settings
        # UI let an operator switch any persona to n8n_agents, and this
        # script silently undid that choice on the very next deploy.
        if not meta_preserved:
            supabase_client.update_workflow_binding(
                meta_binding["id"],
                {
                    "channel": "whatsapp",
                    "provider": "meta_cloud",
                    "n8n_workflow_id": None,
                    "provider_secret_ciphertext": meta_ciphertext,
                    "metadata": _canonical_metadata(meta_binding),
                },
            )
        else:
            supabase_client.update_workflow_binding(
                meta_binding["id"],
                {"provider_secret_ciphertext": meta_ciphertext},
            )
        if not evolution_preserved:
            supabase_client.update_workflow_binding(
                evolution_binding["id"],
                {
                    "channel": "whatsapp",
                    "provider": "evolution_baileys",
                    "n8n_workflow_id": None,
                    "metadata": _canonical_metadata(evolution_binding),
                },
            )
        for persona, binding, preserved in (
            (meta_persona, meta_binding, meta_preserved),
            (evolution_persona, evolution_binding, evolution_preserved),
        ):
            supabase_client.insert_event(
                {
                    "event_type": "whatsapp.binding_hotfix_configured",
                    "entity_type": "workflow_binding",
                    "entity_id": binding["id"],
                    "persona_id": persona["id"],
                    "payload": {
                        "persona_slug": persona["slug"],
                        "provider": binding["provider"],
                        "decision_owner": (
                            "n8n_agents" if preserved else "deterministic"
                        ),
                        "transport_mode": "provider_direct",
                        "credential_preserved": True,
                        "preserved_existing_n8n_agents_config": preserved,
                    },
                },
                source="scripts.configure_whatsapp_hotfix_bindings",
            )

    print({
        "ok": True,
        "applied": args.apply,
        "bindings": [
            {
                "persona_slug": args.meta_persona,
                "binding_id": meta_binding["id"],
                "provider": "meta_cloud",
                "encrypted_secret_present": bool(meta_ciphertext),
                "preserved_existing_n8n_agents_config": meta_preserved,
            },
            {
                "persona_slug": args.evolution_persona,
                "binding_id": evolution_binding["id"],
                "provider": "evolution_baileys",
                "encrypted_secret_present": True,
                "preserved_existing_n8n_agents_config": evolution_preserved,
            },
        ],
    })


if __name__ == "__main__":
    main()
