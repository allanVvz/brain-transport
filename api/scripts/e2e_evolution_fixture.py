"""Prepare and clean the local-only Evolution E2E fixture.

This script is intentionally available only as a Docker/QA orchestration helper.
It never prints credentials or complete workflow bindings. The caller keeps the
small restore snapshot in memory and sends it back through stdin during cleanup.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services import supabase_client
from services.whatsapp_providers import get_provider


def _require_qa() -> None:
    if (os.environ.get("ENVIRONMENT") or "").strip().lower() != "qa":
        raise SystemExit("Evolution E2E fixture is restricted to ENVIRONMENT=qa.")
    evolution_url = (os.environ.get("EVOLUTION_API_URL") or "").strip()
    if evolution_url != "http://evolution-api:8080":
        raise SystemExit("Evolution E2E fixture requires the local evolution-api service.")


def _persona(slug: str) -> dict[str, Any]:
    row = supabase_client.get_persona(slug)
    if not row:
        raise RuntimeError(f"Persona not found: {slug}")
    return row


def _plain_bindings(persona_id: str) -> list[dict[str, Any]]:
    return (
        supabase_client.get_client()
        .table("workflow_bindings")
        .select("*")
        .eq("persona_id", persona_id)
        .execute()
        .data
        or []
    )


def _is_meta_binding(row: dict[str, Any]) -> bool:
    provider = row.get("provider")
    return provider == "meta_cloud" or (
        provider in {None, ""}
        and bool(row.get("whatsapp_phone_number_id"))
    )


def prepare(slug: str) -> None:
    persona = _persona(slug)
    client = supabase_client.get_client()
    rows = _plain_bindings(persona["id"])
    if any(row.get("provider") == "evolution_baileys" for row in rows):
        raise RuntimeError(
            "An Evolution binding already exists for the fixture persona; "
            "refusing to reuse or overwrite it."
        )

    snapshot = {
        "persona_id": persona["id"],
        "persona_slug": persona["slug"],
        "meta_bindings": [
            {
                "id": row["id"],
                "active": row.get("active"),
                "connection_status": row.get("connection_status"),
                "updated_at": row.get("updated_at"),
            }
            for row in rows
            if row.get("active") and _is_meta_binding(row)
        ],
    }
    fixture_binding = (
        client.table("workflow_bindings")
        .insert({
            "persona_id": persona["id"],
            "workflow_name": "E2E Baita Evolution connected",
            "channel": "whatsapp",
            "provider": "evolution_baileys",
            "provider_instance_key": "e2e-baita-connected",
            "connection_status": "connected",
            "last_connection_at": datetime.now(timezone.utc).isoformat(),
            "active": True,
            "metadata": {
                "e2e_fixture": True,
                "connection_state": "synthetic_connected",
                "qr_requested": False,
            },
        })
        .execute()
        .data
        or []
    )
    if not fixture_binding:
        raise RuntimeError("Could not create the isolated connected Evolution fixture.")
    snapshot["fixture_binding_id"] = fixture_binding[0]["id"]
    # The snapshot contains only fields that cleanup may restore; it contains no provider
    # token, API key, phone-number identifier, or encrypted secret.
    print(json.dumps(snapshot, separators=(",", ":")))


def audit_message(binding_id: str, external_message_id: str) -> None:
    client = supabase_client.get_client()
    message_rows = (
        client.table("messages")
        .select("id")
        .eq("channel_binding_id", binding_id)
        .eq("external_message_id", external_message_id)
        .eq("direction", "inbound")
        .execute()
        .data
        or []
    )
    buffer_rows = (
        client.table("lead_buffer")
        .select("id")
        .eq("channel_binding_id", binding_id)
        .eq("external_message_id", external_message_id)
        .eq("direction", "inbound")
        .execute()
        .data
        or []
    )
    print(json.dumps({
        "inbound_messages": len(message_rows),
        "inbound_buffers": len(buffer_rows),
    }, separators=(",", ":")))


def _safe_fixture_email(value: str) -> str:
    email = value.strip().lower()
    if not (
        email.startswith("qa+baita-evolution-")
        and email.endswith("@example.invalid")
    ):
        raise RuntimeError("Refusing to delete a non-fixture user.")
    return email


def cleanup(slug: str, emails: list[str], snapshot: dict[str, Any]) -> None:
    persona = _persona(slug)
    if snapshot.get("persona_id") != persona["id"] or snapshot.get("persona_slug") != slug:
        raise RuntimeError("Fixture restore snapshot does not match the requested persona.")

    client = supabase_client.get_client()
    provider = get_provider("evolution_baileys")
    evolution_rows = [
        row for row in _plain_bindings(persona["id"])
        if row.get("provider") == "evolution_baileys"
    ]
    provider_errors: list[str] = []
    for row in evolution_rows:
        try:
            provider.delete_instance(row)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                provider_errors.append(f"HTTP {exc.response.status_code}")
        except Exception as exc:  # Docker teardown still removes the isolated instance volume.
            provider_errors.append(type(exc).__name__)
        client.table("workflow_bindings").delete().eq("id", row["id"]).execute()

    for item in snapshot.get("meta_bindings") or []:
        client.table("workflow_bindings").update({
            "active": item.get("active"),
            "connection_status": item.get("connection_status"),
            "updated_at": item.get("updated_at"),
        }).eq("id", item["id"]).eq("persona_id", persona["id"]).execute()

    deleted_users = 0
    for raw_email in emails:
        email = _safe_fixture_email(raw_email)
        users = (
            client.table("app_users")
            .select("id")
            .eq("email", email)
            .execute()
            .data
            or []
        )
        for user in users:
            client.table("user_persona_access").delete().eq("user_id", user["id"]).execute()
            client.table("app_users").delete().eq("id", user["id"]).execute()
            deleted_users += 1

    print(json.dumps({
        "ok": not provider_errors,
        "deleted_bindings": len(evolution_rows),
        "deleted_users": deleted_users,
        "restored_meta_bindings": len(snapshot.get("meta_bindings") or []),
        "provider_cleanup_errors": provider_errors,
    }, separators=(",", ":")))


def main() -> None:
    _require_qa()
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "audit-message", "cleanup"])
    parser.add_argument("--persona")
    parser.add_argument("--email", action="append", default=[])
    parser.add_argument("--binding-id")
    parser.add_argument("--external-message-id")
    args = parser.parse_args()

    if args.action == "prepare":
        if not args.persona:
            raise SystemExit("prepare requires --persona")
        prepare(args.persona)
        return
    if args.action == "audit-message":
        if not args.binding_id or not args.external_message_id:
            raise SystemExit(
                "audit-message requires --binding-id and --external-message-id"
            )
        audit_message(args.binding_id, args.external_message_id)
        return

    raw = sys.stdin.read()
    if not raw:
        raise SystemExit("Cleanup requires the restore snapshot on stdin.")
    if not args.persona:
        raise SystemExit("cleanup requires --persona")
    cleanup(args.persona, args.email, json.loads(raw))


if __name__ == "__main__":
    main()
