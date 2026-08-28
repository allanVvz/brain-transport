"""Idempotently seed Aurora's portal account and canonical Graph JSON v2.

The temporary password is generated only when the account does not exist and
is written once to an operator-selected mode-0600 file. It is never printed.
"""
from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services import (
    auth_service,
    supabase_client,
)

PERSONA_CONFIG = {
    "portal": {
        "business_model": "appointment",
        "calendar_provider": "manual",
        "conversion_stage": "fechado",
        "stage_labels": {"fechado": "Agendado"},
        "availability_policy": "always_confirm",
        "schedule_notice": "Disponibilidade e horário sempre sujeitos à confirmação.",
        "automation_mode": "ai_with_handoff",
        "responsible_team": "Equipe Aurora",
    }
}


def ensure_persona() -> dict:
    existing = supabase_client.get_persona("aurora") or {}
    config = {**(existing.get("config") or {}), **PERSONA_CONFIG}
    supabase_client.upsert_persona({
        "slug": "aurora",
        "name": "Aurora Estética Automotiva",
        "tone": "premium, objetivo, cordial",
        "products": [
            "Avaliação inicial",
            "Lavagem detalhada",
            "Higienização interna",
            "Polimento técnico",
        ],
        "config": config,
        "active": True,
    })
    persona = supabase_client.get_persona("aurora")
    if not persona:
        raise RuntimeError("Aurora persona was not persisted")
    return persona


def _write_new_credential(path_value: str, temporary_password: str) -> Path:
    output = Path(path_value)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                "email: aurora@brain.com\n"
                f"password: {temporary_password}\n"
                "must_change_password: true\n"
            )
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return output


def ensure_account(persona: dict, *, credential_output_file: str | None) -> bool:
    client = supabase_client.get_client()
    rows = (
        client.table("app_users")
        .select("id,email")
        .eq("email", "aurora@brain.com")
        .limit(1)
        .execute()
        .data
        or []
    )
    account_created = False
    if rows:
        user = rows[0]
    else:
        if not credential_output_file:
            raise RuntimeError(
                "AURORA_CREDENTIAL_OUTPUT_FILE is required before creating "
                "the one-time Aurora account."
            )
        temporary_password = f"Aurora-{secrets.token_urlsafe(12)}"
        credential_path = _write_new_credential(
            credential_output_file,
            temporary_password,
        )
        try:
            created = client.table("app_users").insert({
                "email": "aurora@brain.com",
                "username": "aurora",
                "password_hash": auth_service.hash_password(temporary_password),
                "name": "Gestor Aurora",
                "role": "user",
                "account_type": "client",
                "must_change_password": True,
                "is_active": True,
            }).execute().data or []
        except Exception:
            credential_path.unlink(missing_ok=True)
            raise
        if not created:
            credential_path.unlink(missing_ok=True)
            raise RuntimeError("Aurora account was not persisted")
        user = created[0]
        account_created = True

    client.table("user_persona_access").upsert({
        "user_id": user["id"],
        "client_id": "aurora",
        "persona_id": persona["id"],
        "persona_slug": "aurora",
        "can_view": True,
        "can_edit": True,
        "can_manage": True,
    }, on_conflict="user_id,persona_id").execute()
    return account_created


def publish_graph() -> dict:
    from scripts.publish_aurora_graph import publish

    return publish()


def main() -> None:
    credential_output_file = (
        os.environ.get("AURORA_CREDENTIAL_OUTPUT_FILE") or ""
    ).strip() or None
    persona = ensure_persona()
    account_created = ensure_account(
        persona,
        credential_output_file=credential_output_file,
    )
    result = publish_graph()
    print(f"Aurora ready: /clientes/aurora/mensagens (graph v{result['version']})")
    if account_created:
        print(
            "One-time credential written with mode 0600 to: "
            f"{credential_output_file}"
        )
        print("The password is not printed and will not be reset on subsequent runs.")
    else:
        print("Aurora account already exists; password preserved and not displayed.")


if __name__ == "__main__":
    main()
