import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from services import auth_service, supabase_client
from utils.tls import configure_trust_store


def upsert_user(args: argparse.Namespace) -> dict:
    client = supabase_client.get_client()
    payload = {
        "email": args.email.strip().lower(),
        "username": args.username.strip().lower() if args.username else None,
        "password_hash": auth_service.hash_password(args.password),
        "name": args.name,
        "role": args.role,
        "account_type": args.account_type,
        "must_change_password": args.must_change_password,
        "is_active": not args.inactive,
    }
    result = client.table("app_users").upsert(payload, on_conflict="email").execute()
    user = (result.data or [None])[0]
    if not user:
        user = client.table("app_users").select("*").eq("email", payload["email"]).maybe_single().execute().data
    if not user:
        raise RuntimeError("Could not create user")
    return user


def grant_personas(user: dict, persona_slugs: list[str], can_edit: bool, can_manage: bool) -> None:
    if not persona_slugs:
        return
    client = supabase_client.get_client()
    for slug in persona_slugs:
        persona = supabase_client.get_persona(slug)
        if not persona:
            raise RuntimeError(f"Persona not found: {slug}")
        client.table("user_persona_access").upsert(
            {
                "user_id": user["id"],
                "client_id": persona.get("client_id") or persona.get("slug"),
                "persona_id": persona["id"],
                "persona_slug": persona.get("slug"),
                "can_view": True,
                "can_edit": can_edit,
                "can_manage": can_manage,
            },
            on_conflict="user_id,persona_id",
        ).execute()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    configure_trust_store()
    parser = argparse.ArgumentParser(description="Create or update an Brain AI login user.")
    parser.add_argument("--email", default=os.environ.get("AI_BRAIN_SEED_ADMIN_EMAIL"))
    parser.add_argument("--username", default=os.environ.get("AI_BRAIN_SEED_ADMIN_USERNAME"))
    parser.add_argument("--password", default=os.environ.get("AI_BRAIN_SEED_ADMIN_PASSWORD"))
    parser.add_argument("--name", default=os.environ.get("AI_BRAIN_SEED_ADMIN_NAME") or "Brain AI Admin")
    parser.add_argument("--role", choices=["admin", "user", "viewer", "operator"], default=os.environ.get("AI_BRAIN_SEED_ADMIN_ROLE") or "admin")
    parser.add_argument("--account-type", choices=["internal", "agency", "client"], default="internal")
    parser.add_argument("--must-change-password", action="store_true")
    parser.add_argument("--persona", action="append", default=[], help="Persona slug to grant. Repeat for multiple personas.")
    parser.add_argument("--can-edit", action="store_true")
    parser.add_argument("--can-manage", action="store_true")
    parser.add_argument("--inactive", action="store_true")
    args = parser.parse_args()

    if not args.email or not args.password:
        raise SystemExit("Set --email and --password, or AI_BRAIN_SEED_ADMIN_EMAIL / AI_BRAIN_SEED_ADMIN_PASSWORD.")

    user = upsert_user(args)
    grant_personas(user, args.persona, args.can_edit, args.can_manage)
    print(f"User ready: {user['email']} ({user['role']})")


if __name__ == "__main__":
    main()
