"""Audit and cleanup helper for the local, real Baita WhatsApp demonstration."""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services import auth_service, graph_json_v2_store, supabase_client
from services.whatsapp_providers import get_provider


def _require_qa() -> None:
    if (os.environ.get("ENVIRONMENT") or "").strip().lower() != "qa":
        raise SystemExit("The Baita demo helper requires ENVIRONMENT=qa.")


def _persona() -> dict[str, Any]:
    persona = supabase_client.get_persona("baita-conveniencia")
    if not persona:
        raise RuntimeError("Persona baita-conveniencia not found.")
    return persona


def _snapshot(email: str) -> dict[str, Any]:
    persona = _persona()
    current = graph_json_v2_store.load_current(persona["slug"])
    if not current:
        raise RuntimeError(
            "Baita has no sufficient published graph evidence; refusing to switch its channel."
        )
    graph_version, graph = current
    nodes = list(graph.nodes or [])
    evidence = [
        node for node in nodes
        if node.node_type.lower() != "persona"
        and (
            str(node.label or "").strip()
            or str(node.data.get("markdown") or "").strip()
            or str(node.data.get("summary") or "").strip()
            or str(node.data.get("content") or "").strip()
        )
    ]
    if (
        graph.status != "published"
        or not graph.validation.is_valid
        or len(evidence) < 1
        or len(graph.edges) < 1
    ):
        raise RuntimeError(
            "Baita has no sufficient published graph evidence; refusing to switch its channel."
        )
    client = supabase_client.get_client()
    bindings = (
        client.table("workflow_bindings").select("*")
        .eq("persona_id", persona["id"]).eq("channel", "whatsapp")
        .execute().data or []
    )
    if any(row.get("provider") == "evolution_baileys" for row in bindings):
        raise RuntimeError("An Evolution binding already exists; run cleanup or audit it first.")
    meta = [
        {
            "id": row["id"],
            "active": bool(row.get("active")),
            "connection_status": row.get("connection_status"),
        }
        for row in bindings
        if row.get("provider") == "meta_cloud" and row.get("whatsapp_phone_number_id")
    ]
    if not any(row["active"] for row in meta):
        raise RuntimeError("Baita has no active configured Meta binding to preserve.")
    existing = (
        client.table("app_users").select("id").eq("email", email.strip().lower())
        .execute().data or []
    )
    if existing:
        raise RuntimeError("The demo email already belongs to an account; use a new email.")
    return {
        "persona_id": persona["id"],
        "persona_slug": persona["slug"],
        "email": email.strip().lower(),
        "meta_bindings": meta,
        "published_evidence_count": len(evidence),
        "graph_version": graph_version,
        "graph_checksum": graph_json_v2_store.checksum_graph(graph),
    }


def snapshot(email: str) -> None:
    print(json.dumps(_snapshot(email), separators=(",", ":")))


def prepare(email: str) -> None:
    state = _snapshot(email)
    client = supabase_client.get_client()
    temporary_password = secrets.token_urlsafe(18)
    created = client.table("app_users").insert({
        "email": state["email"],
        "username": None,
        "name": "Gestor Demo Baita",
        "role": "user",
        "account_type": "client",
        "must_change_password": True,
        "password_hash": auth_service.hash_password(temporary_password),
        "is_active": True,
    }).execute().data or []
    if not created:
        raise RuntimeError("Could not create the disposable Baita manager.")
    user = created[0]
    try:
        client.table("user_persona_access").insert({
            "user_id": user["id"],
            "client_id": state["persona_slug"],
            "persona_id": state["persona_id"],
            "persona_slug": state["persona_slug"],
            "can_view": True,
            "can_edit": True,
            "can_manage": True,
        }).execute()
    except Exception:
        client.table("app_users").delete().eq("id", user["id"]).execute()
        raise
    state["created_user_id"] = user["id"]
    state["portal_url"] = f"/clientes/{state['persona_slug']}/mensagens"
    state["temporary_password"] = temporary_password
    print(json.dumps(state, separators=(",", ":")))


def resume(state: dict[str, Any]) -> None:
    persona = _persona()
    if state.get("persona_id") != persona["id"] or state.get("persona_slug") != persona["slug"]:
        raise RuntimeError("Resume state does not match Baita.")
    user_id = state.get("created_user_id")
    email = str(state.get("email") or "").strip().lower()
    if not user_id or not email:
        raise RuntimeError("Resume state has no disposable manager.")
    temporary_password = secrets.token_urlsafe(18)
    rows = (
        supabase_client.get_client().table("app_users").update({
            "password_hash": auth_service.hash_password(temporary_password),
            "must_change_password": True,
        }).eq("id", user_id).eq("email", email).execute().data or []
    )
    if not rows:
        raise RuntimeError("Disposable manager no longer exists.")
    print(json.dumps({**state, "temporary_password": temporary_password}, separators=(",", ":")))


def cleanup(state: dict[str, Any]) -> None:
    persona = _persona()
    if state.get("persona_id") != persona["id"] or state.get("persona_slug") != persona["slug"]:
        raise RuntimeError("Cleanup state does not match Baita.")
    client = supabase_client.get_client()
    evolution = (
        client.table("workflow_bindings").select("*")
        .eq("persona_id", persona["id"]).eq("provider", "evolution_baileys")
        .execute().data or []
    )
    provider_errors = []
    for binding in evolution:
        try:
            get_provider("evolution_baileys").delete_instance(binding)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                provider_errors.append(f"HTTP {exc.response.status_code}")
        except Exception as exc:
            provider_errors.append(type(exc).__name__)
        binding_id = binding["id"]
        client.table("messages").delete().eq("channel_binding_id", binding_id).execute()
        client.table("lead_buffer").delete().eq("channel_binding_id", binding_id).execute()
        client.table("leads").delete().eq("channel_binding_id", binding_id).execute()
        client.table("workflow_bindings").delete().eq("id", binding_id).execute()
    for item in state.get("meta_bindings") or []:
        client.table("workflow_bindings").update({
            "active": bool(item.get("active")),
            "connection_status": item.get("connection_status"),
        }).eq("id", item["id"]).eq("persona_id", persona["id"]).execute()
    user_id = state.get("created_user_id")
    if user_id:
        client.table("user_persona_access").delete().eq("user_id", user_id).execute()
        client.table("app_users").delete().eq("id", user_id).eq("email", state.get("email")).execute()
    print(json.dumps({
        "ok": not provider_errors,
        "deleted_evolution_bindings": len(evolution),
        "restored_meta_bindings": len(state.get("meta_bindings") or []),
        "deleted_invite": bool(user_id),
        "provider_cleanup_errors": provider_errors,
    }, separators=(",", ":")))


def main() -> None:
    _require_qa()
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["snapshot", "prepare", "resume", "cleanup"])
    parser.add_argument("--email")
    args = parser.parse_args()
    if args.action in {"snapshot", "prepare"}:
        if not args.email:
            raise SystemExit("--email is required.")
        (prepare if args.action == "prepare" else snapshot)(args.email)
        return
    raw = sys.stdin.read()
    if not raw:
        raise SystemExit(f"{args.action} state is required on stdin.")
    (resume if args.action == "resume" else cleanup)(json.loads(raw))


if __name__ == "__main__":
    main()

