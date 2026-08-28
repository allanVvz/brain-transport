"""Read-only manifest for active SDR branches without a referential service.

This script never mutates data. The returned expected revision can be supplied
to ``repair_conversation_ledger_branch_v1(..., p_apply=true)`` only in a
separately reviewed and authorized production step.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from services import supabase_client  # noqa: E402


def build_manifest(persona_id: str | None = None) -> dict:
    query = (
        supabase_client.get_client()
        .table("conversation_ledgers")
        .select("id,persona_id,lead_ref,journey_id,active_branch_node_id,revision")
        .not_.is_("active_branch_node_id", "null")
        .order("persona_id")
        .order("lead_ref")
    )
    if persona_id:
        query = query.eq("persona_id", persona_id)
    ledgers = getattr(query.execute(), "data", None) or []
    candidates: list[dict] = []
    for ledger in ledgers:
        preview = supabase_client.get_client().rpc(
            "repair_conversation_ledger_branch_v1",
            {
                "p_ledger_id": ledger["id"],
                "p_expected_revision": ledger["revision"],
                "p_apply": False,
            },
        ).execute()
        result = getattr(preview, "data", None) or {}
        if isinstance(result, list):
            result = result[0] if result else {}
        if result.get("reason") == "active_branch_without_referential_service":
            candidates.append({
                "ledger_id": ledger["id"],
                "persona_id": ledger["persona_id"],
                "lead_ref": ledger["lead_ref"],
                "journey_id": ledger.get("journey_id"),
                "active_branch_node_id": ledger["active_branch_node_id"],
                "expected_revision": ledger["revision"],
                "proposed_action": result.get("proposed_action"),
            })
    return {
        "mode": "dry_run",
        "mutation_performed": False,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = build_manifest(args.persona_id)
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
