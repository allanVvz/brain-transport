"""Add qualification metadata to existing leads without changing their stages.

Dry-run is the default. Use ``--apply`` only after the release is deployed and
the reported counts have been audited.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services import lead_qualification, supabase_client


def _object(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _state_from_logs(lead_id: str) -> tuple[dict, str, list[str]]:
    logs = supabase_client.get_agent_logs(lead_id=lead_id, limit=100) or []
    for row in logs:
        metadata = _object(row.get("metadata"))
        output = _object(row.get("output"))
        decision = _object(output.get("decision"))
        response = _object(output.get("response"))
        state = _object(response.get("cart_state"))
        intent = str(
            decision.get("intent")
            or metadata.get("intent")
            or _object(row.get("decision")).get("intent")
            or ""
        )
        evidence = (
            decision.get("evidence_node_ids")
            or metadata.get("evidence_node_ids")
            or []
        )
        if state or intent:
            return state, intent, list(evidence)
    return {}, "", []


def run(*, apply: bool = False, limit: int = 10000) -> dict:
    rows = supabase_client.get_leads(limit=limit, offset=0) or []
    stats = {"seen": len(rows), "eligible": 0, "updated": 0, "errors": []}
    for lead in rows:
        metadata = dict(lead.get("metadata") or {})
        runtime = dict(metadata.get("conversation_runtime") or {})
        state = dict(
            metadata.get("conversation_state")
            or metadata.get("vitoria_state")
            or {}
        )
        intent = str(
            runtime.get("last_intent")
            or metadata.get("last_intent")
            or "backfill"
        )
        evidence_node_ids = list(runtime.get("evidence_node_ids") or [])
        log_state, log_intent, log_evidence = _state_from_logs(str(lead.get("id") or ""))
        if not state:
            state = log_state
        if intent == "backfill" and log_intent:
            intent = log_intent
        if not evidence_node_ids:
            evidence_node_ids = log_evidence
        if not state and not runtime and not metadata.get("qualification"):
            continue
        stats["eligible"] += 1
        try:
            qualification, _ = lead_qualification.calculate(
                previous=metadata.get("qualification"),
                business_model=str(state.get("business_model") or "sales"),
                intent=intent,
                state=state,
                current_stage=lead.get("stage"),
                evidence_node_ids=evidence_node_ids,
                update_stage=False,
            )
            if apply:
                supabase_client.update_lead(
                    int(lead["id"]),
                    {"metadata": {**metadata, "qualification": qualification}},
                )
                stats["updated"] += 1
        except Exception as exc:
            stats["errors"].append({"lead_id": lead.get("id"), "error": str(exc)})
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()
    print(run(apply=args.apply, limit=args.limit))

