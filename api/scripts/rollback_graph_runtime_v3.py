"""Rollback one persona to a previous GraphRAG publication/runtime binding."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_ROOT))

from services import supabase_client


def rollback(persona_slug: str, target_version: int, *, runtime_version: str | None) -> dict:
    persona = supabase_client.get_persona(persona_slug)
    if not persona:
        raise LookupError(f"persona not found: {persona_slug}")
    activation = supabase_client.get_client().rpc("rollback_graph_publication_v3", {
        "p_persona_id": persona["id"], "p_target_version": target_version,
    }).execute().data
    bindings: list[str] = []
    for binding in supabase_client.get_workflow_bindings(str(persona["id"])):
        if not binding.get("id"):
            continue
        metadata = dict(binding.get("metadata") or {})
        if runtime_version:
            metadata["runtime_version"] = runtime_version
        else:
            metadata.pop("runtime_version", None)
            metadata.pop("shadow_runtime_version", None)
            metadata["pipeline_contract"] = "conversation_v1"
        supabase_client.update_workflow_binding_metadata(str(binding["id"]), metadata)
        bindings.append(str(binding["id"]))
    return {"persona_slug": persona_slug, "activation": activation, "binding_ids": bindings}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("persona_slug")
    parser.add_argument("target_version", type=int)
    parser.add_argument("--runtime-version")
    args = parser.parse_args()
    print(json.dumps(rollback(
        args.persona_slug, args.target_version, runtime_version=args.runtime_version
    ), ensure_ascii=False, default=str))
