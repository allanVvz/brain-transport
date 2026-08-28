"""Compile current canonical knowledge graphs into GraphRAG v3 publications."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_ROOT))

from services import graph_compiler_v3, supabase_client


def _record_pending_validation(persona: dict, errors: list[str]) -> dict:
    """Persist the exact gap without inventing any commercial knowledge."""
    source = supabase_client.get_or_create_manual_source()
    slug = str(persona["slug"])
    content = "GraphRAG v3 publication blocked:\n" + "\n".join(f"- {error}" for error in errors)
    item_payload = {
        "persona_id": persona["id"], "source_id": source.get("id"),
        "status": "pending", "content_type": "briefing",
        "title": "GraphRAG v3 — lacunas de publicação",
        "content": content, "metadata": {
            "source": "pending_source", "validation_status": "pending_validation",
            "graph_runtime": "graph_agent_runtime_v3", "validation_errors": errors,
            "technical_gap": True,
        },
        "file_path": f"pending/{slug}/graph-agent-runtime-v3-gaps.md",
        "file_type": "md", "tags": ["graphrag-v3", "pending-validation"],
        "agent_visibility": [],
        "content_hash": graph_compiler_v3.canonical_checksum(content),
        "canonical_key": f"{slug}:briefing:graph-agent-runtime-v3-gaps",
        "canonical_hash": graph_compiler_v3.canonical_checksum({"slug": slug, "content": content}),
    }
    existing = supabase_client.get_knowledge_item_by_path(item_payload["file_path"])
    if existing and existing.get("id"):
        supabase_client.update_knowledge_item(str(existing["id"]), item_payload)
        item = {**existing, **item_payload}
    else:
        item = supabase_client.insert_knowledge_item(item_payload)
    node = supabase_client.upsert_knowledge_node({
        "persona_id": persona["id"], "source_table": "knowledge_items",
        "source_id": item.get("id"), "node_type": "briefing",
        "slug": "graph-agent-runtime-v3-gaps",
        "title": "GraphRAG v3 — lacunas de publicação", "summary": content[:400],
        "tags": ["graphrag-v3", "pending-validation"],
        "metadata": {
            "source": "pending_source", "validation_status": "pending_validation",
            "validation_errors": errors, "technical_gap": True,
        },
        "status": "pending_validation",
    })
    rows, _ = supabase_client.list_all_knowledge_graph(persona_id=str(persona["id"]), limit_nodes=10000)
    root = next((row for row in rows if row.get("node_type") == "persona"), None)
    if root and node and root.get("id") != node.get("id"):
        supabase_client.upsert_knowledge_edge(
            str(root["id"]), str(node["id"]), "contains",
            persona_id=str(persona["id"]),
            metadata={"active": True, "primary_tree": True, "technical_gap": True},
        )
    return {"item_id": item.get("id"), "node_id": (node or {}).get("id"), "errors": errors}


def _set_binding_mode(persona_id: str, *, activate_runtime: bool, shadow: bool) -> list[str]:
    changed: list[str] = []
    for binding in supabase_client.get_workflow_bindings(persona_id):
        if not binding.get("id"):
            continue
        metadata = dict(binding.get("metadata") or {})
        if activate_runtime:
            metadata["runtime_version"] = "graph_agent_runtime_v3"
            metadata["pipeline_contract"] = "conversation_v3"
            metadata.pop("shadow_runtime_version", None)
        elif shadow:
            metadata["shadow_runtime_version"] = "graph_agent_runtime_v3"
        else:
            continue
        supabase_client.update_workflow_binding_metadata(str(binding["id"]), metadata)
        changed.append(str(binding["id"]))
    return changed


def run(
    slugs: list[str],
    *,
    activate_publication: bool,
    activate_runtime: bool,
    shadow: bool,
    compile_only: bool = False,
) -> list[dict]:
    personas = [supabase_client.get_persona(slug) for slug in slugs] if slugs else supabase_client.get_personas()
    results: list[dict] = []
    for persona in [value for value in personas if value]:
        try:
            if compile_only:
                node_rows, edge_rows = supabase_client.list_all_knowledge_graph(
                    persona_id=str(persona["id"]), limit_nodes=10000
                )
                document = graph_compiler_v3.compile_graph(
                    persona=persona,
                    node_rows=node_rows,
                    edge_rows=edge_rows,
                )
                results.append({
                    "persona_slug": persona["slug"],
                    "ok": True,
                    "compiler_version": document["compiler_version"],
                    "checksum": document["checksum"],
                    "node_count": len(document["node_by_id"]),
                    "edge_count": len(document["edges"]),
                    "branch_count": len(document["branch_contracts"]),
                    "projection_manifest": document["projection_manifest"],
                })
                continue
            compiled = graph_compiler_v3.compile_persona_publication(
                str(persona["slug"]), activate=activate_publication
            )
            binding_ids = _set_binding_mode(
                str(persona["id"]), activate_runtime=activate_runtime, shadow=shadow
            )
            results.append({
                "persona_slug": persona["slug"], "ok": True,
                "publication": compiled.get("publication"), "binding_ids": binding_ids,
            })
        except graph_compiler_v3.GraphCompilationError as exc:
            if compile_only:
                results.append({
                    "persona_slug": persona["slug"],
                    "ok": False,
                    "compiler_version": graph_compiler_v3.COMPILER_VERSION,
                    "validation_errors": exc.errors,
                })
                continue
            gap = _record_pending_validation(persona, exc.errors)
            supabase_client.insert_event({
                "event_type": "graph_v3.backfill_pending_validation",
                "entity_type": "persona", "entity_id": str(persona["id"]),
                "persona_id": persona["id"], "payload": gap,
            }, level="warning", source="scripts.backfill_graph_publications_v3")
            results.append({"persona_slug": persona["slug"], "ok": False, "pending_validation": gap})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("personas", nargs="*")
    parser.add_argument("--activate-publication", action="store_true")
    parser.add_argument("--activate-runtime", action="store_true")
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="validate and checksum the current graph without database writes or embeddings",
    )
    args = parser.parse_args()
    if args.activate_runtime and not args.activate_publication:
        parser.error("--activate-runtime requires --activate-publication")
    print(json.dumps(run(
        args.personas, activate_publication=args.activate_publication,
        activate_runtime=args.activate_runtime, shadow=args.shadow,
        compile_only=args.compile_only,
    ), ensure_ascii=False, default=str))
