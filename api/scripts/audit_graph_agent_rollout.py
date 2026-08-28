"""Emit a secret-free audit of GraphRAG publications, bindings and n8n wiring."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_ROOT))

from services import n8n_client, supabase_client


def _checksum(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _count(table: str, publication_id: str, *, non_null: str | None = None) -> int:
    query = (
        supabase_client.get_client().table(table).select("publication_id", count="exact")
        .eq("publication_id", publication_id)
    )
    if non_null:
        query = query.not_.is_(non_null, "null")
    response = query.limit(1).execute()
    return int(response.count or 0)


def _mask_phone(value: Any) -> str | None:
    digits = "".join(character for character in str(value or "") if character.isdigit())
    return f"{digits[:2]}***{digits[-4:]}" if len(digits) >= 6 else None


def audit(slug: str) -> dict[str, Any]:
    persona = supabase_client.get_persona(slug)
    if not persona:
        raise RuntimeError(f"persona not found: {slug}")
    bindings = [
        row for row in supabase_client.get_workflow_bindings(str(persona["id"]))
        if row.get("active")
    ]
    binding = bindings[0] if len(bindings) == 1 else {}
    metadata = dict(binding.get("metadata") or {})
    integration = supabase_client.get_persona_integration_connection(
        str(persona["id"]), "deepseek"
    ) or {}
    config = dict(integration.get("config_json") or {})
    workflow_id = str(config.get("n8n_workflow_id") or binding.get("n8n_workflow_id") or "")
    workflow = n8n_client.get_workflow(workflow_id) if workflow_id else None
    nodes = (workflow or {}).get("nodes") or []
    topology = {
        "nodes": sorted((str(node.get("id")), str(node.get("type"))) for node in nodes),
        "connections": (workflow or {}).get("connections") or {},
    }
    publication = supabase_client.get_active_graph_publication(str(persona["id"]))
    projection = None
    if publication:
        publication_id = str(publication["id"])
        projection = {
            "coordinates": _count("graph_node_coordinates", publication_id),
            "memberships": _count("graph_branch_memberships", publication_id),
            "contracts": _count("graph_branch_contracts", publication_id),
            "entries": _count("knowledge_rag_entries", publication_id),
            "chunks": _count("knowledge_rag_chunks", publication_id),
            "embeddings": _count(
                "knowledge_rag_chunks", publication_id, non_null="embedding"
            ),
        }
    portal = (persona.get("config") or {}).get("portal") or {}
    return {
        "persona_slug": slug,
        "persona_id": persona["id"],
        "automation_mode": portal.get("automation_mode"),
        "binding": {
            "id": binding.get("id"), "provider": binding.get("provider"),
            "phone": _mask_phone(binding.get("whatsapp_number")),
            "decision_owner": metadata.get("decision_owner"),
            "runtime_version": metadata.get("runtime_version"),
            "pipeline_contract": metadata.get("pipeline_contract"),
            "model": metadata.get("model") or config.get("model"),
            "reply_source": metadata.get("reply_source") or config.get("reply_source"),
        },
        "workflow": {
            "id": workflow_id or None,
            "active": (workflow or {}).get("active"),
            "template": config.get("workflow_template"),
            "rendered_checksum": config.get("workflow_checksum"),
            "structural_checksum": _checksum(topology) if workflow else None,
            "node_ids": [item[0] for item in topology["nodes"]],
        },
        "publication": ({
            "id": publication["id"], "version": publication["version"],
            "checksum": publication["checksum"],
            "compiler_version": publication["compiler_version"],
            "projection_manifest": (publication.get("document_json") or {}).get(
                "projection_manifest"
            ),
            "projected_counts": projection,
        } if publication else None),
    }


if __name__ == "__main__":
    slugs = sys.argv[1:] or ["aurora", "vz-lupas"]
    report = [audit(slug) for slug in slugs]
    report.append({
        "same_workflow_structure": len({
            row["workflow"]["structural_checksum"] for row in report
            if row["workflow"]["structural_checksum"]
        }) == 1
    })
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))

