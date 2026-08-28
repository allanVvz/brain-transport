"""Inventory or idempotently repair ready inbound media missing graph links.

Dry-run is the default. Applying repairs is a separate, explicit operation and
does not delete assets, messages, nodes, edges, storage objects, or RAG data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from services import conversation_graph, supabase_client


def _edge_exists(*, source_id: str | None = None, target_id: str | None = None, relation: str) -> bool:
    query = supabase_client.get_client().table("knowledge_edges").select("id,metadata")
    if source_id:
        query = query.eq("source_node_id", source_id)
    if target_id:
        query = query.eq("target_node_id", target_id)
    rows = query.eq("relation_type", relation).limit(20).execute().data or []
    return any((row.get("metadata") or {}).get("active", True) is not False for row in rows)


def inspect_asset(asset: dict[str, Any]) -> dict[str, Any]:
    asset_id = str(asset.get("id") or "")
    persona_id = str(asset.get("persona_id") or "")
    lead_id = asset.get("lead_id")
    node = supabase_client.get_knowledge_node_for_source(
        "assets", asset_id, persona_id=persona_id
    ) if asset_id and persona_id else None
    node_id = str((node or {}).get("id") or "")
    conversation = None
    if persona_id and lead_id:
        rows = (
            supabase_client.get_client().table("knowledge_nodes")
            .select("id").eq("persona_id", persona_id).eq("node_type", "conversation")
            .eq("slug", f"conversa-{lead_id}").limit(1).execute().data or []
        )
        conversation = rows[0] if rows else None
    conversation_id = str((conversation or {}).get("id") or "")
    conversation_edge = bool(
        node_id and conversation_id and _edge_exists(
            source_id=conversation_id, target_id=node_id, relation="uses_asset"
        )
    )
    gallery_edge = bool(
        node_id and _edge_exists(source_id=node_id, relation="gallery_asset")
    )
    missing = [
        name for name, present in (
            ("asset_node", bool(node_id)),
            ("conversation_node", bool(conversation_id)),
            ("conversation_asset_edge", conversation_edge),
            ("asset_gallery_edge", gallery_edge),
        ) if not present
    ]
    return {
        "asset_id": asset_id,
        "persona_id": persona_id,
        "lead_id": lead_id,
        "asset_node_id": node_id or None,
        "conversation_node_id": conversation_id or None,
        "missing": missing,
    }


def run(*, apply: bool, persona_id: str | None = None, asset_ids: list[str] | None = None) -> dict[str, Any]:
    query = (
        supabase_client.get_client().table("assets")
        .select("id,persona_id,lead_id,status,upload_context,created_at")
        .eq("status", "ready").eq("upload_context", "whatsapp_inbound")
    )
    if persona_id:
        query = query.eq("persona_id", persona_id)
    if asset_ids:
        query = query.in_("id", asset_ids)
    assets = query.order("created_at").limit(5000).execute().data or []
    findings = [inspect_asset(asset) for asset in assets]
    incomplete = [item for item in findings if item["missing"]]
    repaired: list[dict[str, Any]] = []
    if apply:
        for item in incomplete:
            repaired.append({
                "asset_id": item["asset_id"],
                "graph_attachment": conversation_graph.attach_inbound_asset(item["asset_id"]),
            })
    return {
        "mode": "apply" if apply else "dry-run",
        "ready_assets_scanned": len(assets),
        "incomplete_count": len(incomplete),
        "incomplete": incomplete,
        "repaired": repaired,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Inventory only (default).")
    mode.add_argument("--apply", action="store_true", help="Create missing nodes/edges idempotently.")
    parser.add_argument("--persona-id")
    parser.add_argument("--asset-id", action="append", dest="asset_ids")
    args = parser.parse_args()
    report = run(apply=bool(args.apply), persona_id=args.persona_id, asset_ids=args.asset_ids)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
