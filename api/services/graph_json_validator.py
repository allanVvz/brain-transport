from __future__ import annotations

import json
import sys
from collections import defaultdict, deque

try:
    from services import supabase_client
except ModuleNotFoundError:
    from pathlib import Path

    API_DIR = Path(__file__).resolve().parents[1]
    if str(API_DIR) not in sys.path:
        sys.path.insert(0, str(API_DIR))
    from services import supabase_client

ALLOWED_NEXT = {
    "persona": {"brand"},
    "brand": {"briefing"},
    "briefing": {"campaign"},
    "campaign": {"audience"},
    "audience": {"product_group"},
    "product_group": {"product"},
    "product": {"faq"},
    "faq": {"embed"},
}


def _validate_graph(graph_json: dict) -> dict:
    nodes = graph_json.get("nodes") or []
    edges = graph_json.get("edges") or []
    node_map = {str(n.get("id")): n for n in nodes if n.get("id")}
    outgoing = defaultdict(list)
    for edge in edges:
        src = str(edge.get("source") or "")
        tgt = str(edge.get("target") or "")
        if src and tgt:
            outgoing[src].append(tgt)

    errors: list[str] = []
    for edge in edges:
        src = node_map.get(str(edge.get("source") or ""))
        tgt = node_map.get(str(edge.get("target") or ""))
        if not src or not tgt:
            errors.append(f"EDGE_NODE_MISSING:{edge.get('id')}")
            continue
        src_type = str(src.get("type") or "")
        tgt_type = str(tgt.get("type") or "")
        edge_type = str(edge.get("edge_type") or "reference")
        if tgt_type == "embed" and src_type != "faq":
            errors.append(f"EMBED_SOURCE_NOT_FAQ:{edge.get('id')}")
        if src_type == "product" and tgt_type == "embed":
            errors.append(f"PRODUCT_DIRECT_TO_EMBED:{edge.get('id')}")
        if edge_type == "main" and src_type in ALLOWED_NEXT and tgt_type not in ALLOWED_NEXT[src_type]:
            errors.append(f"INVALID_EDGE:{src_type}->{tgt_type}:{edge.get('id')}")

    persona_nodes = [n for n in nodes if str(n.get("type") or "") == "persona"]
    reachable = set()
    for root in persona_nodes:
        q = deque([str(root["id"])])
        while q:
            nid = q.popleft()
            if nid in reachable:
                continue
            reachable.add(nid)
            for tgt in outgoing.get(nid, []):
                if tgt not in reachable:
                    q.append(tgt)

    disconnected = [
        str(n.get("id"))
        for n in nodes
        if n.get("id")
        and str(n.get("id")) not in reachable
        and str(n.get("type") or "") not in {"gallery", "embedded"}
    ]
    if disconnected:
        errors.append(f"DISCONNECTED_NODES:{','.join(disconnected[:20])}")

    return {"is_valid": len(errors) == 0, "errors": errors, "node_count": len(nodes), "edge_count": len(edges)}


def validate_document_id(document_id: str) -> dict:
    rows = supabase_client.list_system_events(
        entity_type="graph_document",
        event_types=["graph_document_published"],
        entity_id=document_id,
        limit=5,
    )
    if not rows and ":v" in document_id:
        try:
            persona_slug, brand_slug, _ver = document_id.rsplit(":", 2)
            candidates = supabase_client.list_system_events(
                entity_type="graph_document",
                event_types=["graph_document_published"],
                limit=200,
            )
            rows = [
                r
                for r in candidates
                if (r.get("payload") or {}).get("persona_slug") == persona_slug
                and ((r.get("payload") or {}).get("brand_slug") or "default") == brand_slug
            ]
            rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        except Exception:
            rows = []
    if not rows:
        return {"is_valid": False, "errors": [f"DOCUMENT_NOT_FOUND:{document_id}"]}
    payload = (rows[0].get("payload") or {})
    graph_json = payload.get("graph_json") or {}
    result = _validate_graph(graph_json)
    result["document_id"] = document_id
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"is_valid": False, "errors": ["USAGE: python -m ai-brain.api.services.graph_json_validator <doc-id>"]}))
        raise SystemExit(2)
    doc_id = sys.argv[1]
    print(json.dumps(validate_document_id(doc_id), ensure_ascii=False))

