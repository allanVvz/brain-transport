"""Explainable context resolution over one activated action subgraph."""
from __future__ import annotations

import math
import re
from collections import deque
from typing import Any

from schemas.graph_json_v2 import DEFAULT_RELATION_WEIGHTS, GraphJson, Node
from services import graph_json_v2_store, graph_json_v21_adapter, graph_markdown


INTENT_PROFILES: dict[str, dict[str, Any]] = {
    "product_interest": {
        "relations": {"contains", "supports", "answers", "represents", "uses_asset", "applies_to", "targets", "references"},
        "types": {"product": 1.0, "offer": 1.0, "copy": 0.95, "faq": 0.95, "asset": 0.85, "campaign": 0.7, "product_group": 0.7, "brand": 0.6, "rule": 0.7, "tone": 0.65},
        "distance": 3,
    },
    "company": {
        "relations": {"contains", "supports", "answers", "applies_to", "represents", "references"},
        "types": {"brand": 1.0, "briefing": 0.95, "asset": 0.85, "rule": 0.85, "tone": 0.85, "faq": 0.9},
        "distance": 2,
    },
    "campaign": {
        "relations": {"contains", "targets", "supports", "uses_asset", "represents", "answers", "applies_to", "references"},
        "types": {"campaign": 1.0, "audience": 0.95, "product_group": 0.9, "product": 0.9, "offer": 0.9, "copy": 0.9, "asset": 0.85, "faq": 0.85, "brand": 0.65, "briefing": 0.7},
        "distance": 3,
    },
}


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[\wÀ-ÿ-]+", value.lower()) if len(token) > 1}


def _search_text(node: Node) -> str:
    aliases = (node.spec or {}).get("aliases") or (node.data or {}).get("aliases") or []
    return " ".join([
        node.slug, node.title or node.label or "", " ".join(map(str, aliases)),
        str((node.data or {}).get("summary") or ""),
        str((node.data or {}).get("markdown") or ""),
    ])


def _seed_score(node: Node, query: str, query_tokens: set[str], seed_refs: set[str]) -> float:
    if node.id in seed_refs or node.slug in seed_refs:
        return 1.0
    normalized = query.strip().lower()
    if normalized and normalized == node.slug.lower():
        return 1.0
    aliases = {str(item).lower() for item in ((node.spec or {}).get("aliases") or [])}
    if normalized in aliases:
        return 0.95
    haystack = _search_text(node).lower()
    if normalized and normalized in haystack:
        return 0.9
    terms = _tokens(haystack)
    if not query_tokens or not terms:
        return 0.0
    lexical = len(query_tokens & terms) / max(1, len(query_tokens))
    return min(0.89, 0.65 * lexical)


def resolve_context(
    *,
    persona_slug: str,
    destination_id: str,
    graph_version: int,
    intent: str,
    query: str,
    seed_refs: list[str] | None = None,
    max_nodes: int = 24,
    max_tokens: int = 8000,
) -> dict[str, Any]:
    loaded = graph_json_v2_store.load_activated_version(persona_slug, graph_version)
    if loaded is None:
        raise LookupError("Activated graph version not found")
    graph = graph_json_v21_adapter.upgrade_to_v21(loaded)
    graph.graph_version = graph_version
    graph = graph_markdown.canonicalize_graph(graph, reject_markdown_drift=False)
    action = next(
        (
            node for node in graph.nodes
            if node.node_class == "action" and node.action
            and node.action.destination_id == destination_id
            and node.action.enabled
        ),
        None,
    )
    if action is None:
        raise PermissionError("Destination is not enabled in this graph version")

    by_id = {node.id: node for node in graph.nodes}
    grant_edges = {
        edge.source: edge
        for edge in graph.edges
        if edge.relation_type == "publishes_to"
        and edge.target == action.id
        and edge.lifecycle.status == "active"
        and by_id.get(edge.source)
        and by_id[edge.source].lifecycle.status in {"approved", "active"}
    }
    allowed_ids = set(grant_edges)
    profile = INTENT_PROFILES.get(intent, INTENT_PROFILES["product_interest"])
    allowed_relations = set(profile["relations"])
    max_distance = int(profile["distance"])
    query_tokens = _tokens(query)
    requested_seeds = set(seed_refs or [])
    seed_scores = {
        node_id: _seed_score(by_id[node_id], query, query_tokens, requested_seeds)
        for node_id in allowed_ids
    }
    seeds = sorted(seed_scores, key=seed_scores.get, reverse=True)
    seeds = [node_id for node_id in seeds if seed_scores[node_id] > 0][:8]
    if not seeds:
        seeds = sorted(allowed_ids)[:1]
        for seed in seeds:
            seed_scores[seed] = 0.1

    adjacency: dict[str, list[tuple[str, str, float]]] = {}
    for edge in graph.edges:
        relation = edge.relation_type or "references"
        if edge.lifecycle.status != "active" or relation not in allowed_relations:
            continue
        if edge.source not in allowed_ids or edge.target not in allowed_ids:
            continue
        weight = float(edge.weight or DEFAULT_RELATION_WEIGHTS.get(relation, 0.5))
        adjacency.setdefault(edge.source, []).append((edge.target, relation, weight))
        adjacency.setdefault(edge.target, []).append((edge.source, relation, weight))

    best: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        queue = deque([(seed, [seed], [], 1.0)])
        visited_distance: dict[str, int] = {seed: 0}
        while queue:
            node_id, path, relations, path_weight = queue.popleft()
            distance = len(path) - 1
            node = by_id[node_id]
            seed_value = seed_scores.get(seed, 0.0)
            distance_decay = 1.0 / (distance + 1)
            type_priority = float(profile["types"].get(node.node_type, 0.5))
            final = (
                0.40 * seed_value
                + 0.25 * path_weight
                + 0.15 * distance_decay
                + 0.10 * type_priority
                + 0.10 * 1.0
            )
            candidate = {
                "node_id": node_id,
                "node_type": node.node_type,
                "markdown": node.markdown.content if node.markdown else (node.data or {}).get("markdown", ""),
                "why": {
                    "seed_node_id": seed,
                    "path": path,
                    "relations": relations,
                    "distance": distance,
                    "scores": {
                        "seed": round(seed_value, 4),
                        "path": round(path_weight, 4),
                        "distance": round(distance_decay, 4),
                        "type_priority": round(type_priority, 4),
                        "final": round(final, 4),
                    },
                    "graph_version": graph_version,
                    "destination_id": destination_id,
                    "grant_edge_id": grant_edges[node_id].id,
                },
            }
            if node_id not in best or final > best[node_id]["why"]["scores"]["final"]:
                best[node_id] = candidate
            if distance >= max_distance:
                continue
            for neighbor, relation, weight in adjacency.get(node_id, []):
                next_distance = distance + 1
                if visited_distance.get(neighbor, math.inf) <= next_distance:
                    continue
                visited_distance[neighbor] = next_distance
                queue.append((neighbor, [*path, neighbor], [*relations, relation], path_weight * weight))

    ranked = sorted(best.values(), key=lambda item: item["why"]["scores"]["final"], reverse=True)
    selected: list[dict[str, Any]] = []
    token_count = 0
    for item in ranked:
        estimated = max(1, len(item["markdown"]) // 4)
        if selected and token_count + estimated > max_tokens:
            continue
        selected.append(item)
        token_count += estimated
        if len(selected) >= max_nodes:
            break
    return {
        "persona_slug": persona_slug,
        "destination_id": destination_id,
        "action_node_id": action.id,
        "graph_version": graph_version,
        "graph_checksum": graph.content_checksum or graph_json_v2_store.checksum_graph(graph),
        "intent": intent,
        "query": query,
        "items": selected,
        "node_count": len(selected),
        "estimated_tokens": token_count,
    }
