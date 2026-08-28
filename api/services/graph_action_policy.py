"""Restricted deterministic auto-publication policy evaluator."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

from schemas.graph_json_v2 import Edge, GraphJson


def _get(value: dict[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _ancestors(graph: GraphJson, node_id: str) -> list[str]:
    parent_by_child = {
        edge.target: edge.source
        for edge in graph.edges
        if edge.relation_type == "contains" and edge.lifecycle.status == "active"
    }
    out: list[str] = []
    visited: set[str] = set()
    current = node_id
    while current in parent_by_child and current not in visited:
        visited.add(current)
        current = parent_by_child[current]
        out.append(current)
    return out


def _condition_matches(graph: GraphJson, node_payload: dict[str, Any], condition: dict[str, Any]) -> bool:
    if "field" in condition:
        actual = _get(node_payload, str(condition["field"]))
        op = condition.get("op", "eq")
        expected = condition.get("value")
        if op == "eq":
            return actual == expected
        if op == "in":
            return actual in (expected or [])
        if op == "contains":
            return expected in (actual or [])
        return False
    relation = condition.get("relation_exists")
    if isinstance(relation, dict) and relation.get("type") == "contains":
        by_id = {node.id: node for node in graph.nodes}
        return any(
            by_id.get(ancestor) and by_id[ancestor].node_type == relation.get("ancestor_type")
            for ancestor in _ancestors(graph, str(node_payload.get("id")))
        )
    return False


def _policy_matches(graph: GraphJson, node_payload: dict[str, Any], policy: dict[str, Any]) -> bool:
    conditions = policy.get("conditions") or {}
    all_conditions = conditions.get("all") if isinstance(conditions, dict) else []
    any_conditions = conditions.get("any") if isinstance(conditions, dict) else []
    if all_conditions and not all(_condition_matches(graph, node_payload, item) for item in all_conditions):
        return False
    if any_conditions and not any(_condition_matches(graph, node_payload, item) for item in any_conditions):
        return False
    return True


def evaluate(graph: GraphJson) -> list[dict[str, Any]]:
    by_id = {node.id: node for node in graph.nodes}
    existing = {
        (edge.source, edge.target, (edge.grant.policy_id if edge.grant else None)): edge
        for edge in graph.edges
        if edge.relation_type == "publishes_to"
    }
    changes: list[dict[str, Any]] = []
    for action in [node for node in graph.nodes if node.node_class == "action" and node.action]:
        policies = action.action.policy.auto_connect
        for policy in policies:
            if not isinstance(policy, dict) or not policy.get("enabled", True):
                continue
            policy_id = str(policy.get("policy_id") or "")
            if not policy_id:
                continue
            accepted = set(policy.get("accepted_node_types") or action.action.accepted_node_types)
            for node in [item for item in graph.nodes if item.node_class == "knowledge"]:
                key = (node.id, action.id, policy_id)
                matches = (
                    node.node_type in accepted
                    and node.lifecycle.status in {"approved", "active"}
                    and _policy_matches(graph, node.model_dump(mode="json"), policy)
                )
                prior = existing.get(key)
                if matches and (prior is None or prior.lifecycle.status != "active"):
                    changes.append({
                        "op": "activate", "source": node.id, "target": action.id,
                        "policy_id": policy_id, "edge_id": prior.id if prior else None,
                    })
                elif not matches and prior is not None and prior.lifecycle.status == "active":
                    changes.append({"op": "revoke", "edge_id": prior.id, "source": node.id, "target": action.id, "policy_id": policy_id})
    return changes


def apply(graph: GraphJson) -> tuple[GraphJson, list[dict[str, Any]]]:
    result = GraphJson.model_validate(deepcopy(graph.model_dump(mode="json")))
    changes = evaluate(result)
    by_edge_id = {edge.id: edge for edge in result.edges}
    by_node = {node.id: node for node in result.nodes}
    for change in changes:
        if change["op"] == "revoke":
            by_edge_id[change["edge_id"]].lifecycle.status = "revoked"
            continue
        node = by_node[change["source"]]
        material = json.dumps(node.model_dump(mode="json", exclude={"markdown"}), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        revision_checksum = "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()
        if change.get("edge_id") in by_edge_id:
            prior = by_edge_id[change["edge_id"]]
            prior.lifecycle.status = "active"
            if prior.grant:
                prior.grant.source_approval_checksum = revision_checksum
            change["source_approval_checksum"] = revision_checksum
            continue
        digest = hashlib.sha256(
            f"{change['source']}:{change['target']}:{change['policy_id']}:{node.lifecycle.revision}".encode("utf-8")
        ).hexdigest()[:20]
        edge = Edge.model_validate({
            "id": f"edge:policy:{digest}",
            "source": change["source"],
            "target": change["target"],
            "relation_type": "publishes_to",
            "relation_class": "publication",
            "primary_tree": False,
            "lifecycle": {"status": "active"},
            "grant": {
                "mode": "policy",
                "policy_id": change["policy_id"],
                "reason": "Automatic action policy",
                "source_approval_checksum": revision_checksum,
            },
            "metadata": {"source_revision": node.lifecycle.revision},
        })
        result.edges.append(edge)
        by_edge_id[edge.id] = edge
        change["edge_id"] = edge.id
        change["source_approval_checksum"] = revision_checksum
    return result, changes

