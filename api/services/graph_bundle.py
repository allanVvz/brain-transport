"""Pure GraphBundle compilation and publication planning.

The bundle is an authoring contract.  This module deliberately performs no
database writes, embedding calls or publication activation.  It adapts a
declarative bundle to the existing pure GraphRAG v3 compiler and produces the
human-reviewable plan required before any publish operation.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from services import graph_compiler_v3


BUNDLE_VERSION = "1.0"
PUBLISHABLE_STATUSES = set(graph_compiler_v3.PUBLISHED_STATUSES)
DETACHED_TERMINAL_TYPES = {"embed", "embedded", "gallery"}
SALES_QUALIFICATION_COPY_KEYS = (
    "summary_template",
    "confirmation_question",
    "completion_message",
    "correction_prompt",
    "incomplete_handoff_template",
)


class GraphBundleError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = list(dict.fromkeys(str(error) for error in errors if error))
        super().__init__("; ".join(self.errors))


def normalize_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Validate and deterministically normalize an authoring bundle."""
    errors: list[str] = []
    raw = deepcopy(bundle) if isinstance(bundle, dict) else {}
    if str(raw.get("bundle_version") or "") != BUNDLE_VERSION:
        errors.append(f"unsupported_bundle_version:{raw.get('bundle_version')}")

    persona = raw.get("persona") if isinstance(raw.get("persona"), dict) else {}
    persona_id = str(persona.get("id") or "").strip()
    persona_slug = str(persona.get("slug") or "").strip()
    if not persona_id:
        errors.append("bundle_persona_id_required")
    if not persona_slug:
        errors.append("bundle_persona_slug_required")

    raw_nodes = raw.get("nodes") if isinstance(raw.get("nodes"), list) else []
    raw_edges = raw.get("edges") if isinstance(raw.get("edges"), list) else []
    nodes: list[dict[str, Any]] = []
    node_ids: set[str] = set()
    persona_nodes: list[str] = []
    for index, value in enumerate(raw_nodes):
        if not isinstance(value, dict):
            errors.append(f"bundle_node_invalid:{index}")
            continue
        node = deepcopy(value)
        node_id = str(node.get("id") or "").strip()
        node_type = str(node.get("node_type") or "").strip().lower()
        slug = str(node.get("slug") or "").strip()
        if not node_id:
            errors.append(f"bundle_node_id_required:{index}")
            continue
        if node_id in node_ids:
            errors.append(f"bundle_duplicate_node_id:{node_id}")
            continue
        node_ids.add(node_id)
        if not node_type:
            errors.append(f"bundle_node_type_required:{node_id}")
        if not slug:
            errors.append(f"bundle_node_slug_required:{node_id}")
        if node_type == "persona":
            persona_nodes.append(node_id)
            if slug != persona_slug:
                errors.append(f"bundle_persona_slug_mismatch:{slug}:{persona_slug}")
        data = deepcopy(node.get("data")) if isinstance(node.get("data"), dict) else {}
        source = str(data.get("source") or "").strip()
        data["source"] = source or "pending_source"
        summary = str(node.get("summary") or "")
        if not summary and not any(
            data.get(key) for key in ("content", "question", "answer", "facts")
        ):
            errors.append(f"bundle_node_content_required:{node_id}")
        status = str(node.get("status") or "pending_validation").lower()
        projection_node_id = str(node.get("projection_node_id") or "").strip()
        try:
            namespace = UUID(persona_id)
        except (ValueError, TypeError):
            namespace = uuid5(NAMESPACE_URL, persona_id)
        if not projection_node_id:
            projection_node_id = str(uuid5(namespace, node_id))
        try:
            UUID(projection_node_id)
        except (ValueError, TypeError):
            errors.append(f"bundle_projection_node_id_invalid:{node_id}")
        declared_data_status = str(
            data.get("validation_status") or data.get("status") or ""
        ).strip().lower()
        if declared_data_status and declared_data_status != status:
            errors.append(
                f"bundle_node_status_conflict:{node_id}:{status}:{declared_data_status}"
            )
        if status not in PUBLISHABLE_STATUSES:
            errors.append(f"bundle_node_not_publishable:{node_id}:{status}")
        nodes.append({
            "id": node_id,
            "node_type": node_type,
            "slug": slug,
            "title": str(node.get("title") or slug or node_id),
            "summary": summary,
            "tags": sorted({str(tag) for tag in (node.get("tags") or []) if str(tag)}),
            "status": status,
            "projection_node_id": projection_node_id,
            "data": data,
        })
    if len(persona_nodes) != 1:
        errors.append(f"bundle_requires_one_persona_node:{len(persona_nodes)}")
    elif persona_node := next(
        (node for node in nodes if node["id"] == persona_nodes[0]), None
    ):
        persona_data = persona_node.get("data") or {}
        if str(persona_data.get("business_model") or "").strip().lower() == "sales":
            conversation_policy = persona_data.get("conversation_policy")
            qualification = (
                conversation_policy.get("qualification")
                if isinstance(conversation_policy, dict)
                else None
            )
            qualification = qualification if isinstance(qualification, dict) else {}
            for key in SALES_QUALIFICATION_COPY_KEYS:
                value = qualification.get(key)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"bundle_sales_qualification_copy_required:{key}")

    edges: list[dict[str, Any]] = []
    edge_ids: set[str] = set()
    for index, value in enumerate(raw_edges):
        if not isinstance(value, dict):
            errors.append(f"bundle_edge_invalid:{index}")
            continue
        edge = deepcopy(value)
        edge_id = str(edge.get("id") or "").strip()
        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        if not edge_id:
            errors.append(f"bundle_edge_id_required:{index}")
            continue
        if edge_id in edge_ids:
            errors.append(f"bundle_duplicate_edge_id:{edge_id}")
            continue
        edge_ids.add(edge_id)
        if source not in node_ids:
            errors.append(f"bundle_edge_source_missing:{edge_id}:{source}")
        if target not in node_ids:
            errors.append(f"bundle_edge_target_missing:{edge_id}:{target}")
        try:
            weight = float(edge.get("weight") or 1.0)
        except (TypeError, ValueError):
            errors.append(f"bundle_edge_weight_invalid:{edge_id}")
            weight = 1.0
        edges.append({
            "id": edge_id,
            "source": source,
            "target": target,
            "relation_type": str(edge.get("relation_type") or "contains"),
            "weight": weight,
            "metadata": (
                deepcopy(edge.get("metadata"))
                if isinstance(edge.get("metadata"), dict)
                else {}
            ),
        })

    logical_edges: set[tuple[str, str, str]] = set()
    for edge in edges:
        logical_key = (edge["source"], edge["target"], edge["relation_type"])
        if logical_key in logical_edges:
            errors.append(
                "bundle_duplicate_logical_edge:"
                f"{edge['source']}:{edge['target']}:{edge['relation_type']}"
            )
        logical_edges.add(logical_key)

    if len(persona_nodes) == 1:
        root_id = persona_nodes[0]
        parents: dict[str, str] = {}
        for edge in edges:
            if edge["relation_type"] != "contains":
                continue
            target = edge["target"]
            prior = parents.get(target)
            if prior and prior != edge["source"]:
                errors.append(f"bundle_primary_parent_ambiguous:{target}")
            parents[target] = edge["source"]
        if root_id in parents:
            errors.append(f"bundle_persona_cannot_have_parent:{root_id}")
        type_by_id = {node["id"]: node["node_type"] for node in nodes}
        for node_id in sorted(node_ids - {root_id}):
            if node_id not in parents:
                if type_by_id.get(node_id) not in DETACHED_TERMINAL_TYPES:
                    errors.append(f"bundle_primary_parent_missing:{node_id}")
                continue
            visited = {node_id}
            current = node_id
            while current in parents:
                current = parents[current]
                if current in visited:
                    errors.append(f"bundle_primary_cycle:{node_id}:{current}")
                    break
                visited.add(current)
            if current != root_id:
                errors.append(f"bundle_path_not_rooted_in_persona:{node_id}:{current}")

    metadata = (
        deepcopy(raw.get("metadata"))
        if isinstance(raw.get("metadata"), dict)
        else {}
    )
    embedding_profile = (
        metadata.get("embedding_profile")
        if isinstance(metadata.get("embedding_profile"), dict)
        else {}
    )
    provider = str(embedding_profile.get("embedding_provider") or "").strip()
    model = str(embedding_profile.get("embedding_model") or "").strip()
    try:
        dimension = int(embedding_profile.get("embedding_dimension"))
    except (TypeError, ValueError):
        dimension = 0
    if provider not in {"local", "openai"}:
        errors.append("bundle_embedding_provider_invalid")
    if not model:
        errors.append("bundle_embedding_model_required")
    if dimension != graph_compiler_v3.EMBEDDING_DIMENSION:
        errors.append("bundle_embedding_dimension_invalid")
    metadata["embedding_profile"] = {
        "embedding_provider": provider,
        "embedding_model": model,
        "embedding_dimension": dimension,
    }

    if errors:
        raise GraphBundleError(errors)
    return {
        "bundle_version": BUNDLE_VERSION,
        "persona": {"id": persona_id, "slug": persona_slug},
        "metadata": metadata,
        "nodes": sorted(nodes, key=lambda item: item["id"]),
        "edges": sorted(edges, key=lambda item: item["id"]),
    }


def _compiler_inputs(bundle: dict[str, Any]) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    normalized = normalize_bundle(bundle)
    row_by_node_id: dict[str, dict[str, Any]] = {}
    node_rows: list[dict[str, Any]] = []
    for node in normalized["nodes"]:
        row = {
            "id": node["projection_node_id"],
            "node_type": node["node_type"],
            "slug": node["slug"],
            "title": node["title"],
            "summary": node["summary"],
            "tags": node["tags"],
            "status": node["status"],
            "metadata": {
                "graph_json_node_id": node["id"],
                **deepcopy(node["data"]),
            },
        }
        row_by_node_id[node["id"]] = row
        node_rows.append(row)
    edge_rows = [
        {
            "id": edge["id"],
            "source_node_id": row_by_node_id[edge["source"]]["id"],
            "target_node_id": row_by_node_id[edge["target"]]["id"],
            "relation_type": edge["relation_type"],
            "weight": edge["weight"],
            "metadata": {
                "active": True,
                "graph_json_edge_id": edge["id"],
                **deepcopy(edge["metadata"]),
            },
        }
        for edge in normalized["edges"]
    ]
    return (
        normalized["persona"],
        node_rows,
        edge_rows,
        normalized["metadata"]["embedding_profile"],
    )


def compile_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Compile one bundle through the existing generic v3 compiler."""
    persona, node_rows, edge_rows, embedding_profile = _compiler_inputs(bundle)
    return graph_compiler_v3.compile_graph(
        persona=persona,
        node_rows=node_rows,
        edge_rows=edge_rows,
        embedding_profile=embedding_profile,
    )


def _chunk_identities(document: dict[str, Any] | None) -> set[str]:
    manifest = (document or {}).get("projection_manifest") or {}
    profile = {
        "provider": manifest.get("embedding_provider"),
        "model": manifest.get("embedding_model"),
        "dimension": manifest.get("embedding_dimension"),
    }
    projected_node_ids = {
        str(node_id)
        for members in ((document or {}).get("branch_memberships") or {}).values()
        for node_id in members
    }
    return {
        graph_compiler_v3.canonical_checksum({
            "chunk_checksum": str(chunk["checksum"]),
            "embedding_profile": profile,
        })
        for node_id, node in ((document or {}).get("node_by_id") or {}).items()
        if str(node_id) in projected_node_ids
        for chunk in graph_compiler_v3.semantic_chunks(node)
        if chunk.get("checksum")
    }


def _node_changes(
    current: dict[str, Any] | None,
    candidate: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    before = (current or {}).get("node_by_id") or {}
    after = candidate.get("node_by_id") or {}
    before_ids, after_ids = set(before), set(after)
    def fingerprint(node: dict[str, Any]) -> str:
        semantic = {
            key: value for key, value in node.items()
            if key != "projection_node_id"
        }
        return graph_compiler_v3.canonical_checksum(semantic)

    changed = sorted(
        node_id
        for node_id in before_ids & after_ids
        if fingerprint(before[node_id]) != fingerprint(after[node_id])
    )
    return sorted(after_ids - before_ids), changed, sorted(before_ids - after_ids)


def _edge_changes(
    current: dict[str, Any] | None,
    candidate: dict[str, Any],
) -> tuple[list[str], list[str], list[str], set[str]]:
    before = {str(edge.get("id")): edge for edge in (current or {}).get("edges") or []}
    after = {str(edge.get("id")): edge for edge in candidate.get("edges") or []}
    before_ids, after_ids = set(before), set(after)
    changed = sorted(
        edge_id for edge_id in before_ids & after_ids
        if graph_compiler_v3.canonical_checksum(before[edge_id])
        != graph_compiler_v3.canonical_checksum(after[edge_id])
    )
    affected_nodes = {
        str(edge.get(endpoint) or "")
        for edge_id in (before_ids ^ after_ids) | set(changed)
        for edge in (before.get(edge_id), after.get(edge_id))
        if edge
        for endpoint in ("source", "target")
        if edge.get(endpoint)
    }
    return (
        sorted(after_ids - before_ids),
        changed,
        sorted(before_ids - after_ids),
        affected_nodes,
    )


def _affected_branches(
    current: dict[str, Any] | None,
    candidate: dict[str, Any],
    changed_node_ids: set[str],
) -> list[str]:
    affected: set[str] = set()
    for document in (current or {}, candidate):
        for branch, members in (document.get("branch_memberships") or {}).items():
            if changed_node_ids.intersection(members):
                affected.add(str(branch))
    return sorted(affected)


def _breaking_changes(
    current: dict[str, Any] | None,
    candidate: dict[str, Any],
    removed_nodes: list[str],
    removed_edges: list[str],
) -> list[str]:
    if not current:
        return []
    changes = [f"node_removed:{node_id}" for node_id in removed_nodes]
    if current.get("compiler_version") != candidate.get("compiler_version"):
        changes.append(
            "compiler_version_changed:"
            f"{current.get('compiler_version')}:{candidate.get('compiler_version')}"
        )
    changes.extend(f"edge_removed:{edge_id}" for edge_id in removed_edges)
    before_contracts = current.get("branch_contracts") or {}
    after_contracts = candidate.get("branch_contracts") or {}
    for branch in sorted(set(before_contracts) - set(after_contracts)):
        changes.append(f"branch_removed:{branch}")
    for branch in sorted(set(before_contracts) & set(after_contracts)):
        before_required = set(before_contracts[branch].get("required_fields") or [])
        after_required = set(after_contracts[branch].get("required_fields") or [])
        for field in sorted(after_required - before_required):
            changes.append(f"required_field_added:{branch}:{field}")
        contract_keys = (
            "fields", "questions", "completion", "claims", "handoff",
            "handoff_rules", "conversation_policy", "collection_policy",
            "field_labels",
        )
        before_contract = {
            key: before_contracts[branch].get(key) for key in contract_keys
        }
        after_contract = {
            key: after_contracts[branch].get(key) for key in contract_keys
        }
        if (
            graph_compiler_v3.canonical_checksum(before_contract)
            != graph_compiler_v3.canonical_checksum(after_contract)
        ):
            changes.append(f"branch_contract_changed:{branch}")
        if (
            before_contracts[branch].get("branch_path_checksum")
            != after_contracts[branch].get("branch_path_checksum")
            or before_contracts[branch].get("closure_checksum")
            != after_contracts[branch].get("closure_checksum")
        ):
            changes.append(f"branch_structure_changed:{branch}")
    return changes


def _current_document_errors(
    current: dict[str, Any] | None,
    persona: dict[str, Any],
) -> list[str]:
    if current is None:
        return []
    errors: list[str] = []
    if current.get("schema_version") != "3.0":
        errors.append("baseline_schema_version_invalid")
    current_persona = current.get("persona") or {}
    if str(current_persona.get("id") or "") != persona["id"]:
        errors.append("baseline_persona_id_mismatch")
    if str(current_persona.get("slug") or "") != persona["slug"]:
        errors.append("baseline_persona_slug_mismatch")
    if not str(current.get("compiler_version") or ""):
        errors.append("baseline_compiler_version_missing")
    nodes = current.get("nodes")
    node_by_id = current.get("node_by_id")
    edges = current.get("edges")
    memberships = current.get("branch_memberships")
    contracts = current.get("branch_contracts")
    manifest = current.get("projection_manifest")
    if not isinstance(nodes, list) or not isinstance(node_by_id, dict):
        errors.append("baseline_nodes_invalid")
    if not isinstance(edges, list):
        errors.append("baseline_edges_invalid")
    if not isinstance(memberships, dict) or not isinstance(contracts, dict):
        errors.append("baseline_branch_contract_invalid")
    if not isinstance(manifest, dict) or not all(
        manifest.get(key) is not None
        for key in ("embedding_provider", "embedding_model", "embedding_dimension")
    ):
        errors.append("baseline_projection_manifest_invalid")
    if isinstance(nodes, list) and isinstance(node_by_id, dict):
        listed_ids = {
            str(node.get("id") or "") for node in nodes if isinstance(node, dict)
        }
        if listed_ids != set(node_by_id):
            errors.append("baseline_node_index_mismatch")
        if isinstance(edges, list):
            if any(
                not isinstance(edge, dict)
                or str(edge.get("source") or "") not in listed_ids
                or str(edge.get("target") or "") not in listed_ids
                for edge in edges
            ):
                errors.append("baseline_edge_reference_invalid")
        if isinstance(memberships, dict) and isinstance(contracts, dict):
            for branch, members in memberships.items():
                contract = contracts.get(branch)
                if (
                    branch not in listed_ids
                    or not isinstance(members, dict)
                    or not isinstance(contract, dict)
                    or str(contract.get("branch_anchor_node_id") or "") != str(branch)
                    or any(str(node_id) not in listed_ids for node_id in members)
                ):
                    errors.append(f"baseline_branch_membership_invalid:{branch}")
    checksum = str(current.get("checksum") or "")
    payload = deepcopy(current)
    payload.pop("checksum", None)
    if not checksum or graph_compiler_v3.canonical_checksum(payload) != checksum:
        errors.append("baseline_checksum_invalid")
    return errors


def build_publication_plan(
    bundle: dict[str, Any],
    *,
    current_document: dict[str, Any] | None = None,
    next_version: int = 1,
) -> dict[str, Any]:
    """Return a complete dry-run plan; never publish or activate anything."""
    try:
        normalized = normalize_bundle(bundle)
        pending_source_nodes = [
            node["id"] for node in normalized["nodes"]
            if str((node.get("data") or {}).get("source") or "").strip().lower()
            == "pending_source"
        ]
        if pending_source_nodes:
            raise GraphBundleError([
                f"bundle_node_source_pending:{node_id}"
                for node_id in pending_source_nodes
            ])
        publication_allowed = normalized["metadata"].get("publication_allowed") is True
        internal_test_allowed = (
            normalized["metadata"].get("internal_wa_validator_test_allowed") is True
        )
        baseline_errors = _current_document_errors(
            current_document, normalized["persona"]
        )
        if baseline_errors:
            raise GraphBundleError(baseline_errors)
        draft_checksum = graph_compiler_v3.canonical_checksum(normalized)
        candidate = compile_bundle(normalized)
    except (GraphBundleError, graph_compiler_v3.GraphCompilationError) as exc:
        errors = list(getattr(exc, "errors", [str(exc)]))
        return {
            "plan_version": "1.0",
            "disposition": "blocked",
            "publication_allowed": False,
            "approval_scope": "none",
            "draft_checksum": graph_compiler_v3.canonical_checksum(bundle),
            "runtime_checksum": None,
            "next_version": int(next_version),
            "nodes_added": 0,
            "nodes_changed": 0,
            "nodes_removed": 0,
            "node_changes": {"added": [], "changed": [], "removed": []},
            "edges_added": 0,
            "edges_changed": 0,
            "edges_removed": 0,
            "edge_changes": {"added": [], "changed": [], "removed": []},
            "branches_affected": [],
            "chunks_reused": 0,
            "chunks_to_embed": 0,
            "estimated_embedding_cost": {
                "amount": None,
                "currency": "USD",
                "reason": "plan_blocked",
            },
            "breaking_contract_changes": [],
            "validation_errors": errors,
        }

    added, changed, removed = _node_changes(current_document, candidate)
    edge_added, edge_changed, edge_removed, edge_affected_nodes = _edge_changes(
        current_document, candidate
    )
    changed_ids = set(added) | set(changed) | set(removed) | edge_affected_nodes
    current_chunks = _chunk_identities(current_document)
    candidate_chunks = _chunk_identities(candidate)
    breaking = _breaking_changes(
        current_document, candidate, removed, edge_removed
    )
    return {
        "plan_version": "1.0",
        "disposition": (
            "awaiting_approval"
            if publication_allowed
            else "approved_for_internal_test"
            if internal_test_allowed
            else "dry_run_complete"
        ),
        "publication_allowed": publication_allowed,
        "approval_scope": (
            "publication_plan"
            if publication_allowed
            else "internal_wa_validator"
            if internal_test_allowed
            else "dry_run_only"
        ),
        "draft_checksum": draft_checksum,
        "runtime_checksum": candidate["checksum"],
        "next_version": int(next_version),
        "nodes_added": len(added),
        "nodes_changed": len(changed),
        "nodes_removed": len(removed),
        "node_changes": {"added": added, "changed": changed, "removed": removed},
        "edges_added": len(edge_added),
        "edges_changed": len(edge_changed),
        "edges_removed": len(edge_removed),
        "edge_changes": {
            "added": edge_added,
            "changed": edge_changed,
            "removed": edge_removed,
        },
        "branches_affected": _affected_branches(
            current_document, candidate, changed_ids
        ),
        "chunks_reused": len(candidate_chunks & current_chunks),
        "chunks_to_embed": len(candidate_chunks - current_chunks),
        "estimated_embedding_cost": {
            "amount": None,
            "currency": "USD",
            "reason": "provider_pricing_not_configured",
        },
        "breaking_contract_changes": breaking,
        "validation_errors": [],
        "compiler_version": candidate["compiler_version"],
        "candidate_document": candidate,
    }

