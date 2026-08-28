"""Compile editable knowledge tables into immutable GraphRAG v3 publications.

This is deliberately a compiler, not a dialog engine.  Commercial concepts are
read from node metadata/capabilities; the compiler only proves structural and
declarative consistency and projects search material for each branch closure.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from services import graph_conversation_contract, supabase_client


COMPILER_VERSION = "graph-compiler-v3.6.2"
FAQ_PROJECTION_CONTRACT = "v1"
LOCAL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 1536
_local_embedding_model: Any | None = None
_local_embedding_model_name: str | None = None
CONTRACT_DOCUMENT = Path(__file__).resolve().parents[1] / "contracts" / "graph-agent-runtime-v3.md"
PUBLISHED_STATUSES = {"approved", "active", "validated", "ativo", "embedded"}
FACT_STATUSES = {"known", "unknown", "declined", "needs_confirmation", "invalid"}
STRUCTURAL_RELATIONS = {"contains"}
RAG_CONTENT_TYPES = {
    "faq", "product", "service", "product_group", "offer", "brand", "campaign",
    "rule", "tone", "copy", "briefing", "audience", "asset", "entity", "general_note",
}
TECHNICAL_METADATA_KEYS = {
    "active", "content_hash", "file_path", "graph_checksum", "graph_json_id",
    "graph_json_import", "graph_json_node_id", "graph_json_parent_id",
    "graph_version", "knowledge_item_id", "schema_version", "session_id",
}


class GraphCompilationError(ValueError):
    def __init__(self, errors: Iterable[str]):
        self.errors = list(dict.fromkeys(str(item) for item in errors if item))
        super().__init__("; ".join(self.errors))


def canonical_checksum(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_content_checksum(value: str) -> str:
    """Hash rendered card text consistently across compiler and runtime."""
    canonical = unicodedata.normalize(
        "NFC", str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def embedding_provider() -> str:
    configured = (os.environ.get("GRAPH_RAG_EMBEDDING_PROVIDER") or "auto").strip().lower()
    if configured not in {"auto", "openai", "local"}:
        raise RuntimeError("GRAPH_RAG_EMBEDDING_PROVIDER must be auto, openai, or local")
    if configured == "auto":
        return "openai" if (os.environ.get("OPENAI_API_KEY") or "").strip() else "local"
    return configured


def embedding_model_name() -> str:
    if embedding_provider() == "openai":
        return (os.environ.get("GRAPH_RAG_EMBEDDING_MODEL") or "text-embedding-3-small").strip()
    return (os.environ.get("GRAPH_RAG_LOCAL_EMBEDDING_MODEL") or LOCAL_EMBEDDING_MODEL).strip()


def _fit_embedding_dimension(vector: Any) -> list[float]:
    values = [float(item) for item in vector]
    if not values or len(values) > EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"embedding provider returned invalid native dimension {len(values)}"
        )
    return values + [0.0] * (EMBEDDING_DIMENSION - len(values))


def _stable_node_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(
        metadata.get("graph_json_node_id")
        or row.get("canonical_key")
        or f"node:{row.get('node_type') or 'knowledge'}:{row.get('slug') or row.get('id')}"
    )


def _stable_edge_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("graph_json_edge_id") or f"edge:{row.get('id')}")


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row.get("metadata") or {})


def _runtime_node_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Exclude publication provenance that cannot change dialogue behavior."""
    metadata = _metadata(row)
    metadata.pop("graph_bundle_draft_checksum", None)
    return metadata


def _runtime_edge_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Keep layout/provenance fields out of the semantic runtime checksum."""
    metadata = _metadata(row)
    metadata.pop("graph_bundle_draft_checksum", None)
    metadata.pop("primary_tree", None)
    return metadata


def _capability(data: dict[str, Any], name: str) -> bool:
    capabilities = data.get("capabilities")
    if isinstance(capabilities, dict):
        value = capabilities.get(name)
        if isinstance(value, dict):
            return value.get("enabled", True) is True
        if value is True:
            return True
    if isinstance(capabilities, list) and name in capabilities:
        return True
    return data.get(name) is True or data.get(f"is_{name}") is True


def is_branch_anchor(node: dict[str, Any]) -> bool:
    """Anchors are capability-owned; node_type is never consulted."""
    return _capability(node.get("data") or {}, "branch_anchor")


def _published(row: dict[str, Any]) -> bool:
    return (
        _metadata(row).get("active", True) is not False
        and str(row.get("status") or "").lower() in PUBLISHED_STATUSES
    )


def _select_persona_node(
    nodes: list[dict[str, Any]], errors: list[str]
) -> dict[str, Any] | None:
    """Select the canonical persona without depending on database row order."""
    candidates = [node for node in nodes if node["node_type"] == "persona"]
    projected = [
        node
        for node in candidates
        if (node.get("data") or {}).get("graph_json_import") is True
    ]
    authored = [
        node
        for node in candidates
        if str(node.get("slug") or "").lower() != "self"
        and str((node.get("data") or {}).get("role") or "").lower() != "root"
    ]
    eligible = projected if projected else (authored if authored else candidates)
    if len(eligible) == 1:
        return eligible[0]
    if len(eligible) > 1:
        errors.append(
            "ambiguous_persona_node:"
            + ",".join(sorted(str(node.get("id") or "") for node in eligible))
        )
    return None


def _condition_valid(condition: Any, field_keys: set[str]) -> bool:
    if condition in (None, {}, []):
        return True
    if not isinstance(condition, dict):
        return False
    if "all" in condition or "any" in condition:
        key = "all" if "all" in condition else "any"
        values = condition.get(key)
        return isinstance(values, list) and bool(values) and all(
            _condition_valid(value, field_keys) for value in values
        )
    field = str(condition.get("field") or "")
    operator = str(condition.get("operator") or "equals")
    return bool(field in field_keys and operator in {
        "equals", "not_equals", "in", "not_in", "exists", "not_exists",
        "greater_than", "greater_than_or_equal", "less_than", "less_than_or_equal",
    })


def _validation_tuning_valid(validation: dict[str, Any]) -> bool:
    """Optional field-validation tuning must be usable, or absent.

    `model_confidence_min`, `min_tokens`, `max_tokens` and
    `confirmation_policy` let the graph tell the runtime how much to trust
    the model on one field. A publication that declares them wrong is worse
    than one that declares nothing, because the runtime would fall back to
    its generic default while the operator believes the graph is in charge.
    So the publish gate refuses it instead.
    """
    confidence = validation.get("model_confidence_min")
    if confidence is not None:
        try:
            if not 0.0 <= float(confidence) <= 1.0:
                return False
        except (TypeError, ValueError):
            return False
    bounds: list[int | None] = []
    for raw in (validation.get("min_tokens"), validation.get("max_tokens")):
        if raw is None:
            bounds.append(None)
            continue
        try:
            bounds.append(int(raw))
        except (TypeError, ValueError):
            return False
    minimum, maximum = bounds
    if minimum is not None and minimum < 1:
        return False
    if maximum is not None and maximum < (minimum or 1):
        return False
    policy = validation.get("confirmation_policy")
    if policy is not None and str(policy) not in {"always", "last_resort"}:
        return False
    return True


def _compiled_field_validation(item: dict[str, Any]) -> dict[str, Any]:
    raw = item.get("validation")
    validation = dict(raw) if isinstance(raw, dict) else {}
    mode = str(
        validation.get("mode")
        or ("schema" if isinstance(item.get("value_schema"), dict) and item.get("value_schema") else "")
    ).strip().lower()
    if mode == "enum":
        raw_values = validation.get("values") or validation.get("canonical_values") or []
        values: list[dict[str, Any]] = []
        for raw_value in raw_values if isinstance(raw_values, list) else []:
            if isinstance(raw_value, dict):
                canonical = raw_value.get("value", raw_value.get("canonical"))
                aliases = raw_value.get("aliases") or []
            else:
                canonical = raw_value
                aliases = []
            if canonical in (None, ""):
                continue
            values.append({
                "value": canonical,
                "aliases": [str(alias) for alias in aliases if str(alias).strip()],
            })
        validation["values"] = values
    validation["mode"] = mode
    return validation


def _compiled_confirmation_policy(item: dict[str, Any]) -> dict[str, str]:
    authored = item.get("confirmation")
    authored = authored if isinstance(authored, dict) else {}
    return {
        "capability": str(authored.get("capability") or "common_fact"),
        "template_key": str(authored.get("template_key") or "fact"),
        "context_guidance": str(authored.get("context_guidance") or ""),
    }


def _field_declarations(
    node: dict[str, Any],
    *,
    branch_node: dict[str, Any] | None = None,
    persona_node: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    data = node.get("data") or {}
    qualification = data.get("qualification")
    sources: list[Any] = []
    if isinstance(data.get("fields"), list):
        sources.extend(data["fields"])
    if isinstance(qualification, dict) and isinstance(qualification.get("fields"), list):
        sources.extend(qualification["fields"])
    result: list[dict[str, Any]] = []
    for raw in sources:
        item = {"key": raw} if isinstance(raw, str) else dict(raw) if isinstance(raw, dict) else {}
        key = str(item.get("key") or "").strip()
        if not key:
            continue
        validation = _compiled_field_validation(item)
        accepted = item.get("accepted_statuses")
        if not isinstance(accepted, list) or not accepted:
            accepted = ["known", *(["unknown"] if item.get("accepts_unknown") else [])]
        if validation.get("semantic_type") == "human_full_name":
            accepted = [*accepted, "needs_confirmation", "invalid"]
        result.append({
            **item,
            "key": key,
            "owner_node_id": str(
                graph_conversation_contract.resolve_field_owner_node_id(
                    item, branch_node=branch_node, persona_node=persona_node,
                    declaration_node=node,
                ) or node["id"]
            ),
            "question_node_id": str(item.get("question_node_id") or "") or None,
            "required": item.get("required", True) is True,
            "accepted_statuses": list(dict.fromkeys(str(value) for value in accepted)),
            "value_schema": item.get("value_schema") if isinstance(item.get("value_schema"), dict) else {},
            "validation": validation,
            "normalization": item.get("normalization") or item.get("normalization_hint"),
            "depends_on": [str(value) for value in item.get("depends_on") or [] if value],
            "condition": item.get("condition"),
            "priority": float(item.get("priority") or 0.5),
            "overwrite_policy": str(item.get("overwrite_policy") or "explicit_correction"),
            "confirmation": _compiled_confirmation_policy(item),
        })
    return result


def _question_text(node: dict[str, Any]) -> str:
    data = node.get("data") or {}
    content = data.get("content") if isinstance(data.get("content"), dict) else {}
    return str(content.get("question") or data.get("question") or node.get("title") or "").strip()


def _question_paraphrases(node: dict[str, Any]) -> list[str]:
    """Alternative published wordings for the same question.

    The runtime may need to ask one field more than once. Without published
    alternatives its only options are the identical sentence or silence, so
    the graph -- never the code -- owns every other way of asking.
    """
    data = node.get("data") or {}
    content = data.get("content") if isinstance(data.get("content"), dict) else {}
    values = content.get("paraphrases") or data.get("paraphrases") or []
    if not isinstance(values, list):
        return []
    return [text for value in values if isinstance(value, str) and (text := value.strip())]


def _faq_question_answer(node: dict[str, Any]) -> tuple[str, str]:
    """Return only explicit FAQ fields; a display title is not a FAQ question."""
    data = node.get("data") or {}
    content = data.get("content") if isinstance(data.get("content"), dict) else {}
    question = str(content.get("question") or data.get("question") or "").strip()
    answer = str(
        content.get("answer") or content.get("resposta") or data.get("answer") or ""
    ).strip()
    return question, answer


def _self_evidenced_faq(node: dict[str, Any]) -> bool:
    claims = (node.get("data") or {}).get("claims")
    return isinstance(claims, list) and any(
        isinstance(claim, dict)
        and claim.get("policy")
        and [str(value) for value in claim.get("evidence_node_ids") or []] == [node["id"]]
        for claim in claims
    )


def _topological_fields(
    fields: list[dict[str, Any]], *, preferred_order: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Order fields by dependencies, then by the graph-declared sequence."""
    by_key = {field["key"]: field for field in fields}
    incoming = {key: set(field.get("depends_on") or []) for key, field in by_key.items()}
    preferred = {
        str(key): index for index, key in enumerate(preferred_order or []) if key
    }

    def order_key(key: str) -> tuple[int, float, str]:
        field = by_key[key]
        return (
            preferred.get(key, len(preferred)),
            -float(field.get("priority") or 0),
            key,
        )

    errors = [
        f"field_dependency_missing:{key}:{dependency}"
        for key, dependencies in incoming.items() for dependency in dependencies
        if dependency not in by_key
    ]
    queue = deque(sorted(
        (key for key, dependencies in incoming.items() if not dependencies),
        key=order_key,
    ))
    ordered: list[dict[str, Any]] = []
    while queue:
        key = queue.popleft()
        ordered.append(by_key[key])
        for candidate in sorted(incoming, key=order_key):
            if key in incoming[candidate]:
                incoming[candidate].remove(key)
                if not incoming[candidate] and by_key[candidate] not in ordered and candidate not in queue:
                    queue.append(candidate)
        queue = deque(sorted(queue, key=order_key))
    cyclic = sorted(key for key, dependencies in incoming.items() if dependencies)
    errors.extend(f"field_dependency_cycle:{key}" for key in cyclic)
    return ordered, errors


def branch_selection_field_key(
    contracts: dict[str, dict[str, Any]], persona_node: dict[str, Any] | None,
) -> str | None:
    """Resolve the graph-declared field that selects a branch.

    Sales graphs can declare ``conversation_policy.branch_selection.field_key``.
    Appointment graphs retain the legacy derivation from required_fields as a
    compatibility adapter. No client, product, service or audience key is
    hardcoded here.
    """
    if not contracts or not persona_node:
        return None
    by_anchor = {
        anchor: {
            str(field.get("key") or ""): field
            for field in contract.get("fields") or []
        }
        for anchor, contract in contracts.items()
    }
    persona_data = persona_node.get("data") or {}
    conversation_policy = persona_data.get("conversation_policy") or {}
    declared_selection = conversation_policy.get("branch_selection") or {}
    explicit_key = str(declared_selection.get("field_key") or "").strip()
    appointment_policy = persona_data.get("appointment_policy") or {}
    candidates = [
        explicit_key,
        *(str(value or "") for value in appointment_policy.get("required_fields") or []),
    ]
    for key in candidates:
        if not key:
            continue
        # Support branches may omit the selector. Where declared, it must be
        # owned by that branch, and at least two branches must participate.
        declarantes = {
            anchor: campos[key] for anchor, campos in by_anchor.items() if key in campos
        }
        if len(declarantes) < 2:
            continue
        if all(
            str(field.get("owner_node_id") or "") == anchor
            for anchor, field in declarantes.items()
        ):
            return key
    return None


def _with_confirmable_status(field: dict[str, Any]) -> dict[str, Any]:
    """Autoriza `needs_confirmation` no campo de selecao de galho.

    O runtime so promove o servico a `known` com evidencia exata; uma resolucao
    aproximada vira `needs_confirmation` ate o cliente confirmar -- e o
    contrato precisa aceitar esse estado intermediario, senao o fato fica num
    limbo que trava a conversa: nao e `known` (nenhum galho ativo, toda
    proposta com `keep` rejeitada) e nao esta faltando (a qualificacao parece
    completa). Mesmo tratamento que `human_full_name` ja recebe.
    """
    accepted = [
        str(value) for value in field.get("accepted_statuses") or [] if value
    ] or ["known"]
    return {
        **field,
        "accepted_statuses": list(dict.fromkeys([*accepted, "needs_confirmation"])),
        "branch_selection_field": True,
        "confirmation": {
            **dict(field.get("confirmation") or {}),
            "capability": "branch_selector",
            "template_key": "service_selection",
        },
    }



def _selector_shared_field(field: dict[str, Any], persona_id: str) -> dict[str, Any]:
    """O campo de servico visto de fora de qualquer galho."""
    return _with_confirmable_status({
        **field,
        "owner_node_id": persona_id,
        "scope": "persona",
        "branch_selection_field": True,
        "overwrite_policy": "never",
    })


def _common_persona_contract(
    contracts: dict[str, dict[str, Any]], persona_node: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compile fields that are safe to collect before selecting a service."""
    if not contracts or not persona_node:
        return {}
    anchors = list(contracts)
    first = contracts[anchors[0]]
    by_anchor = {
        anchor: {
            str(field.get("key") or ""): field
            for field in contract.get("fields") or []
        }
        for anchor, contract in contracts.items()
    }
    persona_id = str(persona_node.get("id") or "")
    persona_data = persona_node.get("data") or {}
    persona_policy = persona_data.get("appointment_policy") or {}
    branch_selection = (
        (persona_data.get("conversation_policy") or {}).get("branch_selection")
        or {}
    )
    preferred = [
        str(value) for value in persona_policy.get("required_fields") or [] if value
    ]
    explicit_selector = str(branch_selection.get("field_key") or "").strip()
    if explicit_selector and explicit_selector not in preferred:
        preferred.insert(0, explicit_selector)

    selector_key = branch_selection_field_key(contracts, persona_node)

    shared: list[dict[str, Any]] = []
    for field in first.get("fields") or []:
        key = str(field.get("key") or "")
        declarations = [by_anchor[anchor].get(key) for anchor in anchors]
        if not key or not all(declarations):
            continue
        if all(
            str(item.get("owner_node_id") or "") == persona_id
            for item in declarations
        ):
            shared.append(dict(field))
        elif key == selector_key:
            shared.append(_selector_shared_field(field, persona_id))

    # O seletor pode nao existir no anchor varrido acima (galhos de suporte nao
    # o declaram). Sem isto, o campo de servico ficava fora do contrato comum e
    # a qualificacao parecia completa so com o nome.
    if selector_key and not any(
        str(field.get("key") or "") == selector_key for field in shared
    ):
        declarado = next(
            (campos[selector_key] for campos in by_anchor.values()
             if selector_key in campos),
            None,
        )
        if declarado:
            shared.append(_selector_shared_field(declarado, persona_id))

    order = {key: index for index, key in enumerate(preferred)}
    shared.sort(key=lambda field: (
        order.get(str(field.get("key") or ""), len(order)),
        -float(field.get("priority") or 0),
        str(field.get("key") or ""),
    ))
    question_ids = {
        str(field.get("question_node_id") or "")
        for field in shared if field.get("question_node_id")
    }
    questions: dict[str, dict[str, Any]] = {}
    for contract in contracts.values():
        for question_id in question_ids:
            question = (contract.get("questions") or {}).get(question_id)
            if question:
                questions[question_id] = question
    # Uma claim autorizada por TODOS os galhos nao depende de qual galho o
    # cliente escolher -- e o caso das FAQs projetadas de `global_context`.
    # Sem isto o contrato comum publicava `claims: []`, e o cliente que compra
    # perguntando ("como funciona o PPF?") recebia
    # `claim_not_authorized`/`claim_evidence_not_authorized`: a proposta do
    # modelo era descartada inteira e o turno caia em `published_fallback`
    # antes de qualquer servico poder ser escolhido. Precos e regras de um
    # servico especifico continuam de fora -- so um galho os declara.
    shared_claims = [
        claim for claim in first.get("claims") or []
        if all(
            any(
                canonical_checksum(candidate) == canonical_checksum(claim)
                for candidate in contract.get("claims") or []
            )
            for contract in contracts.values()
        )
    ]
    shared_closure = set.intersection(*(
        {str(value) for value in contract.get("closure_node_ids") or []}
        for contract in contracts.values()
    )) if contracts else set()
    # A evidencia de uma claim compartilhada precisa estar dentro do closure
    # comum, senao o checker rejeita por `claim_node_evidence_outside_package`.
    claim_evidence = {
        str(value)
        for claim in shared_claims
        for value in claim.get("evidence_node_ids") or []
    }
    closure = list(dict.fromkeys([
        persona_id,
        *[str(field.get("owner_node_id") or "") for field in shared],
        *question_ids,
        *sorted(shared_closure & claim_evidence),
    ]))
    required = [
        str(field.get("key") or "")
        for field in shared if field.get("required", True)
    ]
    return {
        "branch_anchor_node_id": None,
        "branch_path_checksum": "",
        "closure_checksum": canonical_checksum(closure),
        "closure_node_ids": [value for value in closure if value],
        "fields": shared,
        "required_fields": required,
        "questions": questions,
        "claims": shared_claims,
        "completion": {"required_fields": required},
        "conversation_policy": dict(first.get("conversation_policy") or {}),
        "field_labels": dict(first.get("field_labels") or {}),
        "handoff": {},
        "handoff_rule_node_ids": [],
        "handoff_rules": [],
        "collection_policy": dict(first.get("collection_policy") or {}),
        "compiler_version": COMPILER_VERSION,
    }


def compile_graph(
    *, persona: dict[str, Any], node_rows: list[dict[str, Any]], edge_rows: list[dict[str, Any]],
    embedding_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    db_to_stable: dict[str, str] = {}
    nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in node_rows:
        if not _published(row):
            continue
        stable_id = _stable_node_id(row)
        if stable_id in seen:
            errors.append(f"duplicate_node_id:{stable_id}")
            continue
        seen.add(stable_id)
        db_to_stable[str(row.get("id"))] = stable_id
        data = _runtime_node_metadata(row)
        nodes.append({
            "id": stable_id,
            "projection_node_id": str(row.get("id")),
            "node_type": (
                "embedded" if str(row.get("node_type") or "").lower() == "embed"
                else str(row.get("node_type") or "knowledge").lower()
            ),
            "slug": str(row.get("slug") or stable_id),
            "title": str(row.get("title") or row.get("slug") or stable_id),
            "summary": str(row.get("summary") or ""),
            "tags": list(row.get("tags") or []),
            "status": str(row.get("status") or ""),
            "data": data,
        })
    persona_node = _select_persona_node(nodes, errors)
    if persona_node is not None:
        canonical_persona_id = str(persona_node["id"])
        nodes = [
            node
            for node in nodes
            if node["node_type"] != "persona" or node["id"] == canonical_persona_id
        ]
        retained_ids = {str(node["id"]) for node in nodes}
        db_to_stable = {
            projection_id: stable_id
            for projection_id, stable_id in db_to_stable.items()
            if stable_id in retained_ids
        }
    nodes.sort(key=lambda node: str(node["id"]))
    node_by_id = {node["id"]: node for node in nodes}
    if not nodes:
        errors.append("publication_has_no_published_nodes")

    edges: list[dict[str, Any]] = []
    for row in edge_rows:
        source_metadata = _metadata(row)
        if source_metadata.get("active", True) is False:
            continue
        source = db_to_stable.get(str(row.get("source_node_id")))
        target = db_to_stable.get(str(row.get("target_node_id")))
        if not source or not target:
            continue
        relation = str(row.get("relation_type") or "contains")
        edges.append({
            "id": _stable_edge_id(row), "source": source, "target": target,
            "relation_type": relation, "weight": float(row.get("weight") or 1),
            "metadata": _runtime_edge_metadata(row),
            "primary": relation in STRUCTURAL_RELATIONS,
        })
    edges.sort(key=lambda edge: str(edge["id"]))

    parents: dict[str, tuple[str, str]] = {}
    children: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if not edge["primary"]:
            continue
        target = edge["target"]
        if target in parents and parents[target][0] != edge["source"]:
            errors.append(f"ambiguous_primary_path:{target}")
            continue
        parents[target] = (edge["source"], edge["id"])
        children[edge["source"]].append(target)

    coordinates: dict[str, dict[str, Any]] = {}
    for node_id in sorted(node_by_id):
        path_nodes = [node_id]
        path_edges: list[str] = []
        current = node_id
        visited = {current}
        while current in parents:
            parent_id, edge_id = parents[current]
            if parent_id in visited:
                errors.append(f"primary_path_cycle:{node_id}:{parent_id}")
                break
            visited.add(parent_id)
            path_nodes.insert(0, parent_id)
            path_edges.insert(0, edge_id)
            current = parent_id
        anchors = [candidate for candidate in path_nodes if is_branch_anchor(node_by_id[candidate])]
        anchor = anchors[-1] if anchors else None
        path_payload = {"path_node_ids": path_nodes, "path_edge_ids": path_edges}
        coordinates[node_id] = {
            "node_id": node_id,
            "branch_anchor_node_id": anchor,
            **path_payload,
            "depth": len(path_edges),
            "path_checksum": canonical_checksum(path_payload),
        }

    anchors = sorted(node["id"] for node in nodes if is_branch_anchor(node))
    if not anchors:
        errors.append("publication_has_no_branch_anchor_capability")

    def descendants(root: str) -> set[str]:
        result: set[str] = set()
        queue = deque([(root, 0)])
        while queue:
            current, _distance = queue.popleft()
            if current in result:
                continue
            result.add(current)
            queue.extend((child, _distance + 1) for child in children.get(current, []))
        return result

    persona_data = (persona_node or {}).get("data") or {}
    conversation_policy = persona_data.get("conversation_policy")
    conversation_policy = (
        conversation_policy if isinstance(conversation_policy, dict) else {}
    )
    appointment_policy = persona_data.get("appointment_policy")
    appointment_policy = (
        appointment_policy if isinstance(appointment_policy, dict) else {}
    )
    embedded_faq_ids = {
        edge["source"]
        for edge in edges
        if node_by_id.get(edge["source"], {}).get("node_type") == "faq"
        and node_by_id.get(edge["target"], {}).get("node_type") == "embedded"
    }
    eligible_faq_ids: set[str] = set()
    for node_id in embedded_faq_ids:
        node = node_by_id[node_id]
        data = node.get("data") or {}
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        role = str(metadata.get("role") or data.get("role") or "")
        if (
            node.get("node_type") == "faq"
            and role != "qualification_question"
            and all(_faq_question_answer(node))
            and _self_evidenced_faq(node)
        ):
            eligible_faq_ids.add(node_id)
    global_context_members: set[str] = set()
    for node in nodes:
        if _capability(node["data"], "global_context"):
            global_context_members.update(descendants(node["id"]))

    memberships: dict[str, dict[str, dict[str, Any]]] = {}
    contracts: dict[str, dict[str, Any]] = {}
    for anchor in anchors:
        primary_members = descendants(anchor)
        path = coordinates[anchor]["path_node_ids"]
        primary_members.update(path)
        reasons: dict[str, str] = {
            node_id: ("branch_descendant" if node_id not in path else "branch_ancestor")
            for node_id in primary_members
        }
        for node_id in sorted(global_context_members):
            reasons[node_id] = (
                "global_context_capability"
                if _capability(node_by_id[node_id]["data"], "global_context")
                else "global_context_descendant"
            )
        # A semantic edge expands a branch only when the graph explicitly says
        # it participates in branch scope.  In particular, publication edges
        # to a shared Embedded action must never connect otherwise independent
        # product closures through that shared sink.
        frontier = set(reasons)
        for edge in edges:
            if edge["primary"] or not (
                (edge.get("metadata") or {}).get("include_in_branch") is True
                or edge["relation_type"] == "applies_to"
            ):
                continue
            if edge["source"] in frontier and edge["target"] not in reasons:
                reasons[edge["target"]] = f"semantic:{edge['relation_type']}"
            elif edge["target"] in frontier and edge["source"] not in reasons:
                reasons[edge["source"]] = f"semantic:{edge['relation_type']}"

        fields_by_key: dict[str, dict[str, Any]] = {}
        for node_id in sorted(reasons):
            node = node_by_id[node_id]
            for field in _field_declarations(
                node, branch_node=node_by_id[anchor], persona_node=persona_node
            ):
                prior = fields_by_key.get(field["key"])
                if prior and prior != field:
                    errors.append(f"ambiguous_field_declaration:{anchor}:{field['key']}")
                    continue
                fields_by_key[field["key"]] = field
        # Question references are explicit closure members, but a question
        # already rooted below another anchor cannot be imported into this
        # branch. Root-level questions may be shared deliberately.
        for field in fields_by_key.values():
            question_id = field.get("question_node_id")
            question_anchor = (
                (coordinates.get(question_id) or {}).get("branch_anchor_node_id")
                if question_id else None
            )
            if question_anchor and question_anchor != anchor:
                errors.append(
                    f"field_question_wrong_scope:{anchor}:{field['key']}:{question_id}"
                )
            elif question_id:
                reasons.setdefault(question_id, "field_question")

        raw_completion = node_by_id[anchor]["data"].get("completion")
        raw_completion = raw_completion if isinstance(raw_completion, dict) else {}
        persona_policy = (
            ((persona_node or {}).get("data") or {}).get("appointment_policy") or {}
        )
        preferred_field_order = list(
            raw_completion.get("required_fields")
            or persona_policy.get("required_fields")
            or []
        )
        ordered_fields, dependency_errors = _topological_fields(
            list(fields_by_key.values()), preferred_order=preferred_field_order,
        )
        errors.extend(f"{anchor}:{error}" for error in dependency_errors)
        field_keys = set(fields_by_key)
        for field in ordered_fields:
            owner = field["owner_node_id"]
            question_id = field.get("question_node_id")
            if owner not in reasons:
                errors.append(f"field_owner_unreachable:{anchor}:{field['key']}:{owner}")
            if field.get("scope") == "persona" and persona_node is None:
                errors.append(
                    f"field_owner_persona_scope_no_persona_node:{anchor}:{field['key']}"
                )
            if field.get("scope") not in {None, "persona", "branch", "declaration"}:
                errors.append(f"invalid_field_scope:{anchor}:{field['key']}")
            validation = field.get("validation") or {}
            validation_mode = str(validation.get("mode") or "")
            if validation_mode not in {"enum", "schema", "semantic"}:
                errors.append(f"field_validation_mode_missing:{anchor}:{field['key']}")
            elif validation_mode == "enum":
                values = validation.get("values") or []
                canonical = [item.get("value") for item in values if isinstance(item, dict)]
                if not canonical or len({json.dumps(value, sort_keys=True) for value in canonical}) != len(canonical):
                    errors.append(f"field_enum_values_invalid:{anchor}:{field['key']}")
            elif validation_mode == "schema" and not field.get("value_schema"):
                errors.append(f"field_schema_missing:{anchor}:{field['key']}")
            elif validation_mode == "semantic":
                description = str(validation.get("description") or "").strip()
                definition = (
                    validation.get("semantic_type")
                    or validation.get("examples")
                    or validation.get("rules")
                )
                if not description or not definition:
                    errors.append(f"field_semantic_definition_missing:{anchor}:{field['key']}")
            if not _validation_tuning_valid(validation):
                errors.append(f"field_validation_tuning_invalid:{anchor}:{field['key']}")
            if not set(field["accepted_statuses"]).issubset(FACT_STATUSES):
                errors.append(f"invalid_accepted_statuses:{anchor}:{field['key']}")
            if field["overwrite_policy"] not in {"never", "explicit_correction", "higher_confidence", "always"}:
                errors.append(f"invalid_overwrite_policy:{anchor}:{field['key']}")
            if not _condition_valid(field.get("condition"), field_keys):
                errors.append(f"invalid_field_condition:{anchor}:{field['key']}")
            if not question_id:
                errors.append(f"field_question_missing:{anchor}:{field['key']}")
            elif question_id not in node_by_id:
                errors.append(f"field_question_not_published:{anchor}:{field['key']}:{question_id}")
            elif question_id not in reasons:
                errors.append(f"field_question_wrong_scope:{anchor}:{field['key']}:{question_id}")
            elif not _question_text(node_by_id[question_id]):
                errors.append(f"field_question_empty:{anchor}:{field['key']}:{question_id}")

        closure = sorted(reasons)
        member_rows: dict[str, dict[str, Any]] = {}
        anchor_depth = coordinates[anchor]["depth"]
        for node_id in closure:
            coord = coordinates.get(node_id) or {}
            if node_id in path:
                distance = abs(anchor_depth - int(coord.get("depth") or 0))
            elif anchor in (coord.get("path_node_ids") or []):
                distance = int(coord.get("depth") or 0) - anchor_depth
            else:
                distance = 1
            member_rows[node_id] = {
                "branch_node_id": anchor,
                "node_id": node_id,
                "graph_distance": max(0, distance),
                "inclusion_reason": reasons[node_id],
                "structural_weight": round(1 / (1 + max(0, distance)), 6),
            }
        memberships[anchor] = member_rows

        handoff_rules = [
            node_id for node_id in closure
            if _capability(node_by_id[node_id]["data"], "handoff_rule")
            or isinstance(node_by_id[node_id]["data"].get("handoff_rule"), dict)
        ]
        handoff = node_by_id[anchor]["data"].get("handoff")
        if handoff and not handoff_rules:
            errors.append(f"handoff_without_published_rule:{anchor}")
        handoff_contract_rules = [
            {
                "node_id": node_id,
                **dict(node_by_id[node_id]["data"].get("handoff_rule") or {}),
            }
            for node_id in handoff_rules
        ]
        if isinstance(handoff, dict):
            for key, value in handoff.items():
                if (
                    isinstance(value, str)
                    and (key.startswith("on_") or key.endswith("rule_node_id"))
                    and value not in handoff_rules
                ):
                    errors.append(f"handoff_references_non_rule:{anchor}:{value}")

        claims: list[dict[str, Any]] = []
        for node_id in closure:
            raw_claims = node_by_id[node_id]["data"].get("claims") or []
            for raw in raw_claims if isinstance(raw_claims, list) else []:
                if not isinstance(raw, dict):
                    continue
                claim = {**raw, "owner_node_id": str(raw.get("owner_node_id") or node_id)}
                if not claim.get("policy"):
                    errors.append(f"commercial_claim_without_policy:{anchor}:{node_id}")
                evidence_nodes = [str(value) for value in claim.get("evidence_node_ids") or []]
                if not evidence_nodes:
                    errors.append(f"commercial_claim_without_evidence:{anchor}:{node_id}")
                for evidence_node_id in evidence_nodes:
                    if evidence_node_id not in reasons:
                        errors.append(
                            f"commercial_claim_evidence_outside_scope:{anchor}:{node_id}:{evidence_node_id}"
                        )
                claims.append(claim)

        for node_id in closure:
            node = node_by_id[node_id]
            data = node.get("data") or {}
            metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
            if node.get("node_type") != "faq" or node_id not in embedded_faq_ids or (
                metadata.get("role") or data.get("role")
            ) == "qualification_question":
                continue
            if not str(data.get("answer") or "").strip():
                continue
            raw_claims = data.get("claims")
            if not isinstance(raw_claims, list) or not raw_claims:
                errors.append(f"factual_faq_without_claim:{anchor}:{node_id}")
                continue
            for raw in raw_claims:
                if not isinstance(raw, dict):
                    errors.append(f"factual_faq_invalid_claim:{anchor}:{node_id}")
                    continue
                if not raw.get("policy"):
                    errors.append(f"factual_faq_claim_without_policy:{anchor}:{node_id}")
                evidence = [str(value) for value in raw.get("evidence_node_ids") or []]
                if evidence != [node_id]:
                    errors.append(f"factual_faq_claim_without_self_evidence:{anchor}:{node_id}")

        completion = raw_completion or {"required_fields": [
            field["key"] for field in ordered_fields if field["required"]
        ]}
        closure_checksum = canonical_checksum(closure)
        contract = {
            "branch_anchor_node_id": anchor,
            "branch_path_checksum": coordinates[anchor]["path_checksum"],
            "closure_checksum": closure_checksum,
            "closure_node_ids": closure,
            "fields": ordered_fields,
            "required_fields": [field["key"] for field in ordered_fields if field["required"]],
            "questions": {
                field["question_node_id"]: {
                    "field_key": field["key"],
                    "text": _question_text(node_by_id[field["question_node_id"]]),
                    "paraphrases": _question_paraphrases(node_by_id[field["question_node_id"]]),
                    "depends_on": field["depends_on"],
                    "condition": field.get("condition"),
                }
                for field in ordered_fields if field.get("question_node_id") in node_by_id
            },
            "claims": claims,
            "completion": completion,
            "conversation_policy": conversation_policy,
            "field_labels": dict(
                conversation_policy.get("field_labels")
                or appointment_policy.get("field_labels")
                or {}
            ),
            "handoff": handoff if isinstance(handoff, dict) else {},
            "handoff_rule_node_ids": handoff_rules,
            "handoff_rules": handoff_contract_rules,
            "eligible_faq_node_ids": sorted(
                node_id for node_id in eligible_faq_ids if node_id in closure
            ),
            "faq_projection_contract": FAQ_PROJECTION_CONTRACT,
            "collection_policy": (
                dict(qualification_policy)
                if isinstance((qualification_policy := (
                    (((persona_node or {}).get("data") or {}).get("conversation_policy") or {})
                    .get("qualification")
                )), dict)
                else {}
            ),
            "compiler_version": COMPILER_VERSION,
        }
        contracts[anchor] = contract

    # Cross-branch field consistency check: same field key should have
    # consistent owner_node_id across branches unless explicitly scoped to branch.
    # This detects the class of bugs fixed 2026-08-10 (nome_cliente divergence).
    all_fields_by_key: dict[str, dict[str, dict[str, Any]]] = {}
    for anchor, contract in contracts.items():
        for field in contract.get("fields") or []:
            key = field["key"]
            if key not in all_fields_by_key:
                all_fields_by_key[key] = {}
            all_fields_by_key[key][anchor] = field

    for field_key, by_anchor in all_fields_by_key.items():
        if len(by_anchor) <= 1:
            continue
        owners = {
            anchor: str(field.get("owner_node_id") or "")
            for anchor, field in by_anchor.items()
        }
        unique_owners = set(owners.values())
        if len(unique_owners) > 1:
            all_locally_scoped = all(
                by_anchor[anchor].get("scope") in {"branch", "declaration"}
                for anchor in by_anchor
            )
            if not all_locally_scoped:
                branches_with_owners = [
                    f"{anchor}(owner={owners[anchor][:8]}..., scope={by_anchor[anchor].get('scope', 'branch')})"
                    for anchor in sorted(by_anchor)
                ]
                errors.append(
                    f"inconsistent_field_owner:{field_key}:{', '.join(branches_with_owners)}"
                )

    declared_branch_selection = str(
        ((conversation_policy.get("branch_selection") or {}).get("field_key")) or ""
    ).strip()
    if (
        declared_branch_selection
        and branch_selection_field_key(contracts, persona_node)
        != declared_branch_selection
    ):
        errors.append(
            f"branch_selection_field_invalid:{declared_branch_selection}"
        )

    if errors:
        raise GraphCompilationError(errors)
    contract_markdown = CONTRACT_DOCUMENT.read_text(encoding="utf-8")
    projected_nodes = {
        node_id: semantic_chunks(node)
        for node_id, node in node_by_id.items()
        if any(node_id in members for members in memberships.values())
    }
    branch_chunk_counts = {
        branch: sum(len(projected_nodes.get(node_id) or []) for node_id in members)
        for branch, members in memberships.items()
    }
    resolved_embedding_profile = dict(embedding_profile or {})
    projection_manifest = {
        "entry_count": sum(bool(chunks) for chunks in projected_nodes.values()),
        "chunk_count": sum(branch_chunk_counts.values()),
        "branch_chunk_counts": branch_chunk_counts,
        "embedding_dimension": int(
            resolved_embedding_profile.get("embedding_dimension") or EMBEDDING_DIMENSION
        ),
        "embedding_provider": str(
            resolved_embedding_profile.get("embedding_provider") or embedding_provider()
        ),
        "embedding_model": str(
            resolved_embedding_profile.get("embedding_model") or embedding_model_name()
        ),
    }
    projected_eligible_faq_ids = sorted(
        node_id for node_id in eligible_faq_ids
        if any(node_id in members for members in memberships.values())
    )
    # O fato de servico e escrito com o galho como dono, entao o contrato de
    # galho tambem precisa aceitar o estado intermediario -- nao so o comum.
    selection_key = branch_selection_field_key(contracts, persona_node)
    if selection_key:
        for contract in contracts.values():
            contract["fields"] = [
                _with_confirmable_status(field)
                if str(field.get("key") or "") == selection_key else field
                for field in contract.get("fields") or []
            ]
    common_contract = _common_persona_contract(contracts, persona_node)
    confirmation_templates = appointment_policy.get("confirmation_templates")
    confirmation_templates = (
        dict(confirmation_templates)
        if isinstance(confirmation_templates, dict) else {}
    )
    service_resolution_policy = appointment_policy.get("service_resolution")
    service_resolution_policy = (
        dict(service_resolution_policy)
        if isinstance(service_resolution_policy, dict)
        else {
            "max_edit_distance": 3,
            "min_text_similarity": 0.80,
            "min_semantic_score": 0.78,
            "min_semantic_margin": 0.08,
        }
    )
    document = {
        "schema_version": "3.0",
        "compiler_version": COMPILER_VERSION,
        "compiler_contract": {
            "path": "api/contracts/graph-agent-runtime-v3.md",
            "checksum": canonical_checksum(contract_markdown),
        },
        "persona": {"id": str(persona["id"]), "slug": str(persona["slug"])},
        "node_by_id": node_by_id,
        "nodes": nodes,
        "edges": edges,
        "parents": {key: value[0] for key, value in parents.items()},
        "children": {key: sorted(value) for key, value in children.items()},
        "coordinates": coordinates,
        "branch_anchors": anchors,
        "branch_memberships": memberships,
        "branch_contracts": contracts,
        "common_contract": common_contract,
        "confirmation_templates": confirmation_templates,
        "service_resolution_policy": service_resolution_policy,
        "faq_projection_contract": FAQ_PROJECTION_CONTRACT,
        "eligible_faq_node_ids": projected_eligible_faq_ids,
        "projection_manifest": projection_manifest,
    }
    document["checksum"] = canonical_checksum(document)
    return document


def semantic_chunks(node: dict[str, Any]) -> list[dict[str, Any]]:
    """Split one node by semantic role instead of arbitrary character windows."""
    data = node.get("data") or {}
    values: list[tuple[str, str]] = []
    faq_question, faq_answer = _faq_question_answer(node)
    if node.get("node_type") == "faq" and faq_question and faq_answer:
        values.append(("faq", f"Pergunta: {faq_question}\nResposta: {faq_answer}"))
    question = _question_text(node)
    if question:
        values.append(("question", question))
    content = data.get("content")
    if isinstance(content, dict):
        answer = str(content.get("answer") or content.get("resposta") or "").strip()
        if answer:
            values.append(("answer", answer))
    aliases = data.get("aliases")
    if isinstance(aliases, list) and aliases:
        values.append(("aliases", "\n".join(str(value) for value in aliases if value)))
    facts = data.get("facts") or data.get("structured_facts")
    if facts:
        values.append(("structured_facts", json.dumps(facts, ensure_ascii=False, sort_keys=True)))
    for key, kind in (
        ("claims", "claims"),
        ("handoff_rule", "rule"),
        ("validators", "validators"),
        ("rules", "rules"),
    ):
        structured = data.get(key)
        if structured:
            values.append((kind, json.dumps(structured, ensure_ascii=False, sort_keys=True)))
    raw_body = data.get("markdown") or (
        data.get("content") if isinstance(data.get("content"), str) else ""
    )
    body = str(raw_body).strip()
    if not body:
        body = "\n".join(value for value in [node.get("title"), node.get("summary")] if value)
    if body:
        # Preserve paragraphs; only long documents receive bounded windows.
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
        current = ""
        for paragraph in paragraphs or [body]:
            if current and len(current) + len(paragraph) + 2 > 1800:
                values.append(("content", current))
                current = paragraph
            else:
                current = f"{current}\n\n{paragraph}".strip()
        if current:
            values.append(("content", current))
    for field in _field_declarations(node):
        values.append(("field_intent", " ".join(filter(None, [
            field["key"], str(field.get("normalization") or ""), str(field.get("question_node_id") or "")
        ]))))
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for kind, text in values:
        normalized = text.strip()
        checksum = canonical_checksum({"kind": kind, "text": normalized})
        if not normalized or checksum in seen:
            continue
        seen.add(checksum)
        deduped.append({"kind": kind, "text": normalized, "checksum": checksum})
    return deduped


def _local_embeddings(
    texts: list[str], *, input_type: str, model_name: str | None = None,
) -> list[list[float]]:
    global _local_embedding_model, _local_embedding_model_name
    from fastembed import TextEmbedding

    resolved_model_name = str(model_name or embedding_model_name())
    if _local_embedding_model is None or _local_embedding_model_name != resolved_model_name:
        _local_embedding_model = TextEmbedding(
            model_name=resolved_model_name,
            cache_dir=(
                os.environ.get("FASTEMBED_CACHE_PATH")
                or "/opt/venv/fastembed-cache"
            ),
        )
        _local_embedding_model_name = resolved_model_name
    method_name = "query_embed" if input_type == "query" else "passage_embed"
    method = getattr(_local_embedding_model, method_name, None)
    generated = method(texts) if callable(method) else _local_embedding_model.embed(texts)
    return [_fit_embedding_dimension(vector) for vector in generated]


def warm_embedding_model() -> bool:
    """Initialize the configured local model once during application startup."""
    if embedding_provider() != "local":
        return False
    _local_embeddings(["brain-ai-warmup"], input_type="query")
    return True


def generate_embeddings(
    texts: list[str],
    *,
    input_type: str = "passage",
) -> list[list[float]]:
    if not texts:
        return []
    provider = embedding_provider()
    if provider == "local":
        vectors = _local_embeddings(texts, input_type=input_type)
    else:
        key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is required by the configured embedding provider"
            )
        from openai import OpenAI

        response = OpenAI(api_key=key).embeddings.create(
            model=embedding_model_name(), input=texts, dimensions=EMBEDDING_DIMENSION
        )
        vectors = [
            _fit_embedding_dimension(item.embedding)
            for item in sorted(response.data, key=lambda item: item.index)
        ]
    if len(vectors) != len(texts):
        raise RuntimeError("embedding provider returned an invalid batch size")
    return vectors


def generate_embeddings_for_profile(
    texts: list[str], *, profile: dict[str, Any], input_type: str = "passage",
) -> list[list[float]]:
    """Generate vectors with the exact provider/model approved in the plan."""
    if not texts:
        return []
    provider = str(profile.get("embedding_provider") or "").strip().lower()
    model = str(profile.get("embedding_model") or "").strip()
    dimension = int(profile.get("embedding_dimension") or 0)
    if dimension != EMBEDDING_DIMENSION:
        raise RuntimeError("approved embedding dimension does not match runtime")
    if provider == "local":
        vectors = _local_embeddings(
            texts, input_type=input_type, model_name=model
        )
    elif provider == "openai":
        key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required by the approved embedding profile")
        from openai import OpenAI

        response = OpenAI(api_key=key).embeddings.create(
            model=model, input=texts, dimensions=EMBEDDING_DIMENSION
        )
        vectors = [
            _fit_embedding_dimension(item.embedding)
            for item in sorted(response.data, key=lambda item: item.index)
        ]
    else:
        raise RuntimeError(f"unsupported approved embedding provider: {provider}")
    if len(vectors) != len(texts):
        raise RuntimeError("embedding provider returned an invalid batch size")
    return vectors


def openai_embeddings(texts: list[str]) -> list[list[float]]:
    """Backward-compatible passage embedding entry point."""
    return generate_embeddings(texts, input_type="passage")


def query_embeddings(texts: list[str]) -> list[list[float]]:
    return generate_embeddings(texts, input_type="query")


def compile_persona_publication(
    persona_slug: str,
    *,
    activate: bool = True,
    embedder: Callable[[list[str]], list[list[float]]] | None = None,
    embedding_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile, persist, embed, and atomically activate one persona snapshot."""
    persona = supabase_client.get_persona(persona_slug)
    if not persona:
        raise LookupError(f"persona not found: {persona_slug}")
    node_rows, edge_rows = supabase_client.list_all_knowledge_graph(
        persona_id=str(persona["id"]), limit_nodes=10000
    )
    document = compile_graph(
        persona=persona,
        node_rows=node_rows,
        edge_rows=edge_rows,
        embedding_profile=embedding_profile,
    )
    client = supabase_client.get_client()
    existing_response = (
        client.table("graph_publications").select("*")
        .eq("persona_id", persona["id"]).eq("checksum", document["checksum"])
        .maybe_single().execute()
    )
    existing = existing_response.data if existing_response is not None else None
    if existing:
        publication = existing
        if activate and publication.get("status") != "active":
            activation = client.rpc("activate_graph_publication_v3", {"p_publication_id": publication["id"]}).execute().data
            return {"publication": publication, "activation": activation, "cached": True}
        return {"publication": publication, "activation": None, "cached": True}
    latest = (
        client.table("graph_publications").select("version")
        .eq("persona_id", persona["id"]).order("version", desc=True).limit(1).execute().data or []
    )
    version = int(latest[0]["version"] if latest else 0) + 1
    publication = (client.table("graph_publications").insert({
        "persona_id": persona["id"], "version": version,
        "checksum": document["checksum"], "document_json": document,
        "status": "building", "compiler_version": COMPILER_VERSION,
    }).execute().data or [None])[0]
    if not publication:
        raise RuntimeError("graph publication insert returned no row")
    publication_id = publication["id"]
    try:
        coordinates = [{**coordinate, "publication_id": publication_id}
                       for coordinate in document["coordinates"].values()]
        client.table("graph_node_coordinates").insert(coordinates).execute()
        memberships = [
            {**member, "publication_id": publication_id}
            for values in document["branch_memberships"].values() for member in values.values()
        ]
        client.table("graph_branch_memberships").insert(memberships).execute()
        contracts = [{
            "publication_id": publication_id, "branch_node_id": branch,
            "path_checksum": contract["branch_path_checksum"],
            "closure_checksum": contract["closure_checksum"],
            "contract_json": contract, "compiler_version": COMPILER_VERSION,
        } for branch, contract in document["branch_contracts"].items()]
        client.table("graph_branch_contracts").insert(contracts).execute()

        embedding_model = str(
            document["projection_manifest"].get("embedding_model")
            or embedding_model_name()
        )
        batch_embed = embedder or (
            (lambda texts: generate_embeddings_for_profile(
                texts, profile=embedding_profile, input_type="passage"
            ))
            if embedding_profile else openai_embeddings
        )
        for node_id, node in document["node_by_id"].items():
            chunks = semantic_chunks(node)
            if not chunks:
                continue
            branch_ids = sorted(
                branch for branch, members in document["branch_memberships"].items()
                if node_id in members
            )
            if not branch_ids:
                continue
            vectors = batch_embed([chunk["text"] for chunk in chunks])
            if len(vectors) != len(chunks):
                raise RuntimeError(f"embedding count mismatch for {node_id}")
            entry = (client.table("knowledge_rag_entries").insert({
                "persona_id": persona["id"], "publication_id": publication_id,
                "source_graph_node_id": node_id, "source_node_id": node["projection_node_id"],
                "content_type": (
                    node["node_type"] if node["node_type"] in RAG_CONTENT_TYPES else "general_note"
                ), "semantic_level": 50,
                "title": node["title"], "content": "\n\n".join(chunk["text"] for chunk in chunks),
                "summary": node.get("summary") or node["title"],
                "canonical_key": f"graphrag-v3:{publication_id}:{node_id}",
                "slug": node["slug"], "language": "pt-BR", "status": "validated",
                "tags": node.get("tags") or [node["node_type"]], "entities": [],
                "products": [], "campaigns": [], "metadata": {
                    "compiler_version": COMPILER_VERSION, "publication_id": publication_id,
                    "source_node_id": node_id, "graph_checksum": document["checksum"],
                }, "embedding_model": embedding_model, "confidence": 1, "importance": 1,
                "validated_at": datetime.now(timezone.utc).isoformat(),
                "projection_status": "ready", "graph_version": version,
                "graph_checksum": document["checksum"],
            }).execute().data or [None])[0]
            if not entry:
                raise RuntimeError(f"RAG entry insert returned no row for {node_id}")
            rows: list[dict[str, Any]] = []
            index = 0
            coordinate = document["coordinates"][node_id]
            fields = _field_declarations(node)
            for branch in branch_ids:
                for chunk, vector in zip(chunks, vectors, strict=True):
                    rows.append({
                        "rag_entry_id": entry["id"], "persona_id": persona["id"],
                        "publication_id": publication_id, "source_graph_node_id": node_id,
                        "source_node_id": node["projection_node_id"],
                        "branch_anchor_node_id": branch, "path_checksum": coordinate["path_checksum"],
                        "chunk_index": index, "chunk_kind": chunk["kind"],
                        "chunk_checksum": chunk["checksum"], "chunk_text": chunk["text"],
                        "chunk_summary": node.get("summary") or node["title"],
                        "embedding": vector, "embedding_model": embedding_model,
                        "embedded_at": datetime.now(timezone.utc).isoformat(),
                        "projection_status": "ready", "graph_version": version,
                        "graph_checksum": document["checksum"], "metadata": {
                            "publication_id": publication_id, "source_node_id": node_id,
                            "branch_anchor_node_id": branch,
                            "path_node_ids": coordinate["path_node_ids"],
                            "path_edge_ids": coordinate["path_edge_ids"],
                            "path_checksum": coordinate["path_checksum"],
                            "field_keys": [field["key"] for field in fields],
                            "priority": max([field["priority"] for field in fields] or [0.5]),
                            "provenance": {"projection_node_id": node["projection_node_id"]},
                            **({
                                "faq_question": _faq_question_answer(node)[0],
                                "faq_answer": _faq_question_answer(node)[1],
                                "faq_aliases": [
                                    str(alias) for alias in (node.get("data") or {}).get("aliases") or []
                                    if str(alias).strip()
                                ],
                                "faq_projection_contract": FAQ_PROJECTION_CONTRACT,
                            } if chunk["kind"] == "faq" else {}),
                        },
                    })
                    index += 1
            if rows:
                client.table("knowledge_rag_chunks").insert(rows).execute()
        client.table("graph_publications").update({"status": "compiled"}).eq("id", publication_id).execute()
        activation = None
        if activate:
            activation = client.rpc("activate_graph_publication_v3", {"p_publication_id": publication_id}).execute().data
        return {"publication": {**publication, "status": "active" if activate else "compiled"},
                "activation": activation, "cached": False}
    except Exception:
        client.table("graph_publications").update({"status": "failed"}).eq("id", publication_id).execute()
        raise
