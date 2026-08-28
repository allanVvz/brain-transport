"""Declarative conversation contracts compiled from a published Graph JSON.

The graph owns branches, fields, questions and handoff rules.  This module
only derives coordinates and checks model proposals against those facts; it
does not contain customer-, vertical- or field-specific conversation logic.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import deque
from copy import deepcopy
from typing import Any, Iterable

from schemas.graph_json_v2 import Edge, EdgeLifecycle, GraphJson, Node, PublicationGrant


FACT_STATUSES = {"known", "unknown", "declined", "needs_confirmation", "invalid"}
APPROVED = {"approved", "active", "validated", "ativo", "embedded"}
BRANCH_TYPES = {"product", "service"}
PERMANENT_TYPES = {"persona", "brand", "briefing", "tone", "rule"}
QUESTION_RELATIONS = {"answers", "answers_question", "references", "applies_to"}


def resolve_field_owner_node_id(
    field: dict[str, Any],
    *,
    branch_node: dict[str, Any] | None = None,
    persona_node: dict[str, Any] | None = None,
    declaration_node: dict[str, Any] | None = None,
) -> str:
    """Resolve owner_node_id for a field declaration based on its scope.

    ``scope: "declaration"`` binds the fact to the node that declares it. This
    lets Persona, Campaign and branch nodes publish different commercial notes
    without teaching the runtime any field or customer names. ``persona`` and
    ``branch`` remain supported for existing publications.

    This generalizes the pattern already applied to Aurora's product branches
    (publish_aurora_graph.py:90) to all branch types and all personas,
    with explicit declarative control via the `scope` field.

    Args:
        field: Field declaration dict, optionally containing `scope` and `owner_node_id`
        branch_node: The branch anchor node dict (required if scope != "persona")
        persona_node: The persona node dict (required if scope == "persona")

    Returns:
        Resolved owner_node_id (string)
    """
    # Explicit owner_node_id always wins
    if field.get("owner_node_id"):
        return str(field["owner_node_id"])

    # Scope-driven resolution
    scope = field.get("scope", "branch")
    if scope == "declaration" and declaration_node:
        return str(declaration_node.get("id"))
    if scope == "persona" and persona_node:
        return str(persona_node.get("id"))
    if scope == "branch" and branch_node:
        return str(branch_node.get("id"))

    # Fallback: branch if available, else persona
    if branch_node:
        return str(branch_node.get("id"))
    if persona_node:
        return str(persona_node.get("id"))

    # Ultimate fallback: empty string (compiler will flag as error)
    return ""


def _fold(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _checksum(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _active_primary_edges(graph: GraphJson) -> list[Any]:
    """Return the real hierarchy edges for both Graph JSON generations."""
    result = [
        edge for edge in graph.edges
        if edge.lifecycle.status == "active"
        and edge.primary_tree is True
        and (
            graph.schema_version == "2.0"
            or (edge.relation_type or edge.relation) == "contains"
        )
    ]
    if result:
        return result

    # Compatibility for old fixtures whose parent_id predates explicit edges.
    by_pair = {(edge.source, edge.target): edge for edge in graph.edges}
    synthetic: list[Any] = []
    for node in graph.nodes:
        if not node.parent_id:
            continue
        edge = by_pair.get((node.parent_id, node.id))
        if edge is not None:
            synthetic.append(edge)
    return synthetic


def hierarchy(graph: GraphJson) -> tuple[dict[str, tuple[str, str]], dict[str, list[str]]]:
    parents: dict[str, tuple[str, str]] = {}
    children: dict[str, list[str]] = {}
    for edge in _active_primary_edges(graph):
        parents[edge.target] = (edge.source, edge.id)
        children.setdefault(edge.source, []).append(edge.target)
    by_id = {node.id: node for node in graph.nodes}
    for node in graph.nodes:
        if node.id in parents or not node.parent_id or node.parent_id not in by_id:
            continue
        parents[node.id] = (node.parent_id, f"parent:{node.parent_id}:{node.id}")
        children.setdefault(node.parent_id, []).append(node.id)
    return parents, children


def coordinate_for_node(
    graph: GraphJson,
    node_id: str,
    *,
    graph_version: int | None = None,
) -> dict[str, Any]:
    """Derive a stable node coordinate from actual hierarchy edges."""
    by_id = {node.id: node for node in graph.nodes}
    parents, _ = hierarchy(graph)
    if node_id not in by_id:
        raise LookupError(f"graph node not found: {node_id}")
    node_ids = [node_id]
    edge_ids: list[str] = []
    current = node_id
    seen = {current}
    while current in parents:
        parent_id, edge_id = parents[current]
        if parent_id in seen:
            raise ValueError(f"graph hierarchy cycle at {parent_id}")
        seen.add(parent_id)
        node_ids.insert(0, parent_id)
        edge_ids.insert(0, edge_id)
        current = parent_id
    branch_anchor = next(
        (
            candidate_id for candidate_id in reversed(node_ids)
            if by_id[candidate_id].node_type in BRANCH_TYPES
        ),
        None,
    )
    version = graph_version or graph.graph_version
    path_payload = {
        "graph_id": graph.graph_id,
        "graph_version": version,
        "path_node_ids": node_ids,
        "path_edge_ids": edge_ids,
    }
    return {
        "source_node_id": node_id,
        "branch_anchor_node_id": branch_anchor,
        "path_node_ids": node_ids,
        "path_edge_ids": edge_ids,
        "graph_version": version,
        "path_checksum": _checksum(path_payload),
    }


def coordinates(
    graph: GraphJson, *, graph_version: int | None = None
) -> dict[str, dict[str, Any]]:
    return {
        node.id: coordinate_for_node(graph, node.id, graph_version=graph_version)
        for node in graph.nodes
        if node.node_class == "knowledge"
    }


def resolve_branch_anchor(
    graph: GraphJson,
    message: str,
    *,
    previous_anchor_node_id: str | None = None,
) -> dict[str, Any]:
    """Select a branch only on explicit intent; short answers keep the pin."""
    by_id = {node.id: node for node in graph.nodes}
    previous = by_id.get(str(previous_anchor_node_id or ""))
    if previous is not None and previous.node_type not in BRANCH_TYPES:
        previous = None
    query = _fold(message)
    query_tokens = set(query.split())
    ranked: list[tuple[float, str]] = []
    for node in graph.nodes:
        if node.node_type not in BRANCH_TYPES or node.lifecycle.status not in APPROVED:
            continue
        aliases = [
            node.slug,
            node.title or node.label or "",
            *((node.spec or {}).get("aliases") or []),
            *((node.data or {}).get("aliases") or []),
        ]
        score = 0.0
        for raw_alias in aliases:
            alias = _fold(raw_alias)
            if not alias:
                continue
            alias_tokens = set(alias.split())
            if query == alias:
                score = max(score, 1.0)
            elif len(alias) >= 4 and re.search(rf"\b{re.escape(alias)}\b", query):
                score = max(score, 0.98)
            elif alias_tokens:
                overlap = len(query_tokens & alias_tokens) / len(alias_tokens)
                distinctive = any(len(token) >= 4 for token in query_tokens & alias_tokens)
                if overlap >= 0.75 and distinctive:
                    score = max(score, 0.75 + 0.2 * overlap)
        if score:
            ranked.append((score, node.id))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected_id = ranked[0][1] if ranked else (previous.id if previous else None)
    confidence = ranked[0][0] if ranked else (1.0 if previous else 0.0)
    changed = bool(ranked and previous and selected_id != previous.id)
    return {
        "branch_anchor_node_id": selected_id,
        "confidence": round(confidence, 4),
        "explicit_intent": bool(ranked),
        "branch_changed": changed,
    }


def _descendants(children: dict[str, list[str]], node_id: str) -> set[str]:
    result: set[str] = set()
    queue = deque([node_id])
    while queue:
        current = queue.popleft()
        if current in result:
            continue
        result.add(current)
        queue.extend(children.get(current, []))
    return result


def branch_closure(graph: GraphJson, branch_anchor_node_id: str | None) -> set[str]:
    """Return ancestors, branch descendants and graph-applicable policy nodes."""
    by_id = {node.id: node for node in graph.nodes}
    published = {
        node.id for node in graph.nodes
        if node.node_class == "knowledge" and node.lifecycle.status in APPROVED
    }
    if not branch_anchor_node_id or branch_anchor_node_id not in by_id:
        return published
    parents, children = hierarchy(graph)
    result = _descendants(children, branch_anchor_node_id)
    current = branch_anchor_node_id
    while current in parents:
        current = parents[current][0]
        result.add(current)
    result.update(
        node.id for node in graph.nodes
        if node.node_type in PERMANENT_TYPES and node.id in published
    )

    # Include executable questions referenced by any reachable field contract.
    contract = compile_branch_contract(graph, branch_anchor_node_id)
    result.update(
        str(field.get("question_node_id"))
        for field in contract.get("fields") or []
        if field.get("question_node_id")
    )
    result.update(contract.get("handoff_rule_node_ids") or [])

    # One-hop semantic rules/FAQs may be outside the primary tree branch.
    frontier = set(result)
    for edge in graph.edges:
        relation = edge.relation_type or edge.relation or ""
        if edge.lifecycle.status != "active" or relation not in QUESTION_RELATIONS:
            continue
        if edge.source in frontier and edge.target in published:
            candidate = by_id[edge.target]
            if candidate.node_type in {"faq", "rule", "copy", "tone"}:
                result.add(candidate.id)
        if edge.target in frontier and edge.source in published:
            candidate = by_id[edge.source]
            if candidate.node_type in {"faq", "rule", "copy", "tone"}:
                result.add(candidate.id)
    return result & published


def _qualification_question_nodes(graph: GraphJson) -> dict[tuple[str, str], Node]:
    result: dict[tuple[str, str], Node] = {}
    for node in graph.nodes:
        if node.node_type != "faq":
            continue
        data = node.data or {}
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        role = metadata.get("role") or data.get("role")
        field_key = str(metadata.get("field_key") or data.get("field_key") or "").strip()
        owner = str(metadata.get("owner_node_id") or data.get("owner_node_id") or "").strip()
        if role == "qualification_question" and field_key:
            result[(owner, field_key)] = node
            result.setdefault(("", field_key), node)
    return result


def _field_specs(
    value: Any, owner_node_id: str, *, carry_over: bool = False,
) -> dict[str, dict[str, Any]]:
    """Compila specs de field preservando o que o grafo declarou.

    `carry_over` diz se o field atravessa o fim de um pedido. E o default por
    origem, nunca uma lista de nomes: identidade do cliente atravessa, dado do
    pedido nao. O grafo pode declarar `carry_over` no proprio field e vencer o
    default -- `spec = dict(item)` preserva a chave.
    """
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(value, list):
        return result
    for item in value:
        if isinstance(item, str):
            key = item.strip()
            spec: dict[str, Any] = {"key": key, "required": True}
        elif isinstance(item, dict):
            key = str(item.get("key") or "").strip()
            spec = dict(item)
        else:
            continue
        if not key:
            continue
        spec["key"] = key
        spec.setdefault("required", True)
        spec.setdefault("accepts_unknown", False)
        spec.setdefault("owner_node_id", owner_node_id)
        spec["carry_over"] = bool(spec.get("carry_over", carry_over))
        result[key] = spec
    return result


def compile_branch_contract(
    graph: GraphJson, branch_anchor_node_id: str | None
) -> dict[str, Any]:
    """Compile effective fields and questions without knowing field names."""
    by_id = {node.id: node for node in graph.nodes}
    persona = next((node for node in graph.nodes if node.node_type == "persona"), None)
    persona_data = (persona.data or {}) if persona else {}
    policy = persona_data.get("appointment_policy")
    policy = policy if isinstance(policy, dict) else {}
    question_nodes = _qualification_question_nodes(graph)
    question_ids = policy.get("field_question_node_ids")
    question_ids = question_ids if isinstance(question_ids, dict) else {}
    question_texts = policy.get("field_questions")
    question_texts = question_texts if isinstance(question_texts, dict) else {}

    # Preserve the graph-authored qualification order explicitly. Relying on
    # dict insertion order lets qualification field declarations accidentally
    # outrank appointment_policy.required_fields.
    declared_order: list[str] = []
    for value in policy.get("required_fields") or []:
        key = str(value).strip()
        if key and key not in declared_order:
            declared_order.append(key)

    fields: dict[str, dict[str, Any]] = {}
    persona_owner = persona.id if persona else ""
    persona_qualification = persona_data.get("qualification")
    persona_qualification = persona_qualification if isinstance(persona_qualification, dict) else {}
    # Qualificacao da persona e o que o cliente e, nao o que este pedido e:
    # nome e afins seguem valendo no pedido seguinte.
    fields.update(_field_specs(
        persona_qualification.get("fields"), persona_owner, carry_over=True,
    ))
    for key in policy.get("required_fields") or []:
        normalized = str(key).strip()
        if normalized:
            fields.setdefault(normalized, {
                "key": normalized,
                "required": True,
                "accepts_unknown": False,
                "owner_node_id": persona_owner,
                # Data e janela pertencem ao pedido: carregar "12/09" para o
                # pedido seguinte seria errado.
                "carry_over": False,
            })

    branch = by_id.get(str(branch_anchor_node_id or ""))
    if branch is not None:
        data = branch.data or {}
        qualification = data.get("qualification")
        qualification = qualification if isinstance(qualification, dict) else {}
        fields.update(_field_specs(qualification.get("fields"), branch.id))
        booking = data.get("booking")
        booking = booking if isinstance(booking, dict) else {}
        for key in booking.get("required_fields") or []:
            normalized = str(key).strip()
            if normalized:
                if normalized not in declared_order:
                    declared_order.append(normalized)
                fields.setdefault(normalized, {
                    "key": normalized,
                    "required": True,
                    "accepts_unknown": False,
                    "owner_node_id": branch.id,
                    "carry_over": False,
                })

        conditional = policy.get("conditional_fields")
        conditional = conditional if isinstance(conditional, dict) else {}
        for key, branches in conditional.items():
            if branch.slug not in (branches or []):
                fields.pop(str(key), None)
                continue
            if str(key) not in declared_order:
                declared_order.append(str(key))
            fields.setdefault(str(key), {
                "key": str(key),
                "required": True,
                "accepts_unknown": False,
                "owner_node_id": branch.id,
                "carry_over": False,
            })

    ordered: list[dict[str, Any]] = []
    ordered_keys = [key for key in declared_order if key in fields]
    ordered_keys.extend(key for key in fields if key not in ordered_keys)
    for key in ordered_keys:
        spec = fields[key]
        owner = str(spec.get("owner_node_id") or persona_owner)
        question_id = str(spec.get("question_node_id") or question_ids.get(key) or "")
        question_node = by_id.get(question_id) if question_id else None
        if question_node is None:
            question_node = question_nodes.get((owner, key)) or question_nodes.get(("", key))
            question_id = question_node.id if question_node else ""
        question = ""
        if question_node is not None:
            node_data = question_node.data or {}
            content = node_data.get("content") if isinstance(node_data.get("content"), dict) else {}
            question = str(
                content.get("question") or node_data.get("question")
                or question_node.title or question_node.label or ""
            ).strip()
        if not question:
            question = str(question_texts.get(key) or "").strip()
        ordered.append({
            **spec,
            "key": key,
            "owner_node_id": owner,
            "question_node_id": question_id or None,
            "question": question,
            "depends_on": [str(item) for item in spec.get("depends_on") or [] if item],
        })

    required = [field["key"] for field in ordered if field.get("required", True)]
    branch_coordinate = (
        coordinate_for_node(graph, branch.id) if branch is not None else None
    )
    handoff_rule_ids = [
        node.id for node in graph.nodes
        if node.node_type == "rule"
        and bool((node.data or {}).get("handoff") or (node.data or {}).get("confirmation_required"))
    ]
    texts = policy.get("texts") if isinstance(policy.get("texts"), dict) else {}
    confirmation_text = str(
        policy.get("confirmation_text")
        or texts.get("completion")
        or texts.get("complemento_confirmacao")
        or ""
    ).strip()
    return {
        "branch_anchor_node_id": branch.id if branch else None,
        "branch_slug": branch.slug if branch else None,
        "branch_path_checksum": (branch_coordinate or {}).get("path_checksum"),
        "required_fields": required,
        "fields": ordered,
        "strict_order": bool(policy.get("strict_order", True)),
        "confirmation_required": bool(
            policy.get("confirmation_required")
            or ((branch.data or {}).get("booking") or {}).get("confirmation_required")
            if branch is not None else policy.get("confirmation_required")
        ),
        "handoff_rule_node_ids": handoff_rule_ids,
        "confirmation_text": confirmation_text or None,
    }


def ledger_from_state(state: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    facts = deepcopy(state.get("facts") or {})
    request = state.get("appointment_request")
    request = request if isinstance(request, dict) else {}
    fields = {field["key"]: field for field in contract.get("fields") or []}
    for key, value in request.items():
        if key not in fields or value in (None, "") or key in facts:
            continue
        facts[key] = {
            "status": "known",
            "value": value,
            "source_message_id": None,
            "owner_node_id": fields[key].get("owner_node_id"),
        }
    return {
        "active_branch_node_id": state.get("active_branch_node_id") or contract.get("branch_anchor_node_id"),
        "active_path_checksum": state.get("active_path_checksum") or contract.get("branch_path_checksum"),
        "facts": facts,
        "asked_question_node_ids": list(state.get("asked_question_node_ids") or []),
    }


def field_is_resolved(fact: dict[str, Any] | None, field: dict[str, Any]) -> bool:
    if not isinstance(fact, dict):
        return False
    status = str(fact.get("status") or "")
    if status == "known":
        return fact.get("value") not in (None, "")
    return status == "unknown" and bool(field.get("accepts_unknown"))


def missing_fields(ledger: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    facts = ledger.get("facts") or {}
    return [
        field["key"] for field in contract.get("fields") or []
        if field.get("required", True) and not field_is_resolved(facts.get(field["key"]), field)
    ]


def apply_extracted_facts(
    ledger: dict[str, Any],
    facts: Iterable[dict[str, Any]],
    contract: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    result = deepcopy(ledger)
    result.setdefault("facts", {})
    declarations = {field["key"]: field for field in contract.get("fields") or []}
    errors: list[str] = []
    for raw in facts:
        key = str(raw.get("field_key") or "").strip()
        declaration = declarations.get(key)
        if declaration is None:
            errors.append(f"undeclared_field:{key or '<empty>'}")
            continue
        owner = str(raw.get("owner_node_id") or declaration.get("owner_node_id") or "")
        if owner != str(declaration.get("owner_node_id") or ""):
            errors.append(f"field_owner_mismatch:{key}")
            continue
        status = str(raw.get("status") or "known")
        if status not in FACT_STATUSES:
            errors.append(f"invalid_fact_status:{key}:{status}")
            continue
        value = raw.get("value")
        if status == "known" and value in (None, ""):
            errors.append(f"known_fact_without_value:{key}")
            continue
        if status != "known":
            value = None
        value_type = str(declaration.get("value_type") or "")
        if status == "known" and value_type == "year" and not re.fullmatch(r"\d{4}", str(value).strip()):
            status = "invalid"
            value = None
        result["facts"][key] = {
            "status": status,
            "value": value,
            "source_message_id": raw.get("source_message_id"),
            "owner_node_id": owner,
        }
    return result, errors


# --------------------------------------------------------------------------
# Price disclosure guard (shared with services.conversation_runtime)
#
# A persona whose published policy says
# ``appointment_policy.price_disclosure == "human_only"`` must never let the
# agent state money for a service â€” not a value, not a range, not an
# approximation, not an installment.  The regex and the carve-out live here,
# next to the rest of the graph-contract proof checks, so the runtime guard
# and check_proposal() enforce exactly the same rule from one definition.
# --------------------------------------------------------------------------

# Any number immediately followed by one of these reads as a duration, a
# count or a percentage, never as money ("cerca de 2 horas", "48 horas",
# "10%", "4x sem juros").
_NOT_MONEY_UNIT = (
    r"(?!\s*(?:%|x\b|h\b|hs\b|hora|horas|min\b|minuto|minutos|dia|dias|"
    r"semana|semanas|m[eÃª]s|meses|ano|anos|km|kg|vezes|parcelas))"
)
# ``(?![\d.,])`` stops the number from being truncated so the unit lookahead
# always inspects what follows the *whole* figure ("30 minutos", not "0 â€¦").
_AMOUNT = r"\d[\d.,]*(?![\d.,])" + _NOT_MONEY_UNIT
_PRICE_LEAD_IN = (
    r"cust\w*|sai\s+por|fic(?:a|am|ou)\s+(?:em|por)?|"
    r"valor(?:es)?\s*(?:de|Ã©|e|em|:)?|pre[Ã§c]o\w*\s*(?:de|Ã©|e|:)?|"
    r"or[Ã§c]amento\w*\s*(?:de|Ã©|e|:)?|investimento\s*(?:de|Ã©|e|:)?|"
    r"a\s+partir\s+d[eoa]s?|em\s+torno\s+de|por\s+volta\s+de|cerca\s+de|"
    r"aproximadamente|faixa\s+de|entre|por\s+apenas|apenas"
)
_MONETARY_FIGURE = re.compile(
    r"r\$\s*\d"
    rf"|{_AMOUNT}\s*(?:reais|real\b|conto?s?\b|pila\b)"
    rf"|(?:{_PRICE_LEAD_IN})\s*(?:r\$\s*)?{_AMOUNT}",
    re.IGNORECASE,
)


def persona_appointment_policy(graph: GraphJson) -> dict[str, Any]:
    """Published ``appointment_policy`` of this graph's persona node."""
    persona = next((node for node in graph.nodes if node.node_type == "persona"), None)
    data = (persona.data or {}) if persona is not None else {}
    policy = data.get("appointment_policy")
    return policy if isinstance(policy, dict) else {}


def price_disclosure_is_human_only(policy: dict[str, Any] | None) -> bool:
    return (
        str((policy or {}).get("price_disclosure") or "").strip().lower()
        == "human_only"
    )


def _node_data(node: Any) -> dict[str, Any]:
    """Read ``data`` from a Node model or from a flattened rag_node dict."""
    data = node.get("data") if isinstance(node, dict) else getattr(node, "data", None)
    return data if isinstance(data, dict) else {}


def carries_payment_policy_claim(nodes: Iterable[Any] | None) -> bool:
    """True when any given node publishes a ``payment_policy`` claim.

    Payment terms (deposit threshold, installment plans) are money figures the
    graph explicitly authorizes the agent to state, so they must survive the
    price guard.  The authorization is read from the node's own published
    claims â€” never from the wording of the reply.
    """
    for node in nodes or ():
        for claim in _node_data(node).get("claims") or []:
            if isinstance(claim, dict) and str(
                claim.get("claim_type") or ""
            ) == "payment_policy":
                return True
    return False


def states_monetary_figure(text: str | None) -> bool:
    """True when the text states a value, range or approximation in money."""
    return bool(text) and bool(_MONETARY_FIGURE.search(text))


def reply_discloses_blocked_price(
    text: str | None,
    policy: dict[str, Any] | None,
    cited_nodes: Iterable[Any] | None = (),
) -> bool:
    """The reply states money the persona's policy reserves for a human."""
    if not price_disclosure_is_human_only(policy):
        return False
    if not states_monetary_figure(text):
        return False
    return not carries_payment_policy_claim(cited_nodes)


def _question_field_map(contract: dict[str, Any]) -> dict[str, str]:
    return {
        str(field.get("question_node_id")): field["key"]
        for field in contract.get("fields") or []
        if field.get("question_node_id")
    }


def check_proposal(
    *,
    graph: GraphJson,
    contract: dict[str, Any],
    ledger: dict[str, Any],
    proposal: dict[str, Any],
    package_node_ids: set[str],
    repair_context_node_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Proof-check a model proposal and return a commit-ready factual ledger."""
    errors: list[str] = []
    repair_ids: set[str] = set()
    by_id = {node.id: node for node in graph.nodes}
    branch_id = contract.get("branch_anchor_node_id")
    closure = branch_closure(graph, branch_id)
    available = set(package_node_ids) | set(repair_context_node_ids or set())
    if proposal.get("branch_anchor_node_id") != branch_id:
        errors.append("branch_anchor_mismatch")
    if proposal.get("branch_path_checksum") != contract.get("branch_path_checksum"):
        errors.append("branch_path_checksum_mismatch")

    cited = [str(item) for item in proposal.get("cited_node_ids") or [] if item]
    for node_id in cited:
        if node_id not in by_id or node_id not in closure:
            errors.append(f"cited_node_outside_branch:{node_id}")
        elif node_id not in available:
            errors.append(f"cited_node_outside_package:{node_id}")
            repair_ids.add(node_id)

    next_ledger, fact_errors = apply_extracted_facts(
        ledger, proposal.get("extracted_facts") or [], contract
    )
    errors.extend(fact_errors)
    missing = missing_fields(next_ledger, contract)
    resolved = set(contract.get("required_fields") or []) - set(missing)
    question_map = _question_field_map(contract)
    next_question = str(proposal.get("next_question_node_id") or "")
    expected_field = missing[0] if missing else None
    selected_field = question_map.get(next_question)
    if missing:
        if not next_question:
            errors.append("missing_next_question")
        elif selected_field not in missing:
            errors.append(f"question_not_for_missing_field:{next_question}")
        elif contract.get("strict_order") and selected_field != expected_field:
            errors.append(f"question_out_of_order:{next_question}")
        else:
            declaration = next(
                field for field in contract.get("fields") or []
                if field["key"] == selected_field
            )
            unmet = [item for item in declaration.get("depends_on") or [] if item not in resolved]
            if unmet:
                errors.append(f"question_dependencies_missing:{next_question}")
            if next_question not in closure:
                errors.append(f"question_outside_branch:{next_question}")
            elif next_question not in available:
                errors.append(f"question_outside_package:{next_question}")
                repair_ids.add(next_question)
            question = str(declaration.get("question") or "").strip()
            reply = str(proposal.get("reply") or "")
            if question and question not in reply:
                errors.append(f"reply_missing_published_question:{next_question}")
    elif next_question:
        errors.append("question_after_completion")

    if bool(proposal.get("qualification_complete")) != (not missing):
        errors.append("qualification_completion_mismatch")
    if proposal.get("handoff_requested") and not (
        contract.get("confirmation_required") and not missing
    ):
        errors.append("handoff_not_authorized_by_graph")

    reply = str(proposal.get("reply") or "")
    policy = persona_appointment_policy(graph)
    if price_disclosure_is_human_only(policy):
        # This persona reserves every price statement for a human, so price
        # evidence in the graph does NOT authorize the reply â€” only a cited
        # node publishing a payment_policy claim does (deposit threshold,
        # installment terms), and the check covers ranges and approximations,
        # not just "R$ ...".
        cited_nodes = [by_id[node_id] for node_id in cited if node_id in by_id]
        if reply_discloses_blocked_price(reply, policy, cited_nodes):
            errors.append("price_disclosure_requires_human")
    elif re.search(r"r\$\s*\d", reply, re.IGNORECASE):
        has_price_evidence = any(
            node_id in by_id and bool((by_id[node_id].data or {}).get("price"))
            for node_id in cited
        )
        if not has_price_evidence:
            errors.append("price_without_graph_evidence")

    repairable = bool(repair_ids) and all(
        error.startswith(("cited_node_outside_package", "question_outside_package"))
        for error in errors
    )
    return {
        "valid": not errors,
        "errors": errors,
        "repair_required": repairable,
        "repair_node_ids": sorted(repair_ids),
        "ledger": next_ledger,
        "missing_fields": missing,
        "next_question_node_id": next_question or None,
    }


def fallback_question(contract: dict[str, Any], missing: list[str]) -> tuple[str | None, str | None]:
    if not missing:
        return None, None
    field = next(
        (item for item in contract.get("fields") or [] if item["key"] == missing[0]),
        None,
    )
    if not field:
        return None, None
    question = str(field.get("question") or "").strip() or None
    question_id = str(field.get("question_node_id") or "").strip() or None
    return question_id, question


def materialize_qualification_questions(graph: GraphJson) -> GraphJson:
    """Compile operator-authored question strings into executable FAQ nodes.

    This is a schema adapter, not copy generation: it only materializes exact
    questions already present in the persona's published policy.
    """
    result = GraphJson.model_validate(graph.model_dump(mode="json"))
    if result.schema_version != "2.1":
        return result
    persona = next((node for node in result.nodes if node.node_type == "persona"), None)
    if persona is None:
        return result
    data = dict(persona.data or {})
    policy = data.get("appointment_policy")
    if not isinstance(policy, dict):
        return result
    questions = policy.get("field_questions")
    required = policy.get("required_fields")
    if not isinstance(questions, dict) or not isinstance(required, list):
        return result
    by_id = {node.id: node for node in result.nodes}
    edge_ids = {edge.id for edge in result.edges}
    mapped = dict(policy.get("field_question_node_ids") or {})
    embedded_actions = [
        node for node in result.nodes
        if node.node_class == "action" and node.node_type == "embedded" and node.action
    ]
    executable_fields = [str(item).strip() for item in required if str(item).strip()]
    conditional = policy.get("conditional_fields")
    if isinstance(conditional, dict):
        executable_fields.extend(str(key).strip() for key in conditional if str(key).strip())
    for candidate in result.nodes:
        if candidate.node_type not in BRANCH_TYPES:
            continue
        candidate_data = candidate.data or {}
        booking = candidate_data.get("booking")
        if isinstance(booking, dict):
            executable_fields.extend(
                str(item).strip() for item in booking.get("required_fields") or []
                if str(item).strip()
            )
        qualification = candidate_data.get("qualification")
        if isinstance(qualification, dict):
            executable_fields.extend(
                str(item.get("key") if isinstance(item, dict) else item).strip()
                for item in qualification.get("fields") or []
                if str(item.get("key") if isinstance(item, dict) else item).strip()
            )
    # Alternative wordings for the same question, authored beside the question
    # itself. The runtime needs them to re-ask a pending field without
    # repeating one sentence verbatim; copy stays operator-owned either way.
    paraphrase_policy = policy.get("field_question_paraphrases")
    if not isinstance(paraphrase_policy, dict):
        paraphrase_policy = {}
    for raw_key in dict.fromkeys(executable_fields):
        question = str(questions.get(raw_key) or "").strip()
        if not question:
            continue
        paraphrases = [
            text for value in (paraphrase_policy.get(raw_key) or [])
            if isinstance(value, str) and (text := value.strip()) and text != question
        ]
        node_id = str(mapped.get(raw_key) or f"faq:qualification:{persona.slug}:{raw_key}")
        mapped[raw_key] = node_id
        if node_id in by_id:
            # Re-materializing must keep an already published node in sync,
            # otherwise newly authored paraphrases would never reach the graph.
            existing = by_id[node_id]
            existing_data = dict(existing.data or {})
            existing_content = dict(existing_data.get("content") or {})
            existing_data["paraphrases"] = paraphrases
            existing_content["paraphrases"] = paraphrases
            existing_data["content"] = existing_content
            existing.data = existing_data
        else:
            node = Node(
                id=node_id,
                node_type="faq",
                slug=f"qualification-{raw_key.replace('_', '-')}",
                title=question,
                label=question,
                parent_id=persona.id,
                data={
                    "question": question,
                    "paraphrases": paraphrases,
                    "content": {"question": question, "paraphrases": paraphrases},
                    "metadata": {
                        "role": "qualification_question",
                        "field_key": raw_key,
                        "owner_node_id": persona.id,
                    },
                    "source": persona.provenance.source or "pending_source",
                    "status": "approved",
                },
                lifecycle={"status": "approved", "revision": 1},
                provenance={
                    "source": persona.provenance.source or "pending_source",
                    "reason": "materialized from appointment_policy.field_questions",
                },
            )
            result.nodes.append(node)
            by_id[node_id] = node
        contains_id = f"edge:qualification:{persona.id}:{node_id}"
        if contains_id not in edge_ids:
            result.edges.append(Edge(
                id=contains_id,
                source=persona.id,
                target=node_id,
                relation_type="contains",
                relation_class="hierarchy",
                primary_tree=True,
                lifecycle=EdgeLifecycle(status="active"),
            ))
            edge_ids.add(contains_id)
        for action in embedded_actions:
            publish_id = f"edge:publish:{node_id}:{action.id}"
            if publish_id in edge_ids:
                continue
            result.edges.append(Edge(
                id=publish_id,
                source=node_id,
                target=action.id,
                relation_type="publishes_to",
                relation_class="publication",
                primary_tree=False,
                lifecycle=EdgeLifecycle(status="active"),
                grant=PublicationGrant(
                    mode="policy",
                    policy_id="appointment_policy.field_questions",
                    reason="Executable graph-owned qualification question",
                ),
            ))
            edge_ids.add(publish_id)
    policy["field_question_node_ids"] = mapped
    data["appointment_policy"] = policy
    persona.data = data
    persona.spec = {**(persona.spec or {}), "appointment_policy": policy}
    return result

