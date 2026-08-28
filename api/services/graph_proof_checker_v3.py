"""Generic proof checker for structured GraphRAG model proposals."""
from __future__ import annotations

import difflib
import re
import unicodedata
from copy import deepcopy
from typing import Any

try:
    from jsonschema import Draft202012Validator
except ModuleNotFoundError:  # local bootstrap before requirements are rebuilt
    Draft202012Validator = None  # type: ignore[assignment]


VALID_STATUSES = {"known", "unknown", "declined", "needs_confirmation", "invalid"}
_MEDIA_PLACEHOLDER = re.compile(
    r"^\[o cliente enviou (?:um|uma) (?:audio|imagem|video|documento)\]$",
    re.IGNORECASE,
)
_CORRECTION_MARKER = re.compile(
    r"\b(corrig|corre[cç][aã]o|na verdade|retific|correction|actually|i meant)\w*\b",
    re.IGNORECASE,
)
_FINAL_CONFIRMATION = re.compile(
    r"\b(confirmad[oa]|reservad[oa]|agendad[oa]|fechad[oa]|booked|confirmed)\b",
    re.IGNORECASE,
)
_NAME_SEPARATORS = {"-", "'", "’"}
# Generic floor/ceiling for a complete human name. The graph overrides them
# per field via `validation.min_tokens` / `validation.max_tokens`.
NAME_MIN_TOKENS = 2
NAME_MAX_TOKENS = 6
_NON_NAME_PHRASES = {
    "oi", "ola", "bom dia", "boa tarde", "boa noite", "tudo bem",
    "e ai", "e ae", "eai", "eae",
}


def _literal_span(message: str, span: Any) -> bool:
    # A quiet burst is persisted as multiple physical messages but dispatched
    # as one canonical text separated by newlines. Whitespace folding preserves
    # literal word order while allowing evidence to span adjacent burst members.
    value = re.sub(r"\s+", " ", str(span or "")).strip()
    canonical = re.sub(r"\s+", " ", str(message or "")).strip()
    return bool(value and value in canonical)


def name_token_bounds(validation: Any) -> tuple[int, int]:
    """Token limits for a name field, published by the graph.

    The graph owns how long a complete name may be in its own market; the
    code only carries a generic default so a publication that says nothing
    keeps working. Invalid or inverted bounds fall back to the default
    instead of rejecting every name -- the publish gate is what refuses a
    malformed declaration, not the runtime.
    """
    published = validation if isinstance(validation, dict) else {}
    try:
        minimum = int(published.get("min_tokens") or NAME_MIN_TOKENS)
        maximum = int(published.get("max_tokens") or NAME_MAX_TOKENS)
    except (TypeError, ValueError):
        return NAME_MIN_TOKENS, NAME_MAX_TOKENS
    if minimum < 1 or maximum < minimum:
        return NAME_MIN_TOKENS, NAME_MAX_TOKENS
    return minimum, maximum


def is_human_full_name(
    value: Any, *, min_tokens: int = NAME_MIN_TOKENS, max_tokens: int = NAME_MAX_TOKENS,
) -> bool:
    """Validate a complete human name without assuming a language or fixture."""
    text = unicodedata.normalize("NFC", str(value or "")).strip()
    if not text or re.search(r"(?:https?://|www\.|@)", text, re.IGNORECASE):
        return False
    if _fold(text) in _NON_NAME_PHRASES:
        return False
    tokens = text.split()
    if not min_tokens <= len(tokens) <= max_tokens:
        return False
    for token in tokens:
        if not token or token[0] in _NAME_SEPARATORS or token[-1] in _NAME_SEPARATORS:
            return False
        previous_separator = False
        for char in token:
            if char in _NAME_SEPARATORS:
                if previous_separator:
                    return False
                previous_separator = True
                continue
            if not unicodedata.category(char).startswith("L"):
                return False
            previous_separator = False
    return True


def _condition_matches(condition: Any, facts: dict[str, Any]) -> bool:
    if condition in (None, {}, []):
        return True
    if "all" in condition:
        return all(_condition_matches(item, facts) for item in condition["all"])
    if "any" in condition:
        return any(_condition_matches(item, facts) for item in condition["any"])
    fact = facts.get(str(condition.get("field") or "")) or {}
    exists = fact.get("status") in {"known", "unknown", "declined"}
    actual = fact.get("value")
    expected = condition.get("value")
    operator = condition.get("operator") or "equals"
    if operator == "exists":
        return exists
    if operator == "not_exists":
        return not exists
    if operator == "equals":
        return actual == expected
    if operator == "not_equals":
        return actual != expected
    if operator == "in":
        return actual in (expected or [])
    if operator == "not_in":
        return actual not in (expected or [])
    try:
        return {
            "greater_than": actual > expected,
            "greater_than_or_equal": actual >= expected,
            "less_than": actual < expected,
            "less_than_or_equal": actual <= expected,
        }[operator]
    except (KeyError, TypeError):
        return False


def field_resolved(field: dict[str, Any], fact: dict[str, Any] | None) -> bool:
    if not fact:
        return False
    status = str(fact.get("status") or "")
    if status in {"needs_confirmation", "invalid"}:
        return False
    if status not in set(field.get("accepted_statuses") or ["known"]):
        return False
    return status != "known" or fact.get("value") not in (None, "")


def fact_compatible(field: dict[str, Any], fact: dict[str, Any] | None) -> bool:
    """Check whether a stored fact remains valid under a new publication."""
    if not fact or fact.get("owner_node_id") != field.get("owner_node_id"):
        return False
    if not field_resolved(field, fact):
        return False
    if fact.get("status") == "known":
        return _schema_error(field.get("value_schema") or {}, fact.get("value")) is None
    return fact.get("value") is None


def _resolved_for_field_owner(field: dict[str, Any], fact: dict[str, Any] | None) -> bool:
    """Qualification requires a known value from the declared owner.

    Confirmed live 2026-08-06: field keys are shared across every product's
    field declarations (nome_cliente, modelo_veiculo, servico, ...), each
    with its own owner_node_id. field_resolved() alone only checks status
    and value, never owner_node_id — so a fact accepted while a different
    branch was active (same key, different owner_node_id) silently counted
    as resolved for the new branch too after a switch. fact_compatible()
    already carries this owner check, but until now it only ran when the
    graph publication changed, never on an ordinary in-conversation branch
    switch within the same publication. Requiring the same match here
    closes that gap for every branch switch, generically, without pulling
    in fact_compatible()'s schema re-validation (meant for publication
    changes, not plain branch switches).
    """
    return bool(
        fact
        and fact.get("owner_node_id") == field.get("owner_node_id")
        and fact.get("status") == "known"
        and fact.get("value") not in (None, "")
    )


def _applicable_required_fields(contract: dict[str, Any], facts: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        field for field in contract.get("fields") or []
        if field.get("required", True)
        and _condition_matches(field.get("condition"), facts)
    ]


def pending_fields(contract: dict[str, Any], facts: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        field for field in _applicable_required_fields(contract, facts)
        if not _resolved_for_field_owner(field, facts.get(field["key"]))
    ]


def askable_pending_fields(
    contract: dict[str, Any], facts: dict[str, Any],
) -> list[dict[str, Any]]:
    """Pending fields that have not exhausted their published-question limit."""
    return [
        field for field in pending_fields(contract, facts)
        if not (
            (fact := facts.get(field["key"]))
            and fact.get("owner_node_id") == field.get("owner_node_id")
            and fact.get("status") in {"unknown", "declined"}
        )
    ]


def required_field_count(contract: dict[str, Any], facts: dict[str, Any]) -> int:
    """Count of a branch's currently-applicable required fields (resolved or
    not) -- the denominator for a 0-50% qualification progress score."""
    return len(_applicable_required_fields(contract, facts))


def aggregate_missing_fields(
    branch_contracts: dict[str, dict[str, Any]],
    active_branch_anchors: list[str],
    facts_by_key: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Union of pending fields across every simultaneously-active branch.

    Supports N services per appointment (branch_action "add"): each active
    branch keeps its OWN required fields, and a field owned by a specific
    branch anchor (e.g. "servico") must be resolved separately per branch,
    while a field owned by a shared node (e.g. "nome_cliente", owned by the
    persona) naturally de-duplicates because every branch's contract
    resolves it against the *same* owner's fact. `facts_by_key` holds every
    *current* fact for the ledger grouped by field_key -- unlike the single-
    branch flat `facts` dict (one fact per field_key, the shape every other
    function here still uses and that this function does not change), more
    than one branch being active can mean more than one current fact shares
    a field_key (same key, different owner_node_id), which is exactly why
    this needs its own facts shape rather than reusing the flat one.
    """
    def _fact_for_owner(key: str, owner_node_id: Any) -> dict[str, Any] | None:
        return next(
            (fact for fact in facts_by_key.get(key, []) if fact.get("owner_node_id") == owner_node_id),
            None,
        )

    seen: set[tuple[str, str]] = set()
    aggregated: list[dict[str, Any]] = []
    for anchor in active_branch_anchors:
        contract = branch_contracts.get(anchor) or {}
        scoped_facts = {
            field["key"]: fact
            for field in contract.get("fields") or []
            if (fact := _fact_for_owner(field["key"], field.get("owner_node_id"))) is not None
        }
        for field in pending_fields(contract, scoped_facts):
            dedupe_key = (field["key"], str(field.get("owner_node_id") or anchor))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            aggregated.append(field)
    return aggregated


def aggregate_askable_fields(
    branch_contracts: dict[str, dict[str, Any]],
    active_branch_anchors: list[str],
    facts_by_key: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Aggregate pending fields excluding owners explicitly marked unknown."""
    return [
        field
        for field in aggregate_missing_fields(
            branch_contracts, active_branch_anchors, facts_by_key,
        )
        if not any(
            fact.get("owner_node_id") == field.get("owner_node_id")
            and fact.get("status") in {"unknown", "declined"}
            for fact in facts_by_key.get(str(field.get("key") or ""), [])
        )
    ]


def aggregate_required_field_count(
    branch_contracts: dict[str, dict[str, Any]],
    active_branch_anchors: list[str],
    facts_by_key: dict[str, list[dict[str, Any]]],
) -> int:
    """Count applicable requirements once per field owner across services."""
    seen: set[tuple[str, str]] = set()
    for anchor in active_branch_anchors:
        contract = branch_contracts.get(anchor) or {}
        scoped_facts = {
            str(field.get("key") or ""): next(
                (
                    fact for fact in facts_by_key.get(str(field.get("key") or ""), [])
                    if fact.get("owner_node_id") == field.get("owner_node_id")
                ),
                None,
            )
            for field in contract.get("fields") or []
        }
        for field in _applicable_required_fields(contract, scoped_facts):
            seen.add((
                str(field.get("key") or ""),
                str(field.get("owner_node_id") or anchor),
            ))
    return len(seen)


def _claim_policy(contract: dict[str, Any], claim_type: str) -> list[dict[str, Any]]:
    return [
        policy for policy in contract.get("claims") or []
        if str(policy.get("claim_type") or policy.get("type") or "other") == claim_type
    ]


def handoff_rule_matches(
    rule: dict[str, Any], *, facts: dict[str, Any], qualification_complete: bool
) -> bool:
    condition = rule.get("condition")
    if condition in (None, {}, []):
        return True
    if condition == "qualification_complete":
        return qualification_complete
    if condition == "qualification_incomplete":
        return not qualification_complete
    return isinstance(condition, dict) and _condition_matches(condition, facts)


def _schema_error(schema: dict[str, Any], value: Any) -> str | None:
    """Use Draft 2020-12 in runtime; keep local bootstrap import-safe."""
    if Draft202012Validator is not None:
        errors = list(Draft202012Validator(schema).iter_errors(value))
        return errors[0].message if errors else None
    for candidate in schema.get("anyOf") or []:
        if _schema_error(candidate, value) is None:
            return None
    if schema.get("anyOf"):
        return "value does not match anyOf"
    expected = schema.get("type")
    accepted = expected if isinstance(expected, list) else [expected] if expected else []
    type_map = {
        "string": str, "number": (int, float), "integer": int,
        "boolean": bool, "object": dict, "array": list, "null": type(None),
    }
    if accepted and not any(isinstance(value, type_map[kind]) and not (
        kind in {"number", "integer"} and isinstance(value, bool)
    ) for kind in accepted if kind in type_map):
        return f"value is not of type {accepted}"
    if isinstance(value, str):
        if len(value) < int(schema.get("minLength") or 0):
            return "string is too short"
        if schema.get("pattern") and not re.search(str(schema["pattern"]), value):
            return "string does not match pattern"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return "number is below minimum"
        if "maximum" in schema and value > schema["maximum"]:
            return "number is above maximum"
    if "enum" in schema and value not in schema["enum"]:
        return "value is not in enum"
    return None


def _canonical_field_value(
    field: dict[str, Any], value: Any, evidence_span: Any,
) -> tuple[Any, str | None]:
    """Validate and normalize a value using only its compiled declaration."""
    validation = field.get("validation") or {}
    mode = str(validation.get("mode") or "schema")
    if mode == "enum":
        candidates = {_fold(str(value)), _fold(str(evidence_span))}
        for item in validation.get("values") or []:
            if not isinstance(item, dict):
                continue
            canonical = item.get("value")
            published = [canonical, *(item.get("aliases") or [])]
            if any(_fold(str(option)) in candidates for option in published):
                return canonical, None
        return value, "value is not in published enum"
    if mode == "semantic":
        if not isinstance(value, str) or not value.strip():
            return value, "semantic value must be non-empty text"
        semantic_type = str(validation.get("semantic_type") or "")
        if semantic_type == "human_full_name":
            minimum, maximum = name_token_bounds(validation)
            if not is_human_full_name(value, min_tokens=minimum, max_tokens=maximum):
                return value, "value is not a valid human full name"
        elif semantic_type == "human_name":
            folded = _fold(value)
            tokens = folded.split()
            if not (1 <= len(tokens) <= 6) or any(any(char.isdigit() for char in token) for token in tokens):
                return value, "value is not a plausible human name"
        schema_error = _schema_error(field.get("value_schema") or {}, value)
        return value.strip(), schema_error
    return value, _schema_error(field.get("value_schema") or {}, value)


def check_service_operations(
    *,
    document: dict[str, Any],
    message: str,
    operations: list[dict[str, Any]],
    active_branch_node_ids: list[str],
    consumed_service_spans: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prove a first-class service-set transition independently of fields."""
    anchors = set(document.get("branch_anchors") or [])
    after = list(dict.fromkeys(str(value) for value in active_branch_node_ids if value))
    errors: list[str] = []
    applied: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for operation in operations:
        action = str(operation.get("action") or "")
        anchor = str(operation.get("branch_anchor_node_id") or "")
        identity = (action, anchor)
        if identity in seen:
            errors.append(f"duplicate_service_operation:{action}:{anchor}")
            continue
        seen.add(identity)
        if action not in {"add", "keep", "drop"}:
            errors.append(f"invalid_service_operation:{action}")
            continue
        if anchor not in anchors:
            errors.append(f"service_anchor_not_published:{anchor}")
            continue
        expected_checksum = str(
            ((document.get("coordinates") or {}).get(anchor) or {}).get("path_checksum")
            or ""
        )
        if str(operation.get("branch_path_checksum") or "") != expected_checksum:
            errors.append(f"service_path_checksum_mismatch:{anchor}")
        if not _literal_span(message, operation.get("evidence_span")):
            errors.append(f"service_evidence_not_literal:{anchor}")
        evidence_type = str(operation.get("evidence_type") or "")
        if evidence_type not in {
            "exact_catalog", "confirmed_candidate", "explicit_change",
        }:
            errors.append(f"service_evidence_type_not_authorized:{anchor}")
        if evidence_type == "explicit_change" and action != "drop":
            errors.append(f"service_explicit_change_only_authorizes_drop:{anchor}")
        registered = next(
            (
                span for span in consumed_service_spans or []
                if _fold(str(span.get("text") or ""))
                == _fold(str(operation.get("evidence_span") or ""))
                and str(span.get("branch_anchor_node_id") or anchor) == anchor
                and str(span.get("evidence_type") or "exact_catalog") == evidence_type
            ),
            None,
        )
        if registered is None:
            errors.append(f"service_evidence_not_consumed:{anchor}")
        if action == "add":
            if anchor in after:
                errors.append(f"service_add_duplicate:{anchor}")
            else:
                after.append(anchor)
        elif action == "keep":
            if anchor not in after:
                errors.append(f"service_keep_inactive:{anchor}")
        elif anchor not in after:
            errors.append(f"service_drop_inactive:{anchor}")
        else:
            after.remove(anchor)
        applied.append(dict(operation))
    return {
        "valid": not errors,
        "errors": errors,
        "operations": applied,
        "previous_active_branch_node_ids": list(active_branch_node_ids),
        "next_active_branch_node_ids": after,
    }


def check(
    *,
    publication: dict[str, Any],
    contract: dict[str, Any],
    ledger: dict[str, Any],
    proposal: dict[str, Any],
    message: str,
    source_message_id: str,
    package_node_ids: set[str],
    package_chunk_ids: set[str],
    active_branch_node_id: str | None,
    branch_selection_allowed: bool,
    branch_switch_allowed: bool,
    package_chunk_sources: dict[str, str] | None = None,
    active_branch_node_ids: list[str] | None = None,
    additional_fields: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    repair: list[dict[str, Any]] = []
    document = publication.get("document_json") or {}
    branch = str(proposal.get("branch_anchor_node_id") or "")
    action = str(proposal.get("branch_action") or "keep")
    anchors = set(document.get("branch_anchors") or [])
    if publication.get("status") != "active":
        errors.append("publication_not_active")
    if publication.get("checksum") != ledger.get("graph_checksum"):
        errors.append("publication_checksum_mismatch")
    if action != "none":
        if branch not in anchors:
            errors.append("branch_not_published")
        if proposal.get("branch_path_checksum") != contract.get("branch_path_checksum"):
            errors.append("branch_path_checksum_mismatch")
    branch_span = proposal.get("branch_evidence_span")
    active_set = set(active_branch_node_ids or ([active_branch_node_id] if active_branch_node_id else []))
    if active_branch_node_id:
        active_set.add(active_branch_node_id)
    if action == "none":
        if branch or proposal.get("branch_path_checksum"):
            errors.append("none_with_branch_anchor")
    elif action == "keep":
        if not active_set:
            # Confirmed live 2026-08-08: with no active branch yet, "keep"
            # never validated anything (unlike "select"/"switch", which
            # require branch_selection_allowed/branch_switch_allowed and a
            # literal evidence span). A retrieval-only fallback branch
            # picked by graph_agent_runtime_v3._fallback_retrieval_branch()
            # for a message with zero branch signal (e.g. just a name) could
            # get silently promoted to the *real* active branch the moment
            # the model echoed it back as "keep" -- one the customer never
            # actually selected. Two turns later, an explicit switch request
            # to that exact branch was then rejected outright because it was
            # already (wrongly) active. "keep" now requires an active branch
            # to mean anything, same as "select"/"switch" require their own
            # authorization.
            errors.append("keep_without_active_branch")
        elif branch not in active_set:
            errors.append("keep_changed_branch")
    elif action == "select":
        if not branch_selection_allowed or active_branch_node_id:
            errors.append("branch_select_not_authorized")
        if not _literal_span(message, branch_span):
            errors.append("branch_evidence_not_literal")
    elif action == "switch":
        if not branch_switch_allowed or not active_branch_node_id or branch == active_branch_node_id:
            errors.append("branch_switch_not_authorized")
        if not _literal_span(message, branch_span):
            errors.append("branch_evidence_not_literal")
    elif action == "add":
        # A customer asking for an additional service without dropping the
        # one already in progress (e.g. "quero higienização interna E
        # polimento") -- unlike "switch", the previously active branch(es)
        # stay active too. Requires the same authorization/evidence bar as
        # "switch" (it is, after all, still an unsolicited branch-scope
        # change) plus a check "switch" doesn't need: the branch must not
        # already be one of the active ones, since re-adding an active
        # branch is meaningless and would only invite duplicate-fact bugs.
        if not active_set:
            errors.append("add_without_active_branch")
        elif branch in active_set:
            errors.append("add_duplicate_branch")
        elif not branch_switch_allowed:
            errors.append("add_not_authorized")
        if not _literal_span(message, branch_span):
            errors.append("branch_evidence_not_literal")
    else:
        errors.append("invalid_branch_action")

    closure = set(contract.get("closure_node_ids") or [])
    chunk_sources = package_chunk_sources or {}
    for node_id in proposal.get("cited_node_ids") or []:
        if node_id not in closure:
            errors.append(f"cited_node_outside_branch:{node_id}")
        elif node_id not in package_node_ids:
            errors.append(f"cited_node_outside_package:{node_id}")
            repair.append({"kind": "node", "id": node_id})
    for chunk_id in proposal.get("cited_chunk_ids") or []:
        if chunk_id not in package_chunk_ids:
            errors.append(f"cited_chunk_outside_package:{chunk_id}")
            repair.append({"kind": "chunk", "id": chunk_id})
        elif chunk_sources.get(chunk_id) and chunk_sources[chunk_id] not in closure:
            errors.append(f"cited_chunk_outside_branch:{chunk_id}")

    next_ledger = deepcopy(ledger)
    facts = next_ledger.setdefault("facts", {})
    # Keyed by (key, owner_node_id), not just key: the branch-selector field
    # is deliberately declared once per branch, same key ("servico"),
    # different owner -- a plain-key merge would let one branch's field
    # object clobber another's. `additional_fields` (the union of every
    # currently active branch's own fields, not just the turn's focused
    # contract) is what lets a second active branch's own fact validate at
    # all instead of failing as undeclared/owner-mismatched purely because
    # this turn's contract is scoped to whichever branch is focused.
    fields_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    fields_by_key_any_owner: dict[str, dict[str, Any]] = {}
    field_order_keys: list[str] = []
    for declared_field in [*(contract.get("fields") or []), *(additional_fields or [])]:
        field_key = str(declared_field.get("key") or "")
        field_owner = str(declared_field.get("owner_node_id") or "")
        if not field_key:
            continue
        fields_by_identity.setdefault((field_key, field_owner), declared_field)
        fields_by_key_any_owner.setdefault(field_key, declared_field)
        if field_key not in field_order_keys:
            field_order_keys.append(field_key)
    accepted_facts: list[dict[str, Any]] = []
    field_validation: list[dict[str, Any]] = []
    proposal_facts = proposal.get("extracted_facts") or []
    duplicate_keys = {
        str(fact.get("field_key") or "") for fact in proposal_facts
        if sum(
            str(candidate.get("field_key") or "") == str(fact.get("field_key") or "")
            for candidate in proposal_facts
        ) > 1
    }
    errors.extend(f"duplicate_extracted_fact:{key}" for key in sorted(duplicate_keys) if key)
    field_order = {key: index for index, key in enumerate(field_order_keys)}
    for fact in sorted(
        proposal_facts,
        key=lambda value: field_order.get(str(value.get("field_key") or ""), len(field_order)),
    ):
        key = str(fact.get("field_key") or "")
        if key in duplicate_keys:
            continue
        fact_error_count = len(errors)
        owner_for_lookup = str(fact.get("owner_node_id") or "")
        field = fields_by_identity.get((key, owner_for_lookup))
        if not field:
            if key in fields_by_key_any_owner:
                errors.append(f"field_owner_mismatch:{key}")
            else:
                errors.append(f"undeclared_field:{key}")
            continue
        if fact.get("source_message_id") != source_message_id:
            errors.append(f"fact_source_message_mismatch:{key}")
        if not _literal_span(message, fact.get("evidence_span")):
            errors.append(f"fact_evidence_not_literal:{key}")
        status = str(fact.get("status") or "")
        if status not in VALID_STATUSES or status not in set(field.get("accepted_statuses") or ["known"]):
            errors.append(f"fact_status_not_accepted:{key}:{status}")
        value = fact.get("value")
        if status == "known":
            if _MEDIA_PLACEHOLDER.fullmatch(str(value or "").strip()):
                errors.append(f"fact_evidence_placeholder:{key}")
            value, schema_error = _canonical_field_value(
                field, value, fact.get("evidence_span"),
            )
            if schema_error:
                errors.append(f"fact_schema_invalid:{key}:{schema_error}")
        elif value is not None:
            errors.append(f"non_known_fact_has_value:{key}")
        previous = facts.get(key)
        if previous and (previous.get("value"), previous.get("status")) != (value, status):
            policy = field.get("overwrite_policy") or "explicit_correction"
            if policy == "never":
                errors.append(f"fact_overwrite_forbidden:{key}")
            elif (
                policy == "explicit_correction"
                and previous.get("status") != "unknown"
                and not _CORRECTION_MARKER.search(message or "")
            ):
                # An exact graph-authorized branch switch is itself explicit:
                # the customer literally named one unique title/alias in this
                # inbound. Requiring filler words such as "na verdade" made
                # the branch change pass while its corresponding fact failed.
                graph_explicit_switch = (
                    action == "switch"
                    and branch_switch_allowed
                    and fact.get("owner_node_id") == branch
                    and bool(_literal_span(message, fact.get("evidence_span")))
                )
                # Settling a candidate the graph itself left at
                # "needs_confirmation" (e.g. the branch selector field,
                # which graph_compiler_v3._with_confirmable_status always
                # authorizes for that intermediate state) into "known" is
                # confirmation, not correction -- the customer is finally
                # answering something that was never actually settled, not
                # overwriting a fact they previously stated. Requiring
                # "na verdade"/"corrigindo" language here blocked exactly
                # that resolution once the branch focus had moved on to
                # another active service (confirmed live: an "add" that
                # should have closed a pending servico candidate instead
                # failed as an unauthorized overwrite).
                resolving_pending_confirmation = (
                    previous.get("status") == "needs_confirmation" and status == "known"
                )
                if not graph_explicit_switch and not resolving_pending_confirmation:
                    errors.append(f"fact_correction_not_explicit:{key}")
            elif policy == "higher_confidence" and float(fact.get("confidence") or 0) <= float(previous.get("confidence") or 0):
                errors.append(f"fact_overwrite_confidence_too_low:{key}")
        for dependency in field.get("depends_on") or []:
            dependency_field = fields_by_key_any_owner.get(dependency) or {}
            if not _resolved_for_field_owner(dependency_field, facts.get(dependency)):
                errors.append(f"fact_dependency_unsatisfied:{key}:{dependency}")
        if not _condition_matches(field.get("condition"), facts):
            errors.append(f"fact_condition_not_met:{key}")
        if len(errors) != fact_error_count:
            field_validation.append({
                "field_key": key,
                "owner_node_id": fact.get("owner_node_id"),
                "valid": False,
                "errors": errors[fact_error_count:],
            })
            continue
        accepted = {
            **fact, "value": value, "field_key": key,
            "source_message_id": source_message_id,
            "confidence": float(fact.get("confidence") or 0),
        }
        facts[key] = accepted
        accepted_facts.append(accepted)
        field_validation.append({
            "field_key": key,
            "owner_node_id": fact.get("owner_node_id"),
            "valid": True,
            "status": status,
            "canonical_value": value if status == "known" else None,
            "errors": [],
        })

    missing = pending_fields(contract, facts)
    missing_keys = [field["key"] for field in missing]
    question_id = proposal.get("next_question_node_id")
    questions = contract.get("questions") or {}
    askable = askable_pending_fields(contract, facts)
    if askable:
        expected_question_id = askable[0].get("question_node_id")
        question = questions.get(str(question_id or ""))
        if question_id != expected_question_id:
            errors.append("next_question_not_first_missing_field")
        elif not question or question.get("field_key") != askable[0].get("key"):
            errors.append("next_question_not_for_pending_field")
        else:
            if any(dependency in missing_keys for dependency in question.get("depends_on") or []):
                errors.append("next_question_dependencies_unsatisfied")
    elif missing and question_id:
        errors.append("question_after_all_pending_fields_deferred")
    elif question_id:
        errors.append("question_after_completion")
    # qualification_complete is 100% derivable from `missing` -- the same
    # reasoning as re-deriving "servico" from active_branch_node_id
    # server-side (graph_agent_runtime_v3.decide()) rather than trusting a
    # separately model-declared value. Confirmed live 2026-08-08: the model
    # regularly mis-self-reports this (both false positives and false
    # negatives) right at qualification's last field, and the proposal was
    # rejected wholesale over a value the caller can just compute -- costing
    # a natural closing reply for a generic fallback question at exactly the
    # moment it mattered most. The authoritative value is returned below;
    # callers should use that instead of proposal["qualification_complete"].

    for claim in proposal.get("claims") or []:
        claim_type = str(claim.get("claim_type") or "other")
        evidence_nodes = set(claim.get("evidence_node_ids") or [])
        evidence_chunks = set(claim.get("evidence_chunk_ids") or [])
        policies = _claim_policy(contract, claim_type)
        if not policies:
            errors.append(f"claim_not_authorized:{claim_type}")
        if not evidence_nodes and not evidence_chunks:
            errors.append(f"claim_without_evidence:{claim_type}")
        if not evidence_nodes.issubset(package_node_ids & closure):
            errors.append(f"claim_node_evidence_outside_package:{claim_type}")
            repair.extend({"kind": "node", "id": value} for value in evidence_nodes - package_node_ids)
        if not evidence_chunks.issubset(package_chunk_ids):
            errors.append(f"claim_chunk_evidence_outside_package:{claim_type}")
            repair.extend({"kind": "chunk", "id": value} for value in evidence_chunks - package_chunk_ids)
        authorized_nodes = {
            str(value) for policy in policies
            for value in policy.get("evidence_node_ids") or []
        }
        # ``availability`` is also how the proposal schema represents the
        # harmless question "vocês fazem este serviço?". The active,
        # published branch is authoritative evidence that the service is
        # offered, but it must not authorize schedule/slot availability.
        # Expand evidence only for the exact boolean service-existence claim;
        # agenda remains governed by its published operational rule.
        claim_value = claim.get("value")
        # Production publications key branch_contracts by anchor but older
        # documents do not duplicate that key inside the contract value. The
        # proof call already carries authoritative active-branch state, so use
        # the proposal branch only when that state proves it is active.
        branch_anchor = str(contract.get("branch_anchor_node_id") or "")
        if not branch_anchor and branch in active_set:
            branch_anchor = branch
        if (
            claim_type == "availability"
            and isinstance(claim_value, dict)
            and claim_value.get("available") is True
            and set(claim_value).issubset({"available"})
            and branch_anchor
        ):
            authorized_nodes.add(branch_anchor)
        authorized_chunks = {
            str(value) for policy in policies
            for value in policy.get("evidence_chunk_ids") or []
        }
        unsupported_nodes = evidence_nodes - authorized_nodes
        unsupported_chunks = {
            chunk_id for chunk_id in evidence_chunks
            if chunk_id not in authorized_chunks
            and chunk_sources.get(chunk_id) not in authorized_nodes
        }
        if unsupported_nodes or unsupported_chunks:
            errors.append(f"claim_evidence_not_authorized:{claim_type}")

    handoff_rules = contract.get("handoff_rules") or []
    handoff_required = any(
        handoff_rule_matches(
            rule, facts=facts, qualification_complete=not missing
        ) for rule in handoff_rules
    )
    if proposal.get("handoff_requested") is True and not handoff_required:
        errors.append("handoff_not_authorized")
    # Routing is server-owned. A model may request an authorized handoff, but
    # omitting the flag can never keep a terminal qualification on the SDR
    # route; graph_agent_runtime_v3 derives and commits that transition.
    if _FINAL_CONFIRMATION.search(str(proposal.get("reply") or "")) and missing:
        errors.append("premature_final_confirmation")

    repair = list({(item["kind"], item["id"]): item for item in repair if item.get("id")}.values())
    repair_only = bool(errors) and all("outside_package" in error for error in errors)
    return {
        "valid": not errors, "errors": errors,
        "repair_required": repair_only and bool(repair),
        "repair_requirements": repair, "ledger": next_ledger,
        "accepted_facts": accepted_facts, "missing_fields": missing_keys,
        "next_question_node_id": question_id,
        "qualification_complete": not missing,
        "handoff_required": handoff_required,
        "required_field_count": required_field_count(contract, facts),
        "field_validation": field_validation,
    }


def _fold(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _question_already_asked(question: str, text: str) -> bool:
    """True when `text` already asks `question`, even personalized/reworded.

    Confirmed live 2026-08-08: a literal substring match breaks the moment
    the model personalizes the canonical question -- swapping "o carro" for
    the customer's actual model ("o Onix", "o Civic") -- so the same
    question got silently appended a second time in the same message.
    Content-word overlap alone (first attempt at this fix, same day) still
    missed a short question whose *only* real content word is exactly the
    one that gets personalized away (e.g. "Qual é a cor do veículo?" ->
    "...a cor do seu Onix?" -- "veículo" is the one content word, and it's
    gone). Matching contiguous character runs between the two folded
    strings catches that: the shared prefix/suffix around the swapped word
    still accounts for most of the question's length, even when word-level
    overlap alone would not. Checking both signals (word overlap OR
    character-run coverage) covers substitutions in either a single word or
    the sentence's structure, without weakening detection of a genuinely
    different question -- unrelated questions share only a few short/
    common words either way.
    """
    if question.casefold() in text.casefold():
        return True
    q_folded = _fold(question)
    if not q_folded:
        return False
    # Compare each interrogative sentence, never the whole reply. Summing
    # every matching character run across an acknowledgement plus an
    # unrelated question made ordinary Portuguese glue words look like a
    # match (for example, the graph expected the objective while the model
    # repeated "como voce se chama?"). SequenceMatcher.ratio keeps order
    # and penalizes those scattered coincidences, while still recognizing a
    # personalized noun substitution such as "cor do veiculo" ->
    # "cor do seu Onix".
    candidates = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", str(text or ""))
        if "?" in sentence
    ]
    content_tokens = {token for token in q_folded.split() if len(token) >= 4}
    for candidate in candidates:
        candidate_folded = _fold(candidate)
        candidate_tokens = set(candidate_folded.split())
        if (
            len(content_tokens) >= 2
            and len(content_tokens & candidate_tokens) / len(content_tokens) >= 0.7
        ):
            return True
        if difflib.SequenceMatcher(
            None, q_folded, candidate_folded, autojunk=False,
        ).ratio() >= 0.68:
            return True
    return False


def compose_published_question(
    *, reply: str, next_question_node_id: str | None, contract: dict[str, Any]
) -> str:
    """Preserve acknowledgement/answer prose and emit exactly one graph question."""
    text = str(reply or "").strip()
    if not next_question_node_id:
        return text
    question = str(((contract.get("questions") or {}).get(next_question_node_id) or {}).get("text") or "").strip()
    if not question:
        return text
    if _question_already_asked(question, text) and text.count("?") <= 1:
        return text
    # The model may have drafted a natural acknowledgement followed by a
    # question for a different field. Routing/field order belongs to the
    # graph, so retain declarative sentences but discard every interrogative
    # sentence before appending the single authorized question.
    sentences = re.split(r"(?<=[.!?])\s+|[\r\n]+", text)
    declarative = " ".join(
        sentence.strip() for sentence in sentences
        if sentence.strip() and "?" not in sentence
    ).strip()
    return f"{declarative}\n\n{question}".strip()


def validate_natural_summary(reply: str, *, informed_values: list[str]) -> bool:
    """Accept a model-written qualification summary only if it stays grounded.

    Every collected value must appear in the reply (accent/case-insensitive
    substring match -- the model can still phrase the surrounding sentence
    freely), the reply must carry exactly one confirmation question, and it
    may not use vocabulary implying the order is already final (that word
    choice is reserved for the turn after the customer actually confirms).
    """
    text = str(reply or "").strip()
    if not text or text.count("?") != 1:
        return False
    if _FINAL_CONFIRMATION.search(text):
        return False
    folded = _fold(text)
    for value in informed_values:
        candidate = _fold(str(value or ""))
        if candidate and candidate not in folded:
            return False
    return True
