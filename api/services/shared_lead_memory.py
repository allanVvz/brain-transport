"""Agent-neutral projection of existing conversation storage.

This module deliberately owns no persistence.  It projects journeys, ledgers,
facts, proofs, outcomes and messages returned by graph_turn_context_batch_v4.
"""
from __future__ import annotations

from typing import Any

from schemas.conversation import SharedLeadMemory, SharedMemoryFact


POLICY_VERSION = "shared_lead_memory_v1"
MAX_RECENT_MESSAGES = 8
MAX_SEMANTIC_TOKENS = 3000


def _carry_over_keys(document: dict[str, Any]) -> set[str]:
    return {
        str(field.get("key") or "")
        for contract in [
            document.get("common_contract") or {},
            *((document.get("branch_contracts") or {}).values()),
        ]
        for field in contract.get("fields") or []
        if field.get("carry_over") and field.get("key")
    }


def _branch_selection_keys(document: dict[str, Any]) -> set[str]:
    return {
        str(field.get("key") or "")
        for contract in [
            document.get("common_contract") or {},
            *((document.get("branch_contracts") or {}).values()),
        ]
        for field in contract.get("fields") or []
        if field.get("branch_selection_field") and field.get("key")
    }


def _memory_fact(row: dict[str, Any], *, reuse_policy: str) -> SharedMemoryFact:
    metadata = dict(row.get("metadata") or {})
    return SharedMemoryFact(
        key=_text(row.get("field_key"), "unknown"),
        value=row.get("value", row.get("value_json")),
        owner_node_id=_text(row.get("owner_node_id"), "unknown"),
        status=_text(row.get("status"), "invalid"),
        confidence=float(row["confidence"]) if row.get("confidence") is not None else None,
        source=_text(metadata.get("source"), "conversation_fact"),
        journey_id=_text(row.get("journey_id")) or None,
        recorded_at=_text(row.get("updated_at") or row.get("created_at")) or None,
        source_message_id=_text(row.get("source_message_id")) or None,
        reuse_policy=reuse_policy,
        metadata=metadata,
    )


def _text(value: Any, default: str = "") -> str:
    return str(value) if value not in (None, "") else default


def _row_marker(row: dict[str, Any]) -> str:
    return _text(row.get("updated_at") or row.get("created_at") or row.get("revision"))


def _latest_current_facts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current = sorted(
        (row for row in rows if row.get("is_current") is not False), key=_row_marker,
    )
    latest = {
        (_text(row.get("field_key")), _text(row.get("owner_node_id"))): row
        for row in current
    }
    return list(latest.values())


def _partition_facts(
    facts: list[dict[str, Any]], carry_over: set[str], selection_keys: set[str],
) -> tuple[list[SharedMemoryFact], list[SharedMemoryFact]]:
    profile: list[SharedMemoryFact] = []
    historical: list[SharedMemoryFact] = []
    for row in facts:
        key = _text(row.get("field_key"))
        reusable = key in carry_over and key not in selection_keys and row.get("status") == "known"
        policy = "carry_over" if reusable else (
            "branch_history_only" if key in selection_keys else "historical_only"
        )
        (profile if reusable else historical).append(
            _memory_fact(row, reuse_policy=policy)
        )
    return profile, historical


def _agent_activity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    activity = []
    for proof in rows:
        final = proof.get("final_decision") or {}
        result = proof.get("proof_result") or {}
        activity.append({
            "at": proof.get("created_at"),
            "agent": result.get("agent_slug") or final.get("agent_slug") or "legacy_agent",
            "role": result.get("agent_role") or final.get("route") or "legacy",
            "policy_version": result.get("policy_version") or "legacy",
            "journey_action": result.get("journey_action"),
            "canonical_inbound_id": proof.get("canonical_inbound_id"),
        })
    return activity


def _bounded_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining_chars = MAX_SEMANTIC_TOKENS * 4
    for row in reversed(messages[-MAX_RECENT_MESSAGES:]):
        content = str(row.get("content") or row.get("texto") or "")
        if not content:
            continue
        clipped = content[-remaining_chars:]
        if not clipped:
            break
        selected.append({
            key: value for key, value in row.items()
            if key in {"id", "role", "direction", "created_at", "external_message_id", "metadata"}
        } | {"content": clipped})
        remaining_chars -= len(clipped)
        if remaining_chars <= 0:
            break
    return list(reversed(selected))


def _pending_items(
    profile: list[SharedMemoryFact], historical: list[SharedMemoryFact],
    current: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    current_id = _text((current or {}).get("id"))
    return [
        fact.model_dump(mode="json")
        for fact in [*profile, *historical]
        if fact.status in {"needs_confirmation", "invalid"}
        and (not current_id or fact.journey_id in {None, current_id})
    ]


def _outcomes(batch: dict[str, Any]) -> list[dict[str, Any]]:
    values = batch.get("journey_outcomes")
    return list(values if values is not None else batch.get("journeys") or [])


def project_shared_lead_memory(
    *, batch: dict[str, Any], document: dict[str, Any], messages: list[dict[str, Any]],
) -> SharedLeadMemory:
    raw_facts = list(batch.get("memory_facts") or batch.get("facts") or [])
    facts = _latest_current_facts(raw_facts)
    profile, historical = _partition_facts(
        facts, _carry_over_keys(document), _branch_selection_keys(document),
    )

    current = batch.get("journey") or None
    return SharedLeadMemory(
        profile_facts=profile,
        current_journey=current,
        historical_facts=historical,
        journey_outcomes=_outcomes(batch),
        pending_items=_pending_items(profile, historical, current),
        recent_messages=_bounded_messages(messages),
        agent_activity=_agent_activity(list(batch.get("agent_activity") or [])),
        policy_version=POLICY_VERSION,
    )


def facts_by_key(memory: SharedLeadMemory) -> dict[str, list[dict[str, Any]]]:
    """Authoritative complete fact map; contract projection happens later."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for fact in [*memory.profile_facts, *memory.historical_facts]:
        grouped.setdefault(fact.key, []).append({
            "field_key": fact.key,
            "value": fact.value,
            "owner_node_id": fact.owner_node_id,
            "status": fact.status.value,
            "confidence": fact.confidence,
            "source_message_id": fact.source_message_id,
            "metadata": {**fact.metadata, "reuse_policy": fact.reuse_policy},
            "journey_id": fact.journey_id,
        })
    return grouped
