"""Convert WA Validator evidence into reviewable Sofia authoring proposals.

This module is pure. It never writes graph state, calls a publisher, or turns
an observed failure into an invented business fact.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def _proposal_kind(topic: str) -> tuple[str, list[str], str, str]:
    folded = topic.lower()
    if any(value in folded for value in ("branch", "service_operation", "selector")):
        return (
            "branch_resolution_review",
            ["audience", "rule"],
            "graph.create_card_draft",
            "Revisar aliases e a política declarativa de seleção do galho.",
        )
    if any(value in folded for value in ("missing_field", "question", "repetition")):
        return (
            "qualification_contract_review",
            ["faq", "rule"],
            "graph.create_card_draft",
            "Revisar field, question_node_id e orçamento de repetição no contrato publicado.",
        )
    if any(value in folded for value in ("claim", "evidence", "knowledge")):
        return (
            "knowledge_gap",
            ["knowledge_item", "faq"],
            "graph.create_card_draft",
            "Solicitar fonte humana; criar conteúdo apenas como pending_source/pending_validation.",
        )
    if any(value in folded for value in ("lineage", "checksum", "publication")):
        return (
            "publication_drift",
            [],
            "graph.validate_patch",
            "Reconciliar versão/checksum; não alterar conteúdo para mascarar drift.",
        )
    return (
        "runtime_or_instrumentation_review",
        ["rule"],
        "graph.validate_patch",
        "Localizar a falha entre runtime, proof e contrato antes de propor uma edição.",
    )


def build_sofia_review(
    *, persona_slug: str, session_id: str, gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    proposals: list[dict[str, Any]] = []
    for gap in gaps:
        topic = str(gap.get("topic") or "unknown_gap")
        evidence = str(gap.get("evidence") or "")
        kind, node_types, tool, recommendation = _proposal_kind(topic)
        identity = json.dumps(
            [persona_slug, session_id, topic, evidence],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        proposals.append({
            "proposal_id": "sofia-gap-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
            "kind": kind,
            "status": "pending_human_review",
            "source": f"wa_validator_session:{session_id}",
            "persona_slug": persona_slug,
            "evidence": {"topic": topic, "detail": evidence},
            "target_node_types": node_types,
            "recommended_tool": tool,
            "recommendation": recommendation,
            "automatic_mutation": False,
            "publication_allowed": False,
        })
    return {
        "status": "pending_human_review" if proposals else "no_gaps",
        "automatic_mutation": False,
        "tool_sequence": ["graph.create_card_draft", "graph.validate_patch"],
        "proposals": proposals,
    }
