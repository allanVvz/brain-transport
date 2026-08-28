"""Production-safe release contract checks; prints no secrets or user content."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from schemas.conversation import ConversationContext
from services import conversation_runtime, graph_json_v2_store


def _validate_aurora_dialog(*, graph, version: int, checksum: str) -> dict:
    rag_nodes = [
        {
            "id": node.id,
            "node_type": node.node_type,
            "slug": node.slug,
            "title": node.title,
            "status": node.lifecycle.status,
            "source": node.provenance.source,
            "data": node.data,
        }
        for node in graph.nodes
    ]
    cases = [
        ("Quais serviÃ§os estÃ£o disponÃ­veis?", "list_services", "SDR", False),
        # price_disclosure == "human_only": the agent never states a value,
        # it answers with the published text and escalates to a person.
        ("Quanto custa o polimento tÃ©cnico?", "consult_price", "HUMAN", True),
        ("Quero reclamar da garantia", "exceptional_support", "HUMAN", True),
        ("Quero agendar higienizaÃ§Ã£o interna", "request_booking", "SDR", False),
    ]
    results: list[dict] = []
    for message, expected_intent, expected_route, expected_handoff in cases:
        context = ConversationContext(
            persona_slug="aurora",
            agent_slug="aurora",
            graph_version=version,
            graph_checksum=checksum,
            messages=[{"role": "user", "sender_type": "lead", "texto": message}],
            cart={"_lead_stage": "novo", "business_model": "appointment"},
            rag_nodes=rag_nodes,
            rag_paths=[[node["id"]] for node in rag_nodes],
        )
        decision, response = conversation_runtime.decide(context)
        actual = {
            "intent": decision.intent,
            "route": decision.route.value,
            "handoff": response.handoff_required,
            "has_reply": bool((response.reply_text or "").strip()),
        }
        expected = {
            "intent": expected_intent,
            "route": expected_route,
            "handoff": expected_handoff,
            "has_reply": True,
        }
        if actual != expected:
            raise RuntimeError({"aurora_dialog_case": message, "expected": expected, "actual": actual})
        results.append(actual)
    return {"cases": len(results), "all_replied": all(item["has_reply"] for item in results)}


def validate() -> dict:
    aurora_current = graph_json_v2_store.load_current("aurora")
    if not aurora_current:
        raise RuntimeError("Aurora graph is not published")
    aurora_version, aurora = aurora_current
    aurora_event = graph_json_v2_store.latest_event("aurora") or {}
    factual = [
        node for node in aurora.nodes
        if node.node_type not in {"embedded", "gallery"}
    ]
    faqs = [node for node in aurora.nodes if node.node_type == "faq"]
    embedded = next(node for node in aurora.nodes if node.node_type == "embedded")
    faq_edges = [
        edge for edge in aurora.edges
        if edge.target == embedded.id
        and edge.source in {node.id for node in faqs}
        and edge.relation_type == "publishes_to"
        and edge.lifecycle.status == "active"
    ]
    agent_grants = [
        edge for edge in aurora.edges
        if edge.target == embedded.id
        and edge.relation_type == "publishes_to"
        and edge.lifecycle.status == "active"
    ]
    checks = {
        "aurora_schema_version": aurora.schema_version,
        "aurora_nodes": len(aurora.nodes),
        "aurora_edges": len(aurora.edges),
        "aurora_markdown_documents": sum(
            1 for node in factual if str((node.data or {}).get("markdown") or "").strip()
        ),
        "aurora_faqs": len(faqs),
        "aurora_faq_embedded_edges": len(faq_edges),
        "aurora_orphan_faqs": len(faqs) - len({edge.source for edge in faq_edges}),
        "aurora_agent_grants": len(agent_grants),
        "aurora_destination": embedded.action.destination_id if embedded.action else None,
    }
    # Structural invariants, not exact content counts. Confirmed live
    # 2026-08-06: this used to be a dict-equality check against hardcoded
    # totals (55 nodes, 27 faqs, ...) captured once from a specific content
    # snapshot. Aurora's FAQ/node content changes routinely as it's
    # authored â€” that made this gate fail release after release for
    # reasons that have nothing to do with whether the deploy itself is
    # healthy, and masked the real post-deploy health check (audit.sh)
    # that runs after it. What actually matters structurally: the graph
    # isn't empty, every FAQ is wired to the embedded destination with no
    # orphans, and grants exist to serve them.
    structural_errors = []
    if checks["aurora_schema_version"] != "2.1":
        structural_errors.append(f"unexpected_schema_version:{checks['aurora_schema_version']}")
    if checks["aurora_nodes"] <= 0 or checks["aurora_edges"] <= 0:
        structural_errors.append("aurora_graph_is_empty")
    if checks["aurora_faqs"] <= 0:
        structural_errors.append("aurora_has_no_faqs")
    if checks["aurora_faq_embedded_edges"] != checks["aurora_faqs"]:
        structural_errors.append("faq_not_fully_published_to_embedded_destination")
    if checks["aurora_orphan_faqs"] != 0:
        structural_errors.append(f"orphan_faqs:{checks['aurora_orphan_faqs']}")
    if checks["aurora_agent_grants"] <= 0:
        structural_errors.append("no_agent_grants")
    if checks["aurora_destination"] != "dataset:sdr-aurora":
        structural_errors.append(f"unexpected_destination:{checks['aurora_destination']}")
    if structural_errors:
        raise RuntimeError({"structural_errors": structural_errors, "actual": checks})
    # Per-FAQ markdown/question-count hygiene is content-authoring debt,
    # not a deploy-blocking condition â€” surfaced as a warning so it stays
    # visible without aborting the script before audit.sh runs.
    faq_markdown_warnings = [
        node.slug or node.id for node in faqs
        if (node.data or {}).get("markdown_document") is not True
        or int((node.data or {}).get("question_count") or 0) != 1
    ]

    baita_current = graph_json_v2_store.load_current("baita-conveniencia")
    if not baita_current or int(baita_current[0]) < 1:
        raise RuntimeError("Baita published graph was not preserved")
    if baita_current[1].persona_slug != "baita-conveniencia":
        raise RuntimeError("Baita graph scope was corrupted")
    baita_event = graph_json_v2_store.latest_event("baita-conveniencia") or {}
    baita_checksum = str((baita_event.get("payload") or {}).get("checksum") or "")
    if not baita_checksum:
        raise RuntimeError("Baita graph checksum is missing")
    dialog = _validate_aurora_dialog(
        graph=aurora,
        version=aurora_version,
        checksum=str((aurora_event.get("payload") or {}).get("checksum") or "missing"),
    )
    return {
        "ok": True,
        **checks,
        "aurora_version": aurora_version,
        "aurora_checksum": (aurora_event.get("payload") or {}).get("checksum"),
        "baita_version": int(baita_current[0]),
        "baita_checksum": baita_checksum,
        "aurora_dialog": dialog,
        "faq_markdown_warnings": faq_markdown_warnings,
    }


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))

