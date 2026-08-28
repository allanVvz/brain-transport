"""Deterministic, additive lead qualification shared by sales and appointments."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from services import graph_agent_runtime_v3


VERSION = "qualification_v1"
MANUAL_TERMINAL_STAGES = {"fechado", "perdido"}
STAGES = ("novo", "contatado", "engajado", "qualificado", "oportunidade")
POINTS = {
    "first_contact": 5, "discovery": 4, "product_or_service": 10,
    "price_query": 10, "request_started": 20, "quantity": 10, "name": 5,
    "complete_address": 15, "vehicle_model": 5, "vehicle_size": 5,
    "vehicle_condition": 5, "date": 10, "time_window": 10,
    "completed_request": 20,
}
LABELS = {
    "first_contact": "Primeiro contato",
    "discovery": "Descoberta ou listagem",
    "product_or_service": "Produto ou serviço identificado",
    "price_query": "Consulta de preço",
    "request_started": "Pedido, orçamento ou agendamento iniciado",
    "quantity": "Quantidade informada", "name": "Nome informado",
    "complete_address": "Endereço completo", "vehicle_model": "Modelo do veículo",
    "vehicle_size": "Porte do veículo", "vehicle_condition": "Condição do veículo",
    "date": "Data solicitada", "time_window": "Período solicitado",
    "completed_request": "Solicitação completa ou confirmada",
}


def stage_for_score(score: int) -> str:
    if score <= 0:
        return "novo"
    if score <= 14:
        return "contatado"
    if score <= 34:
        return "engajado"
    # Qualification is the highest automatic stage. Opportunity is a later
    # explicit commercial decision, not a side effect of accumulating points.
    return "qualificado"


def _has_complete_address(state: dict[str, Any]) -> bool:
    address = state.get("address")
    if isinstance(address, dict):
        return bool(address.get("street") and address.get("number"))
    return bool(isinstance(address, str) and len(address.strip()) >= 8)


def _detected_signals(
    *, business_model: str, intent: str, state: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    appointment = business_model == "appointment"
    request = dict(state.get("appointment_request") or {})
    items = list(state.get("items") or [])
    decision_product = state.get("_decision_product_slug")
    detected = {
        "first_contact": {"intent": intent, "evidence": "deterministic_interaction"}
    }

    def add(key: str, evidence: Any) -> None:
        detected[key] = {"intent": intent, "evidence": evidence}

    if intent in {"greeting", "list_services", "consult_category"}:
        add("discovery", intent)
    if request.get("service_slug") or decision_product or items or intent in {"consult_service", "consult_product"}:
        add(
            "product_or_service",
            request.get("service_slug")
            or decision_product
            or [i.get("product_slug") for i in items],
        )
    if (
        intent in {"consult_price", "complete_booking_request"}
        or any(i.get("unit_price") is not None for i in items)
    ):
        add("price_query", intent)
    if (
        intent in {
            "request_quote", "request_booking", "provide_vehicle",
            "provide_condition", "provide_date", "provide_time_window",
            "add_item", "change_quantity", "provide_name", "provide_address",
            "confirm_order", "complete_booking_request",
        }
        or items
        or state.get("conversation_state") in {
            "collecting", "awaiting_name", "awaiting_address", "awaiting_confirmation",
        }
    ):
        add("request_started", intent)
    if items and any(int(i.get("quantity") or 0) > 0 for i in items):
        add("quantity", [i.get("quantity") for i in items])
    name = request.get("customer_name") if appointment else state.get("customer_name")
    if name:
        add("name", name)
    if not appointment and _has_complete_address(state):
        add("complete_address", state.get("address"))
    if appointment:
        for field, signal in (
            ("vehicle_model", "vehicle_model"), ("vehicle_size", "vehicle_size"),
            ("condition", "vehicle_condition"), ("desired_date", "date"),
            ("time_window", "time_window"),
        ):
            if request.get(field):
                add(signal, request.get(field))
    if intent in {"confirm_order", "complete_booking_request"} or state.get("confirmation_status") == "confirmed_pending_human":
        add("completed_request", intent)
    return detected


def calculate(
    *, previous: dict[str, Any] | None, business_model: str, intent: str,
    state: dict[str, Any] | None, current_stage: str | None,
    evidence_node_ids: list[str] | None = None, update_stage: bool = True,
) -> tuple[dict[str, Any], str]:
    previous = dict(previous or {})
    accumulated = {
        str(item.get("key")): dict(item)
        for item in previous.get("signals") or []
        if isinstance(item, dict) and item.get("key") in POINTS
    }
    for key, detail in _detected_signals(
        business_model=business_model, intent=intent, state=dict(state or {}),
    ).items():
        if key not in accumulated:
            accumulated[key] = {
                "key": key, "label": LABELS[key], "points": POINTS[key],
                "intent": detail.get("intent"), "evidence": detail.get("evidence"),
                "evidence_node_ids": list(evidence_node_ids or []),
            }
    signals = [accumulated[key] for key in POINTS if key in accumulated]
    score = min(100, sum(int(item["points"]) for item in signals))
    calculated_stage = stage_for_score(score)
    current = str(current_stage or "novo").lower()
    if current in MANUAL_TERMINAL_STAGES:
        next_stage, stage_source = current, "manual_terminal"
    elif not update_stage:
        next_stage, stage_source = current, "backfill_preserved"
    else:
        try:
            next_stage = STAGES[max(STAGES.index(current), STAGES.index(calculated_stage))]
        except ValueError:
            next_stage = calculated_stage
        stage_source = "deterministic_qualification"
    qualification = {
        "version": VERSION, "score": score, "signals": signals,
        "points_by_signal": {item["key"]: item["points"] for item in signals},
        "source_intent": intent,
        "source_evidence_node_ids": list(evidence_node_ids or []),
        "calculated_stage": calculated_stage, "stage": next_stage,
        "stage_source": stage_source,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    return qualification, next_stage


def _v3_progress_score(qualification: dict[str, Any]) -> int:
    """0-50%, reaching exactly 50 the moment qualification_complete flips
    true -- the same moment the graph's handoff_rule authorizes handoff.
    """
    if qualification.get("complete"):
        return 50
    total = int(qualification.get("required_field_count") or 0)
    if total > 0:
        resolved_count = int(qualification.get("resolved_required_count") or 0)
    else:
        # Leads persisted before this field was added never got
        # required_field_count/resolved_required_count -- fall back to the
        # coarser field-key counts that were always present so old leads
        # still show a proportional score instead of reading 0 forever.
        resolved_count = len(qualification.get("resolved_fields") or [])
        total = resolved_count + len(qualification.get("missing_fields") or [])
    if total <= 0:
        return 0
    return min(50, round(50 * resolved_count / total))


def score_for_display(lead: dict[str, Any]) -> int:
    """Single source of truth for the displayed qualification_score.

    Manual pipeline stage always wins over any computed value:
      - fechado          -> 100
      - oportunidade     -> 75
      - nao_qualificado  -> 0 (explicit manual disqualification)
      - perdido          -> the score snapshotted the moment the lead was
                             manually moved to "perdido" (a historical
                             high-water mark, never recomputed)
      - otherwise, v3 leads get the 0-50% branch-progress formula and
        legacy leads keep their additive 0-100 score unchanged.
    """
    metadata = dict(lead.get("metadata") or {})
    qualification = dict(metadata.get("qualification") or {})
    stage = str(lead.get("stage") or "").strip().lower()
    if stage == "fechado":
        return 100
    if stage == "oportunidade":
        return 75
    if stage == "nao_qualificado":
        return 0
    if stage == "perdido":
        frozen = qualification.get("score_at_perdido")
        return int(frozen) if frozen is not None else int(qualification.get("score") or 0)
    if qualification.get("version") == graph_agent_runtime_v3.RUNTIME_VERSION:
        return _v3_progress_score(qualification)
    return int(qualification.get("score") or 0)


def decorate_lead(lead: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(lead.get("metadata") or {})
    qualification = dict(metadata.get("qualification") or {})
    lead_id = str(lead.get("lead_id") or "")
    canonical_validation = (
        dict(metadata.get("validation") or {})
        if isinstance(metadata.get("validation"), dict)
        else {}
    )
    validation = bool(canonical_validation.get("is_validation")) or lead_id.startswith("validator_") or bool(
        metadata.get("validation_scenario")
        or metadata.get("validator_scenario")
        or metadata.get("validation_session_id")
        or metadata.get("validator_session_id")
        or metadata.get("e2e_run")
    )
    source = canonical_validation.get("source")
    if not source and metadata.get("e2e_run"):
        source = "release_e2e"
    if not source and validation:
        source = "webscraping" if lead_id.startswith("validator_") else "legacy"
    run_id = (
        canonical_validation.get("run_id")
        or metadata.get("e2e_run")
        or metadata.get("validation_session_id")
        or metadata.get("validator_session_id")
    )
    return {
        **lead,
        "qualification": qualification,
        "qualification_score": score_for_display(lead),
        "qualification_signals": qualification.get("signals") or [],
        "validation": {
            "is_validation": validation,
            "source": source,
            "run_id": run_id,
            "counterpart_persona_slug": canonical_validation.get("counterpart_persona_slug"),
            "scenario": (
                canonical_validation.get("scenario")
                or metadata.get("validation_scenario")
                or metadata.get("validator_scenario")
            ),
            "session_id": (
                canonical_validation.get("session_id")
                or metadata.get("validation_session_id")
                or metadata.get("validator_session_id")
            ),
        },
    }


def is_validation_lead(lead: dict[str, Any]) -> bool:
    return bool(decorate_lead(lead).get("validation", {}).get("is_validation"))


def filter_validation_scope(
    rows: list[dict[str, Any]],
    scope: str = "exclude",
) -> list[dict[str, Any]]:
    normalized = str(scope or "exclude").strip().lower()
    if normalized not in {"exclude", "only", "all"}:
        normalized = "exclude"
    decorated = [decorate_lead(row) for row in rows]
    if normalized == "all":
        return decorated
    expected = normalized == "only"
    return [
        row
        for row in decorated
        if bool(row.get("validation", {}).get("is_validation")) is expected
    ]
