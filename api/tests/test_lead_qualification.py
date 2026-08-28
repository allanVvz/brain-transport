import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from services import lead_qualification
from services import conversation_runtime


def _apply(previous, *, model, intent, state, stage="novo"):
    return lead_qualification.calculate(
        previous=previous,
        business_model=model,
        intent=intent,
        state=state,
        current_stage=stage,
        evidence_node_ids=["node:test"],
    )


def test_complete_appointment_caps_at_100_and_qualifies():
    state = {
        "business_model": "appointment",
        "appointment_request": {
            "customer_name": "Ana",
            "service_slug": "lavagem",
            "vehicle_model": "Corolla",
            "vehicle_size": "sedan",
            "condition": "com manchas",
            "desired_date": "31/07",
            "time_window": "tarde",
        },
        "conversation_state": "handoff",
    }
    qualification, stage = _apply(
        None,
        model="appointment",
        intent="complete_booking_request",
        state=state,
    )
    assert qualification["score"] == 100
    assert stage == "qualificado"


def test_complete_sales_order_scores_95_and_qualifies():
    state = {
        "items": [{"product_slug": "agua", "quantity": 2, "unit_price": 5}],
        "customer_name": "Ana",
        "address": {"street": "Rua QA", "number": "100"},
        "confirmation_status": "confirmed_pending_human",
        "conversation_state": "handoff",
    }
    qualification, stage = _apply(
        None, model="sales", intent="confirm_order", state=state,
    )
    assert qualification["score"] == 95
    assert stage == "qualificado"


def test_release_e2e_and_webscraping_are_only_visible_in_validation_scope():
    rows = [
        {"id": 1, "lead_id": "real", "metadata": {}},
        {"id": 2, "lead_id": "validator_session", "metadata": {}},
        {"id": 3, "lead_id": "e2e", "metadata": {"e2e_run": "release-1"}},
    ]
    assert [row["id"] for row in lead_qualification.filter_validation_scope(rows)] == [1]
    assert [
        row["id"]
        for row in lead_qualification.filter_validation_scope(rows, "only")
    ] == [2, 3]


def test_model_fields_append_name_and_address_without_overwriting_state():
    state = conversation_runtime._apply_model_fields(
        {
            "customer_name": "Nome confirmado",
            "address": {"street": "Rua existente"},
        },
        {
            "fields": {
                "customer_name": "Nome sugerido",
                "delivery_address": {
                    "street": "Rua sugerida",
                    "number": "123",
                    "city": "SÃ£o Paulo",
                },
            },
        },
        business_model="sales",
    )

    assert state["customer_name"] == "Nome confirmado"
    assert state["address"] == {
        "street": "Rua existente",
        "number": "123",
        "city": "SÃ£o Paulo",
    }


def test_complete_model_fields_make_sales_lead_qualified_but_not_opportunity():
    state = conversation_runtime._apply_model_fields(
        {"items": [{"product_slug": "produto-1", "quantity": 1, "unit_price": 10}]},
        {
            "fields": {
                "customer_name": "VitÃ³ria",
                "delivery_address": {"street": "Rua A", "number": "10"},
            },
        },
        business_model="sales",
    )

    qualification, stage = lead_qualification.calculate(
        previous=None,
        business_model="sales",
        intent="add_item",
        state=state,
        current_stage="novo",
    )

    assert stage == "qualificado"
    assert qualification["calculated_stage"] == "qualificado"


def test_signals_are_idempotent_and_terminal_stage_is_preserved():
    state = {"items": [{"product_slug": "agua", "quantity": 1, "unit_price": 5}]}
    first, _ = _apply(None, model="sales", intent="add_item", state=state)
    second, stage = _apply(
        first, model="sales", intent="add_item", state=state, stage="fechado",
    )
    assert second["score"] == first["score"]
    assert len(second["signals"]) == len(first["signals"])
    assert stage == "fechado"


def test_complaint_does_not_receive_commercial_signals():
    qualification, stage = _apply(
        None, model="appointment", intent="exceptional_support",
        state={"business_model": "appointment", "conversation_state": "handoff"},
    )
    assert qualification["score"] == 5
    assert [item["key"] for item in qualification["signals"]] == ["first_contact"]
    assert stage == "contatado"

