"""supabase_client.merge_commercial_note â€” shared by the admin
(routes.leads) and client-portal (routes.portal) lead-update endpoints,
so an edit from either surface lands the same way: the display-only
commercial_note mirror AND conversation_state.appointment_request (the
AI's actual working memory) both get the new values, and any
now-answered field drops out of missing_fields.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services import supabase_client


def test_merge_commercial_note_fills_appointment_request_and_clears_missing():
    metadata = {
        "conversation_state": {
            "missing_fields": ["vehicle_size", "condicao", "data_desejada"],
            "appointment_request": {"nome_cliente": "Allan"},
        },
    }
    merged = supabase_client.merge_commercial_note(
        metadata, {"condicao": "risco fundo na porta", "data_desejada": "amanha"},
    )
    state = merged["conversation_state"]
    assert state["appointment_request"]["condicao"] == "risco fundo na porta"
    assert state["appointment_request"]["data_desejada"] == "amanha"
    assert state["appointment_request"]["nome_cliente"] == "Allan"
    assert state["missing_fields"] == ["vehicle_size"]
    assert merged["commercial_note"]["condicao"] == "risco fundo na porta"
    assert merged["commercial_note"]["source"] == "manual"


def test_merge_commercial_note_drops_blank_keys_and_values():
    merged = supabase_client.merge_commercial_note(
        {}, {"": "x", "  ": "y", "servico": "  ", "modelo_veiculo": " Onix "},
    )
    assert merged["commercial_note"]["modelo_veiculo"] == "Onix"
    assert "servico" not in merged["commercial_note"]
    assert list(merged["commercial_note"].keys()) == ["modelo_veiculo", "updated_at", "source"]


def test_merge_commercial_note_handles_empty_starting_metadata():
    merged = supabase_client.merge_commercial_note(None, {"servico": "pintura"})
    assert merged["commercial_note"]["servico"] == "pintura"
    assert merged["conversation_state"]["appointment_request"]["servico"] == "pintura"

