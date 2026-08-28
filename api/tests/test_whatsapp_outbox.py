import pytest
from fastapi import HTTPException

from services.whatsapp_outbox import _recipient_for_lead


def test_recipient_prefers_canonical_external_identity():
    lead = {
        "external_contact_id": "5511999999999",
        "telefone": "5511888888888",
        "metadata": {"identities": {"remote_jid_alt": "5511777777777@s.whatsapp.net"}},
    }

    assert _recipient_for_lead(lead) == "5511777777777"


def test_recipient_falls_back_to_phone_for_manual_lead():
    assert _recipient_for_lead({"telefone": "+55 (11) 98888-7777"}) == "5511988887777"


@pytest.mark.parametrize("lead", [{}, {"telefone": "123"}, {"external_contact_id": "abc@lid"}])
def test_recipient_rejects_missing_or_invalid_identity(lead):
    with pytest.raises(HTTPException) as exc:
        _recipient_for_lead(lead)
    assert exc.value.status_code == 409

