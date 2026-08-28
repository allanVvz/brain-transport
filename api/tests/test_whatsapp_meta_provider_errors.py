"""Regression coverage for Meta Cloud error-detail capture.

Before this, a rejected send only ever surfaced the generic httpx status
line ("Client error '400 Bad Request' for url ...") in logs/dashboard.
Meta's own error body (unapproved template, wrong parameter count, invalid
recipient, ...) was read by nobody and discarded, so every failure looked
identical and undiagnosable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.whatsapp_providers.meta import MetaWhatsAppProvider


def _binding() -> dict:
    return {
        "whatsapp_phone_number_id": "949967854877404",
        "provider_secret_ciphertext": "encrypted",
    }


def _fake_response(status_code: int, body: dict) -> httpx.Response:
    request = httpx.Request("POST", "https://graph.facebook.com/v21.0/x/messages")
    return httpx.Response(status_code, json=body, request=request)


@pytest.fixture(autouse=True)
def _fake_credential(monkeypatch):
    monkeypatch.setattr(
        "services.whatsapp_providers.meta.secret_store.decrypt_secret",
        lambda _ciphertext: json.dumps({"access_token": "token-123", "api_version": "v21.0"}),
    )


def test_send_template_surfaces_meta_error_message(monkeypatch):
    meta_error_body = {
        "error": {
            "message": "(#132001) Template name does not exist in the translation",
            "type": "OAuthException",
            "code": 132001,
            "error_subcode": None,
            "fbtrace_id": "AbCdEf",
        }
    }
    monkeypatch.setattr(
        "services.whatsapp_providers.meta.httpx.post",
        lambda *_a, **_k: _fake_response(400, meta_error_body),
    )

    provider = MetaWhatsAppProvider()
    with pytest.raises(httpx.HTTPStatusError) as exc:
        provider.send_template(
            _binding(), "5551992623375",
            template_name="boas_vindas", template_language="pt_BR", components=[],
        )

    message = str(exc.value)
    assert "400" in message
    assert "code=132001" in message
    assert "Template name does not exist" in message


def test_send_text_surfaces_meta_error_message(monkeypatch):
    meta_error_body = {"error": {"message": "Invalid recipient phone number", "code": 100}}
    monkeypatch.setattr(
        "services.whatsapp_providers.meta.httpx.post",
        lambda *_a, **_k: _fake_response(400, meta_error_body),
    )

    provider = MetaWhatsAppProvider()
    with pytest.raises(httpx.HTTPStatusError) as exc:
        provider.send_text(_binding(), "invalid", "oi")

    assert "Invalid recipient phone number" in str(exc.value)


def test_non_json_error_body_falls_back_to_raw_text(monkeypatch):
    request = httpx.Request("POST", "https://graph.facebook.com/v21.0/x/messages")
    response = httpx.Response(500, text="upstream timeout", request=request)
    monkeypatch.setattr(
        "services.whatsapp_providers.meta.httpx.post",
        lambda *_a, **_k: response,
    )

    provider = MetaWhatsAppProvider()
    with pytest.raises(httpx.HTTPStatusError) as exc:
        provider.send_text(_binding(), "5551992623375", "oi")

    assert "upstream timeout" in str(exc.value)


def test_successful_send_is_unaffected(monkeypatch):
    request = httpx.Request("POST", "https://graph.facebook.com/v21.0/x/messages")
    ok_response = httpx.Response(200, json={"messages": [{"id": "wamid.123"}]}, request=request)
    monkeypatch.setattr(
        "services.whatsapp_providers.meta.httpx.post",
        lambda *_a, **_k: ok_response,
    )

    provider = MetaWhatsAppProvider()
    result = provider.send_text(_binding(), "5551992623375", "oi")

    assert result["messages"][0]["id"] == "wamid.123"

