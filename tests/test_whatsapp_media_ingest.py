"""Inbound WhatsApp media must be captured, not discarded.

Regression tests for the gap this feature closes: the Evolution normalizer
only ever read `conversation` and `extendedTextMessage.text`, so an
imageMessage / audioMessage / documentMessage produced an empty string. A
customer sending a voice note asking for a price reached the operator as
"[mensagem sem texto]" and the agent as "".

The tests cover the three points where the file could still be lost:
  1. the provider normalizers (Evolution, Meta, mock) must emit a descriptor
  2. the webhook must persist an asset and hold dispatch
  3. a text-only message must be completely unaffected
"""
from __future__ import annotations

import hashlib
import hmac
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "api"
for path in (API_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# ── normalizers ──────────────────────────────────────────────────────────

def _evolution_event(message: dict) -> dict:
    from services.whatsapp_providers.evolution import EvolutionWhatsAppProvider

    payload = {
        "event": "messages.upsert",
        "instance": "brain-test",
        "data": {
            "key": {"id": "WAMID1", "remoteJid": "5511999999999@s.whatsapp.net"},
            "message": message,
        },
    }
    return EvolutionWhatsAppProvider().normalize_webhook(payload)[0]


def test_evolution_extracts_voice_note():
    event = _evolution_event({
        "audioMessage": {
            "mimetype": "audio/ogg; codecs=opus",
            "ptt": True,
            "seconds": 7,
            "fileLength": 4211,
        }
    })
    media = event["media"]
    assert media["kind"] == "audio"
    assert media["voice_note"] is True
    assert media["duration_seconds"] == 7
    assert media["reading_status"] == "pending"
    assert media["fetch_ref"]["strategy"] == "evolution_base64"
    # The key is what Evolution needs to decrypt and return the bytes.
    assert media["fetch_ref"]["message_key"]["id"] == "WAMID1"


def test_evolution_image_caption_becomes_the_message_text():
    """A captioned photo must keep its caption as the user's turn."""
    event = _evolution_event({
        "imageMessage": {"mimetype": "image/jpeg", "caption": "quanto custa essa?"}
    })
    assert event["media"]["kind"] == "image"
    assert event["text"] == "quanto custa essa?"
    assert event["media"]["caption"] == "quanto custa essa?"


def test_evolution_document_keeps_filename():
    event = _evolution_event({
        "documentMessage": {"mimetype": "application/pdf", "fileName": "pedido.pdf"}
    })
    assert event["media"]["kind"] == "document"
    assert event["media"]["filename"] == "pedido.pdf"


def test_evolution_text_message_has_no_media():
    """The text path must be untouched by media support."""
    event = _evolution_event({"conversation": "oi, tudo bem?"})
    assert event["media"] is None
    assert event["text"] == "oi, tudo bem?"


def test_meta_describes_media_and_ignores_text():
    from services.whatsapp_providers.meta import MetaWhatsAppProvider

    media = MetaWhatsAppProvider.describe_media({
        "type": "audio",
        "audio": {"id": "MEDIA-9", "mime_type": "audio/ogg", "voice": True},
    })
    assert media["kind"] == "audio"
    assert media["voice_note"] is True
    assert media["fetch_ref"] == {"strategy": "meta_graph", "media_id": "MEDIA-9"}

    assert MetaWhatsAppProvider.describe_media({"type": "text", "text": {"body": "oi"}}) is None


def test_sticker_is_treated_as_an_image():
    from services.whatsapp_providers.meta import MetaWhatsAppProvider

    media = MetaWhatsAppProvider.describe_media({
        "type": "sticker", "sticker": {"id": "S1", "mime_type": "image/webp"},
    })
    assert media["kind"] == "image"


# ── webhook persistence ──────────────────────────────────────────────────

@pytest.fixture
def app_client(monkeypatch):
    from routes import evolution_webhook as mod

    monkeypatch.setenv("EVOLUTION_WEBHOOK_HMAC_SECRET", "test-secret")
    app = FastAPI()
    app.include_router(mod.router)
    return app, mod, TestClient(app)


def _token(binding_id: str) -> str:
    from services import auth_service

    return auth_service._b64encode(
        hmac.new(b"test-secret", binding_id.encode(), hashlib.sha256).digest()
    )


def _wire_binding(monkeypatch, mod, binding_id="binding-media"):
    binding = {
        "id": binding_id,
        "persona_id": "persona-1",
        "provider": "evolution_baileys",
        "provider_instance_key": "brain-test",
        "active": True,
        "metadata": {"mode": "active"},
    }
    monkeypatch.setattr(mod.supabase_client, "get_workflow_binding_by_id", lambda _id: binding)
    monkeypatch.setattr(mod.supabase_client, "ensure_channel_lead", lambda **_k: {"id": 42})
    return binding


def test_webhook_registers_the_asset_and_holds_dispatch(monkeypatch, app_client):
    app, mod, client = app_client
    binding_id = "binding-media"
    _wire_binding(monkeypatch, mod, binding_id)

    enqueued = {}
    monkeypatch.setattr(
        mod.supabase_client,
        "enqueue_whatsapp_envelope",
        lambda **kwargs: enqueued.update(kwargs) or {
            "buffer_id": "buf-1", "message_id": "WAMID-AUDIO", "message_row_id": 7,
        },
    )
    registered = {}
    monkeypatch.setattr(
        mod.media_ingest,
        "register_inbound_media",
        lambda **kwargs: registered.update(kwargs) or {"id": "asset-1"},
    )

    holds = []
    real_debounce = mod.supabase_client.debounce_available_at
    monkeypatch.setattr(
        mod.supabase_client,
        "debounce_available_at",
        lambda seconds=3: holds.append(seconds) or real_debounce(seconds),
    )

    resp = client.post(
        f"/webhooks/evolution/{binding_id}",
        json={
            "event": "messages.upsert",
            "instance": "brain-test",
            "data": {
                "key": {"id": "WAMID-AUDIO", "remoteJid": "5511999999999@s.whatsapp.net"},
                "message": {"audioMessage": {"mimetype": "audio/ogg; codecs=opus", "ptt": True}},
            },
        },
        headers={"x-brain-webhook-token": _token(binding_id)},
    )

    assert resp.status_code == 202
    assert resp.json()["accepted"] == 1

    # Dispatch is held long enough for the worker to read the file.
    assert holds == [mod.media_ingest.MEDIA_HOLD_SECONDS]

    payload = enqueued["buffer"]["payload"]
    assert payload["media"]["kind"] == "audio"
    # Never an empty user turn, even before the transcription lands.
    assert payload["text"] == "[o cliente enviou um audio]"
    assert enqueued["message"]["content"] == payload["text"]

    assert registered["buffer_id"] == "buf-1"
    assert registered["message_row_id"] == 7
    assert registered["descriptor"]["kind"] == "audio"


def test_registration_uses_the_internal_message_row_and_projects_the_asset(monkeypatch):
    from services import media_ingest

    inserted = {}
    linked = []
    monkeypatch.setattr(media_ingest, "resolve_campaign_attribution", lambda *_args: {})
    monkeypatch.setattr(
        media_ingest.supabase_client,
        "insert_inbound_media_asset",
        lambda **kwargs: inserted.update(kwargs) or {"id": "asset-1"},
    )
    monkeypatch.setattr(
        media_ingest.supabase_client,
        "link_inbound_media_asset_to_message",
        lambda message_row_id, asset_id: linked.append((message_row_id, asset_id)) or True,
    )

    asset = media_ingest.register_inbound_media(
        persona_id="persona-1",
        lead={"id": 42},
        descriptor={"kind": "image", "mime": "image/jpeg"},
        buffer_id="buf-1",
        message_row_id=2378,
        binding_id="binding-1",
    )

    assert asset == {"id": "asset-1"}
    assert inserted["message_id"] == 2378
    assert linked == [(2378, "asset-1")]


def test_text_message_is_not_held(monkeypatch, app_client):
    """A plain text message must keep the 3s debounce, not the media hold."""
    app, mod, client = app_client
    binding_id = "binding-media"
    _wire_binding(monkeypatch, mod, binding_id)

    enqueued = {}
    monkeypatch.setattr(
        mod.supabase_client,
        "enqueue_whatsapp_envelope",
        lambda **kwargs: enqueued.update(kwargs) or {
            "buffer_id": "buf-2", "message_id": "WAMID-TEXT", "message_row_id": 8,
        },
    )

    def _no_media(**_kwargs):
        raise AssertionError("a text message must not register media")

    monkeypatch.setattr(mod.media_ingest, "register_inbound_media", _no_media)

    holds = []
    real_debounce = mod.supabase_client.debounce_available_at
    monkeypatch.setattr(
        mod.supabase_client,
        "debounce_available_at",
        lambda seconds=3: holds.append(seconds) or real_debounce(seconds),
    )

    resp = client.post(
        f"/webhooks/evolution/{binding_id}",
        json={
            "event": "messages.upsert",
            "instance": "brain-test",
            "data": {
                "key": {"id": "WAMID-TXT", "remoteJid": "5511999999999@s.whatsapp.net"},
                "message": {"conversation": "bom dia"},
            },
        },
        headers={"x-brain-webhook-token": _token(binding_id)},
    )

    assert resp.status_code == 202
    assert holds == [3]
    assert "media" not in enqueued["buffer"]["payload"]
    assert enqueued["buffer"]["payload"]["text"] == "bom dia"
