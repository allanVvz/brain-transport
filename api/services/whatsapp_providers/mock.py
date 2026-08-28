"""In-memory WhatsApp provider for safe E2E testing. Never touches the
network â€” no Evolution instance, no Meta Graph API call, no real WhatsApp
session â€” so a Baita<->Aurora E2E run can exercise
whatsapp_dispatch_worker._dispatch_outbound end to end without any risk of
sending a real message.

Implements the same shape as EvolutionWhatsAppProvider/MetaWhatsAppProvider
(services.whatsapp_providers.base.WhatsAppProvider) so it's a drop-in
substitute wherever get_provider(...) is called.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any

from services.whatsapp_providers.evolution import EvolutionWhatsAppProvider

logger = logging.getLogger("whatsapp_providers.mock")

# Smallest valid PNG (1x1, transparent). Returned by get_media_base64 so a
# media E2E run has real decodable bytes without shipping a fixture file.
_STUB_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


class MockWhatsAppProvider:
    # registry.get_provider() constructs a fresh instance per call site, so
    # the log below is a *class* attribute: every instance in this process
    # shares it. That's what lets a test build its own MockWhatsAppProvider()
    # to assert on sends made by a completely different call site (e.g. the
    # dispatch worker running in a background thread/process within the
    # same test session).
    _lock = threading.Lock()
    sent: list[dict[str, Any]] = []
    instances: dict[str, dict[str, Any]] = {}
    # Overridable by a test that wants to drive a specific file through the
    # ingest pipeline (e.g. a real .ogg to exercise transcription).
    media_bytes: bytes = _STUB_PNG

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls.sent = []
            cls.instances = {}
            cls.media_bytes = _STUB_PNG

    @staticmethod
    def _name(binding: dict[str, Any]) -> str:
        name = str(binding.get("provider_instance_key") or binding.get("id") or "")
        if not name:
            raise RuntimeError("mock binding has no instance key or id")
        return name

    def provision_instance(
        self,
        instance_name: str,
        token: str,
        webhook_url: str,
        *,
        webhook_token: str,
    ) -> dict[str, Any]:
        with self._lock:
            self.instances[instance_name] = {
                "token": token,
                "webhook_url": webhook_url,
                "webhook_token": webhook_token,
                "state": "connecting",
            }
        return {"instance": {"instanceName": instance_name}, "status": "connecting"}

    def get_connection_status(self, binding: dict[str, Any]) -> dict[str, Any]:
        name = self._name(binding)
        state = self.instances.get(name, {}).get("state", "connected")
        return {"instance": {"state": state}}

    def get_qr_code(self, binding: dict[str, Any]) -> dict[str, Any]:
        # No real Baileys session exists to pair with, so the mock skips
        # straight to "connected" instead of returning a fake QR image.
        name = self._name(binding)
        with self._lock:
            self.instances.setdefault(name, {"token": None, "webhook_url": None, "webhook_token": None})
            self.instances[name]["state"] = "connected"
        return {"status": "connected", "qr": None}

    def send_text(self, binding: dict[str, Any], recipient: str, text: str) -> dict[str, Any]:
        return self._record(binding, recipient, {"kind": "text", "text": text})

    def send_media(self, binding: dict[str, Any], recipient: str, media: dict[str, Any]) -> dict[str, Any]:
        return self._record(binding, recipient, {"kind": "media", "media": media})

    def _record(self, binding: dict[str, Any], recipient: str, content: dict[str, Any]) -> dict[str, Any]:
        external_id = f"mock-{uuid.uuid4().hex}"
        entry = {
            "id": external_id,
            "binding_id": binding.get("id"),
            "recipient": recipient,
            "content": content,
            "sent_at": time.time(),
        }
        with self._lock:
            self.sent.append(entry)
        logger.info(
            "mock_whatsapp_send binding=%s recipient=%s external_id=%s kind=%s",
            binding.get("id"), recipient, external_id, content.get("kind"),
        )
        # whatsapp_dispatch_worker._dispatch_outbound reads external id from
        # result["key"]["id"] (Evolution shape), result["messages"][0]["id"]
        # (Meta shape), result["messageId"], or result["id"] â€” cover the
        # simplest two so the mock works regardless of which provider a
        # binding under test claims to be.
        return {"id": external_id, "messageId": external_id}

    def restart_instance(self, binding: dict[str, Any]) -> dict[str, Any]:
        name = self._name(binding)
        with self._lock:
            if name in self.instances:
                self.instances[name]["state"] = "connecting"
        return {"status": "restarted"}

    def logout_instance(self, binding: dict[str, Any]) -> dict[str, Any]:
        name = self._name(binding)
        with self._lock:
            self.instances.pop(name, None)
        return {"status": "logged_out"}

    def normalize_webhook(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        # Mirrors EvolutionWhatsAppProvider.normalize_webhook's output shape
        # (api/scripts/e2e_evolution_fixture.py and the real evolution
        # webhook route already speak this shape), so a test can post a
        # synthetic Evolution-style payload through the same code path
        # regardless of which provider a binding under test claims to be.
        event = str(payload.get("event") or payload.get("type") or "MESSAGES_UPSERT").upper()
        data = payload.get("data") or {}
        key = data.get("key") or {}
        # Reuse Evolution's own extractor so a fixture exercising media takes
        # the exact same path the real provider does.
        media, caption = EvolutionWhatsAppProvider._extract_media(data.get("message") or {})
        if media:
            media["fetch_ref"]["message_key"] = key
        return [{
            "event_type": event,
            "instance": payload.get("instance") or data.get("instance"),
            "external_message_id": key.get("id") or data.get("messageId") or f"mock-in-{uuid.uuid4().hex}",
            "external_contact_id": key.get("remoteJid") or data.get("remoteJid"),
            "remote_jid": key.get("remoteJid") or data.get("remoteJid"),
            "remote_jid_alt": key.get("remoteJidAlt") or data.get("remoteJidAlt"),
            "from_me": bool(key.get("fromMe") or data.get("fromMe")),
            "status": data.get("status"),
            "text": data.get("text") or (data.get("message") or {}).get("conversation") or caption or "",
            "media": media,
            "raw": data,
        }]

    def get_media_base64(self, binding: dict[str, Any], message_key: dict[str, Any]) -> bytes:
        """Deterministic stand-in for the Evolution media download."""
        return self.media_bytes

