from __future__ import annotations

import base64
import os
import time
from typing import Any

import httpx

from services import secret_store
from services.whatsapp_providers import media as _media
from utils.tls import get_ca_bundle_path


class EvolutionWhatsAppProvider:
    def __init__(self) -> None:
        self.base_url = (os.environ.get("EVOLUTION_API_URL") or "http://evolution-api:8080").rstrip("/")
        self.api_key = (os.environ.get("EVOLUTION_AUTHENTICATION_API_KEY") or "").strip()

    def _request(self, method: str, path: str, *, json: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("EVOLUTION_AUTHENTICATION_API_KEY is not configured")
        response = None
        for attempt in range(12):
            try:
                with httpx.Client(
                    base_url=self.base_url,
                    timeout=httpx.Timeout(20.0, connect=5.0),
                    follow_redirects=False,
                    verify=get_ca_bundle_path(),
                    headers={"apikey": self.api_key, "Content-Type": "application/json"},
                ) as client:
                    response = client.request(method, path, json=json)
                break
            except httpx.ConnectError:
                if attempt == 11:
                    raise
                time.sleep(1.0)
        if response is None:
            raise RuntimeError("Evolution API did not return a response")
        response.raise_for_status()
        if not response.content:
            return {"ok": True}
        body = response.json()
        return body if isinstance(body, dict) else {"data": body}

    def provision_instance(
        self,
        instance_name: str,
        token: str,
        webhook_url: str,
        *,
        webhook_token: str,
    ) -> dict[str, Any]:
        return self._request("POST", "/instance/create", json={
            "instanceName": instance_name,
            "integration": "WHATSAPP-BAILEYS",
            "token": token,
            # Provisioning and QR consent are separate portal actions. This
            # avoids rotating a live session or producing a QR unexpectedly.
            "qrcode": False,
            "webhook": {
                "enabled": True,
                "url": webhook_url,
                "webhookByEvents": False,
                "webhookBase64": False,
                "headers": {"X-Brain-Webhook-Token": webhook_token},
                "events": [
                    "MESSAGES_UPSERT", "MESSAGES_UPDATE", "SEND_MESSAGE",
                    "CONNECTION_UPDATE", "QRCODE_UPDATED",
                ],
            },
            "settings": {"syncFullHistory": False, "groupsIgnore": True},
        })

    @staticmethod
    def _name(binding: dict[str, Any]) -> str:
        name = str(binding.get("provider_instance_key") or "")
        if not name:
            raise RuntimeError("Evolution binding has no instance key")
        return name

    @staticmethod
    def _token(binding: dict[str, Any]) -> str | None:
        return secret_store.decrypt_secret(binding.get("provider_secret_ciphertext"))

    def get_connection_status(self, binding: dict[str, Any]) -> dict[str, Any]:
        return self._request("GET", f"/instance/connectionState/{self._name(binding)}")

    def get_qr_code(self, binding: dict[str, Any]) -> dict[str, Any]:
        connection = self.get_connection_status(binding)
        state = str((connection.get("instance") or {}).get("state") or connection.get("state") or "").lower()
        if state in {"open", "connected"}:
            return {"status": "connected", "qr": None}
        path = f"/instance/connect/{self._name(binding)}"
        normalized_qr = None
        # Evolution generates the PNG from the Baileys callback after the
        # connection has entered "connecting". Keep the wait bounded and never
        # persist or log the returned QR payload.
        for attempt in range(12):
            body = self._request("GET", path)
            qr = body.get("base64") or (body.get("qrcode") or {}).get("base64")
            normalized_qr = self._normalize_qr(qr)
            if normalized_qr:
                break
            if attempt < 11:
                time.sleep(0.75)
        return {
            "status": "qr_ready" if normalized_qr else "connecting",
            "qr": {"base64": normalized_qr} if normalized_qr else None,
        }

    @staticmethod
    def _normalize_qr(value: Any) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        candidate = value.strip()
        encoded = candidate
        if candidate.startswith("data:"):
            prefix, separator, encoded = candidate.partition(",")
            if not separator or prefix.lower() != "data:image/png;base64":
                return None
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            return None
        if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
            return None
        return f"data:image/png;base64,{encoded}"

    def send_text(self, binding: dict[str, Any], recipient: str, text: str) -> dict[str, Any]:
        return self._request("POST", f"/message/sendText/{self._name(binding)}", json={
            "number": recipient,
            "text": text,
        })

    def send_media(self, binding: dict[str, Any], recipient: str, media: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/message/sendMedia/{self._name(binding)}", json={
            "number": recipient,
            **media,
        })

    def restart_instance(self, binding: dict[str, Any]) -> dict[str, Any]:
        return self._request("PUT", f"/instance/restart/{self._name(binding)}")

    def logout_instance(self, binding: dict[str, Any]) -> dict[str, Any]:
        return self._request("DELETE", f"/instance/logout/{self._name(binding)}")

    def delete_instance(self, binding: dict[str, Any]) -> dict[str, Any]:
        return self._request("DELETE", f"/instance/delete/{self._name(binding)}")

    def get_media_base64(self, binding: dict[str, Any], message_key: dict[str, Any]) -> bytes:
        """Download the bytes of a received media message.

        Baileys keeps the media encrypted on WhatsApp's CDN, so there is no
        plain URL to GET: Evolution decrypts it on demand and hands it back
        base64-encoded. Called from the media ingest worker, never from the
        webhook â€” the webhook must return before this round trip completes.
        """
        body = self._request(
            "POST",
            f"/chat/getBase64FromMediaMessage/{self._name(binding)}",
            json={"message": {"key": message_key}, "convertToMp4": False},
        )
        encoded = body.get("base64") or (body.get("data") or {}).get("base64")
        if not encoded:
            raise RuntimeError("Evolution returned no base64 payload for media message")
        return base64.b64decode(encoded, validate=True)

    # Message containers Baileys uses for attachments, in the order we probe
    # them. ``sticker`` is last because a sticker reply can ride alongside a
    # quoted image and we prefer the real attachment.
    _MEDIA_CONTAINERS = (
        ("imageMessage", "image"),
        ("audioMessage", "audio"),
        ("videoMessage", "video"),
        ("documentMessage", "document"),
        ("documentWithCaptionMessage", "document"),
        ("stickerMessage", "image"),
    )

    @classmethod
    def _extract_media(cls, message: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        """Return ``(descriptor, caption)`` for the first attachment found."""
        for container, hint in cls._MEDIA_CONTAINERS:
            node = message.get(container)
            if not isinstance(node, dict):
                continue
            # documentWithCaptionMessage wraps a real documentMessage.
            if container == "documentWithCaptionMessage":
                inner = (node.get("message") or {}).get("documentMessage")
                if isinstance(inner, dict):
                    node = inner
            mime = node.get("mimetype") or node.get("mimeType")
            caption = str(node.get("caption") or "")
            return (
                _media.build_descriptor(
                    provider="evolution_baileys",
                    kind=_media.kind_for_mime(mime, hint),
                    mime=mime,
                    filename=node.get("fileName") or node.get("filename"),
                    caption=caption,
                    size=node.get("fileLength") or node.get("fileSize"),
                    voice_note=bool(node.get("ptt")),
                    duration_seconds=node.get("seconds"),
                    # Evolution needs the whole message key back to locate and
                    # decrypt the file; it is small, so we carry it verbatim.
                    fetch_ref={"strategy": "evolution_base64"},
                ),
                caption,
            )
        return None, ""

    def normalize_webhook(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        event = str(payload.get("event") or payload.get("type") or "").upper().replace(".", "_")
        data = payload.get("data") or {}
        key = data.get("key") or {}
        remote_jid = key.get("remoteJid") or data.get("remoteJid")
        remote_alt = key.get("remoteJidAlt") or data.get("remoteJidAlt")
        external_contact = remote_alt or remote_jid
        message = data.get("message") or {}
        media, caption = self._extract_media(message)
        if media:
            # The message key is the fetch handle; store it alongside the
            # descriptor so the worker needs nothing else from the raw event.
            media["fetch_ref"]["message_key"] = key
        text = (
            message.get("conversation")
            or (message.get("extendedTextMessage") or {}).get("text")
            or data.get("text")
            # A captioned photo carries its text on the attachment itself.
            or caption
            or ""
        )
        return [{
            "event_type": event,
            "instance": payload.get("instance") or data.get("instance"),
            "external_message_id": key.get("id") or data.get("messageId"),
            "external_contact_id": external_contact,
            "remote_jid": remote_jid,
            "remote_jid_alt": remote_alt,
            "from_me": bool(key.get("fromMe") or data.get("fromMe")),
            "status": data.get("status") or data.get("update", {}).get("status"),
            "text": text,
            "media": media,
            "raw": data,
        }]

