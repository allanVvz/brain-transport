from __future__ import annotations

import json
from typing import Any

import httpx

from services import secret_store
from services.whatsapp_providers import media as _media


def _extract_meta_error_detail(response: httpx.Response) -> str:
    """Pull Graph API's own error message out of a failed response.

    Meta returns a structured body (`{"error": {"code", "message", ...}}`)
    that explains exactly why a send was rejected (unapproved template,
    parameter mismatch, invalid recipient, ...). Without this, only the
    generic 'N Bad Request' status survives into our logs/dashboard,
    leaving every failure equally undiagnosable.
    """
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500]
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return json.dumps(payload)[:500]
    parts = [
        f"code={error.get('code')}",
        f"subcode={error.get('error_subcode')}",
        f"message={error.get('message')}",
    ]
    if error.get("error_data"):
        parts.append(f"details={error.get('error_data')}")
    return "; ".join(str(part) for part in parts)[:500]


def _raise_for_status_with_detail(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = _extract_meta_error_detail(response)
        raise httpx.HTTPStatusError(
            f"{exc} | meta_error: {detail}" if detail else str(exc),
            request=exc.request,
            response=exc.response,
        ) from exc


def _credential(binding: dict[str, Any]) -> tuple[str, str]:
    """Decrypt a binding's Meta credential into ``(token, api_version)``.

    The credential is stored either as a JSON document or, for older
    bindings, as the bare access token.
    """
    secret = secret_store.decrypt_secret(binding.get("provider_secret_ciphertext"))
    if not secret:
        raise RuntimeError("Meta binding has no valid credential")
    try:
        credential = json.loads(secret)
        token = str(credential.get("access_token") or "")
        api_version = str(credential.get("api_version") or "v21.0")
    except json.JSONDecodeError:
        token, api_version = secret, "v21.0"
    if not token:
        raise RuntimeError("Meta binding credential has no access token")
    return token, api_version


class MetaWhatsAppProvider:
    """Binding-owned Meta Cloud transport."""

    def send_text(self, binding: dict[str, Any], recipient: str, text: str) -> dict[str, Any]:
        if not binding.get("whatsapp_phone_number_id"):
            raise RuntimeError("Meta binding has no phone number id")
        token, api_version = _credential(binding)
        response = httpx.post(
            f"https://graph.facebook.com/{api_version}/{binding['whatsapp_phone_number_id']}/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"messaging_product": "whatsapp", "to": recipient, "type": "text",
                  "text": {"body": text}}, timeout=30.0,
        )
        _raise_for_status_with_detail(response)
        return response.json()

    def send_template(
        self,
        binding: dict[str, Any],
        recipient: str,
        *,
        template_name: str,
        template_language: str,
        components: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not binding.get("whatsapp_phone_number_id"):
            raise RuntimeError("Meta binding has no phone number id")
        token, api_version = _credential(binding)
        template: dict[str, Any] = {"name": template_name, "language": {"code": template_language}}
        if components:
            template["components"] = components
        response = httpx.post(
            f"https://graph.facebook.com/{api_version}/{binding['whatsapp_phone_number_id']}/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"messaging_product": "whatsapp", "to": recipient, "type": "template",
                  "template": template}, timeout=30.0,
        )
        _raise_for_status_with_detail(response)
        return response.json()

    def provision_instance(self, *_args, **_kwargs):
        raise NotImplementedError("Meta provisioning is managed by the existing integration flow")

    def normalize_webhook(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [{"event_type": "META_RAW", "raw": payload}]

    # Meta message types that carry an attachment. Unlike Baileys, the payload
    # holds only an opaque media id â€” the bytes need two authenticated calls,
    # which is why the webhook records the id and the worker does the fetching.
    _MEDIA_TYPES = ("image", "audio", "video", "document", "sticker")

    @classmethod
    def describe_media(cls, message: dict[str, Any]) -> dict[str, Any] | None:
        """Build the canonical media descriptor from a Meta inbound message.

        Called by the ``/webhooks/whatsapp`` route, which parses Meta's
        envelope itself rather than going through ``normalize_webhook``.
        """
        kind_hint = str(message.get("type") or "").strip().lower()
        if kind_hint not in cls._MEDIA_TYPES:
            return None
        node = message.get(kind_hint)
        if not isinstance(node, dict):
            return None
        media_id = node.get("id")
        if not media_id:
            return None
        mime = node.get("mime_type")
        return _media.build_descriptor(
            provider="meta_cloud",
            kind=_media.kind_for_mime(mime, kind_hint),
            mime=mime,
            filename=node.get("filename"),
            caption=node.get("caption"),
            # Meta reports the size only on documents; audio/image omit it.
            size=node.get("file_size"),
            voice_note=bool(node.get("voice")),
            fetch_ref={"strategy": "meta_graph", "media_id": str(media_id)},
        )

    def get_media_bytes(self, binding: dict[str, Any], media_id: str) -> bytes:
        """Download a received attachment from the Graph API.

        Two steps: resolve the media id to a short-lived CDN URL, then fetch
        that URL. The second call still needs the bearer token â€” the URL is
        scoped, not public.
        """
        token, api_version = _credential(binding)
        headers = {"Authorization": f"Bearer {token}"}
        lookup = httpx.get(
            f"https://graph.facebook.com/{api_version}/{media_id}",
            headers=headers,
            timeout=30.0,
        )
        _raise_for_status_with_detail(lookup)
        url = (lookup.json() or {}).get("url")
        if not url:
            raise RuntimeError(f"Meta returned no download URL for media {media_id}")
        download = httpx.get(url, headers=headers, timeout=60.0, follow_redirects=True)
        _raise_for_status_with_detail(download)
        return download.content

    # Meta Cloud has no equivalent of these Evolution/Baileys instance
    # concepts (QR pairing, restart, logout of a local session). Explicit
    # stubs so a generic `binding.provider`-dispatched call fails with a
    # clear message instead of a bare AttributeError.
    def get_connection_status(self, binding: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Meta Cloud bindings have no connection-status polling; use webhook status callbacks")

    def get_qr_code(self, binding: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Meta Cloud bindings are not paired via QR code")

    def restart_instance(self, binding: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Meta Cloud bindings have no local instance to restart")

    def logout_instance(self, binding: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Meta Cloud bindings have no local instance to log out")

    def send_media(self, binding: dict[str, Any], recipient: str, media: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Meta Cloud media send is not implemented yet")

