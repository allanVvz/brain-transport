"""Canonical WhatsApp outbox creation, always bound to the lead channel."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import os
from urllib.parse import urlparse

from fastapi import HTTPException

from services import event_emitter, supabase_client

# Fallback when a binding does not set metadata.duplicate_guard_window_seconds.
# Any workflow_bindings row can override or disable this per persona/channel
# without a code change (see _observe_duplicate_content).
DEFAULT_DUPLICATE_GUARD_WINDOW_SECONDS = 300


def _recipient_for_lead(lead: dict[str, Any]) -> str:
    """Return the canonical WhatsApp recipient or fail before queueing.

    A manual send must never create an ambiguous outbox item which a worker
    could later route to a stale JID or another channel.
    """
    import re

    identities = (lead.get("metadata") or {}).get("identities") or {}
    # Inbound webhooks always populate a canonical remote JID/external id.
    # Manually imported/operator-created leads may only have `telefone`, so
    # use that as a safe fallback after normalising it below.  The old code
    # rejected those leads even when the phone was valid and the binding was
    # healthy, producing a misleading "recipient unavailable" 409.
    recipient = str(
        identities.get("remote_jid_alt")
        or lead.get("external_contact_id")
        or lead.get("telefone")
        or ""
    )
    if recipient.endswith("@s.whatsapp.net"):
        recipient = recipient.split("@", 1)[0]
    if not recipient or "@lid" in recipient:
        raise HTTPException(409, "Destinatario WhatsApp ausente ou invalido.")
    recipient = re.sub(r"\D", "", recipient)
    # E.164 is at most 15 digits.  The lower bound intentionally accepts
    # national test numbers while still rejecting ids and empty placeholders.
    if not 8 <= len(recipient) <= 15:
        raise HTTPException(409, "Destinatario WhatsApp ausente ou invalido.")
    return recipient


def validate_direct_binding(binding: dict[str, Any]) -> None:
    """Validate the provider-direct contract shared by every persona."""
    metadata = binding.get("metadata") or {}
    decision_owner = metadata.get("decision_owner")
    if decision_owner not in {"deterministic", "n8n_agents"}:
        raise HTTPException(409, "Dono da decisao de mensageria invalido.")
    if metadata.get("transport_mode") != "provider_direct":
        raise HTTPException(409, "A mensageria deve usar transporte direto pelo provider.")
    if (
        metadata.get("outbound_webhook_url")
        or metadata.get("n8n_outbound_webhook_url")
    ):
        raise HTTPException(
            409,
            "Webhooks n8n de saida nao sao permitidos no transporte direto.",
        )
    conversation_url = str(metadata.get("conversation_webhook_url") or "").strip()
    if decision_owner == "deterministic" and (
        binding.get("n8n_workflow_id") or conversation_url
    ):
        raise HTTPException(
            409,
            "Webhooks n8n nao sao permitidos no binding deterministico.",
        )
    if decision_owner == "n8n_agents":
        expected_base = str(os.environ.get("N8N_BASE_URL") or "").rstrip("/")
        parsed = urlparse(conversation_url)
        if (
            not binding.get("n8n_workflow_id")
            or not conversation_url
            or not expected_base
            or not conversation_url.startswith(f"{expected_base}/webhook/")
            or parsed.scheme not in {"http", "https"}
        ):
            raise HTTPException(409, "Workflow conversacional n8n invalido.")

    provider = binding.get("provider")
    status = str(binding.get("connection_status") or "").lower()
    if provider == "meta_cloud":
        if not binding.get("whatsapp_phone_number_id"):
            raise HTTPException(409, "Mensageria Meta sem whatsapp_phone_number_id.")
        if not binding.get("provider_secret_ciphertext"):
            raise HTTPException(409, "Mensageria Meta sem credencial.")
        if status not in {"connected", "open"}:
            raise HTTPException(409, "Mensageria Meta nao esta conectada.")
        return
    if provider == "evolution_baileys":
        if not binding.get("provider_instance_key"):
            raise HTTPException(409, "Mensageria Evolution sem instancia.")
        if not binding.get("provider_secret_ciphertext"):
            raise HTTPException(409, "Mensageria Evolution sem credencial.")
        if status not in {"connected", "open"}:
            raise HTTPException(409, "Mensageria Evolution ainda aguarda conexao ou QR Code.")
        return
    raise HTTPException(409, "Provider de mensageria nao suportado.")


def resolve_lead_binding(lead: dict[str, Any]) -> dict[str, Any]:
    binding_id = lead.get("channel_binding_id")
    binding = (
        supabase_client.get_workflow_binding_by_id(binding_id)
        if binding_id
        else None
    )
    if binding and binding.get("persona_id") != lead.get("persona_id"):
        raise HTTPException(403, "O canal selecionado pertence a outra persona.")
    if not binding or not binding.get("active"):
        binding = supabase_client.get_active_whatsapp_binding(lead.get("persona_id"))
        if not binding:
            raise HTTPException(409, "Mensageria da persona nao configurada.")
        if lead.get("id"):
            supabase_client.update_lead(
                int(lead["id"]),
                {"channel_binding_id": binding["id"]},
            )
        lead["channel_binding_id"] = binding["id"]
    metadata = binding.get("metadata") or {}
    if metadata.get("safety_paused") or binding.get("connection_status") == "safety_paused":
        raise HTTPException(409, "O canal esta pausado por seguranca.")
    validate_direct_binding(binding)
    return binding


def _observe_duplicate_content(
    *, lead: dict[str, Any], binding: dict[str, Any], text: str, correlation_id: str,
) -> dict[str, Any] | None:
    """Record repeated copy without turning content into message identity.

    Row-identity idempotency (idempotency_key/correlation_id, checked by the
    caller before this runs) only catches a literal re-dispatch of the same
    outbox row. It does not catch an operator or agent typing the same
    answer again as a brand-new send because delivery looked ambiguous
    (missing ACK from the provider) — that produces a second, distinct row
    that sails through as a distinct turn. This observer records that quality
    signal generically; canonical identity, not text, decides idempotency.

    A content duplicate is not a binding-level safety anomaly, so callers
    must not treat it as one: it must never pause the lead or sweep sibling
    buffer rows (confirmed live 2026-08-10 — a legitimate new turn that
    happened to generate the same reply text tripped this guard, which used
    to call record_whatsapp_safety_violation and cascaded into hours of
    silence for the whole backlog). The caller still enqueues the distinct
    outbound after emitting this warning.
    """
    metadata = binding.get("metadata") or {}
    if metadata.get("duplicate_guard_enabled") is False:
        return None
    window_seconds = metadata.get("duplicate_guard_window_seconds")
    if not isinstance(window_seconds, (int, float)) or window_seconds <= 0:
        window_seconds = DEFAULT_DUPLICATE_GUARD_WINDOW_SECONDS
    normalized = supabase_client.normalize_whatsapp_text(text)
    if not normalized:
        return None
    duplicate = supabase_client.find_recent_duplicate_whatsapp_outbound(
        lead_ref=lead["id"],
        channel_binding_id=binding["id"],
        normalized_text=normalized,
        window_seconds=int(window_seconds),
    )
    if not duplicate:
        return None
    event_emitter.emit(
        "whatsapp.duplicate_content_suppressed",
        entity_type="lead",
        entity_id=str(lead.get("id") or ""),
        persona_id=lead.get("persona_id"),
        payload={
            "correlation_id": correlation_id,
            "duplicate_buffer_id": duplicate.get("id"),
            "duplicate_status": duplicate.get("status"),
            "window_seconds": int(window_seconds),
        },
        level="warning",
        source="services.whatsapp_outbox",
    )
    return duplicate


def prepare_outbound_envelope(
    *, lead: dict[str, Any], text: str, sender_type: str,
    message_id: str, correlation_id: str, idempotency_key: str | None = None,
    initial_status: str = "pending_send", metadata: dict[str, Any] | None = None,
    media: dict[str, Any] | None = None, template: dict[str, Any] | None = None,
    campaign_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate routing and build the canonical DB envelope without writing it."""
    if initial_status not in {"pending_send", "awaiting_proof"}:
        raise ValueError("invalid outbound initial status")
    binding = resolve_lead_binding(lead)
    _recipient_for_lead(lead)
    lock_key = idempotency_key or correlation_id
    _observe_duplicate_content(
        lead=lead, binding=binding, text=text, correlation_id=correlation_id,
    )
    scope_fields = {
        "message_origin": "campaign",
        "campaign_id": campaign_scope.get("campaign_id"),
        "campaign_revision": campaign_scope.get("campaign_revision"),
        "campaign_recipient_id": campaign_scope.get("campaign_recipient_id"),
        "policy_checksum": campaign_scope.get("policy_checksum"),
    } if campaign_scope else {}
    return {
        "binding": binding,
        "lock_key": lock_key,
        "buffer": {
            "persona_id": lead["persona_id"],
            "lead_ref": lead["id"],
            "channel_binding_id": binding["id"],
            "whatsapp_phone_number_id": binding.get("whatsapp_phone_number_id"),
            "direction": "outbound",
            "payload": {"text": text, "sender_type": sender_type, "media": media, "template": template},
            "status": initial_status,
            "batch_key": f"{lead['persona_id']}:{lead['id']}",
            "idempotency_key": lock_key,
            "correlation_id": correlation_id,
            **scope_fields,
            **({"campaign_step": campaign_scope.get("campaign_step")} if campaign_scope else {}),
        },
        "message": {
            "lead_id": lead["id"],
            "role": "human" if sender_type == "human" else "assistant",
            "content": text, "direction": "outbound", "status": "pending",
            "channel": "whatsapp", "sender_id": message_id,
            "whatsapp_phone_number_id": binding.get("whatsapp_phone_number_id"),
            "channel_binding_id": binding["id"], "correlation_id": correlation_id,
            "metadata": metadata or {}, "created_at": datetime.now(timezone.utc).isoformat(),
            **scope_fields,
        },
    }


def enqueue_outbound(*, lead: dict[str, Any], text: str, sender_type: str,
                     message_id: str, correlation_id: str,
                     idempotency_key: str | None = None,
                     initial_status: str = "pending_send",
                     metadata: dict[str, Any] | None = None,
                     media: dict[str, Any] | None = None,
                     template: dict[str, Any] | None = None,
                     campaign_scope: dict[str, Any] | None = None) -> dict[str, Any]:
    """Queue one outbound WhatsApp send.

    `campaign_scope` (campaign_id/campaign_revision/campaign_recipient_id/
    campaign_step/policy_checksum) is optional and additive: when absent this
    behaves exactly as the ordinary 1:1 conversation path (message_origin
    stays 'conversation'). When present it tags the row message_origin
    'campaign' so the dispatch worker and campaign audit views can attribute
    it back to a specific campaign send.
    """
    prepared = prepare_outbound_envelope(
        lead=lead, text=text, sender_type=sender_type, message_id=message_id,
        correlation_id=correlation_id, idempotency_key=idempotency_key,
        initial_status=initial_status, metadata=metadata, media=media,
        template=template, campaign_scope=campaign_scope,
    )
    binding = prepared["binding"]
    lock_key = idempotency_key or correlation_id
    existing = supabase_client.get_whatsapp_buffer_by_idempotency(lock_key)
    if existing:
        if (
            existing.get("lead_ref") != lead["id"]
            or existing.get("channel_binding_id") != binding["id"]
        ):
            raise HTTPException(
                409,
                "A chave idempotente ja pertence a outra mensagem.",
            )
        return {
            "buffer_id": existing["id"],
            "message_id": message_id,
            "status": existing.get("status") or "pending_send",
            "deduplicated": True,
            "binding": binding,
        }
    envelope = supabase_client.enqueue_whatsapp_envelope(
        buffer=prepared["buffer"], message=prepared["message"],
    )
    if envelope.get("deduplicated"):
        existing = supabase_client.get_whatsapp_buffer_by_idempotency(lock_key)
        if (
            not existing
            or existing.get("lead_ref") != lead["id"]
            or existing.get("channel_binding_id") != binding["id"]
        ):
            raise HTTPException(
                409,
                "A chave idempotente ja pertence a outra mensagem.",
            )
    return {**envelope, "binding": binding}
