"""Durable WhatsApp inbox/outbox dispatcher.

Meta and the dashboard only write durable rows.  This worker leases rows from
Postgres, calls Brain for inbound decisions and n8n only for transport.  It is
safe to run more than one instance because leasing uses SKIP LOCKED.
"""
from __future__ import annotations

import json
import os
import re
import socket
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from services import (
    conversation_runtime,
    event_emitter,
    n8n_client,
    supabase_client,
    sre_logger,
    whatsapp_outbox,
)
from services.whatsapp_providers import get_provider
from workers.base_worker import BaseWorker


def _retry_delay(attempt: int) -> int:
    """Bounded exponential backoff in seconds (5, 10, ... up to 5 minutes)."""
    return min(300, 5 * (2 ** max(0, attempt - 1)))


# Below this length an exact text match against a recent outbound message is
# not a reliable echo signal â€” short greetings ("oi", "ok", "sim") are
# equally likely to be typed by a real customer replying in kind as to be a
# genuine WhatsApp-side echo of the bot's own message. Confirmed live
# 2026-08-04: a customer replying "oi" (matching the bot's own earlier
# greeting) got permanently handed off by this guard on every message.
_ECHO_GUARD_MIN_LENGTH = 12


def _queue_wait_ms(row: dict[str, Any]) -> int | None:
    try:
        created = datetime.fromisoformat(str(row.get("created_at") or "").replace("Z", "+00:00"))
        return max(0, int((datetime.now(timezone.utc) - created).total_seconds() * 1000))
    except (TypeError, ValueError):
        return None


class WhatsAppDispatchWorker(BaseWorker):
    name = "WhatsAppDispatchWorker"
    interval = int(os.environ.get("WHATSAPP_DISPATCH_INTERVAL", "2"))

    def __init__(self) -> None:
        super().__init__()
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}"
        self.concurrency = max(
            1, int(os.environ.get("WHATSAPP_DISPATCH_CONCURRENCY", "4")),
        )

    def _run_cycle(self) -> None:
        rows = supabase_client.claim_whatsapp_buffer(self.worker_id)
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            group_key = str(
                row.get("batch_key")
                or (f"lead:{row['lead_ref']}" if row.get("lead_ref") is not None else row["id"])
            )
            groups[group_key].append(row)
        ordered_groups = [
            sorted(group, key=lambda row: (str(row.get("created_at") or ""), str(row["id"])))
            for group in groups.values()
        ]
        if len(ordered_groups) <= 1 or self.concurrency == 1:
            for group in ordered_groups:
                self._dispatch_group(group)
            return
        with ThreadPoolExecutor(
            max_workers=min(self.concurrency, len(ordered_groups)),
            thread_name_prefix="whatsapp-dispatch",
        ) as executor:
            list(executor.map(self._dispatch_group, ordered_groups))

    def _dispatch_group(self, rows: list[dict[str, Any]]) -> None:
        """Keep one lead FIFO while allowing unrelated leads in parallel."""
        for row in rows:
            try:
                if row.get("direction") == "inbound":
                    self._dispatch_inbound(row)
                else:
                    self._dispatch_outbound(row)
            except Exception as exc:  # one poison message must not block the queue
                self._retry_or_dead_letter(row, exc)

    def _begin_attempt(self, row: dict[str, Any], kind: str) -> bool:
        if supabase_client.mark_whatsapp_attempt(
            row["id"],
            self.worker_id,
            kind,
        ):
            row["_attempt_started"] = kind
            return True
        reason = f"compromised {kind} re-execution"
        if row.get("channel_binding_id"):
            supabase_client.record_whatsapp_safety_violation(
                binding_id=row["channel_binding_id"],
                lead_ref=row.get("lead_ref"),
                violation_key=f"reexecution:{kind}:{row['id']}",
                reason=reason,
            )
        supabase_client.complete_whatsapp_buffer(
            row["id"],
            "waiting_human",
            error=reason,
        )
        return False

    def _dispatch_inbound(self, row: dict[str, Any]) -> None:
        payload = row.get("payload") or {}
        persona = supabase_client.get_persona_by_id(row["persona_id"])
        if not persona:
            raise RuntimeError("persona for inbound buffer no longer exists")
        lead = supabase_client.get_lead_by_ref(row.get("lead_ref")) if row.get("lead_ref") else {}
        inbound_text = str(payload.get("text") or "").strip()
        if inbound_text and len(inbound_text) >= _ECHO_GUARD_MIN_LENGTH and row.get("lead_ref"):
            recent_messages = supabase_client.get_messages(str(row["lead_ref"]), limit=20) or []
            recent_outbound_echo = next(
                (
                    message
                    for message in reversed(recent_messages)
                    if message.get("direction") == "outbound"
                    and str(message.get("content") or message.get("texto") or "").strip() == inbound_text
                ),
                None,
            )
            if recent_outbound_echo:
                supabase_client.handoff_whatsapp_lead(int(row["lead_ref"]))
                supabase_client.complete_whatsapp_buffer(row["id"], "waiting_human", error="recent outbound echo")
                event_emitter.emit(
                    "whatsapp.bot_loop_suppressed",
                    entity_type="lead",
                    entity_id=str(row["lead_ref"]),
                    persona_id=row["persona_id"],
                    payload={
                        "correlation_id": row.get("correlation_id"),
                        "matched_outbound_id": recent_outbound_echo.get("id"),
                    },
                    source="workers.whatsapp",
                )
                return
        # Per-lead handoff is the only pause authority. A former persona-wide
        # ``automation_mode`` gate could leave the eyebrow toggle showing
        # "IA ativa" while silently parking every inbound in waiting_human.
        if (lead or {}).get("ai_paused"):
            supabase_client.complete_whatsapp_buffer(row["id"], "waiting_human")
            event_emitter.emit(
                "whatsapp.inbound_waiting_human",
                entity_type="lead",
                entity_id=str(row.get("lead_ref") or ""),
                persona_id=row["persona_id"],
                payload={
                    "correlation_id": row.get("correlation_id"),
                    "ai_paused": bool((lead or {}).get("ai_paused")),
                },
                source="workers.whatsapp",
            )
            return
        binding = supabase_client.get_workflow_binding_by_id(row.get("channel_binding_id"))
        binding_metadata = (binding or {}).get("metadata") or {}
        if (
            not binding
            or not binding.get("active")
            or binding.get("persona_id") != row.get("persona_id")
        ):
            supabase_client.complete_whatsapp_buffer(
                row["id"],
                "waiting_human",
                error="inbound binding is missing, inactive or cross-persona",
            )
            return
        if (
            binding_metadata.get("safety_paused")
            or binding.get("connection_status") == "safety_paused"
        ):
            if row.get("lead_ref"):
                supabase_client.handoff_whatsapp_lead(int(row["lead_ref"]))
            supabase_client.complete_whatsapp_buffer(
                row["id"],
                "waiting_human",
                error="binding safety paused",
            )
            return
        decision_owner = binding_metadata.get("decision_owner")
        if decision_owner not in {"deterministic", "n8n_agents"}:
            supabase_client.complete_whatsapp_buffer(
                row["id"],
                "waiting_human",
                error="unsupported binding decision owner",
            )
            return
        if not self._begin_attempt(row, "decision"):
            return
        if decision_owner == "deterministic":
            result = conversation_runtime.execute_pipeline(
                persona_slug=persona["slug"],
                lead_ref=int(row["lead_ref"]),
                message=str(payload.get("text") or ""),
                message_id=row.get("external_message_id"),
                correlation_id=str(row.get("correlation_id") or row["id"]),
                phone_number_id=row.get("whatsapp_phone_number_id"),
                channel_binding_id=row.get("channel_binding_id"),
                inbound_buffer_id=row["id"],
            )
            if result.get("burst_superseded"):
                event_emitter.emit(
                    "whatsapp.inbound_burst_superseded",
                    entity_type="lead", entity_id=str(row.get("lead_ref") or ""),
                    persona_id=row["persona_id"],
                    payload={"correlation_id": row.get("correlation_id"), "buffer_id": row["id"]},
                    source="workers.whatsapp",
                )
                return
            if not result.get("handoff"):
                supabase_client.complete_whatsapp_buffer(row["id"], "sent")
            event_emitter.emit(
                "whatsapp.inbound_dispatched",
                entity_type="lead",
                entity_id=str(row.get("lead_ref") or ""),
                persona_id=row["persona_id"],
                payload={
                    "correlation_id": row.get("correlation_id"),
                    "queue_wait_ms": _queue_wait_ms(row),
                    "conversation_mode": "deterministic",
                    "pipeline_contract": binding_metadata.get("pipeline_contract") or "conversation_v1",
                    "classifier": result.get("classifier"),
                    "route": result.get("route"),
                    "handoff": bool(result.get("handoff")),
                },
                source="workers.whatsapp",
            )
            return
        if decision_owner == "n8n_agents":
            webhook_url = str(
                binding_metadata.get("conversation_webhook_url") or ""
            ).strip()
            if not webhook_url:
                raise RuntimeError("n8n_agents conversation webhook is not configured")
            token = (os.environ.get("AI_BRAIN_WEBHOOK_TOKEN") or "").strip()
            status, body = n8n_client.send_to_webhook(
                webhook_url,
                {
                    "buffer_id": row["id"],
                    "lead_ref": row.get("lead_ref"),
                    "persona_id": row["persona_id"],
                    "persona_slug": persona["slug"],
                    "phone_number_id": row["whatsapp_phone_number_id"],
                    "channel_binding_id": row.get("channel_binding_id"),
                    "external_message_id": row.get("external_message_id"),
                    "correlation_id": row.get("correlation_id"),
                    "message": payload.get("text") or "",
                    "pipeline_contract": binding_metadata.get("pipeline_contract") or "conversation_v3",
                    "decision_owner": "n8n_agents",
                },
                secret=token or None,
                timeout=45.0,
                # Defense-in-depth only -- the real fix is that /internal/
                # conversations/commit now returns a small, trimmed envelope
                # (see conversation_runtime.dispatch_result_envelope), so
                # this limit should never be the thing standing between a
                # successful commit and a readable response. Kept well above
                # the trimmed envelope's realistic size, not raised to
                # "big enough to swallow the old bug" (confirmed live
                # 2026-08-10: the untrimmed response measured 80438 bytes).
                response_limit=262_144,
            )
            if not 200 <= status < 300:
                raise RuntimeError(f"n8n conversation returned HTTP {status}")
            try:
                n8n_result = json.loads(body)
            except Exception:
                n8n_result = None
            if not isinstance(n8n_result, dict) or not any(
                key in n8n_result
                for key in ("ok", "technical_failure", "handoff", "message_id")
            ):
                raise RuntimeError(
                    "n8n conversation returned an invalid result contract"
                )
            if n8n_result.get("technical_failure"):
                event_emitter.emit(
                    "whatsapp.inbound_decision_failed",
                    entity_type="lead",
                    entity_id=str(row.get("lead_ref") or ""),
                    persona_id=row["persona_id"],
                    payload={
                        "correlation_id": row.get("correlation_id"),
                        "buffer_id": row.get("id"),
                        "pipeline_contract": binding_metadata.get("pipeline_contract"),
                    },
                    level="error",
                    source="workers.whatsapp",
                )
                return
            if n8n_result.get("burst_superseded"):
                event_emitter.emit(
                    "whatsapp.inbound_burst_superseded",
                    entity_type="lead", entity_id=str(row.get("lead_ref") or ""),
                    persona_id=row["persona_id"],
                    payload={"correlation_id": row.get("correlation_id"), "buffer_id": row["id"]},
                    source="workers.whatsapp",
                )
                return
            if not n8n_result.get("handoff"):
                supabase_client.complete_whatsapp_buffer(row["id"], "sent")
            event_emitter.emit(
                "whatsapp.inbound_dispatched",
                entity_type="lead",
                entity_id=str(row.get("lead_ref") or ""),
                persona_id=row["persona_id"],
                payload={
                    "correlation_id": row.get("correlation_id"),
                    "queue_wait_ms": _queue_wait_ms(row),
                    "conversation_mode": "n8n_agents",
                    "pipeline_contract": binding_metadata.get("pipeline_contract") or "conversation_v3",
                    "handoff": bool(n8n_result.get("handoff")),
                },
                source="workers.whatsapp",
            )
            return

    def _dispatch_outbound(self, row: dict[str, Any]) -> None:
        binding = supabase_client.get_workflow_binding_by_id(row.get("channel_binding_id"))
        if (
            not binding
            or not binding.get("active")
            or binding.get("persona_id") != row.get("persona_id")
        ):
            supabase_client.complete_whatsapp_buffer(
                row["id"],
                "waiting_human",
                error="active binding does not match outbound persona",
            )
            return
        metadata = binding.get("metadata") or {}
        if (
            metadata.get("safety_paused")
            or binding.get("connection_status") == "safety_paused"
        ):
            supabase_client.complete_whatsapp_buffer(
                row["id"],
                "waiting_human",
                error="binding safety paused",
            )
            return
        try:
            whatsapp_outbox.validate_direct_binding(binding)
        except Exception as exc:
            detail = getattr(exc, "detail", None) or str(exc)
            supabase_client.complete_whatsapp_buffer(
                row["id"],
                "waiting_human",
                error=str(detail),
            )
            return
        transport_mode = "provider_direct"

        correlation_id = str(row.get("correlation_id") or "")
        if not correlation_id:
            supabase_client.complete_whatsapp_buffer(
                row["id"],
                "waiting_human",
                error="outbound correlation id is missing",
            )
            return
        lead = (
            supabase_client.get_lead_by_ref(row.get("lead_ref"))
            if row.get("lead_ref")
            else {}
        )
        identities = ((lead or {}).get("metadata") or {}).get("identities") or {}
        recipient = str(
            identities.get("remote_jid_alt")
            or (lead or {}).get("external_contact_id")
            or (lead or {}).get("telefone")
            or ""
        )
        if recipient.endswith("@s.whatsapp.net"):
            recipient = recipient.split("@", 1)[0]
        if not recipient or "@lid" in recipient:
            supabase_client.complete_whatsapp_buffer(
                row["id"],
                "waiting_human",
                error="recipient identification unavailable",
            )
            return
        recipient = re.sub(r"\D", "", recipient)
        if not recipient or not 8 <= len(recipient) <= 15:
            supabase_client.complete_whatsapp_buffer(
                row["id"],
                "waiting_human",
                error="recipient identification unavailable",
            )
            return

        if transport_mode == "provider_direct":
            outbound_payload = row.get("payload") or {}
            provider = get_provider(binding.get("provider"))
            if outbound_payload.get("media") and binding.get("provider") == "evolution_baileys":
                media = outbound_payload["media"]
                signed_url = supabase_client._storage_signed_url(
                    media.get("bucket"), media.get("path"), 3600
                )
                if not signed_url:
                    raise RuntimeError("could not create signed media URL")
                if not self._begin_attempt(row, "provider"):
                    return
                result = provider.send_media(binding, recipient, {
                    "mediatype": (
                        "image" if str(media.get("mime") or "").startswith("image/")
                        else "audio" if str(media.get("mime") or "").startswith("audio/")
                        else "document"
                    ),
                    "mimetype": media.get("mime"),
                    "media": signed_url,
                    "fileName": media.get("filename"),
                    "caption": str(outbound_payload.get("text") or ""),
                })
            elif outbound_payload.get("template") and binding.get("provider") == "meta_cloud":
                template = outbound_payload["template"] or {}
                if not self._begin_attempt(row, "provider"):
                    return
                result = provider.send_template(
                    binding,
                    recipient,
                    template_name=str(template.get("name") or ""),
                    template_language=str(template.get("language") or "pt_BR"),
                    components=template.get("components") or [],
                )
            else:
                if outbound_payload.get("media"):
                    supabase_client.complete_whatsapp_buffer(
                        row["id"],
                        "waiting_human",
                        error="binding provider does not support canonical media send",
                    )
                    return
                if not self._begin_attempt(row, "provider"):
                    return
                result = provider.send_text(
                    binding,
                    recipient,
                    str(outbound_payload.get("text") or ""),
                )
            external_id = (
                (result.get("key") or {}).get("id")
                or ((result.get("messages") or [{}])[0] or {}).get("id")
                or result.get("messageId")
                or result.get("id")
            )
            if not external_id:
                raise RuntimeError("provider returned no external message id")
            supabase_client.complete_whatsapp_outbound(
                row["id"],
                binding_id=binding["id"],
                correlation_id=correlation_id,
                wamid=str(external_id),
                success=True,
            )
            event_emitter.emit(
                "whatsapp.outbound_accepted", entity_type="lead",
                entity_id=str(row.get("lead_ref") or ""), persona_id=row["persona_id"],
                payload={"buffer_id": row["id"], "correlation_id": correlation_id,
                         "provider": binding.get("provider"), "external_message_id": str(external_id)},
                source="workers.whatsapp",
            )
            return

        webhook_url = str(metadata.get("n8n_outbound_webhook_url") or "").strip()
        if not webhook_url:
            supabase_client.complete_whatsapp_buffer(
                row["id"],
                "waiting_human",
                error="outbound n8n webhook is not configured",
            )
            return
        if not self._begin_attempt(row, "provider"):
            return
        status, _ = n8n_client.send_to_webhook(webhook_url, {
            "buffer_id": row["id"], "lead_ref": row.get("lead_ref"), "persona_id": row["persona_id"],
            "phone_number_id": binding.get("whatsapp_phone_number_id"), "channel_binding_id": binding["id"], "to": recipient,
            "text": (row.get("payload") or {}).get("text", ""),
            "correlation_id": correlation_id,
        }, secret=(os.environ.get("AI_BRAIN_WEBHOOK_TOKEN") or None))
        if not 200 <= status < 300:
            raise RuntimeError(f"n8n outbound returned HTTP {status}")
        # The controlled adapter commits the external id through
        # /internal/whatsapp/outbound-result before responding.  If it did not,
        # this processing row expires to waiting_human and is never resent.

    def _retry_or_dead_letter(self, row: dict[str, Any], exc: Exception) -> None:
        attempts, maximum = int(row.get("attempt_count") or 1), int(row.get("max_attempts") or 5)
        error = f"{type(exc).__name__}: {str(exc)[:800]}"
        attempt_kind = row.get("_attempt_started")
        if attempt_kind == "decision" and row.get("direction") == "inbound":
            try:
                reconciliation = supabase_client.reconcile_committed_graph_inbound(
                    str(row["id"]), error
                )
            except Exception:
                reconciliation = {}
            if reconciliation.get("reconciled") or (
                reconciliation.get("ok")
                and reconciliation.get("reason") == "already_reconciled"
            ):
                event_emitter.emit(
                    "whatsapp.inbound_commit_reconciled",
                    entity_type="lead",
                    entity_id=str(row.get("lead_ref") or ""),
                    persona_id=row.get("persona_id"),
                    payload={
                        "buffer_id": row.get("id"),
                        "correlation_id": row.get("correlation_id"),
                        "outbound_id": reconciliation.get("outbound_id"),
                    },
                    level="warning",
                    source="workers.whatsapp",
                )
                return
        if attempt_kind:
            if row.get("channel_binding_id"):
                # level="partial" only quarantines this row to waiting_human;
                # it leaves sibling inbound rows for the same lead claimable,
                # instead of the implicit level="full" that used to pause
                # the whole lead over one row's exhausted retries.
                supabase_client.record_whatsapp_safety_violation(
                    binding_id=row["channel_binding_id"],
                    lead_ref=row.get("lead_ref"),
                    violation_key=f"attempt-failed:{attempt_kind}:{row['id']}",
                    reason=error,
                    level="partial",
                )
            supabase_client.complete_whatsapp_buffer(row["id"], "waiting_human", error=error)
            return
        if row.get("direction") == "outbound":
            supabase_client.complete_whatsapp_buffer(
                row["id"],
                "waiting_human",
                error=error,
            )
            return
        if attempts >= maximum:
            supabase_client.complete_whatsapp_buffer(row["id"], "dead_letter", error=error)
            sre_logger.error(self.name, f"dead-letter buffer={row['id']}: {error}")
            return
        supabase_client.release_whatsapp_buffer(row["id"], "retry", delay_seconds=_retry_delay(attempts), error=error)

