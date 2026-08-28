"""Campaign delivery domain service.

The service deliberately separates policy resolution, consent resolution and
recipient eligibility. `send_campaign` (delivery two) re-runs the same
eligibility function again, live, immediately before admitting each outbox
row, rather than trusting the frozen preview.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from fastapi import HTTPException

from services import event_emitter, graph_json_v2_store, supabase_client, whatsapp_outbox


DEFAULT_CONTACT_POLICY: dict[str, Any] = {
    "campaign_kind": "consent_request",
    "purpose": "ofertas_e_novidades",
    "max_unanswered_attempts_per_lead": 3,
    "retry_schedule_hours": [48, 72],
    "cooldown_after_limit_days": 30,
    "max_total_sends": 500,
    "max_unique_leads": 400,
    "daily_send_limit": 100,
    "hourly_send_limit": 20,
    "warning_thresholds": [0.70, 0.85, 0.95, 1.0],
    "response_hold_hours": 24,
    "attribution_window_hours": 72,
    "conversion_window_days": 7,
}

_POLICY_INT_FIELDS = {
    "max_unanswered_attempts_per_lead": (1, 10),
    "cooldown_after_limit_days": (1, 365),
    "max_total_sends": (1, 100000),
    "max_unique_leads": (1, 100000),
    "daily_send_limit": (1, 100000),
    "hourly_send_limit": (1, 10000),
    "response_hold_hours": (1, 168),
    "attribution_window_hours": (1, 720),
    "conversion_window_days": (1, 365),
}
_POLICY_FIELDS = set(DEFAULT_CONTACT_POLICY)


def rollout_one_enabled(persona: dict[str, Any] | None = None) -> bool:
    configured = (os.environ.get("BULK_CAMPAIGNS_ROLLOUT1_ENABLED") or "").strip().lower()
    if configured:
        globally_enabled = configured in {"1", "true", "yes", "on"}
    else:
        globally_enabled = (os.environ.get("ENVIRONMENT") or "").strip().lower() in {
            "qa", "test", "preview", "development", "local",
        }
    persona_config = ((persona or {}).get("config") or {}).get("bulk_campaigns") or {}
    return globally_enabled and persona_config.get("enabled", True) is not False


def meta_mock_enabled() -> bool:
    environment = (os.environ.get("ENVIRONMENT") or "").strip().lower()
    configured = (os.environ.get("AI_BRAIN_META_MOCK") or "").strip().lower()
    return environment in {"qa", "test", "preview", "development", "local"} and configured in {
        "1", "true", "yes", "on",
    }


def meta_provider_ready(binding: dict[str, Any] | None) -> bool:
    if meta_mock_enabled():
        return True
    if not binding or binding.get("provider") != "meta_cloud":
        return False
    return bool(
        str(binding.get("connection_status") or "").lower() in {"connected", "open"}
        and binding.get("provider_secret_ciphertext")
        and binding.get("whatsapp_phone_number_id")
    )


def evolution_provider_ready(binding: dict[str, Any] | None) -> bool:
    if not binding or binding.get("provider") != "evolution_baileys":
        return False
    return bool(
        str(binding.get("connection_status") or "").lower() in {"connected", "open"}
        and binding.get("provider_secret_ciphertext")
        and binding.get("provider_instance_key")
    )


def resolve_provider_ready(binding: dict[str, Any] | None, provider: str) -> bool:
    """Dispatches readiness by provider so Evolution is a normal, first-class
    option once its binding is connected â€” not a separate/limited mode."""
    if provider == "meta_cloud":
        return meta_provider_ready(binding)
    if provider == "evolution_baileys":
        return evolution_provider_ready(binding)
    return False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_checksum(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _uuid_or_none(value: str | None) -> str | None:
    try:
        return str(uuid.UUID(str(value))) if value else None
    except (TypeError, ValueError, AttributeError):
        return None


def _slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower()).strip("-")
    return value or "campanha"


def _chunks(values: list[Any], size: int = 500) -> Iterable[list[Any]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


def _rows(query) -> list[dict]:
    return supabase_client._q(query)


def normalize_contact_policy(policy: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(policy or {}) - _POLICY_FIELDS)
    if unknown:
        raise HTTPException(422, f"Campos de politica desconhecidos: {', '.join(unknown)}.")
    normalized = dict(DEFAULT_CONTACT_POLICY)
    normalized.update({key: value for key, value in (policy or {}).items() if value is not None})
    for field, (minimum, maximum) in _POLICY_INT_FIELDS.items():
        try:
            value = int(normalized[field])
        except (TypeError, ValueError):
            value = int(DEFAULT_CONTACT_POLICY[field])
        normalized[field] = min(max(value, minimum), maximum)

    retry_hours: list[int] = []
    for raw in normalized.get("retry_schedule_hours") or []:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if 1 <= value <= 24 * 30 and value not in retry_hours:
            retry_hours.append(value)
    normalized["retry_schedule_hours"] = retry_hours[:9]

    thresholds: list[float] = []
    for raw in normalized.get("warning_thresholds") or []:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if 0 < value <= 1 and value not in thresholds:
            thresholds.append(value)
    normalized["warning_thresholds"] = sorted(thresholds) or list(DEFAULT_CONTACT_POLICY["warning_thresholds"])

    kind = str(normalized.get("campaign_kind") or "consent_request")
    if kind not in {"consent_request", "promotional"}:
        raise HTTPException(422, "Tipo de campanha invalido.")
    normalized["campaign_kind"] = kind
    normalized["purpose"] = str(normalized.get("purpose") or "").strip()
    if not normalized["purpose"]:
        raise HTTPException(422, "A finalidade da campanha e obrigatoria.")
    normalized["max_unique_leads"] = min(
        normalized["max_unique_leads"], normalized["max_total_sends"]
    )
    normalized["hourly_send_limit"] = min(
        normalized["hourly_send_limit"], normalized["daily_send_limit"]
    )
    return normalized


def resolve_contact_policy(
    *,
    persona: dict[str, Any],
    audience: dict[str, Any],
    campaign_overrides: dict[str, Any] | None,
    campaign_kind: str,
    purpose: str,
) -> tuple[dict[str, Any], str]:
    """Resolve default < persona < audience < campaign, then normalizes."""
    persona_policy = ((persona.get("config") or {}).get("contact_policy") or {})
    audience_policy = ((audience.get("metadata") or {}).get("contact_policy") or {})
    resolved = {
        **DEFAULT_CONTACT_POLICY,
        **persona_policy,
        **audience_policy,
        **(campaign_overrides or {}),
        "campaign_kind": campaign_kind,
        "purpose": purpose,
    }
    resolved = normalize_contact_policy(resolved)
    return resolved, _canonical_checksum(resolved)


def _latest_consents(
    *, persona_id: str, lead_ids: list[int], channel: str, purpose: str
) -> dict[int, dict[str, Any]]:
    if not lead_ids:
        return {}
    latest: dict[int, dict[str, Any]] = {}
    client = supabase_client.get_client()
    for group in _chunks(lead_ids):
        rows = _rows(
            client.table("contact_consents")
            .select("*")
            .eq("persona_id", persona_id)
            .eq("channel", channel)
            .eq("purpose", purpose)
            .in_("lead_id", group)
            .order("effective_at", desc=True)
            .order("created_at", desc=True)
            .limit(5000)
        )
        for row in rows:
            lead_id = int(row["lead_id"])
            latest.setdefault(lead_id, row)
    return latest


def resolve_applicable_consent(
    latest: dict[int, dict[str, Any]], lead_id: int
) -> tuple[str, str | None]:
    consent = latest.get(int(lead_id))
    if not consent:
        return "unknown", None
    valid_until = consent.get("valid_until")
    if valid_until and consent.get("status") not in {"refused", "revoked"}:
        try:
            expiry = datetime.fromisoformat(str(valid_until).replace("Z", "+00:00"))
            if expiry < datetime.now(timezone.utc):
                return "unknown", consent.get("id")
        except ValueError:
            return "unknown", consent.get("id")
    return str(consent.get("status") or "unknown"), consent.get("id")


def _semantic_membership_map(persona_id: str, lead_ids: list[int]) -> dict[int, str]:
    if not lead_ids:
        return {}
    client = supabase_client.get_client()
    audiences = _rows(
        client.table("audiences")
        .select("id,metadata,source_type,slug")
        .eq("persona_id", persona_id)
        .limit(1000)
    )
    semantic_ids = {
        row["id"] for row in audiences
        if str((row.get("metadata") or {}).get("kind") or "semantic_group") == "semantic_group"
        and row.get("source_type") != "import"
        and row.get("slug") != "import"
    }
    if not semantic_ids:
        return {}
    result: dict[int, str] = {}
    for group in _chunks(lead_ids):
        memberships = _rows(
            client.table("lead_audience_memberships")
            .select("lead_id,audience_id")
            .in_("lead_id", group)
            .in_("audience_id", list(semantic_ids))
            .limit(5000)
        )
        for row in memberships:
            result[int(row["lead_id"])] = str(row["audience_id"])
    return result


def _campaign_leads(import_batch_ids: list[str]) -> tuple[list[dict[str, Any]], dict[int, set[str]]]:
    client = supabase_client.get_client()
    rows = _rows(
        client.table("lead_import_rows")
        .select("batch_id,lead_id,status")
        .in_("batch_id", import_batch_ids)
        .in_("status", ["created", "updated"])
        .limit(100000)
    )
    batches_by_lead: dict[int, set[str]] = {}
    for row in rows:
        if row.get("lead_id") is None:
            continue
        batches_by_lead.setdefault(int(row["lead_id"]), set()).add(str(row["batch_id"]))
    lead_ids = sorted(batches_by_lead)
    leads: list[dict[str, Any]] = []
    for group in _chunks(lead_ids):
        leads.extend(_rows(client.table("leads").select("*").in_("id", group).limit(len(group))))
    leads.sort(key=lambda row: int(row["id"]))
    return leads, batches_by_lead


def _valid_phone(lead: dict[str, Any]) -> bool:
    phone = re.sub(r"\D", "", str(lead.get("telefone") or lead.get("lead_id") or ""))
    return 8 <= len(phone) <= 15


def evaluate_recipient_eligibility(
    *,
    lead: dict[str, Any],
    selected_audience_id: str,
    semantic_audience_id: str | None,
    campaign_kind: str,
    consent_status: str,
    provider_ready: bool,
    persona_id: str | None = None,
) -> tuple[bool, str | None, str, str]:
    """Pure, ordered recipient gate used by preview and future send admission."""
    if not lead.get("persona_id"):
        return False, "missing_persona", "policy_blocked", "unknown"
    if persona_id and str(lead.get("persona_id")) != str(persona_id):
        return False, "persona_mismatch", "policy_blocked", "unknown"
    if not semantic_audience_id:
        return False, "missing_semantic_group", "policy_blocked", "unknown"
    if semantic_audience_id != selected_audience_id:
        return False, "semantic_group_mismatch", "policy_blocked", "unknown"
    if not _valid_phone(lead):
        return False, "invalid_number", "none", "invalid_number"
    if consent_status in {"refused", "revoked"}:
        return False, "consent_revoked_or_refused", "unsubscribed", "valid"
    if consent_status == "review_required":
        return False, "consent_review_required", "policy_blocked", "valid"
    if campaign_kind == "promotional" and consent_status != "granted":
        return False, "consent_not_granted", "policy_blocked", "valid"
    if not provider_ready:
        return False, "provider_unavailable", "policy_blocked", "valid"
    return True, None, "none", "valid"


EVOLUTION_DELIVERY_CAVEAT = (
    "Envios por Evolution podem aparecer como 'enviado' mesmo sem confirmacao "
    "real de entrega no aparelho, por uma limitacao conhecida do WhatsApp/"
    "Baileys (identidades @lid). Ver docs/architecture/WHATSAPP_N8N_RUNTIME.md."
)


def has_active_conversation_window(persona_id: str, lead_id: int, hours: int = 24) -> bool:
    """Heuristic proxy for Meta's 24h customer-service window.

    Meta doesn't expose real window state via an API, so this only checks for
    a recent inbound message from this lead. A false positive here still
    fails safely: the provider call itself rejects a plain-text send outside
    the real window with a genuine Graph API error.
    """
    messages = supabase_client.get_messages(str(lead_id), limit=20) or []
    for message in reversed(messages):
        if message.get("direction") != "inbound":
            continue
        created_at = message.get("created_at")
        if not created_at:
            return False
        try:
            sent_at = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        except ValueError:
            return False
        return (datetime.now(timezone.utc) - sent_at) <= timedelta(hours=hours)
    return False


def _with_lead_name_fallbacks(variables: dict[str, Any], lead: dict[str, Any]) -> dict[str, Any]:
    """Templates commonly greet the lead by name using either a named
    placeholder ({{nome}}) or Meta's positional style ({{1}}); default both
    to the lead's name so an operator doesn't have to configure this per
    template for the common case."""
    resolved = dict(variables)
    name = lead.get("nome") or lead.get("name") or ""
    resolved.setdefault("nome", name)
    resolved.setdefault("1", name)
    return resolved


def _build_meta_components(template_row: dict[str, Any], variables: dict[str, Any]) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    for component in template_row.get("meta_component_schema") or []:
        placeholders = component.get("placeholders") or []
        if not placeholders:
            continue
        components.append({
            "type": component.get("type") or "body",
            "parameters": [{"type": "text", "text": str(variables.get(name, ""))} for name in placeholders],
        })
    return components


def _render_evolution_body(template_row: dict[str, Any], variables: dict[str, Any]) -> str:
    body = str(template_row.get("evolution_body_template") or "")
    for key, value in variables.items():
        body = body.replace("{{" + str(key) + "}}", str(value))
    return body


def resolve_send_payload(
    *,
    provider: str,
    send_mode: str,
    revision_content: dict[str, Any],
    template: dict[str, Any] | None,
    lead: dict[str, Any],
    persona_id: str,
) -> dict[str, Any]:
    """Decide template vs. plain text for one recipient.

    Meta: plain text is only ever allowed when explicitly flagged as a
    controlled test AND there's evidence of an active conversation window for
    this specific lead; otherwise an approved template is mandatory.
    Evolution has no Meta-style approval/window constraint â€” it sends the
    template's rendered body (if attached) or the frozen reference message.
    """
    if provider != "meta_cloud":
        if template:
            variables = _with_lead_name_fallbacks(revision_content.get("variables") or {}, lead)
            text = _render_evolution_body(template, variables).strip()
        else:
            text = str(revision_content.get("message") or "").strip()
        if not text:
            return {"mode": "blocked", "reason": "message_required"}
        return {"mode": "text", "text": text}

    if template:
        variables = _with_lead_name_fallbacks(revision_content.get("variables") or {}, lead)
        components = _build_meta_components(template, variables)
        preview_text = str(revision_content.get("message") or template.get("meta_template_name") or "")
        return {
            "mode": "template",
            "text": preview_text,
            "template": {
                "name": template.get("meta_template_name"),
                "language": template.get("meta_template_language") or "pt_BR",
                "components": components,
            },
        }

    if send_mode == "controlled_test" and has_active_conversation_window(persona_id, int(lead["id"])):
        text = str(revision_content.get("message") or "").strip()
        if not text:
            return {"mode": "blocked", "reason": "message_required"}
        return {"mode": "text", "text": text}

    return {"mode": "blocked", "reason": "template_required"}


def _extract_placeholders(text: str) -> list[str]:
    seen: list[str] = []
    for match in re.findall(r"\{\{\s*(\w+)\s*\}\}", text or ""):
        if match not in seen:
            seen.append(match)
    return seen


def list_message_templates(persona_id: str, provider: str) -> list[dict[str, Any]]:
    return _rows(
        supabase_client.get_client().table("message_templates").select("*")
        .eq("persona_id", persona_id).eq("provider", provider).eq("status", "active")
        .order("created_at", desc=True).limit(200)
    )


def create_message_template(payload: dict[str, Any], *, actor_user_id: str | None) -> dict[str, Any]:
    persona_id = str(payload.get("persona_id") or "")
    provider = str(payload.get("provider") or "")
    template_key = str(payload.get("template_key") or "").strip()
    body = str(payload.get("body") or "").strip()
    if not persona_id or provider not in {"meta_cloud", "evolution_baileys"} or not template_key or not body:
        raise HTTPException(422, "persona_id, provider, template_key e corpo do template sao obrigatorios.")

    row: dict[str, Any] = {
        "persona_id": persona_id,
        "provider": provider,
        "template_key": template_key,
        "status": "active",
        "created_by_user_id": _uuid_or_none(actor_user_id),
    }
    placeholders = _extract_placeholders(body)
    if provider == "meta_cloud":
        name = str(payload.get("meta_template_name") or "").strip()
        if not name:
            raise HTTPException(422, "Nome do template aprovado na Meta e obrigatorio.")
        row.update({
            "meta_template_name": name,
            "meta_template_language": str(payload.get("meta_template_language") or "pt_BR").strip(),
            "meta_template_category": str(payload.get("meta_template_category") or "MARKETING").strip(),
            "meta_component_schema": [{"type": "body", "placeholders": placeholders}] if placeholders else [],
        })
    else:
        row.update({
            "evolution_body_template": body,
            "evolution_variables": placeholders,
        })
    result = supabase_client.get_client().table("message_templates").insert(row).execute()
    data = getattr(result, "data", None)
    return data[0] if isinstance(data, list) and data else row


def preview_campaign(payload: dict[str, Any]) -> dict[str, Any]:
    persona_id = str(payload.get("persona_id") or "")
    audience_id = str(payload.get("audience_id") or "")
    import_ids = [str(value) for value in (payload.get("import_batch_ids") or []) if value]
    if not persona_id or not audience_id or not import_ids:
        raise HTTPException(422, "Persona, grupo e ao menos um import sao obrigatorios.")

    persona = supabase_client.get_persona_by_id(persona_id)
    audience = supabase_client.get_audience(audience_id)
    if not persona or not audience or audience.get("persona_id") != persona_id:
        raise HTTPException(404, "Persona ou grupo nao encontrado.")
    if not rollout_one_enabled(persona):
        raise HTTPException(503, "Campanhas em massa estao desabilitadas para esta persona.")
    if str((audience.get("metadata") or {}).get("kind") or "semantic_group") != "semantic_group":
        raise HTTPException(422, "A campanha exige um grupo semantico.")

    client = supabase_client.get_client()
    batches = _rows(client.table("lead_import_batches").select("*").in_("id", import_ids).limit(len(import_ids)))
    by_id = {str(row["id"]): row for row in batches}
    if len(by_id) != len(set(import_ids)):
        raise HTTPException(404, "Um ou mais imports nao foram encontrados.")
    for batch_id in import_ids:
        batch = by_id[batch_id]
        if batch.get("persona_id") != persona_id:
            raise HTTPException(403, "Import de outra persona nao pode ser usado.")
        if batch.get("status") != "completed":
            raise HTTPException(409, "Somente imports concluidos podem entrar na campanha.")

    campaign_kind = str(payload.get("campaign_kind") or "consent_request")
    purpose = str(payload.get("purpose") or "").strip()
    policy, policy_checksum = resolve_contact_policy(
        persona=persona,
        audience=audience,
        campaign_overrides=payload.get("policy_overrides") or {},
        campaign_kind=campaign_kind,
        purpose=purpose,
    )
    provider = str(payload.get("provider") or "meta_cloud")
    if provider not in {"meta_cloud", "evolution_baileys"}:
        raise HTTPException(422, "Provider de campanha invalido.")
    binding = supabase_client.get_active_whatsapp_binding(persona_id)
    provider_ready = resolve_provider_ready(binding, provider)

    leads, batches_by_lead = _campaign_leads(import_ids)
    lead_ids = [int(row["id"]) for row in leads]
    memberships = _semantic_membership_map(persona_id, lead_ids)
    consents = _latest_consents(persona_id=persona_id, lead_ids=lead_ids, channel="whatsapp", purpose=purpose)

    recipients: list[dict[str, Any]] = []
    blocked = Counter()
    for lead in leads:
        lead_id = int(lead["id"])
        consent_status, consent_id = resolve_applicable_consent(consents, lead_id)
        eligible, reason, suppression, contact_status = evaluate_recipient_eligibility(
            lead=lead,
            selected_audience_id=audience_id,
            semantic_audience_id=memberships.get(lead_id),
            campaign_kind=campaign_kind,
            consent_status=consent_status,
            provider_ready=provider_ready,
            persona_id=persona_id,
        )
        if reason:
            blocked[reason] += 1
        recipients.append({
            "lead_id": lead_id,
            "name": lead.get("nome") or lead.get("name") or "Lead",
            "eligible": eligible,
            "blocked_reason": reason,
            "consent_status": consent_status,
            "applicable_consent_id": consent_id,
            "sequence_status": "eligible" if eligible else "blocked",
            "suppression_status": suppression,
            "contact_status": contact_status,
            "import_batch_ids": sorted(batches_by_lead.get(lead_id, set())),
        })

    result = {
        "persona_id": persona_id,
        "audience": audience,
        "imports": [by_id[value] for value in import_ids],
        "campaign_kind": campaign_kind,
        "purpose": purpose,
        "provider": provider,
        "provider_ready": provider_ready,
        "policy": policy,
        "policy_checksum": policy_checksum,
        "counts": {
            "selected_unique": len(recipients),
            "eligible": sum(1 for row in recipients if row["eligible"]),
            "blocked": sum(1 for row in recipients if not row["eligible"]),
            "duplicate_rows_removed": max(0, sum(len(value) for value in batches_by_lead.values()) - len(recipients)),
        },
        "blocked_reasons": dict(sorted(blocked.items())),
        "recipients": recipients,
    }
    result["preview_checksum"] = _canonical_checksum({
        "persona_id": persona_id,
        "audience_id": audience_id,
        "import_batch_ids": sorted(set(import_ids)),
        "campaign_kind": campaign_kind,
        "purpose": purpose,
        "provider": provider,
        "policy_checksum": policy_checksum,
        "content": {
            "template_name": payload.get("template_name"),
            "template_language": payload.get("template_language") or "pt_BR",
            "template_id": payload.get("template_id"),
            "send_mode": payload.get("send_mode") or "",
            "message": payload.get("message"),
            "variables": payload.get("variables") or {},
            "assets": payload.get("assets") or [],
        },
        "recipients": [
            {
                "lead_id": row["lead_id"],
                "consent_status": row["consent_status"],
                "sequence_status": row["sequence_status"],
                "blocked_reason": row["blocked_reason"],
            }
            for row in recipients
        ],
    })
    return result


def _current_graph_snapshot(persona: dict[str, Any]) -> tuple[int | None, str | None]:
    loaded = graph_json_v2_store.load_current(str(persona.get("slug") or ""))
    if not loaded:
        return None, None
    version, graph = loaded
    return version, graph_json_v2_store.checksum_graph(graph)


def create_campaign_draft(payload: dict[str, Any], *, actor_user_id: str | None) -> dict[str, Any]:
    idempotency_key = str(payload.get("idempotency_key") or "").strip()
    reason = str(payload.get("reason") or "").strip()
    expected_revision = int(payload.get("expected_revision") or 0)
    if not idempotency_key or not reason:
        raise HTTPException(422, "idempotency_key e reason sao obrigatorios.")
    if expected_revision != 0:
        raise HTTPException(409, "Uma nova campanha deve usar expected_revision=0.")

    client = supabase_client.get_client()
    existing = _rows(
        client.table("campaigns").select("id")
        .eq("idempotency_key", idempotency_key).limit(1)
    )
    if existing:
        return get_campaign_detail(str(existing[0]["id"]))

    preview = preview_campaign(payload)
    if preview["preview_checksum"] != str(payload.get("expected_preview_checksum") or ""):
        raise HTTPException(409, "A elegibilidade mudou; execute o preview novamente.")
    if preview["counts"]["selected_unique"] == 0:
        raise HTTPException(422, "Os imports selecionados nao possuem leads utilizaveis.")
    persona = supabase_client.get_persona_by_id(preview["persona_id"]) or {}
    graph_version, graph_checksum = _current_graph_snapshot(persona)
    campaign_id = str(uuid.uuid4())
    revision_id = str(uuid.uuid4())
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(422, "Nome da campanha e obrigatorio.")
    now = _now_iso()
    content_snapshot = {
        "template_name": payload.get("template_name"),
        "template_language": payload.get("template_language") or "pt_BR",
        "template_id": payload.get("template_id"),
        "send_mode": payload.get("send_mode") or "",
        "message": payload.get("message"),
        "variables": payload.get("variables") or {},
        "assets": payload.get("assets") or [],
    }
    audience_snapshot = {
        "id": preview["audience"].get("id"),
        "slug": preview["audience"].get("slug"),
        "name": preview["audience"].get("name"),
        "description": preview["audience"].get("description"),
        "metadata": preview["audience"].get("metadata") or {},
    }
    database_actor_id = _uuid_or_none(actor_user_id)
    campaign_row = {
        "id": campaign_id,
        "persona_id": preview["persona_id"],
        "slug": f"{_slugify(name)}-{campaign_id[:8]}",
        "name": name,
        "status": "draft",
        "format": "whatsapp_template",
        "campaign_kind": preview["campaign_kind"],
        "purpose": preview["purpose"],
        "channel": "whatsapp",
        "provider": preview["provider"],
        "audience_id": preview["audience"]["id"],
        "current_revision": 1,
        "idempotency_key": idempotency_key,
        "reason": reason,
        "created_by_user_id": database_actor_id,
        "metadata": {"kind": "operational_campaign", "delivery_rollout": 1},
        "created_at": now,
        "updated_at": now,
    }
    revision_snapshot = {
        "campaign_id": campaign_id,
        "revision": 1,
        "campaign_kind": preview["campaign_kind"],
        "purpose": preview["purpose"],
        "objective": payload.get("objective"),
        "channel": "whatsapp",
        "provider": preview["provider"],
        "graph_version": graph_version,
        "graph_checksum": graph_checksum,
        "audience": audience_snapshot,
        "content": content_snapshot,
        "policy": preview["policy"],
        "policy_checksum": preview["policy_checksum"],
        "import_batch_ids": [row["id"] for row in preview["imports"]],
    }
    revision_row = {
        "id": revision_id,
        "campaign_id": campaign_id,
        "persona_id": preview["persona_id"],
        "revision": 1,
        "audience_id": preview["audience"]["id"],
        "campaign_kind": preview["campaign_kind"],
        "purpose": preview["purpose"],
        "objective": payload.get("objective"),
        "channel": "whatsapp",
        "provider": preview["provider"],
        "graph_version": graph_version,
        "graph_checksum": graph_checksum,
        "audience_snapshot": audience_snapshot,
        "content_snapshot": content_snapshot,
        "policy_snapshot": preview["policy"],
        "policy_checksum": preview["policy_checksum"],
        "revision_checksum": _canonical_checksum(revision_snapshot),
        "status": "draft",
        "created_by_user_id": database_actor_id,
        "template_id": payload.get("template_id") or None,
        "reason": reason,
        "created_at": now,
    }
    recipient_rows = [
        {
            "lead_id": row["lead_id"],
            "applicable_consent_id": row["applicable_consent_id"],
            "consent_status": row["consent_status"],
            "sequence_status": row["sequence_status"],
            "suppression_status": row["suppression_status"],
            "contact_status": row["contact_status"],
            "blocked_reason": row["blocked_reason"],
        }
        for row in preview["recipients"]
    ]
    event_payload = {
        "campaign_revision": 1,
        "policy_checksum": preview["policy_checksum"],
        "revision_checksum": revision_row["revision_checksum"],
        "selected_unique": preview["counts"]["selected_unique"],
        "eligible": preview["counts"]["eligible"],
        "blocked": preview["counts"]["blocked"],
        "idempotency_key": idempotency_key,
        "reason": reason,
        "actor_user_id": actor_user_id,
    }
    try:
        result = client.rpc("create_campaign_draft_v1", {
            "p_campaign": campaign_row,
            "p_revision": revision_row,
            "p_import_ids": [row["id"] for row in preview["imports"]],
            "p_recipients": recipient_rows,
            "p_event": event_payload,
        }).execute().data
    except Exception:
        # A concurrent replay may win the unique idempotency key.
        existing = _rows(
            client.table("campaigns").select("id")
            .eq("idempotency_key", idempotency_key).limit(1)
        )
        if not existing:
            raise
        campaign_id = str(existing[0]["id"])
    else:
        rpc_result = result if isinstance(result, dict) else (result[0] if result else {})
        campaign_id = str(rpc_result.get("campaign_id") or campaign_id)
    return get_campaign_detail(campaign_id)


def list_campaigns(persona_id: str | None = None) -> list[dict[str, Any]]:
    query = supabase_client.get_client().table("campaigns").select("*").eq("metadata->>kind", "operational_campaign")
    if persona_id:
        query = query.eq("persona_id", persona_id)
    campaigns = _rows(query.order("created_at", desc=True).limit(500))
    for campaign in campaigns:
        recipients = _rows(
            supabase_client.get_client().table("campaign_recipients")
            .select("sequence_status,blocked_reason")
            .eq("campaign_id", campaign["id"])
            .eq("campaign_revision", campaign.get("current_revision") or 1)
            .limit(100000)
        )
        campaign["counts"] = {
            "selected_unique": len(recipients),
            "eligible": sum(row.get("sequence_status") == "eligible" for row in recipients),
            "blocked": sum(row.get("sequence_status") == "blocked" for row in recipients),
        }
    return campaigns


def get_campaign_detail(campaign_id: str) -> dict[str, Any]:
    client = supabase_client.get_client()
    campaign_rows = _rows(client.table("campaigns").select("*").eq("id", campaign_id).limit(1))
    if not campaign_rows:
        raise HTTPException(404, "Campanha nao encontrada.")
    campaign = campaign_rows[0]
    revision_rows = _rows(
        client.table("campaign_revisions").select("*")
        .eq("campaign_id", campaign_id)
        .eq("revision", campaign.get("current_revision") or 1)
        .limit(1)
    )
    revision = revision_rows[0] if revision_rows else None
    recipients = _rows(
        client.table("campaign_recipients").select("*")
        .eq("campaign_id", campaign_id)
        .eq("campaign_revision", campaign.get("current_revision") or 1)
        .order("created_at")
        .limit(100000)
    )
    import_ids: list[str] = []
    imports: list[dict[str, Any]] = []
    if revision:
        links = _rows(client.table("campaign_revision_imports").select("import_batch_id").eq("campaign_revision_id", revision["id"]).limit(1000))
        import_ids = [str(row["import_batch_id"]) for row in links]
        if import_ids:
            imports = _rows(client.table("lead_import_batches").select("*").in_("id", import_ids).limit(len(import_ids)))
    reasons = Counter(row.get("blocked_reason") for row in recipients if row.get("blocked_reason"))
    provider = (revision or {}).get("provider")
    if recipients:
        send_rows = _rows(
            client.table("lead_buffer")
            .select("campaign_recipient_id,status,external_message_id,last_error,updated_at,created_at")
            .eq("campaign_id", campaign_id)
            .eq("campaign_revision", campaign.get("current_revision") or 1)
            .order("created_at", desc=True)
            .limit(100000)
        )
        latest_send: dict[str, dict[str, Any]] = {}
        for row in send_rows:
            key = str(row.get("campaign_recipient_id") or "")
            if key and key not in latest_send:
                latest_send[key] = row
        for row in recipients:
            send_row = latest_send.get(str(row["id"]))
            row["send"] = {
                "provider": provider,
                "status": send_row.get("status"),
                "external_message_id": send_row.get("external_message_id"),
                "error": send_row.get("last_error"),
                "attempted_at": send_row.get("updated_at") or send_row.get("created_at"),
            } if send_row else None
    result = {
        "campaign": campaign,
        "revision": revision,
        "imports": imports,
        "counts": {
            "selected_unique": len(recipients),
            "eligible": sum(row.get("sequence_status") == "eligible" for row in recipients),
            "blocked": sum(row.get("sequence_status") == "blocked" for row in recipients),
            "responded": sum(row.get("response_attributed_at") is not None for row in recipients),
        },
        "blocked_reasons": dict(sorted(reasons.items())),
        "recipients": recipients,
    }
    if provider == "evolution_baileys":
        result["delivery_confidence_caveat"] = EVOLUTION_DELIVERY_CAVEAT
    return result


def update_campaign_status(
    campaign_id: str,
    *,
    expected_revision: int,
    status: str,
    idempotency_key: str,
    reason: str,
    actor_user_id: str | None,
) -> dict[str, Any]:
    if status not in {"paused", "cancelled"}:
        raise HTTPException(422, "Transicao nao suportada na Entrega 1.")
    if not idempotency_key or not reason:
        raise HTTPException(422, "idempotency_key e reason sao obrigatorios.")
    detail = get_campaign_detail(campaign_id)
    campaign = detail["campaign"]
    if int(campaign.get("current_revision") or 1) != int(expected_revision):
        raise HTTPException(409, "A campanha foi alterada; recarregue antes de continuar.")
    if campaign.get("status") in {"cancelled", "completed"}:
        raise HTTPException(409, "Campanha encerrada nao pode mudar de estado.")
    supabase_client.get_client().rpc("transition_campaign_status_v1", {
        "p_campaign_id": campaign_id,
        "p_expected_revision": expected_revision,
        "p_status": status,
        "p_idempotency_key": idempotency_key,
        "p_reason": reason,
        "p_actor_user_id": _uuid_or_none(actor_user_id),
    }).execute()
    if status == "cancelled":
        # Recipients that never got admitted stay visibly excluded; anything
        # already 'queued' has a real outbox row and follows the outbox's own
        # lifecycle instead (never force-cancelled, same as 1:1 conversation
        # sends today).
        supabase_client.get_client().table("campaign_recipients").update({
            "sequence_status": "cancelled",
        }).eq("campaign_id", campaign_id).eq(
            "campaign_revision", expected_revision
        ).eq("sequence_status", "eligible").execute()
    return get_campaign_detail(campaign_id)


def _mark_recipient_send_failed(recipient_id: str, blocked_reason: str) -> None:
    supabase_client.get_client().table("campaign_recipients").update({
        "sequence_status": "send_failed",
        "blocked_reason": blocked_reason,
    }).eq("id", recipient_id).execute()


def send_campaign(
    campaign_id: str,
    *,
    expected_revision: int,
    idempotency_key: str,
    reason: str | None,
    actor_user_id: str | None,
) -> dict[str, Any]:
    if not idempotency_key:
        raise HTTPException(422, "idempotency_key e obrigatorio.")
    detail = get_campaign_detail(campaign_id)
    campaign = detail["campaign"]
    revision = detail["revision"]
    if not revision:
        raise HTTPException(409, "Campanha sem revisao congelada.")
    if int(campaign.get("current_revision") or 1) != int(expected_revision):
        raise HTTPException(409, "A campanha foi alterada; recarregue antes de continuar.")
    if campaign.get("status") not in {"draft", "validated", "running"}:
        raise HTTPException(409, "Campanha nao esta pronta para envio.")

    provider = str(revision.get("provider") or "meta_cloud")
    if provider not in {"meta_cloud", "evolution_baileys"}:
        raise HTTPException(422, "Provider de campanha invalido.")

    clean_reason = str(reason or "").strip()
    if provider == "meta_cloud" and not clean_reason:
        raise HTTPException(422, "reason e obrigatorio para envio via Meta.")

    persona_id = str(campaign["persona_id"])
    binding = supabase_client.get_active_whatsapp_binding(persona_id)
    if not resolve_provider_ready(binding, provider):
        raise HTTPException(409, "Canal de mensageria desta persona nao esta pronto para envio.")

    content = revision.get("content_snapshot") or {}
    send_mode = str(content.get("send_mode") or "")
    template_row: dict[str, Any] | None = None
    template_id = revision.get("template_id")
    if template_id:
        rows = _rows(
            supabase_client.get_client().table("message_templates").select("*")
            .eq("id", template_id).limit(1)
        )
        template_row = rows[0] if rows else None

    audience_id = str(revision.get("audience_id") or "")
    campaign_kind = str(revision.get("campaign_kind") or "consent_request")
    purpose = str(revision.get("purpose") or "")

    client = supabase_client.get_client()
    recipients = _rows(
        client.table("campaign_recipients").select("*")
        .eq("campaign_id", campaign_id)
        .eq("campaign_revision", campaign.get("current_revision") or 1)
        .eq("sequence_status", "eligible")
        .limit(100000)
    )
    if not recipients:
        raise HTTPException(422, "Nenhum destinatario elegivel para envio.")

    lead_ids = [int(row["lead_id"]) for row in recipients]
    leads_by_id: dict[int, dict[str, Any]] = {}
    for group in _chunks(lead_ids):
        for lead in _rows(client.table("leads").select("*").in_("id", group).limit(len(group))):
            leads_by_id[int(lead["id"])] = lead
    memberships = _semantic_membership_map(persona_id, lead_ids)
    consents = _latest_consents(persona_id=persona_id, lead_ids=lead_ids, channel="whatsapp", purpose=purpose)

    queued = 0
    failed = 0
    for recipient in recipients:
        lead_id = int(recipient["lead_id"])
        lead = leads_by_id.get(lead_id)
        if not lead:
            _mark_recipient_send_failed(recipient["id"], "lead_not_found")
            failed += 1
            continue
        consent_status, _consent_id = resolve_applicable_consent(consents, lead_id)
        eligible, block_reason, _suppression, _contact_status = evaluate_recipient_eligibility(
            lead=lead,
            selected_audience_id=audience_id,
            semantic_audience_id=memberships.get(lead_id),
            campaign_kind=campaign_kind,
            consent_status=consent_status,
            provider_ready=True,
            persona_id=persona_id,
        )
        if not eligible:
            _mark_recipient_send_failed(recipient["id"], block_reason or "not_eligible")
            failed += 1
            continue

        resolved = resolve_send_payload(
            provider=provider,
            send_mode=send_mode,
            revision_content=content,
            template=template_row,
            lead=lead,
            persona_id=persona_id,
        )
        if resolved["mode"] == "blocked":
            _mark_recipient_send_failed(recipient["id"], resolved["reason"])
            failed += 1
            continue

        step = int(recipient.get("commercial_attempt_count") or 0) + 1
        try:
            whatsapp_outbox.enqueue_outbound(
                lead=lead,
                text=resolved["text"],
                sender_type="campaign",
                message_id=f"campaign:{campaign_id}:{recipient['id']}:{step}",
                correlation_id=f"campaign:{campaign_id}:{revision['revision']}:{recipient['id']}:{step}",
                idempotency_key=f"campaign-send:{campaign_id}:{revision['revision']}:{recipient['id']}:{step}",
                template=resolved.get("template"),
                campaign_scope={
                    "campaign_id": campaign_id,
                    "campaign_revision": revision["revision"],
                    "campaign_recipient_id": recipient["id"],
                    "campaign_step": step,
                    "policy_checksum": recipient.get("policy_checksum"),
                },
            )
        except HTTPException as exc:
            _mark_recipient_send_failed(recipient["id"], f"enqueue_error:{exc.status_code}")
            failed += 1
            continue

        client.table("campaign_recipients").update({
            "sequence_status": "queued",
            "commercial_attempt_count": step,
            "last_commercial_attempt_at": _now_iso(),
        }).eq("id", recipient["id"]).execute()
        queued += 1

    if queued and campaign.get("status") != "running":
        client.table("campaigns").update({
            "status": "running", "updated_at": _now_iso(),
        }).eq("id", campaign_id).execute()

    event_emitter.emit(
        "campaign_send_started",
        entity_type="campaign",
        entity_id=campaign_id,
        persona_id=persona_id,
        payload={
            "campaign_revision": revision["revision"],
            "idempotency_key": idempotency_key,
            "reason": clean_reason,
            "actor_user_id": actor_user_id,
            "queued": queued,
            "failed": failed,
        },
        source="campaigns.send",
    )

    return get_campaign_detail(campaign_id)

