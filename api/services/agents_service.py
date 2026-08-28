# -*- coding: utf-8 -*-
"""
Agents (per-persona bots) and role assignments (sdr / closer / followup).

Schema lives in supabase/migrations/007_agents_routing.sql:
  - agents (one row per bot, scoped per persona)
  - persona_role_assignments (which agent — or NULL=human — handles each role)

This module is the single source of truth for resolving "who handles this
lead now?". /process calls resolve_for_stage(persona_id, funnel_stage) and
either runs the resolved agent or pauses the AI for human handoff.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from services import graph_agent_runtime_v3, supabase_client
from services.graph_agent_runtime_v3 import RESUME_ANSWER_WINDOW_SECONDS

logger = logging.getLogger("agents_service")

VALID_ROLES = ("sdr", "closer", "followup")
_ROLE_ASSIGNMENTS_TABLE_MISSING = False

# Funnel stage → role.
# Conservative defaults: most stages map to SDR; fechamento/oportunidade →
# closer; pos_venda / follow_up → followup.
_STAGE_TO_ROLE = {
    "novo":          "sdr",
    "contato":       "sdr",
    "qualificacao":  "sdr",
    "qualificado":   "sdr",
    "interessado":   "sdr",
    "oportunidade":  "closer",
    "negociacao":    "closer",
    "fechamento":    "closer",
    "fechado":       "closer",
    "pos_venda":     "followup",
    "follow_up":     "followup",
    "follow-up":     "followup",
}


def role_for_stage(funnel_stage: Optional[str]) -> str:
    return _STAGE_TO_ROLE.get((funnel_stage or "").lower(), "sdr")


# ── agents CRUD ──────────────────────────────────────────────────

def list_agents(persona_id: Optional[str] = None, include_inactive: bool = False) -> list:
    client = supabase_client.get_client()
    try:
        q = client.table("agents").select("*").order("created_at", desc=False)
        if persona_id:
            q = q.eq("persona_id", persona_id)
        if not include_inactive:
            q = q.eq("active", True)
        return supabase_client._q(q)
    except Exception as exc:
        logger.warning("list_agents failed: %s", exc)
        return []


def get_agent(agent_id: str) -> Optional[dict]:
    if not agent_id:
        return None
    client = supabase_client.get_client()
    return supabase_client._one(
        client.table("agents").select("*").eq("id", agent_id).maybe_single()
    )


def create_agent(data: dict) -> dict:
    client = supabase_client.get_client()
    return supabase_client._insert_one(client.table("agents").insert(data))


def update_agent(agent_id: str, data: dict) -> Optional[dict]:
    client = supabase_client.get_client()
    try:
        result = client.table("agents").update(data).eq("id", agent_id).execute()
        if result and result.data:
            return result.data[0]
    except Exception as exc:
        logger.warning("update_agent failed: %s", exc)
    return None


def deactivate_agent(agent_id: str) -> bool:
    return update_agent(agent_id, {"active": False}) is not None


# ── role assignments ─────────────────────────────────────────────

def get_role_assignments(persona_id: str) -> dict:
    """Return {role: agent_id_or_None}. Always includes all VALID_ROLES."""
    global _ROLE_ASSIGNMENTS_TABLE_MISSING
    out = {role: None for role in VALID_ROLES}
    if not persona_id:
        return out
    if _ROLE_ASSIGNMENTS_TABLE_MISSING:
        return out
    client = supabase_client.get_client()
    try:
        result = (
            client.table("persona_role_assignments")
            .select("role,agent_id,active")
            .eq("persona_id", persona_id)
            .execute()
        )
        rows = result.data or []
        for row in rows:
            if row.get("role") in VALID_ROLES and row.get("active", True):
                out[row["role"]] = row.get("agent_id")
    except Exception as exc:
        if _is_missing_role_assignments_table(exc):
            _ROLE_ASSIGNMENTS_TABLE_MISSING = True
            logger.warning(
                "persona_role_assignments table is missing; falling back to human handoff until migration 007 is applied"
            )
            return out
        logger.warning("get_role_assignments failed: %s", exc)
    return out


def set_role_assignment(persona_id: str, role: str, agent_id: Optional[str]) -> dict:
    if role not in VALID_ROLES:
        raise ValueError(f"role must be one of {VALID_ROLES}")
    client = supabase_client.get_client()
    payload = {
        "persona_id": persona_id,
        "role": role,
        "agent_id": agent_id,
        "active": True,
    }
    try:
        result = (
            client.table("persona_role_assignments")
            .upsert(payload, on_conflict="persona_id,role")
            .execute()
        )
        return (result.data or [{}])[0]
    except Exception as exc:
        logger.warning("set_role_assignment failed: %s", exc)
        return {}


def _is_missing_role_assignments_table(exc: Exception) -> bool:
    text = str(exc)
    return (
        "persona_role_assignments" in text
        and ("PGRST205" in text or "schema cache" in text or "Could not find the table" in text)
    )


# ── runtime resolver ─────────────────────────────────────────────

def resolve_for_stage(
    persona_slug_or_id: str, funnel_stage: str
) -> tuple[Optional[dict], str]:
    """Resolve the agent that should answer for (persona, funnel_stage).

    Args:
        persona_slug_or_id: persona slug ("tock-fatal") or UUID.
        funnel_stage: lead's current funnel stage.

    Returns:
        (agent_record_or_None, role)
        - agent_record None  →  human handles this role for this persona.
        - empty role assignment row missing  →  also returns None (human).
    """
    role = role_for_stage(funnel_stage)
    persona_id = _resolve_persona_id(persona_slug_or_id)
    if not persona_id:
        return None, role

    assignments = get_role_assignments(persona_id)
    agent_id = assignments.get(role)
    if not agent_id:
        return None, role
    return get_agent(agent_id), role


def _resolve_persona_id(persona_slug_or_id: str) -> Optional[str]:
    if not persona_slug_or_id:
        return None
    # Looks like UUID (36 chars with dashes) — pass through.
    if len(persona_slug_or_id) == 36 and persona_slug_or_id.count("-") == 4:
        return persona_slug_or_id
    persona = supabase_client.get_persona(persona_slug_or_id)
    return persona.get("id") if persona else None


# ── lead pause/resume ────────────────────────────────────────────

def pause_lead(lead_ref: int) -> bool:
    try:
        supabase_client.update_lead(lead_ref, {"handoff_level": "full"})
        return True
    except Exception as exc:
        logger.warning("pause_lead failed: %s", exc)
        return False


def acknowledge_partial_handoff(lead_ref: int) -> bool:
    """Clear a 'partial' handoff flag once a human has reviewed it.

    Unlike resume_lead, a partial handoff never stopped the AI or parked
    lead_buffer rows as waiting_human, so there's nothing to reset or
    requeue here — just clear the flag.
    """
    try:
        supabase_client.update_lead(lead_ref, {"handoff_level": "none"})
        return True
    except Exception as exc:
        logger.warning("acknowledge_partial_handoff failed: %s", exc)
        return False


def _cleared_conversation_state_metadata(lead: dict) -> Optional[dict]:
    """Clear a lead's sticky "handoff" flag so /process actually retries.

    conversation_runtime persists the deterministic engines' working state
    under metadata.conversation_state (or metadata.vitoria_state for legacy
    Baita leads — same fallback conversation_runtime._build_context uses).
    Both DeterministicAppointment and DeterministicSDR short-circuit with an
    empty reply the moment that state's own "conversation_state" field is
    "handoff", regardless of handoff_level. Left untouched, resuming a lead
    just makes it silently re-pause on the next inbound message instead of
    trying to answer. Only the sticky flag and the stale clarification
    counter are reset here — collected fields (appointment_request, items,
    etc.) must survive the resume. This is the legacy engines' format only
    — v3's equivalent sticky state lives in conversation_ledgers and is
    handled separately by _reset_v3_ledger_if_applicable.
    """
    metadata = dict(lead.get("metadata") or {})
    for key in ("conversation_state", "vitoria_state"):
        cart_state = metadata.get(key)
        if isinstance(cart_state, dict) and cart_state.get("conversation_state") == "handoff":
            cart_state = dict(cart_state)
            cart_state["conversation_state"] = ""
            cart_state["clarification_attempts"] = 0
            metadata = {**metadata, key: cart_state}
            return metadata
    return None


def resume_lead(lead_ref: int) -> bool:
    update_payload: dict = {"handoff_level": "none"}
    lead: Optional[dict] = None
    try:
        lead = supabase_client.get_lead_by_ref(lead_ref)
    except Exception as exc:
        # Best-effort: a lookup failure must not block the resume itself,
        # it just means the sticky flag (if any) won't be cleared this time.
        logger.warning("resume_lead lead lookup failed: %s", exc)
    if lead:
        try:
            metadata = _cleared_conversation_state_metadata(lead)
            if metadata is None:
                metadata = dict(lead.get("metadata") or {})
            # A resumed lead may still be leaning on facts collected before
            # this pause (name, vehicle, service) that could be stale by
            # now -- the next reply must confirm them instead of silently
            # assuming they still hold. Consumed and cleared by
            # graph_agent_runtime_v3.build_context on the next turn.
            metadata["pending_reconfirmation"] = True
            update_payload["metadata"] = metadata
        except Exception as exc:
            logger.warning("resume_lead conversation-state clearing failed: %s", exc)
        # Preserve the v3 ledger. Resume is not a new journey and therefore
        # must not erase its authoritative branch or asked-question history.
    try:
        supabase_client.update_lead(lead_ref, update_payload)
    except Exception as exc:
        logger.warning("resume_lead failed: %s", exc)
        return False
    # Respect the customer's own timing. An inbound that has been parked
    # longer than the published window is not a conversation waiting to be
    # continued -- answering it hours later reads as an agent talking to
    # itself. The buffer rows stay exactly where they are; the next message
    # the customer sends starts the conversation again.
    window = resume_answer_window(lead or {})
    _LAST_RESUME_WINDOW[lead_ref] = window
    if not window.get("may_speak"):
        _LAST_REQUEUED.pop(lead_ref, None)
        logger.info(
            "resume_lead stays silent lead_ref=%s reason=%s",
            lead_ref, window.get("reason"),
        )
        return True
    try:
        requeued = supabase_client.requeue_waiting_human_whatsapp_buffer(lead_ref)
        if requeued:
            logger.info("resume_lead requeued %d waiting_human message(s)", requeued)
            _LAST_REQUEUED[lead_ref] = int(requeued)
        else:
            _LAST_REQUEUED.pop(lead_ref, None)
    except Exception as exc:
        # handoff_level is already cleared; a requeue failure must not be
        # reported as a failed resume, just logged for follow-up.
        logger.warning("resume_lead requeue failed: %s", exc)
    return True


# How many inbounds the last resume handed back to the queue, per lead. A
# resume that requeued something must not also send a proactive notice: the
# agent is about to answer the customer's own pending message, and two
# outbounds for one resume is exactly the duplication AGENTS.md 26 forbids.
_LAST_REQUEUED: dict[int, int] = {}

# What the last resume decided about speaking at all, per lead, so the notice
# does not have to reload the published graph to reach the same conclusion.
_LAST_RESUME_WINDOW: dict[int, dict] = {}


def _seconds_since_unanswered_customer_message(lead: dict) -> float | None:
    """Age of the customer's last message, or None if it was already answered.

    Only the tail matters: if the newest row is inbound, the customer spoke
    last and is still waiting. If it is outbound, there is nothing pending
    and the agent has no standing to open the conversation on its own.
    """
    try:
        rows = supabase_client.get_messages(str(lead.get("id")), limit=5) or []
    except Exception as exc:
        logger.warning("resume window message lookup failed: %s", exc)
        return None
    if not rows or str(rows[-1].get("direction") or "") != "inbound":
        return None
    raw = str(rows[-1].get("created_at") or "").strip()
    if not raw:
        return None
    try:
        sent_at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if sent_at.tzinfo is None:
        sent_at = sent_at.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - sent_at).total_seconds()


def resume_answer_window(lead: dict) -> dict:
    """Whether a resumed agent may still speak, per the published window.

    The agent speaks on a resume only while the customer's own unanswered
    message is recent enough to make a reply feel like a continuation of the
    same conversation. Past that, the turn belongs to the customer: the agent
    waits to be addressed. A campaign dispatch or a brand-new conversation is
    a different door and is unaffected by this.
    """
    if not lead:
        return {"may_speak": False, "reason": "lead_unavailable"}
    published = _reactivation_policy(lead)
    try:
        window = int(
            published.get("answer_pending_inbound_within_seconds")
            or RESUME_ANSWER_WINDOW_SECONDS
        )
    except (TypeError, ValueError):
        window = RESUME_ANSWER_WINDOW_SECONDS
    age = _seconds_since_unanswered_customer_message(lead)
    if age is None:
        return {
            "may_speak": False,
            "reason": "no_unanswered_customer_message",
            "window_seconds": window,
        }
    if age > window:
        return {
            "may_speak": False,
            "reason": "resume_window_expired",
            "window_seconds": window,
            "age_seconds": round(age),
        }
    return {
        "may_speak": True,
        "reason": "unanswered_customer_message_within_window",
        "window_seconds": window,
        "age_seconds": round(age),
    }


def _reactivation_policy(lead: dict) -> dict:
    """The persona's published reactivation block: copy and timing, one place."""
    from services import context_cards, graph_agent_runtime_v3

    persona_slug = str(lead.get("persona_slug") or "")
    if not persona_slug:
        try:
            persona = supabase_client.get_persona_by_id(str(lead.get("persona_id") or ""))
        except Exception as exc:
            logger.warning("reactivation policy persona lookup failed: %s", exc)
            return {}
        persona_slug = str((persona or {}).get("slug") or "")
    if not persona_slug:
        return {}
    try:
        _version, _checksum, graph = context_cards.current_graph(persona_slug)
    except Exception as exc:
        logger.warning("reactivation policy graph load failed: %s", exc)
        return {}
    document = {"nodes": [node.model_dump(mode="json") for node in graph.nodes]}
    persona_node = graph_agent_runtime_v3._persona_node(document) or {}
    policy = ((persona_node.get("data") or {}).get("conversation_policy") or {})
    return policy.get("reactivation") or {}


def reactivation_notice(lead_ref: int, *, reason: str) -> dict:
    """Announce, once, that the AI is back — when nothing else will speak.

    `reason` mirrors the `by` already carried by the `lead.ai_resumed` event
    ("manual", "journey_closed"). Copy is published in the persona graph under
    `conversation_policy.reactivation`; a persona that publishes none stays
    silent instead of receiving runtime-authored text.
    """
    from services import whatsapp_outbox

    window = _LAST_RESUME_WINDOW.pop(lead_ref, None)
    if _LAST_REQUEUED.pop(lead_ref, 0):
        return {"sent": False, "skipped": "pending_inbound_will_be_answered"}
    try:
        lead = supabase_client.get_lead_by_ref(lead_ref)
    except Exception as exc:
        # Courtesy message only. The AI is already resumed and the customer is
        # not blocked by this, so a lookup failure must never fail the resume.
        logger.warning("reactivation_notice lead lookup failed: %s", exc)
        return {"sent": False, "skipped": "lead_lookup_failed"}
    if not lead:
        return {"sent": False, "skipped": "lead_not_found"}
    if str(lead.get("handoff_level") or "none") != "none":
        return {"sent": False, "skipped": "human_owns_the_conversation"}
    # The same window the resume itself obeyed: reusing the decision avoids a
    # second graph load, and recomputing it covers a notice that did not come
    # straight from a resume.
    if window is None:
        window = resume_answer_window(lead)
    # A recent unanswered inbound is the turn that must speak. This check is
    # deliberately independent of the requeue count: audio can still be in
    # media/transcription processing and therefore not yet be claimable as a
    # waiting_human row. Enqueuing a courtesy notice in that gap creates a
    # second outbound before the canonical inbound decision exists.
    if window.get("reason") == "unanswered_customer_message_within_window":
        return {"sent": False, "skipped": "pending_inbound_will_be_answered"}
    if not window.get("may_speak"):
        return {
            "sent": False,
            "skipped": str(window.get("reason") or "resume_window_expired"),
        }

    try:
        text = _reactivation_text(lead, reason=reason)
    except Exception as exc:
        logger.warning("reactivation_notice copy lookup failed: %s", exc)
        return {"sent": False, "skipped": "copy_lookup_failed"}
    if not text:
        return {"sent": False, "skipped": "no_published_copy"}

    # One notice per resume, not per click: the key is the resume itself.
    resumed_at = str(lead.get("updated_at") or "")[:19]
    message_id = f"reactivation:{lead_ref}:{resumed_at}"
    try:
        result = whatsapp_outbox.enqueue_outbound(
            lead=lead, text=text, sender_type="agent",
            message_id=message_id, correlation_id=message_id,
            idempotency_key=message_id,
            metadata={"reactivation_reason": reason, "automatic": True},
        )
    except Exception as exc:
        # The AI is already resumed and the customer is not blocked by this.
        logger.warning("reactivation_notice enqueue failed: %s", exc)
        return {"sent": False, "skipped": "enqueue_failed"}
    return {"sent": not result.get("deduplicated"), "message_id": message_id}


def _reactivation_text(lead: dict, *, reason: str) -> str:
    """Pick the published opening that matches why the AI came back."""
    from services import graph_agent_runtime_v3

    published = _reactivation_policy(lead)
    variants = published.get(_reactivation_key(lead, reason=reason)) or []
    return graph_agent_runtime_v3._unrepeated_variant(
        [str(value) for value in variants], _recent_agent_texts(lead)
    )


def _reactivation_key(lead: dict, *, reason: str) -> str:
    if reason != "journey_closed":
        return "manual"
    outcome = str((lead.get("metadata") or {}).get("journey_outcome") or "")
    return "journey_cancelled" if "cancel" in outcome else "journey_completed"


def _recent_agent_texts(lead: dict) -> list[str]:
    """So a second resume does not replay the first notice word for word."""
    try:
        messages = supabase_client.get_messages(str(lead.get("id")), limit=6) or []
    except Exception:
        return []
    return [
        text for row in messages
        if str(row.get("direction") or "") == "outbound"
        if (text := str(row.get("texto") or row.get("content") or "").strip())
    ]
