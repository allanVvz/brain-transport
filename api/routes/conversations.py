"""Internal, token-authenticated conversation steps orchestrated by n8n."""
from __future__ import annotations

import hmac
import os
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import Field, field_validator

from schemas.conversation import (
    AgentResponse,
    ConversationContext,
    ConversationDecision,
    StrictModel,
)
from services import conversation_runtime


router = APIRouter(prefix="/internal/conversations", tags=["conversations"])


def _authorize(token: str | None) -> None:
    expected = (os.environ.get("AI_BRAIN_WEBHOOK_TOKEN") or "").strip()
    if not expected:
        raise HTTPException(503, "internal webhook token is not configured")
    if expected and not hmac.compare_digest(
        (token or "").encode("utf-8"),
        expected.encode("utf-8"),
    ):
        raise HTTPException(401, "invalid webhook token")


class ContextRequest(StrictModel):
    persona_slug: str
    lead_ref: int
    message: str
    message_id: str | None = None
    # Turn/trace id for observability -- the same lead_buffer.id already
    # sent as inbound_buffer_id to /commit, forwarded here too so every
    # step of a turn (context/decide/commit) logs under one shared id.
    # Optional so a not-yet-updated n8n workflow doesn't hard-fail.
    trace_id: str | None = None

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("message must not be blank")
        return normalized


class DecisionRequest(StrictModel):
    context: ConversationContext
    model_observation: dict | None = None
    trace_id: str | None = None
    # ConversationContext carries no lead identity of its own (by design --
    # /decide reasons only from context + model_observation) -- forwarded
    # separately, purely for observability logging (lead_id column).
    lead_ref: int | None = None


class CommitRequest(StrictModel):
    lead_ref: int
    context: ConversationContext
    decision: ConversationDecision
    response: AgentResponse
    correlation_id: str
    phone_number_id: str | None = None
    channel_binding_id: str
    inbound_buffer_id: str
    n8n_execution_id: str | None = None


class FailSafeHandoffRequest(StrictModel):
    lead_ref: int
    reason: str
    correlation_id: str
    diagnostic: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None


class TechnicalFailureRequest(StrictModel):
    lead_ref: int
    buffer_id: str
    reason: str
    correlation_id: str
    diagnostic: dict[str, Any] = Field(default_factory=dict)


@router.post("/context", response_model=ConversationContext)
def context(
    body: ContextRequest,
    x_webhook_token: str | None = Header(None, alias="X-Webhook-Token"),
) -> ConversationContext:
    _authorize(x_webhook_token)
    try:
        return conversation_runtime.build_context(**body.model_dump())
    except conversation_runtime.PublishedGraphUnavailable as exc:
        raise HTTPException(409, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc


@router.post("/decide")
def decide(
    body: DecisionRequest,
    x_webhook_token: str | None = Header(None, alias="X-Webhook-Token"),
) -> dict:
    _authorize(x_webhook_token)
    decision, response = conversation_runtime.decide(
        body.context,
        model_observation=body.model_observation,
        trace_id=body.trace_id,
        lead_ref=body.lead_ref,
    )
    return {
        "decision": decision.model_dump(mode="json"),
        "response": response.model_dump(mode="json"),
    }


@router.post("/commit")
def commit(
    body: CommitRequest,
    x_webhook_token: str | None = Header(None, alias="X-Webhook-Token"),
) -> dict:
    _authorize(x_webhook_token)
    try:
        result = conversation_runtime.commit(
            lead_ref=body.lead_ref,
            context=body.context,
            decision=body.decision,
            response=body.response,
            correlation_id=body.correlation_id,
            phone_number_id=body.phone_number_id,
            channel_binding_id=body.channel_binding_id,
            inbound_buffer_id=body.inbound_buffer_id,
            expected_decision_owner="n8n_agents",
            n8n_execution_id=body.n8n_execution_id,
        )
        return conversation_runtime.dispatch_result_envelope(
            result, correlation_id=body.correlation_id
        )
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except conversation_runtime.ConversationCommitFailed as exc:
        raise HTTPException(409, detail=exc.canonical_result()) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/fail-safe-handoff")
def fail_safe_handoff(
    body: FailSafeHandoffRequest,
    x_webhook_token: str | None = Header(None, alias="X-Webhook-Token"),
) -> dict:
    _authorize(x_webhook_token)
    lead = conversation_runtime.supabase_client.get_lead_by_ref(body.lead_ref) or {}
    conversation_runtime.supabase_client.handoff_whatsapp_lead(body.lead_ref)
    if body.diagnostic:
        conversation_runtime.supabase_client.insert_event(
            {
                "event_type": "n8n.workflow_step_failed",
                "entity_type": "lead",
                "entity_id": str(body.lead_ref),
                "persona_id": lead.get("persona_id"),
                "payload": {
                    "lead_ref": body.lead_ref,
                    "correlation_id": body.correlation_id,
                    "trace_id": body.trace_id,
                    **body.diagnostic,
                },
            },
            level="error",
            source="routes.conversations",
        )
    conversation_runtime.supabase_client.insert_event(
        {
            "event_type": "conversation.fail_safe_handoff",
            "entity_type": "lead",
            "entity_id": str(body.lead_ref),
            "persona_id": lead.get("persona_id"),
            "payload": {**body.model_dump(), "trace_id": body.trace_id},
        },
        level="error",
        source="routes.conversations",
    )
    conversation_runtime.emit_turn_event(
        agent_name="conversation.error",
        trace_id=body.trace_id,
        lead_ref=body.lead_ref,
        persona_id=lead.get("persona_id"),
        status="error",
        error_msg=body.reason[:1000],
        metadata={"conversation_id": body.lead_ref, "step": "fail_safe_handoff"},
    )
    return {"ok": True, "handoff": True, "ai_paused": True}


@router.post("/technical-failure")
def technical_failure(
    body: TechnicalFailureRequest,
    x_webhook_token: str | None = Header(None, alias="X-Webhook-Token"),
) -> dict:
    """Quarantine a failed turn without inventing a commercial handoff."""
    _authorize(x_webhook_token)
    lead = conversation_runtime.supabase_client.get_lead_by_ref(body.lead_ref) or {}
    conversation_runtime.supabase_client.complete_whatsapp_buffer(
        body.buffer_id,
        "dead_letter",
        error=body.reason[:1000],
    )
    conversation_runtime.supabase_client.insert_event(
        {
            "event_type": "conversation.technical_failure",
            "entity_type": "lead",
            "entity_id": str(body.lead_ref),
            "persona_id": lead.get("persona_id"),
            "payload": {**body.model_dump(), "trace_id": body.buffer_id},
        },
        level="error",
        source="routes.conversations",
    )
    conversation_runtime.emit_turn_event(
        agent_name="conversation.error",
        trace_id=body.buffer_id,
        lead_ref=body.lead_ref,
        persona_id=lead.get("persona_id"),
        status="error",
        error_msg=body.reason[:1000],
        metadata={"conversation_id": body.lead_ref, "step": "technical_failure"},
    )
    return {
        "ok": False,
        "technical_failure": True,
        "handoff": False,
        "ai_paused": bool(lead.get("ai_paused")),
    }

