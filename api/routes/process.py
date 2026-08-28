import hmac
import json
import logging
import os
import re
import time
from fastapi import APIRouter, Header, HTTPException
from schemas.events import LeadEvent
from core import context_builder, classifier, decision_engine, insight_engine
from agents.sdr import SDRAgent
from agents.closer import CloserAgent
from services import agents_service, event_emitter, supabase_client
from services.deterministic_sdr import load_catalog
from pathlib import Path
from datetime import datetime, timezone

router = APIRouter(tags=["process"])
logger = logging.getLogger("process")

_AGENTS = {
    "SDR": SDRAgent,
    "CLOSER": CloserAgent,
}

# Map agents_service role â†’ legacy agent class key.
# Followup falls back to SDR until a FollowupAgent class exists.
_ROLE_TO_AGENT_KEY = {
    "sdr":      "SDR",
    "closer":   "CLOSER",
    "followup": "SDR",
}


@router.post("/process")
async def process(
    event: LeadEvent,
    x_webhook_token: str | None = Header(default=None, alias="X-Webhook-Token"),
):
    t0 = time.monotonic()
    correlation_id = f"classifier:{event.lead_ref or event.lead_id or int(time.time() * 1000)}"

    # â”€â”€ Gate 0: persona routing mode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # When persona.process_mode == 'n8n', the Brain AI delegates the reply
    # to n8n. We only persist the inbound message and resolve the lead so
    # the dashboard sees the conversation. n8n is responsible for replying.
    persona_routing: dict | None = None
    if event.persona_slug:
        try:
            persona_routing = supabase_client.get_persona_routing(event.persona_slug)
        except Exception as exc:
            logger.warning("get_persona_routing failed: %s", exc)
            persona_routing = None
    expected_webhook_token = (
        (persona_routing or {}).get("inbound_webhook_token")
        or (os.environ.get("AI_BRAIN_WEBHOOK_TOKEN") or "").strip()
    )
    is_production = (os.environ.get("ENVIRONMENT") or "").strip().lower() == "production"
    if is_production and not expected_webhook_token:
        raise HTTPException(503, "webhook token is not configured")
    if expected_webhook_token and not hmac.compare_digest(
        (x_webhook_token or "").encode("utf-8"), str(expected_webhook_token).encode("utf-8")
    ):
        raise HTTPException(401, "invalid webhook token")
    if persona_routing and persona_routing.get("process_mode") == "n8n":
        # Resolve/create lead bound to the persona, persist inbound message,
        # then hand control back to n8n.
        try:
            lead_row = supabase_client.ensure_lead_for_persona(
                lead_id=event.lead_id,
                lead_ref=event.lead_ref,
                persona_slug_or_id=event.persona_slug,
                nome=event.nome,
                stage=event.stage,
                canal=event.canal,
                mensagem=event.mensagem,
                interesse_produto=event.interesse_produto,
                cidade=event.cidade,
                cep=event.cep,
                whatsapp_phone_number_id=event.whatsapp_phone_number_id,
            ) or {}
        except Exception as exc:
            logger.warning("ensure_lead_for_persona (n8n mode) failed: %s", exc)
            lead_row = {}
        resolved_ref = lead_row.get("id") or event.lead_ref
        if event.mensagem and resolved_ref is not None:
            try:
                supabase_client.insert_message({
                    "lead_ref": resolved_ref,
                    "message_id": f"n8n_in_{int(time.time() * 1000)}_{resolved_ref}",
                    "sender_type": "client",
                    "role": "user",
                    "canal": event.canal or "whatsapp",
                    "texto": event.mensagem,
                    "direction": "inbound",
                    "nome": event.nome or lead_row.get("nome"),
                    "status": "received",
                    "whatsapp_phone_number_id": event.whatsapp_phone_number_id or lead_row.get("whatsapp_phone_number_id"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })
            except Exception as exc:
                logger.warning("insert_message (n8n mode) failed: %s", exc)
        latency_ms = int((time.monotonic() - t0) * 1000)
        return {
            "reply": None,
            "stage_update": event.stage or lead_row.get("stage") or "novo",
            "agent_used": "N8N_DELEGATED",
            "score": 0,
            "detected_fields": {},
            "latency_ms": latency_ms,
            "lead_ref": resolved_ref,
        }

    ctx = context_builder.build(event)

    # â”€â”€ Gate 1: lead.ai_paused â†’ operador humano cuidando, nÃ£o responde â”€â”€
    lead_data = supabase_client.get_lead_by_ref(ctx.lead.ref) if ctx.lead.ref else supabase_client.get_lead(event.lead_id)
    lead_data = lead_data or {}
    if lead_data.get("ai_paused"):
        latency_ms = int((time.monotonic() - t0) * 1000)
        return {
            "reply": None,
            "stage_update": ctx.lead.stage,
            "agent_used": "PAUSED",
            "score": 0,
            "detected_fields": {},
            "latency_ms": latency_ms,
        }

    try:
        event_emitter.emit(
            "classifier_started",
            entity_type="lead",
            entity_id=str(ctx.lead.ref or ctx.lead.id or event.lead_id or ""),
            persona_id=getattr(ctx, "persona_id", None),
            payload={
                "correlation_id": correlation_id,
                "lead_ref": ctx.lead.ref,
                "lead_id": ctx.lead.id,
                "persona_slug": event.persona_slug or getattr(ctx, "persona_slug", None),
                "stage": ctx.lead.stage,
                "message_preview": (ctx.mensagem or "")[:240],
            },
            source="process.classifier",
        )
        classification = classifier.classify(ctx)
        supabase_client.insert_agent_log({
            "lead_id": str(ctx.lead.ref or ctx.lead.id or event.lead_id or ""),
            "agent_type": "Classifier",
            "action": "[INFO] classifier completed",
            "decision": json.dumps(classification, ensure_ascii=False)[:500],
            "metadata": {
                "level": "INFO",
                "component": "Classifier",
                "message": "classifier completed",
                "ts": datetime.now(timezone.utc).isoformat(),
                "correlation_id": correlation_id,
                "persona_slug": event.persona_slug or getattr(ctx, "persona_slug", None),
                "route_hint": classification.get("route_hint"),
            },
            "input": {
                "lead_ref": ctx.lead.ref,
                "lead_id": ctx.lead.id,
                "message": ctx.mensagem,
            },
            "output": classification,
        })
        event_emitter.emit(
            "classifier_completed",
            entity_type="lead",
            entity_id=str(ctx.lead.ref or ctx.lead.id or event.lead_id or ""),
            persona_id=getattr(ctx, "persona_id", None),
            payload={
                "correlation_id": correlation_id,
                "lead_ref": ctx.lead.ref,
                "lead_id": ctx.lead.id,
                "persona_slug": event.persona_slug or getattr(ctx, "persona_slug", None),
                "intent": classification.get("intent"),
                "interest_level": classification.get("interest_level"),
                "urgency": classification.get("urgency"),
                "fit": classification.get("fit"),
                "route_hint": classification.get("route_hint"),
            },
            source="process.classifier",
        )
    except Exception as exc:
        logger.warning("classifier failed, using SDR defaults: %s", exc)
        classification = {
            "intent": "duvida_geral",
            "interest_level": "medio",
            "urgency": "baixa",
            "fit": "neutro",
            "objections": [],
            "summary": "classifier unavailable",
            "route_hint": "SDR",
        }
        supabase_client.insert_agent_log({
            "lead_id": str(ctx.lead.ref or ctx.lead.id or event.lead_id or ""),
            "agent_type": "Classifier",
            "action": f"[ERROR] classifier failed: {str(exc)[:160]}",
            "decision": str(exc)[:500],
            "metadata": {
                "level": "ERROR",
                "component": "Classifier",
                "message": f"classifier failed: {exc}",
                "traceback": str(exc),
                "ts": datetime.now(timezone.utc).isoformat(),
                "correlation_id": correlation_id,
                "defaults_used": True,
                "persona_slug": event.persona_slug or getattr(ctx, "persona_slug", None),
            },
            "input": {
                "lead_ref": ctx.lead.ref,
                "lead_id": ctx.lead.id,
                "message": ctx.mensagem,
            },
            "output": classification,
        })
        event_emitter.emit(
            "classifier_failed",
            entity_type="lead",
            entity_id=str(ctx.lead.ref or ctx.lead.id or event.lead_id or ""),
            persona_id=getattr(ctx, "persona_id", None),
            payload={
                "correlation_id": correlation_id,
                "lead_ref": ctx.lead.ref,
                "lead_id": ctx.lead.id,
                "persona_slug": event.persona_slug or getattr(ctx, "persona_slug", None),
                "error": str(exc),
                "defaults_used": True,
            },
            source="process.classifier",
        )
    ctx.classification = classification

    score, tags, funnel_stage = decision_engine.compute_score(ctx)
    ctx.score = score
    ctx.tags = tags
    ctx.funnel_stage = funnel_stage

    # â”€â”€ Gate 2: role assignment para essa persona+stage estÃ¡ em humano? â”€â”€
    # Se sim, pausa o lead e devolve sem rodar agente (handoff).
    handoff_reason: str | None = None
    try:
        agent_record, role = agents_service.resolve_for_stage(
            event.persona_slug or ctx.persona_slug, funnel_stage,
        )
    except Exception as exc:
        logger.warning("resolve_for_stage failed: %s", exc)
        agent_record, role = None, "sdr"

    deterministic_mode = bool(event.persona_slug and load_catalog(Path(__file__).resolve().parents[1].parent / "docs" / "sdr", event.persona_slug).products)
    if agent_record is None and event.persona_slug and not deterministic_mode:
        # Nenhum agente atribuÃ­do a esse role para essa persona â†’ humano.
        # Pausa o lead pra o /process nÃ£o voltar a tentar enquanto operador
        # responde via dashboard. Manual resume via /leads/{id}/resume-ai.
        if ctx.lead.ref:
            try:
                supabase_client.update_lead(ctx.lead.ref, {"handoff_level": "full"})
            except Exception as exc:
                logger.warning("auto-pause update_lead failed: %s", exc)
        try:
            event_emitter.emit(
                "lead.handoff",
                entity_type="lead",
                entity_id=str(ctx.lead.ref or ctx.lead.id),
                payload={
                    "role": role,
                    "stage": funnel_stage,
                    "persona_slug": event.persona_slug,
                    "reason": "no_agent_for_role",
                },
                source="process",
            )
        except Exception:
            pass
        handoff_reason = f"no agent for role={role}"

    if handoff_reason:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return {
            "reply": None,
            "stage_update": funnel_stage,
            "agent_used": "HUMAN_HANDOFF",
            "score": score,
            "detected_fields": {},
            "latency_ms": latency_ms,
            "handoff_reason": handoff_reason,
        }

    # DecisÃ£o de rota: usa role do assignment se disponÃ­vel; senÃ£o decision_engine.
    route = _ROLE_TO_AGENT_KEY.get(role, decision_engine.decide(ctx))
    ctx.route_hint = route

    agent_result: dict = {"reply": None, "agent": route, "detected_fields": {}}

    agent_cls = _AGENTS.get(route) or _AGENTS.get("SDR")
    if agent_cls:
        try:
            agent_result = await agent_cls().run(ctx)
        except Exception as e:
            print(f"[process] agent error: {e}")

    latency_ms = int((time.monotonic() - t0) * 1000)

    # Resolve lead_ref: Supabase row id when available, otherwise WA phone number as int
    resolved_lead_ref: int | None = ctx.lead.ref
    if resolved_lead_ref is None:
        try:
            digits = re.sub(r"\D", "", ctx.lead.id or "")
            if digits:
                resolved_lead_ref = int(digits)
        except Exception:
            pass

    # A guarded agent may send one acknowledgement while handing the lead to
    # an operator.  Pause atomically before the outbox can process another
    # inbound message, so no automated follow-up can leak past the handoff.
    if agent_result.get("handoff") and ctx.lead.ref:
        try:
            supabase_client.handoff_whatsapp_lead(ctx.lead.ref)
        except Exception as exc:
            logger.warning("guarded handoff failed (non-fatal): %s", exc)

    if agent_result.get("reply"):
        try:
            supabase_client.insert_message({
                "lead_ref": resolved_lead_ref,
                "message_id": f"ai_{int(time.time() * 1000)}_{resolved_lead_ref}",
                "sender_type": "agent",
                "canal": ctx.lead.canal,
                "texto": agent_result["reply"],
                "role": "assistant",
                "direction": "outbound",
                "Lead_Stage": funnel_stage,
                "nome": ctx.lead.nome,
                "whatsapp_phone_number_id": lead_data.get("whatsapp_phone_number_id"),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {"sdr_state": (agent_result.get("detected_fields") or {}).get("sdr_state")},
            })
        except Exception as exc:
            logger.warning("insert_message failed (non-fatal): %s", exc)

        if ctx.lead.ref:
            try:
                update = {"stage": funnel_stage, "ultima_mensagem": ctx.mensagem}
                df = agent_result.get("detected_fields", {}) or {}
                if df.get("produto"):
                    update["interesse_produto"] = df["produto"]
                if df.get("cidade"):
                    update["cidade"] = df["cidade"]
                if df.get("cep"):
                    update["cep"] = df["cep"]
                if df.get("nome"):
                    update["nome"] = df["nome"]
                if df.get("sdr_state"):
                    current_metadata = lead_data.get("metadata") or {}
                    update["metadata"] = {**current_metadata, "sdr_state": df["sdr_state"]}
                supabase_client.update_lead(ctx.lead.ref, update)
            except Exception as exc:
                logger.warning("update_lead failed (non-fatal): %s", exc)

    insight_engine.record(ctx, agent_result, latency_ms)

    return {
        "reply": agent_result.get("reply"),
        "stage_update": funnel_stage,
        "agent_used": route,
        "score": score,
        "detected_fields": agent_result.get("detected_fields", {}),
        "latency_ms": latency_ms,
    }

