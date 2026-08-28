from __future__ import annotations

import json
import hashlib
import hmac
import os
import time
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Iterator

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from services import (
    agents_service,
    auth_service,
    campaigns_service,
    context_cards as context_cards_service,
    event_emitter,
    journey_outcome,
    knowledge_graph,
    lead_qualification,
    secret_store,
    supabase_client,
    whatsapp_outbox,
)
from routes import agents as agents_routes
from routes.agents import JourneyEventBody, JourneyStateBody
from services.whatsapp_providers import get_provider

router = APIRouter(prefix="/portal", tags=["portal"])

PIPELINE_STAGES = [
    ("novo", "Novo"),
    ("nao_qualificado", "NÃ£o qualificado"),
    ("contatado", "Contatado"),
    ("engajado", "Engajado"),
    ("qualificado", "Qualificado"),
    ("oportunidade", "Oportunidade"),
    ("fechado", "Fechado"),
    ("perdido", "Perdido"),
]


class LeadPatchBody(BaseModel):
    stage: str | None = None
    nome: str | None = None
    email: str | None = None
    cidade: str | None = None
    cep: str | None = None
    notes: str | None = None
    tags: list[str] | None = None
    interesse_produto: str | None = None
    commercial_note: dict[str, str] | None = None


class WhatsAppProviderBody(BaseModel):
    provider: str
    confirmed: bool = False


class PortalCampaignPreviewBody(BaseModel):
    persona_slug: str
    name: str | None = None
    objective: str | None = None
    purpose: str
    campaign_kind: str
    import_batch_ids: list[str]
    audience_id: str
    provider: str = "meta_cloud"
    template_name: str | None = None
    template_language: str = "pt_BR"
    template_id: str | None = None
    send_mode: str | None = None
    message: str | None = None
    variables: dict[str, Any] = {}
    assets: list[dict[str, Any]] = []
    policy_overrides: dict[str, Any] = {}


class PortalCampaignCreateBody(PortalCampaignPreviewBody):
    name: str
    expected_revision: int = 0
    expected_preview_checksum: str
    idempotency_key: str
    reason: str


class PortalCampaignStatusBody(BaseModel):
    expected_revision: int
    idempotency_key: str
    reason: str


class PortalCampaignSendBody(BaseModel):
    expected_revision: int
    idempotency_key: str
    reason: str | None = None


class PortalTemplateCreateBody(BaseModel):
    persona_slug: str
    provider: str
    template_key: str
    body: str
    meta_template_name: str | None = None
    meta_template_language: str = "pt_BR"
    meta_template_category: str = "MARKETING"


def _require_internal_admin(request: Request) -> dict[str, Any]:
    user = auth_service.current_user(request)
    if not auth_service.is_admin(user):
        raise HTTPException(403, "Troca de provider disponivel apenas na administracao.")
    return user


def _direct_binding_metadata(binding: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = {
        **((binding or {}).get("metadata") or {}),
        "decision_owner": "deterministic",
        "conversation_mode": "deterministic",
        "transport_mode": "provider_direct",
        "pipeline_contract": "conversation_v1",
    }
    metadata.pop("outbound_webhook_url", None)
    metadata.pop("n8n_outbound_webhook_url", None)
    metadata.pop("conversation_webhook_url", None)
    return metadata


def _persona(slug: str, request: Request, capability: str = "view") -> dict[str, Any]:
    persona = supabase_client.get_persona(slug)
    if not persona:
        raise HTTPException(404, "Persona nao encontrada.")
    auth_service.assert_persona_capability(
        request, capability, persona_id=persona["id"], persona_slug=slug
    )
    return persona


def _lead(lead_id: int, persona_id: str) -> dict[str, Any]:
    row = supabase_client.get_lead_by_ref(lead_id) or {}
    if not row or row.get("persona_id") != persona_id:
        raise HTTPException(404, "Lead nao encontrado.")
    return row


def _binding_for_persona(persona_id: str) -> dict[str, Any] | None:
    return next(
        (
            row for row in supabase_client.get_workflow_bindings(persona_id)
            if row.get("active") and row.get("channel") == "whatsapp"
        ),
        None,
    )


def _whatsapp_bindings(persona_id: str) -> list[dict[str, Any]]:
    return [row for row in supabase_client.get_workflow_bindings(persona_id) if row.get("channel") == "whatsapp"]


def _evolution_webhook_target(
    binding_id: str,
    public_base: str,
) -> tuple[str, str]:
    secret = (
        os.environ.get("EVOLUTION_WEBHOOK_HMAC_SECRET")
        or os.environ.get("AI_BRAIN_AUTH_SECRET")
        or ""
    )
    if not secret:
        raise HTTPException(503, "Segredo do webhook Evolution nao configurado.")
    callback_token = auth_service._b64encode(
        hmac.new(secret.encode(), binding_id.encode(), hashlib.sha256).digest()
    )
    return (
        f"{public_base.rstrip('/')}/webhooks/evolution/{binding_id}",
        callback_token,
    )


@router.get("/conversations")
def conversations(request: Request, persona_slug: str = Query(...)):
    persona = _persona(persona_slug, request)
    rows = supabase_client.get_conversations(hours=720, persona_id=persona["id"])
    leads_by_ref = supabase_client.get_leads_by_refs([
        int(row["lead_ref"])
        for row in rows
        if row.get("lead_ref") is not None
    ])
    resumos = journey_outcome.summaries_for_leads(persona["id"], list(leads_by_ref))
    business_model = journey_outcome.business_models_for_personas(
        [persona["id"]]
    ).get(persona["id"], journey_outcome.SALES)
    decorated = []
    for row in rows:
        ref = int(row.get("lead_ref") or 0)
        lead = leads_by_ref.get(ref, {})
        extra = lead_qualification.decorate_lead(lead)
        if extra.get("validation", {}).get("is_validation"):
            continue
        decorated.append({
            **row,
            "qualification": extra.get("qualification") or {},
            "qualification_score": extra.get("qualification_score") or 0,
            "qualification_signals": extra.get("qualification_signals") or [],
            "validation": extra.get("validation") or {},
            **(resumos.get(ref) or {
                "journey_outcome": None, "lead_converted": False,
                "journey_is_open": False, "journey_sequence": 0,
            }),
            # O carimbo permanente vence a derivacao por `converted_at`.
            "lead_converted": journey_outcome.lead_converted(lead)
            or bool((resumos.get(ref) or {}).get("lead_converted")),
            "business_model": business_model,
        })
    return decorated


@router.get("/client-pages")
def client_pages(request: Request):
    """Return every authorized persona, including channel-onboarding states."""
    user = auth_service.current_user(request)
    access_rows = auth_service.allowed_access(request)
    pages = []
    for persona in auth_service.filter_personas_for_user(
        user, auth_service.get_auth_personas(), access_rows
    ):
        binding = _binding_for_persona(persona["id"])
        capability = (
            {
                "capabilities": {
                    "view": True,
                    "edit": True,
                    "manage": True,
                    "manage_members": True,
                }
            }
            if auth_service.is_admin(user)
            else auth_service.assert_persona_capability(
                request,
                "view",
                persona_id=persona["id"],
                persona_slug=persona["slug"],
            )
        )
        pages.append({
            "slug": persona["slug"], "name": persona["name"],
            "url": f"/clientes/{persona['slug']}/mensagens",
            "capabilities": capability.get("capabilities") or {},
            "channel": {
                "configured": bool(binding),
                "provider": (binding or {}).get("provider"),
                "status": (
                    (binding or {}).get("connection_status")
                    or ("connected" if binding and binding.get("active") else "disabled")
                ),
            },
        })
    return pages


@router.get("/conversations/{lead_id}/messages")
def conversation_messages(
    lead_id: int,
    request: Request,
    persona_slug: str = Query(...),
    limit: int = Query(50, ge=1, le=100),
    after: str | None = Query(None),
    before: str | None = Query(None),
):
    persona = _persona(persona_slug, request)
    lead = _lead(lead_id, persona["id"])
    if lead_qualification.is_validation_lead(lead):
        raise HTTPException(404, "Conversa nao encontrada.")
    # Reuse the authenticated message cursor contract; portal authorization
    # has already proved that this lead belongs to the requested persona.
    from routes.messages import _message_page
    return _message_page(lead_id, limit=limit, after=after, before=before)


@router.get("/knowledge/chat-context")
def knowledge_chat_context(
    request: Request,
    persona_slug: str = Query(...),
    lead_ref: int = Query(...),
    q: str | None = Query(None),
    response_message_id: str | None = Query(None),
    limit: int = Query(12, ge=1, le=50),
):
    """Expose the operator evidence inside the authorized client portal."""
    persona = _persona(persona_slug, request)
    _lead(lead_ref, persona["id"])
    context = knowledge_graph.get_chat_context(
        lead_ref=lead_ref,
        persona_id=persona["id"],
        user_text=q,
        limit=limit,
    )
    # Context cards only need the latest conversational window. Full proof
    # history is fetched explicitly through the paginated messages endpoint.
    messages = supabase_client.get_messages(str(lead_ref), limit=50)
    try:
        projection_nodes, _projection_edges = supabase_client.list_all_knowledge_graph(
            persona_id=str(persona["id"]),
            limit_nodes=5000,
        )
    except Exception:
        projection_nodes = list(context.get("nodes") or [])
    try:
        turn = context_cards_service.response_context(
            persona_slug=persona_slug,
            persona_id=str(persona["id"]),
            lead_ref=lead_ref,
            messages=messages,
            response_message_id=response_message_id,
            query=q or "",
            projection_nodes=projection_nodes,
            limit=limit,
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {**knowledge_graph.with_operator_context(context, limit=limit), **turn}


@router.post("/messages", status_code=202)
async def send_message(request: Request, persona_slug: str = Query(...)):
    content_type = request.headers.get("content-type") or ""
    media: dict[str, Any] | None = None
    raw_client_message_id = ""
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        lead_id = int(str(form.get("lead_id") or "0"))
        text = str(form.get("text") or "").strip()
        raw_client_message_id = str(form.get("client_message_id") or "").strip()
        upload = form.get("file")
        if upload and hasattr(upload, "read"):
            data = await upload.read()
            mime = str(getattr(upload, "content_type", "") or "application/octet-stream").lower()
            filename = re.sub(r"[^A-Za-z0-9._-]+", "-", str(getattr(upload, "filename", "") or "arquivo"))[:120]
            allowed = (
                mime.startswith("image/")
                or mime.startswith("audio/")
                or mime in {
                    "application/pdf",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "text/plain",
                }
            )
            if not allowed:
                raise HTTPException(415, "Tipo de arquivo nao permitido.")
            if len(data) > 25 * 1024 * 1024:
                raise HTTPException(413, "Arquivo excede 25 MiB.")
            if mime == "application/pdf" and not data.startswith(b"%PDF"):
                raise HTTPException(415, "Arquivo PDF invalido.")
            if mime.startswith("image/") and not data.startswith(
                (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF8", b"RIFF")
            ):
                raise HTTPException(415, "Imagem invalida.")
            media = {"data": data, "mime": mime, "filename": filename}
    else:
        body = await request.json()
        lead_id = int(body.get("lead_id") or 0)
        text = str(body.get("text") or "").strip()
        raw_client_message_id = str(body.get("client_message_id") or "").strip()

    persona = _persona(persona_slug, request, "edit")
    lead = _lead(lead_id, persona["id"])
    if not text and not media:
        raise HTTPException(400, "Mensagem vazia.")
    try:
        client_message_id = str(uuid.UUID(raw_client_message_id))
    except (ValueError, AttributeError) as exc:
        raise HTTPException(422, "client_message_id UUID e obrigatorio.") from exc

    binding = whatsapp_outbox.resolve_lead_binding(lead)

    message_id = f"manual:{client_message_id}"
    existing = supabase_client.get_whatsapp_buffer_by_idempotency(message_id)
    if existing:
        if (
            existing.get("lead_ref") != lead["id"]
            or existing.get("channel_binding_id") != binding["id"]
        ):
            raise HTTPException(409, "client_message_id ja pertence a outra mensagem.")
        return {
            "ok": True,
            "message_id": message_id,
            "status": existing.get("status") or "pending_send",
            "buffer_id": existing["id"],
            "deduplicated": True,
        }

    media_metadata = None
    outbox_media = None
    if media:
        bucket = "message-media"
        if not supabase_client.ensure_bucket(bucket, public=False):
            raise HTTPException(503, "Storage de mensagens indisponivel.")
        path = f"{persona['id']}/{lead['id']}/{uuid.uuid4().hex}-{media['filename']}"
        supabase_client.upload_to_storage(bucket, path, media["data"], media["mime"])
        media_metadata = {
            "bucket": bucket, "path": path, "mime": media["mime"],
            "filename": media["filename"],
        }
        outbox_media = dict(media_metadata)

    result = whatsapp_outbox.enqueue_outbound(
        lead=lead,
        text=text,
        sender_type="human",
        message_id=message_id,
        correlation_id=message_id,
        idempotency_key=message_id,
        metadata={
            "source": "portal",
            "media": media_metadata,
            "client_message_id": client_message_id,
        },
        media=outbox_media,
    )
    return {
        "ok": True,
        "message_id": result.get("message_id") or message_id,
        "status": result.get("status") or "pending_send",
        "buffer_id": result["buffer_id"],
        "deduplicated": bool(result.get("deduplicated")),
    }


@router.get("/leads")
def leads(request: Request, persona_slug: str = Query(...), limit: int = Query(500, le=2000)):
    persona = _persona(persona_slug, request)
    rows = supabase_client.get_leads_for_persona_ids([persona["id"]], limit=limit, offset=0)
    return journey_outcome.decorate_leads(
        lead_qualification.filter_validation_scope(rows, "exclude"), persona["id"],
    )


@router.get("/leads/{lead_id}")
def lead_detail(lead_id: int, request: Request, persona_slug: str = Query(...)):
    persona = _persona(persona_slug, request)
    lead = _lead(lead_id, persona["id"])
    if lead_qualification.is_validation_lead(lead):
        raise HTTPException(404, "Lead nao encontrada.")
    return journey_outcome.decorate_leads(
        [lead_qualification.decorate_lead(lead)], persona["id"],
    )[0]


@router.patch("/leads/{lead_id}")
def update_lead(lead_id: int, body: LeadPatchBody, request: Request, persona_slug: str = Query(...)):
    persona = _persona(persona_slug, request, "edit")
    lead = _lead(lead_id, persona["id"])
    excluded = {"notes", "tags", "commercial_note"}
    payload = {key: value for key, value in body.model_dump().items() if value is not None and key not in excluded}
    if body.stage and body.stage not in {item[0] for item in PIPELINE_STAGES}:
        raise HTTPException(400, "Etapa invalida.")
    metadata: dict[str, Any] | None = None
    if body.stage == "perdido" and lead.get("stage") != "perdido":
        # Freeze the score right before the lead goes cold -- a historical
        # high-water mark of how far it got, not recomputed afterward and
        # never overwritten by a later re-PATCH while already "perdido".
        metadata = dict(lead.get("metadata") or {})
        qualification = dict(metadata.get("qualification") or {})
        qualification["score_at_perdido"] = lead_qualification.score_for_display(lead)
        metadata["qualification"] = qualification
    if body.notes is not None or body.tags is not None:
        metadata = dict(metadata if metadata is not None else (lead.get("metadata") or {}))
        portal = dict(metadata.get("portal") or {})
        if body.notes is not None:
            portal["notes"] = body.notes
        if body.tags is not None:
            portal["tags"] = body.tags
        metadata["portal"] = portal
    if body.commercial_note is not None:
        metadata = supabase_client.merge_commercial_note(
            metadata if metadata is not None else (lead.get("metadata") or {}),
            body.commercial_note,
        )
    if metadata is not None:
        payload["metadata"] = metadata
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    rows = supabase_client.get_client().table("leads").update(payload).eq("id", lead_id).eq("persona_id", persona["id"]).execute().data or []
    return rows[0] if rows else {}


def _set_ai(lead_id: int, request: Request, persona_slug: str, paused: bool):
    persona = _persona(persona_slug, request, "edit")
    lead = _lead(lead_id, persona["id"])
    ok = agents_service.pause_lead(lead_id) if paused else agents_service.resume_lead(lead_id)
    if not ok:
        raise HTTPException(500, "Falha ao atualizar IA.")
    if not paused:
        metadata = dict(lead.get("metadata") or {})
        state = dict(metadata.get("vitoria_state") or {})
        state.pop("conversation_state", None)
        state.pop("confirmation_status", None)
        state["clarification_attempts"] = 0
        metadata["vitoria_state"] = state
        supabase_client.update_lead(lead_id, {"metadata": metadata})
    return {"ok": True, "lead_id": lead_id, "ai_paused": paused}


@router.post("/leads/{lead_id}/journey-events")
def journey_event(
    lead_id: int, body: JourneyEventBody, request: Request,
    persona_slug: str = Query(...),
):
    """Desfecho comercial do pedido, na variante do portal.

    A rota gemea em `/agents` nao serve o portal: o middleware de auth so
    libera `/portal/*`, `/auth/*` e a midia de asset para contas `client`,
    entao um operador do portal levava 403 antes de chegar a rota. Aqui o
    escopo tambem fica mais estreito -- `_lead` confirma que a lead pertence a
    persona da URL, checagem que a rota de `/agents` nao faz.
    """
    persona = _persona(persona_slug, request, "edit")
    _lead(lead_id, persona["id"])
    user = auth_service.current_user(request)
    return agents_routes.record_journey_event(lead_id, body, str(user["id"]))


@router.post("/leads/{lead_id}/journey-state")
def journey_state(
    lead_id: int, body: JourneyStateBody, request: Request,
    persona_slug: str = Query(...),
):
    """Estado do pedido escolhido pelo closer, na variante do portal.

    O `business_model` da persona decide o par de eventos por tras do alvo:
    produto e comprado e entregue, servico e agendado e concluido.
    """
    persona = _persona(persona_slug, request, "edit")
    _lead(lead_id, persona["id"])
    user = auth_service.current_user(request)
    offering = journey_outcome.business_models_for_personas(
        [persona["id"]]
    ).get(persona["id"], journey_outcome.SALES)
    return agents_routes.set_journey_state(
        lead_id, body, str(user["id"]), offering=offering,
    )


@router.post("/leads/{lead_id}/ai/pause")
def pause_ai(lead_id: int, request: Request, persona_slug: str = Query(...)):
    return _set_ai(lead_id, request, persona_slug, True)


@router.post("/leads/{lead_id}/ai/resume")
def resume_ai(lead_id: int, request: Request, persona_slug: str = Query(...)):
    return _set_ai(lead_id, request, persona_slug, False)


@router.post("/leads/{lead_id}/ai/acknowledge")
def acknowledge_handoff(lead_id: int, request: Request, persona_slug: str = Query(...)):
    """Clear a 'partial' handoff flag once a human has reviewed it.

    Unlike /ai/resume, a partial handoff never stopped the AI -- there's
    nothing to requeue, just the flag to clear.
    """
    persona = _persona(persona_slug, request, "edit")
    _lead(lead_id, persona["id"])
    ok = agents_service.acknowledge_partial_handoff(lead_id)
    if not ok:
        raise HTTPException(500, "Falha ao confirmar handoff.")
    return {"ok": True, "lead_id": lead_id, "handoff_level": "none"}


@router.post("/leads/{lead_id}/handoff")
def handoff(lead_id: int, request: Request, persona_slug: str = Query(...)):
    persona = _persona(persona_slug, request, "edit")
    _lead(lead_id, persona["id"])
    supabase_client.handoff_whatsapp_lead(lead_id)
    return {"ok": True, "lead_id": lead_id, "ai_paused": True, "handoff": True}


def _campaign_for_persona(campaign_id: str, persona_id: str) -> dict[str, Any]:
    detail = campaigns_service.get_campaign_detail(campaign_id)
    if str(detail["campaign"].get("persona_id")) != str(persona_id):
        raise HTTPException(404, "Campanha nao encontrada.")
    return detail


@router.get("/campaigns")
def portal_campaigns(request: Request, persona_slug: str = Query(...)):
    persona = _persona(persona_slug, request)
    return campaigns_service.list_campaigns(persona["id"])


@router.get("/campaigns/provider-health")
def portal_campaign_provider_health(request: Request, persona_slug: str = Query(...)):
    persona = _persona(persona_slug, request)
    rollout_enabled = campaigns_service.rollout_one_enabled(persona)
    binding = supabase_client.get_active_whatsapp_binding(persona["id"])
    mock_enabled = campaigns_service.meta_mock_enabled()
    if not binding and not mock_enabled:
        return {
            "provider": None, "ready": False, "status": "not_configured",
            "campaigns_enabled": False, "rollout_one_enabled": rollout_enabled,
        }
    provider = "meta_cloud" if mock_enabled else binding.get("provider")
    status = "mock" if mock_enabled else str(binding.get("connection_status") or "unknown").lower()
    ready = True if mock_enabled else campaigns_service.resolve_provider_ready(binding, provider)
    return {
        "provider": provider,
        "ready": ready,
        "status": status,
        "campaigns_enabled": rollout_enabled and provider in {"meta_cloud", "evolution_baileys"},
        "rollout_one_enabled": rollout_enabled,
        "mock": mock_enabled,
    }


@router.get("/lead-imports")
def portal_lead_imports(request: Request, persona_slug: str = Query(...), limit: int = Query(20, le=100)):
    persona = _persona(persona_slug, request)
    batches = supabase_client.list_lead_import_batches(persona["id"], limit=limit)
    batches.sort(key=lambda row: row.get("created_at") or "", reverse=True)
    return batches


@router.get("/audiences")
def portal_audiences(request: Request, persona_slug: str = Query(...)):
    persona = _persona(persona_slug, request)
    try:
        rows = supabase_client.list_persona_audiences(persona["id"]) or []
    except Exception:
        return []
    return rows


@router.post("/campaigns/preview")
def portal_campaign_preview(body: PortalCampaignPreviewBody, request: Request):
    persona = _persona(body.persona_slug, request, "edit")
    payload = {**body.model_dump(exclude={"persona_slug"}), "persona_id": persona["id"]}
    return campaigns_service.preview_campaign(payload)


@router.post("/campaigns")
def portal_create_campaign(body: PortalCampaignCreateBody, request: Request):
    persona = _persona(body.persona_slug, request, "edit")
    user = auth_service.current_user(request)
    payload = {**body.model_dump(exclude={"persona_slug"}), "persona_id": persona["id"]}
    return campaigns_service.create_campaign_draft(payload, actor_user_id=user.get("id"))


@router.post("/campaigns/{campaign_id}/pause")
def portal_pause_campaign(campaign_id: str, body: PortalCampaignStatusBody, request: Request, persona_slug: str = Query(...)):
    persona = _persona(persona_slug, request, "edit")
    detail = _campaign_for_persona(campaign_id, persona["id"])
    if detail["campaign"].get("status") not in {"draft", "validated", "scheduled", "running", "paused"}:
        raise HTTPException(409, "Campanha nao pode ser pausada neste estado.")
    return campaigns_service.update_campaign_status(
        campaign_id, expected_revision=body.expected_revision, status="paused",
        idempotency_key=body.idempotency_key, reason=body.reason,
        actor_user_id=auth_service.current_user(request).get("id"),
    )


@router.post("/campaigns/{campaign_id}/cancel")
def portal_cancel_campaign(campaign_id: str, body: PortalCampaignStatusBody, request: Request, persona_slug: str = Query(...)):
    persona = _persona(persona_slug, request, "edit")
    _campaign_for_persona(campaign_id, persona["id"])
    return campaigns_service.update_campaign_status(
        campaign_id, expected_revision=body.expected_revision, status="cancelled",
        idempotency_key=body.idempotency_key, reason=body.reason,
        actor_user_id=auth_service.current_user(request).get("id"),
    )


@router.post("/campaigns/{campaign_id}/send")
def portal_send_campaign(campaign_id: str, body: PortalCampaignSendBody, request: Request, persona_slug: str = Query(...)):
    persona = _persona(persona_slug, request, "edit")
    detail = _campaign_for_persona(campaign_id, persona["id"])
    if detail["campaign"].get("status") not in {"draft", "validated", "running"}:
        raise HTTPException(409, "Campanha nao pode ser enviada neste estado.")
    return campaigns_service.send_campaign(
        campaign_id, expected_revision=body.expected_revision,
        idempotency_key=body.idempotency_key, reason=body.reason,
        actor_user_id=auth_service.current_user(request).get("id"),
    )


@router.get("/templates")
def portal_list_templates(request: Request, persona_slug: str = Query(...), provider: str = Query(...)):
    persona = _persona(persona_slug, request)
    return campaigns_service.list_message_templates(persona["id"], provider)


@router.post("/templates")
def portal_create_template(body: PortalTemplateCreateBody, request: Request):
    persona = _persona(body.persona_slug, request, "edit")
    user = auth_service.current_user(request)
    payload = {**body.model_dump(exclude={"persona_slug"}), "persona_id": persona["id"]}
    return campaigns_service.create_message_template(payload, actor_user_id=user.get("id"))


@router.get("/pipeline")
def pipeline(request: Request, persona_slug: str = Query(...)):
    persona = _persona(persona_slug, request)
    rows = [
        lead_qualification.decorate_lead(row)
        for row in supabase_client.get_leads_for_persona_ids([persona["id"]], limit=2000, offset=0)
    ]
    portal_config = dict((persona.get("config") or {}).get("portal") or {})
    configured_labels = dict(portal_config.get("stage_labels") or {})
    conversion_stage = str(portal_config.get("conversion_stage") or "fechado")
    stages = [
        {
            "id": slug,
            "label": configured_labels.get(slug) or label,
            "leads": [row for row in rows if (row.get("stage") or "novo") == slug],
        }
        for slug, label in PIPELINE_STAGES
    ]
    return {
        "stages": stages,
        "summary": {
            "total": len(rows),
            "conversion_stage": conversion_stage,
            "converted": sum(
                1 for row in rows if (row.get("stage") or "novo") == conversion_stage
            ),
            "open": sum(
                1
                for row in rows
                if (row.get("stage") or "novo") not in {conversion_stage, "perdido"}
            ),
        },
        "business_model": portal_config.get("business_model") or "sales",
    }


@router.get("/personas/{slug}/channels/whatsapp")
def whatsapp_channel(slug: str, request: Request):
    persona = _persona(slug, request)
    can_manage_provider = auth_service.is_admin(auth_service.current_user(request))
    binding = _binding_for_persona(persona["id"])
    if not binding:
        return {
            "configured": False,
            "status": "disabled",
            "can_manage_provider": can_manage_provider,
            "available_providers": (
                ["meta_cloud", "evolution_baileys"]
                if can_manage_provider else []
            ),
        }
    return {
        "configured": True,
        "provider": binding.get("provider") or "meta_cloud",
        "status": binding.get("connection_status") or ("connected" if binding.get("active") else "disabled"),
        "last_connection_at": binding.get("last_connection_at"),
        "can_manage_provider": can_manage_provider,
        "available_providers": (
            ["meta_cloud", "evolution_baileys"]
            if can_manage_provider else []
        ),
    }


@router.post("/personas/{slug}/channels/whatsapp/provider")
def select_whatsapp_provider(slug: str, body: WhatsAppProviderBody, request: Request):
    """Switch transport only after an explicit confirmation.

    Secrets stay in the binding ciphertext.  The previous binding is retained,
    inactive, so a failed provider call can be rolled back safely.
    """
    _require_internal_admin(request)
    persona = _persona(slug, request, "manage")
    if body.provider not in {"meta_cloud", "evolution_baileys"}:
        raise HTTPException(400, "Provider invalido.")
    if not body.confirmed:
        raise HTTPException(400, "Confirme a troca do canal WhatsApp.")
    bindings = _whatsapp_bindings(persona["id"])
    current = next((item for item in bindings if item.get("active")), None)
    target = next((item for item in bindings if item.get("provider") == body.provider), None)

    if body.provider == "meta_cloud":
        if (
            not target
            or not target.get("whatsapp_phone_number_id")
            or not target.get("provider_secret_ciphertext")
        ):
            raise HTTPException(409, "Nao existe um binding Meta configurado para reativar.")
        supabase_client.update_workflow_binding_metadata(
            target["id"],
            _direct_binding_metadata(target),
        )
        if target.get("connection_status") not in {"connected", "open"}:
            raise HTTPException(409, "Binding Meta configurado, mas nao conectado.")
        result = supabase_client.activate_whatsapp_binding(
            persona_id=persona["id"],
            binding_id=target["id"],
            provider="meta_cloud",
            source="admin.settings",
        )
        supabase_client.update_persona_routing(
            slug,
            {
                "process_mode": "internal",
                "outbound_webhook_url": None,
                "outbound_webhook_secret": None,
            },
        )
        return {**result, "status": target.get("connection_status")}

    if (os.environ.get("EVOLUTION_ENABLED") or "").lower() not in {"1", "true", "yes"}:
        raise HTTPException(503, "Evolution API desabilitada.")
    needs_provision = not target or target.get("connection_status") in {
        None, "disabled", "disconnected", "failed",
    }
    if needs_provision:
        instance_key = f"brain-{persona['slug'][:30]}-{persona['id'][:8]}"
        instance_token = os.urandom(32).hex()
        public_base = (os.environ.get("AI_BRAIN_PUBLIC_API_URL") or "").rstrip("/")
        if not public_base:
            raise HTTPException(503, "AI_BRAIN_PUBLIC_API_URL nao configurada.")
        if target:
            instance_key = str(target.get("provider_instance_key") or instance_key)
            instance_token = (
                secret_store.decrypt_secret(target.get("provider_secret_ciphertext"))
                or instance_token
            )
            supabase_client.update_workflow_binding(
                target["id"],
                {
                    "provider_instance_key": instance_key,
                    "provider_secret_ciphertext": secret_store.encrypt_secret(instance_token),
                    "connection_status": "provisioning",
                    "active": False,
                },
            )
            supabase_client.update_workflow_binding_metadata(
                target["id"],
                _direct_binding_metadata(target),
            )
            target = {
                **target,
                "provider_instance_key": instance_key,
                "provider_secret_ciphertext": secret_store.encrypt_secret(instance_token),
                "connection_status": "provisioning",
                "active": False,
            }
        else:
            target = supabase_client.upsert_workflow_binding({
                "persona_id": persona["id"], "workflow_name": f"Evolution {persona['slug']}",
                "channel": "whatsapp", "provider": "evolution_baileys",
                "provider_instance_key": instance_key,
                "provider_secret_ciphertext": secret_store.encrypt_secret(instance_token),
                "connection_status": "provisioning", "active": False,
                "metadata": {
                    "provider_version": os.environ.get("EVOLUTION_API_VERSION", "2.3.7"),
                    **_direct_binding_metadata(),
                },
            })
        webhook_url, callback = _evolution_webhook_target(
            target["id"],
            public_base,
        )
        try:
            get_provider("evolution_baileys").provision_instance(
                instance_key,
                instance_token,
                webhook_url,
                webhook_token=callback,
            )
        except Exception:
            # A draft without a provisioned remote instance must not become active.
            supabase_client.update_workflow_binding(target["id"], {"active": False, "connection_status": "failed"})
            raise HTTPException(502, "Falha ao provisionar Evolution; o canal anterior foi preservado.")
        supabase_client.update_workflow_binding(
            target["id"],
            {"active": False, "connection_status": "connecting"},
        )
        target = {**target, "connection_status": "connecting", "active": False}

    supabase_client.update_workflow_binding_metadata(
        target["id"],
        _direct_binding_metadata(target),
    )
    result = supabase_client.activate_whatsapp_binding(
        persona_id=persona["id"],
        binding_id=target["id"],
        provider="evolution_baileys",
        source="admin.settings",
    )
    supabase_client.update_persona_routing(
        slug,
        {
            "process_mode": "internal",
            "outbound_webhook_url": None,
            "outbound_webhook_secret": None,
        },
    )
    return {
        **result,
        "provider": "evolution_baileys",
        "status": target.get("connection_status") or "connecting",
    }


@router.post("/personas/{slug}/channels/whatsapp/evolution/provision")
def provision_evolution(slug: str, request: Request):
    _require_internal_admin(request)
    return select_whatsapp_provider(
        slug,
        WhatsAppProviderBody(provider="evolution_baileys", confirmed=True),
        request,
    )


def _evolution_action(slug: str, request: Request, action: str):
    persona = _persona(slug, request, "manage")
    binding = _binding_for_persona(persona["id"])
    if not binding or binding.get("provider") != "evolution_baileys":
        raise HTTPException(404, "Canal Evolution nao configurado.")
    provider = get_provider("evolution_baileys")
    try:
        result = getattr(provider, action)(binding)
    except httpx.HTTPStatusError as exc:
        if action != "get_qr_code" or exc.response.status_code != 404 or not binding.get("active"):
            raise
        instance_key = str(binding.get("provider_instance_key") or "")
        instance_token = secret_store.decrypt_secret(binding.get("provider_secret_ciphertext"))
        public_base = (os.environ.get("AI_BRAIN_PUBLIC_API_URL") or "").rstrip("/")
        if not instance_key or not instance_token or not public_base:
            raise HTTPException(503, "Binding Evolution incompleto para reprovisionamento.") from exc
        webhook_url, callback = _evolution_webhook_target(
            binding["id"],
            public_base,
        )
        provider.provision_instance(
            instance_key,
            instance_token,
            webhook_url,
            webhook_token=callback,
        )
        supabase_client.update_workflow_binding(binding["id"], {"connection_status": "connecting"})
        result = provider.get_qr_code(binding)
    if action == "logout_instance":
        supabase_client.update_workflow_binding(binding["id"], {"connection_status": "disconnected"})
    elif action == "get_qr_code" and result.get("status") == "qr_ready":
        supabase_client.update_workflow_binding(binding["id"], {"connection_status": "qr_ready"})
    return result


@router.post("/personas/{slug}/channels/whatsapp/evolution/connect")
def connect_evolution(slug: str, request: Request):
    response = _evolution_action(slug, request, "get_qr_code")
    return JSONResponse(response, headers={"Cache-Control": "no-store"})


@router.post("/personas/{slug}/channels/whatsapp/evolution/restart")
def restart_evolution(slug: str, request: Request):
    _require_internal_admin(request)
    return _evolution_action(slug, request, "restart_instance")


@router.post("/personas/{slug}/channels/whatsapp/evolution/logout")
def logout_evolution(slug: str, request: Request):
    _require_internal_admin(request)
    return _evolution_action(slug, request, "logout_instance")


@router.post("/personas/{slug}/channels/whatsapp/evolution/revoke-and-reconnect")
def revoke_and_reconnect_evolution(slug: str, request: Request):
    """Revoke the device session and issue a fresh QR code (admin only)."""
    _require_internal_admin(request)
    _evolution_action(slug, request, "logout_instance")
    response = _evolution_action(slug, request, "get_qr_code")
    return JSONResponse(response, headers={"Cache-Control": "no-store"})


@router.get("/events/stream")
def events_stream(request: Request, persona_slug: str = Query(...)):
    persona = _persona(persona_slug, request)
    allowed = {
        "message.new", "lead.ai_paused", "lead.ai_resumed",
        "whatsapp.connection_changed", "lead.updated",
    }

    def generate() -> Iterator[str]:
        cursor = request.headers.get("last-event-id")
        while True:
            if await_disconnected(request):
                break
            rows = supabase_client.get_events(limit=50, persona_id=persona["id"])
            rows = list(reversed(rows))
            for row in rows:
                event_id = str(row.get("id") or "")
                if cursor and event_id == cursor:
                    cursor = None
                    continue
                if cursor:
                    continue
                if row.get("event_type") not in allowed:
                    continue
                safe = {
                    "type": row.get("event_type"),
                    "entity_id": row.get("entity_id"),
                    "created_at": row.get("created_at"),
                }
                yield f"id: {event_id}\nevent: update\ndata: {json.dumps(safe)}\n\n"
            yield ": heartbeat\n\n"
            time.sleep(10)

    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})


def await_disconnected(_request: Request) -> bool:
    # Sync generator cannot await Request.is_disconnected; failed writes close it.
    return False

