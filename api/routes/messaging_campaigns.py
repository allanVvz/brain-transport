from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from services import auth_service, campaigns_service, supabase_client


router = APIRouter(prefix="/messaging", tags=["messaging-campaigns"])


class CampaignPreviewBody(BaseModel):
    persona_id: str
    name: str | None = None
    objective: str | None = None
    purpose: str
    campaign_kind: Literal["consent_request", "promotional"]
    import_batch_ids: list[str] = Field(min_length=1)
    audience_id: str
    provider: Literal["meta_cloud", "evolution_baileys"] = "meta_cloud"
    template_name: str | None = None
    template_language: str = "pt_BR"
    template_id: str | None = None
    send_mode: str | None = None
    message: str | None = None
    variables: dict[str, Any] = Field(default_factory=dict)
    assets: list[dict[str, Any]] = Field(default_factory=list)
    policy_overrides: dict[str, Any] = Field(default_factory=dict)


class CampaignCreateBody(CampaignPreviewBody):
    expected_revision: int = 0
    expected_preview_checksum: str = Field(min_length=16, max_length=100)
    idempotency_key: str = Field(min_length=8, max_length=200)
    reason: str = Field(min_length=3, max_length=500)


class CampaignStatusBody(BaseModel):
    expected_revision: int = Field(gt=0)
    idempotency_key: str = Field(min_length=8, max_length=200)
    reason: str = Field(min_length=3, max_length=500)


class CampaignSendBody(BaseModel):
    expected_revision: int = Field(gt=0)
    idempotency_key: str = Field(min_length=8, max_length=200)
    # Required for Meta sends, enforced in campaigns_service.send_campaign
    # (can't be conditionally required at the schema level since it depends
    # on the revision's provider, which lives in the DB).
    reason: str | None = Field(default=None, max_length=500)


class MessageTemplateCreateBody(BaseModel):
    persona_id: str
    provider: Literal["meta_cloud", "evolution_baileys"]
    template_key: str = Field(min_length=1, max_length=120)
    body: str = Field(min_length=1)
    meta_template_name: str | None = None
    meta_template_language: str = "pt_BR"
    meta_template_category: str = "MARKETING"


def _assert_campaign_access(request: Request, campaign_id: str, capability: str = "view") -> dict:
    detail = campaigns_service.get_campaign_detail(campaign_id)
    auth_service.assert_persona_capability(
        request,
        capability,
        persona_id=detail["campaign"].get("persona_id"),
    )
    return detail


@router.get("/campaigns")
def list_campaigns(request: Request, persona_id: str | None = Query(None)):
    user = auth_service.current_user(request)
    if persona_id:
        auth_service.assert_persona_capability(request, "view", persona_id=persona_id)
        return campaigns_service.list_campaigns(persona_id)
    if not auth_service.is_admin(user):
        rows: list[dict] = []
        for allowed_id in auth_service.allowed_persona_ids(request):
            rows.extend(campaigns_service.list_campaigns(allowed_id))
        rows.sort(key=lambda row: row.get("created_at") or "", reverse=True)
        return rows
    return campaigns_service.list_campaigns()


@router.post("/campaigns/preview")
def preview_campaign(body: CampaignPreviewBody, request: Request):
    auth_service.assert_persona_capability(request, "edit", persona_id=body.persona_id)
    return campaigns_service.preview_campaign(body.model_dump())


@router.post("/campaigns")
def create_campaign(body: CampaignCreateBody, request: Request):
    auth_service.assert_persona_capability(request, "edit", persona_id=body.persona_id)
    user = auth_service.current_user(request)
    return campaigns_service.create_campaign_draft(
        body.model_dump(),
        actor_user_id=user.get("id"),
    )


@router.get("/campaigns/{campaign_id}")
def get_campaign(campaign_id: str, request: Request):
    return _assert_campaign_access(request, campaign_id)


@router.post("/campaigns/{campaign_id}/pause")
def pause_campaign(campaign_id: str, body: CampaignStatusBody, request: Request):
    detail = _assert_campaign_access(request, campaign_id, "edit")
    if detail["campaign"].get("status") not in {"draft", "validated", "scheduled", "running", "paused"}:
        raise HTTPException(409, "Campanha nao pode ser pausada neste estado.")
    return campaigns_service.update_campaign_status(
        campaign_id,
        expected_revision=body.expected_revision,
        status="paused",
        idempotency_key=body.idempotency_key,
        reason=body.reason,
        actor_user_id=auth_service.current_user(request).get("id"),
    )


@router.post("/campaigns/{campaign_id}/cancel")
def cancel_campaign(campaign_id: str, body: CampaignStatusBody, request: Request):
    _assert_campaign_access(request, campaign_id, "edit")
    return campaigns_service.update_campaign_status(
        campaign_id,
        expected_revision=body.expected_revision,
        status="cancelled",
        idempotency_key=body.idempotency_key,
        reason=body.reason,
        actor_user_id=auth_service.current_user(request).get("id"),
    )


@router.post("/campaigns/{campaign_id}/send")
def send_campaign(campaign_id: str, body: CampaignSendBody, request: Request):
    detail = _assert_campaign_access(request, campaign_id, "edit")
    if detail["campaign"].get("status") not in {"draft", "validated", "running"}:
        raise HTTPException(409, "Campanha nao pode ser enviada neste estado.")
    return campaigns_service.send_campaign(
        campaign_id,
        expected_revision=body.expected_revision,
        idempotency_key=body.idempotency_key,
        reason=body.reason,
        actor_user_id=auth_service.current_user(request).get("id"),
    )


@router.get("/templates")
def list_templates(request: Request, persona_id: str = Query(...), provider: str = Query(...)):
    auth_service.assert_persona_capability(request, "view", persona_id=persona_id)
    return campaigns_service.list_message_templates(persona_id, provider)


@router.post("/templates")
def create_template(body: MessageTemplateCreateBody, request: Request):
    auth_service.assert_persona_capability(request, "edit", persona_id=body.persona_id)
    user = auth_service.current_user(request)
    return campaigns_service.create_message_template(body.model_dump(), actor_user_id=user.get("id"))


@router.get("/provider-health")
def provider_health(request: Request, persona_id: str = Query(...)):
    auth_service.assert_persona_capability(request, "view", persona_id=persona_id)
    persona = supabase_client.get_persona_by_id(persona_id)
    rollout_enabled = campaigns_service.rollout_one_enabled(persona)
    binding = supabase_client.get_active_whatsapp_binding(persona_id)
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

