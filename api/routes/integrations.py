from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from services import auth_service, integration_service, supabase_client

router = APIRouter(prefix="/integrations", tags=["integrations"])


class IntegrationCredentialsBody(BaseModel):
    enabled: Optional[bool] = None
    service_account_json: Optional[Any] = None
    spreadsheet_id: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    endpoint: Optional[str] = None
    reply_source: Optional[str] = None
    base_id: Optional[str] = None
    # Meta (WhatsApp Business catalog)
    access_token: Optional[str] = None
    business_id: Optional[str] = None
    catalog_id: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class IntegrationValidateBody(BaseModel):
    service_account_json: Optional[Any] = None
    spreadsheet_id: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    endpoint: Optional[str] = None
    reply_source: Optional[str] = None
    base_id: Optional[str] = None
    access_token: Optional[str] = None
    business_id: Optional[str] = None
    catalog_id: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class WhatsAppBindingBody(BaseModel):
    phone_number_id: str
    whatsapp_number: str | None = None
    workflow_name: str = "WA â€” Outbound Sender"
    n8n_workflow_id: str | None = None
    business_id: str | None = None
    waba_id: str | None = None
    verified_name: str | None = None
    webhook_url: str | None = None
    mode: str = "disabled"
    allowlist: list[str] = []
    agent_id: str | None = None
    conversation_mode: str | None = None


def _current_user_id(request: Request) -> str:
    return auth_service.current_user(request).get("id") or ""


def _persona_or_404(slug: str, request: Request) -> dict[str, Any]:
    persona = supabase_client.get_persona(slug)
    if not persona:
        raise HTTPException(404, "Persona nao encontrada")
    auth_service.assert_persona_access(request, persona_id=persona["id"], persona_slug=slug)
    return persona


def _require_internal_admin(request: Request) -> dict[str, Any]:
    user = auth_service.current_user(request)
    if not auth_service.is_admin(user):
        raise HTTPException(403, "Configuracao de mensageria disponivel apenas para administradores.")
    return user


def _public_binding(binding: dict[str, Any] | None) -> dict[str, Any] | None:
    """Expose routing state without ever serializing integration secrets."""
    if not binding:
        return None
    metadata = binding.get("metadata") or {}
    return {
        "id": binding.get("id"), "persona_id": binding.get("persona_id"),
        "workflow_name": binding.get("workflow_name"), "n8n_workflow_id": binding.get("n8n_workflow_id"),
        "channel": binding.get("channel"),
        "provider": binding.get("provider"),
        "connection_status": binding.get("connection_status"),
        "whatsapp_number": binding.get("whatsapp_number"),
        "whatsapp_phone_number_id": binding.get("whatsapp_phone_number_id"),
        "active": bool(binding.get("active")),
        "metadata": {
            key: metadata.get(key)
            for key in (
                "business_id",
                "waba_id",
                "verified_name",
                "mode",
                "allowlist",
                "agent_id",
                "conversation_mode",
                "decision_owner",
                "transport_mode",
                "pipeline_contract",
                "runtime_version",
                "model",
                "reply_source",
            )
            if metadata.get(key) is not None
        },
    }


def _to_payload(body: BaseModel) -> dict[str, Any]:
    return body.model_dump(exclude_none=True)


def _handle_validation_error(exc: Exception) -> None:
    raise HTTPException(
        status_code=400,
        detail={
            "status": "invalid_credentials",
            "message": str(exc),
        },
    ) from exc


@router.get("/catalog")
def integrations_catalog():
    return integration_service.list_catalog()


@router.get("")
def list_integrations(request: Request):
    return integration_service.list_user_integrations(_current_user_id(request))


@router.get("/user")
def list_user_integrations(request: Request):
    return integration_service.list_user_integrations(_current_user_id(request))


@router.put("/user/{service}")
def upsert_user_integration(service: str, body: IntegrationCredentialsBody, request: Request):
    try:
        return integration_service.save_user_integration(
            _current_user_id(request),
            service,
            enabled=bool(body.enabled),
            credentials=_to_payload(body),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown integration: {service}") from exc
    except integration_service.IntegrationValidationError as exc:
        _handle_validation_error(exc)


@router.post("/user/{service}/validate")
def validate_user_integration(service: str, request: Request, body: Optional[IntegrationValidateBody] = None):
    try:
        payload = _to_payload(body or IntegrationValidateBody())
        return integration_service.validate_user_integration(
            _current_user_id(request),
            service,
            credentials=payload or None,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown integration: {service}") from exc
    except integration_service.IntegrationValidationError as exc:
        _handle_validation_error(exc)


@router.delete("/user/{service}/credentials")
def delete_user_integration_credentials(service: str, request: Request):
    try:
        return integration_service.delete_user_credentials(_current_user_id(request), service)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown integration: {service}") from exc


@router.get("/personas/{slug}")
def list_persona_integrations(slug: str, request: Request):
    _require_internal_admin(request)
    persona = _persona_or_404(slug, request)
    return integration_service.list_persona_integrations(persona["id"])


@router.put("/personas/{slug}/{service}")
def upsert_persona_integration(
    slug: str,
    service: str,
    body: IntegrationCredentialsBody,
    request: Request,
):
    user = _require_internal_admin(request)
    persona = _persona_or_404(slug, request)
    try:
        return integration_service.save_persona_integration(
            persona_id=persona["id"],
            actor_user_id=user.get("id") or "",
            service=service,
            enabled=bool(body.enabled),
            credentials=_to_payload(body),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown integration: {service}") from exc
    except integration_service.IntegrationValidationError as exc:
        _handle_validation_error(exc)


@router.post("/personas/{slug}/{service}/validate")
def validate_persona_integration(
    slug: str,
    service: str,
    request: Request,
    body: Optional[IntegrationValidateBody] = None,
):
    user = _require_internal_admin(request)
    persona = _persona_or_404(slug, request)
    try:
        return integration_service.validate_persona_integration(
            persona_id=persona["id"],
            actor_user_id=user.get("id") or "",
            service=service,
            credentials=_to_payload(body or IntegrationValidateBody()) or None,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown integration: {service}") from exc
    except integration_service.IntegrationValidationError as exc:
        _handle_validation_error(exc)


@router.delete("/personas/{slug}/{service}/credentials")
def delete_persona_integration_credentials(
    slug: str,
    service: str,
    request: Request,
):
    user = _require_internal_admin(request)
    persona = _persona_or_404(slug, request)
    try:
        return integration_service.delete_persona_credentials(
            persona_id=persona["id"],
            actor_user_id=user.get("id") or "",
            service=service,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown integration: {service}") from exc


@router.get("/meta/whatsapp/personas/{slug}")
def get_whatsapp_binding(slug: str, request: Request):
    _require_internal_admin(request)
    persona = _persona_or_404(slug, request)
    binding = next((
        b for b in supabase_client.get_workflow_bindings(persona["id"])
        if b.get("provider") == "meta_cloud"
    ), None)
    connection = supabase_client.get_persona_integration_connection(persona["id"], "meta_whatsapp") or {}
    return {"persona": {"id": persona["id"], "slug": persona["slug"]}, "binding": _public_binding(binding), "meta_configured": bool(connection.get("secret_ciphertext"))}


@router.put("/meta/whatsapp/personas/{slug}/binding")
def put_whatsapp_binding(slug: str, body: WhatsAppBindingBody, request: Request):
    _require_internal_admin(request)
    persona = _persona_or_404(slug, request)
    phone_number_id = body.phone_number_id.strip()
    if not phone_number_id:
        raise HTTPException(400, "phone_number_id obrigatorio")
    if body.mode not in {"disabled", "test_allowlist", "active"}:
        raise HTTPException(400, "mode invalido")
    allowlist = [item.strip() for item in body.allowlist if item and item.strip()]
    if body.mode == "test_allowlist" and not allowlist:
        raise HTTPException(400, "test_allowlist exige allowlist")
    if (
        body.conversation_mode not in {None, "deterministic"}
        or (body.webhook_url or "").strip()
        or (body.n8n_workflow_id or "").strip()
    ):
        raise HTTPException(400, "Mensageria direta nao aceita n8n_adapter ou webhooks n8n.")
    connection = (
        supabase_client.get_persona_integration_connection(persona["id"], "meta_whatsapp")
        or {}
    )
    provider_secret_ciphertext = connection.get("secret_ciphertext")
    if body.mode != "disabled" and not provider_secret_ciphertext:
        raise HTTPException(409, "Token Meta nao configurado para este binding.")

    # A provider number is a routing boundary, not reusable setup data.  Do
    # this check before upsert/activation so a new persona cannot create a
    # shadow binding for another persona's number.  Moving an existing number
    # remains an explicit reassignment operation with its own pause/audit
    # contract.
    foreign_binding = next(
        (
            binding
            for binding in supabase_client.get_workflow_bindings_by_phone_number_id(phone_number_id)
            if str(binding.get("persona_id") or "") != str(persona["id"])
        ),
        None,
    )
    if foreign_binding:
        raise HTTPException(
            409,
            "phone_number_id ja pertence a outra persona; use a reassociacao auditada do binding.",
        )
    metadata = {
        "business_id": body.business_id,
        "waba_id": body.waba_id,
        "verified_name": body.verified_name,
        "mode": body.mode,
        "allowlist": allowlist,
        "agent_id": body.agent_id,
        "conversation_mode": "deterministic",
        "decision_owner": "deterministic",
        "transport_mode": "provider_direct",
        "pipeline_contract": "conversation_v1",
    }
    existing = next((
        row for row in supabase_client.get_workflow_bindings(persona["id"])
        if row.get("provider") == "meta_cloud"
    ), None)
    try:
        binding = supabase_client.upsert_workflow_binding({
            "persona_id": persona["id"],
            "workflow_name": body.workflow_name,
            "n8n_workflow_id": None,
            "whatsapp_number": body.whatsapp_number,
            "whatsapp_phone_number_id": phone_number_id,
            "channel": "whatsapp",
            "provider": "meta_cloud",
            "provider_secret_ciphertext": provider_secret_ciphertext,
            "connection_status": (
                "connected" if body.mode != "disabled" else "disabled"
            ),
            "active": bool(existing and existing.get("active") and body.mode != "disabled"),
            "metadata": metadata,
        })
    except Exception as exc:
        raise HTTPException(409, "phone_number_id ja possui binding ativo") from exc
    result = {}
    if body.mode != "disabled":
        result = supabase_client.activate_whatsapp_binding(
            persona_id=persona["id"],
            binding_id=binding["id"],
            provider="meta_cloud",
            source="admin.settings.meta",
        )
        supabase_client.update_persona_routing(
            slug,
            {
                "process_mode": "internal",
                "outbound_webhook_url": None,
                "outbound_webhook_secret": None,
            },
        )
        binding = {
            **binding,
            "active": True,
            "connection_status": "connected",
        }
    supabase_client.insert_event({
        "event_type": "whatsapp.binding_updated",
        "entity_type": "workflow_binding",
        "entity_id": binding.get("id") or body.phone_number_id,
        "persona_id": persona["id"],
        "payload": {
            "mode": body.mode,
            "provider": "meta_cloud",
            "binding_id": binding.get("id"),
            "rebound_leads": result.get("rebound_leads", 0),
        },
    }, source="integrations.whatsapp")
    return {**(_public_binding(binding) or {}), "rebound_leads": result.get("rebound_leads", 0)}


@router.post("/meta/whatsapp/personas/{slug}/validate")
def validate_whatsapp_binding(slug: str, request: Request):
    _require_internal_admin(request)
    persona = _persona_or_404(slug, request)
    binding = next((b for b in supabase_client.get_workflow_bindings(persona["id"]) if b.get("active")), None)
    if not binding:
        raise HTTPException(400, "Binding WhatsApp nao configurado")
    if not binding.get("provider_secret_ciphertext"):
        raise HTTPException(400, "Token Meta nao configurado")
    return {"ok": True, "phone_number_id": binding.get("whatsapp_phone_number_id"), "token_masked": True}


@router.post("/meta/whatsapp/personas/{slug}/test")
def test_whatsapp_webhook(slug: str, request: Request):
    _require_internal_admin(request)
    persona = _persona_or_404(slug, request)
    binding = next((b for b in supabase_client.get_workflow_bindings(persona["id"]) if b.get("active")), None)
    if not binding:
        raise HTTPException(400, "Binding WhatsApp nao configurado")
    supabase_client.insert_event({"event_type": "whatsapp.webhook_test_requested", "entity_type": "workflow_binding", "entity_id": binding.get("id") or "unknown", "persona_id": persona["id"], "payload": {"phone_number_id": binding.get("whatsapp_phone_number_id")}}, source="integrations.whatsapp")
    return {"ok": True, "synthetic": True, "correlation_id": f"wa-test:{binding.get('id')}"}

