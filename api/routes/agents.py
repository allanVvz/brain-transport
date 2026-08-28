"""Authenticated SDR journey and commercial conversion contracts."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field, field_validator, model_validator

from routes.conversations import _authorize as authorize_internal
from services import agents_service, auth_service, event_emitter, supabase_client

router = APIRouter(prefix="/agents", tags=["agents"])
internal_router = APIRouter(prefix="/internal/agents", tags=["agents"])


class PurchaseCompletedBody(BaseModel):
    idempotency_key: str = Field(min_length=1, max_length=200)
    source: str = Field(min_length=1, max_length=100)
    occurred_at: datetime
    external_ref: str | None = Field(default=None, max_length=300)
    amount_minor: int | None = Field(default=None, ge=0)
    currency: str | None = None
    items: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if len(normalized) != 3 or not normalized.isalpha():
            raise ValueError("currency must be a three-letter ISO code")
        return normalized

    @model_validator(mode="after")
    def require_currency_for_amount(self) -> "PurchaseCompletedBody":
        if self.amount_minor is not None and self.currency is None:
            raise ValueError("currency is required when amount_minor is present")
        return self


class ConversionStatusBody(BaseModel):
    status: Literal["cancelled", "partially_refunded", "refunded"]
    idempotency_key: str = Field(min_length=1, max_length=200)
    refunded_amount_minor: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JourneyEventBody(BaseModel):
    event_type: Literal[
        "converted", "conversion_reverted", "sale_recorded", "appointment_booked",
        "delivered", "service_completed", "cancelled",
    ]
    idempotency_key: str = Field(min_length=1, max_length=200)
    source: str = Field(min_length=1, max_length=100)
    occurred_at: datetime
    external_ref: str | None = Field(default=None, max_length=300)
    amount_minor: int | None = Field(default=None, ge=0)
    currency: str | None = None
    items: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str | None) -> str | None:
        return PurchaseCompletedBody.validate_currency(value)

    @model_validator(mode="after")
    def require_currency_for_amount(self) -> "JourneyEventBody":
        if self.amount_minor is not None and self.currency is None:
            raise ValueError("currency is required when amount_minor is present")
        if self.event_type not in {"sale_recorded", "appointment_booked"} and (
            self.amount_minor is not None or self.items
        ):
            raise ValueError("commercial values are only accepted for conversion events")
        return self


class JourneyStateBody(BaseModel):
    """Estado-alvo do pedido, escolhido pelo closer humano.

    Sem `idempotency_key`: isto nao e um evento append-only, e a afirmacao de
    onde o pedido esta agora. Reenviar o mesmo alvo devolve `changed=false`.
    """
    target: Literal["qualificado", "convertido", "vendido", "entregue", "cancelado"]
    source: str = Field(min_length=1, max_length=100)
    occurred_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


def _lead_or_404(lead_ref: int) -> dict:
    lead = supabase_client.get_lead_by_ref(lead_ref) or {}
    if not lead or not lead.get("persona_id"):
        raise HTTPException(404, "Lead not found")
    return lead


def record_journey_event(
    lead_ref: int, body: JourneyEventBody, responsible_user_id: str | None,
) -> dict:
    lead = _lead_or_404(lead_ref)
    try:
        return supabase_client.record_conversation_journey_event(
            p_persona_id=lead["persona_id"], p_lead_ref=lead_ref,
            p_event_type=body.event_type,
            p_idempotency_key=body.idempotency_key, p_source=body.source,
            p_occurred_at=body.occurred_at.isoformat(), p_external_ref=body.external_ref,
            p_amount_minor=body.amount_minor, p_currency=body.currency,
            p_items=body.items, p_metadata=body.metadata,
            p_responsible_user_id=responsible_user_id,
        )
    except Exception as exc:
        raise HTTPException(409, "Journey event could not be recorded") from exc


def set_journey_state(
    lead_ref: int, body: JourneyStateBody, responsible_user_id: str | None,
    *, offering: str = "sales",
) -> dict:
    """Aplica o estado-alvo e, ao fechar o pedido, devolve a IA ao atendimento.

    Fechar por conclusao ou cancelamento reinicia o ciclo: `resume_lead` zera
    `handoff_level`, marca `pending_reconfirmation` e devolve a fila os inbounds
    que ficaram parqueados em `waiting_human` durante o handoff. Sem isso o lead
    seguiria mudo indefinidamente -- nenhum worker reclama `waiting_human`.
    """
    lead = _lead_or_404(lead_ref)
    try:
        result = supabase_client.set_conversation_journey_state(
            p_persona_id=lead["persona_id"], p_lead_ref=lead_ref,
            p_target=body.target, p_source=body.source,
            p_occurred_at=body.occurred_at.isoformat(), p_offering=offering,
            p_responsible_user_id=responsible_user_id, p_metadata=body.metadata,
        )
    except Exception as exc:
        raise HTTPException(409, _state_conflict_detail(exc)) from exc

    if result.get("changed") and result.get("journey_closed"):
        resumed = agents_service.resume_lead(lead_ref)
        result["ai_resumed"] = bool(resumed)
        if resumed:
            # A origem precisa ser auditavel: isto nao foi um humano clicando
            # em "retomar", foi o fechamento do pedido religando o ciclo.
            event_emitter.emit(
                "lead.ai_resumed", entity_type="lead", entity_id=str(lead_ref),
                payload={"ai_paused": False, "by": "journey_closed",
                         "target": body.target},
            )
            result["notice"] = agents_service.reactivation_notice(
                lead_ref, reason="journey_closed"
            )
    elif result.get("changed") and body.target == "qualificado":
        # Selecionar "qualificado" manualmente e o mesmo desfecho que o SDR
        # atinge sozinho ao confirmar a qualificacao -- e o SDR e a unica
        # agente do lead, entao os dois caminhos devem pausar a IA do mesmo
        # jeito. O caminho automatico ja grava handoff_level='full' em
        # conversation_runtime.commit(); aqui cobrimos so a correcao manual.
        paused = agents_service.pause_lead(lead_ref)
        result["ai_paused"] = bool(paused)
        if paused:
            event_emitter.emit(
                "lead.ai_paused", entity_type="lead", entity_id=str(lead_ref),
                payload={"ai_paused": True, "by": "journey_qualified",
                         "target": body.target},
            )
    return result


def _state_conflict_detail(exc: Exception) -> str:
    """Traduz as guardas nomeadas da RPC; o resto continua generico."""
    texto = str(exc)
    if "a newer order already exists" in texto:
        return "Ja existe um pedido mais novo para esta lead; nao da para reabrir o anterior."
    if "lead already converted" in texto:
        return "Esta lead ja converteu: qualificado nao esta mais disponivel."
    if "no journey for this lead" in texto:
        return "Esta lead ainda nao tem pedido."
    return "Journey state could not be changed"


def _record_purchase(lead_ref: int, body: PurchaseCompletedBody, responsible_user_id: str | None) -> dict:
    return record_journey_event(
        lead_ref,
        JourneyEventBody(event_type="sale_recorded", **body.model_dump()),
        responsible_user_id,
    )


@router.post("/leads/{lead_ref}/journey-events")
def journey_event(lead_ref: int, body: JourneyEventBody, request: Request) -> dict:
    lead = _lead_or_404(lead_ref)
    user = auth_service.current_user(request)
    auth_service.assert_persona_capability(request, "edit", persona_id=lead["persona_id"])
    return record_journey_event(lead_ref, body, str(user["id"]))


@router.post("/leads/{lead_ref}/journey-state")
def journey_state(lead_ref: int, body: JourneyStateBody, request: Request) -> dict:
    lead = _lead_or_404(lead_ref)
    user = auth_service.current_user(request)
    auth_service.assert_persona_capability(request, "edit", persona_id=lead["persona_id"])
    return set_journey_state(lead_ref, body, str(user["id"]))


@internal_router.post("/leads/{lead_ref}/journey-events")
def journey_event_internal(
    lead_ref: int, body: JourneyEventBody,
    x_webhook_token: str | None = Header(None, alias="X-Webhook-Token"),
) -> dict:
    authorize_internal(x_webhook_token)
    return record_journey_event(lead_ref, body, None)


@router.post("/leads/{lead_ref}/purchase-completed")
def purchase_completed(lead_ref: int, body: PurchaseCompletedBody, request: Request) -> dict:
    lead = _lead_or_404(lead_ref)
    user = auth_service.current_user(request)
    auth_service.assert_persona_capability(request, "edit", persona_id=lead["persona_id"])
    return _record_purchase(lead_ref, body, str(user["id"]))


@internal_router.post("/leads/{lead_ref}/purchase-completed")
def purchase_completed_internal(
    lead_ref: int, body: PurchaseCompletedBody,
    x_webhook_token: str | None = Header(None, alias="X-Webhook-Token"),
) -> dict:
    authorize_internal(x_webhook_token)
    return _record_purchase(lead_ref, body, None)


@router.post("/sales-conversions/{conversion_id}/status")
def transition_conversion(conversion_id: UUID, body: ConversionStatusBody, request: Request) -> dict:
    user = auth_service.current_user(request)
    result = supabase_client.get_client().table("sales_conversions").select(
        "id,persona_id"
    ).eq("id", str(conversion_id)).maybe_single().execute()
    conversion = getattr(result, "data", None) or {}
    if not conversion:
        raise HTTPException(404, "Conversion not found")
    auth_service.assert_persona_capability(request, "edit", persona_id=conversion["persona_id"])
    try:
        return supabase_client.transition_sales_conversion_status(
            p_conversion_id=str(conversion_id), p_status=body.status,
            p_idempotency_key=body.idempotency_key,
            p_refunded_amount_minor=body.refunded_amount_minor,
            p_metadata=body.metadata, p_responsible_user_id=str(user["id"]),
        )
    except Exception as exc:
        raise HTTPException(409, "Conversion status could not be changed") from exc

