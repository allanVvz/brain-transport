import logging
import os
import secrets
import time
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from schemas.persona import PersonaCreate, PersonaUpdate
from services import public_site
from services import auth_service, deepseek_n8n_service, supabase_client, vault_sync

router = APIRouter(prefix="/personas", tags=["personas"])
logger = logging.getLogger("personas")


class RoutingUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    process_mode: str | None = None
    conversation_mode: str | None = None
    outbound_webhook_url: str | None = None
    outbound_webhook_secret: str | None = None
    inbound_webhook_token: str | None = None
    rotate_inbound_token: bool | None = None


class PublicSiteUpdate(BaseModel):
    site_slug: Optional[str] = None
    site_name: Optional[str] = None
    format_key: Optional[str] = None
    default_collection_slug: Optional[str] = None
    whatsapp_phone: Optional[str] = None
    whatsapp_message_template: Optional[str] = None
    catalog_url: Optional[str] = None


_KNOWN_CONVERSATION_MODES = {"deterministic", "n8n_agents", "orquestrador"}


def _active_binding(persona_id: str | None) -> dict:
    """The routing switch that actually governs live dispatch â€” read
    straight from the active binding, not the legacy process_mode column.

    Confirmed live this session: decision_owner can be changed directly on
    a binding (as production hotfixes did repeatedly) without process_mode
    ever being touched, leaving any UI that only reads process_mode showing
    a stale value while automation silently runs a different engine.
    """
    if not persona_id:
        return {}
    for binding in supabase_client.get_workflow_bindings(persona_id):
        if binding.get("active"):
            return binding
    return {}


def _routing_readiness(
    routing: dict,
    live_binding: dict,
    conversation_mode: str,
) -> dict:
    """Report whether the selected engine can actually execute a v3 turn.

    Selection and readiness are intentionally separate. A failed switch must
    not leave the dashboard presenting the previous legacy engine as healthy.
    """
    persona_id = str(routing.get("id") or "")
    metadata = live_binding.get("metadata") or {}
    publication = (
        supabase_client.get_active_graph_publication(persona_id)
        if persona_id and conversation_mode in {"deterministic", "n8n_agents"}
        else None
    )
    blocked_reasons: list[str] = []
    if not live_binding:
        blocked_reasons.append("active_binding_missing")
    if conversation_mode in {"deterministic", "n8n_agents"} and not publication:
        blocked_reasons.append("active_graph_publication_v3_missing")

    paused = bool(
        metadata.get("safety_paused")
        or live_binding.get("connection_status") == "safety_paused"
    )
    if conversation_mode == "deterministic":
        if metadata.get("runtime_version") != "graph_agent_runtime_v3":
            blocked_reasons.append("legacy_deterministic_runtime_unverified")
    elif conversation_mode == "n8n_agents":
        integration = (
            supabase_client.get_persona_integration_connection(persona_id, "deepseek")
            or {}
        )
        integration_config = integration.get("config_json") or {}
        if not integration.get("enabled"):
            blocked_reasons.append("deepseek_integration_disabled")
        if not integration_config.get("n8n_credential_id"):
            blocked_reasons.append("deepseek_n8n_credential_missing")
        if not live_binding.get("n8n_workflow_id"):
            blocked_reasons.append("n8n_workflow_missing")
        if metadata.get("pipeline_contract") != "conversation_v3":
            blocked_reasons.append("conversation_v3_contract_missing")
        if metadata.get("runtime_version") != "graph_agent_runtime_v3":
            blocked_reasons.append("graph_agent_runtime_v3_missing")
    else:
        blocked_reasons.append("engine_not_implemented")

    operational = not paused and not blocked_reasons
    return {
        "operational": operational,
        "operational_state": "paused" if paused else ("ready" if operational else "blocked"),
        "blocked_reasons": blocked_reasons,
        "active_graph_publication": (
            {
                "id": publication.get("id"),
                "version": publication.get("version"),
                "checksum": publication.get("checksum"),
                "compiler_version": publication.get("compiler_version"),
            }
            if publication else None
        ),
    }


def _mask_routing(routing: dict, *, include_readiness: bool = False) -> dict:
    """Hide secret/token values from GET responses; return only presence flag."""
    if not routing:
        return routing
    process_mode = routing.get("process_mode") or "internal"
    live_binding = _active_binding(routing.get("id"))
    live_metadata = live_binding.get("metadata") or {}
    live_decision_owner = live_metadata.get("decision_owner")
    conversation_mode = (
        live_decision_owner
        if live_decision_owner in _KNOWN_CONVERSATION_MODES
        else ("n8n_agents" if process_mode == "n8n" else "deterministic")
    )
    response = {
        "slug": routing.get("slug"),
        "id": routing.get("id"),
        "process_mode": process_mode,
        "conversation_mode": conversation_mode,
        "pipeline_contract": live_metadata.get("pipeline_contract") or "conversation_v1",
        "classifier": live_metadata.get("runtime_version") or (
            "deterministic_v1" if conversation_mode == "deterministic" else None
        ),
        "field_extractor": live_metadata.get("model"),
        "model_required": conversation_mode == "n8n_agents",
        "outbound_webhook_url": routing.get("outbound_webhook_url"),
        "has_outbound_webhook_secret": bool(routing.get("outbound_webhook_secret")),
        "has_inbound_webhook_token": bool(routing.get("inbound_webhook_token")),
        "migration_applied": bool(routing.get("migration_applied")),
        "routing_source": routing.get("routing_source") or "default",
    }
    if include_readiness:
        response["readiness"] = _routing_readiness(
            routing, live_binding, conversation_mode
        )
    return response


@router.get("")
def list_personas(request: Request):
    user = auth_service.current_user(request)
    access = auth_service.allowed_access(request)
    return auth_service.filter_personas_for_user(user, supabase_client.get_personas(), access)


@router.get("/{slug}")
def get_persona(slug: str, request: Request):
    persona = supabase_client.get_persona(slug)
    if not persona:
        raise HTTPException(404, "Persona not found")
    auth_service.assert_persona_access(request, persona_id=persona.get("id"), persona_slug=slug)
    return persona


@router.post("")
def create_persona(body: PersonaCreate, request: Request):
    if not auth_service.is_admin(auth_service.current_user(request)):
        raise HTTPException(403, "Apenas admin pode criar personas")
    payload = body.model_dump()
    config = dict(payload.get("config") or {})
    config.setdefault(
        "public_site",
        public_site.default_public_site_config(payload),
    )
    payload["config"] = config
    supabase_client.upsert_persona(payload)
    vault_sync.ensure_persona_vault_structure(body.slug)
    return supabase_client.get_persona(body.slug)


@router.patch("/{slug}")
def update_persona(slug: str, body: PersonaUpdate, request: Request):
    if not auth_service.is_admin(auth_service.current_user(request)):
        raise HTTPException(403, "Apenas admin pode editar personas")
    supabase_client.upsert_persona({"slug": slug, **body.model_dump(exclude_none=True)})
    return supabase_client.get_persona(slug)


def _public_site_response(persona: dict) -> dict:
    formats = supabase_client.list_public_site_formats(enabled_only=True)
    return {
        "persona_slug": persona.get("slug"),
        "persona_id": persona.get("id"),
        "catalog_url": persona.get("catalog_url"),
        "formats": formats,
        "config": public_site.normalize_public_site_config(persona, formats),
        "site": public_site.public_site_payload(persona, formats, catalog_url=persona.get("catalog_url")),
    }


def _assert_unique_site_slug(persona_slug: str, site_slug: str, formats: list[dict]) -> None:
    for row in supabase_client.get_personas():
        if row.get("slug") == persona_slug:
            continue
        existing = public_site.normalize_public_site_config(row, formats).get("site_slug")
        if existing == site_slug:
            raise HTTPException(409, "site_slug ja esta em uso por outra persona")


@router.get("/{slug}/public-site")
def get_public_site(slug: str, request: Request):
    persona = supabase_client.get_persona(slug)
    if not persona:
        raise HTTPException(404, "Persona not found")
    auth_service.assert_persona_access(request, persona_id=persona.get("id"), persona_slug=slug)
    return _public_site_response(persona)


@router.patch("/{slug}/public-site")
def update_public_site(slug: str, body: PublicSiteUpdate, request: Request):
    if not auth_service.is_admin(auth_service.current_user(request)):
        raise HTTPException(403, "Apenas admin pode editar o site publico")
    persona = supabase_client.get_persona(slug)
    if not persona:
        raise HTTPException(404, "Persona not found")
    formats = supabase_client.list_public_site_formats(enabled_only=True)
    current = public_site.normalize_public_site_config(persona, formats)
    patch = body.model_dump(exclude_unset=True)
    catalog_url_provided = "catalog_url" in patch
    catalog_url = patch.pop("catalog_url", None)
    if "format_key" in patch and not public_site.format_by_key(formats, str(patch["format_key"] or "")):
        raise HTTPException(422, "format_key deve existir em public_site_formats e estar ativo")
    next_config = {**current, **patch}
    next_config = public_site.normalize_public_site_config(
        {"slug": persona.get("slug"), "name": persona.get("name"), "config": {"public_site": next_config}},
        formats,
    )
    if not next_config.get("site_slug"):
        raise HTTPException(422, "site_slug obrigatorio")
    if not next_config.get("site_name"):
        raise HTTPException(422, "site_name obrigatorio")
    _assert_unique_site_slug(slug, next_config["site_slug"], formats)
    merged_config = dict(persona.get("config") or {})
    merged_config["public_site"] = next_config
    merged_config["default_collection_slug"] = next_config["default_collection_slug"]
    if catalog_url_provided:
        updated = supabase_client.update_persona_config(slug, merged_config, catalog_url=(catalog_url or None))
    else:
        updated = supabase_client.update_persona_config(slug, merged_config)
    return _public_site_response(updated or {**persona, "config": merged_config})


@router.get("/{slug}/routing")
def get_routing(slug: str, request: Request):
    routing = supabase_client.get_persona_routing(slug)
    if not routing:
        raise HTTPException(404, "Persona not found")
    auth_service.assert_persona_access(request, persona_id=routing.get("id"), persona_slug=slug)
    return _mask_routing(routing, include_readiness=True)


@router.patch("/{slug}/routing")
def update_routing(slug: str, body: RoutingUpdate, request: Request):
    if not auth_service.is_admin(auth_service.current_user(request)):
        raise HTTPException(403, "Apenas admin pode editar routing")
    current = supabase_client.get_persona_routing(slug)
    if not current:
        raise HTTPException(404, "Persona not found")
    if not current.get("migration_applied"):
        raise HTTPException(
            409,
            "Migration 011 not applied. Apply supabase/migrations/011_persona_routing.sql before editing persona routing.",
        )
    payload: dict = {}
    rotated_token: str | None = None
    requested_conversation_mode = body.conversation_mode
    aliases = {
        "deterministic": "internal",
        "n8n_agents": "n8n",
    }
    if requested_conversation_mode is not None:
        if requested_conversation_mode not in _KNOWN_CONVERSATION_MODES:
            raise HTTPException(
                400,
                "conversation_mode must be 'deterministic', 'n8n_agents' or 'orquestrador'",
            )
        normalized = aliases.get(requested_conversation_mode)
        if normalized is not None:
            if body.process_mode is not None and body.process_mode != normalized:
                raise HTTPException(400, "routing mode fields conflict")
            payload["process_mode"] = normalized
        # "orquestrador" has no legacy process_mode equivalent yet (that
        # column only understands internal/n8n) â€” leave it untouched
        # rather than write an invalid value into a still-read legacy field.
    if body.process_mode is not None:
        if body.process_mode not in {"internal", "n8n"}:
            raise HTTPException(400, "process_mode must be 'internal' or 'n8n'")
        payload["process_mode"] = body.process_mode
    if body.outbound_webhook_url is not None:
        if body.outbound_webhook_url.strip():
            raise HTTPException(
                400,
                "provider_direct nao aceita webhook de saida n8n",
            )
        payload["outbound_webhook_url"] = None
    if body.outbound_webhook_secret is not None:
        if body.outbound_webhook_secret:
            raise HTTPException(
                400,
                "provider_direct nao aceita secret de webhook de saida n8n",
            )
        payload["outbound_webhook_secret"] = None
    if body.inbound_webhook_token is not None:
        payload["inbound_webhook_token"] = body.inbound_webhook_token or None
    if body.rotate_inbound_token:
        rotated_token = secrets.token_urlsafe(32)
        payload["inbound_webhook_token"] = rotated_token
    if not payload and requested_conversation_mode is None:
        return _mask_routing(current, include_readiness=True)
    if requested_conversation_mode is not None:
        conversation_mode = requested_conversation_mode
        deepseek_config: dict = {}
        if conversation_mode == "n8n_agents":
            integration = (
                supabase_client.get_persona_integration_connection(
                    current.get("id"),
                    "deepseek",
                )
                or {}
            )
            deepseek_config = dict(integration.get("config_json") or {})
            if not integration.get("enabled") or not deepseek_config.get("n8n_credential_id"):
                raise HTTPException(
                    409,
                    "Configure a chave DeepSeek da persona em Ferramentas antes de ativar o n8n.",
                )
            # Build (or rebuild) the live n8n workflow *before* pointing the
            # binding at it, so switching to n8n_agents always ends with a
            # real, existing workflow id and webhook path â€” including the
            # first time this persona is ever switched, when no workflow
            # exists yet. The credential is reused as-is; it's the only
            # place the raw DeepSeek key still lives once saved.
            full_persona = supabase_client.get_persona(slug) or current
            if not supabase_client.get_active_graph_publication(str(current.get("id") or "")):
                raise HTTPException(
                    409,
                    "Ative uma publicacao GraphRAG v3 consistente antes de ativar o modelo.",
                )
            deepseek_config = deepseek_n8n_service.resync_workflow_for_persona(
                full_persona, deepseek_config,
            )
            workflow_check = deepseek_n8n_service.check_workflow_wiring(
                deepseek_config
            )
            supabase_client.insert_event(
                {
                    "event_type": (
                        "n8n.workflow_validation.succeeded"
                        if workflow_check["ok"]
                        else "n8n.workflow_validation.failed"
                    ),
                    "entity_type": "persona",
                    "entity_id": str(current.get("id") or slug),
                    "persona_id": current.get("id"),
                    "payload": {
                        "persona_slug": slug,
                        "reason": workflow_check.get("reason"),
                        "diagnostics": workflow_check.get("diagnostics") or {},
                    },
                },
                level="info" if workflow_check["ok"] else "error",
                source="routes.personas",
            )
            if not workflow_check["ok"]:
                raise HTTPException(409, workflow_check["reason"])
            supabase_client.save_persona_integration_connection({
                "persona_id": current.get("id"),
                "service": "deepseek",
                "config_json": deepseek_config,
            })
        for binding in supabase_client.get_workflow_bindings(current.get("id")):
            if not binding.get("id") or not binding.get("active"):
                continue
            metadata = {
                **(binding.get("metadata") or {}),
                "conversation_mode": conversation_mode,
                "decision_owner": conversation_mode,
                "pipeline_contract": (
                    deepseek_config.get("pipeline_contract") or "conversation_v3"
                    if conversation_mode == "n8n_agents" else "conversation_v1"
                ),
                "transport_mode": "provider_direct",
            }
            binding_update: dict = {"metadata": metadata}
            if conversation_mode == "n8n_agents":
                n8n_base = str(os.environ.get("N8N_BASE_URL") or "").rstrip("/")
                webhook_path = str(
                    deepseek_config.get("conversation_webhook_path") or ""
                ).strip("/")
                if not n8n_base or not webhook_path:
                    raise HTTPException(409, "Endpoint interno do n8n nao configurado.")
                metadata["conversation_webhook_url"] = (
                    f"{n8n_base}/webhook/{webhook_path}"
                )
                metadata["runtime_version"] = (
                    deepseek_config.get("runtime_version")
                    or "graph_agent_runtime_v3"
                )
                metadata["model"] = deepseek_config.get("model")
                metadata["reply_source"] = deepseek_config.get("reply_source")
                binding_update["n8n_workflow_id"] = deepseek_config["n8n_workflow_id"]
            else:
                metadata.pop("conversation_webhook_url", None)
                metadata.pop("runtime_version", None)
                binding_update["n8n_workflow_id"] = None
            updated_binding = supabase_client.update_workflow_binding(
                str(binding["id"]),
                binding_update,
            )
            if conversation_mode == "n8n_agents":
                effective_binding = updated_binding or {
                    **binding,
                    **binding_update,
                    "persona_id": binding.get("persona_id") or current.get("id"),
                    "metadata": metadata,
                }
                binding_check = deepseek_n8n_service.check_binding(
                    persona=full_persona,
                    binding=effective_binding,
                    deepseek_config=deepseek_config,
                    n8n_base_url=n8n_base,
                )
                supabase_client.insert_event(
                    {
                        "event_type": (
                            "n8n.binding_validation.succeeded"
                            if binding_check["ok"]
                            else "n8n.binding_validation.failed"
                        ),
                        "entity_type": "workflow_binding",
                        "entity_id": str(binding["id"]),
                        "persona_id": current.get("id"),
                        "payload": {
                            "persona_slug": slug,
                            "reason": binding_check.get("reason"),
                            "diagnostics": binding_check.get("diagnostics") or {},
                        },
                    },
                    level="info" if binding_check["ok"] else "error",
                    source="routes.personas",
                )
                if not binding_check["ok"]:
                    raise HTTPException(409, binding_check["reason"])
        if payload:
            supabase_client.update_persona_routing(slug, payload)
        supabase_client.insert_event(
            {
                "event_type": "conversation.mode_updated",
                "entity_type": "persona",
                "entity_id": str(current.get("id") or slug),
                "persona_id": current.get("id"),
                "payload": {
                    "persona_slug": slug,
                    "conversation_mode": conversation_mode,
                    "pipeline_contract": (
                        deepseek_config.get("pipeline_contract")
                        if conversation_mode == "n8n_agents"
                        else "conversation_v1"
                    ),
                    "classifier": (
                        deepseek_config.get("runtime_version")
                        if conversation_mode == "n8n_agents"
                        else "deterministic_v1"
                    ),
                    "field_extractor": (
                        deepseek_config.get("model")
                        if conversation_mode == "n8n_agents"
                        else None
                    ),
                },
            },
            source="routes.personas",
        )
    elif payload:
        supabase_client.update_persona_routing(slug, payload)
    final = supabase_client.get_persona_routing(slug) or current
    response = _mask_routing(final, include_readiness=True)
    if rotated_token:
        response["inbound_webhook_token"] = rotated_token
    return response


@router.post("/{slug}/routing/test")
def test_outbound_webhook(slug: str, request: Request):
    """Fires a synthetic ping at the persona's outbound webhook so the
    operator can confirm n8n is wired up correctly. Returns the upstream
    status code and any error string."""
    routing = supabase_client.get_persona_routing(slug)
    if not routing:
        raise HTTPException(404, "Persona not found")
    auth_service.assert_persona_access(request, persona_id=routing.get("id"), persona_slug=slug)
    url = routing.get("outbound_webhook_url")
    if not url:
        raise HTTPException(400, "outbound_webhook_url not configured")
    headers = {"Content-Type": "application/json"}
    secret = routing.get("outbound_webhook_secret")
    if secret:
        headers["X-Webhook-Secret"] = secret
    payload = {
        "ping": True,
        "persona_slug": slug,
        "issued_at": int(time.time()),
        "source": "ai-brain.routing.test",
    }
    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=10.0)
        return {"ok": 200 <= resp.status_code < 300, "status": resp.status_code, "body": resp.text[:500]}
    except Exception as exc:
        logger.warning("routing test failed for %s: %s", slug, exc)
        return {"ok": False, "status": None, "error": str(exc)[:300]}

