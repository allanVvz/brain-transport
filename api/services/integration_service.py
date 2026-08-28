from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import gspread
import httpx
from google.oauth2.service_account import Credentials

from services import deepseek_n8n_service, n8n_client, secret_store, supabase_client
from utils.tls import get_ca_bundle_path

CATALOG: list[dict[str, Any]] = [
    {
        "service": "google_sheets",
        "label": "Google Sheets",
        "description": "Planilhas usadas como fonte de conhecimento e operacao.",
        "scope": "user",
        "requires_credentials": True,
        "user_managed": True,
    },
    {
        "service": "airtable",
        "label": "Airtable",
        "description": "Base operacional para CRM e sincronizacoes estruturadas.",
        "scope": "user",
        "requires_credentials": True,
        "user_managed": True,
    },
    {
        "service": "n8n",
        "label": "n8n",
        "description": "Automacoes e espelhamento de execucoes.",
        "scope": "system",
        "requires_credentials": False,
        "user_managed": False,
    },
    {
        "service": "openai",
        "label": "ChatGPT / OpenAI",
        "description": "Modelos GPT, embeddings e chat da persona.",
        "scope": "persona",
        "requires_credentials": True,
        "user_managed": True,
    },
    {
        "service": "anthropic",
        "label": "Claude / Anthropic",
        "description": "Modelos Claude e fallback de IA da persona.",
        "scope": "persona",
        "requires_credentials": True,
        "user_managed": True,
    },
    {
        "service": "deepseek",
        "label": "DeepSeek",
        "description": "Modelo do workflow conversacional n8n da persona.",
        "scope": "persona",
        "requires_credentials": True,
        "user_managed": True,
    },
    {
        "service": "whatsapp",
        "label": "WhatsApp",
        "description": "Canal de entrada e saida para atendimento.",
        "scope": "system",
        "requires_credentials": False,
        "user_managed": False,
    },
    {
        "service": "figma_mcp",
        "label": "Figma MCP",
        "description": "Ferramentas de design e contexto visual no protocolo MCP.",
        "scope": "system",
        "requires_credentials": False,
        "user_managed": False,
    },
    {
        "service": "meta",
        "label": "Meta",
        "description": "Catalogo WhatsApp Business (Graph API).",
        "scope": "persona",
        "requires_credentials": True,
        "user_managed": True,
    },
    {
        "service": "meta_whatsapp",
        "label": "Meta WhatsApp",
        "description": "Token de mensageria (Graph API) usado para vincular o numero WhatsApp.",
        "scope": "persona",
        "requires_credentials": True,
        "user_managed": True,
    },
]

CATALOG_BY_SERVICE = {item["service"]: item for item in CATALOG}
_GOOGLE_SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
_PLACEHOLDER_MARKERS = {"", "your-airtable-key", "your-api-key", "changeme", "placeholder"}


class IntegrationValidationError(ValueError):
    pass


def list_catalog() -> list[dict[str, Any]]:
    return [dict(item) for item in CATALOG]


def get_catalog_item(service: str) -> dict[str, Any]:
    item = CATALOG_BY_SERVICE.get(service)
    if not item:
        raise KeyError(service)
    return dict(item)


def is_user_managed(service: str) -> bool:
    return bool(CATALOG_BY_SERVICE.get(service, {}).get("user_managed"))


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _http_client(timeout: float = 10.0) -> httpx.Client:
    return httpx.Client(timeout=timeout, verify=get_ca_bundle_path())


def _normalize_google_payload(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    raw = payload.get("service_account_json")
    if not raw:
        raise IntegrationValidationError("service_account_json is required.")
    if isinstance(raw, dict):
        secret_payload = raw
    else:
        try:
            secret_payload = json.loads(str(raw))
        except Exception as exc:
            raise IntegrationValidationError("service_account_json must be valid JSON.") from exc
    return json.dumps(secret_payload, ensure_ascii=False), {
        "spreadsheet_id": (payload.get("spreadsheet_id") or "").strip() or None,
    }


def _normalize_airtable_payload(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    api_key = (payload.get("api_key") or "").strip()
    base_id = (payload.get("base_id") or "").strip()
    if not api_key:
        raise IntegrationValidationError("api_key is required.")
    if not base_id:
        raise IntegrationValidationError("base_id is required.")
    return api_key, {"base_id": base_id}


def _normalize_meta_payload(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    access_token = (payload.get("access_token") or "").strip()
    business_id = (payload.get("business_id") or "").strip()
    catalog_id = (payload.get("catalog_id") or "").strip()
    if not access_token:
        raise IntegrationValidationError("access_token is required.")
    if not catalog_id:
        raise IntegrationValidationError("catalog_id is required.")
    return access_token, {"business_id": business_id or None, "catalog_id": catalog_id}


def _normalize_meta_whatsapp_payload(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    access_token = (payload.get("access_token") or "").strip()
    business_id = (payload.get("business_id") or "").strip()
    if not access_token:
        raise IntegrationValidationError("access_token is required.")
    return access_token, {"business_id": business_id or None}


def _normalize_llm_api_key_payload(service: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    api_key = (payload.get("api_key") or "").strip()
    if not api_key:
        raise IntegrationValidationError("api_key is required.")
    lower = api_key.lower()
    if lower in _PLACEHOLDER_MARKERS:
        raise IntegrationValidationError(f"{service} credential is a placeholder.")
    if service == "openai" and not api_key.startswith(("sk-", "sk-proj-", "sk-svcacct-")):
        raise IntegrationValidationError("OpenAI api_key has an invalid format.")
    if service == "anthropic" and not api_key.startswith("sk-ant-"):
        raise IntegrationValidationError("Anthropic api_key has an invalid format.")
    if service == "deepseek" and not api_key.startswith("sk-"):
        raise IntegrationValidationError("DeepSeek api_key has an invalid format.")
    if service != "deepseek":
        return api_key, {}
    model = str(
        payload.get("model")
        or os.environ.get("DEEPSEEK_CONVERSATION_MODEL")
        or "deepseek-v4-flash"
    ).strip()
    endpoint = str(
        payload.get("endpoint")
        or os.environ.get("DEEPSEEK_CONVERSATION_ENDPOINT")
        or "https://api.deepseek.com/chat/completions"
    ).strip()
    reply_source = str(payload.get("reply_source") or model).strip()
    if not model or not endpoint.startswith("https://"):
        raise IntegrationValidationError(
            "DeepSeek model binding requires model and HTTPS endpoint."
        )
    return api_key, {
        "model": model,
        "endpoint": endpoint,
        "reply_source": reply_source,
    }


def normalize_credentials(service: str, payload: Optional[dict[str, Any]]) -> tuple[Optional[str], dict[str, Any]]:
    body = dict(payload or {})
    if service == "google_sheets":
        return _normalize_google_payload(body)
    if service == "airtable":
        return _normalize_airtable_payload(body)
    if service in {"openai", "anthropic", "deepseek"}:
        return _normalize_llm_api_key_payload(service, body)
    if service == "meta":
        return _normalize_meta_payload(body)
    if service == "meta_whatsapp":
        return _normalize_meta_whatsapp_payload(body)
    return None, {}


def validate_google_sheets(service_account_json: str, spreadsheet_id: Optional[str] = None) -> tuple[str, Optional[str], Optional[int]]:
    try:
        info = json.loads(service_account_json)
        creds = Credentials.from_service_account_info(info, scopes=_GOOGLE_SCOPES)
    except Exception as exc:
        raise IntegrationValidationError(f"Invalid Google service account JSON: {exc}") from exc

    started = time.monotonic()
    try:
        client = gspread.authorize(creds)
        if spreadsheet_id:
            client.open_by_key(spreadsheet_id)
        latency_ms = int((time.monotonic() - started) * 1000)
        return "connected", None, latency_ms
    except Exception as exc:
        raise IntegrationValidationError(f"Google Sheets validation failed: {exc}") from exc


def validate_airtable(api_key: str, base_id: str) -> tuple[str, Optional[str], Optional[int]]:
    if api_key.strip().lower() in _PLACEHOLDER_MARKERS:
        raise IntegrationValidationError("Airtable credential is a placeholder.")
    started = time.monotonic()
    try:
        with _http_client(timeout=8.0) as client:
            response = client.get(
                f"https://api.airtable.com/v0/meta/bases/{base_id}/tables",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        latency_ms = int((time.monotonic() - started) * 1000)
        if response.status_code == 200:
            return "connected", None, latency_ms
        if response.status_code in {401, 403}:
            raise IntegrationValidationError("Airtable rejected the credentials.")
        raise IntegrationValidationError(f"Airtable validation failed with HTTP {response.status_code}.")
    except IntegrationValidationError:
        raise
    except Exception as exc:
        raise IntegrationValidationError(f"Airtable validation failed: {exc}") from exc


def validate_meta(
    access_token: str,
    catalog_id: str,
    *,
    fetch: Optional[Any] = None,
) -> tuple[str, Optional[str], Optional[int]]:
    """Validate Meta catalog credentials by listing one product.

    `fetch(access_token, catalog_id)` is injectable so tests run offline.
    Returns ("healthy", None, latency) on success.
    """
    if (access_token or "").strip().lower() in _PLACEHOLDER_MARKERS:
        raise IntegrationValidationError("Meta access token is a placeholder.")
    if not catalog_id:
        raise IntegrationValidationError("Meta catalog_id is required.")
    started = time.monotonic()
    try:
        if fetch is not None:
            fetch(access_token, catalog_id)
        else:
            with _http_client(timeout=8.0) as client:
                response = client.get(
                    f"https://graph.facebook.com/v19.0/{catalog_id}/products",
                    params={"access_token": access_token, "limit": 1},
                )
                if response.status_code in {401, 403}:
                    raise IntegrationValidationError("Meta rejected the credentials.")
                if response.status_code != 200:
                    raise IntegrationValidationError(f"Meta validation failed with HTTP {response.status_code}.")
        latency_ms = int((time.monotonic() - started) * 1000)
        return "healthy", None, latency_ms
    except IntegrationValidationError:
        raise
    except Exception as exc:
        raise IntegrationValidationError(f"Meta validation failed: {exc}") from exc


def validate_meta_whatsapp(
    access_token: str,
    business_id: Optional[str] = None,
    *,
    fetch: Optional[Any] = None,
) -> tuple[str, Optional[str], Optional[int]]:
    """Validate a Meta WhatsApp messaging token, independent of any catalog.

    `fetch(access_token, business_id)` is injectable so tests run offline.
    Returns ("healthy", None, latency) on success.
    """
    if (access_token or "").strip().lower() in _PLACEHOLDER_MARKERS:
        raise IntegrationValidationError("Meta access token is a placeholder.")
    started = time.monotonic()
    try:
        if fetch is not None:
            fetch(access_token, business_id)
        else:
            with _http_client(timeout=8.0) as client:
                response = client.get(
                    "https://graph.facebook.com/v21.0/me",
                    params={"access_token": access_token},
                )
                if response.status_code in {401, 403}:
                    raise IntegrationValidationError("Meta rejected the credentials.")
                if response.status_code != 200:
                    raise IntegrationValidationError(f"Meta validation failed with HTTP {response.status_code}.")
        latency_ms = int((time.monotonic() - started) * 1000)
        return "healthy", None, latency_ms
    except IntegrationValidationError:
        raise
    except Exception as exc:
        raise IntegrationValidationError(f"Meta validation failed: {exc}") from exc


def validate_deepseek(
    api_key: str,
    *,
    model: str | None = None,
) -> tuple[str, Optional[str], Optional[int]]:
    started = time.monotonic()
    try:
        with _http_client(timeout=8.0) as client:
            response = client.get(
                "https://api.deepseek.com/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        latency_ms = int((time.monotonic() - started) * 1000)
        if response.status_code in {401, 403}:
            raise IntegrationValidationError("DeepSeek rejeitou a chave informada.")
        if response.status_code != 200:
            raise IntegrationValidationError(
                f"DeepSeek indisponivel (HTTP {response.status_code})."
            )
        models = {
            str(item.get("id") or "")
            for item in (response.json().get("data") or [])
        }
        requested_model = str(
            model
            or os.environ.get("DEEPSEEK_CONVERSATION_MODEL")
            or "deepseek-v4-flash"
        ).strip()
        if requested_model not in models:
            raise IntegrationValidationError(
                f"DeepSeek nao disponibilizou o modelo configurado {requested_model}."
            )
        return "connected", None, latency_ms
    except IntegrationValidationError:
        raise
    except Exception as exc:
        raise IntegrationValidationError(
            f"Falha ao validar a chave DeepSeek: {exc}"
        ) from exc


def get_meta_credentials(user_id: str) -> dict[str, Any]:
    """Return decrypted Meta credentials for the import service. Never logged."""
    row = supabase_client.get_user_integration_connection(user_id, "meta") or {}
    access_token = secret_store.decrypt_secret(row.get("secret_ciphertext"))
    if not access_token:
        raise IntegrationValidationError("Meta integration is not configured. Configure it in Tools first.")
    config = row.get("config_json") or {}
    return {
        "access_token": access_token,
        "business_id": config.get("business_id"),
        "catalog_id": config.get("catalog_id"),
    }


def validate_credentials(service: str, *, secret_value: str, config_json: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    config = dict(config_json or {})
    if service == "google_sheets":
        status, error, latency = validate_google_sheets(secret_value, config.get("spreadsheet_id"))
    elif service == "airtable":
        status, error, latency = validate_airtable(secret_value, str(config.get("base_id") or ""))
    elif service in {"openai", "anthropic"}:
        _normalize_llm_api_key_payload(service, {"api_key": secret_value})
        status, error, latency = "connected", None, None
    elif service == "deepseek":
        _normalize_llm_api_key_payload(service, {"api_key": secret_value})
        status, error, latency = validate_deepseek(secret_value)
    elif service == "meta":
        status, error, latency = validate_meta(secret_value, str(config.get("catalog_id") or ""))
    elif service == "meta_whatsapp":
        status, error, latency = validate_meta_whatsapp(secret_value, config.get("business_id"))
    else:
        raise IntegrationValidationError(f"Unsupported user-managed service: {service}")
    return {
        "status": status,
        "last_error": error,
        "last_validated_at": _utcnow(),
        "response_ms": latency,
    }


def _merge_user_integration(service: str, row: Optional[dict[str, Any]]) -> dict[str, Any]:
    catalog = get_catalog_item(service)
    connection = row or {}
    connection_config = connection.get("config_json") or {}
    configured = bool(
        connection.get("secret_ciphertext")
        or connection_config.get("n8n_credential_id")
    )
    enabled = bool(connection.get("enabled")) if configured else False
    status = str(connection.get("status") or ("disabled" if not enabled else "never_validated"))
    if not configured:
        status = "never_validated"
        if connection.get("enabled"):
            enabled = False
    elif not enabled and status == "connected":
        status = "disabled"
    elif not enabled and not connection.get("status"):
        status = "disabled"
    return {
        "service": service,
        "label": catalog["label"],
        "description": catalog["description"],
        "scope": catalog["scope"],
        "enabled": enabled,
        "status": status,
        "requires_credentials": True,
        "configured": configured,
        "config_json": connection_config,
        "last_validated_at": connection.get("last_validated_at"),
        "last_error": connection.get("last_error"),
    }


def _merge_system_integration(service: str, row: Optional[dict[str, Any]]) -> dict[str, Any]:
    catalog = get_catalog_item(service)
    status_row = row or {}
    status = str(status_row.get("status") or "unknown")
    configured = system_service_has_runtime_credentials(service)
    return {
        "service": service,
        "label": catalog["label"],
        "description": catalog["description"],
        "scope": catalog["scope"],
        "enabled": configured and status not in {"down", "disabled"},
        "status": status,
        "requires_credentials": False,
        "configured": configured,
        "last_validated_at": status_row.get("last_check"),
        "last_error": status_row.get("error_message"),
        "response_ms": status_row.get("response_ms"),
    }


def list_user_integrations(user_id: str) -> list[dict[str, Any]]:
    user_rows = {row["service"]: row for row in supabase_client.list_user_integration_connections(user_id)}
    system_rows = {row["service"]: row for row in supabase_client.get_integration_statuses(persona_id=None)}
    merged: list[dict[str, Any]] = []
    for item in CATALOG:
        service = item["service"]
        if item["user_managed"]:
            merged.append(_merge_user_integration(service, user_rows.get(service)))
        else:
            merged.append(_merge_system_integration(service, system_rows.get(service)))
    return merged


def get_user_integration_state(user_id: str, service: str) -> dict[str, Any]:
    if not is_user_managed(service):
        raise KeyError(service)
    return _merge_user_integration(service, supabase_client.get_user_integration_connection(user_id, service))


def list_persona_integrations(persona_id: str) -> list[dict[str, Any]]:
    persona_rows = {
        row["service"]: row
        for row in supabase_client.list_persona_integration_connections(persona_id)
    }
    system_rows = {
        row["service"]: row
        for row in supabase_client.get_integration_statuses(persona_id=persona_id)
    }
    merged: list[dict[str, Any]] = []
    for item in CATALOG:
        service = item["service"]
        if item["user_managed"]:
            merged.append(_merge_user_integration(service, persona_rows.get(service)))
        else:
            merged.append(_merge_system_integration(service, system_rows.get(service)))
    return merged


def get_persona_integration_state(persona_id: str, service: str) -> dict[str, Any]:
    if not is_user_managed(service):
        raise KeyError(service)
    return _merge_user_integration(
        service,
        supabase_client.get_persona_integration_connection(persona_id, service),
    )


def _build_update_payload(
    *,
    user_id: str,
    service: str,
    existing: Optional[dict[str, Any]],
    enabled: bool,
    secret_value: Optional[str],
    config_json: Optional[dict[str, Any]],
    validation: Optional[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "user_id": user_id,
        "service": service,
        "enabled": enabled,
        "status": (validation or {}).get("status") or ("disabled" if not enabled else existing.get("status") if existing else "never_validated"),
        "config_json": config_json if config_json is not None else (existing.get("config_json") if existing else {}),
        "secret_ciphertext": secret_store.encrypt_secret(secret_value) if secret_value is not None else (existing.get("secret_ciphertext") if existing else None),
        "last_validated_at": (validation or {}).get("last_validated_at"),
        "last_error": (validation or {}).get("last_error"),
    }
    if not enabled and payload.get("status") == "connected":
        payload["status"] = "disabled"
    return payload


def save_user_integration(
    user_id: str,
    service: str,
    *,
    enabled: bool,
    credentials: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if not is_user_managed(service):
        raise KeyError(service)
    existing = supabase_client.get_user_integration_connection(user_id, service) or {}
    secret_value, config_json = normalize_credentials(service, credentials) if credentials else (None, None)

    if enabled:
        if secret_value is None:
            decrypted = secret_store.decrypt_secret(existing.get("secret_ciphertext"))
            if not decrypted:
                raise IntegrationValidationError("Credentials are required before enabling this integration.")
            secret_value = decrypted
        if config_json is None:
            config_json = existing.get("config_json") or {}
        validation = validate_credentials(service, secret_value=secret_value, config_json=config_json)
    else:
        validation = {
            "status": "disabled",
            "last_error": None,
            "last_validated_at": existing.get("last_validated_at"),
        }

    payload = _build_update_payload(
        user_id=user_id,
        service=service,
        existing=existing,
        enabled=enabled,
        secret_value=secret_value,
        config_json=config_json,
        validation=validation,
    )
    supabase_client.upsert_user_integration_connection(payload)
    return get_user_integration_state(user_id, service)


def validate_user_integration(
    user_id: str,
    service: str,
    *,
    credentials: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if not is_user_managed(service):
        raise KeyError(service)
    existing = supabase_client.get_user_integration_connection(user_id, service) or {}
    secret_value, config_json = normalize_credentials(service, credentials) if credentials else (None, None)
    if secret_value is None:
        secret_value = secret_store.decrypt_secret(existing.get("secret_ciphertext"))
    if config_json is None:
        config_json = existing.get("config_json") or {}
    if not secret_value:
        raise IntegrationValidationError("Credentials are required before validation.")

    validation = validate_credentials(service, secret_value=secret_value, config_json=config_json)
    payload = _build_update_payload(
        user_id=user_id,
        service=service,
        existing=existing,
        enabled=bool(existing.get("enabled")),
        secret_value=secret_value if credentials else None,
        config_json=config_json,
        validation=validation,
    )
    supabase_client.upsert_user_integration_connection(payload)
    return get_user_integration_state(user_id, service)


def delete_user_credentials(user_id: str, service: str) -> dict[str, Any]:
    if not is_user_managed(service):
        raise KeyError(service)
    supabase_client.upsert_user_integration_connection(
        {
            "user_id": user_id,
            "service": service,
            "enabled": False,
            "status": "never_validated",
            "config_json": {},
            "secret_ciphertext": None,
            "last_validated_at": None,
            "last_error": None,
        }
    )
    return get_user_integration_state(user_id, service)


def save_persona_integration(
    *,
    persona_id: str,
    actor_user_id: str,
    service: str,
    enabled: bool,
    credentials: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if not is_user_managed(service):
        raise KeyError(service)
    existing = (
        supabase_client.get_persona_integration_connection(persona_id, service)
        or {}
    )
    if service == "deepseek":
        if not enabled:
            return delete_persona_credentials(
                persona_id=persona_id,
                actor_user_id=actor_user_id,
                service=service,
            )
        secret_value, model_binding = normalize_credentials(service, credentials)
        if not secret_value:
            raise IntegrationValidationError("DeepSeek api_key is required.")
        validate_deepseek(secret_value, model=model_binding.get("model"))
        persona = supabase_client.get_persona_by_id(persona_id) or {}
        try:
            config_json = deepseek_n8n_service.provision(
                persona=persona,
                api_key=secret_value,
                previous_config=existing.get("config_json") or {},
                model_binding=model_binding,
            )
        except Exception as exc:
            raise IntegrationValidationError(
                f"DeepSeek/n8n provisioning failed: {exc}"
            ) from exc
        supabase_client.save_persona_integration_connection(
            {
                "persona_id": persona_id,
                "user_id": actor_user_id,
                "service": service,
                "enabled": True,
                "status": "connected",
                "config_json": config_json,
                "secret_ciphertext": None,
                "last_validated_at": _utcnow(),
                "last_error": None,
            }
        )
        return get_persona_integration_state(persona_id, service)
    secret_value, config_json = (
        normalize_credentials(service, credentials)
        if credentials
        else (None, None)
    )
    if enabled:
        if secret_value is None:
            secret_value = secret_store.decrypt_secret(existing.get("secret_ciphertext"))
        if not secret_value:
            raise IntegrationValidationError(
                "Credentials are required before enabling this integration."
            )
        if config_json is None:
            config_json = existing.get("config_json") or {}
        validation = validate_credentials(
            service,
            secret_value=secret_value,
            config_json=config_json,
        )
    else:
        validation = {
            "status": "disabled",
            "last_error": None,
            "last_validated_at": existing.get("last_validated_at"),
        }
    payload = _build_update_payload(
        user_id=actor_user_id,
        service=service,
        existing=existing,
        enabled=enabled,
        secret_value=secret_value,
        config_json=config_json,
        validation=validation,
    )
    payload["persona_id"] = persona_id
    supabase_client.save_persona_integration_connection(payload)
    return get_persona_integration_state(persona_id, service)


def validate_persona_integration(
    *,
    persona_id: str,
    actor_user_id: str,
    service: str,
    credentials: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if not is_user_managed(service):
        raise KeyError(service)
    existing = (
        supabase_client.get_persona_integration_connection(persona_id, service)
        or {}
    )
    if service == "deepseek":
        ok, latency = n8n_client.ping()
        if not existing.get("enabled") or not (
            existing.get("config_json") or {}
        ).get("n8n_credential_id"):
            raise IntegrationValidationError("DeepSeek is not provisioned in n8n.")
        if not ok:
            raise IntegrationValidationError("n8n is unavailable.")
        wiring = deepseek_n8n_service.check_workflow_wiring(existing.get("config_json") or {})
        if not wiring["ok"]:
            existing["status"] = "error"
            existing["last_error"] = wiring["reason"]
            existing["last_validated_at"] = _utcnow()
            supabase_client.save_persona_integration_connection(existing)
            supabase_client.insert_event(
                {
                    "event_type": "deepseek.workflow_wiring_invalid",
                    "entity_type": "persona",
                    "entity_id": persona_id,
                    "persona_id": persona_id,
                    "payload": {
                        "reason": wiring["reason"],
                        "diagnostics": wiring.get("diagnostics") or {},
                    },
                },
                level="error",
                source="services.integration_service",
            )
            raise IntegrationValidationError(wiring["reason"])
        existing["last_validated_at"] = _utcnow()
        existing["status"] = "connected"
        existing["last_error"] = None
        supabase_client.save_persona_integration_connection(existing)
        state = get_persona_integration_state(persona_id, service)
        state["response_ms"] = latency
        state["workflow_diagnostics"] = wiring.get("diagnostics") or {}
        supabase_client.insert_event(
            {
                "event_type": "deepseek.workflow_wiring_valid",
                "entity_type": "persona",
                "entity_id": persona_id,
                "persona_id": persona_id,
                "payload": {
                    "response_ms": latency,
                    "diagnostics": wiring.get("diagnostics") or {},
                },
            },
            source="services.integration_service",
        )
        return state
    secret_value, config_json = (
        normalize_credentials(service, credentials)
        if credentials
        else (None, None)
    )
    if secret_value is None:
        secret_value = secret_store.decrypt_secret(existing.get("secret_ciphertext"))
    if config_json is None:
        config_json = existing.get("config_json") or {}
    if not secret_value:
        raise IntegrationValidationError("Credentials are required before validation.")
    validation = validate_credentials(
        service,
        secret_value=secret_value,
        config_json=config_json,
    )
    payload = _build_update_payload(
        user_id=actor_user_id,
        service=service,
        existing=existing,
        enabled=bool(existing.get("enabled")),
        secret_value=secret_value if credentials else None,
        config_json=config_json,
        validation=validation,
    )
    payload["persona_id"] = persona_id
    supabase_client.save_persona_integration_connection(payload)
    return get_persona_integration_state(persona_id, service)


def delete_persona_credentials(
    *,
    persona_id: str,
    actor_user_id: str,
    service: str,
) -> dict[str, Any]:
    if not is_user_managed(service):
        raise KeyError(service)
    if service == "deepseek":
        existing = (
            supabase_client.get_persona_integration_connection(persona_id, service)
            or {}
        )
        deepseek_n8n_service.revoke(existing.get("config_json") or {})
    supabase_client.save_persona_integration_connection(
        {
            "persona_id": persona_id,
            "user_id": actor_user_id,
            "service": service,
            "enabled": False,
            "status": "never_validated",
            "config_json": {},
            "secret_ciphertext": None,
            "last_validated_at": None,
            "last_error": None,
        }
    )
    return get_persona_integration_state(persona_id, service)


def get_enabled_persona_secret(persona_id: str, service: str) -> Optional[str]:
    row = (
        supabase_client.get_persona_integration_connection(persona_id, service)
        or {}
    )
    if not row.get("enabled"):
        return None
    return secret_store.decrypt_secret(row.get("secret_ciphertext"))


def system_service_has_runtime_credentials(service: str) -> bool:
    if service == "n8n":
        return bool((os.environ.get("N8N_BASE_URL") or "").strip() and (os.environ.get("N8N_API_KEY") or "").strip())
    if service == "supabase":
        return bool((os.environ.get("SUPABASE_URL") or "").strip() and (os.environ.get("SUPABASE_SERVICE_KEY") or "").strip())
    if service == "openai":
        return bool((os.environ.get("OPENAI_API_KEY") or "").strip())
    if service == "anthropic":
        return bool((os.environ.get("ANTHROPIC_API_KEY") or "").strip())
    return True
