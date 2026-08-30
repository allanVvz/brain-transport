import os
import re
import base64
import time
import unicodedata
import json
from datetime import datetime, timedelta, timezone
import httpx
from supabase import create_client, Client, ClientOptions
from typing import Any, Optional

from services.public_site import DEFAULT_FORMATS

_client: Optional[Client] = None
_UNSET = object()
EXPECTED_DB_ROLE = "brain_transport"
_TRANSIENT_ERROR_MARKERS = (
    "Server disconnected",
    "RemoteProtocolError",
    "ReadError",
    "ConnectError",
    "TimeoutException",
    "Connection reset",
)


def _supabase_ssl_verify() -> bool:
    raw = (os.environ.get("SUPABASE_SSL_VERIFY") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    runtime = (os.environ.get("ENV") or os.environ.get("PYTHON_ENV") or "").strip().lower()
    return runtime == "production"


def _validated_db_jwt() -> str:
    token = (os.environ.get("BRAIN_DB_JWT") or "").strip()
    try:
        payload_segment = token.split(".")[1]
        padding = "=" * (-len(payload_segment) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_segment + padding))
    except (IndexError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("BRAIN_DB_JWT must be a valid role-scoped JWT") from exc
    role = str(payload.get("role") or "")
    if role != EXPECTED_DB_ROLE:
        raise RuntimeError(
            f"BRAIN_DB_JWT role must be {EXPECTED_DB_ROLE!r}, got {role!r}"
        )
    return token


def get_client() -> Client:
    global _client
    if _client is None:
        if (os.environ.get("SUPABASE_OFFLINE") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise RuntimeError(
                "Supabase access is disabled for this deterministic test run"
            )
        timeout_seconds = float(os.environ.get("SUPABASE_HTTP_TIMEOUT_SECONDS") or "120")
        http_client = httpx.Client(verify=_supabase_ssl_verify(), timeout=timeout_seconds)
        _client = create_client(
            os.environ["SUPABASE_URL"],
            _validated_db_jwt(),
            options=ClientOptions(httpx_client=http_client),
        )
    return _client


def _reset_client() -> None:
    global _client
    _client = None


def _is_transient_transport_error(exc: Exception) -> bool:
    text = f"{type(exc).__module__}.{type(exc).__name__}: {exc}"
    return any(marker in text for marker in _TRANSIENT_ERROR_MARKERS)


def _execute_with_retry(query, retries: int = 4):
    """Run a PostgREST query with exponential backoff on transient transport errors.

    Supabase Edge / PostgREST occasionally drops connections under load
    ("Server disconnected", "RemoteProtocolError"). Retries with a fresh client
    have proven to recover most of these without operator-visible failure.
    """
    retries = int(os.environ.get("SUPABASE_RETRY_ATTEMPTS") or retries)
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return query.execute()
        except Exception as exc:
            last_exc = exc
            if not _is_transient_transport_error(exc) or attempt >= retries:
                raise
            _reset_client()
            # Backoff: 0.25, 0.5, 1.0, 2.0, 4.0 seconds. Caps at ~7.75s total.
            time.sleep(min(0.25 * (2 ** attempt), 4.0))
    if last_exc:
        raise last_exc
    return None


# â”€â”€ Safe query helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# All public functions use _q() / _one() so that:
#   â€¢ A None result never causes AttributeError
#   â€¢ A missing table returns a safe default instead of a 500

def _q(query) -> list:
    """Execute a list query; return [] on None or any exception."""
    try:
        result = _execute_with_retry(query)
        if result is None:
            return []
        return result.data or []
    except Exception as exc:
        try:
            from services import sre_logger
            sre_logger.error("supabase_client", f"query failed: {exc}", exc)
        except Exception:
            pass
        return []


def _one(query) -> Optional[dict]:
    """Execute a single-row query (maybe_single); return None on error."""
    try:
        result = _execute_with_retry(query)
        if result is None:
            return None
        return result.data
    except Exception as exc:
        try:
            from services import sre_logger
            sre_logger.error("supabase_client", f"query failed: {exc}", exc)
        except Exception:
            pass
        return None

_LEADS_MISSING_COLUMNS: set[str] = set()

def update_lead(lead_ref: int, data: dict) -> None:
    _execute_with_retry(get_client().table("leads").update(data).eq("id", lead_ref))

def get_lead_by_ref(lead_ref: int) -> Optional[dict]:
    """Fetch a lead row by its integer primary key (`leads.id`)."""
    return _one(get_client().table("leads").select("*").eq("id", lead_ref).maybe_single())


def get_leads_by_refs(lead_refs: list[int], *, chunk_size: int = 100) -> dict[int, dict]:
    """Fetch lead snapshots in bounded batches for conversation decoration.

    Keeping each ``in`` list small avoids the oversized PostgREST URLs seen
    in production while replacing one request per conversation with one or
    two bounded requests.
    """
    refs = sorted({int(value) for value in lead_refs if value is not None})
    rows: dict[int, dict] = {}
    for index in range(0, len(refs), max(1, min(chunk_size, 100))):
        chunk = refs[index:index + max(1, min(chunk_size, 100))]
        for row in _q(get_client().table("leads").select("*").in_("id", chunk)):
            if row.get("id") is not None:
                rows[int(row["id"])] = row
    return rows

def get_audience(audience_id: str) -> Optional[dict]:
    return _one(get_client().table("audiences").select("*").eq("id", audience_id).maybe_single())


def get_audience_by_slug(persona_id: str, audience_slug: str) -> Optional[dict]:
    if not persona_id or not audience_slug:
        return None
    return _one(
        get_client()
        .table("audiences")
        .select("*")
        .eq("persona_id", persona_id)
        .eq("slug", audience_slug)
        .maybe_single()
    )

def lead_has_membership(lead_id: int, persona_id: str, audience_id: Optional[str] = None) -> bool:
    rows = _q(
        get_client()
        .table("lead_audience_memberships")
        .select("lead_id,audience_id")
        .eq("lead_id", lead_id)
        .limit(500)
    )
    if not rows:
        return False
    audience_ids = [row.get("audience_id") for row in rows if row.get("audience_id")]
    if not audience_ids:
        return False
    audience_q = get_client().table("audiences").select("id").in_("id", audience_ids).eq("persona_id", persona_id)
    if audience_id:
        audience_q = audience_q.eq("id", audience_id)
    return bool(_q(audience_q.limit(1)))


def _audience_ids_for_persona(persona_id: str, audience_id: Optional[str] = None, audience_slug: Optional[str] = None) -> list[str]:
    if audience_id:
        audience = get_audience(audience_id)
        return [audience["id"]] if audience and audience.get("persona_id") == persona_id else []
    if audience_slug:
        audience = get_audience_by_slug(persona_id, audience_slug)
        return [audience["id"]] if audience else []
    rows = _q(get_client().table("audiences").select("id").eq("persona_id", persona_id))
    return [row.get("id") for row in rows if row.get("id")]


def get_lead_refs_for_audience_scope(
    *,
    persona_id: str,
    audience_id: Optional[str] = None,
    audience_slug: Optional[str] = None,
) -> list[int]:
    audience_ids = _audience_ids_for_persona(persona_id, audience_id=audience_id, audience_slug=audience_slug)
    if not audience_ids:
        return []
    rows = _q(
        get_client()
        .table("lead_audience_memberships")
        .select("lead_id")
        .in_("audience_id", audience_ids)
        .limit(5000)
    )
    return sorted({int(row["lead_id"]) for row in rows if row.get("lead_id") is not None})

# â”€â”€ Messages â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_messages(lead_id: str, limit: int = 30) -> list:
    """
    Fetch the most recent `limit` messages for a lead, returned in ascending
    (chronological chat) order.

    The DB query orders descending so `.limit()` keeps the newest rows, then
    `_sort_messages_for_chat` flips them back to display order. Querying
    ascending-then-limit (the previous behavior) silently returned the
    *oldest* `limit` messages for any lead with more history than that —
    every caller expecting recent context (AI conversation history, the
    bot-echo-loop guard) was reading from the start of the conversation
    instead. Confirmed live 2026-08-02: a lead with 143 messages had its
    echo-loop guard permanently matching a 6-day-old outbound reply because
    that old row never left the first-20-messages window.
    The self-hosted schema uses ``messages.lead_id``.  ``lead_ref`` remains a
    response compatibility alias through ``_normalize_message_row``.
    """
    client = get_client()

    # Primary: lead_id in the self-hosted/local-first schema.
    try:
        import re as _re
        digits = _re.sub(r"\D", "", lead_id or "")
        if digits and len(digits) <= 10:
            rows = _q(
                client.table("messages")
                .select("*")
                .eq("lead_id", int(digits))
                .order("created_at", desc=True)
                .order("id", desc=True)
                .limit(limit)
            )
            if rows:
                return _sort_messages_for_chat([_normalize_message_row(row) for row in rows])
    except Exception:
        pass

    # Name lookup belongs to leads; messages do not carry a duplicated nome.
    if lead_id and not lead_id.isdigit():
        lead = _one(client.table("leads").select("id").eq("nome", lead_id).maybe_single())
        if lead and lead.get("id") is not None:
            rows = _q(
                client.table("messages")
                .select("*")
                .eq("lead_id", lead["id"])
                .order("created_at", desc=True)
                .order("id", desc=True)
                .limit(limit)
            )
            return _sort_messages_for_chat([_normalize_message_row(row) for row in rows])

    return []


def get_messages_page(
    lead_ref: int, *, limit: int = 50,
    after_created_at: str | None = None, after_id: int | None = None,
    before_created_at: str | None = None, before_id: int | None = None,
) -> list:
    result = get_client().rpc(
        "messages_page",
        {
            "p_lead_id": lead_ref,
            "p_limit": max(1, min(int(limit), 100)) + 1,
            "p_after_created_at": after_created_at,
            "p_after_id": after_id,
            "p_before_created_at": before_created_at,
            "p_before_id": before_id,
        },
    ).execute()
    rows = [_normalize_message_row(row) for row in (result.data or [])]
    return _sort_messages_for_chat(_hydrate_message_media_asset_refs(rows))


def _hydrate_message_media_asset_refs(rows: list[dict]) -> list[dict]:
    """Project persisted inbound assets onto their conversation messages.

    ``assets.message_id`` is the canonical relationship. Mirroring ``asset_id``
    into message metadata makes the dashboard renderer cheap, but the read path
    must not depend on that denormalized write succeeding during webhook ingest.
    """
    message_ids = [
        int(row["id"])
        for row in rows
        if row.get("id") is not None
        and isinstance(row.get("metadata"), dict)
        and (
            (row.get("metadata") or {}).get("media")
            or (row.get("metadata") or {}).get("asset_id")
        )
    ]
    if not message_ids:
        return rows
    assets = _q(
        get_client().table("assets")
        .select("id,message_id,status")
        .in_("message_id", message_ids)
        .eq("upload_context", "whatsapp_inbound")
    )
    return _project_message_media_asset_refs(rows, assets)


def _project_message_media_asset_refs(
    rows: list[dict], assets: list[dict],
) -> list[dict]:
    """Pure projection used by the dashboard read path and regression tests."""
    by_message_id = {
        int(asset["message_id"]): asset
        for asset in assets
        if asset.get("message_id") is not None and asset.get("id")
    }
    hydrated: list[dict] = []
    for row in rows:
        asset = by_message_id.get(int(row["id"])) if row.get("id") is not None else None
        if not asset:
            hydrated.append(row)
            continue
        metadata = {
            **(row.get("metadata") or {}),
            "asset_id": str(asset["id"]),
            "media_asset_status": asset.get("status"),
        }
        hydrated.append({**row, "metadata": metadata})
    return hydrated


def _sort_messages_for_chat(rows: list) -> list:
    """Return chat messages in human-readable order.

    Some WhatsApp/n8n flows persist the assistant reply row milliseconds
    before the inbound row that triggered it. Those rows share the same
    WhatsApp id, with the reply stored as `ai_reply.<wamid>`. For display and
    API consumers, the inbound message must come before its generated reply.
    """
    from datetime import datetime

    def parse_ts(value: str | None) -> float:
        if not value:
            return 0.0
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.0

    base_by_message_id = {
        row.get("message_id"): row
        for row in rows
        if row.get("message_id") and not str(row.get("message_id")).startswith("ai_reply.")
    }

    def row_id(row: dict) -> int:
        try:
            return int(row.get("id") or 0)
        except Exception:
            return 0

    def key(row: dict):
        message_id = str(row.get("message_id") or "")
        own_ts = parse_ts(row.get("created_at"))
        own_id = row_id(row)
        if message_id.startswith("ai_reply."):
            base = base_by_message_id.get(message_id.removeprefix("ai_reply."))
            if base:
                return (parse_ts(base.get("created_at")), row_id(base), 1, own_ts, own_id)
        return (own_ts, own_id, 0, own_ts, own_id)

    return sorted(rows, key=key)

def _normalize_message_row(row: dict) -> dict:
    normalized = dict(row or {})
    if "texto" not in normalized and normalized.get("content") is not None:
        normalized["texto"] = normalized.get("content")
    if "canal" not in normalized and normalized.get("channel") is not None:
        normalized["canal"] = normalized.get("channel")
    if "lead_ref" not in normalized and normalized.get("lead_id") is not None:
        normalized["lead_ref"] = normalized.get("lead_id")
    if "sender_type" not in normalized and normalized.get("role") is not None:
        role = str(normalized.get("role") or "").lower()
        normalized["sender_type"] = "client" if role in {"user", "client", "human"} else "ai"
    if "message_id" not in normalized and normalized.get("sender_id") is not None:
        normalized["message_id"] = normalized.get("sender_id")
    return normalized


def get_recent_messages(hours: int = 24, limit: int = 500, persona_id: Optional[str] = None, lead_refs: Optional[list[int]] = None) -> list:
    from datetime import datetime, timedelta
    since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    client = get_client()
    q = (
        client.table("messages")
        .select("*")
        .gte("created_at", since)
        .order("created_at", desc=True)
        .limit(limit)
    )
    if lead_refs is not None:
        if not lead_refs:
            return []
        q = q.in_("lead_id", lead_refs)
    elif persona_id:
        leads = _q(
            client.table("leads")
            .select("id")
            .eq("persona_id", persona_id)
        )
        lead_refs = [lead.get("id") for lead in leads if lead.get("id") is not None]
        if not lead_refs:
            return []
        q = q.in_("lead_id", lead_refs)
    return [_normalize_message_row(row) for row in _q(q)]


def get_conversations(hours: int = 168, limit: int = 1000, persona_id: Optional[str] = None, lead_refs: Optional[list[int]] = None) -> list:
    """
    Returns the last message per unique conversation.

    ``messages.lead_id`` is the canonical key.  The response retains
    ``lead_ref`` for dashboard compatibility.
    """
    from datetime import datetime, timedelta
    since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    client = get_client()
    requested_lead_refs = list(lead_refs) if lead_refs is not None else None
    if lead_refs is None and persona_id:
        scoped_leads = _q(client.table("leads").select("id").eq("persona_id", persona_id))
        lead_refs = [lead.get("id") for lead in scoped_leads if lead.get("id") is not None]
        if not lead_refs:
            return []

    messages_q = (
        client.table("messages")
        .select("id,lead_id,role,content,created_at,direction,status,channel,sender_id")
        .gte("created_at", since)
        .order("created_at", desc=True)
        .limit(limit)
    )
    if lead_refs is not None:
        if not lead_refs:
            return []
        messages_q = messages_q.in_("lead_id", lead_refs)
    rows = [_normalize_message_row(row) for row in _q(messages_q)]
    lead_refs = sorted({row.get("lead_ref") for row in rows if row.get("lead_ref") is not None})
    leads_by_ref: dict = {}
    for idx in range(0, len(lead_refs), 200):
        chunk = lead_refs[idx:idx + 200]
        for lead in _q(
            client.table("leads")
            .select("id,lead_id,nome,persona_id,stage,interesse_produto")
            .in_("id", chunk)
        ):
            leads_by_ref[lead.get("id")] = lead

    seen: dict = {}
    for row in rows:
        lead_ref = row.get("lead_ref")
        lead = leads_by_ref.get(lead_ref) or {}
        if persona_id and lead.get("persona_id") != persona_id and requested_lead_refs is None:
            continue
        key = f"lead:{lead_ref}" if lead_ref is not None else f"message:{row.get('id') or 'unknown'}"
        if key not in seen:
            seen[key] = {
                "key": key,
                "nome": lead.get("nome") or key,
                "lead_id": lead.get("lead_id"),
                "lead_ref": lead_ref,
                "persona_id": lead.get("persona_id"),
                "interesse_produto": lead.get("interesse_produto"),
                "Lead_Stage": lead.get("stage") or "novo",
                "last_message": row.get("texto") or row.get("content") or "",
                "last_direction": row.get("direction") or "",
                "last_sender_type": row.get("sender_type") or "",
                "last_at": row.get("created_at") or "",
            }
    return list(seen.values())


def insert_message(data: dict) -> None:
    client = get_client()
    sender_type = str(data.get("sender_type") or "").lower()
    direction = str(data.get("direction") or "").lower()
    role = data.get("role")
    if not role:
        role = "assistant" if sender_type in {"ai", "assistant"} or direction == "outbound" else "user"

    mapped = {
        "lead_id": data.get("lead_id") or data.get("lead_ref"),
        "role": role,
        "content": data.get("content") or data.get("texto") or "",
        "direction": data.get("direction"),
        "status": data.get("status"),
        "channel": data.get("channel") or data.get("canal"),
        "sender_id": data.get("sender_id") or data.get("message_id"),
        "whatsapp_phone_number_id": data.get("whatsapp_phone_number_id"),
        "external_message_id": data.get("external_message_id"),
        "channel_binding_id": data.get("channel_binding_id"),
        "correlation_id": data.get("correlation_id"),
        "metadata": data.get("metadata"),
        "created_at": data.get("created_at"),
    }
    mapped = {k: v for k, v in mapped.items() if v is not None}
    try:
        _execute_with_retry(client.table("messages").insert(mapped))
        return
    except Exception as exc:
        text = str(exc)
        if data.get("external_message_id") and any(
            marker in text.lower() for marker in ("duplicate", "unique", "23505")
        ):
            return
        current_column_mismatch = any(
            marker in text
            for marker in (
                "messages.lead_id does not exist",
                "messages.role does not exist",
                "messages.content does not exist",
                "messages.channel does not exist",
                "messages.sender_id does not exist",
            )
        )
        if not current_column_mismatch:
            raise
    # Compatibility fallback for a legacy remote schema.
    _execute_with_retry(client.table("messages").insert(data))


# â”€â”€ Knowledge Graph: nodes & edges (migration 008) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# All functions are defensive: missing tables (e.g., migration 008 not applied)
# return safe defaults so the rest of the system keeps working.

_KG_TABLES_MISSING = False  # flipped to True on PGRST205 to short-circuit

def _storage_signed_url(bucket: str | None, path: str | None, expires_in: int = 86400) -> Optional[str]:
    if not bucket or not path:
        return None
    try:
        signed = get_client().storage.from_(bucket).create_signed_url(path, expires_in)
        signed_url = signed.get("signedURL") if isinstance(signed, dict) else getattr(signed, "signed_url", None) or getattr(signed, "signedURL", None)
        if not signed_url:
            return None
        internal_base = (os.environ.get("SUPABASE_URL") or "").rstrip("/")
        public_base = (os.environ.get("SUPABASE_PUBLIC_URL") or internal_base).rstrip("/")
        if signed_url.startswith("http"):
            if internal_base and signed_url.startswith(internal_base):
                return f"{public_base}{signed_url[len(internal_base):]}"
            return signed_url
        base = public_base
        if signed_url.startswith("/object"):
            return f"{base}/storage/v1{signed_url}"
        return f"{base}{signed_url}"
    except Exception:
        return None

# â”€â”€ Registries (migration 009) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Cached in-memory with short TTL â€” config rarely changes and the graph
# endpoint reads them on every request.

_REGISTRY_TTL_SECONDS = 300
_NODE_TYPE_REGISTRY_CACHE: tuple[float, list[dict]] | None = None
_RELATION_TYPE_REGISTRY_CACHE: tuple[float, list[dict]] | None = None

# Defensive fallback: mirrors the seed inserts of migration 009.
# Used when the table is missing or empty (009 partially applied) so the
# graph endpoint still returns useful level/color/icon hints.
_NODE_TYPE_REGISTRY_FALLBACK: list[dict] = [
    {"node_type": "persona",        "label": "Persona",   "default_level":  0, "default_importance": 1.00, "color": "#7c6fff", "icon": "user",        "sort_order":  0},
    {"node_type": "entity",         "label": "Entidade",  "default_level": 10, "default_importance": 0.95, "color": "#7c6fff", "icon": "network",     "sort_order": 10},
    {"node_type": "brand",          "label": "Brand",     "default_level": 20, "default_importance": 0.90, "color": "#a78bfa", "icon": "badge",       "sort_order": 20},
    {"node_type": "campaign",       "label": "Campanha",  "default_level": 30, "default_importance": 0.80, "color": "#fb923c", "icon": "megaphone",   "sort_order": 30},
    {"node_type": "product",        "label": "Produto",   "default_level": 40, "default_importance": 0.85, "color": "#60a5fa", "icon": "box",         "sort_order": 40},
    {"node_type": "offer",          "label": "Oferta",    "default_level": 45, "default_importance": 0.78, "color": "#38bdf8", "icon": "badge-dollar-sign", "sort_order": 45},
    {"node_type": "briefing",       "label": "Briefing",  "default_level": 50, "default_importance": 0.75, "color": "#c084fc", "icon": "file-text",   "sort_order": 50},
    {"node_type": "audience",       "label": "AudiÃªncia", "default_level": 55, "default_importance": 0.70, "color": "#f472b6", "icon": "users",       "sort_order": 55},
    {"node_type": "tone",           "label": "Tom",       "default_level": 60, "default_importance": 0.70, "color": "#22d3ee", "icon": "palette",     "sort_order": 60},
    {"node_type": "rule",           "label": "Regra",     "default_level": 65, "default_importance": 0.80, "color": "#f87171", "icon": "scale",       "sort_order": 65},
    {"node_type": "copy",           "label": "Copy",      "default_level": 70, "default_importance": 0.65, "color": "#64748b", "icon": "text",        "sort_order": 70},
    {"node_type": "faq",            "label": "FAQ",       "default_level": 75, "default_importance": 0.45, "color": "#4ade80", "icon": "circle-help", "sort_order": 75},
    {"node_type": "asset",          "label": "Asset",     "default_level": 80, "default_importance": 0.55, "color": "#f59e0b", "icon": "image",       "sort_order": 80},
    {"node_type": "gallery",        "label": "Gallery",   "default_level":112, "default_importance": 0.82, "color": "#f0abfc", "icon": "images",      "sort_order":112},
    {"node_type": "embedded",       "label": "Golden Dataset", "default_level":120, "default_importance": 0.78, "color": "#ffffff", "icon": "database",    "sort_order":120},
    {"node_type": "tag",            "label": "Tag",       "default_level": 90, "default_importance": 0.30, "color": "#94a3b8", "icon": "tag",         "sort_order": 90},
    {"node_type": "mention",        "label": "MenÃ§Ã£o",    "default_level": 92, "default_importance": 0.25, "color": "#94a3b8", "icon": "at-sign",     "sort_order": 92},
    {"node_type": "knowledge_item", "label": "Fila",      "default_level": 95, "default_importance": 0.40, "color": "#94a3b8", "icon": "inbox",       "sort_order": 95},
    {"node_type": "kb_entry",       "label": "Golden Dataset Entry", "default_level": 95, "default_importance": 0.50, "color": "#94a3b8", "icon": "database",    "sort_order": 96},
]

_RELATION_TYPE_REGISTRY_FALLBACK: list[dict] = [
    {"relation_type": "belongs_to_persona", "label": "pertence Ã  persona", "inverse_label": "possui",        "default_weight": 1.00, "directional": True,  "sort_order":  10},
    {"relation_type": "defines_brand",      "label": "define brand",       "inverse_label": "Ã© definido por", "default_weight": 0.90, "directional": True,  "sort_order":  20},
    {"relation_type": "has_tone",           "label": "usa tom",            "inverse_label": "tom de",         "default_weight": 0.80, "directional": True,  "sort_order":  30},
    {"relation_type": "about_product",      "label": "sobre produto",      "inverse_label": "tem conhecimento", "default_weight": 0.85, "directional": True, "sort_order":  40},
    {"relation_type": "part_of_campaign",   "label": "parte da campanha",  "inverse_label": "contÃ©m",         "default_weight": 0.75, "directional": True,  "sort_order":  50},
    {"relation_type": "supports_campaign",  "label": "apoia campanha",     "inverse_label": "apoiada por",    "default_weight": 0.70, "directional": True,  "sort_order":  55},
    {"relation_type": "answers_question",   "label": "responde pergunta",  "inverse_label": "Ã© respondido por", "default_weight": 0.80, "directional": True, "sort_order":  60},
    {"relation_type": "supports_copy",      "label": "suporta copy",       "inverse_label": "Ã© suportado por", "default_weight": 0.70, "directional": True,  "sort_order":  70},
    {"relation_type": "uses_asset",         "label": "usa asset",          "inverse_label": "Ã© usado por",    "default_weight": 0.65, "directional": True,  "sort_order":  80},
    {"relation_type": "gallery_asset",      "label": "na gallery",         "inverse_label": "contÃ©m",         "default_weight": 0.90, "directional": True,  "sort_order":  82},
    {"relation_type": "briefed_by",         "label": "briefado por",       "inverse_label": "briefa",         "default_weight": 0.70, "directional": True,  "sort_order":  90},
    {"relation_type": "same_topic_as",      "label": "mesmo tÃ³pico",       "inverse_label": "mesmo tÃ³pico",   "default_weight": 0.45, "directional": False, "sort_order": 100},
    {"relation_type": "duplicate_of",       "label": "duplicado de",       "inverse_label": "tem duplicado",  "default_weight": 1.00, "directional": True,  "sort_order": 110},
    {"relation_type": "derived_from",       "label": "derivado de",        "inverse_label": "origina",        "default_weight": 0.90, "directional": True,  "sort_order": 120},
    {"relation_type": "contains",           "label": "contÃ©m",             "inverse_label": "contido em",     "default_weight": 0.75, "directional": True,  "sort_order": 130},
    {"relation_type": "has_tag",            "label": "tem tag",            "inverse_label": "marca",          "default_weight": 0.30, "directional": True,  "sort_order": 200},
    {"relation_type": "mentions",           "label": "menciona",           "inverse_label": "mencionado por", "default_weight": 0.30, "directional": True,  "sort_order": 210},
    {"relation_type": "visible_to_agent",   "label": "visÃ­vel para agente", "inverse_label": "vÃª",            "default_weight": 0.50, "directional": True,  "sort_order": 220},
]

def ping_supabase() -> tuple[bool, Optional[str]]:
    try:
        _execute_with_retry(get_client().table("app_users").select("id").limit(1))
        return True, None
    except Exception as exc:
        return False, str(exc)


def get_health_history(limit: int = 30) -> list:
    rows = _q(
        get_client().table("system_health")
        .select("*")
        .order("snapshot_at", desc=True)
        .limit(limit)
    )
    return list(reversed(rows))


# â”€â”€ Integration Status â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def upsert_integration_status(data: dict) -> None:
    client = get_client()
    persona_id = data.get("persona_id")
    service = data["service"]
    if persona_id is None:
        # maybe_single() throws 406 if duplicates exist â€” use limit(1) instead
        rows = client.table("integration_status").select("id").is_("persona_id", "null").eq("service", service).limit(1).execute()
        if rows.data:
            row_id = rows.data[0]["id"]
            client.table("integration_status").update(data).eq("id", row_id).execute()
        else:
            client.table("integration_status").insert(data).execute()
    else:
        client.table("integration_status").upsert(data, on_conflict="persona_id,service").execute()


def get_integration_statuses(persona_id: Optional[str] = None) -> list:
    client = get_client()
    q = client.table("integration_status").select("*").order("service").order("last_check", desc=True)
    if persona_id:
        q = q.eq("persona_id", persona_id)
    rows = _q(q)
    seen: set[str] = set()
    result = []
    for row in rows:
        key = f"{row.get('persona_id')}:{row['service']}"
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def list_user_integration_connections(user_id: str) -> list[dict[str, Any]]:
    if not user_id:
        return []
    return _q(
        get_client()
        .table("user_integration_connections")
        .select("*")
        .eq("user_id", user_id)
        .order("service")
    )


def get_user_integration_connection(user_id: str, service: str) -> Optional[dict[str, Any]]:
    if not user_id or not service:
        return None
    rows = _q(
        get_client()
        .table("user_integration_connections")
        .select("*")
        .eq("user_id", user_id)
        .eq("service", service)
        .limit(1)
    )
    return rows[0] if rows else None


def upsert_user_integration_connection(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    payload = dict(data or {})
    now_iso = datetime.now(timezone.utc).isoformat()
    payload.setdefault("config_json", {})
    payload["updated_at"] = now_iso
    payload.setdefault("created_at", now_iso)
    result = _execute_with_retry(
        get_client()
        .table("user_integration_connections")
        .upsert(payload, on_conflict="user_id,service")
    )
    rows = result.data or []
    if rows:
        return rows[0]
    return get_user_integration_connection(payload.get("user_id"), payload.get("service"))


def list_persona_integration_connections(persona_id: str) -> list[dict[str, Any]]:
    if not persona_id:
        return []
    return _q(
        get_client()
        .table("user_integration_connections")
        .select("*")
        .eq("persona_id", persona_id)
        .order("service")
    )


def get_persona_integration_connection(
    persona_id: str,
    service: str,
) -> Optional[dict[str, Any]]:
    if not persona_id or not service:
        return None
    rows = _q(
        get_client()
        .table("user_integration_connections")
        .select("*")
        .eq("persona_id", persona_id)
        .eq("service", service)
        .limit(1)
    )
    return rows[0] if rows else None


def save_persona_integration_connection(
    data: dict[str, Any],
) -> Optional[dict[str, Any]]:
    payload = dict(data or {})
    persona_id = str(payload.get("persona_id") or "")
    service = str(payload.get("service") or "")
    if not persona_id or not service:
        raise ValueError("persona_id and service are required")
    now_iso = datetime.now(timezone.utc).isoformat()
    payload.setdefault("config_json", {})
    payload["updated_at"] = now_iso
    existing = get_persona_integration_connection(persona_id, service)
    if existing:
        rows = (
            _execute_with_retry(
                get_client()
                .table("user_integration_connections")
                .update(payload)
                .eq("id", existing["id"])
            ).data
            or []
        )
        return rows[0] if rows else get_persona_integration_connection(persona_id, service)
    payload.setdefault("created_at", now_iso)
    rows = (
        _execute_with_retry(
            get_client().table("user_integration_connections").insert(payload)
        ).data
        or []
    )
    return rows[0] if rows else get_persona_integration_connection(persona_id, service)


# â”€â”€ Personas â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_personas() -> list:
    return _q(get_client().table("personas").select("*").eq("active", True))


def get_persona(slug: str) -> Optional[dict]:
    return _one(get_client().table("personas").select("*").eq("slug", slug).maybe_single())


def get_persona_by_id(persona_id: str) -> Optional[dict]:
    return _one(get_client().table("personas").select("*").eq("id", persona_id).maybe_single())

_PERSONA_ROUTING_FIELDS = (
    "process_mode",
    "outbound_webhook_url",
    "outbound_webhook_secret",
    "inbound_webhook_token",
)

_MISSING_COLUMN_RE = re.compile(r"Could not find the '([^']+)' column of '([^']+)'")

# â”€â”€ Agent Logs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_AGENT_LOGS_SCHEMA_MODE: Optional[str] = None


def _detect_agent_logs_schema_mode() -> str:
    global _AGENT_LOGS_SCHEMA_MODE
    if _AGENT_LOGS_SCHEMA_MODE:
        return _AGENT_LOGS_SCHEMA_MODE
    client = get_client()
    try:
        client.table("agent_logs").select("agent_type").limit(1).execute()
        _AGENT_LOGS_SCHEMA_MODE = "modern"
        return _AGENT_LOGS_SCHEMA_MODE
    except Exception as exc:
        text = str(exc)
        if "agent_type" in text and ("does not exist" in text or "42703" in text):
            _AGENT_LOGS_SCHEMA_MODE = "legacy"
            return _AGENT_LOGS_SCHEMA_MODE
    try:
        client.table("agent_logs").select("agent_name").limit(1).execute()
        _AGENT_LOGS_SCHEMA_MODE = "legacy"
    except Exception:
        _AGENT_LOGS_SCHEMA_MODE = "modern"
    return _AGENT_LOGS_SCHEMA_MODE

def insert_agent_log(data: dict) -> None:
    payload = dict(data or {})
    meta = payload.get("metadata") or {}
    level = str(
        meta.get("level")
        or ("ERROR" if str(payload.get("action") or "").startswith("[ERROR]") else "INFO")
    ).lower()
    legacy_payload = {
        "lead_id": payload.get("lead_id"),
        "persona_id": payload.get("persona_id"),
        "agent_name": payload.get("agent_type") or payload.get("agent_name") or meta.get("component") or "agent",
        "input": payload.get("input") if isinstance(payload.get("input"), dict) else (meta.get("input") or {}),
        "output": payload.get("output") if isinstance(payload.get("output"), dict) else {
            "action": payload.get("action"),
            "decision": payload.get("decision"),
            "metadata": meta,
        },
        "latency_ms": payload.get("latency_ms") or meta.get("latency_ms"),
        "model_used": payload.get("model_used") or meta.get("model_used"),
        "status": "error" if level == "error" else ("timeout" if level == "timeout" else "success"),
        "error_msg": payload.get("decision") if level == "error" else payload.get("error_msg"),
    }
    # Compose bootstraps the legacy table first and later expands it with the
    # modern columns. Insert the compatible superset first so NOT NULL legacy
    # fields are satisfied without intentionally generating a database error
    # for every log line.
    hybrid_payload = {
        **legacy_payload,
        **payload,
        "agent_name": legacy_payload["agent_name"],
        "status": legacy_payload["status"],
        "input": legacy_payload["input"],
        "output": legacy_payload["output"],
    }

    mode = _detect_agent_logs_schema_mode()
    attempts = (
        [hybrid_payload, payload, legacy_payload]
        if mode == "modern"
        else [legacy_payload, hybrid_payload, payload]
    )
    last_exc: Exception | None = None
    for candidate in attempts:
        try:
            _execute_with_retry(get_client().table("agent_logs").insert(candidate))
            return
        except Exception as exc:
            last_exc = exc
    if last_exc:
        raise last_exc

# Mirrors the CHECK constraint on knowledge_items.content_type from
# supabase/migrations/002_knowledge_platform.sql. Keep in sync if the constraint changes.
KNOWLEDGE_ITEM_CONTENT_TYPES: frozenset[str] = frozenset({
    # Canonical fractal types (migration 039 + knowledge_taxonomy).
    "persona", "brand", "briefing", "campaign", "audience",
    "product_group", "product", "offer", "copy", "faq", "gallery", "asset",
    # Non-canonical but still accepted as input (kept for backwards-compat).
    "prompt", "maker_material", "tone", "competitor",
    "rule", "entity", "other",
})

KNOWLEDGE_ITEM_STATUSES: frozenset[str] = frozenset({
    "pending", "approved", "rejected", "embedded", "needs_update", "pending_regeneration",
})

_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_KNOWLEDGE_ITEMS_MISSING_COLUMNS: set[str] = set()

_APPROVED_SNAPSHOTS_MISSING = False

def get_active_whatsapp_binding(persona_id: Optional[str]) -> Optional[dict]:
    if not persona_id:
        return None
    return _one(
        get_client().table("workflow_bindings").select("*")
        .eq("persona_id", persona_id)
        .eq("channel", "whatsapp")
        .eq("active", True)
        .maybe_single()
    )

def get_active_workflow_binding_by_phone_number_id(phone_number_id: str) -> Optional[dict]:
    """Resolve routing exclusively by the business number, never by lead phone."""
    if not phone_number_id:
        return None
    return _one(
        get_client().table("workflow_bindings").select("*")
        .eq("whatsapp_phone_number_id", phone_number_id).eq("active", True).maybe_single()
    )

def get_workflow_binding_by_id(binding_id: Optional[str]) -> Optional[dict]:
    if not binding_id:
        return None
    return _one(
        get_client().table("workflow_bindings").select("*")
        .eq("id", binding_id).maybe_single()
    )


def update_workflow_binding(binding_id: str, payload: dict) -> dict:
    from datetime import datetime, timezone
    update = {**payload, "updated_at": datetime.now(timezone.utc).isoformat()}
    rows = (
        get_client().table("workflow_bindings").update(update)
        .eq("id", binding_id).execute().data or []
    )
    return rows[0] if rows else {}


def ensure_channel_lead(
    *,
    persona_id: str,
    channel_binding_id: str,
    external_contact_id: str,
    display_name: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    existing = _one(
        get_client().table("leads").select("*")
        .eq("persona_id", persona_id)
        .eq("channel_binding_id", channel_binding_id)
        .eq("external_contact_id", external_contact_id)
        .maybe_single()
    )
    if existing:
        return existing
    # A lead may already exist for this same contact but only carry
    # `telefone` (digits-only), not `external_contact_id` — e.g. leads
    # created through the legacy /process route. Without this fallback,
    # every inbound webhook for that contact spawns a permanent duplicate
    # lead instead of continuing the existing conversation. Confirmed live
    # 2026-08-01 on a real Baita customer (two "Allan" leads, messages
    # split across them).
    normalized_phone = re.sub(r"\D", "", external_contact_id or "")
    if normalized_phone:
        phone_match = _one(
            get_client().table("leads").select("*")
            .eq("persona_id", persona_id)
            .eq("telefone", normalized_phone)
            .is_("external_contact_id", "null")
            .maybe_single()
        )
        if phone_match:
            update_lead(phone_match["id"], {
                "external_contact_id": external_contact_id,
                "channel_binding_id": channel_binding_id,
            })
            return {
                **phone_match,
                "external_contact_id": external_contact_id,
                "channel_binding_id": channel_binding_id,
            }
    payload = {
        "lead_id": f"channel:{channel_binding_id}:{external_contact_id}",
        "persona_id": persona_id,
        "channel_binding_id": channel_binding_id,
        "external_contact_id": external_contact_id,
        "nome": display_name,
        "stage": "novo",
        "origem": "whatsapp",
        "metadata": metadata or {},
    }
    try:
        rows = get_client().table("leads").insert(payload).execute().data or []
        return rows[0] if rows else payload
    except Exception as exc:
        if not any(token in str(exc).lower() for token in ("duplicate", "unique", "23505")):
            raise
        return _one(
            get_client().table("leads").select("*")
            .eq("persona_id", persona_id)
            .eq("channel_binding_id", channel_binding_id)
            .eq("external_contact_id", external_contact_id)
            .maybe_single()
        ) or {}


def debounce_available_at(seconds: int = 3) -> str:
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) + timedelta(seconds=max(0, seconds))).isoformat()

def enqueue_whatsapp_envelope(
    *,
    buffer: dict,
    message: dict,
) -> dict:
    """Atomically create or resolve a WhatsApp message + durable buffer."""
    result = get_client().rpc(
        "enqueue_whatsapp_envelope",
        {"p_buffer": buffer, "p_message": message},
    ).execute()
    payload = getattr(result, "data", None)
    if isinstance(payload, list):
        payload = payload[0] if payload else None
    if not isinstance(payload, dict) or not payload.get("buffer_id"):
        raise RuntimeError("enqueue_whatsapp_envelope returned an invalid result")
    return payload


def get_whatsapp_buffer_by_idempotency(idempotency_key: str) -> Optional[dict]:
    if not idempotency_key:
        return None
    return _one(
        get_client()
        .table("lead_buffer")
        .select("*")
        .eq("idempotency_key", idempotency_key)
        .maybe_single()
    )


def get_whatsapp_buffer(buffer_id: str) -> Optional[dict]:
    if not buffer_id:
        return None
    return _one(
        get_client()
        .table("lead_buffer")
        .select("id,direction,status,lead_ref")
        .eq("id", buffer_id)
        .maybe_single()
    )


def normalize_whatsapp_text(text: str | None) -> str:
    """Collapse whitespace and case so near-identical retries compare equal."""
    import re

    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def find_recent_duplicate_whatsapp_outbound(
    *,
    lead_ref: int,
    channel_binding_id: str,
    normalized_text: str,
    window_seconds: int,
) -> Optional[dict]:
    """Return the most recent outbound row for this lead/channel whose text
    normalizes to `normalized_text` and landed within `window_seconds`, or
    None. Row identity (idempotency_key/correlation_id) already guards a
    literal re-dispatch of the same send; this guards a *new* send whose
    content matches one already in flight or unconfirmed, generically for
    any persona/binding.
    """
    if not normalized_text or window_seconds <= 0:
        return None
    from datetime import datetime, timedelta, timezone

    since = (datetime.now(timezone.utc) - timedelta(seconds=window_seconds)).isoformat()
    result = (
        get_client()
        .table("lead_buffer")
        .select("id,payload,status,created_at")
        .eq("lead_ref", lead_ref)
        .eq("channel_binding_id", channel_binding_id)
        .eq("direction", "outbound")
        .gte("created_at", since)
        .order("created_at", desc=True)
        .limit(20)
        .execute()
    )
    for row in result.data or []:
        candidate = (row.get("payload") or {}).get("text")
        if normalize_whatsapp_text(candidate) == normalized_text:
            return row
    return None


def claim_whatsapp_buffer(worker_id: str, limit: int = 20, lease_seconds: int = 60) -> list[dict]:
    result = get_client().rpc("claim_whatsapp_buffer", {
        "p_worker": worker_id, "p_limit": limit, "p_lease_seconds": lease_seconds,
    }).execute()
    rows = result.data or []
    # Migration 113 preserves the SQL SETOF lead_buffer contract while
    # exposing the burst identity as convenient top-level worker fields.
    for row in rows:
        payload = row.get("payload") or {}
        for key in (
            "canonical_id", "burst_member_ids", "evidence_messages", "burst_version",
        ):
            if key in payload:
                row[key] = payload[key]
        row.setdefault("canonical_id", row.get("id"))
    return rows


def mark_whatsapp_attempt(buffer_id: str, worker_id: str, kind: str) -> bool:
    result = get_client().rpc(
        "mark_whatsapp_attempt",
        {"p_buffer_id": buffer_id, "p_worker": worker_id, "p_kind": kind},
    ).execute()
    payload = getattr(result, "data", None)
    if isinstance(payload, list):
        payload = payload[0] if payload else False
    return payload is True


def record_whatsapp_safety_violation(
    *,
    binding_id: str,
    lead_ref: int | None,
    violation_key: str,
    reason: str,
    level: str = "full",
) -> dict:
    result = get_client().rpc(
        "record_whatsapp_safety_violation",
        {
            "p_binding_id": binding_id,
            "p_lead_ref": lead_ref,
            "p_violation_key": violation_key,
            "p_reason": reason[:500],
            "p_level": level,
        },
    ).execute()
    payload = getattr(result, "data", None)
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    return payload if isinstance(payload, dict) else {}

def complete_whatsapp_buffer(buffer_id: str, status: str, error: str | None = None) -> None:
    from datetime import datetime, timezone
    # Keep the chat projection in step with terminal outbound outbox states.
    # Without this, an operator sees a forever "pending" bubble even though
    # lead_buffer contains the actionable failure reason.
    buffer = _one(
        get_client().table("lead_buffer")
        .select("direction,channel_binding_id,correlation_id")
        .eq("id", buffer_id)
        .maybe_single()
    ) or {}
    _execute_with_retry(get_client().table("lead_buffer").update({
        "status": status, "last_error": error, "locked_at": None, "locked_by": None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", buffer_id))
    if buffer.get("direction") == "outbound" and buffer.get("channel_binding_id") and buffer.get("correlation_id"):
        messages_query = (
            get_client().table("messages")
            .select("id,metadata")
            .eq("channel_binding_id", buffer["channel_binding_id"])
            .eq("correlation_id", buffer["correlation_id"])
            .not_.in_("status", ["sent", "delivered", "read"])
        )
        message_rows = _q(messages_query)
        _execute_with_retry(
            get_client().table("messages").update({"status": status})
            .eq("channel_binding_id", buffer["channel_binding_id"])
            .eq("correlation_id", buffer["correlation_id"])
            .not_.in_("status", ["sent", "delivered", "read"])
        )
        if error:
            for message in message_rows:
                merged_metadata = {
                    **(message.get("metadata") or {}),
                    "outbox_error": str(error)[:800],
                    "outbox_buffer_id": buffer_id,
                }
                _execute_with_retry(
                    get_client().table("messages").update({"metadata": merged_metadata})
                    .eq("id", message["id"])
                )


def release_whatsapp_buffer(buffer_id: str, status: str, *, delay_seconds: int, error: str | None, decrement_attempt: bool = False) -> None:
    from datetime import datetime, timedelta, timezone
    payload = {
        "status": status, "last_error": error, "locked_at": None, "locked_by": None,
        "available_at": (datetime.now(timezone.utc) + timedelta(seconds=max(0, delay_seconds))).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _execute_with_retry(get_client().table("lead_buffer").update(payload).eq("id", buffer_id))

def reconcile_committed_graph_inbound(buffer_id: str, reason: str) -> dict:
    result = get_client().rpc(
        "reconcile_committed_graph_inbound",
        {"p_buffer_id": buffer_id, "p_reason": reason},
    ).execute()
    payload = getattr(result, "data", None)
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    return payload if isinstance(payload, dict) else {}


def update_whatsapp_delivery_by_binding(
    binding_id: str,
    external_message_id: str,
    status: str,
) -> None:
    if not binding_id or not external_message_id:
        return
    raw_status = str(status).upper()
    # Provider PENDING is not an outbox instruction.  It is audit-only: an
    # external callback must never make a committed row dispatchable again.
    if raw_status in {"PENDING", "PENDING_SEND", "UNKNOWN", ""}:
        insert_event({"event_type": "whatsapp.delivery_ack_ignored", "entity_type": "workflow_binding",
                      "entity_id": binding_id, "payload": {"external_message_id": external_message_id,
                      "status": raw_status}}, source="whatsapp.delivery")
        return
    normalized = {
        "SERVER_ACK": "sent",
        "SENT": "sent",
        "DELIVERY_ACK": "delivered",
        "DELIVERED": "delivered",
        "READ": "read",
        "PLAYED": "read",
        "ERROR": "failed",
        "FAILED": "failed",
    }.get(raw_status)
    if not normalized:
        insert_event(
            {
                "event_type": "whatsapp.delivery_ack_ignored",
                "entity_type": "workflow_binding",
                "entity_id": binding_id,
                "payload": {
                    "external_message_id": external_message_id,
                    "status": raw_status,
                },
            },
            source="whatsapp.delivery",
        )
        return
    get_client().rpc(
        "reconcile_whatsapp_delivery",
        {
            "p_binding_id": binding_id, "p_external_message_id": external_message_id,
            "p_status": normalized,
        },
    ).execute()


def complete_whatsapp_outbound(
    buffer_id: str,
    *,
    binding_id: str,
    correlation_id: str,
    wamid: str | None,
    success: bool,
    error: str | None = None,
    execution_id: str | None = None,
) -> dict:
    result = get_client().rpc(
        "complete_whatsapp_outbound_result",
        {
            "p_buffer_id": buffer_id,
            "p_binding_id": binding_id,
            "p_correlation_id": correlation_id,
            "p_external_message_id": wamid,
            "p_success": success,
            "p_error": error,
            "p_execution_id": execution_id,
        },
    ).execute()
    payload = getattr(result, "data", None)
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    return payload if isinstance(payload, dict) else {}

def handoff_whatsapp_lead(lead_ref: int, *, level: str = "full") -> None:
    """Atomically set handoff_level and (for level='full') quarantine queued work."""
    _execute_with_retry(
        get_client().rpc(
            "handoff_whatsapp_lead", {"p_lead_ref": lead_ref, "p_level": level}
        )
    )

# â”€â”€ System Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Columns that exist in the physical system_events BASE TABLE.
# Any key not in this set is silently dropped before insert to prevent PGRST204.
_SYSTEM_EVENTS_COLUMNS = frozenset({
    "event_type", "entity_type", "entity_id",
    "persona_id", "payload", "level", "source",
})


def insert_event(
    data: dict,
    level: str = "info",
    source: Optional[str] = None,
) -> Optional[dict]:
    """
    Fire-and-forget event insert. Never raises â€” if the DB is unavailable
    the calling code continues uninterrupted.

    Only columns present in _SYSTEM_EVENTS_COLUMNS are forwarded so that
    adding extra keys to `data` never causes a PGRST204 schema-cache error.
    """
    try:
        row = {k: v for k, v in data.items() if k in _SYSTEM_EVENTS_COLUMNS}
        row.setdefault("payload", {})
        row.setdefault("level", level)
        if source:
            row["source"] = source
        result = get_client().table("system_events").insert(row).execute()
        return (result.data or [None])[0]
    except Exception as exc:
        try:
            from services import sre_logger
            sre_logger.error("supabase_client", f"insert_event failed: {exc}", exc)
        except Exception:
            pass
        return None

def update_pipeline_status(service: str, data: dict) -> None:
    get_client().table("pipeline_status").update(data).eq("service", service).execute()

# â”€â”€ Storage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def upload_to_storage(bucket: str, path: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """Upload bytes to Supabase Storage; returns the public URL."""
    client = get_client()
    client.storage.from_(bucket).upload(path, data, {"content-type": content_type, "upsert": "true"})
    return client.storage.from_(bucket).get_public_url(path)

def ensure_bucket(name: str, public: bool = False) -> bool:
    """Make sure a Supabase Storage bucket exists. Idempotent.

    Migration 033 tries to seed `assets-raw` / `assets-derived` via
    `INSERT INTO storage.buckets`, but the SQL path requires storage-admin
    privileges and silently misses on fresh projects. This helper closes the
    gap at boot time so /assets/upload never 502s on a missing bucket.

    Returns True if the bucket exists (created or pre-existing), False on
    failure. Never raises.
    """
    try:
        client = get_client()
        try:
            existing = client.storage.list_buckets() or []
        except Exception:
            existing = []
        names = {b.get("name") if isinstance(b, dict) else getattr(b, "name", None) for b in existing}
        if name in names:
            return True
        client.storage.create_bucket(name, options={"public": public})
        return True
    except Exception as exc:
        msg = str(exc).lower()
        # supabase-py raises StorageApiError with statusCode=409 / "already exists"
        # when the bucket exists but list_buckets() failed to enumerate it.
        if "already exists" in msg or "duplicate" in msg or "409" in msg:
            return True
        try:
            from services import sre_logger
            sre_logger.warn("supabase_client", f"ensure_bucket({name}) failed: {exc}")
        except Exception:
            pass
        return False


# â”€â”€ Assets / asset_readings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def insert_asset(data: dict) -> dict:
    result = get_client().table("assets").insert(data).execute()
    return (result.data or [{}])[0]


def update_asset(asset_id: str, patch: dict) -> dict:
    if not asset_id:
        return {}
    result = (
        get_client().table("assets")
        .update(patch)
        .eq("id", asset_id)
        .execute()
    )
    return (result.data or [{}])[0]

def insert_asset_reading(data: dict) -> dict:
    result = get_client().table("asset_readings").insert(data).execute()
    return (result.data or [{}])[0]


# ── Inbound WhatsApp media ───────────────────────────────────────────────
# Files a lead sends over WhatsApp land in the PRIVATE `whatsapp-media`
# bucket, never in the public `assets-raw` used by marketing uploads.
WHATSAPP_MEDIA_BUCKET = "whatsapp-media"


def _inbound_asset_type(kind: str, mime: Optional[str]) -> str:
    """Map a media descriptor onto the assets.type CHECK constraint.

    The constraint allows image|video|audio|pdf|text|copy|campaign|template;
    anything else a customer might attach (docx, xlsx, ...) is stored as
    `text`, which is the generic bucket rather than a claim about content.
    """
    if kind in ("image", "audio", "video"):
        return kind
    if (mime or "").split(";")[0].strip().lower() == "application/pdf":
        return "pdf"
    return "text"


def insert_inbound_media_asset(
    *,
    persona_id: str,
    lead_id: int,
    message_id: Optional[int],
    descriptor: dict,
    campaign_id: Optional[str] = None,
    campaign_recipient_id: Optional[str] = None,
) -> Optional[dict]:
    """Register a received attachment before its bytes have been fetched.

    Created in `status='reading'`: the row exists so the media ingest worker
    has something to claim, but nothing is downloadable yet. Returns None when
    the message already produced an asset (the unique partial index in
    migration 119), which is the expected outcome of a provider retry.
    """
    kind = str(descriptor.get("kind") or "document")
    filename = descriptor.get("filename") or f"{kind}-{lead_id}"
    try:
        return insert_asset({
            "persona_id": persona_id,
            "lead_id": lead_id,
            "message_id": message_id,
            "campaign_id": campaign_id,
            "campaign_recipient_id": campaign_recipient_id,
            "type": _inbound_asset_type(kind, descriptor.get("mime")),
            "name": filename,
            "source": "whatsapp",
            "upload_context": "whatsapp_inbound",
            "status": "reading",
            "storage_bucket": WHATSAPP_MEDIA_BUCKET,
            "mime_type": descriptor.get("mime"),
            "file_size": descriptor.get("size"),
            "original_filename": filename,
            "metadata": {
                "media": descriptor,
                "direction": "inbound",
                "reading_status": "pending",
                # Customer-sent media is never a marketing asset: it must not
                # acquire a landing slot or reach the public site.
                "validation_status": "not_applicable",
                "upload_context": "whatsapp_inbound",
            },
        })
    except Exception as exc:
        text = str(exc).lower()
        if "duplicate" in text or "unique" in text or "23505" in text:
            return None
        raise


def link_inbound_media_asset_to_message(message_row_id: int, asset_id: str) -> bool:
    """Idempotently mirror a canonical inbound asset onto message metadata."""
    if not message_row_id or not asset_id:
        return False
    row = _one(
        get_client().table("messages")
        .select("id,metadata")
        .eq("id", int(message_row_id))
        .maybe_single()
    ) or {}
    if not row:
        return False
    metadata = {**(row.get("metadata") or {}), "asset_id": str(asset_id)}
    result = (
        get_client().table("messages")
        .update({"metadata": metadata})
        .eq("id", int(message_row_id))
        .execute()
    )
    return bool(result.data)


def claim_pending_media_assets(limit: int = 10) -> list:
    """Assets whose bytes still need fetching and reading."""
    result = (
        get_client().table("assets")
        .select("*")
        .eq("upload_context", "whatsapp_inbound")
        .eq("status", "reading")
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return result.data or []


def resolve_media_buffer(
    buffer_id: str,
    text: str,
    *,
    reading_status: str = "completed",
    debounce_seconds: int = 3,
) -> dict:
    """Publish the extracted text and release the dispatch hold.

    Wraps the SQL function from migration 119 — it has to be one statement so
    the quiet-burst string_agg cannot race with it and lose the transcription.
    """
    result = get_client().rpc("resolve_media_buffer", {
        "p_buffer_id": buffer_id,
        "p_text": text,
        "p_reading_status": reading_status,
        "p_debounce_seconds": debounce_seconds,
    }).execute()
    payload = getattr(result, "data", None)
    if isinstance(payload, list):
        payload = payload[0] if payload else None
    return payload if isinstance(payload, dict) else {"resolved": False}
