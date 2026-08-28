import os
import re
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
            os.environ["SUPABASE_SERVICE_KEY"],
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


def _insert_one(query) -> dict:
    """Execute an insert and return the first row.

    Re-raises on any database error so callers see the real cause (CHECK violations,
    NOT NULL, FK, etc.) instead of receiving a silent {}. Returns {} only when the
    insert succeeded but PostgREST returned no row data â€” an anomalous shape that
    callers can recover from via a follow-up lookup.
    """
    try:
        result = _execute_with_retry(query)
    except Exception as exc:
        try:
            from services import sre_logger
            sre_logger.error("supabase_client", f"insert failed: {exc}", exc)
        except Exception:
            pass
        raise
    if result is None or not result.data:
        return {}
    return result.data[0]


def _slugify(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    return text.strip("-") or "item"


# â”€â”€ Leads â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_lead(lead_id: str) -> Optional[dict]:
    client = get_client()
    try:
        import re as _re
        digits = _re.sub(r"\D", "", lead_id or "")
        if digits and len(digits) <= 10:
            row = _one(client.table("leads").select("*").eq("id", int(digits)).maybe_single())
            if row:
                return row
    except Exception:
        pass
    return _one(client.table("leads").select("*").eq("lead_id", lead_id).maybe_single())


def _resolve_persona_id(persona_slug_or_id: Optional[str]) -> Optional[str]:
    if not persona_slug_or_id:
        return None
    if len(persona_slug_or_id) == 36 and persona_slug_or_id.count("-") == 4:
        return persona_slug_or_id
    persona = get_persona(persona_slug_or_id)
    return persona.get("id") if persona else None


_LEADS_MISSING_COLUMNS: set[str] = set()


def _missing_column_from_error(exc: Exception) -> Optional[str]:
    """Detect PGRST204 'Could not find the X column' and extract column name."""
    text = str(exc)
    if "PGRST204" not in text and "schema cache" not in text:
        return None
    import re as _re
    m = _re.search(r"Could not find the '([^']+)' column", text)
    return m.group(1) if m else None


def _strip_known_missing_columns(payload: dict) -> dict:
    if not _LEADS_MISSING_COLUMNS:
        return payload
    return {k: v for k, v in payload.items() if k not in _LEADS_MISSING_COLUMNS}


def _execute_lead_write(query_factory, payload: dict, *, max_retries: int = 3) -> Optional[dict]:
    """Run a leads INSERT/UPDATE, learning and retrying around missing columns.

    Postgres + PostgREST will reject the whole write when any payload key is
    not in the schema cache (e.g. canal column missing). Instead of swallowing
    silently, we strip the offending column and retry, so the row still lands.
    """
    cleaned = _strip_known_missing_columns(payload)
    last_exc: Exception | None = None
    for _ in range(max_retries):
        try:
            result = _execute_with_retry(query_factory(cleaned))
            return result
        except Exception as exc:
            missing = _missing_column_from_error(exc)
            if not missing or missing not in cleaned:
                last_exc = exc
                break
            _LEADS_MISSING_COLUMNS.add(missing)
            cleaned = {k: v for k, v in cleaned.items() if k != missing}
            last_exc = exc
    if last_exc:
        try:
            from services import sre_logger
            sre_logger.error("supabase_client", f"lead write failed: {last_exc}", last_exc)
        except Exception:
            pass
    return None


def ensure_lead_for_persona(
    *,
    lead_id: str,
    persona_slug_or_id: Optional[str],
    lead_ref: Optional[int] = None,
    nome: Optional[str] = None,
    stage: Optional[str] = None,
    canal: Optional[str] = None,
    mensagem: Optional[str] = None,
    interesse_produto: Optional[str] = None,
    cidade: Optional[str] = None,
    cep: Optional[str] = None,
    whatsapp_phone_number_id: Optional[str] = None,
) -> Optional[dict]:
    """Ensure an inbound lead is tied to the intended persona branch.

    If an existing lead has no persona, assign it. If it already belongs to a
    different persona and no explicit lead_ref was provided, keep it unchanged
    to avoid moving a real lead between clients by phone/name collision.
    """
    if not lead_id and lead_ref is None:
        return None
    from datetime import datetime, timezone

    client = get_client()
    persona_id = _resolve_persona_id(persona_slug_or_id)
    if not whatsapp_phone_number_id and persona_id:
        whatsapp_phone_number_id = get_default_whatsapp_phone_number_id(persona_id)
    if lead_ref is not None:
        existing = get_lead_by_ref(lead_ref)
    elif persona_id:
        existing = _one(
            client.table("leads").select("*")
            .eq("persona_id", persona_id).eq("lead_id", lead_id).maybe_single()
        )
    else:
        existing = get_lead(lead_id)
    now_iso = datetime.now(timezone.utc).isoformat()

    update: dict = {
        "last_update": now_iso,
        "updated_at": now_iso,
    }
    if nome:
        update["nome"] = nome
    if stage:
        update["stage"] = stage
    if canal:
        update["canal"] = canal
        # Mirror canal into origem so dashboards filtering by source still
        # work even when the canal column is absent in the schema cache.
        update.setdefault("origem", canal)
    if mensagem:
        update["ultima_mensagem"] = mensagem
    if interesse_produto:
        update["interesse_produto"] = interesse_produto
    if cidade:
        update["cidade"] = cidade
    if cep:
        update["cep"] = cep
    if whatsapp_phone_number_id:
        update["whatsapp_phone_number_id"] = whatsapp_phone_number_id
    if lead_id:
        update["lead_id"] = lead_id
        digits = "".join(ch for ch in lead_id if ch.isdigit())
        if digits:
            update["telefone"] = digits

    if existing:
        current_persona = existing.get("persona_id")
        if persona_id and (not current_persona or lead_ref is not None or current_persona == persona_id):
            update["persona_id"] = persona_id
        elif current_persona and persona_id and current_persona != persona_id:
            update = {k: v for k, v in update.items() if k in {"last_update", "updated_at", "ultima_mensagem"}}
        result = _execute_lead_write(
            lambda payload: client.table("leads").update(payload).eq("id", existing["id"]),
            update,
        )
        if result and getattr(result, "data", None):
            return (result.data or [{**existing, **update}])[0]
        return {**existing, **_strip_known_missing_columns(update)}

    payload = {
        **update,
        "lead_id": lead_id,
        "nome": nome,
        "stage": stage or "novo",
        "canal": canal or "whatsapp",
        "persona_id": persona_id,
        "ai_enabled": True,
        "created_at": now_iso,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    payload.setdefault("origem", payload.get("canal"))
    result = _execute_lead_write(
        lambda body: client.table("leads").insert(body),
        payload,
    )
    if result and getattr(result, "data", None):
        return result.data[0]
    return None


def get_leads(
    persona_slug: Optional[str] = None, limit: int = 100, offset: int = 0,
    since_hours: int | None = None,
) -> list:
    try:
        q = get_client().table("leads").select("*")
        if persona_slug:
            persona_id = _resolve_persona_id(persona_slug) or persona_slug
            q = q.eq("persona_id", persona_id)
        if since_hours is not None:
            q = q.gte(
                "created_at",
                (datetime.now(timezone.utc) - timedelta(hours=max(1, int(since_hours)))).isoformat(),
            )
        return _q(q.order("updated_at", desc=True).range(offset, offset + limit - 1))
    except Exception as exc:
        try:
            from services import sre_logger
            sre_logger.error("supabase_client", f"get_leads failed: {exc}", exc)
        except Exception:
            pass
        return []


def get_leads_for_persona_ids(
    persona_ids: list[str], limit: int = 100, offset: int = 0,
    since_hours: int | None = None,
) -> list:
    ids = [pid for pid in persona_ids if pid]
    if not ids:
        return []
    try:
        q = (
            get_client()
            .table("leads")
            .select("*")
            .in_("persona_id", ids)
        )
        if since_hours is not None:
            q = q.gte(
                "created_at",
                (datetime.now(timezone.utc) - timedelta(hours=max(1, int(since_hours)))).isoformat(),
            )
        return _q(q.order("updated_at", desc=True).range(offset, offset + limit - 1))
    except Exception as exc:
        try:
            from services import sre_logger
            sre_logger.error("supabase_client", f"get_leads_for_persona_ids failed: {exc}", exc)
        except Exception:
            pass
        return []


def update_lead(lead_ref: int, data: dict) -> None:
    _execute_with_retry(get_client().table("leads").update(data).eq("id", lead_ref))


def merge_commercial_note(metadata: dict, commercial_note: dict[str, str]) -> dict:
    """Apply a manual commercial-note edit into a lead's metadata.

    Shared by the admin (`routes.leads`) and client-portal (`routes.portal`)
    lead-update endpoints so an edit made from either surface lands the
    same way: the display-only `commercial_note` mirror AND
    `conversation_state.appointment_request` (the AI's actual working
    memory) both get the new values, and any now-answered field is
    dropped from `missing_fields` so the next reply doesn't ask again.
    """
    clean_note = {
        str(k).strip(): str(v).strip()
        for k, v in commercial_note.items()
        if str(k).strip() and str(v).strip()
    }
    metadata = dict(metadata or {})
    metadata["commercial_note"] = {
        **clean_note,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "manual",
    }
    conversation_state = dict(metadata.get("conversation_state") or {})
    appointment_request = dict(conversation_state.get("appointment_request") or {})
    appointment_request.update(clean_note)
    conversation_state["appointment_request"] = appointment_request
    missing_fields = list(conversation_state.get("missing_fields") or [])
    conversation_state["missing_fields"] = [
        field for field in missing_fields if field not in clean_note
    ]
    metadata["conversation_state"] = conversation_state
    return metadata


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


def get_audiences(persona_id: Optional[str] = None) -> list[dict]:
    q = get_client().table("audiences").select("*").order("is_system").order("name")
    if persona_id:
        q = q.eq("persona_id", persona_id)
    return _q(q)


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


def create_audience(data: dict) -> dict:
    metadata = dict(data.get("metadata") or {})
    metadata.setdefault(
        "kind",
        "legacy_import_bucket" if (data.get("source_type") == "import" or data.get("slug") == "import") else "semantic_group",
    )
    payload = {
        "persona_id": data.get("persona_id"),
        "slug": _slugify(data.get("slug") or data.get("name") or "audience"),
        "name": data.get("name") or "Audience",
        "description": data.get("description"),
        "source_type": data.get("source_type") or "manual",
        "is_system": bool(data.get("is_system", False)),
        "created_by_user_id": data.get("created_by_user_id"),
        "metadata": metadata,
    }
    return _insert_one(get_client().table("audiences").insert(payload))


def update_audience(audience_id: str, data: dict) -> Optional[dict]:
    payload = {
        "name": data.get("name"),
        "description": data.get("description"),
        "updated_at": data.get("updated_at"),
        "metadata": data.get("metadata"),
    }
    if data.get("slug"):
        payload["slug"] = _slugify(data["slug"])
    payload = {k: v for k, v in payload.items() if v is not None}
    if not payload:
        return get_audience(audience_id)
    result = _execute_with_retry(get_client().table("audiences").update(payload).eq("id", audience_id))
    return (result.data or [None])[0] if result else None


def ensure_system_audience(
    persona_id: str,
    *,
    slug: str,
    name: str,
    description: Optional[str] = None,
    source_type: str = "manual",
    created_by_user_id: Optional[str] = None,
) -> Optional[dict]:
    existing = get_audience_by_slug(persona_id, slug)
    payload = {
        "persona_id": persona_id,
        "slug": _slugify(slug),
        "name": name,
        "description": description,
        "source_type": source_type,
        "is_system": True,
        "created_by_user_id": created_by_user_id,
    }
    if existing:
        return update_audience(existing["id"], payload) or {**existing, **payload}
    return create_audience(payload)


def _is_import_audience(audience: Optional[dict]) -> bool:
    """The `import` system bucket is an operational leads grouping (source_type
    'import'), not a semantic audience of the persona tree. It must never appear
    as a graph audience node nor as a Leads filter pill."""
    if not audience:
        return False
    slug = str(audience.get("slug") or "").strip().lower()
    source_type = str(audience.get("source_type") or "").strip().lower()
    return slug == "import" or source_type == "import"


def ensure_import_audience(persona_id: str, created_by_user_id: Optional[str] = None) -> Optional[dict]:
    return ensure_system_audience(
        persona_id,
        slug="import",
        name="Import",
        description="Audience padrao para todos os imports CSV/Bulk da persona.",
        source_type="import",
        created_by_user_id=created_by_user_id,
    )


def ensure_system_audiences_for_persona(
    persona_id: Optional[str],
    *,
    created_by_user_id: Optional[str] = None,
) -> dict:
    """Garante que a persona tenha as audiences system padrao.

    Idempotente: pode ser chamado em qualquer entrypoint (listagem, move,
    share, login) sem efeito colateral alem de criar a `import` audience caso
    nao exista. Devolve {'import': <audience_dict_or_None>}.

    Falhas sao silenciadas para que um problema na criacao da audience nao
    derrube o endpoint chamador. Endpoints continuam funcionando com lista
    vazia de audiences caso a criacao falhe.
    """
    if not persona_id:
        return {"import": None}
    try:
        imp = ensure_import_audience(persona_id, created_by_user_id=created_by_user_id)
    except Exception as exc:
        try:
            from services import sre_logger
            sre_logger.warn("supabase_client", f"ensure_system_audiences_for_persona failed: {exc}", exc)
        except Exception:
            pass
        imp = None
    return {"import": imp}


def materialize_graph_audiences_for_persona(persona_id: Optional[str]) -> list[dict]:
    """Reconcile graph audience nodes into the operational `audiences` table.

    Audiences created via Sofia or the Graph tab live as `knowledge_nodes`
    (node_type='audience'). For them to be usable as Leads filters (and as
    move/share targets) they must also exist as `audiences` rows. This bridge is
    idempotent by (persona_id, slug): it only creates rows that do not yet exist
    and never touches the `import` bucket or archived nodes. Rows it creates are
    tagged `source_type='graph'` so `sync_audience_node` skips them and we do not
    spawn a duplicate node back into the tree.
    """
    if not persona_id:
        return []
    try:
        nodes = list_knowledge_nodes_by_type(["audience"], persona_id=persona_id, limit=500)
    except Exception:
        return []
    created: list[dict] = []
    for node in nodes or []:
        slug = _slugify(node.get("slug") or node.get("title") or "")
        if not slug or slug == "import":
            continue
        meta = node.get("metadata") or {}
        if str(meta.get("source_type") or "").strip().lower() == "import":
            continue
        if str(node.get("status") or "").strip().lower() == "archived":
            continue
        if get_audience_by_slug(persona_id, slug):
            continue
        try:
            row = create_audience({
                "persona_id": persona_id,
                "slug": slug,
                "name": node.get("title") or slug,
                "description": node.get("summary"),
                "source_type": "graph",
            })
        except Exception:
            row = None
        if row:
            created.append(row)
    return created


def list_persona_audiences(persona_id: Optional[str]) -> list[dict]:
    """Operational audiences for the Leads tab filters.

    Returns the persona's real audiences (reconciled with graph-created ones)
    minus the internal `import` bucket. This is the single source the Leads UI
    should consume so that any audience created in the Graph/Sofia automatically
    becomes a Leads filter, while `import` never shows as a semantic audience.
    """
    if not persona_id:
        return []
    try:
        materialize_graph_audiences_for_persona(persona_id)
    except Exception:
        pass
    rows = [row for row in (get_audiences(persona_id=persona_id) or []) if not _is_import_audience(row)]
    by_slug = {str(row.get("slug") or "").strip().lower(): row for row in rows}
    # Union with graph audience nodes (read-only) so anything highlighted as an
    # audience in the Graph becomes a Leads filter even if row materialization
    # did not persist. Persona-scoped, never the `import` bucket, never archived.
    try:
        nodes = list_knowledge_nodes_by_type(["audience"], persona_id=persona_id, limit=500)
    except Exception:
        nodes = []
    for node in nodes or []:
        slug = _slugify(node.get("slug") or node.get("title") or "")
        if not slug or slug == "import" or slug in by_slug:
            continue
        meta = node.get("metadata") or {}
        if str(meta.get("source_type") or "").strip().lower() == "import":
            continue
        if str(node.get("status") or "").strip().lower() == "archived":
            continue
        synthesized = {
            "id": node.get("id"),
            "persona_id": persona_id,
            "slug": slug,
            "name": node.get("title") or slug,
            "description": node.get("summary"),
            "source_type": "graph",
            "is_system": False,
            "from_graph_node": True,
        }
        by_slug[slug] = synthesized
        rows.append(synthesized)
    return rows


def get_lead_memberships(lead_id: int) -> list[dict]:
    rows = _q(
        get_client()
        .table("lead_audience_memberships")
        .select("id,lead_id,audience_id,membership_type,created_by_user_id,created_at")
        .eq("lead_id", lead_id)
        .order("created_at")
    )
    if not rows:
        return []
    audience_ids = [row.get("audience_id") for row in rows if row.get("audience_id")]
    audiences_by_id = (
        {
            row.get("id"): row
            for row in _q(get_client().table("audiences").select("*").in_("id", audience_ids))
        }
        if audience_ids
        else {}
    )
    return [{**row, "audience": audiences_by_id.get(row.get("audience_id"))} for row in rows]


def ensure_lead_membership(
    lead_id: int,
    audience_id: str,
    *,
    membership_type: str = "primary",
    created_by_user_id: Optional[str] = None,
) -> Optional[dict]:
    if not lead_id or not audience_id:
        return None
    payload = {
        "lead_id": lead_id,
        "audience_id": audience_id,
        "membership_type": membership_type,
        "created_by_user_id": created_by_user_id,
    }
    result = _execute_with_retry(
        get_client().table("lead_audience_memberships").upsert(payload, on_conflict="lead_id,audience_id")
    )
    return (result.data or [payload])[0] if result else payload


def delete_lead_membership(lead_id: int, audience_id: str) -> None:
    _execute_with_retry(
        get_client().table("lead_audience_memberships").delete().eq("lead_id", lead_id).eq("audience_id", audience_id)
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


def get_leads_for_audience_scope(
    *,
    persona_id: str,
    audience_id: Optional[str] = None,
    audience_slug: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    since_hours: int | None = None,
) -> list[dict]:
    lead_refs = get_lead_refs_for_audience_scope(persona_id=persona_id, audience_id=audience_id, audience_slug=audience_slug)
    if not lead_refs:
        return []
    query = (
        get_client().table("leads").select("*")
        .in_("id", lead_refs).order("updated_at", desc=True)
    )
    if since_hours is not None:
        query = query.gte(
            "created_at",
            (datetime.now(timezone.utc) - timedelta(hours=max(1, int(since_hours)))).isoformat(),
        )
    query = query.range(offset, offset + limit - 1)
    rows = _q(query)
    page_refs = [int(row["id"]) for row in rows if row.get("id") is not None]
    memberships_map = {lead_id: get_lead_memberships(lead_id) for lead_id in page_refs}
    return [{**row, "memberships": memberships_map.get(row.get("id"), [])} for row in rows]


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


def backdate_lead_messages(lead_ref: int, hours: float) -> int:
    """Test-only: shift a lead's messages.created_at backward by N hours.

    Used by the WA Validator to simulate a genuine time gap between a
    scripted phase and a later one (see migration 104).
    """
    result = get_client().rpc(
        "backdate_lead_messages", {"p_lead_ref": lead_ref, "p_hours": hours}
    ).execute()
    return int(getattr(result, "data", 0) or 0)


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


def _kg_unavailable(exc: Exception) -> bool:
    """Detect 'table not found' from PostgREST/Supabase, regardless of message wording."""
    text = str(exc)
    return (
        "knowledge_nodes" in text or "knowledge_edges" in text
    ) and ("PGRST205" in text or "schema cache" in text or "Could not find the table" in text)


def _unique_violation(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "duplicate key value violates unique constraint" in text
        or "unique constraint" in text
        or "23505" in text
    )


def upsert_knowledge_node(data: dict) -> Optional[dict]:
    """Idempotent upsert of a knowledge node, keyed by (persona_id, node_type, slug).

    `data` should at minimum contain node_type, slug, title.
    Returns the inserted/updated row, or None if the table is missing.
    """
    global _KG_TABLES_MISSING
    if _KG_TABLES_MISSING:
        return None
    from datetime import datetime, timezone

    required = {"node_type", "slug", "title"}
    if not required.issubset(data.keys()):
        raise ValueError(f"upsert_knowledge_node missing keys: {required - set(data.keys())}")

    client = get_client()
    persona_id = data.get("persona_id")
    try:
        q = (
            client.table("knowledge_nodes")
            .select("id,metadata,tags,summary,title,status")
            .eq("node_type", data["node_type"])
            .eq("slug", data["slug"])
        )
        q = q.eq("persona_id", persona_id) if persona_id else q.is_("persona_id", "null")
        existing = (q.limit(1).execute().data or [None])[0]
    except Exception as exc:
        if _kg_unavailable(exc):
            _KG_TABLES_MISSING = True
            return None
        if _unique_violation(exc):
            # Parallel approvals can race between the select and insert.
            # Treat the duplicate as a successful idempotent upsert.
            q = (
                client.table("knowledge_nodes")
                .select("*")
                .eq("node_type", data["node_type"])
                .eq("slug", data["slug"])
            )
            q = q.eq("persona_id", persona_id) if persona_id else q.is_("persona_id", "null")
            existing = (q.limit(1).execute().data or [None])[0]
            if existing:
                return existing
        raise

    now_iso = datetime.now(timezone.utc).isoformat()
    if existing:
        # Merge tags & metadata to keep prior context.
        merged_tags = sorted(set((existing.get("tags") or []) + (data.get("tags") or [])))
        merged_meta = {**(existing.get("metadata") or {}), **(data.get("metadata") or {})}
        update = {
            "title":    data.get("title") or existing.get("title"),
            "summary":  data.get("summary") or existing.get("summary"),
            "tags":     merged_tags,
            "metadata": merged_meta,
            "status":   data.get("status") or existing.get("status") or "active",
            "updated_at": now_iso,
        }
        if data.get("source_table"):
            update["source_table"] = data["source_table"]
        if data.get("source_id"):
            update["source_id"] = data["source_id"]
        for field in ("level", "importance", "confidence"):
            if data.get(field) is not None:
                update[field] = data[field]
        try:
            r = client.table("knowledge_nodes").update(update).eq("id", existing["id"]).execute()
            return (r.data or [{**existing, **update}])[0]
        except Exception as exc:
            if _kg_unavailable(exc):
                _KG_TABLES_MISSING = True
                return None
            raise

    payload = dict(data)
    payload.setdefault("tags", [])
    payload.setdefault("metadata", {})
    payload.setdefault("status", "active")
    payload["created_at"] = now_iso
    payload["updated_at"] = now_iso
    try:
        r = client.table("knowledge_nodes").insert(payload).execute()
        return (r.data or [{}])[0]
    except Exception as exc:
        if _kg_unavailable(exc):
            _KG_TABLES_MISSING = True
            return None
        if _unique_violation(exc):
            q = (
                client.table("knowledge_nodes")
                .select("*")
                .eq("node_type", data["node_type"])
                .eq("slug", data["slug"])
            )
            q = q.eq("persona_id", persona_id) if persona_id else q.is_("persona_id", "null")
            existing = (q.limit(1).execute().data or [None])[0]
            if existing:
                return existing
        raise


def get_knowledge_node(node_id: str) -> Optional[dict]:
    """Fetch a single knowledge node by UUID."""
    global _KG_TABLES_MISSING
    if _KG_TABLES_MISSING or not node_id:
        return None
    try:
        return _one(get_client().table("knowledge_nodes").select("*").eq("id", node_id).maybe_single())
    except Exception as exc:
        if _kg_unavailable(exc):
            _KG_TABLES_MISSING = True
            return None
        raise


def update_knowledge_node(node_id: str, data: dict, *, mark_related_faqs: bool = True) -> Optional[dict]:
    global _KG_TABLES_MISSING
    if _KG_TABLES_MISSING or not node_id or not data:
        return None
    try:
        client = get_client()
        payload = dict(data)
        payload.setdefault("updated_at", __import__("datetime").datetime.utcnow().isoformat())
        result = client.table("knowledge_nodes").update(payload).eq("id", node_id).execute()
        updated = (result.data or [payload])[0]
        node_type = updated.get("node_type")
        if not node_type:
            row = (client.table("knowledge_nodes").select("id,node_type,persona_id,source_id").eq("id", node_id).limit(1).execute().data or [None])[0]
        else:
            row = updated
        if mark_related_faqs and row and row.get("node_type") in {"brand", "briefing", "audience", "product", "offer", "copy", "rule"}:
            _mark_persona_faqs_pending_regeneration(
                row.get("persona_id"),
                changed_source_id=row.get("source_id") or node_id,
                now_iso=payload["updated_at"],
            )
        return updated
    except Exception as exc:
        if _kg_unavailable(exc):
            _KG_TABLES_MISSING = True
            return None
        raise


def list_graph_json_projection_nodes(persona_id: str, limit: int = 10000) -> list[dict]:
    """List only nodes owned by the regenerable Graph JSON projection."""
    if _KG_TABLES_MISSING or not persona_id:
        return []
    try:
        rows = _q(
            get_client()
            .table("knowledge_nodes")
            .select("id,node_type,source_id,status,metadata")
            .eq("persona_id", persona_id)
            .limit(limit)
        )
        return [
            row
            for row in rows
            if (row.get("metadata") or {}).get("graph_json_import") is True
        ]
    except Exception as exc:
        if _kg_unavailable(exc):
            return []
        raise


def list_graph_json_projection_edges(persona_id: str, limit: int = 10000) -> list[dict]:
    """List only edges owned by the regenerable Graph JSON projection."""
    if _KG_TABLES_MISSING or not persona_id:
        return []
    try:
        rows = _q(
            get_client()
            .table("knowledge_edges")
            .select("id,source_node_id,target_node_id,relation_type,metadata")
            .eq("persona_id", persona_id)
            .limit(limit)
        )
        return [
            row
            for row in rows
            if (row.get("metadata") or {}).get("graph_json_edge_id")
        ]
    except Exception as exc:
        if _kg_unavailable(exc):
            return []
        raise


def ensure_persona_knowledge_node(persona_id: str) -> Optional[dict]:
    """Ensure the graph has a protected semantic root for a persona."""
    if not persona_id:
        return None
    persona = None
    try:
        persona = _one(get_client().table("personas").select("id,slug,name").eq("id", persona_id).maybe_single())
    except Exception:
        persona = None
    return upsert_knowledge_node({
        "persona_id": persona_id,
        "node_type": "persona",
        "slug": "self",
        "title": (persona or {}).get("name") or "Persona",
        "summary": "Raiz protegida da persona no grafo.",
        "tags": ["persona"],
        "metadata": {"role": "root", "protected": True},
        "status": "active",
        "level": 0,
        "importance": 1.0,
        "confidence": 1.0,
    })


def ensure_gallery_node(persona_id: str) -> Optional[dict]:
    """Ensure a protected Gallery node exists for a persona."""
    if not persona_id:
        return None
    return upsert_knowledge_node({
        "persona_id": persona_id,
        "node_type": "gallery",
        "slug": "gallery-default",
        "title": "Gallery",
        "summary": "Bloco protegido para materiais visuais. Nodes ligados aqui aparecem em Assets.",
        "tags": ["gallery", "assets", "visual"],
        "metadata": {
            "protected": True,
            "system_node": True,
            "asset_scope": "visual_media",
            "open_url": "/marketing/assets",
        },
        "status": "active",
        "level": 112,
        "importance": 0.82,
        "confidence": 1.0,
    })


def sync_audience_node(audience: dict) -> Optional[dict]:
    if not audience or not audience.get("persona_id") or not audience.get("id"):
        return None
    # Never mirror the `import` bucket (operational, not semantic) nor a
    # graph-sourced audience (it already exists as a node) into the tree.
    if _is_import_audience(audience) or str(audience.get("source_type") or "").strip().lower() == "graph":
        return None
    persona = get_persona_by_id(audience["persona_id"]) or {}
    node = upsert_knowledge_node({
        "persona_id": audience["persona_id"],
        "source_table": "audiences",
        "source_id": audience["id"],
        "node_type": "audience",
        "slug": audience.get("slug") or _slugify(audience.get("name") or "audience"),
        "title": audience.get("name") or "Audience",
        "summary": audience.get("description") or "Publico ou grupo operacional de leads.",
        "tags": ["audience", audience.get("source_type") or "manual"],
        "metadata": {
            **(audience.get("metadata") or {}),
            "audience_id": audience.get("id"),
            "audience_slug": audience.get("slug"),
            "source_type": audience.get("source_type"),
            "is_system": audience.get("is_system"),
            "open_url": f"/leads?audience={audience.get('slug', '')}",
            "persona_slug": persona.get("slug"),
        },
        "status": "active",
        "level": 55,
        "importance": 0.72,
        "confidence": 1.0,
    })
    # Lazy import to avoid circular dependency between supabase_client and knowledge_graph
    from services import knowledge_graph as _kg
    persona_root = _kg._ensure_persona_root(audience["persona_id"])
    if node and persona_root:
        upsert_knowledge_edge(
            source_node_id=persona_root["id"],
            target_node_id=node["id"],
            relation_type="contains",
            persona_id=audience["persona_id"],
            weight=1,
            metadata={"primary_tree": True, "created_from": "audiences"},
        )
    return node


def ensure_embedded_node(persona_id: str) -> Optional[dict]:
    """Ensure a protected Embedded/Golden Dataset destination node exists for a persona."""
    if not persona_id:
        return None
    try:
        existing = (
            get_client()
            .table("knowledge_nodes")
            .select("*")
            .eq("persona_id", persona_id)
            .in_("node_type", ["embed", "embedded"])
            .limit(100)
            .execute()
            .data
            or []
        )
        active = [
            row
            for row in existing
            if (row.get("metadata") or {}).get("active", True) is not False
        ]
        if active:
            active.sort(
                key=lambda row: (
                    (row.get("metadata") or {}).get("graph_json_import") is True,
                    row.get("slug") in {"embedded", "embedded-default"},
                ),
                reverse=True,
            )
            return active[0]
    except Exception as exc:
        if not _kg_unavailable(exc):
            raise
    return upsert_knowledge_node({
        "persona_id": persona_id,
        "node_type": "embed",
        "slug": "embedded-default",
        "title": "Embedded",
        "summary": "Destino protegido para FAQs publicados no Golden Dataset e enviados ao RAG.",
        "tags": ["rag", "embedded", "golden-dataset", "default"],
        "metadata": {
            "protected": True,
            "system_node": True,
            # The protected output sink has no primary Persona edge. This also
            # keeps the legacy primary-edge trigger from manufacturing an
            # invalid Persona -> Embed connection on older installations.
            "graph_json_id": "protected-output-sink",
            "rag_index": "default",
            "open_url": "/kb",
        },
        "status": "active",
        "level": 120,
        "importance": 0.78,
        "confidence": 1.0,
    })


def _edge_is_inactive(edge: dict | None) -> bool:
    metadata = (edge or {}).get("metadata") or {}
    return metadata.get("active") is False


def _primary_tree_metadata(metadata: Optional[dict]) -> dict:
    merged = dict(metadata or {})
    merged["primary_tree"] = True
    merged["active"] = True
    merged.pop("deleted_at", None)
    merged.pop("deleted_from", None)
    return merged


def demote_duplicate_primary_edges_for_pair(source_node_id: str, target_node_id: str, except_relation_type: Optional[str] = None) -> int:
    """Keep one visible primary-tree edge per source -> target pair.

    Alternate relation labels can remain as lineage/semantic edges, but they
    must not render as separate primary-tree branches.
    """
    if _KG_TABLES_MISSING or not source_node_id or not target_node_id:
        return 0
    client = get_client()
    rows = _q(
        client.table("knowledge_edges")
        .select("id,relation_type,metadata")
        .eq("source_node_id", source_node_id)
        .eq("target_node_id", target_node_id)
        .limit(500)
    )
    changed = 0
    for row in rows:
        if except_relation_type and (row.get("relation_type") or "").lower() == except_relation_type.lower():
            continue
        metadata = row.get("metadata") or {}
        if metadata.get("primary_tree") is not True or metadata.get("active") is False:
            continue
        next_metadata = {
            **metadata,
            "primary_tree": False,
            "visual_hidden": True,
            "active": True,
            "demoted_from_primary_tree": "duplicate_source_target",
        }
        _execute_with_retry(
            client.table("knowledge_edges").update({"metadata": next_metadata}).eq("id", row["id"])
        )
        changed += 1
    return changed


def upsert_knowledge_edge(
    source_node_id: str,
    target_node_id: str,
    relation_type: str,
    persona_id: Optional[str] = None,
    weight: float = 1.0,
    metadata: Optional[dict] = None,
) -> Optional[dict]:
    """Idempotent upsert keyed by (source_node_id, target_node_id, relation_type).

    Existing soft-deleted edges are reactivated. For primary tree paths, this
    also deactivates any previous active primary path pointing at the target.
    """
    global _KG_TABLES_MISSING
    if _KG_TABLES_MISSING:
        return None
    if not source_node_id or not target_node_id or not relation_type:
        return None
    if source_node_id == target_node_id:
        return None  # don't allow self-loops

    client = get_client()
    original_relation_type = relation_type
    try:
        existing_q = (
            client.table("knowledge_edges")
            .select("*")
            .eq("source_node_id", source_node_id)
            .eq("target_node_id", target_node_id)
            .eq("relation_type", relation_type)
            .limit(1)
            .execute()
        )
        existing = (existing_q.data or [None])[0]
    except Exception as exc:
        if _kg_unavailable(exc):
            _KG_TABLES_MISSING = True
            return None
        if _unique_violation(exc):
            existing_q = (
                client.table("knowledge_edges")
                .select("*")
                .eq("source_node_id", source_node_id)
                .eq("target_node_id", target_node_id)
                .eq("relation_type", relation_type)
                .limit(1)
                .execute()
            )
            existing = (existing_q.data or [None])[0]
            if existing:
                return existing
        raise
    requested_metadata = dict(metadata or {})
    is_primary_path = requested_metadata.get("primary_tree") is True
    if is_primary_path and (relation_type or "").lower() == "about_product":
        try:
            type_rows = (
                client.table("knowledge_nodes")
                .select("id,node_type")
                .in_("id", [source_node_id, target_node_id])
                .limit(2)
                .execute()
                .data
                or []
            )
            types = {row.get("id"): (row.get("node_type") or "").lower() for row in type_rows}
            if types.get(source_node_id) == "audience" and types.get(target_node_id) == "product":
                requested_metadata.setdefault("canonicalized_relation_from", relation_type)
                relation_type = "offers_product"
                if relation_type != original_relation_type:
                    existing_q = (
                        client.table("knowledge_edges")
                        .select("*")
                        .eq("source_node_id", source_node_id)
                        .eq("target_node_id", target_node_id)
                        .eq("relation_type", relation_type)
                        .limit(1)
                        .execute()
                    )
                    existing = (existing_q.data or [None])[0]
        except Exception:
            pass
    if is_primary_path and (relation_type or "").lower() == "belongs_to_persona":
        try:
            source_rows = (
                client.table("knowledge_nodes")
                .select("id,node_type")
                .eq("id", source_node_id)
                .limit(1)
                .execute()
                .data
                or []
            )
            source_type = (source_rows[0].get("node_type") if source_rows else "").lower()
            if source_type == "persona":
                active_parents = (
                    client.table("knowledge_edges")
                    .select("source_node_id,relation_type,metadata")
                    .eq("target_node_id", target_node_id)
                    .limit(500)
                    .execute()
                    .data
                    or []
                )
                non_persona_primary = [
                    edge for edge in active_parents
                    if edge.get("source_node_id") != source_node_id
                    and not _edge_is_inactive(edge)
                    and (edge.get("metadata") or {}).get("primary_tree") is True
                ]
                if non_persona_primary:
                    requested_metadata["primary_tree"] = False
                    requested_metadata["visual_hidden"] = True
                    is_primary_path = False
        except Exception:
            pass

    if is_primary_path:
        deactivate_primary_paths_for_target(target_node_id, except_source_node_id=source_node_id)
        demote_duplicate_primary_edges_for_pair(source_node_id, target_node_id, except_relation_type=relation_type)

    if existing:
        update_data = {
            "persona_id": persona_id,
            "weight": weight,
            "metadata": _primary_tree_metadata(requested_metadata) if is_primary_path else {**(existing.get("metadata") or {}), **requested_metadata, "active": True},
        }
        r = client.table("knowledge_edges").update(update_data).eq("id", existing["id"]).execute()
        return (r.data or [{**existing, **update_data}])[0]
    try:
        insert_metadata = _primary_tree_metadata(requested_metadata) if is_primary_path else requested_metadata
        r = client.table("knowledge_edges").insert({
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
            "relation_type": relation_type,
            "persona_id": persona_id,
            "weight": weight,
            "metadata": insert_metadata,
        }).execute()
        return (r.data or [{}])[0]
    except Exception as exc:
        if _kg_unavailable(exc):
            _KG_TABLES_MISSING = True
            return None
        raise


def update_knowledge_edge(edge_id: str, data: dict) -> Optional[dict]:
    """Replace selected fields on one edge by UUID without metadata merging."""
    global _KG_TABLES_MISSING
    if _KG_TABLES_MISSING or not edge_id or not data:
        return None
    try:
        payload = dict(data)
        result = (
            get_client()
            .table("knowledge_edges")
            .update(payload)
            .eq("id", edge_id)
            .execute()
        )
        return (result.data or [{"id": edge_id, **payload}])[0]
    except Exception as exc:
        if _kg_unavailable(exc):
            _KG_TABLES_MISSING = True
            return None
        raise


def deactivate_primary_paths_for_target(target_node_id: str, except_source_node_id: Optional[str] = None) -> int:
    """Soft-disable active primary-tree paths to a target node."""
    if _KG_TABLES_MISSING or not target_node_id:
        return 0
    from datetime import datetime, timezone

    client = get_client()
    rows = _q(
        client.table("knowledge_edges")
        .select("id,source_node_id,metadata")
        .eq("target_node_id", target_node_id)
        .limit(500)
    )
    changed = 0
    now_iso = datetime.now(timezone.utc).isoformat()
    for row in rows:
        metadata = row.get("metadata") or {}
        if row.get("source_node_id") == except_source_node_id:
            continue
        if metadata.get("primary_tree") is not True or metadata.get("active") is False:
            continue
        next_metadata = {
            **metadata,
            "active": False,
            "primary_tree": False,
            "visual_hidden": True,
            "deleted_at": now_iso,
            "deleted_from": "graph_ui_reparent",
        }
        _execute_with_retry(
            client.table("knowledge_edges").update({"metadata": next_metadata}).eq("id", row["id"])
        )
        changed += 1
    return changed


def delete_knowledge_edge(edge_id: str) -> bool:
    """Soft-delete a knowledge edge by id. Returns True when the request succeeds."""
    global _KG_TABLES_MISSING
    if _KG_TABLES_MISSING or not edge_id:
        return False
    from datetime import datetime, timezone

    try:
        client = get_client()
        row = _one(client.table("knowledge_edges").select("*").eq("id", edge_id).maybe_single())
        if not row:
            return False
        metadata = row.get("metadata") or {}
        metadata = {
            **metadata,
            "active": False,
            "deleted_at": datetime.now(timezone.utc).isoformat(),
            "deleted_from": "graph_ui",
        }
        _execute_with_retry(client.table("knowledge_edges").update({"metadata": metadata}).eq("id", edge_id))
        if row.get("relation_type") == "gallery_asset":
            mark_gallery_asset_inactive_by_edge(edge_id)
        return True
    except Exception as exc:
        if _kg_unavailable(exc):
            _KG_TABLES_MISSING = True
            return False
        raise


def hard_delete_knowledge_edge(edge_id: str) -> bool:
    """Hard-delete a knowledge edge by id. Scoped callers must validate ownership first."""
    global _KG_TABLES_MISSING
    if _KG_TABLES_MISSING or not edge_id:
        return False
    try:
        client = get_client()
        row = _one(client.table("knowledge_edges").select("id").eq("id", edge_id).maybe_single())
        if not row:
            return False
        _execute_with_retry(client.table("knowledge_edges").delete().eq("id", edge_id))
        return True
    except Exception as exc:
        if _kg_unavailable(exc):
            _KG_TABLES_MISSING = True
            return False
        raise


def get_knowledge_edge(edge_id: str) -> Optional[dict]:
    if not edge_id:
        return None
    return _one(get_client().table("knowledge_edges").select("*").eq("id", edge_id).maybe_single())


def get_knowledge_node_for_source(
    source_table: str,
    source_id: str,
    *,
    persona_id: Optional[str] = None,
) -> Optional[dict]:
    global _KG_TABLES_MISSING
    if _KG_TABLES_MISSING or not source_table or not source_id:
        return None
    try:
        q = (
            get_client().table("knowledge_nodes")
            .select("*")
            .eq("source_table", source_table)
            .eq("source_id", source_id)
        )
        if persona_id:
            q = q.eq("persona_id", persona_id)
        rows = _q(q.order("created_at", desc=True).limit(5))
        for row in rows:
            if row.get("status") != "deleted":
                return row
        return rows[0] if rows else None
    except Exception as exc:
        if _kg_unavailable(exc):
            _KG_TABLES_MISSING = True
            return None
        raise


def get_knowledge_node_by_slug(
    slug: str,
    *,
    persona_id: Optional[str] = None,
    node_type: Optional[str] = None,
) -> Optional[dict]:
    global _KG_TABLES_MISSING
    if _KG_TABLES_MISSING or not slug:
        return None
    try:
        q = get_client().table("knowledge_nodes").select("*").eq("slug", slug)
        if persona_id:
            q = q.eq("persona_id", persona_id)
        if node_type:
            q = q.eq("node_type", node_type)
        rows = _q(q.limit(20))
        for row in rows:
            if row.get("status") != "deleted":
                return row
        return rows[0] if rows else None
    except Exception as exc:
        if _kg_unavailable(exc):
            _KG_TABLES_MISSING = True
            return None
        raise


def get_knowledge_edge_between(
    source_node_id: str,
    target_node_id: str,
    *,
    relation_type: Optional[str] = None,
) -> Optional[dict]:
    global _KG_TABLES_MISSING
    if _KG_TABLES_MISSING or not source_node_id or not target_node_id:
        return None
    try:
        q = (
            get_client().table("knowledge_edges")
            .select("*")
            .eq("source_node_id", source_node_id)
            .eq("target_node_id", target_node_id)
        )
        if relation_type:
            q = q.eq("relation_type", relation_type)
        rows = _q(q.limit(5))
        for row in rows:
            if not _edge_is_inactive(row):
                return row
        return rows[0] if rows else None
    except Exception as exc:
        if _kg_unavailable(exc):
            _KG_TABLES_MISSING = True
            return None
        raise


def delete_knowledge_node(node_id: str) -> bool:
    """Delete a knowledge node and its graph edges by id."""
    global _KG_TABLES_MISSING
    if _KG_TABLES_MISSING or not node_id:
        return False
    client = get_client()
    try:
        node = _one(client.table("knowledge_nodes").select("id,node_type,metadata").eq("id", node_id).maybe_single())
        metadata = (node or {}).get("metadata") or {}
        if (node or {}).get("node_type") in {"persona", "embedded", "gallery"} or metadata.get("protected") is True:
            return False
        _execute_with_retry(client.table("knowledge_edges").delete().eq("source_node_id", node_id))
        _execute_with_retry(client.table("knowledge_edges").delete().eq("target_node_id", node_id))
        _execute_with_retry(client.table("knowledge_nodes").delete().eq("id", node_id))
        return True
    except Exception as exc:
        if _kg_unavailable(exc):
            _KG_TABLES_MISSING = True
            return False
        raise


def find_knowledge_nodes(
    term: str,
    persona_id: Optional[str] = None,
    node_types: Optional[list[str]] = None,
    limit: int = 25,
) -> list[dict]:
    """Find nodes by slug, title (ILIKE), or tags membership.

    Defensive: returns [] when the table is missing or any error occurs.
    """
    global _KG_TABLES_MISSING
    if _KG_TABLES_MISSING:
        return []
    if not term:
        return []
    client = get_client()
    norm = term.strip().lower()
    slug_norm = norm.replace(" ", "-")
    try:
        # Match by exact slug, then loosen by title/tags if needed.
        q = client.table("knowledge_nodes").select("*").limit(limit)
        if persona_id:
            q = q.eq("persona_id", persona_id)
        if node_types:
            q = q.in_("node_type", node_types)
        # PostgREST `or_` filter â€” slug exact match | title ILIKE | tag contains
        or_clause = f"slug.eq.{slug_norm},title.ilike.*{norm}*,tags.cs.{{{norm}}}"
        rows = q.or_(or_clause).execute().data or []
        return rows
    except Exception as exc:
        if _kg_unavailable(exc):
            _KG_TABLES_MISSING = True
            return []
        return []


def get_knowledge_neighbors(
    node_ids: list[str],
    max_edges: int = 200,
) -> tuple[list[dict], list[dict]]:
    """Return (nodes, edges) within 1 hop of the given node ids.

    Includes the seed nodes themselves. Edges are deduplicated.
    """
    global _KG_TABLES_MISSING
    if _KG_TABLES_MISSING or not node_ids:
        return [], []
    client = get_client()
    seed_ids = list({n for n in node_ids if n})
    try:
        edges_out = (
            client.table("knowledge_edges").select("*")
            .in_("source_node_id", seed_ids).limit(max_edges).execute().data or []
        )
        edges_in = (
            client.table("knowledge_edges").select("*")
            .in_("target_node_id", seed_ids).limit(max_edges).execute().data or []
        )
    except Exception as exc:
        if _kg_unavailable(exc):
            _KG_TABLES_MISSING = True
        return [], []

    edges: dict[str, dict] = {}
    related_ids: set[str] = set(seed_ids)
    for e in [*edges_out, *edges_in]:
        if _edge_is_inactive(e):
            continue
        edges[e["id"]] = e
        related_ids.add(e["source_node_id"])
        related_ids.add(e["target_node_id"])

    try:
        nodes = (
            client.table("knowledge_nodes").select("*")
            .in_("id", list(related_ids)).execute().data or []
        )
    except Exception as exc:
        if _kg_unavailable(exc):
            _KG_TABLES_MISSING = True
        return [], []

    return nodes, list(edges.values())


def list_knowledge_nodes_by_type(
    node_types: list[str],
    persona_id: Optional[str] = None,
    limit: int = 200,
) -> list[dict]:
    """Helper used by chat-context: enumerate canonical product/campaign nodes
    so we can detect mentions in free-form lead text without an LLM call."""
    global _KG_TABLES_MISSING
    if _KG_TABLES_MISSING:
        return []
    client = get_client()
    try:
        q = client.table("knowledge_nodes").select("id,slug,title,node_type,tags,metadata,persona_id").in_("node_type", node_types).limit(limit)
        if persona_id:
            q = q.eq("persona_id", persona_id)
        return q.execute().data or []
    except Exception as exc:
        if _kg_unavailable(exc):
            _KG_TABLES_MISSING = True
        return []


def list_product_collection_nodes(
    *,
    persona_id: Optional[str] = None,
    node_type: str = "product_collection",
    limit: int = 500,
) -> list[dict]:
    global _KG_TABLES_MISSING
    if _KG_TABLES_MISSING:
        return []
    try:
        q = (
            get_client()
            .table("knowledge_nodes")
            .select("*")
            .eq("node_type", node_type)
            .neq("status", "archived")
            .order("title")
            .limit(limit)
        )
        if persona_id:
            q = q.eq("persona_id", persona_id)
        return _q(q.order("updated_at", desc=True).range(offset, offset + limit - 1))
    except Exception as exc:
        if _kg_unavailable(exc):
            _KG_TABLES_MISSING = True
        return []


def list_product_nodes(
    *,
    persona_id: Optional[str] = None,
    collection_slug: Optional[str] = None,
    category_slug: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 500,
) -> list[dict]:
    global _KG_TABLES_MISSING
    if _KG_TABLES_MISSING:
        return []
    try:
        q = (
            get_client()
            .table("knowledge_nodes")
            .select("*")
            .eq("node_type", "product")
            .neq("status", "archived")
            .order("title")
            .limit(limit)
        )
        if persona_id:
            q = q.eq("persona_id", persona_id)
        if status:
            q = q.eq("status", status)
        rows = _q(q)
        if collection_slug:
            rows = [r for r in rows if (r.get("metadata") or {}).get("collection_slug") == collection_slug]
        if category_slug:
            rows = [r for r in rows if (r.get("metadata") or {}).get("category_slug") == category_slug]
        return rows
    except Exception as exc:
        if _kg_unavailable(exc):
            _KG_TABLES_MISSING = True
        return []


def list_edges_for_nodes(node_ids: list[str], *, relation_types: Optional[list[str]] = None, limit: int = 5000) -> list[dict]:
    global _KG_TABLES_MISSING
    ids = sorted({str(node_id) for node_id in (node_ids or []) if node_id})
    if _KG_TABLES_MISSING or not ids:
        return []
    client = get_client()
    try:
        rows_by_id: dict[str, dict] = {}
        # UUID-heavy PostgREST ``in`` filters become long URLs quickly. The
        # unbounded version produced HTTP 414 in production. Bounded chunks
        # preserve the existing contract without a schema change.
        for index in range(0, len(ids), 75):
            chunk = ids[index:index + 75]
            for column in ("source_node_id", "target_node_id"):
                query = (
                    client.table("knowledge_edges")
                    .select("*")
                    .in_(column, chunk)
                    .limit(limit)
                )
                if relation_types:
                    query = query.in_("relation_type", relation_types)
                for row in query.execute().data or []:
                    identity = str(row.get("id") or (
                        str(row.get("source_node_id"))
                        + ":" + str(row.get("target_node_id"))
                        + ":" + str(row.get("relation_type"))
                    ))
                    rows_by_id[identity] = row
        return [
            row for row in list(rows_by_id.values())[:limit]
            if not _edge_is_inactive(row)
        ]
    except Exception as exc:
        if _kg_unavailable(exc):
            _KG_TABLES_MISSING = True
        return []


def list_knowledge_nodes_by_ids(node_ids: list[str]) -> list[dict]:
    global _KG_TABLES_MISSING
    ids = list({node_id for node_id in (node_ids or []) if node_id})
    if _KG_TABLES_MISSING or not ids:
        return []
    try:
        rows: list[dict] = []
        for index in range(0, len(ids), 75):
            chunk = ids[index:index + 75]
            rows.extend(_q(
                get_client().table("knowledge_nodes").select("*")
                .in_("id", chunk).limit(len(chunk))
            ))
        return rows
    except Exception as exc:
        if _kg_unavailable(exc):
            _KG_TABLES_MISSING = True
        return []


def list_all_knowledge_graph(persona_id: Optional[str] = None, limit_nodes: int = 1500) -> tuple[list[dict], list[dict]]:
    """Return all nodes + edges (optionally scoped to persona). Used by /knowledge/graph-data."""
    global _KG_TABLES_MISSING
    if _KG_TABLES_MISSING:
        return [], []
    client = get_client()
    try:
        nq = client.table("knowledge_nodes").select("*").limit(limit_nodes)
        if persona_id:
            nq = nq.eq("persona_id", persona_id)
        nodes = nq.execute().data or []
    except Exception as exc:
        if _kg_unavailable(exc):
            _KG_TABLES_MISSING = True
        return [], []
    if not nodes:
        return [], []
    node_ids = [n["id"] for n in nodes]
    try:
        eq_in_source = client.table("knowledge_edges").select("*").in_("source_node_id", node_ids).limit(5000).execute().data or []
    except Exception:
        eq_in_source = []
    active_edges = [edge for edge in eq_in_source if not _edge_is_inactive(edge)]
    return nodes, active_edges


def _asset_table_unavailable(exc: Exception) -> bool:
    text = str(exc)
    return "assets" in text and ("PGRST205" in text or "schema cache" in text or "Could not find" in text)


def sync_gallery_asset_node(node: dict, edge: dict) -> Optional[dict]:
    """Mirror a Gallery-linked knowledge node into the existing assets table."""
    if not node or not edge:
        return None
    client = get_client()
    metadata = node.get("metadata") or {}
    node_type = (node.get("node_type") or "").lower()
    file_path = metadata.get("file_path") or metadata.get("path") or metadata.get("url")
    ext = str(file_path).rsplit(".", 1)[-1].lower() if file_path and "." in str(file_path) else ""
    asset_type = metadata.get("asset_type") or ("gallery_node" if node_type != "asset" else "asset")
    platform_type = "image" if ext in {"png", "jpg", "jpeg", "svg", "gif", "webp"} else ("campaign" if node_type == "campaign" else "template")
    payload = {
        "persona_id": node.get("persona_id") or edge.get("persona_id"),
        "type": platform_type,
        "name": node.get("title") or node.get("slug") or "Gallery asset",
        "url": metadata.get("url") if metadata.get("url") else None,
        "metadata": {
            **metadata,
            "knowledge_node_id": node.get("id"),
            "knowledge_edge_id": edge.get("id"),
            "source_table": node.get("source_table"),
            "source_id": node.get("source_id"),
            "node_type": node_type,
            "file_path": file_path,
            "gallery_active": True,
        },
        "source": "imported",
        "asset_type": asset_type,
        "asset_function": metadata.get("asset_function") or "gallery_reference",
        "tags": node.get("tags") or [],
        "description": node.get("summary"),
        "embedding_status": "none",
        "approval_status": "approved",
        "knowledge_node_id": node.get("id"),
        "gallery_edge_id": edge.get("id"),
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    try:
        existing = _one(client.table("assets").select("id").eq("knowledge_node_id", node.get("id")).maybe_single())
        if existing:
            result = _execute_with_retry(client.table("assets").update(payload).eq("id", existing["id"]))
        else:
            result = _execute_with_retry(client.table("assets").insert(payload))
        return (result.data or [payload])[0]
    except Exception as exc:
        if _asset_table_unavailable(exc):
            return None
        try:
            from services import sre_logger
            sre_logger.error("supabase_client", f"sync_gallery_asset_node failed: {exc}", exc)
        except Exception:
            pass
        return None


def _faq_publication_payload(node: dict, item: Optional[dict], edge: Optional[dict] = None) -> dict:
    metadata = node.get("metadata") or {}
    question = (
        metadata.get("question")
        or (item or {}).get("title")
        or node.get("title")
        or node.get("slug")
        or "FAQ"
    )
    answer = (
        metadata.get("answer")
        or (item or {}).get("content")
        or node.get("summary")
        or metadata.get("content")
        or metadata.get("description")
        or question
    )
    file_path = normalize_file_path(
        (item or {}).get("file_path")
        or metadata.get("file_path")
        or metadata.get("path")
        or metadata.get("url")
    )
    path_slugs = []
    for value in (
        (item or {}).get("metadata", {}).get("path_slugs"),
        metadata.get("path_slugs"),
    ):
        if isinstance(value, list) and value:
            path_slugs = [str(v) for v in value if v]
            break
    return {
        "question": question,
        "answer": answer,
        "title": question,
        "content": answer,
        "file_path": file_path,
        "path_slugs": path_slugs,
        "tags": (item or {}).get("tags") or node.get("tags") or [],
        "persona_id": (edge or {}).get("persona_id") or node.get("persona_id") or (item or {}).get("persona_id"),
    }


def sync_embedded_kb_node(node: dict, edge: dict) -> Optional[dict]:
    """Publish an Embedded-linked FAQ into the Golden Dataset + RAG tables."""
    if not node or not edge:
        return None
    from datetime import datetime, timezone
    import hashlib

    persona_id = edge.get("persona_id") or node.get("persona_id")
    if not persona_id:
        return None
    metadata = node.get("metadata") or {}
    source_table = node.get("source_table")
    source_id = node.get("source_id")

    item = None
    if source_table == "knowledge_items" and source_id:
        try:
            item = get_knowledge_item(source_id)
        except Exception:
            item = None

    node_type = (node.get("node_type") or (item or {}).get("content_type") or "other").lower()
    if node_type != "faq":
        raise ValueError("Only approved FAQ nodes can be published to the Golden Dataset")
    if item and str(item.get("status") or "").lower() not in {"approved", "embedded"}:
        raise ValueError("Approve the FAQ before publishing it to the Golden Dataset")
    faq_payload = _faq_publication_payload(node, item, edge)
    title = faq_payload["title"]
    content = faq_payload["content"]
    question = faq_payload["question"]
    answer = faq_payload["answer"]
    file_path = faq_payload["file_path"]
    path_slugs = faq_payload["path_slugs"]
    tags = faq_payload["tags"]
    kb_id = "gn_" + hashlib.md5(f"{node.get('id')}:{persona_id}".encode()).hexdigest()[:12]

    if item:
        update_knowledge_item(source_id, {
            "status": "embedded",
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "persona_id": persona_id,
        })

    entry = upsert_kb_entry({
        "kb_id": kb_id,
        "persona_id": persona_id,
        "tipo": "faq",
        "categoria": node_type,
        "titulo": title,
        "conteudo": content,
        "link": file_path,
        "status": "ATIVO",
        "source": "graph_embed",
        "agent_visibility": (item or {}).get("agent_visibility") or ["SDR", "Closer", "Classifier"],
        "tags": tags,
        "embedding_status": "created",
    })
    rag_entry = None
    rag_chunks: list[dict] = []
    if entry:
        from services import knowledge_rag_intake as _rag_intake
        if _rag_intake.is_rag_eligible(node_type):
            rag_entry = upsert_knowledge_rag_entry({
                "persona_id": persona_id,
                "artifact_id": metadata.get("artifact_id"),
                "content_type": "faq",
                "semantic_level": int(node.get("level") or 50),
                "title": title,
                "question": question,
                "answer": answer,
                "content": content,
                "summary": node.get("summary") or content[:280],
                "canonical_key": f"kb:{persona_id}:{kb_id}",
                "slug": _slugify(title),
                "status": "active",
                "tags": tags,
                "products": [title] if node_type == "product" else [],
                "campaigns": [title] if node_type == "campaign" else [],
                "metadata": {
                    **metadata,
                    "kb_entry_id": entry.get("id"),
                    "knowledge_node_id": node.get("id"),
                    "graph_edge_id": edge.get("id"),
                    "node_type": node_type,
                    "original_node_type": node_type,
                    "file_path": file_path,
                    "path": file_path,
                    "path_slugs": path_slugs,
                    "question": question,
                    "answer": answer,
                    "persona_id": persona_id,
                    "source_knowledge_item_id": source_id,
                },
                "confidence": float(node.get("confidence") or 0.8),
                "importance": float(node.get("importance") or 0.7),
            })
            if rag_entry and rag_entry.get("id"):
                rag_chunks = replace_knowledge_rag_chunks(
                    rag_entry["id"],
                    persona_id,
                    [{
                        "chunk_text": content,
                        "chunk_summary": (node.get("summary") or title)[:280],
                        "metadata": {
                            "source": "graph_embed",
                            "file_path": file_path,
                            "path": file_path,
                            "path_slugs": path_slugs,
                            "question": question,
                            "answer": answer,
                            "persona_id": persona_id,
                            "knowledge_item_id": source_id,
                            "knowledge_node_id": node.get("id"),
                        },
                    }],
                )
        if item:
            next_meta = {
                **((get_knowledge_item(source_id) or item).get("metadata") or {}),
                "kb_entry_id": entry.get("id"),
                "knowledge_rag_entry_id": (rag_entry or {}).get("id"),
                "embedded_edge_id": edge.get("id"),
                "golden_dataset_file_path": file_path,
                "golden_dataset_question": question,
                "golden_dataset_answer": answer,
            }
            update_knowledge_item(source_id, {
                "status": "embedded",
                "metadata": next_meta,
            })
    return {
        "item": get_knowledge_item(source_id) if source_id and source_table == "knowledge_items" else item,
        "kb_entry": entry,
        "rag_entry": rag_entry,
        "chunks": rag_chunks,
        "embedded_edge": edge,
    }


def reset_embedded_legacy_publications(persona_id: Optional[str] = None) -> dict:
    """Deactivate Embedded links and remove Golden Dataset/RAG mirrors."""
    client = get_client()
    embedded_q = client.table("knowledge_nodes").select("id,persona_id").eq("node_type", "embedded")
    if persona_id:
        embedded_q = embedded_q.eq("persona_id", persona_id)
    embedded_nodes = _q(embedded_q.limit(500))

    report = {
        "persona_id": persona_id,
        "embedded_nodes": len(embedded_nodes),
        "edges_deactivated": 0,
        "items_reverted": 0,
        "kb_entries_deleted": 0,
        "rag_entries_deleted": 0,
        "kb_mirror_nodes_deleted": 0,
    }

    for embedded in embedded_nodes:
        edge_rows = _q(
            client.table("knowledge_edges")
            .select("*")
            .eq("target_node_id", embedded.get("id"))
            .limit(500)
        )
        for edge in edge_rows:
            if _edge_is_inactive(edge):
                continue
            metadata = {
                **(edge.get("metadata") or {}),
                "active": False,
                "deleted_from": "reset_embedded_legacy_publications",
                "deleted_at": datetime.now(timezone.utc).isoformat(),
            }
            _execute_with_retry(client.table("knowledge_edges").update({"metadata": metadata}).eq("id", edge["id"]))
            report["edges_deactivated"] += 1

            source_node = get_knowledge_node(edge.get("source_node_id") or "")
            edge_meta = edge.get("metadata") or {}
            kb_entry_id = edge_meta.get("kb_entry_id")
            rag_entry_id = edge_meta.get("knowledge_rag_entry_id")

            item = None
            if source_node and source_node.get("source_table") == "knowledge_items" and source_node.get("source_id"):
                item = get_knowledge_item(str(source_node.get("source_id")))
                item_meta = (item or {}).get("metadata") or {}
                kb_entry_id = item_meta.get("kb_entry_id") or kb_entry_id
                rag_entry_id = item_meta.get("knowledge_rag_entry_id") or rag_entry_id

            if rag_entry_id and delete_knowledge_rag_entry(str(rag_entry_id)):
                report["rag_entries_deleted"] += 1
            if kb_entry_id and delete_kb_entry(str(kb_entry_id)):
                report["kb_entries_deleted"] += 1
                kb_node = get_knowledge_node_for_source("kb_entries", str(kb_entry_id), persona_id=embedded.get("persona_id"))
                if kb_node and delete_knowledge_node(str(kb_node.get("id"))):
                    report["kb_mirror_nodes_deleted"] += 1

            if item and item.get("id"):
                next_meta = {
                    **((item.get("metadata") or {})),
                }
                next_meta.pop("kb_entry_id", None)
                next_meta.pop("knowledge_rag_entry_id", None)
                next_meta.pop("embedded_edge_id", None)
                update_knowledge_item(str(item["id"]), {
                    "status": "approved",
                    "metadata": next_meta,
                })
                report["items_reverted"] += 1

    return report


def mark_gallery_asset_inactive_by_edge(edge_id: str) -> None:
    if not edge_id:
        return
    client = get_client()
    try:
        rows = _q(client.table("assets").select("id,metadata").eq("gallery_edge_id", edge_id).limit(50))
        for row in rows:
            metadata = {**(row.get("metadata") or {}), "gallery_active": False}
            _execute_with_retry(client.table("assets").update({"metadata": metadata}).eq("id", row["id"]))
    except Exception:
        return



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


def _asset_display_url(asset_row: dict) -> str:
    signed = _storage_signed_url(asset_row.get("storage_bucket"), asset_row.get("storage_path"))
    return signed or asset_row.get("url") or ""


def asset_display_url(asset_row: dict) -> str:
    """Return the renderable URL for an asset row.

    Storage location is the source of truth. The persisted `url` column is kept
    only as a legacy fallback because signed URLs expire and public URLs do not
    work for private buckets.
    """
    return _asset_display_url(asset_row)


def list_gallery_assets(persona_id: Optional[str] = None, limit: int = 250) -> list[dict]:
    """Return knowledge nodes connected to the protected Gallery node."""
    global _KG_TABLES_MISSING
    if _KG_TABLES_MISSING:
        return []
    client = get_client()
    try:
        gallery_q = client.table("knowledge_nodes").select("id").eq("node_type", "gallery").eq("status", "active")
        if persona_id:
            gallery_q = gallery_q.eq("persona_id", persona_id)
        galleries = gallery_q.limit(100).execute().data or []
        gallery_ids = [row["id"] for row in galleries if row.get("id")]
        if not gallery_ids:
            return []
        source_edges = (
            client.table("knowledge_edges")
            .select("*")
            .eq("relation_type", "gallery_asset")
            .in_("source_node_id", gallery_ids)
            .limit(limit)
            .execute().data or []
        )
        target_edges = (
            client.table("knowledge_edges")
            .select("*")
            .eq("relation_type", "gallery_asset")
            .in_("target_node_id", gallery_ids)
            .limit(limit)
            .execute().data or []
        )
        edges = source_edges + target_edges
        edges = [edge for edge in edges if not _edge_is_inactive(edge)]
        content_ids = [
            edge.get("target_node_id") if edge.get("source_node_id") in gallery_ids else edge.get("source_node_id")
            for edge in edges
        ]
        content_ids = [node_id for node_id in content_ids if node_id]
        if not content_ids:
            return []
        nodes = (
            client.table("knowledge_nodes")
            .select("*")
            .in_("id", content_ids)
            .neq("status", "archived")
            .limit(limit)
            .execute().data or []
        )
    except Exception as exc:
        if _kg_unavailable(exc):
            _KG_TABLES_MISSING = True
        return []
    edge_by_content = {
        (edge.get("target_node_id") if edge.get("source_node_id") in gallery_ids else edge.get("source_node_id")): edge
        for edge in edges
    }

    # Resolve each gallery node to its underlying public.assets row so the
    # /assets page receives real asset UUIDs (not gn:<node_id>) and the
    # actual image URL â€” old gallery nodes used to store an .md companion
    # in metadata.file_path, which made cards render `.md` placeholders
    # and crashed /assets/{id} with `invalid uuid`.
    asset_ids: list[str] = []
    for node in nodes:
        if node.get("source_table") == "assets" and node.get("source_id"):
            asset_ids.append(str(node["source_id"]))
        else:
            meta_aid = (node.get("metadata") or {}).get("asset_id")
            if meta_aid:
                asset_ids.append(str(meta_aid))
    assets_by_id: dict[str, dict] = {}
    if asset_ids:
        try:
            asset_rows = (
                client.table("assets")
                .select("*")
                .in_("id", list({aid for aid in asset_ids if aid}))
                .execute().data or []
            )
            assets_by_id = {row["id"]: row for row in asset_rows if row.get("id")}
        except Exception:
            assets_by_id = {}

    out = []
    for node in nodes:
        metadata = node.get("metadata") or {}
        asset_id = None
        if node.get("source_table") == "assets" and node.get("source_id"):
            asset_id = str(node["source_id"])
        elif metadata.get("asset_id"):
            asset_id = str(metadata["asset_id"])
        asset_row = assets_by_id.get(asset_id) if asset_id else None

        # Orphan gallery node (no public.assets row backing it) â€” skip from
        # the assets list. It is not a visual asset, just a stray markdown
        # reference from a legacy flow.
        if not asset_row:
            continue
        # Markdown companions are inline content of the parent image; never
        # surface them as standalone cards.
        if (asset_row.get("type") or "").lower() == "markdown":
            continue

        asset_meta = asset_row.get("metadata") or {}
        effective_status = asset_meta.get("validation_status") or asset_row.get("status") or node.get("status") or "ready"
        original = asset_row.get("original_filename") or asset_meta.get("original_filename") or asset_row.get("name") or ""
        ext = original.rsplit(".", 1)[-1].lower() if "." in original else ""
        if not ext:
            mime = asset_row.get("mime_type") or ""
            ext = mime.split("/")[-1].lower() if "/" in mime else ""
        storage_path = (
            f"{asset_row.get('storage_bucket')}:{asset_row.get('storage_path')}"
            if asset_row.get("storage_bucket") and asset_row.get("storage_path")
            else asset_meta.get("storage_path") or asset_row.get("url")
        )
        out.append({
            "id": asset_row["id"],
            "title": asset_row.get("name") or original or node.get("title") or "Gallery asset",
            "status": effective_status,
            "content_type": "asset",
            "asset_type": asset_row.get("type") or asset_meta.get("kind") or metadata.get("asset_type"),
            "asset_function": asset_meta.get("asset_function") or metadata.get("asset_function") or "gallery_reference",
            "file_type": ext or None,
            "file_path": storage_path,
            "url": _asset_display_url(asset_row),
            "persona_id": asset_row.get("persona_id") or node.get("persona_id"),
            "created_at": asset_row.get("created_at") or node.get("created_at"),
            "source": "gallery",
            "summary": node.get("summary"),
            "tags": node.get("tags") or [],
            "knowledge_node_id": node.get("id"),
            "gallery_edge_id": (edge_by_content.get(node.get("id")) or {}).get("id"),
        })
    return out


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


def get_node_type_registry() -> list[dict]:
    """Return the knowledge_node_type_registry rows (migration 009).

    Caches the result for _REGISTRY_TTL_SECONDS to avoid querying on every
    request. Falls back to a hardcoded mirror of the seed inserts when the
    table is missing or empty so the graph endpoint stays useful.
    """
    global _NODE_TYPE_REGISTRY_CACHE
    now = time.monotonic()
    if _NODE_TYPE_REGISTRY_CACHE and (now - _NODE_TYPE_REGISTRY_CACHE[0]) < _REGISTRY_TTL_SECONDS:
        return _NODE_TYPE_REGISTRY_CACHE[1]
    rows: list[dict] = []
    try:
        rows = (
            get_client().table("knowledge_node_type_registry")
            .select("node_type,label,default_level,default_importance,color,icon,sort_order,active")
            .execute().data or []
        )
        rows = [r for r in rows if r.get("active", True)]
    except Exception:
        rows = []
    if not rows:
        rows = _NODE_TYPE_REGISTRY_FALLBACK
    _NODE_TYPE_REGISTRY_CACHE = (now, rows)
    return rows


def get_relation_type_registry() -> list[dict]:
    """Return the knowledge_relation_type_registry rows (migration 009).

    Same cache + fallback strategy as get_node_type_registry.
    """
    global _RELATION_TYPE_REGISTRY_CACHE
    now = time.monotonic()
    if _RELATION_TYPE_REGISTRY_CACHE and (now - _RELATION_TYPE_REGISTRY_CACHE[0]) < _REGISTRY_TTL_SECONDS:
        return _RELATION_TYPE_REGISTRY_CACHE[1]
    rows: list[dict] = []
    try:
        rows = (
            get_client().table("knowledge_relation_type_registry")
            .select("relation_type,label,inverse_label,default_weight,directional,sort_order,active")
            .execute().data or []
        )
        rows = [r for r in rows if r.get("active", True)]
    except Exception:
        rows = []
    if not rows:
        rows = _RELATION_TYPE_REGISTRY_FALLBACK
    _RELATION_TYPE_REGISTRY_CACHE = (now, rows)
    return rows


# â”€â”€ Insights â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_insights(status: Optional[str] = None, limit: int = 50) -> list:
    try:
        q = get_client().table("flow_insights").select("*").order("created_at", desc=True).limit(limit)
        if status:
            q = q.eq("status", status)
        return _q(q)
    except Exception as exc:
        try:
            from services import sre_logger
            sre_logger.error("supabase_client", f"get_insights failed: {exc}", exc)
        except Exception:
            pass
        return []


def insert_insight(data: dict) -> None:
    get_client().table("flow_insights").insert(data).execute()


def update_insight(insight_id: str, data: dict) -> None:
    get_client().table("flow_insights").update(data).eq("id", insight_id).execute()


def get_open_insights_titles() -> list[str]:
    rows = _q(get_client().table("flow_insights").select("title").eq("status", "open"))
    return [r["title"] for r in rows if r.get("title")]


# â”€â”€ System Health â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def insert_health_snapshot(data: dict) -> None:
    get_client().table("system_health").insert(data).execute()


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


def get_persona_configs_by_ids(
    persona_ids: list[str], *, chunk_size: int = 50,
) -> dict[str, dict]:
    """`config` de várias personas numa leitura, para decorar listas de leads.

    A tela de Mensagens precisa do `business_model` de cada lead para saber se
    o pedido é compra/entrega ou agendamento/conclusão. Uma consulta por lead
    seria N+1 numa lista de 500.
    """
    ids = sorted({str(value) for value in persona_ids if value})
    if not ids:
        return {}
    size = max(1, min(chunk_size, 100))
    configs: dict[str, dict] = {}
    for index in range(0, len(ids), size):
        for row in _q(
            get_client().table("personas").select("id,config")
            .in_("id", ids[index:index + size])
        ):
            if row.get("id"):
                configs[str(row["id"])] = dict(row.get("config") or {})
    return configs


def upsert_persona(data: dict) -> None:
    get_client().table("personas").upsert(data, on_conflict="slug").execute()


def list_public_site_formats(enabled_only: bool = True) -> list:
    q = get_client().table("public_site_formats").select("*").order("sort_order").order("label")
    if enabled_only:
        q = q.eq("enabled", True)
    rows = _q(q)
    return rows or [dict(row) for row in DEFAULT_FORMATS if row.get("enabled") or not enabled_only]


def get_public_site_format(key: str) -> Optional[dict]:
    if not key:
        return None
    row = _one(get_client().table("public_site_formats").select("*").eq("key", key).eq("enabled", True).maybe_single())
    if row:
        return row
    return next((dict(fmt) for fmt in DEFAULT_FORMATS if fmt.get("key") == key and fmt.get("enabled")), None)


def update_persona_config(slug: str, config: dict, *, catalog_url: Any = _UNSET) -> Optional[dict]:
    payload: dict[str, Any] = {"config": config or {}}
    if catalog_url is not _UNSET:
        payload["catalog_url"] = catalog_url
    _execute_with_retry(get_client().table("personas").update(payload).eq("slug", slug))
    return get_persona(slug)


_PERSONA_ROUTING_FIELDS = (
    "process_mode",
    "outbound_webhook_url",
    "outbound_webhook_secret",
    "inbound_webhook_token",
)


def get_persona_routing(slug: str) -> Optional[dict]:
    """Returns the routing config for a persona, or None if missing.

    Falls back gracefully when migration 011 is not yet applied (older
    columns will be missing â€” the function returns defaults so callers can
    keep working without crashing).
    """
    persona = get_persona(slug)
    if not persona:
        return None
    migration_applied = all(field in persona for field in _PERSONA_ROUTING_FIELDS)
    legacy_bindings = get_workflow_bindings(persona.get("id")) if persona.get("id") else []
    has_legacy_n8n = any(binding.get("active", True) for binding in legacy_bindings)
    process_mode = persona.get("process_mode") if migration_applied else None
    if not process_mode:
        process_mode = "n8n" if has_legacy_n8n else "internal"
    return {
        "slug": persona.get("slug"),
        "id": persona.get("id"),
        "process_mode": process_mode,
        "config": persona.get("config") or {},
        "outbound_webhook_url": persona.get("outbound_webhook_url"),
        "outbound_webhook_secret": persona.get("outbound_webhook_secret"),
        "inbound_webhook_token": persona.get("inbound_webhook_token"),
        "migration_applied": migration_applied,
        "routing_source": "persona_columns" if migration_applied else ("legacy_workflow_binding" if has_legacy_n8n else "default"),
    }


def update_persona_routing(slug: str, data: dict) -> Optional[dict]:
    """Partial update of persona routing fields. Ignores unknown keys."""
    payload = {k: v for k, v in (data or {}).items() if k in _PERSONA_ROUTING_FIELDS}
    if not payload:
        return get_persona_routing(slug)
    try:
        _execute_with_retry(
            get_client().table("personas").update(payload).eq("slug", slug)
        )
    except Exception as exc:
        try:
            from services import sre_logger
            sre_logger.error("supabase_client", f"update_persona_routing failed: {exc}", exc)
        except Exception:
            pass
        raise
    return get_persona_routing(slug)


# â”€â”€ Knowledge Base â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_kb_entries(persona_id: Optional[str] = None, status: str = "ATIVO") -> list:
    q = get_client().table("kb_entries").select("id,persona_id,tipo,categoria,produto,intencao,titulo,conteudo,link,prioridade,status,source,tags,agent_visibility,updated_at")
    if persona_id:
        q = q.eq("persona_id", persona_id)
    if status:
        q = q.eq("status", status)
    return _q(q.order("prioridade"))


def get_kb_entries_for_persona_ids(persona_ids: list[str], status: str = "ATIVO") -> list:
    ids = [pid for pid in persona_ids if pid]
    if not ids:
        return []
    q = get_client().table("kb_entries").select("id,persona_id,tipo,categoria,produto,intencao,titulo,conteudo,link,prioridade,status,source,tags,agent_visibility,updated_at")
    q = q.in_("persona_id", ids)
    if status:
        q = q.eq("status", status)
    return _q(q.order("prioridade"))


def _kb_entry_select():
    return (
        get_client()
        .table("kb_entries")
        .select("id,persona_id,kb_id,tipo,categoria,produto,intencao,titulo,conteudo,link,prioridade,status,source,tags,agent_visibility,updated_at")
    )


def _find_kb_entry_by_key(kb_id: Optional[str], persona_id: Optional[str]) -> Optional[dict]:
    if not kb_id:
        return None
    q = _kb_entry_select().eq("kb_id", kb_id)
    q = q.eq("persona_id", persona_id) if persona_id else q.is_("persona_id", "null")
    return _one(q.maybe_single())


def _log_kb_entry_write_failure(stage: str, payload: dict, exc: Exception) -> None:
    try:
        from services import sre_logger
        sre_logger.error(
            "supabase_client",
            (
                f"kb_entries {stage} failed: {type(exc).__name__}: {exc} "
                f"(kb_id={payload.get('kb_id')!r}, persona_id={payload.get('persona_id')!r}, source={payload.get('source')!r})"
            ),
            exc,
        )
    except Exception:
        pass


_MISSING_COLUMN_RE = re.compile(r"Could not find the '([^']+)' column of '([^']+)'")


def _drop_missing_kb_entry_column(payload: dict, exc: Exception) -> tuple[dict, Optional[str]]:
    match = _MISSING_COLUMN_RE.search(str(exc))
    if not match:
        return payload, None
    column_name, table_name = match.groups()
    if table_name != "kb_entries" or column_name not in payload:
        return payload, None
    sanitized = dict(payload)
    sanitized.pop(column_name, None)
    return sanitized, column_name


def upsert_kb_entry(data: dict) -> dict:
    payload = dict(data or {})
    kb_id = payload.get("kb_id")
    persona_id = payload.get("persona_id")
    last_exc: Exception | None = None

    try:
        result = _execute_with_retry(get_client().table("kb_entries").upsert(payload, on_conflict="kb_id,persona_id"))
        rows = result.data or []
        return rows[0] if rows else (_find_kb_entry_by_key(kb_id, persona_id) or {})
    except Exception as exc:
        last_exc = exc
        _log_kb_entry_write_failure("upsert", payload, exc)
        payload, dropped_column = _drop_missing_kb_entry_column(payload, exc)
        if dropped_column:
            try:
                result = _execute_with_retry(get_client().table("kb_entries").upsert(payload, on_conflict="kb_id,persona_id"))
                rows = result.data or []
                return rows[0] if rows else (_find_kb_entry_by_key(kb_id, persona_id) or {})
            except Exception as retry_exc:
                last_exc = retry_exc
                _log_kb_entry_write_failure(f"upsert-without-{dropped_column}", payload, retry_exc)

    fallback_payload = dict(payload)
    if fallback_payload.get("source") == "graph_embed":
        fallback_payload["source"] = "manual"
        try:
            result = _execute_with_retry(get_client().table("kb_entries").upsert(fallback_payload, on_conflict="kb_id,persona_id"))
            rows = result.data or []
            return rows[0] if rows else (_find_kb_entry_by_key(kb_id, persona_id) or {})
        except Exception as exc:
            last_exc = exc
            _log_kb_entry_write_failure("upsert-fallback-source", fallback_payload, exc)

    existing = _find_kb_entry_by_key(kb_id, persona_id)
    mutable = {
        key: value
        for key, value in fallback_payload.items()
        if key not in {"id", "kb_id", "persona_id", "created_at"}
    }
    try:
        if existing and existing.get("id"):
            result = _execute_with_retry(get_client().table("kb_entries").update(mutable).eq("id", existing["id"]))
            rows = result.data or []
            return rows[0] if rows else (get_kb_entry(existing["id"]) or {**existing, **mutable})
        result = _execute_with_retry(get_client().table("kb_entries").insert(fallback_payload))
        rows = result.data or []
        if rows:
            return rows[0]
        return _find_kb_entry_by_key(kb_id, persona_id) or {}
    except Exception as exc:
        _log_kb_entry_write_failure("manual-write", fallback_payload, exc)
        if last_exc:
            raise exc from last_exc
        raise


def get_kb_entry(entry_id: str) -> Optional[dict]:
    return _one(
        _kb_entry_select()
        .eq("id", entry_id)
        .maybe_single()
    )


def get_kb_entries_by_ids(ids: list) -> dict:
    """Batch lookup; avoids N+1 when enriching graph kb_entry nodes."""
    unique = [i for i in {str(x) for x in (ids or []) if x}]
    if not unique:
        return {}
    rows: list = []
    # Supabase/PostgREST .in_ has a URL length cap; chunk to be safe.
    for start in range(0, len(unique), 200):
        chunk = unique[start:start + 200]
        rows.extend(_q(
            _kb_entry_select()
            .in_("id", chunk)
        ))
    return {str(r["id"]): r for r in rows if r.get("id")}


def update_kb_entry(entry_id: str, data: dict) -> None:
    from datetime import datetime, timezone
    payload = dict(data or {})
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        _execute_with_retry(get_client().table("kb_entries").update(payload).eq("id", entry_id))
    except Exception as exc:
        payload, dropped_column = _drop_missing_kb_entry_column(payload, exc)
        if not dropped_column:
            raise
        _execute_with_retry(get_client().table("kb_entries").update(payload).eq("id", entry_id))


def delete_kb_entry(entry_id: str) -> bool:
    result = _execute_with_retry(get_client().table("kb_entries").delete().eq("id", entry_id))
    return bool(result.data)


def search_kb(query_embedding: list, persona_id: Optional[str] = None, top_k: int = 5) -> list:
    params = {"query_embedding": query_embedding, "match_count": top_k}
    if persona_id:
        params["filter_persona_id"] = persona_id
    return _q(get_client().rpc("match_kb_entries", params))


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


def _normalize_agent_log_row(row: dict) -> dict:
    if not isinstance(row, dict):
        return {}
    if "agent_type" in row or "action" in row or "decision" in row:
        meta = row.get("metadata") or {}
        return {
            **row,
            "agent_type": row.get("agent_type") or meta.get("component") or row.get("agent_name"),
            "action": row.get("action") or meta.get("message") or "",
            "decision": row.get("decision") or meta.get("traceback") or row.get("error_msg") or "",
            "metadata": meta,
            "level": meta.get("level") or ("ERROR" if str(row.get("action") or "").startswith("[ERROR]") else "INFO"),
            "component": meta.get("component") or row.get("agent_type") or row.get("agent_name") or "",
            "message": meta.get("message") or row.get("action") or "",
            "traceback": meta.get("traceback") or row.get("decision") or "",
            "ts": meta.get("ts") or row.get("created_at") or "",
        }

    output = row.get("output") if isinstance(row.get("output"), dict) else {}
    input_payload = row.get("input") if isinstance(row.get("input"), dict) else {}
    status = str(row.get("status") or "success").lower()
    level = "ERROR" if status in {"error", "timeout", "warn", "warning"} or row.get("error_msg") else "INFO"
    message = row.get("error_msg") or output.get("reply") or output.get("summary") or status
    metadata = {
        "level": level,
        "component": row.get("agent_name") or "agent",
        "message": message,
        "traceback": row.get("error_msg") or "",
        "ts": row.get("created_at"),
        "input": input_payload,
        "output": output,
        "latency_ms": row.get("latency_ms"),
        "model_used": row.get("model_used"),
        "legacy_schema": True,
    }
    return {
        **row,
        "agent_type": row.get("agent_name") or "agent",
        "action": f"[{level}] {str(message)[:200]}",
        "decision": row.get("error_msg") or json.dumps(output or input_payload, ensure_ascii=False)[:500],
        "metadata": metadata,
        "level": level,
        "component": row.get("agent_name") or "agent",
        "message": message,
        "traceback": row.get("error_msg") or "",
        "ts": row.get("created_at") or "",
    }


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


def get_agent_logs(
    lead_id: Optional[str] = None,
    limit: int = 50,
    persona_id: Optional[str] = None,
) -> list:
    fetch_limit = min(max(limit * 4, limit), 1000) if persona_id else limit
    q = get_client().table("agent_logs").select("*").order("created_at", desc=True).limit(fetch_limit)
    if lead_id:
        q = q.eq("lead_id", lead_id)
    rows = _q(q)
    normalized = [_normalize_agent_log_row(row) for row in rows]
    if persona_id:
        normalized = [
            row for row in normalized
            if str(
                row.get("persona_id")
                or (row.get("payload") or {}).get("persona_id")
                or (row.get("metadata") or {}).get("persona_id")
                or ""
            ) == str(persona_id)
        ]
    return normalized[:limit]


def get_error_logs(
    component: Optional[str] = None,
    limit: int = 100,
    persona_id: Optional[str] = None,
) -> list:
    rows = get_agent_logs(limit=limit, persona_id=persona_id)
    filtered = []
    for row in rows:
        level = str(row.get("level") or "").upper()
        if level not in {"ERROR", "WARN", "WARNING"}:
            continue
        if component and str(row.get("component") or row.get("agent_type") or "").lower() != component.lower():
            continue
        filtered.append(row)
    return filtered


# â”€â”€ n8n Executions Mirror â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def upsert_n8n_execution(data: dict) -> None:
    get_client().table("n8n_executions").upsert(data, on_conflict="n8n_id").execute()


def get_n8n_executions(limit: int = 100, status: Optional[str] = None) -> list:
    q = (
        get_client().table("n8n_executions")
        .select("*")
        .order("started_at", desc=True)
        .limit(limit)
    )
    if status:
        q = q.eq("status", status)
    return _q(q)


def get_n8n_error_rate(hours: int = 24) -> float:
    from datetime import datetime, timedelta
    since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    all_rows = _q(
        get_client().table("n8n_executions")
        .select("status")
        .gte("started_at", since)
    )
    if not all_rows:
        return 0.0
    errors = sum(1 for r in all_rows if r.get("status") == "error")
    return errors / len(all_rows)


# â”€â”€ Knowledge Sources â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_knowledge_source_by_path(path: str) -> Optional[dict]:
    return _one(get_client().table("knowledge_sources").select("*").eq("path", path).maybe_single())


def insert_knowledge_source(data: dict) -> dict:
    return _insert_one(get_client().table("knowledge_sources").insert(data))


def update_knowledge_source(source_id: str, data: dict) -> None:
    get_client().table("knowledge_sources").update(data).eq("id", source_id).execute()


def get_or_create_manual_source() -> dict:
    existing = _one(get_client().table("knowledge_sources").select("*").eq("source_type", "upload").maybe_single())
    if existing:
        return existing
    r = get_client().table("knowledge_sources").insert({"source_type": "upload", "name": "Manual Upload"}).execute()
    return (r.data or [{}])[0]


# â”€â”€ Knowledge Items â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_knowledge_items(
    status: Optional[str] = None,
    persona_id: Optional[str] = None,
    content_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list:
    q = (
        get_client().table("knowledge_items")
        .select("*")
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
    )
    if status:
        q = q.eq("status", status)
    if persona_id:
        q = q.eq("persona_id", persona_id)
    if content_type:
        q = q.eq("content_type", content_type)
    return _q(q)


def get_knowledge_item(item_id: str) -> Optional[dict]:
    return _one(get_client().table("knowledge_items").select("*").eq("id", item_id).maybe_single())


def normalize_file_path(file_path: Optional[str]) -> Optional[str]:
    if not file_path:
        return None
    normalized = str(file_path).replace("\\", "/").strip()
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def get_knowledge_item_by_path(file_path: str) -> Optional[dict]:
    exact = _one(
        get_client().table("knowledge_items")
        .select("*")
        .eq("file_path", file_path)
        .maybe_single()
    )
    normalized = normalize_file_path(file_path)
    if exact or not normalized or normalized == file_path:
        return exact
    return _one(
        get_client().table("knowledge_items")
        .select("*")
        .eq("file_path", normalized)
        .maybe_single()
    )


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


def validate_knowledge_item_payload(payload: dict) -> list[str]:
    """Return a list of contract violations for a knowledge_items insert payload.

    Empty list = payload is safe to send to the DB. Mirrors NOT NULL, CHECK and
    foreign-key shape requirements from the schema.
    """
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["payload must be a dict"]

    persona_id = payload.get("persona_id")
    if not persona_id:
        errors.append("persona_id is required")
    elif not isinstance(persona_id, str) or not _UUID_RE.match(persona_id):
        errors.append(f"persona_id must be a UUID string, got {persona_id!r}")

    source_id = payload.get("source_id")
    if not source_id:
        errors.append("source_id is required")
    elif not isinstance(source_id, str) or not _UUID_RE.match(source_id):
        errors.append(f"source_id must be a UUID string, got {source_id!r}")

    content_type = payload.get("content_type")
    if not content_type:
        errors.append("content_type is required")
    elif content_type not in KNOWLEDGE_ITEM_CONTENT_TYPES:
        errors.append(
            f"content_type {content_type!r} not allowed; expected one of "
            f"{sorted(KNOWLEDGE_ITEM_CONTENT_TYPES)}"
        )

    title = payload.get("title")
    if not isinstance(title, str) or len(title.strip()) < 3:
        errors.append("title must be a non-empty string of at least 3 chars")

    content = payload.get("content")
    if not isinstance(content, str) or not content.strip():
        errors.append("content must be a non-empty string")

    if "tags" in payload and payload["tags"] is not None and not isinstance(payload["tags"], list):
        errors.append(f"tags must be a list, got {type(payload['tags']).__name__}")

    if "agent_visibility" in payload and payload["agent_visibility"] is not None and not isinstance(payload["agent_visibility"], list):
        errors.append(
            f"agent_visibility must be a list, got {type(payload['agent_visibility']).__name__}"
        )

    if "metadata" in payload and payload["metadata"] is not None and not isinstance(payload["metadata"], dict):
        errors.append(f"metadata must be a dict, got {type(payload['metadata']).__name__}")

    status = payload.get("status")
    if status is not None and status not in KNOWLEDGE_ITEM_STATUSES:
        errors.append(
            f"status {status!r} not allowed; expected one of {sorted(KNOWLEDGE_ITEM_STATUSES)}"
        )

    return errors


def insert_knowledge_item(data: dict) -> dict:
    data.setdefault("updated_at", __import__("datetime").datetime.utcnow().isoformat())
    cleaned = {k: v for k, v in data.items() if k not in _KNOWLEDGE_ITEMS_MISSING_COLUMNS}
    last_exc: Exception | None = None
    for _ in range(4):
        try:
            return _insert_one(get_client().table("knowledge_items").insert(cleaned))
        except Exception as exc:
            missing = _missing_column_from_error(exc)
            if not missing or missing not in cleaned:
                last_exc = exc
                break
            _KNOWLEDGE_ITEMS_MISSING_COLUMNS.add(missing)
            cleaned = {k: v for k, v in cleaned.items() if k != missing}
            last_exc = exc
    if last_exc:
        raise last_exc
    return {}


def _mark_persona_faqs_pending_regeneration(persona_id: Optional[str], *, changed_source_id: Optional[str], now_iso: str) -> None:
    if not persona_id:
        return
    client = get_client()
    stale_meta = {
        "faq_status": "pending_regeneration",
        "needs_update": True,
        "stale_reason": "related_context_changed",
        "changed_source_id": changed_source_id,
        "stale_marked_at": now_iso,
    }
    try:
        faq_items = (
            client.table("knowledge_items")
            .select("id,metadata")
            .eq("persona_id", persona_id)
            .eq("content_type", "faq")
            .execute()
            .data or []
        )
        for item in faq_items:
            metadata = {**(item.get("metadata") or {}), **stale_meta}
            client.table("knowledge_items").update({
                "status": "pending_regeneration",
                "curation_status": "stale",
                "metadata": metadata,
                "updated_at": now_iso,
            }).eq("id", item["id"]).execute()
    except Exception:
        pass
    try:
        faq_nodes = (
            client.table("knowledge_nodes")
            .select("id,metadata")
            .eq("persona_id", persona_id)
            .eq("node_type", "faq")
            .execute()
            .data or []
        )
        for node in faq_nodes:
            metadata = {**(node.get("metadata") or {}), **stale_meta}
            client.table("knowledge_nodes").update({
                "status": "pending_regeneration",
                "metadata": metadata,
                "updated_at": now_iso,
            }).eq("id", node["id"]).execute()
    except Exception:
        pass


def update_knowledge_item(item_id: str, data: dict) -> None:
    now_iso = __import__("datetime").datetime.utcnow().isoformat()
    data["updated_at"] = now_iso
    try:
        client = get_client()
        client.table("knowledge_items").update(data).eq("id", item_id).execute()
        row = (client.table("knowledge_items").select("id,content_type,persona_id").eq("id", item_id).limit(1).execute().data or [None])[0]
        if row and row.get("content_type") in {"brand", "briefing", "audience", "product", "offer", "copy", "rule"}:
            _mark_persona_faqs_pending_regeneration(row.get("persona_id"), changed_source_id=item_id, now_iso=now_iso)
    except Exception as exc:
        try:
            from services import sre_logger
            sre_logger.error("supabase_client", f"update_knowledge_item failed id={item_id}: {exc}", exc)
        except Exception:
            pass
        raise


def withdraw_faq_from_embedded(item_id: str) -> dict:
    """Send an approved/embedded FAQ back to draft and pull it out of Embedded.

    Used when an already-approved FAQ document is edited: the published content
    is now stale, so the FAQ must be re-approved and re-published before it can
    live in the Golden Dataset again. The FAQ node goes to
    `pending_validation`, its FAQ->Embedded edges are soft-deactivated, and its
    RAG entries/chunks are deleted so stale approved content cannot be retrieved.
    Best-effort: any failure is swallowed so it never blocks the edit itself.
    """
    from datetime import datetime, timezone

    now_iso = datetime.now(timezone.utc).isoformat()
    summary: dict = {
        "item_id": item_id,
        "node_ids": [],
        "deactivated_embedded_edges": [],
        "deleted_rag_entry_ids": [],
        "stale_snapshot_ids": [],
    }
    if _KG_TABLES_MISSING or not item_id:
        return summary
    client = get_client()
    try:
        nodes = (
            client.table("knowledge_nodes")
            .select("id,metadata")
            .eq("source_table", "knowledge_items")
            .eq("source_id", item_id)
            .eq("node_type", "faq")
            .execute()
            .data
            or []
        )
    except Exception:
        nodes = []
    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            continue
        summary["node_ids"].append(node_id)
        current_meta = node.get("metadata") or {}
        rag_entry_ids: list[str] = []
        for key in ("knowledge_rag_entry_id", "rag_entry_id"):
            if current_meta.get(key):
                rag_entry_ids.append(str(current_meta.get(key)))
        for key in ("knowledge_rag_entry_ids", "rag_entry_ids"):
            for entry_id in current_meta.get(key) or []:
                if entry_id:
                    rag_entry_ids.append(str(entry_id))
        snapshots = {}
        try:
            snapshots = list_approved_snapshots_for_nodes([node_id])
        except Exception:
            snapshots = {}
        snapshot = snapshots.get(node_id) or {}
        if snapshot.get("rag_entry_id"):
            rag_entry_ids.append(str(snapshot.get("rag_entry_id")))
        snapshot_meta = snapshot.get("metadata") or {}
        for entry_id in snapshot_meta.get("rag_entry_ids") or []:
            if entry_id:
                rag_entry_ids.append(str(entry_id))
        try:
            rows = (
                client.table("knowledge_rag_entries")
                .select("id")
                .eq("source_node_id", node_id)
                .eq("content_type", "faq")
                .execute()
                .data
                or []
            )
            for row in rows:
                if row.get("id"):
                    rag_entry_ids.append(str(row["id"]))
        except Exception:
            pass
        rag_entry_ids = list(dict.fromkeys(rag_entry_ids))
        for rag_entry_id in rag_entry_ids:
            try:
                if delete_knowledge_rag_entry(rag_entry_id):
                    summary["deleted_rag_entry_ids"].append(rag_entry_id)
            except Exception:
                pass
        try:
            metadata = {
                **current_meta,
                "needs_republish": True,
                "n8n_ready": False,
                "snapshot_status": "withdrawn",
                "withdrawn_from_embedded_at": now_iso,
            }
            for key in (
                "knowledge_rag_entry_id",
                "knowledge_rag_entry_ids",
                "knowledge_rag_chunk_ids",
                "rag_entry_id",
                "rag_entry_ids",
                "rag_chunk_ids",
            ):
                metadata.pop(key, None)
            client.table("knowledge_nodes").update(
                {"status": "pending_validation", "metadata": metadata, "updated_at": now_iso}
            ).eq("id", node_id).execute()
        except Exception:
            pass
        try:
            edges = list_edges_for_nodes([node_id]) or []
        except Exception:
            edges = []
        for edge in edges:
            if edge.get("source_node_id") != node_id:
                continue
            target = get_knowledge_node(edge.get("target_node_id"))
            if target and str(target.get("node_type") or "").lower() == "embedded":
                try:
                    if delete_knowledge_edge(edge.get("id")):
                        summary["deactivated_embedded_edges"].append(edge.get("id"))
                except Exception:
                    pass
        if snapshot.get("id"):
            try:
                updated_meta = {
                    **snapshot_meta,
                    "n8n_ready": False,
                    "snapshot_status": "withdrawn",
                    "withdrawn_from_embedded_at": now_iso,
                    "withdrawn_rag_entry_ids": rag_entry_ids,
                }
                for key in ("rag_entry_id", "rag_entry_ids", "rag_chunk_ids"):
                    updated_meta.pop(key, None)
                update_approved_knowledge_snapshot(
                    snapshot["id"],
                    {
                        "status": "stale",
                        "rag_entry_id": None,
                        "metadata": updated_meta,
                        "updated_at": now_iso,
                    },
                )
                summary["stale_snapshot_ids"].append(snapshot["id"])
            except Exception:
                pass
    try:
        item = get_knowledge_item(item_id) or {}
        item_meta = {
            **(item.get("metadata") or {}),
            "needs_republish": True,
            "n8n_ready": False,
            "snapshot_status": "withdrawn",
            "withdrawn_from_embedded_at": now_iso,
        }
        for key in (
            "knowledge_rag_entry_id",
            "knowledge_rag_entry_ids",
            "knowledge_rag_chunk_ids",
            "rag_entry_id",
            "rag_entry_ids",
            "rag_chunk_ids",
        ):
            item_meta.pop(key, None)
        client.table("knowledge_items").update({"metadata": item_meta, "updated_at": now_iso}).eq("id", item_id).execute()
    except Exception:
        pass
    return summary


def delete_knowledge_item(item_id: str) -> bool:
    result = _execute_with_retry(get_client().table("knowledge_items").delete().eq("id", item_id))
    return bool(result.data)


def insert_knowledge_intake_message(data: dict) -> dict:
    return _insert_one(get_client().table("knowledge_intake_messages").insert(data))


def update_knowledge_intake_message(intake_id: str, data: dict) -> None:
    _execute_with_retry(
        get_client().table("knowledge_intake_messages").update(data).eq("id", intake_id)
    )


def upsert_knowledge_rag_entry(data: dict) -> dict:
    from datetime import datetime, timezone

    payload = dict(data)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    result = _execute_with_retry(
        get_client()
        .table("knowledge_rag_entries")
        .upsert(payload, on_conflict="persona_id,canonical_key")
    )
    return (result.data or [{}])[0]


def replace_knowledge_rag_chunks(rag_entry_id: str, persona_id: str, chunks: list[dict]) -> list[dict]:
    client = get_client()
    _execute_with_retry(client.table("knowledge_rag_chunks").delete().eq("rag_entry_id", rag_entry_id))
    if not chunks:
        return []
    payload = []
    for idx, chunk in enumerate(chunks):
        row = dict(chunk)
        row.setdefault("chunk_index", idx)
        row["rag_entry_id"] = rag_entry_id
        row["persona_id"] = persona_id
        payload.append(row)
    result = _execute_with_retry(client.table("knowledge_rag_chunks").insert(payload))
    return result.data or []


def delete_knowledge_rag_entry(rag_entry_id: str) -> bool:
    client = get_client()
    _execute_with_retry(client.table("knowledge_rag_chunks").delete().eq("rag_entry_id", rag_entry_id))
    result = _execute_with_retry(client.table("knowledge_rag_entries").delete().eq("id", rag_entry_id))
    return bool(result.data)


def upsert_knowledge_rag_link(data: dict) -> dict:
    result = _execute_with_retry(
        get_client()
        .table("knowledge_rag_links")
        .upsert(data, on_conflict="source_entry_id,target_entry_id,relation_type")
    )
    return (result.data or [{}])[0]


_APPROVED_SNAPSHOTS_MISSING = False


def _snapshots_unavailable(exc: Exception) -> bool:
    text = str(exc)
    return (
        "approved_knowledge_snapshots" in text
        and ("PGRST205" in text or "schema cache" in text or "Could not find the table" in text or "does not exist" in text)
    )


def upsert_approved_knowledge_snapshot(data: dict) -> dict:
    """Idempotently upsert an approved tree snapshot by persona/canonical_key."""
    global _APPROVED_SNAPSHOTS_MISSING
    if _APPROVED_SNAPSHOTS_MISSING:
        raise RuntimeError("approved_knowledge_snapshots table is not available")
    from datetime import datetime, timezone

    payload = dict(data)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        result = _execute_with_retry(
            get_client()
            .table("approved_knowledge_snapshots")
            .upsert(payload, on_conflict="persona_id,canonical_key")
        )
        return (result.data or [{}])[0]
    except Exception as exc:
        if _snapshots_unavailable(exc):
            _APPROVED_SNAPSHOTS_MISSING = True
        raise


def update_approved_knowledge_snapshot(snapshot_id: str, data: dict) -> Optional[dict]:
    global _APPROVED_SNAPSHOTS_MISSING
    if _APPROVED_SNAPSHOTS_MISSING or not snapshot_id or not data:
        return None
    try:
        result = _execute_with_retry(
            get_client()
            .table("approved_knowledge_snapshots")
            .update(data)
            .eq("id", snapshot_id)
        )
        return (result.data or [data])[0]
    except Exception as exc:
        if _snapshots_unavailable(exc):
            _APPROVED_SNAPSHOTS_MISSING = True
            return None
        raise


def list_approved_snapshots_for_nodes(node_ids: list[str]) -> dict[str, dict]:
    """Return the latest approved/active snapshot for each source_node_id."""
    global _APPROVED_SNAPSHOTS_MISSING
    if _APPROVED_SNAPSHOTS_MISSING or not node_ids:
        return {}
    try:
        rows = _q(
            get_client()
            .table("approved_knowledge_snapshots")
            .select("id,source_node_id,rag_entry_id,status,content_type,canonical_key,metadata,updated_at")
            .in_("source_node_id", node_ids)
            .order("updated_at", desc=True)
            .limit(max(1, len(node_ids) * 3))
        )
    except Exception as exc:
        if _snapshots_unavailable(exc):
            _APPROVED_SNAPSHOTS_MISSING = True
            return {}
        raise
    out: dict[str, dict] = {}
    for row in rows:
        sid = str(row.get("source_node_id") or "")
        if sid and sid not in out:
            out[sid] = row
    return out


def count_knowledge_rag_chunks_by_entry_ids(entry_ids: list[str]) -> dict[str, int]:
    if not entry_ids:
        return {}
    rows = _q(
        get_client()
        .table("knowledge_rag_chunks")
        .select("id,rag_entry_id")
        .in_("rag_entry_id", entry_ids)
        .limit(5000)
    )
    counts: dict[str, int] = {}
    for row in rows:
        rid = str(row.get("rag_entry_id") or "")
        if rid:
            counts[rid] = counts.get(rid, 0) + 1
    return counts


def search_active_rag_chunks(
    *,
    persona_id: str,
    query: str = "",
    limit: int = 12,
    branch_anchor_node_id: str | None = None,
    allowed_node_ids: list[str] | None = None,
    active_path_node_ids: list[str] | None = None,
    unresolved_fields: list[str] | None = None,
    graph_version: int | None = None,
) -> list[dict]:
    """Return persona-scoped Golden Dataset chunks before any legacy fallback.

    Text scoring stays deterministic and local for now; embeddings can replace
    the ranking without changing the caller contract.
    """
    if not persona_id:
        return []
    client = get_client()
    entries = _q(
        client.table("knowledge_rag_entries")
        .select("id,title,content_type,slug,status,metadata,canonical_key")
        .eq("persona_id", persona_id)
        .in_("status", ["active", "approved", "validated", "embedded"])
        .limit(2000)
    )
    entry_by_id = {str(row.get("id")): row for row in entries if row.get("id")}
    if not entry_by_id:
        return []
    chunks = _q(
        client.table("knowledge_rag_chunks")
        .select("id,rag_entry_id,persona_id,chunk_index,chunk_text,chunk_summary,metadata")
        .eq("persona_id", persona_id)
        .in_("rag_entry_id", list(entry_by_id))
        .limit(5000)
    )
    # WhatsApp questions commonly contain accents, punctuation and hyphenated
    # names (for example "Coca-Cola" / "preço").  Normalize both sides so a
    # lexical RAG fallback remains useful before embeddings are available.
    def _rag_terms(value: str) -> set[str]:
        normalized = unicodedata.normalize("NFKD", value or "")
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()
        return {part for part in re.findall(r"[a-z0-9]+", normalized) if len(part) > 1}

    terms = _rag_terms(query)
    allowed = set(allowed_node_ids or [])
    active_path = set(active_path_node_ids or [])
    pending = set(unresolved_fields or [])
    ranked: list[tuple[float, int, dict]] = []
    for chunk in chunks:
        entry = entry_by_id.get(str(chunk.get("rag_entry_id")))
        if not entry:
            continue
        metadata = {
            **(entry.get("metadata") or {}),
            **(chunk.get("metadata") or {}),
        }
        node_id = str(
            metadata.get("graph_json_node_id")
            or metadata.get("canonical_node_id")
            or metadata.get("source_node_id")
            or ""
        )
        chunk_branch = str(metadata.get("branch_anchor_node_id") or "")
        try:
            chunk_version = int(metadata.get("graph_version") or 0)
        except (TypeError, ValueError):
            chunk_version = 0
        if graph_version is not None and chunk_version not in {0, int(graph_version)}:
            continue
        if branch_anchor_node_id and not (
            chunk_branch == branch_anchor_node_id or node_id in allowed
        ):
            continue
        haystack = " ".join(
            [
                str(entry.get("title") or ""),
                str(entry.get("slug") or ""),
                str(chunk.get("chunk_text") or ""),
                str(chunk.get("chunk_summary") or ""),
            ]
        )
        normalized_haystack = " ".join(_rag_terms(haystack))
        semantic_score = sum(1 for term in terms if term in normalized_haystack)
        if terms and semantic_score == 0 and not branch_anchor_node_id:
            continue
        path_ids = set(metadata.get("path_node_ids") or [])
        path_overlap = len(path_ids & active_path) / max(1, len(active_path))
        field_key = str(metadata.get("field_key") or "")
        field_relevance = 1.0 if field_key and field_key in pending else 0.0
        try:
            priority = float(metadata.get("priority") or 0.0)
        except (TypeError, ValueError):
            priority = 0.0
        graph_proximity = 1.0 if chunk_branch == branch_anchor_node_id else path_overlap
        score = (
            float(semantic_score)
            + 1.25 * graph_proximity
            + 0.75 * path_overlap
            + 0.8 * field_relevance
            + 0.2 * priority
        )
        ranked.append(
            (
                score,
                -int(chunk.get("chunk_index") or 0),
                {**chunk, "entry": entry, "source": "golden_dataset"},
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in ranked[:limit]]


def get_active_graph_publication(persona_id: str) -> Optional[dict]:
    if not persona_id:
        return None
    return _one(
        get_client().table("graph_publications").select("*")
        .eq("persona_id", persona_id).eq("status", "active")
        .maybe_single()
    )


def get_graph_publication_by_id(publication_id: str) -> Optional[dict]:
    """Load the immutable turn-pinned publication, regardless of active status."""
    if not publication_id:
        return None
    return _one(
        get_client().table("graph_publications").select("*")
        .eq("id", publication_id).maybe_single()
    )


def get_graph_turn_context_batch_v3(
    *, persona_id: str, lead_ref: int, message_limit: int = 8,
) -> dict:
    """Fetch publication, ledger, facts, branches and recent messages once."""
    result = get_client().rpc(
        "graph_turn_context_batch_v3",
        {
            "p_persona_id": persona_id,
            "p_lead_ref": lead_ref,
            "p_message_limit": max(1, min(int(message_limit), 20)),
        },
    ).execute()
    value = getattr(result, "data", None)
    if isinstance(value, list):
        value = value[0] if value else {}
    return value if isinstance(value, dict) else {}


def get_graph_turn_context_batch_v4(
    *, persona_id: str, lead_ref: int, message_limit: int = 8,
) -> dict:
    """Fetch v4 shared memory, falling back during a rolling deployment."""
    try:
        result = get_client().rpc(
            "graph_turn_context_batch_v4",
            {
                "p_persona_id": persona_id,
                "p_lead_ref": lead_ref,
                "p_message_limit": max(1, min(int(message_limit), 20)),
            },
        ).execute()
        value = getattr(result, "data", None)
        if isinstance(value, list):
            value = value[0] if value else {}
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    return get_graph_turn_context_batch_v3(
        persona_id=persona_id, lead_ref=lead_ref, message_limit=message_limit,
    )


def get_graph_branch_package_v3(
    *, publication_id: str, branch_node_id: str,
    chunk_ids: list[str] | None = None, node_ids: list[str] | None = None,
    limit: int = 12,
) -> dict:
    result = get_client().rpc(
        "graph_branch_package_v3",
        {
            "p_publication_id": publication_id,
            "p_branch_node_id": branch_node_id,
            "p_chunk_ids": chunk_ids or [],
            "p_node_ids": node_ids or [],
            "p_limit": max(1, min(int(limit), 12)),
        },
    ).execute()
    value = getattr(result, "data", None)
    if isinstance(value, list):
        value = value[0] if value else {}
    return value if isinstance(value, dict) else {}


def get_graph_branch_contract(publication_id: str, branch_node_id: str) -> Optional[dict]:
    if not publication_id or not branch_node_id:
        return None
    return _one(
        get_client().table("graph_branch_contracts").select("*")
        .eq("publication_id", publication_id).eq("branch_node_id", branch_node_id)
        .maybe_single()
    )


def get_conversation_ledger(persona_id: str, lead_ref: int) -> Optional[dict]:
    if not persona_id or not lead_ref:
        return None
    journey = _one(
        get_client().table("conversation_journeys").select("id")
        .eq("persona_id", persona_id).eq("lead_ref", lead_ref)
        .eq("is_current", True).maybe_single()
    )
    if not journey:
        return None
    ledger = _one(
        get_client().table("conversation_ledgers").select("*")
        .eq("persona_id", persona_id).eq("lead_ref", lead_ref)
        .eq("journey_id", journey["id"]).maybe_single()
    )
    if not ledger:
        return None
    facts = _q(
        get_client().table("conversation_facts").select("*")
        .eq("ledger_id", ledger["id"]).eq("is_current", True).limit(1000)
    )
    ledger["facts"] = {
        str(row["field_key"]): {
            **row,
            "value": row.get("value_json"),
            "fact_id": row.get("id"),
        }
        for row in facts
    }
    grouped: dict[str, list[dict]] = {}
    for row in facts:
        grouped.setdefault(str(row["field_key"]), []).append({
            **row,
            "value": row.get("value_json"),
            "fact_id": row.get("id"),
        })
    ledger["facts_by_key"] = grouped
    ledger["active_branch_node_ids"] = get_active_ledger_branches(str(ledger["id"]))
    return ledger


def get_lead_carry_over_facts(
    persona_id: str, lead_ref: int, field_keys: list[str],
) -> list[dict]:
    """O valor `known` mais recente de cada campo, em qualquer jornada do lead.

    Diferente de get_journey_ledger_facts (uma jornada so): jornadas/pedidos
    ja registrados sao a fonte de verdade da identidade do cliente, entao a
    busca cobre o historico inteiro do lead, nao so a jornada anterior
    imediata -- um ponteiro de uma jornada so perdia o fato assim que uma
    segunda jornada fechasse antes do campo ser respondido de novo
    (confirmado ao vivo 2026-08-18).
    """
    if not persona_id or not lead_ref or not field_keys:
        return []
    try:
        result = get_client().rpc(
            "conversation_carry_over_facts_by_lead_v1",
            {"p_persona_id": persona_id, "p_lead_ref": lead_ref, "p_field_keys": field_keys},
        ).execute()
        return result.data or []
    except Exception:
        # Rolling-deploy compatibility until migration 129 is applied.
        pass
    ledgers = _q(
        get_client().table("conversation_ledgers").select("id")
        .eq("persona_id", persona_id).eq("lead_ref", lead_ref).limit(200)
    )
    ledger_ids = [row["id"] for row in ledgers if row.get("id")]
    if not ledger_ids:
        return []
    rows = _q(
        get_client().table("conversation_facts").select("*")
        .in_("ledger_id", ledger_ids).in_("field_key", field_keys)
        .eq("is_current", True).eq("status", "known")
        .order("created_at", desc=True).limit(1000)
    )
    latest_by_key: dict[str, dict] = {}
    for row in rows:
        key = str(row.get("field_key") or "")
        if key and key not in latest_by_key:
            latest_by_key[key] = row
    return list(latest_by_key.values())


def get_journey_ledger_facts(journey_id: str) -> list[dict]:
    """Fatos correntes do ledger de uma jornada especifica.

    Usado para atravessar o fim de um pedido: a jornada seguinte nasce com
    ledger proprio e zero fatos, entao a identidade do cliente (nome e o que o
    grafo marcar como `carry_over`) precisa vir do ledger anterior. O servico e
    os campos do galho ficam para tras de proposito.
    """
    if not journey_id:
        return []
    try:
        result = get_client().rpc(
            "conversation_carry_over_facts_v1", {"p_journey_id": journey_id},
        ).execute()
        return result.data or []
    except Exception:
        # Rolling-deploy compatibility until migration 127 is applied.
        pass
    ledger = _one(
        get_client().table("conversation_ledgers").select("id")
        .eq("journey_id", journey_id).maybe_single()
    )
    if not ledger:
        return []
    return _q(
        get_client().table("conversation_facts").select("*")
        .eq("ledger_id", ledger["id"]).eq("is_current", True)
        .eq("status", "known").limit(1000)
    )


def get_current_conversation_journey(persona_id: str, lead_ref: int) -> Optional[dict]:
    if not persona_id or not lead_ref:
        return None
    return _one(
        get_client().table("conversation_journeys").select("*")
        .eq("persona_id", persona_id).eq("lead_ref", lead_ref)
        .eq("is_current", True).maybe_single()
    )


def get_journeys_by_lead_refs(
    persona_id: str, lead_refs: list[int], *, chunk_size: int = 100,
) -> list[dict]:
    """Every journey of many leads, in bounded batches.

    Same reasoning as ``get_leads_by_refs``: the conversation list paints every
    row at once, so one request per lead would be an N+1, and an unbounded
    ``in`` list would blow the PostgREST URL up.

    Deliberately *not* filtered by ``is_current``: a closed order must keep
    describing itself on screen, and the permanent "lead converted" fact lives
    in whichever journey converted first.
    """
    refs = sorted({int(value) for value in lead_refs if value is not None})
    if not persona_id or not refs:
        return []
    size = max(1, min(chunk_size, 100))
    rows: list[dict] = []
    for index in range(0, len(refs), size):
        rows.extend(_q(
            get_client().table("conversation_journeys")
            .select("lead_ref,state,is_current,converted_at,closed_at,sequence,metadata")
            .eq("persona_id", persona_id)
            .in_("lead_ref", refs[index:index + size])
        ))
    return rows


def get_latest_conversation_journey(persona_id: str, lead_ref: int) -> Optional[dict]:
    if not persona_id or not lead_ref:
        return None
    rows = _q(
        get_client().table("conversation_journeys").select("*")
        .eq("persona_id", persona_id).eq("lead_ref", lead_ref)
        .order("sequence", desc=True).limit(1)
    )
    return rows[0] if rows else None


def get_conversation_facts_by_key(ledger_id: str) -> dict:
    """Every current fact for a ledger, grouped by field_key.

    Unlike get_conversation_ledger()'s flat `facts` dict (one fact per
    field_key -- correct only when at most one branch is ever active),
    supporting more than one simultaneously-active branch (branch_action
    "add") means more than one current fact can share a field_key with a
    different owner_node_id (e.g. two branches each with their own current
    "servico"). Grouping by key instead of collapsing to one preserves all
    of them, for graph_proof_checker_v3.aggregate_missing_fields().
    """
    if not ledger_id:
        return {}
    rows = _q(
        get_client().table("conversation_facts").select("*")
        .eq("ledger_id", ledger_id).eq("is_current", True).limit(1000)
    )
    grouped: dict = {}
    for row in rows:
        grouped.setdefault(str(row["field_key"]), []).append({
            **row, "value": row.get("value_json"), "fact_id": row.get("id"),
        })
    return grouped


def get_active_ledger_branches(ledger_id: str) -> list:
    if not ledger_id:
        return []
    try:
        rows = _q(
            get_client().table("conversation_ledger_branches").select("branch_anchor_node_id")
            .eq("ledger_id", ledger_id).eq("state", "active").limit(100)
        )
    except Exception:
        # migration 105 may not be applied yet on a deployment where this
        # code shipped ahead of it -- degrade to "no additional active
        # branches" (today's single-service behavior) instead of breaking
        # every single conversation turn over an optional, additive table.
        return []
    return [str(row["branch_anchor_node_id"]) for row in rows]


def get_ledger_branch_states(ledger_id: str) -> dict:
    """branch_anchor_node_id -> state, every branch ever tracked for this ledger.

    Unlike get_active_ledger_branches (state='active' only -- what every turn
    uses to seed active_branch_node_ids), this also surfaces 'completed'
    branches, so the runtime can tell "still needs its own confirmation cycle"
    apart from "already confirmed, just still open for support" once a
    journey has more than one active branch (product or service -- nothing
    here is offering-specific). Same migration-not-applied-yet tolerance as
    get_active_ledger_branches.
    """
    if not ledger_id:
        return {}
    try:
        rows = _q(
            get_client().table("conversation_ledger_branches")
            .select("branch_anchor_node_id,state")
            .eq("ledger_id", ledger_id).limit(200)
        )
    except Exception:
        return {}
    return {
        str(row.get("branch_anchor_node_id") or ""): str(row.get("state") or "")
        for row in rows or []
    }


def mark_ledger_branches_completed(ledger_id: str, branch_anchor_node_ids: list) -> None:
    """One-time grandfather write: see build_context's use in
    graph_agent_runtime_v3.py. A journey already in post_qualification_support
    the first time this code runs against it has no 'completed' row yet
    (migration 128 shipped after it was handed off); without this, every one
    of its already-confirmed branches would look indistinguishable from a
    brand new, never-confirmed one. Best-effort and idempotent (upsert), same
    tolerance as the other optional writes to this table.
    """
    if not ledger_id or not branch_anchor_node_ids:
        return
    try:
        get_client().table("conversation_ledger_branches").upsert(
            [
                {
                    "ledger_id": ledger_id, "branch_anchor_node_id": anchor,
                    "state": "completed", "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                for anchor in dict.fromkeys(branch_anchor_node_ids)
            ],
            on_conflict="ledger_id,branch_anchor_node_id",
        ).execute()
    except Exception:
        pass


def add_ledger_branch(ledger_id: str, branch_anchor_node_id: str) -> None:
    """Record an additional simultaneously-active branch for a ledger.

    Idempotent (upsert on the (ledger_id, branch_anchor_node_id) unique
    constraint from migration 105) so a retried/duplicated commit never
    raises -- it just leaves the row as 'active', same as the first call.
    Same migration-not-applied-yet tolerance as get_active_ledger_branches:
    this is an audit-trail write, never required for the current turn's own
    correctness, so a failure here must not fail the whole commit.
    """
    if not ledger_id or not branch_anchor_node_id:
        return
    try:
        get_client().table("conversation_ledger_branches").upsert(
            {"ledger_id": ledger_id, "branch_anchor_node_id": branch_anchor_node_id, "state": "active"},
            on_conflict="ledger_id,branch_anchor_node_id",
        ).execute()
    except Exception:
        pass


def get_wa_validator_session(session_id: str) -> Optional[dict]:
    """Read a WA Validator session's data blob, or None if it doesn't exist."""
    if not session_id:
        return None
    row = _one(
        get_client().table("wa_validator_sessions").select("data")
        .eq("id", session_id).maybe_single()
    )
    return row.get("data") if row else None


def upsert_wa_validator_session(
    session_id: str, data: dict, *, persona_slug: Optional[str] = None, flow_id: Optional[str] = None
) -> dict:
    """Write a WA Validator session's full data blob (create or replace)."""
    from datetime import datetime, timezone
    payload = {
        "id": session_id, "data": data,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if persona_slug is not None:
        payload["persona_slug"] = persona_slug
    if flow_id is not None:
        payload["flow_id"] = flow_id
    rows = (
        get_client().table("wa_validator_sessions")
        .upsert(payload, on_conflict="id").execute().data or []
    )
    return rows[0]["data"] if rows else data


def list_wa_validator_sessions(
    limit: int = 100,
    *,
    persona_slug: str | None = None,
    since_hours: int | None = None,
) -> list[dict]:
    """Return a bounded, database-filtered WA Validator session window."""
    from datetime import datetime, timedelta, timezone

    query = get_client().table("wa_validator_sessions").select("data")
    if persona_slug:
        query = query.eq("persona_slug", persona_slug)
    if since_hours is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, int(since_hours)))
        query = query.gte("created_at", cutoff.isoformat())
    rows = _q(query.order("created_at", desc=True).limit(max(1, min(int(limit), 100))))
    return [row["data"] for row in rows if row.get("data")]


def claim_wa_validator_session(session_id: str) -> dict:
    """Atomically transition one validator session from ready to running."""
    result = get_client().rpc(
        "claim_wa_validator_session", {"p_session_id": session_id}
    ).execute()
    value = getattr(result, "data", None)
    if isinstance(value, list):
        value = value[0] if value else {}
    return value if isinstance(value, dict) else {}


def enqueue_wa_validator_session(session_id: str, mode: str) -> dict:
    """Atomically enqueue a ready validator session for the worker process."""
    result = get_client().rpc(
        "enqueue_wa_validator_session",
        {"p_session_id": session_id, "p_mode": mode},
    ).execute()
    value = getattr(result, "data", None)
    if isinstance(value, list):
        value = value[0] if value else {}
    return value if isinstance(value, dict) else {}


def claim_next_wa_validator_session(worker_id: str) -> dict:
    """Lease the oldest queued validator session, if one is available."""
    result = get_client().rpc(
        "claim_next_wa_validator_session", {"p_worker_id": worker_id}
    ).execute()
    value = getattr(result, "data", None)
    if isinstance(value, list):
        value = value[0] if value else {}
    return value if isinstance(value, dict) else {}


def cleanup_wa_validator_artifacts(*, hours: int = 12, dry_run: bool = True) -> dict:
    """Inventory or transactionally remove expired canonical validator data."""
    result = get_client().rpc(
        "cleanup_wa_validator_artifacts",
        {"p_before": f"{max(1, int(hours))} hours", "p_dry_run": bool(dry_run)},
    ).execute()
    value = getattr(result, "data", None)
    if isinstance(value, list):
        value = value[0] if value else {}
    return value if isinstance(value, dict) else {}


def search_graph_rag_v3(
    *,
    persona_id: str,
    publication_id: str,
    branch_node_id: str,
    query: str,
    query_embedding: list[float] | None,
    active_path_node_ids: list[str] | None = None,
    missing_fields: list[str] | None = None,
    limit: int = 24,
) -> list[dict]:
    result = get_client().rpc(
        "graph_hybrid_search_v3",
        {
            "p_persona_id": persona_id,
            "p_publication_id": publication_id,
            "p_branch_node_id": branch_node_id,
            "p_query": query,
            "p_query_embedding": query_embedding,
            "p_active_path_node_ids": active_path_node_ids or [],
            "p_missing_fields": missing_fields or [],
            "p_limit": max(1, min(int(limit), 200)),
        },
    ).execute()
    return result.data or []


def search_graph_faq_v3(
    *,
    persona_id: str,
    publication_id: str,
    branch_node_id: str,
    query: str,
    query_embedding: list[float] | None,
    eligible_faq_node_ids: list[str],
    limit: int = 64,
) -> list[dict]:
    """Search only compiler-approved FAQ chunks inside one branch closure."""
    if not eligible_faq_node_ids:
        return []
    result = get_client().rpc(
        "graph_faq_search_v3",
        {
            "p_persona_id": persona_id,
            "p_publication_id": publication_id,
            "p_branch_node_id": branch_node_id,
            "p_query": query,
            "p_query_embedding": query_embedding,
            "p_eligible_faq_node_ids": eligible_faq_node_ids,
            "p_limit": max(1, min(int(limit), 200)),
        },
    ).execute()
    return result.data or []


def rank_graph_branches_v3(
    *,
    persona_id: str,
    publication_id: str,
    query: str,
    query_embedding: list[float] | None,
    limit: int = 8,
) -> list[dict]:
    result = get_client().rpc(
        "graph_branch_rank_v3",
        {
            "p_persona_id": persona_id,
            "p_publication_id": publication_id,
            "p_query": query,
            "p_query_embedding": query_embedding,
            "p_limit": max(1, min(int(limit), 32)),
        },
    ).execute()
    return result.data or []


def rank_graph_services_v3(
    *,
    persona_id: str,
    publication_id: str,
    query_embedding: list[float] | None,
    limit: int = 8,
) -> list[dict]:
    """Rank only chunks owned by published service anchors."""
    if query_embedding is None:
        return []
    result = get_client().rpc(
        "graph_service_rank_v3",
        {
            "p_persona_id": persona_id,
            "p_publication_id": publication_id,
            "p_query_embedding": query_embedding,
            "p_limit": max(1, min(int(limit), 32)),
        },
    ).execute()
    return result.data or []


def get_graph_rag_repair_chunks(
    *, publication_id: str, branch_node_id: str, requirements: list[dict]
) -> list[dict]:
    node_ids = [str(item.get("id")) for item in requirements if item.get("kind") == "node"]
    chunk_ids = [str(item.get("id")) for item in requirements if item.get("kind") == "chunk"]
    query = (
        get_client().table("knowledge_rag_chunks")
        .select("id,rag_entry_id,source_graph_node_id,branch_anchor_node_id,chunk_text,chunk_summary,chunk_kind,chunk_checksum,path_checksum,metadata")
        .eq("publication_id", publication_id).eq("branch_anchor_node_id", branch_node_id)
    )
    if chunk_ids and node_ids:
        # PostgREST has no portable OR builder across all supported client
        # versions; fetch two narrowly bounded sets and deduplicate below.
        chunks = _q(query.in_("id", chunk_ids).limit(200))
        nodes = _q(
            get_client().table("knowledge_rag_chunks")
            .select("id,rag_entry_id,source_graph_node_id,branch_anchor_node_id,chunk_text,chunk_summary,chunk_kind,chunk_checksum,path_checksum,metadata")
            .eq("publication_id", publication_id).eq("branch_anchor_node_id", branch_node_id)
            .in_("source_graph_node_id", node_ids).limit(200)
        )
        return list({str(row["id"]): row for row in [*chunks, *nodes]}.values())
    if chunk_ids:
        return _q(query.in_("id", chunk_ids).limit(200))
    if node_ids:
        return _q(query.in_("source_graph_node_id", node_ids).limit(200))
    return []


def commit_graph_turn_v3(**payload: Any) -> dict:
    result = get_client().rpc("commit_graph_turn_v3", payload).execute()
    value = getattr(result, "data", None)
    if isinstance(value, list):
        value = value[0] if value else {}
    return value if isinstance(value, dict) else {}


def commit_graph_turn_and_outbox_v3(
    *, turn: dict, outbound_buffer: dict | None,
    outbound_message: dict | None, result_payload: dict,
) -> dict:
    result = get_client().rpc(
        "commit_graph_turn_and_outbox_v3",
        {
            "p_turn": turn,
            "p_outbound_buffer": outbound_buffer,
            "p_outbound_message": outbound_message,
            "p_result": result_payload,
        },
    ).execute()
    value = getattr(result, "data", None)
    if isinstance(value, list):
        value = value[0] if value else {}
    return value if isinstance(value, dict) else {}


def commit_graph_turn_and_outbox_v4(
    *, turn: dict, outbound_buffer: dict | None,
    outbound_message: dict | None, result_payload: dict,
) -> dict:
    """Atomic journey-aware commit with temporary v3 rolling fallback."""
    payload = {
        "p_turn": turn,
        "p_outbound_buffer": outbound_buffer,
        "p_outbound_message": outbound_message,
        "p_result": result_payload,
    }
    try:
        result = get_client().rpc("commit_graph_turn_and_outbox_v4", payload).execute()
        value = getattr(result, "data", None)
        if isinstance(value, list):
            value = value[0] if value else {}
        if isinstance(value, dict):
            return value
    except Exception:
        # A v3 fallback is safe only for a real journey.  Falling back for
        # journey_action=none would recreate the exact phantom-journey bug.
        if str(turn.get("journey_action") or "continue") == "none":
            raise
    return commit_graph_turn_and_outbox_v3(
        turn=turn, outbound_buffer=outbound_buffer,
        outbound_message=outbound_message, result_payload=result_payload,
    )


def audit_conversation_turn_v3(inbound_buffer_id: str) -> dict:
    result = get_client().rpc(
        "audit_conversation_turn_v3", {"p_inbound_id": inbound_buffer_id}
    ).execute()
    value = getattr(result, "data", None)
    if isinstance(value, list):
        value = value[0] if value else {}
    return value if isinstance(value, dict) else {}


def get_conversation_turn_proof(canonical_inbound_id: str) -> Optional[dict]:
    if not canonical_inbound_id:
        return None
    return _one(
        get_client().table("conversation_turn_proofs").select("*")
        .eq("canonical_inbound_id", canonical_inbound_id).maybe_single()
    )


def reset_conversation_ledger_branch_v3(*, persona_id: str, lead_ref: int) -> dict:
    result = get_client().rpc(
        "reset_conversation_ledger_branch_v3",
        {"p_persona_id": persona_id, "p_lead_ref": lead_ref},
    ).execute()
    value = getattr(result, "data", None)
    if isinstance(value, list):
        value = value[0] if value else {}
    return value if isinstance(value, dict) else {}


def record_purchase_completed(**payload: Any) -> dict:
    result = get_client().rpc("record_purchase_completed_v1", payload).execute()
    value = getattr(result, "data", None)
    if isinstance(value, list):
        value = value[0] if value else {}
    return value if isinstance(value, dict) else {}


def set_conversation_journey_state(**payload: Any) -> dict:
    """Estado-alvo do pedido, escolhido por um humano.

    Complementa ``record_conversation_journey_event``: aquele e append-only e
    idempotente (integracao e agente), este calcula o delta ate o alvo e sabe
    voltar atras.
    """
    result = get_client().rpc(
        "set_conversation_journey_state_v1", payload,
    ).execute()
    value = getattr(result, "data", None)
    if isinstance(value, list):
        value = value[0] if value else {}
    return value if isinstance(value, dict) else {}


def record_conversation_journey_event(**payload: Any) -> dict:
    result = get_client().rpc(
        "record_conversation_journey_event_v1", payload,
    ).execute()
    value = getattr(result, "data", None)
    if isinstance(value, list):
        value = value[0] if value else {}
    return value if isinstance(value, dict) else {}


def transition_sales_conversion_status(**payload: Any) -> dict:
    result = get_client().rpc(
        "transition_sales_conversion_status_v1", payload
    ).execute()
    value = getattr(result, "data", None)
    if isinstance(value, list):
        value = value[0] if value else {}
    return value if isinstance(value, dict) else {}


def mark_conversation_journey_qualification(**payload: Any) -> dict:
    result = get_client().rpc(
        "mark_conversation_journey_qualification_v1", payload
    ).execute()
    value = getattr(result, "data", None)
    if isinstance(value, list):
        value = value[0] if value else {}
    return value if isinstance(value, dict) else {}


def list_inactivity_recovery_candidates(*, enabled_from: str, cutoff: str, limit: int = 100) -> list[dict]:
    return _q(
        get_client().table("lead_buffer").select("id,persona_id,lead_ref,created_at")
        .eq("direction", "inbound").eq("status", "buffered")
        .gte("created_at", enabled_from).lte("created_at", cutoff)
        .is_("payload->conversation_commit", "null")
        .order("created_at").limit(limit)
    )


def claim_inactivity_recovery_candidate(*, inbound_id: str) -> dict:
    result = get_client().rpc(
        "claim_inactivity_recovery_candidate_v1", {"p_inbound_id": inbound_id}
    ).execute()
    value = getattr(result, "data", None)
    if isinstance(value, list):
        value = value[0] if value else {}
    return value if isinstance(value, dict) else {}


def find_knowledge_rag_entry_by_slug(
    *,
    persona_id: str,
    content_type: str,
    slug: str,
) -> Optional[dict]:
    return _one(
        get_client()
        .table("knowledge_rag_entries")
        .select("*")
        .eq("persona_id", persona_id)
        .eq("content_type", content_type)
        .eq("slug", slug)
        .maybe_single()
    )


def get_knowledge_item_counts(persona_id: Optional[str] = None) -> dict:
    q = get_client().table("knowledge_items").select("status,content_type")
    if persona_id:
        q = q.eq("persona_id", persona_id)
    rows = _q(q)
    by_status: dict = {}
    by_type: dict = {}
    for r in rows:
        s = r["status"]
        t = r["content_type"]
        by_status[s] = by_status.get(s, 0) + 1
        by_type[t] = by_type.get(t, 0) + 1
    return {"by_status": by_status, "by_type": by_type, "total": len(rows)}


# â”€â”€ Sync Runs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def insert_sync_run(data: dict) -> dict:
    result = _execute_with_retry(get_client().table("sync_runs").insert(data))
    return (result.data or [{}])[0]


def update_sync_run(run_id: str, data: dict) -> None:
    _execute_with_retry(get_client().table("sync_runs").update(data).eq("id", run_id))


def get_sync_runs(limit: int = 20) -> list:
    return _q(
        get_client().table("sync_runs")
        .select("*")
        .order("started_at", desc=True)
        .limit(limit)
    )


def insert_sync_log(data: dict) -> None:
    _execute_with_retry(get_client().table("sync_logs").insert(data))


def get_sync_logs(run_id: str, limit: int = 200) -> list:
    return _q(
        get_client().table("sync_logs")
        .select("*")
        .eq("run_id", run_id)
        .order("created_at", desc=False)
        .limit(limit)
    )


# â”€â”€ Workflow Bindings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_workflow_bindings(persona_id: Optional[str] = None) -> list:
    # Try with relationship join first; fall back to plain select if PGRST205
    try:
        q = get_client().table("workflow_bindings").select("*,personas(name,slug)")
        if persona_id:
            q = q.eq("persona_id", persona_id)
        rows = _q(q)
        if rows is not None:  # _q already handles None, but check for PGRST205 path
            return rows
    except Exception:
        pass
    # Fallback: plain select without relationship join
    q = get_client().table("workflow_bindings").select("*")
    if persona_id:
        q = q.eq("persona_id", persona_id)
    return _q(q)


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


def activate_whatsapp_binding(
    *,
    persona_id: str,
    binding_id: str,
    provider: str,
    source: str = "admin.settings",
) -> dict:
    """Atomically activate a persona transport and rebind its current leads."""
    result = _execute_with_retry(
        get_client().rpc(
            "activate_persona_whatsapp_binding",
            {
                "p_persona_id": persona_id,
                "p_binding_id": binding_id,
                "p_provider": provider,
                "p_source": source,
            },
        )
    )
    payload = result.data
    if isinstance(payload, list):
        return payload[0] if payload else {}
    return payload or {}


def assert_client_portal_schema() -> None:
    """Fail API startup when the Compose migration contract is incomplete."""
    checks = (
        ("app_users", "id,account_type,must_change_password"),
        ("workflow_bindings", "id,channel,provider,provider_instance_key,provider_secret_ciphertext,connection_status"),
        ("leads", "id,channel_binding_id,external_contact_id"),
        ("messages", "id,channel_binding_id"),
        ("lead_buffer", "id,channel_binding_id"),
    )
    for table, fields in checks:
        try:
            get_client().table(table).select(fields).limit(1).execute()
        except Exception as exc:
            raise RuntimeError(
                f"Client portal schema is incomplete at {table}; apply migrations 061/062 with Docker Compose."
            ) from exc


def get_default_whatsapp_phone_number_id(persona_id: Optional[str] = None) -> Optional[str]:
    if not persona_id:
        return None
    for binding in get_workflow_bindings(persona_id):
        value = binding.get("whatsapp_phone_number_id")
        if value and binding.get("active", True):
            return value
    return None


def get_active_workflow_binding_by_phone_number_id(phone_number_id: str) -> Optional[dict]:
    """Resolve routing exclusively by the business number, never by lead phone."""
    if not phone_number_id:
        return None
    return _one(
        get_client().table("workflow_bindings").select("*")
        .eq("whatsapp_phone_number_id", phone_number_id).eq("active", True).maybe_single()
    )


def get_workflow_bindings_by_phone_number_id(phone_number_id: str) -> list:
    """Return every binding that claims a business phone number.

    Normal routing only needs the active binding. Provisioning is stricter:
    a number already recorded for another persona must be reassigned through
    the dedicated, audited flow rather than silently becoming a second
    persona's draft binding.
    """
    if not phone_number_id:
        return []
    return _q(
        get_client().table("workflow_bindings").select("*")
        .eq("whatsapp_phone_number_id", phone_number_id)
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


def enqueue_whatsapp_message(data: dict) -> Optional[dict]:
    """Legacy buffer-only insert.

    New inbound and outbound traffic must use enqueue_whatsapp_envelope so the
    message projection and the idempotency lock commit in one transaction.
    """
    try:
        result = get_client().table("lead_buffer").insert(data).execute()
        return (result.data or [None])[0]
    except Exception as exc:
        # Postgres unique violation is expected when Meta retries a webhook.
        text = str(exc).lower()
        if "duplicate" in text or "unique" in text or "23505" in text:
            return None
        raise


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


def claim_conversation_commit(
    *,
    inbound_buffer_id: str,
    binding_id: str,
    lead_ref: int,
    correlation_id: str,
) -> dict:
    result = get_client().rpc(
        "claim_conversation_commit",
        {
            "p_inbound_buffer_id": inbound_buffer_id,
            "p_binding_id": binding_id,
            "p_lead_ref": lead_ref,
            "p_correlation_id": correlation_id,
        },
    ).execute()
    payload = getattr(result, "data", None)
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    return payload if isinstance(payload, dict) else {}


def complete_conversation_commit(
    *,
    inbound_buffer_id: str,
    binding_id: str,
    lead_ref: int,
    correlation_id: str,
    result_payload: dict,
) -> dict:
    result = get_client().rpc(
        "complete_conversation_commit",
        {
            "p_inbound_buffer_id": inbound_buffer_id,
            "p_binding_id": binding_id,
            "p_lead_ref": lead_ref,
            "p_correlation_id": correlation_id,
            "p_result": result_payload,
        },
    ).execute()
    payload = getattr(result, "data", None)
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    return payload if isinstance(payload, dict) else {}


def finalize_proven_conversation_turn(
    *,
    inbound_buffer_id: str,
    binding_id: str,
    lead_ref: int,
    correlation_id: str,
    outbound_id: str,
    result_payload: dict,
) -> dict:
    result = get_client().rpc(
        "finalize_proven_conversation_turn",
        {
            "p_inbound_buffer_id": inbound_buffer_id,
            "p_binding_id": binding_id,
            "p_lead_ref": lead_ref,
            "p_correlation_id": correlation_id,
            "p_outbound_id": outbound_id,
            "p_result": result_payload,
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


def recover_uncommitted_graph_inbound(buffer_id: str, reason: str) -> dict:
    result = get_client().rpc(
        "recover_uncommitted_graph_inbound",
        {"p_buffer_id": buffer_id, "p_reason": reason},
    ).execute()
    payload = getattr(result, "data", None)
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    return payload if isinstance(payload, dict) else {}


def recover_unsent_committed_outbound(buffer_id: str, reason: str) -> dict:
    result = get_client().rpc(
        "recover_unsent_committed_outbound",
        {"p_buffer_id": buffer_id, "p_reason": reason},
    ).execute()
    payload = getattr(result, "data", None)
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    return payload if isinstance(payload, dict) else {}


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


def set_lead_buffer_waiting_human(lead_ref: int) -> None:
    _execute_with_retry(
        get_client().table("lead_buffer").update({"status": "waiting_human"})
        .eq("lead_ref", lead_ref).in_("status", ["received", "buffered", "processing", "pending_send", "retry"])
    )


def handoff_whatsapp_lead(lead_ref: int, *, level: str = "full") -> None:
    """Atomically set handoff_level and (for level='full') quarantine queued work."""
    _execute_with_retry(
        get_client().rpc(
            "handoff_whatsapp_lead", {"p_lead_ref": lead_ref, "p_level": level}
        )
    )


def requeue_waiting_human_whatsapp_buffer(lead_ref: int) -> int:
    """Move a lead's stuck inbound messages back into the claimable queue.

    Resuming AI on a lead does nothing on its own to messages that piled up
    in `waiting_human` while it was paused — this is the retroactive
    reprocessing step that was missing.
    """
    result = get_client().rpc(
        "requeue_waiting_human_whatsapp_buffer", {"p_lead_ref": lead_ref}
    ).execute()
    return int(getattr(result, "data", 0) or 0)


def handoff_whatsapp_lead_state(
    lead_ref: int,
    *,
    metadata: dict,
    stage: str,
    level: str = "full",
) -> None:
    """Atomically persist the cart/stage and set handoff_level.

    level='partial' keeps the lead's inbound lead_buffer rows claimable (the
    AI keeps running); only level='full' quarantines them as waiting_human.
    """
    _execute_with_retry(
        get_client().rpc(
            "handoff_whatsapp_lead_state",
            {
                "p_lead_ref": lead_ref,
                "p_metadata": metadata,
                "p_stage": stage,
                "p_level": level,
            },
        )
    )


def upsert_workflow_binding(data: dict) -> dict:
    result = get_client().table("workflow_bindings").upsert(
        data, on_conflict="workflow_name,persona_id"
    ).execute()
    return result.data[0] if result.data else {}


def update_workflow_binding_metadata(
    binding_id: str,
    metadata: dict,
) -> dict:
    result = (
        get_client()
        .table("workflow_bindings")
        .update({"metadata": metadata})
        .eq("id", binding_id)
        .execute()
    )
    return result.data[0] if result.data else {}


# â”€â”€ Brand Profiles â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_brand_profile(persona_id: str) -> Optional[dict]:
    return _one(
        get_client().table("brand_profiles")
        .select("*")
        .eq("persona_id", persona_id)
        .maybe_single()
    )


def upsert_brand_profile(data: dict) -> dict:
    result = get_client().table("brand_profiles").upsert(
        data, on_conflict="persona_id"
    ).execute()
    return result.data[0] if result.data else {}


# â”€â”€ Campaigns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_campaigns(persona_id: Optional[str] = None) -> list:
    q = get_client().table("campaigns").select("*").order("created_at", desc=True)
    if persona_id:
        q = q.eq("persona_id", persona_id)
    return _q(q)


# -- Campaign delivery one -------------------------------------------------------

def create_lead_import_batch(data: dict) -> dict:
    return _insert_one(get_client().table("lead_import_batches").insert(data))


def update_lead_import_batch(batch_id: str, data: dict) -> Optional[dict]:
    result = _execute_with_retry(
        get_client().table("lead_import_batches").update(data).eq("id", batch_id)
    )
    return (result.data or [None])[0] if result else None


def insert_lead_import_row(data: dict) -> dict:
    return _insert_one(get_client().table("lead_import_rows").insert(data))


def list_lead_import_batches(persona_id: Optional[str] = None, limit: int = 100) -> list[dict]:
    query = get_client().table("lead_import_batches").select("*").order("created_at", desc=True).limit(limit)
    if persona_id:
        query = query.eq("persona_id", persona_id)
    return _q(query)


def get_lead_import_batch(batch_id: str) -> Optional[dict]:
    return _one(
        get_client().table("lead_import_batches").select("*").eq("id", batch_id).maybe_single()
    )


def list_lead_import_rows(batch_id: str, limit: int = 5000) -> list[dict]:
    return _q(
        get_client().table("lead_import_rows").select("*")
        .eq("batch_id", batch_id).order("row_index").limit(limit)
    )


def replace_lead_semantic_group(
    *,
    lead_id: int,
    persona_id: str,
    audience_id: str,
    created_by_user_id: Optional[str],
    idempotency_key: str,
    reason: str,
) -> dict:
    audience = get_audience(audience_id)
    if not audience or audience.get("persona_id") != persona_id:
        raise ValueError("audience does not belong to persona")
    if str((audience.get("metadata") or {}).get("kind") or "semantic_group") != "semantic_group":
        raise ValueError("audience is not a semantic group")
    result = _execute_with_retry(get_client().rpc("replace_lead_semantic_group_v1", {
        "p_lead_id": lead_id,
        "p_target_persona_id": persona_id,
        "p_audience_id": audience_id,
        "p_created_by_user_id": created_by_user_id,
        "p_idempotency_key": idempotency_key,
        "p_reason": reason,
    }))
    return {"audience": audience, "result": (result.data if result else None)}


def insert_contact_consent(data: dict, audit_payload: Optional[dict] = None) -> dict:
    event = {
        "channel": data.get("channel"),
        "purpose": data.get("purpose"),
        "campaign_id": data.get("campaign_id"),
        "import_batch_id": data.get("import_batch_id"),
        "request_message_id": data.get("request_message_id"),
        "response_message_id": data.get("response_message_id"),
        "idempotency_key": data.get("idempotency_key"),
        **(audit_payload or {}),
    }
    result = _execute_with_retry(get_client().rpc("record_contact_consent_v1", {
        "p_consent": data,
        "p_event": event,
    }))
    payload = result.data if result else None
    return payload if isinstance(payload, dict) else ((payload or [{}])[0])


def get_latest_contact_consent(
    *, lead_id: int, persona_id: str, channel: str, purpose: str,
) -> Optional[dict]:
    rows = list_contact_consents(
        lead_id=lead_id, persona_id=persona_id, channel=channel, purpose=purpose,
    )
    return rows[0] if rows else None


def list_contact_consents(
    *, lead_id: int, persona_id: str, channel: Optional[str] = None, purpose: Optional[str] = None,
) -> list[dict]:
    query = (
        get_client().table("contact_consents").select("*")
        .eq("lead_id", lead_id).eq("persona_id", persona_id)
        .order("effective_at", desc=True).order("created_at", desc=True).limit(500)
    )
    if channel:
        query = query.eq("channel", channel)
    if purpose:
        query = query.eq("purpose", purpose)
    return _q(query)


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


def get_events(
    limit: int = 50,
    event_type: Optional[str] = None,
    persona_id: Optional[str] = None,
    entity_id: Optional[str] = None,
    level: Optional[str] = None,
) -> list:
    q = (
        get_client().table("system_events")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
    )
    if event_type:
        q = q.eq("event_type", event_type)
    if persona_id:
        q = q.eq("persona_id", persona_id)
    if entity_id:
        q = q.eq("entity_id", entity_id)
    if level:
        q = q.eq("level", level)
    return _q(q)


def list_system_events(
    entity_type: Optional[str] = None,
    event_types: Optional[list[str]] = None,
    persona_id: Optional[str] = None,
    entity_id: Optional[str] = None,
    payload_equals: Optional[dict[str, str]] = None,
    since: Optional[str] = None,
    search: Optional[str] = None,
    level: Optional[str] = None,
    limit: int = 100,
) -> list:
    """Audit-trail query over system_events.

    Filters compose with AND. `event_types` is an OR list (uses .in_()).
    `search` does ILIKE over payload::text — slow without an index, OK for
    audit volumes (capped by limit). `since` expects ISO8601.
    """
    q = (
        get_client().table("system_events")
        .select("*")
        .order("created_at", desc=True)
        .limit(max(1, min(int(limit or 100), 500)))
    )
    if entity_type:
        q = q.eq("entity_type", entity_type)
    if event_types:
        q = q.in_("event_type", list(event_types))
    if persona_id:
        q = q.eq("persona_id", persona_id)
    if entity_id:
        q = q.eq("entity_id", entity_id)
    for key, value in (payload_equals or {}).items():
        if key not in {"persona_slug", "brand_slug", "version", "operation_id"}:
            raise ValueError(f"Unsupported system_events payload filter: {key}")
        q = q.eq(f"payload->>{key}", str(value))
    if since:
        q = q.gte("created_at", since)
    if search:
        q = q.ilike("payload", f"%{search}%")
    if level:
        q = q.eq("level", level)
    return _q(q)


def _rpc_json(name: str, params: dict) -> dict:
    """Execute an internal RPC and normalize PostgREST's object/list shapes."""
    result = _execute_with_retry(get_client().rpc(name, params))
    payload = getattr(result, "data", None)
    if isinstance(payload, list):
        payload = payload[0] if payload else None
    if not isinstance(payload, dict):
        raise RuntimeError(f"{name} returned an invalid result")
    return payload


def commit_graph_version_v2(
    *,
    persona_slug: str,
    brand_slug: Optional[str],
    expected_version: int,
    idempotency_key: str,
    reason: str,
    graph_json: dict,
    content_checksum: str,
    source: str,
    authored_by: Optional[str],
) -> dict:
    return _rpc_json(
        "commit_graph_version_v2",
        {
            "p_persona_slug": persona_slug,
            "p_brand_slug": brand_slug,
            "p_expected_version": expected_version,
            "p_idempotency_key": idempotency_key,
            "p_reason": reason,
            "p_graph_json": graph_json,
            "p_content_checksum": content_checksum,
            "p_source": source,
            "p_authored_by": authored_by,
        },
    )


def activate_graph_projection_v2(
    *,
    persona_slug: str,
    brand_slug: Optional[str],
    graph_version: int,
    graph_checksum: str,
    operation_id: str,
    projections: dict,
    source: str,
) -> dict:
    return _rpc_json(
        "activate_graph_projection_v2",
        {
            "p_persona_slug": persona_slug,
            "p_brand_slug": brand_slug,
            "p_graph_version": graph_version,
            "p_graph_checksum": graph_checksum,
            "p_operation_id": operation_id,
            "p_projections": projections,
            "p_source": source,
        },
    )


def record_graph_projection_event_v2(
    *,
    persona_slug: str,
    projection: dict,
    source: str,
) -> dict:
    return _rpc_json(
        "record_graph_projection_event_v2",
        {
            "p_persona_slug": persona_slug,
            "p_projection": projection,
            "p_source": source,
        },
    )


# â”€â”€ Pipeline Status â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_pipeline_statuses() -> list:
    return _q(
        get_client().table("pipeline_status")
        .select("*")
        .order("service")
    )


def update_pipeline_status(service: str, data: dict) -> None:
    get_client().table("pipeline_status").update(data).eq("service", service).execute()


def get_pipeline_metrics(persona_id: Optional[str] = None) -> dict:
    from datetime import datetime, timedelta
    today = (datetime.utcnow() - timedelta(hours=24)).isoformat()
    client = get_client()

    attention_q = (
        client.table("knowledge_items")
        .select("status")
        .in_("status", ["pending", "needs_persona", "needs_category"])
    )
    approved_q = (
        client.table("knowledge_items")
        .select("id")
        .eq("status", "approved")
        .gte("updated_at", today)
    )
    kb_q = (
        client.table("kb_entries")
        .select("id")
        .eq("status", "ATIVO")
    )
    asset_q = (
        client.table("knowledge_items")
        .select("id")
        .eq("content_type", "asset")
        .in_("status", ["pending", "needs_persona"])
    )
    if persona_id:
        attention_q = attention_q.eq("persona_id", persona_id)
        approved_q = approved_q.eq("persona_id", persona_id)
        kb_q = kb_q.eq("persona_id", persona_id)
        asset_q = asset_q.eq("persona_id", persona_id)

    attention_rows = _q(attention_q)
    approved_rows = _q(approved_q)
    kb_rows = _q(kb_q)
    asset_rows = _q(asset_q)
    error_rows = [
        row for row in get_error_logs(limit=500)
        if str(row.get("created_at") or row.get("ts") or "") >= today
    ]

    return {
        "pending_attention": len(attention_rows),
        "approved_today": len(approved_rows),
        "kb_entries": len(kb_rows),
        "assets_pending": len(asset_rows),
        "errors_24h": len(error_rows),
    }


# â”€â”€ Storage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def upload_to_storage(bucket: str, path: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """Upload bytes to Supabase Storage; returns the public URL."""
    client = get_client()
    client.storage.from_(bucket).upload(path, data, {"content-type": content_type, "upsert": "true"})
    return client.storage.from_(bucket).get_public_url(path)


def download_from_storage(bucket: str, path: str) -> bytes:
    """Download bytes from Supabase Storage using the backend service client."""
    return get_client().storage.from_(bucket).download(path)


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


def update_asset_graph_refs(
    asset_id: str,
    *,
    knowledge_node_id: Optional[str] = None,
    gallery_edge_id: Optional[str] = None,
    parent_node_id: Optional[str] = None,
    parent_edge_id: Optional[str] = None,
) -> dict:
    """Persist graph evidence on an asset row.

    Some live databases may not have the top-level graph columns yet. Keep the
    metadata copy as the durable fallback so the app can still enforce the graph
    contract before the migration is applied.
    """
    if not asset_id:
        return {}
    current = get_asset(asset_id) or {}
    metadata = dict(current.get("metadata") or {})
    graph_meta = dict(metadata.get("graph") or {})
    if knowledge_node_id:
        metadata["knowledge_node_id"] = knowledge_node_id
        graph_meta["knowledge_node_id"] = knowledge_node_id
    if gallery_edge_id:
        metadata["gallery_edge_id"] = gallery_edge_id
        graph_meta["gallery_edge_id"] = gallery_edge_id
    if parent_node_id:
        metadata["parent_node_id"] = parent_node_id
        graph_meta["parent_node_id"] = parent_node_id
    if parent_edge_id:
        metadata["parent_edge_id"] = parent_edge_id
        graph_meta["parent_edge_id"] = parent_edge_id
    if graph_meta:
        metadata["graph"] = graph_meta

    patch = {"metadata": metadata}
    if knowledge_node_id:
        patch["knowledge_node_id"] = knowledge_node_id
    if gallery_edge_id:
        patch["gallery_edge_id"] = gallery_edge_id

    try:
        return update_asset(asset_id, patch)
    except Exception as exc:
        text = str(exc).lower()
        missing_graph_cols = (
            "knowledge_node_id" in text
            or "gallery_edge_id" in text
            or "schema cache" in text
            or "could not find" in text
        )
        if not missing_graph_cols:
            raise
        return update_asset(asset_id, {"metadata": metadata})


def get_asset(asset_id: str) -> Optional[dict]:
    if not asset_id:
        return None
    result = (
        get_client().table("assets")
        .select("*")
        .eq("id", asset_id)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    return rows[0] if rows else None


def list_assets(
    persona_id: Optional[str] = None,
    upload_context: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    include_markdown: bool = False,
) -> list:
    q = get_client().table("assets").select("*")
    if persona_id:
        q = q.eq("persona_id", persona_id)
    if upload_context:
        q = q.eq("upload_context", upload_context)
    if status:
        q = q.eq("status", status)
    if not include_markdown:
        # Markdown companions are inline content of an image asset, never
        # standalone cards in the /assets grid.
        q = q.neq("type", "markdown")
    result = q.order("created_at", desc=True).range(offset, offset + limit - 1).execute()
    return result.data or []


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


def list_lead_media_assets(persona_id: str, lead_id: int, limit: int = 50) -> list:
    """Files exchanged with one lead, newest first (media rail in Mensagens)."""
    if not persona_id or not lead_id:
        return []
    result = (
        get_client().table("assets")
        .select("*")
        .eq("persona_id", persona_id)
        .eq("lead_id", lead_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


def list_asset_readings(asset_id: str) -> list:
    if not asset_id:
        return []
    result = (
        get_client().table("asset_readings")
        .select("*")
        .eq("asset_id", asset_id)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []


def is_node_type(node_id: Optional[str], node_type: str) -> bool:
    """Convenience guard used by /assets and edge wrappers."""
    if not node_id:
        return False
    raw = node_id[3:] if node_id.startswith("gn:") else node_id
    result = (
        get_client().table("knowledge_nodes")
        .select("node_type")
        .eq("id", raw)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    return bool(rows and (rows[0] or {}).get("node_type") == node_type)


# â”€â”€ KB Intake tracking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def insert_kb_intake(data: dict) -> dict:
    result = get_client().table("kb_intake").insert(data).execute()
    return (result.data or [{}])[0]


# -- Sofia plan session persistence -------------------------------------------------

def get_sofia_plan_session(session_id: str) -> Optional[dict]:
    key = str(session_id or "").strip().lower()
    if not key:
        return None
    return _one(
        get_client()
        .table("sofia_plan_sessions")
        .select("*")
        .eq("session_id", key)
        .maybe_single()
    )


def upsert_sofia_plan_session(
    *,
    session_id: str,
    persona_slug: str,
    plan_json: dict,
    active_persona_slug: Optional[str] = None,
    last_referenced_node: Optional[dict] = None,
    recent_turns: Optional[list[dict]] = None,
) -> Optional[dict]:
    key = str(session_id or "").strip().lower()
    if not key:
        return None
    row = {
        "session_id": key,
        "persona_slug": str(persona_slug or "").strip().lower(),
        "active_persona_slug": str(active_persona_slug or persona_slug or "").strip().lower() or None,
        "plan_json": plan_json or {},
        "last_referenced_node": last_referenced_node or {},
        "recent_turns": recent_turns or [],
    }
    try:
        result = _execute_with_retry(
            get_client()
            .table("sofia_plan_sessions")
            .upsert(row, on_conflict="session_id")
        )
    except Exception:
        return None
    return (result.data or [None])[0] if result else None


def get_kb_intake_session(session_id: str) -> Optional[dict]:
    row = _one(
        get_client().table("agent_sessions").select("selected_context")
        .eq("id", session_id).eq("coordinator_key", "sofia_kb_intake")
        .maybe_single()
    )
    state = (row or {}).get("selected_context") or {}
    value = state.get("kb_intake_session")
    return value if isinstance(value, dict) else None


def upsert_kb_intake_session(session: dict) -> Optional[dict]:
    session_id = str(session.get("id") or "").strip()
    if not session_id:
        return None
    persona_slug = str((session.get("classification") or {}).get("persona_slug") or "")
    persona = get_persona(persona_slug) if persona_slug else None
    row = {
        "id": session_id,
        "coordinator_key": "sofia_kb_intake",
        "persona_id": (persona or {}).get("id"),
        "selected_context": {"kb_intake_session": session},
        "status": "completed" if session.get("complete") else "active",
    }
    result = _execute_with_retry(
        get_client().table("agent_sessions").upsert(row, on_conflict="id")
    )
    return (result.data or [None])[0] if result else None


def list_kb_intake_sessions(limit: int = 500) -> list[dict]:
    rows = _q(
        get_client().table("agent_sessions").select("id,selected_context,updated_at")
        .eq("coordinator_key", "sofia_kb_intake")
        .order("updated_at", desc=True).limit(max(1, min(limit, 500)))
    )
    return [
        {
            **dict(((row.get("selected_context") or {}).get("kb_intake_session") or {})),
            "_updated_at": row.get("updated_at"),
        }
        for row in rows
        if isinstance((row.get("selected_context") or {}).get("kb_intake_session"), dict)
    ]

# â”€â”€ Knowledge Items: multi-status query â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_knowledge_items_multi(
    statuses: list[str],
    persona_id: Optional[str] = None,
    content_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list:
    q = (
        get_client().table("knowledge_items")
        .select("*")
        .in_("status", statuses)
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
    )
    if persona_id:
        q = q.eq("persona_id", persona_id)
    if content_type:
        q = q.eq("content_type", content_type)
    return _q(q)
