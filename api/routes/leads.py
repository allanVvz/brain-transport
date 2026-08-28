import csv
import hashlib
import io
import re
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field
from services import (
    agents_service,
    auth_service,
    event_emitter,
    journey_outcome,
    knowledge_graph,
    lead_qualification,
    supabase_client,
)

router = APIRouter(prefix="/leads", tags=["leads"])


class LeadAudienceChangeBody(BaseModel):
    target_persona_id: str
    target_audience_id: str | None = None
    target_audience_slug: str | None = None
    source_audience_id: str | None = None
    source_audience_slug: str | None = None


class LeadGroupBody(BaseModel):
    persona_id: str
    audience_id: str
    idempotency_key: str
    reason: str


class ConsentChangeBody(BaseModel):
    persona_id: str
    channel: str = "whatsapp"
    purpose: str
    source: str = "manual"
    request_message_id: int | None = None
    response_message_id: int | None = None
    campaign_id: str | None = None
    import_batch_id: str | None = None
    evidence: dict = Field(default_factory=dict)
    idempotency_key: str
    reason: str


def _database_actor_id(user: dict | None) -> str | None:
    value = (user or {}).get("id")
    try:
        return str(uuid.UUID(str(value))) if value else None
    except (TypeError, ValueError, AttributeError):
        return None


NAME_HEADERS = {"nome", "name", "nome completo", "cliente", "contato", "lead", "first name", "fn"}
LAST_NAME_HEADERS = {"sobrenome", "last name", "last_name", "ln"}
PHONE_HEADERS = {"telefone", "celular", "phone", "whatsapp", "numero", "lead_id", "mobile"}
EMAIL_HEADERS = {"email", "e-mail", "mail"}
CITY_HEADERS = {"cidade", "city", "ct"}
STATE_HEADERS = {"estado", "state", "st", "uf"}
ZIP_HEADERS = {"cep", "zip", "zipcode", "postal_code"}
COUNTRY_HEADERS = {"pais", "country"}


def _normalize_header(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _normalize_phone(value: str) -> str:
    digits = re.sub(r"\D", "", value or "")
    if digits.startswith("55") and len(digits) > 11:
        return digits
    return digits


def _display_name(first_name: str, last_name: str, fallback: str = "") -> str:
    name = " ".join(part for part in [first_name.strip(), last_name.strip()] if part).strip()
    return name or fallback.strip()


def _pick_field(row: dict, candidates: set[str]) -> str:
    normalized = {_normalize_header(k): v for k, v in row.items()}
    for key in candidates:
        if key in normalized and str(normalized[key] or "").strip():
            return str(normalized[key]).strip()
    for key, value in normalized.items():
        if any(candidate in key for candidate in candidates) and str(value or "").strip():
            return str(value).strip()
    return ""


def _decode_csv(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise HTTPException(400, "CSV must be UTF-8 text")


@router.get("")
def list_leads(
    request: Request,
    limit: int = Query(100, le=2000),
    offset: int = 0,
    persona_id: str | None = Query(None),
    persona_slug: str | None = Query(None),
    audience_id: str | None = Query(None),
    audience_slug: str | None = Query(None),
    validation_scope: str = Query("exclude", pattern="^(exclude|only|all)$"),
    hours: int | None = Query(None, ge=1, le=720),
):
    """Lista leads visiveis na persona ativa.

    Visibilidade vem de lead_audience_memberships, nao de leads.persona_id.
    Quando audience_slug=import e a persona ainda nao tem essa system
    audience, garantimos via helper idempotente para nao 404 a UI.
    """
    try:
        user = auth_service.current_user(request)
        if (user.get("account_type") or "internal") == "client":
            validation_scope = "exclude"
        resolved_persona_id = persona_id
        if not resolved_persona_id and persona_slug:
            persona = supabase_client.get_persona(persona_slug)
            resolved_persona_id = persona.get("id") if persona else None
        if resolved_persona_id:
            # Garantir que system audience `import` exista; idempotente.
            try:
                supabase_client.ensure_system_audiences_for_persona(resolved_persona_id)
            except Exception:
                pass
        if resolved_persona_id and (audience_id or audience_slug):
            auth_service.assert_persona_access(request, persona_id=resolved_persona_id, persona_slug=persona_slug)
            try:
                rows = supabase_client.get_leads_for_audience_scope(
                    persona_id=resolved_persona_id,
                    audience_id=audience_id,
                    audience_slug=audience_slug,
                    limit=limit,
                    offset=offset,
                    since_hours=hours if validation_scope == "only" else None,
                ) or []
                return journey_outcome.decorate_leads(
                    lead_qualification.filter_validation_scope(rows, validation_scope)
                )
            except Exception:
                return []
        if persona_id or persona_slug:
            auth_service.assert_persona_access(request, persona_id=persona_id, persona_slug=persona_slug)
        elif not auth_service.is_admin(auth_service.current_user(request)):
            rows = supabase_client.get_leads_for_persona_ids(
                auth_service.allowed_persona_ids(request),
                limit=limit,
                offset=offset,
                since_hours=hours if validation_scope == "only" else None,
            ) or []
            return journey_outcome.decorate_leads(
                lead_qualification.filter_validation_scope(rows, validation_scope)
            )
        rows = supabase_client.get_leads(
            persona_slug=persona_id or persona_slug,
            limit=limit,
            offset=offset,
            since_hours=hours if validation_scope == "only" else None,
        ) or []
        return lead_qualification.filter_validation_scope(rows, validation_scope)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Erro ao buscar leads: {exc}")


_EMPTY_HASHED_HINT = (
    "Nenhum lead valido. Verifique se o CSV tem colunas email/telefone em texto plano. "
    "Audiencias Meta exportadas com hash (sha256) nao sao suportadas; "
    "exporte o CSV sem hashing."
)


def _persist_terminal_batch_event(
    *,
    batch_id: str,
    persona_id: str,
    payload: dict,
    level: str = "info",
) -> dict | None:
    """Grava o evento batch terminal (completed/failed) numa unica chamada."""
    return supabase_client.insert_event(
        {
            "event_type": "lead_import_batch",
            "entity_type": "lead_import",
            "entity_id": batch_id,
            "persona_id": persona_id,
            "payload": payload,
        },
        level=level,
        source="leads.import",
    )


@router.post("/imports")
async def import_leads_csv_v1(
    request: Request,
    file: UploadFile = File(...),
    persona_id: str | None = Form(None),
    audience_id: str | None = Form(None),
    contact_basis: str = Form("imported_without_evidence"),
    consent_purpose: str = Form("ofertas_e_novidades"),
    contact_basis_statement: str | None = Form(None),
):
    """Persist an operational import cohort without projecting it into Graph."""
    if not persona_id:
        raise HTTPException(400, "Selecione uma persona antes de importar leads")
    auth_service.assert_persona_capability(request, "edit", persona_id=persona_id)
    current_user = auth_service.current_user(request)
    database_actor_id = _database_actor_id(current_user)
    filename = file.filename or "leads.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported for now")
    if contact_basis not in {
        "explicit_consent", "customer_initiated", "existing_customer",
        "imported_without_evidence", "legacy_unknown", "unknown",
    }:
        raise HTTPException(422, "Base de contato invalida.")
    if contact_basis == "explicit_consent" and not (contact_basis_statement or "").strip():
        raise HTTPException(422, "Descreva a evidencia do consentimento explicito.")
    if not consent_purpose.strip():
        raise HTTPException(422, "A finalidade do contato e obrigatoria.")

    audience = None
    if audience_id:
        audience = supabase_client.get_audience(audience_id)
        if not audience or audience.get("persona_id") != persona_id:
            raise HTTPException(404, "Grupo semantico nao encontrado nesta persona.")
        if str((audience.get("metadata") or {}).get("kind") or "semantic_group") != "semantic_group":
            raise HTTPException(422, "O destino do import deve ser um grupo semantico.")

    content = await file.read()
    text = _decode_csv(content)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except Exception:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        raise HTTPException(400, "CSV header row is required")

    batch_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    headers = list(reader.fieldnames or [])
    supabase_client.create_lead_import_batch({
        "id": batch_id,
        "persona_id": persona_id,
        "audience_id": audience_id,
        "filename": filename,
        "file_checksum": "sha256:" + hashlib.sha256(content).hexdigest(),
        "source": "csv_upload",
        "contact_basis": contact_basis,
        "status": "processing",
        "provenance": {
            "headers": headers,
            "contact_basis_statement": contact_basis_statement,
            "consent_purpose": consent_purpose,
        },
        "created_by_user_id": database_actor_id,
        "created_at": started_at,
        "updated_at": started_at,
    })

    stats = {
        "total": 0, "valid": 0, "with_phone": 0, "email_only": 0,
        "created": 0, "updated": 0, "invalid": 0,
    }
    preview: list[dict] = []
    try:
        for index, row in enumerate(reader, start=1):
            stats["total"] += 1
            raw_row = {str(k or ""): v for k, v in (row or {}).items()}
            first_name = _pick_field(raw_row, NAME_HEADERS)
            last_name = _pick_field(raw_row, LAST_NAME_HEADERS)
            name = _display_name(first_name, last_name, _pick_field(raw_row, {"nome completo", "name", "nome"}))
            phone = _normalize_phone(_pick_field(raw_row, PHONE_HEADERS))
            email = _pick_field(raw_row, EMAIL_HEADERS)
            city = _pick_field(raw_row, CITY_HEADERS)
            state = _pick_field(raw_row, STATE_HEADERS)
            zip_code = _pick_field(raw_row, ZIP_HEADERS)
            country = _pick_field(raw_row, COUNTRY_HEADERS)
            row_status, row_error, lead_ref = "created", "", None
            has_phone = bool(phone and len(phone) >= 8)
            has_email = bool(email)

            if not has_phone and not has_email:
                row_status, row_error = "invalid", "missing_email_and_phone"
                stats["invalid"] += 1
            elif not has_phone:
                row_status = "email_only"
                stats["valid"] += 1
                stats["email_only"] += 1
            else:
                stats["valid"] += 1
                stats["with_phone"] += 1
                existing_rows = (
                    supabase_client.get_client().table("leads").select("id")
                    .eq("persona_id", persona_id).eq("lead_id", phone).limit(1).execute().data or []
                )
                lead = supabase_client.ensure_lead_for_persona(
                    lead_id=phone,
                    persona_slug_or_id=persona_id,
                    nome=name or None,
                    stage="novo",
                    canal="bulk_import",
                    cidade=city or None,
                    cep=zip_code or None,
                )
                lead_ref = lead.get("id") if lead else None
                if not lead_ref:
                    raise RuntimeError(f"lead row {index} could not be persisted")
                if audience:
                    supabase_client.replace_lead_semantic_group(
                        lead_id=int(lead_ref),
                        persona_id=persona_id,
                        audience_id=audience["id"],
                        created_by_user_id=database_actor_id,
                        idempotency_key=f"import-group:{batch_id}:lead:{lead_ref}",
                        reason=f"Grupo selecionado no import {batch_id}",
                    )
                if existing_rows:
                    row_status = "updated"
                    stats["updated"] += 1
                else:
                    stats["created"] += 1

            parsed = {
                "nome": name, "fn": first_name, "ln": last_name,
                "lead_id": phone, "email": email, "cidade": city,
                "estado": state, "zip": zip_code, "country": country,
            }
            supabase_client.insert_lead_import_row({
                "batch_id": batch_id,
                "persona_id": persona_id,
                "row_index": index,
                "lead_id": lead_ref,
                "status": row_status,
                "error_code": row_error or None,
                "parsed_data": parsed,
            })
            latest_consent = (
                supabase_client.get_latest_contact_consent(
                    lead_id=int(lead_ref), persona_id=persona_id,
                    channel="whatsapp", purpose=consent_purpose,
                )
                if lead_ref else None
            )
            if lead_ref and (contact_basis == "explicit_consent" or not latest_consent):
                consent_status = "granted" if contact_basis == "explicit_consent" else "pending"
                supabase_client.insert_contact_consent({
                    "lead_id": lead_ref,
                    "persona_id": persona_id,
                    "channel": "whatsapp",
                    "purpose": consent_purpose,
                    "status": consent_status,
                    "source": "lead_import",
                    "import_batch_id": batch_id,
                    "effective_at": started_at,
                    "granted_at": started_at if consent_status == "granted" else None,
                    "evidence": {
                        "contact_basis": contact_basis,
                        "statement": contact_basis_statement,
                        "filename": filename,
                        "row_index": index,
                    },
                    "actor_user_id": database_actor_id,
                    "idempotency_key": f"import:{batch_id}:lead:{lead_ref}:purpose:{consent_purpose}",
                }, audit_payload={
                    "contact_basis": contact_basis,
                    "reason": "Consentimento classificado durante import operacional",
                    "actor_ref": current_user.get("id"),
                })
            row_payload = {
                "batch_id": batch_id, "row_index": index, "parsed": parsed,
                "status": row_status, "error": row_error, "lead_ref": lead_ref,
            }
            if row_status != "invalid" and len(preview) < 5:
                preview.append(row_payload)
    except Exception as exc:
        finished_at = datetime.now(timezone.utc).isoformat()
        error = f"{type(exc).__name__}: {exc}"
        supabase_client.update_lead_import_batch(batch_id, {
            "status": "failed", "error": error,
            "total_rows": stats["total"], "valid_rows": stats["valid"],
            "phone_rows": stats["with_phone"], "email_only_rows": stats["email_only"],
            "created_leads": stats["created"], "updated_leads": stats["updated"],
            "invalid_rows": stats["invalid"], "completed_at": finished_at, "updated_at": finished_at,
        })
        _persist_terminal_batch_event(
            batch_id=batch_id,
            persona_id=persona_id,
            payload={
                "batch_id": batch_id, "filename": filename, "status": "failed",
                "started_at": started_at, "finished_at": finished_at,
                "stats": stats, "persona_id": persona_id, "error": error,
            },
            level="error",
        )
        raise HTTPException(500, f"Falha ao importar leads: {exc}")

    finished_at = datetime.now(timezone.utc).isoformat()
    terminal_status = "completed" if stats["valid"] > 0 else "failed"
    error = None
    if stats["total"] == 0:
        error = "CSV sem linhas alem do cabecalho."
    elif stats["valid"] == 0:
        error = _EMPTY_HASHED_HINT
    if error:
        terminal_status = "failed"
    supabase_client.update_lead_import_batch(batch_id, {
        "status": terminal_status, "error": error,
        "total_rows": stats["total"], "valid_rows": stats["valid"],
        "phone_rows": stats["with_phone"], "email_only_rows": stats["email_only"],
        "created_leads": stats["created"], "updated_leads": stats["updated"],
        "invalid_rows": stats["invalid"], "completed_at": finished_at, "updated_at": finished_at,
    })
    batch_payload = {
        "batch_id": batch_id, "filename": filename, "headers": headers,
        "status": terminal_status, "started_at": started_at, "finished_at": finished_at,
        "stats": stats, "preview": preview, "persona_id": persona_id,
        "audience_id": audience_id, "contact_basis": contact_basis, "error": error,
    }
    batch_event = _persist_terminal_batch_event(
        batch_id=batch_id,
        persona_id=persona_id,
        payload={key: value for key, value in batch_payload.items() if key != "preview"},
        level="info" if terminal_status == "completed" else "warn",
    )
    return {
        "ok": terminal_status == "completed", "status": terminal_status,
        "batch": batch_payload, "batch_event": batch_event,
        "preview": preview, "audience": audience, "error": error,
    }


async def _legacy_import_leads_csv(
    request: Request,
    file: UploadFile = File(...),
    persona_id: str | None = Form(None),
):
    if not persona_id:
        raise HTTPException(400, "Selecione uma persona antes de importar leads")
    auth_service.assert_persona_access(request, persona_id=persona_id)
    current_user = auth_service.current_user(request)

    filename = file.filename or "leads.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported for now")

    text = _decode_csv(await file.read())
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except Exception:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        raise HTTPException(400, "CSV header row is required")

    batch_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    headers = list(reader.fieldnames or [])

    # System "import" audience is idempotent â€” safe to ensure even if the batch
    # eventually fails; nothing is attached to it unless we insert memberships.
    import_audience = supabase_client.ensure_import_audience(
        persona_id,
        created_by_user_id=current_user.get("id"),
    )

    stats = {"total": 0, "valid": 0, "with_phone": 0, "email_only": 0, "created": 0, "updated": 0, "invalid": 0}
    preview: list[dict] = []

    try:
        for index, row in enumerate(reader, start=1):
            stats["total"] += 1
            raw_row = {str(k or ""): v for k, v in (row or {}).items()}
            first_name = _pick_field(raw_row, NAME_HEADERS)
            last_name = _pick_field(raw_row, LAST_NAME_HEADERS)
            name = _display_name(first_name, last_name, _pick_field(raw_row, {"nome completo", "name", "nome"}))
            phone = _normalize_phone(_pick_field(raw_row, PHONE_HEADERS))
            email = _pick_field(raw_row, EMAIL_HEADERS)
            city = _pick_field(raw_row, CITY_HEADERS)
            state = _pick_field(raw_row, STATE_HEADERS)
            zip_code = _pick_field(raw_row, ZIP_HEADERS)
            country = _pick_field(raw_row, COUNTRY_HEADERS)
            row_status = "created"
            row_error = ""
            lead_ref = None

            has_phone = bool(phone and len(phone) >= 8)
            has_email = bool(email)
            if not has_phone and not has_email:
                row_status = "invalid"
                row_error = "missing_email_and_phone"
                stats["invalid"] += 1
            elif not has_phone:
                row_status = "email_only"
                stats["valid"] += 1
                stats["email_only"] += 1
            else:
                stats["valid"] += 1
                stats["with_phone"] += 1
                existing = supabase_client.get_lead(phone)
                lead = supabase_client.ensure_lead_for_persona(
                    lead_id=phone,
                    persona_slug_or_id=persona_id,
                    nome=name or None,
                    stage="novo",
                    canal="bulk_import",
                    cidade=city or None,
                    cep=zip_code or None,
                )
                lead_ref = lead.get("id") if lead else None
                if lead_ref and import_audience:
                    supabase_client.ensure_lead_membership(
                        lead_ref,
                        import_audience["id"],
                        membership_type="primary" if not existing else "shared",
                        created_by_user_id=current_user.get("id"),
                    )
                if existing:
                    row_status = "updated"
                    stats["updated"] += 1
                else:
                    stats["created"] += 1

            row_payload = {
                "batch_id": batch_id,
                "filename": filename,
                "row_index": index,
                "raw_row": raw_row,
                "parsed": {
                    "nome": name,
                    "fn": first_name,
                    "ln": last_name,
                    "lead_id": phone,
                    "email": email,
                    "cidade": city,
                    "estado": state,
                    "zip": zip_code,
                    "country": country,
                },
                "status": row_status,
                "error": row_error,
                "lead_ref": lead_ref,
                "persona_id": persona_id,
            }
            supabase_client.insert_event(
                {
                    "event_type": "lead_import_row",
                    "entity_type": "lead_import",
                    "entity_id": batch_id,
                    "persona_id": persona_id,
                    "payload": row_payload,
                },
                level="error" if row_status == "invalid" else "info",
                source="leads.import",
            )
            if row_status != "invalid" and len(preview) < 5:
                preview.append(row_payload)
    except Exception as exc:
        # Any unexpected error mid-loop: persist a single terminal `failed`
        # event so the import never lingers as orphan/running. Re-raise as 500
        # so the frontend surfaces the failure.
        finished_at = datetime.now(timezone.utc).isoformat()
        failed_payload = {
            "batch_id": batch_id,
            "filename": filename,
            "headers": headers,
            "status": "failed",
            "started_at": started_at,
            "finished_at": finished_at,
            "stats": stats,
            "preview": preview,
            "persona_id": persona_id,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _persist_terminal_batch_event(
            batch_id=batch_id, persona_id=persona_id, payload=failed_payload, level="error"
        )
        raise HTTPException(500, f"Falha ao importar leads: {exc}")

    finished_at = datetime.now(timezone.utc).isoformat()

    # Total == 0 â†’ CSV header-only. Refuse without persisting noise.
    if stats["total"] == 0:
        raise HTTPException(400, "CSV sem linhas alem do cabecalho.")

    valid = stats["valid"]
    if valid == 0:
        # Header parsed but no row produced a usable lead. Persist as failed
        # with a descriptive error and skip the legacy audience graph node so
        # the graph nao polui com lixo de import.
        failed_payload = {
            "batch_id": batch_id,
            "filename": filename,
            "headers": headers,
            "status": "failed",
            "started_at": started_at,
            "finished_at": finished_at,
            "stats": stats,
            "preview": preview,
            "persona_id": persona_id,
            "error": _EMPTY_HASHED_HINT,
        }
        _persist_terminal_batch_event(
            batch_id=batch_id, persona_id=persona_id, payload=failed_payload, level="warn"
        )
        return {
            "ok": False,
            "status": "failed",
            "batch": failed_payload,
            "preview": preview,
            "audience_node": None,
            "error": _EMPTY_HASHED_HINT,
        }

    batch_payload = {
        "batch_id": batch_id,
        "filename": filename,
        "headers": headers,
        "status": "completed",
        "started_at": started_at,
        "finished_at": finished_at,
        "stats": stats,
        "preview": preview,
        "persona_id": persona_id,
    }
    batch_event = _persist_terminal_batch_event(
        batch_id=batch_id, persona_id=persona_id, payload=batch_payload, level="info"
    )
    audience_node = _create_audience_node(persona_id=persona_id, batch_payload=batch_payload)
    batch_payload["audience_node_id"] = audience_node.get("id") if audience_node else None
    return {
        "ok": True,
        "status": "completed",
        "batch": batch_payload,
        "batch_event": batch_event,
        "preview": preview,
        "audience_node": audience_node,
    }


_TERMINAL_BATCH_STATUSES = {"completed", "failed"}
_STALE_RUNNING_THRESHOLD_SECONDS = 120


def _reclassify_stale_running(item: dict) -> dict:
    """Mark batches stuck on `running` (or sem status terminal) as `failed`.

    Cobre lixo legado de antes da gravacao atomica do handler â€” o handler novo
    nunca escreve `running`, mas eventos antigos podem persistir. Sem este
    filtro o frontend listava esses batches como 'imports validos' com 0/0/0.
    """
    status = (item.get("status") or "").lower()
    if status in _TERMINAL_BATCH_STATUSES:
        return item
    created = item.get("created_at") or ""
    age_seconds = 0.0
    try:
        ts = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
        age_seconds = (datetime.now(timezone.utc) - ts).total_seconds()
    except Exception:
        age_seconds = 0.0
    if age_seconds < _STALE_RUNNING_THRESHOLD_SECONDS:
        return item
    stats = item.get("stats") or {"total": 0, "valid": 0, "with_phone": 0, "email_only": 0, "created": 0, "updated": 0, "invalid": 0}
    return {
        **item,
        "status": "failed",
        "stats": stats,
        "error": item.get("error") or "Importacao interrompida sem evento terminal (orfa).",
    }


@router.get("/imports")
def list_lead_imports(
    request: Request,
    limit: int = Query(20, le=100),
    persona_id: str | None = Query(None),
):
    if persona_id:
        auth_service.assert_persona_capability(request, "view", persona_id=persona_id)
        persona_ids = [persona_id]
    elif auth_service.is_admin(auth_service.current_user(request)):
        persona_ids = []
    else:
        persona_ids = auth_service.allowed_persona_ids(request)

    batches: list[dict] = []
    if persona_ids:
        for allowed_id in persona_ids:
            batches.extend(supabase_client.list_lead_import_batches(allowed_id, limit=limit))
    else:
        batches = supabase_client.list_lead_import_batches(limit=limit)
    batches.sort(key=lambda row: row.get("created_at") or "", reverse=True)
    projected: list[dict] = []
    for batch in batches[:limit]:
        rows = supabase_client.list_lead_import_rows(batch["id"], limit=10)
        preview = [
            {
                "batch_id": batch["id"], "row_index": row.get("row_index"),
                "parsed": row.get("parsed_data") or {}, "status": row.get("status"),
                "error": row.get("error_code") or "", "lead_ref": row.get("lead_id"),
            }
            for row in rows if row.get("status") != "invalid"
        ][:5]
        projected.append({
            **batch,
            "batch_id": batch["id"],
            "stats": {
                "total": batch.get("total_rows", 0), "valid": batch.get("valid_rows", 0),
                "with_phone": batch.get("phone_rows", 0), "email_only": batch.get("email_only_rows", 0),
                "created": batch.get("created_leads", 0), "updated": batch.get("updated_leads", 0),
                "invalid": batch.get("invalid_rows", 0),
            },
            "preview": preview,
        })
    return projected


@router.get("/imports/{batch_event_id}")
def get_lead_import(batch_event_id: str, request: Request, limit: int = Query(2000, le=5000)):
    batch = supabase_client.get_lead_import_batch(batch_event_id)
    if not batch:
        raise HTTPException(404, "Import nao encontrado.")
    auth_service.assert_persona_capability(request, "view", persona_id=batch.get("persona_id"))
    rows = supabase_client.list_lead_import_rows(batch_event_id, limit=limit)
    parsed_rows = [{
        "batch_id": batch_event_id,
        "row_index": row.get("row_index"),
        "parsed": row.get("parsed_data") or {},
        "status": row.get("status"),
        "error": row.get("error_code") or "",
        "lead_ref": row.get("lead_id"),
    } for row in rows]
    return {
        "batch": {
            **batch,
            "batch_id": batch["id"],
            "stats": {
                "total": batch.get("total_rows", 0), "valid": batch.get("valid_rows", 0),
                "with_phone": batch.get("phone_rows", 0), "email_only": batch.get("email_only_rows", 0),
                "created": batch.get("created_leads", 0), "updated": batch.get("updated_leads", 0),
                "invalid": batch.get("invalid_rows", 0),
            },
        },
        "rows": parsed_rows,
    }


@router.delete("/imports/{batch_id}")
def delete_lead_import(batch_id: str, request: Request):
    batch = supabase_client.get_lead_import_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Import nao encontrado.")
    persona_id = batch.get("persona_id")
    auth_service.assert_persona_capability(request, "edit", persona_id=persona_id)
    archived_at = datetime.now(timezone.utc).isoformat()
    supabase_client.update_lead_import_batch(batch_id, {
        "status": "archived", "archived_at": archived_at, "updated_at": archived_at,
    })
    supabase_client.insert_event(
        {
            "event_type": "lead_import_batch_deleted",
            "entity_type": "lead_import",
            "entity_id": batch_id,
            "persona_id": persona_id,
            "payload": {
                "batch_id": batch_id,
                "filename": batch.get("filename"),
                "archived_at": archived_at,
                "status": "archived",
            },
        },
        level="info",
        source="leads.import",
    )
    return {"ok": True, "batch_id": batch_id, "status": "archived"}


@router.get("/{lead_id}")
def get_lead(lead_id: str, request: Request):
    lead = supabase_client.get_lead(lead_id)
    if not lead:
        raise HTTPException(404, "Lead not found")
    try:
        lead_pk = int(lead.get("id") or 0)
    except (TypeError, ValueError):
        lead_pk = 0
    memberships: list[dict] = []
    if lead_pk > 0:
        try:
            memberships = supabase_client.get_lead_memberships(lead_pk) or []
        except Exception:
            memberships = []
    visible = False
    for membership in memberships:
        audience = membership.get("audience") or {}
        persona_id = audience.get("persona_id")
        if not persona_id:
            continue
        try:
            auth_service.assert_persona_access(request, persona_id=persona_id)
            visible = True
            break
        except HTTPException:
            continue
    if not visible and lead.get("persona_id"):
        # Compatibilidade: leads antigos sem membership ainda podem usar
        # leads.persona_id como fallback de visibilidade.
        auth_service.assert_persona_access(request, persona_id=lead.get("persona_id"))
    return journey_outcome.decorate_leads([lead_qualification.decorate_lead(lead)])[0]


class LeadInfoUpdateBody(BaseModel):
    nome: str | None = None
    interesse_produto: str | None = None
    commercial_note: dict[str, str] | None = None


@router.patch("/{lead_ref}")
def update_lead_info(lead_ref: int, body: LeadInfoUpdateBody, request: Request):
    """Manual edits to a lead's name/product/commercial note.

    Writes land in two places: the lead's own columns (for display) and
    metadata.conversation_state.appointment_request (the AI's working
    memory), so the next reply treats edited fields as already known
    instead of asking for them again.
    """
    lead = supabase_client.get_lead_by_ref(lead_ref)
    if not lead:
        raise HTTPException(404, "Lead nao encontrado")
    if lead.get("persona_id"):
        auth_service.assert_persona_access(request, persona_id=lead["persona_id"])

    update: dict = {}
    if body.nome is not None:
        stripped = body.nome.strip()
        if not stripped:
            raise HTTPException(400, "Nome nao pode ser vazio")
        update["nome"] = stripped
    if body.interesse_produto is not None:
        update["interesse_produto"] = body.interesse_produto.strip() or None

    if body.commercial_note is not None:
        update["metadata"] = supabase_client.merge_commercial_note(
            lead.get("metadata") or {}, body.commercial_note
        )

    if not update:
        raise HTTPException(400, "Nenhum campo para atualizar")

    supabase_client.update_lead(lead_ref, update)
    updated = supabase_client.get_lead_by_ref(lead_ref)
    event_emitter.emit(
        "lead.info_updated",
        entity_type="lead",
        entity_id=str(lead_ref),
        persona_id=lead.get("persona_id"),
        payload={"fields": list(update.keys())},
        source="leads.update_lead_info",
    )
    return {"ok": True, "lead": updated}


def _resolve_target_audience(body: LeadAudienceChangeBody, request: Request) -> dict:
    """Resolve a audience destino. Se o slug requisitado for `import` e
    a persona destino ainda nao tiver, criamos via helper idempotente em vez
    de devolver 404. Para qualquer outro slug nao encontrado, mantemos 404."""
    auth_service.assert_persona_access(request, persona_id=body.target_persona_id)
    audience = None
    if body.target_audience_id:
        audience = supabase_client.get_audience(body.target_audience_id)
    elif body.target_audience_slug:
        audience = supabase_client.get_audience_by_slug(body.target_persona_id, body.target_audience_slug)
        if not audience and body.target_audience_slug == "import":
            audience = supabase_client.ensure_import_audience(body.target_persona_id)
    if not audience or audience.get("persona_id") != body.target_persona_id:
        raise HTTPException(404, "Target audience not found for persona")
    return audience


def _resolve_source_audience(lead_ref: int, body: LeadAudienceChangeBody) -> dict | None:
    if body.source_audience_id:
        return supabase_client.get_audience(body.source_audience_id)
    if body.source_audience_slug and body.target_persona_id:
        return supabase_client.get_audience_by_slug(body.target_persona_id, body.source_audience_slug)
    try:
        memberships = supabase_client.get_lead_memberships(lead_ref) or []
    except Exception:
        memberships = []
    return memberships[0].get("audience") if memberships else None


@router.get("/{lead_id}/memberships")
def lead_memberships(lead_id: str, request: Request):
    """Endpoint defensivo: nunca quebra por lead sem membership.

    Sempre retorna 200 com `{lead, memberships: [...]}`. Para leads
    inexistentes mantemos 404 (e o frontend trata).
    """
    lead = supabase_client.get_lead(lead_id)
    if not lead:
        raise HTTPException(404, "Lead not found")
    try:
        lead_pk = int(lead.get("id") or 0)
    except (TypeError, ValueError):
        lead_pk = 0
    memberships: list[dict] = []
    if lead_pk > 0:
        try:
            memberships = supabase_client.get_lead_memberships(lead_pk) or []
        except Exception:
            memberships = []
    allowed: list[dict] = []
    for membership in memberships:
        audience = membership.get("audience") or {}
        persona_id = audience.get("persona_id")
        try:
            if persona_id:
                auth_service.assert_persona_access(request, persona_id=persona_id)
            allowed.append(membership)
        except HTTPException:
            continue
    return {"lead": lead, "memberships": allowed}


@router.post("/{lead_ref}/group")
def set_lead_group(lead_ref: int, body: LeadGroupBody, request: Request):
    """Replace the single semantic group; import cohort history is untouched."""
    lead = supabase_client.get_lead_by_ref(lead_ref)
    if not lead:
        raise HTTPException(404, "Lead not found")
    user = auth_service.current_user(request)
    current_persona_id = lead.get("persona_id")
    if not auth_service.is_admin(user) and current_persona_id != body.persona_id:
        raise HTTPException(403, "Somente admin pode trocar a persona do lead.")
    auth_service.assert_persona_capability(request, "edit", persona_id=body.persona_id)
    if not body.idempotency_key.strip() or not body.reason.strip():
        raise HTTPException(422, "idempotency_key e reason sao obrigatorios.")
    audience = supabase_client.get_audience(body.audience_id)
    if not audience or audience.get("persona_id") != body.persona_id:
        raise HTTPException(404, "Grupo nao encontrado na persona selecionada.")
    if str((audience.get("metadata") or {}).get("kind") or "semantic_group") != "semantic_group":
        raise HTTPException(422, "Selecione um grupo semantico.")
    result = supabase_client.replace_lead_semantic_group(
        lead_id=lead_ref,
        persona_id=body.persona_id,
        audience_id=body.audience_id,
        created_by_user_id=_database_actor_id(user),
        idempotency_key=body.idempotency_key,
        reason=body.reason,
    )
    return {
        "ok": True,
        "deduplicated": bool((result.get("result") or {}).get("deduplicated")),
        "group": result,
        "lead": supabase_client.get_lead_by_ref(lead_ref),
        "memberships": supabase_client.get_lead_memberships(lead_ref),
    }


def _record_manual_consent(
    lead_ref: int, body: ConsentChangeBody, request: Request, *, status: str,
) -> dict:
    lead = supabase_client.get_lead_by_ref(lead_ref)
    if not lead:
        raise HTTPException(404, "Lead not found")
    if lead.get("persona_id") != body.persona_id:
        raise HTTPException(403, "Lead nao pertence a persona informada.")
    auth_service.assert_persona_capability(request, "edit", persona_id=body.persona_id)
    if not body.purpose.strip() or not body.idempotency_key.strip() or not body.reason.strip():
        raise HTTPException(422, "purpose, idempotency_key e reason sao obrigatorios.")
    now = datetime.now(timezone.utc).isoformat()
    user = auth_service.current_user(request)
    consent = supabase_client.insert_contact_consent({
        "lead_id": lead_ref, "persona_id": body.persona_id,
        "channel": body.channel, "purpose": body.purpose, "status": status,
        "source": body.source, "request_message_id": body.request_message_id,
        "response_message_id": body.response_message_id, "campaign_id": body.campaign_id,
        "import_batch_id": body.import_batch_id, "effective_at": now,
        "granted_at": now if status == "granted" else None,
        "revoked_at": now if status == "revoked" else None,
        "evidence": {**body.evidence, "reason": body.reason},
        "actor_user_id": _database_actor_id(user),
        "idempotency_key": body.idempotency_key,
    }, audit_payload={
        "reason": body.reason,
        "actor_ref": user.get("id"),
    })
    return {"ok": True, "consent": consent}


@router.get("/{lead_ref}/consents")
def list_lead_consents(
    lead_ref: int, request: Request, persona_id: str = Query(...),
    channel: str | None = Query(None), purpose: str | None = Query(None),
):
    lead = supabase_client.get_lead_by_ref(lead_ref)
    if not lead or lead.get("persona_id") != persona_id:
        raise HTTPException(404, "Lead not found")
    auth_service.assert_persona_capability(request, "view", persona_id=persona_id)
    return supabase_client.list_contact_consents(
        lead_id=lead_ref, persona_id=persona_id, channel=channel, purpose=purpose,
    )


@router.post("/{lead_ref}/consent/grant")
def grant_lead_consent(lead_ref: int, body: ConsentChangeBody, request: Request):
    return _record_manual_consent(lead_ref, body, request, status="granted")


@router.post("/{lead_ref}/consent/revoke")
def revoke_lead_consent(lead_ref: int, body: ConsentChangeBody, request: Request):
    return _record_manual_consent(lead_ref, body, request, status="revoked")


@router.post("/{lead_ref}/move")
def move_lead(lead_ref: int, body: LeadAudienceChangeBody, request: Request):
    target = _resolve_target_audience(body, request)
    return set_lead_group(
        lead_ref,
        LeadGroupBody(
            persona_id=body.target_persona_id,
            audience_id=target["id"],
            idempotency_key=f"legacy-move:{uuid.uuid4()}",
            reason="Adaptador legado /move para grupo semantico unico",
        ),
        request,
    )


@router.post("/{lead_ref}/share")
def share_lead(lead_ref: int, body: LeadAudienceChangeBody, request: Request):
    target = _resolve_target_audience(body, request)
    return set_lead_group(
        lead_ref,
        LeadGroupBody(
            persona_id=body.target_persona_id,
            audience_id=target["id"],
            idempotency_key=f"legacy-share:{uuid.uuid4()}",
            reason="Adaptador legado /share convertido em troca de grupo",
        ),
        request,
    )


@router.post("/{lead_ref}/pause-ai")
def pause_ai(lead_ref: int, request: Request):
    """Pausa a IA para esse lead. /process passa a devolver agent_used=PAUSED."""
    lead = supabase_client.get_lead_by_ref(lead_ref)
    if lead and lead.get("persona_id"):
        auth_service.assert_persona_access(request, persona_id=lead.get("persona_id"))
    ok = agents_service.pause_lead(lead_ref)
    if not ok:
        raise HTTPException(500, "Falha ao pausar lead")
    event_emitter.emit(
        "lead.ai_paused",
        entity_type="lead",
        entity_id=str(lead_ref),
        payload={"ai_paused": True, "by": "manual"},
        source="leads.pause_ai",
    )
    return {"ok": True, "lead_ref": lead_ref, "ai_paused": True}


class HandoffBody(BaseModel):
    reason: str
    role: str = "human"


@router.post("/{lead_ref}/handoff")
def handoff(lead_ref: int, body: HandoffBody, request: Request):
    """Pause automation before a human is notified; it is never best-effort."""
    lead = supabase_client.get_lead_by_ref(lead_ref)
    if not lead:
        raise HTTPException(404, "Lead nao encontrado")
    if lead.get("persona_id"):
        auth_service.assert_persona_access(request, persona_id=lead["persona_id"])
    try:
        supabase_client.handoff_whatsapp_lead(lead_ref)
    except Exception as exc:
        raise HTTPException(500, "Falha ao registrar handoff") from exc
    event_emitter.emit("lead.handoff", entity_type="lead", entity_id=str(lead_ref), persona_id=lead.get("persona_id"), payload={"reason": body.reason, "role": body.role, "ai_paused": True}, source="leads.handoff")
    return {"ok": True, "lead_ref": lead_ref, "ai_paused": True, "reason": body.reason}


@router.post("/{lead_ref}/resume-ai")
def resume_ai(lead_ref: int, request: Request):
    """Retoma a IA para esse lead."""
    lead = supabase_client.get_lead_by_ref(lead_ref)
    if lead and lead.get("persona_id"):
        auth_service.assert_persona_access(request, persona_id=lead.get("persona_id"))
    ok = agents_service.resume_lead(lead_ref)
    if not ok:
        raise HTTPException(500, "Falha ao retomar lead")
    event_emitter.emit(
        "lead.ai_resumed",
        entity_type="lead",
        entity_id=str(lead_ref),
        payload={"ai_paused": False, "by": "manual"},
        source="leads.resume_ai",
    )
    notice = agents_service.reactivation_notice(lead_ref, reason="manual")
    return {"ok": True, "lead_ref": lead_ref, "ai_paused": False, "notice": notice}


@router.post("/{lead_ref}/acknowledge-handoff")
def acknowledge_handoff(lead_ref: int, request: Request):
    """Clear a 'partial' handoff flag once a human has reviewed it.

    Unlike /resume-ai, a partial handoff never stopped the AI -- there's
    nothing to requeue, just the flag to clear.
    """
    lead = supabase_client.get_lead_by_ref(lead_ref)
    if lead and lead.get("persona_id"):
        auth_service.assert_persona_access(request, persona_id=lead.get("persona_id"))
    ok = agents_service.acknowledge_partial_handoff(lead_ref)
    if not ok:
        raise HTTPException(500, "Falha ao confirmar handoff")
    event_emitter.emit(
        "lead.handoff_acknowledged",
        entity_type="lead",
        entity_id=str(lead_ref),
        payload={"by": "manual"},
        source="leads.acknowledge_handoff",
    )
    return {"ok": True, "lead_ref": lead_ref, "handoff_level": "none"}

