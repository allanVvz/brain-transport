from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Any, Optional
import os
import time

from services import auth_service, integration_service
from services import supabase_client
from services import graph_bundle_publisher
from schemas.agent_harness import MessageCreate
from services.agent_harness import SofiaAgentHarness
from services.agent_harness_repository import AgentHarnessRepository
from services.catalog_crawler import crawl_catalog_url
from services.kb_intake_service import (
    start_bootstrap_session,
    get_session,
    chat,
    save,
    AVAILABLE_MODELS,
    attach_crawler_capture,
    attach_reading,
    prune_unpersisted_asset_readings,
    update_session_plan,
    _invalid_criar_persona,
    _session_public_state,
    _save_session,
)

router = APIRouter(prefix="/kb-intake", tags=["kb-intake"])


def _assert_json_serializable(result: Any) -> None:
    """Raise if `result` cannot be JSON-encoded the way FastAPI will encode it.

    FastAPI serializes the return value AFTER the route handler returns â€” i.e.
    outside the handler's try/except. A non-serializable field (e.g. a set or a
    custom object stashed in mission_state/plan_state by the tool loop) would
    surface as a raw 500 with no traceback in the response body. Calling this
    inside the handler turns that into a catchable exception so the cause lands
    in `traceback_tail`."""
    from fastapi.encoders import jsonable_encoder
    import json as _json

    _json.dumps(jsonable_encoder(result))


def _json_safe(obj: Any) -> Any:
    """Best-effort coercion to JSON-safe data; never raises."""
    import json as _json

    try:
        return _json.loads(_json.dumps(obj, default=str))
    except Exception:
        return None


class StartBody(BaseModel):
    model: str = "gpt-4o-mini"
    initial_context: str = ""
    agent_key: str = "sofia"
    persona_slug: Optional[str] = None
    source_url: Optional[str] = None
    mode: str = "legacy"
    initial_block_counts: dict[str, int] = Field(default_factory=dict)
    knowledge_plan: Optional[dict[str, Any]] = None
    memory_summary: Optional[str] = None
    bootstrap_llm: bool = True


class MessageBody(BaseModel):
    session_id: str
    message: str


class SaveBody(BaseModel):
    session_id: str
    content: str = ""
    plan_override: Optional[dict[str, Any]] = None


class CrawlBody(BaseModel):
    url: str
    session_id: Optional[str] = None


class IngestDocumentBody(BaseModel):
    text: str
    model: str = "gpt-4o-mini"


class PlanUpdateBody(BaseModel):
    knowledge_plan: dict[str, Any]
    status: Optional[str] = None
    last_change: Optional[str] = None


class ApprovePublicationBody(BaseModel):
    approved_draft_checksum: str
    approved_runtime_checksum: str
    activate: bool = False
    acknowledge_breaking_changes: bool = False


def _assert_session_access(session_id: str, request: Request) -> dict[str, Any]:
    user = auth_service.current_user(request)
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    owner_id = session.get("user_id")
    if owner_id and owner_id != user.get("id"):
        raise HTTPException(403, "Sessao pertence a outro usuario.")
    return session


@router.get("/models")
def list_models(request: Request):
    user = auth_service.current_user(request)
    user_openai = bool(integration_service.get_enabled_user_secret(user.get("id") or "", "openai"))
    user_anthropic = bool(integration_service.get_enabled_user_secret(user.get("id") or "", "anthropic"))
    system_openai = integration_service.system_service_has_runtime_credentials("openai")
    system_anthropic = integration_service.system_service_has_runtime_credentials("anthropic")
    return [
        {
            "id": k,
            "name": v,
            "provider": "anthropic" if k.startswith("claude-") else "openai",
            "configured": bool(user_anthropic or system_anthropic) if k.startswith("claude-") else bool(user_openai or system_openai),
            "credential_source": "user" if k.startswith("claude-") and user_anthropic else "system" if k.startswith("claude-") and system_anthropic else "user" if user_openai and not k.startswith("claude-") else "system" if system_openai and not k.startswith("claude-") else None,
        }
        for k, v in AVAILABLE_MODELS.items()
    ]


@router.post("/start")
def start_session(body: StartBody, request: Request):
    user = auth_service.current_user(request)
    if body.model not in AVAILABLE_MODELS:
        raise HTTPException(400, f"Modelo nao disponivel: {body.model}")
    initial_state = {
        "mode": body.mode,
        "persona_slug": body.persona_slug,
        "source_url": body.source_url,
        "initial_block_counts": body.initial_block_counts,
        "knowledge_plan": body.knowledge_plan,
        "memory_summary": body.memory_summary,
    }
    if (body.mode or "").strip().lower() == "criar" and _invalid_criar_persona(body.persona_slug):
        raise HTTPException(400, "Selecione uma persona especifica antes de criar conhecimento.")
    if body.persona_slug:
        auth_service.assert_persona_access(request, persona_slug=body.persona_slug)
    result = start_bootstrap_session(
        body.model,
        initial_context=body.initial_context,
        agent_key=body.agent_key,
        initial_state=initial_state,
        bootstrap_llm=body.bootstrap_llm,
        user_id=user.get("id"),
    )
    try:
        persona = supabase_client.get_persona(body.persona_slug) if body.persona_slug else None
        if persona and result.get("session_id"):
            harness = SofiaAgentHarness(AgentHarnessRepository())
            durable_session = harness.adopt_legacy_session(
                session_id=str(result["session_id"]), persona_id=str(persona["id"]),
                user_id=str(user.get("id") or ""), source="kb-intake.start",
                selected_context={"persona_slug": body.persona_slug, "mode": body.mode},
                memory_summary=body.memory_summary or "",
            )
            result["agent_session_id"] = durable_session["id"]
    except Exception:
        # Compatibility remains available while migration 088 rolls through all
        # environments; the legacy request itself must not regress.
        pass
    return result


@router.post("/message")
def send_message(body: MessageBody, request: Request):
    legacy_session = _assert_session_access(body.session_id, request)
    harness_run = None
    try:
        repository = AgentHarnessRepository()
        durable = repository.get_session(body.session_id)
        if durable:
            harness_run = SofiaAgentHarness(repository).add_message(
                durable,
                MessageCreate(
                    message=body.message,
                    expected_revision=int(durable.get("revision") or 1),
                    idempotency_key=f"legacy-message:{body.session_id}:{int(time.time() * 1000)}",
                    reason="Mensagem recebida pelo adapter /kb-intake",
                    context=dict(durable.get("selected_context") or {}),
                ),
                user_id=str((auth_service.current_user(request) or {}).get("id") or ""),
            )
    except Exception:
        # A falha do harness nao interrompe o adapter durante o rollout.
        harness_run = None
    try:
        result = chat(body.session_id, body.message)
        # Validate serialization inside the handler so a non-serializable field
        # becomes a catchable error (with traceback) instead of a raw 500 raised
        # by FastAPI during response encoding, after this handler returns.
        _assert_json_serializable(result)
        if harness_run:
            result["agent_session_id"] = body.session_id
            result["agent_run_id"] = harness_run["id"]
        return result
    except Exception as exc:
        # Full safety net: log structured event + return controlled body so
        # the chat never propagates a bare 500 to the operator.
        import traceback as _tb
        tb_text = _tb.format_exc()
        try:
            from services import sre_logger
            sre_logger.error(
                "kb_intake_message",
                f"unhandled in chat() session={(body.session_id or '')[:8]} msg_len={len(body.message or '')}: {exc}",
                exc,
            )
        except Exception:
            pass
        # Try to recover the session safely. Never let _this_ raise.
        try:
            session = get_session(body.session_id)
        except Exception:
            session = None
        try:
            from services.kb_intake_service import _emit_kb_event  # internal helper
            _emit_kb_event(
                "kb_intake_error",
                session=session or {"id": body.session_id, "classification": {}},
                source="kb-intake.message",
                status="failed",
                transcript=False,
                result={
                    "endpoint": "/kb-intake/message",
                    "step": "chat",
                    "session_id": body.session_id,
                    "exception_type": type(exc).__name__,
                    "message": str(exc)[:500],
                    "traceback_tail": tb_text.splitlines()[-12:],
                },
            )
        except Exception:
            pass
        return {
            "ok": False,
            "error_code": "INTERNAL_ERROR",
            "exception_type": type(exc).__name__,
            "message": (
                "Nao consegui processar sua mensagem agora. Sua configuracao "
                "foi mantida â€” tente novamente ou clique em Salvar se ja houver plano."
            ),
            "detail": str(exc)[:300],
            "traceback_tail": tb_text.splitlines()[-12:],
            # Coerce to JSON-safe so this error body itself can never trigger a
            # second (serialization) 500 when mission_state holds odd types.
            "state": _json_safe((session or {}).get("mission_state")) if session else None,
        }


@router.post("/crawl-preview")
def crawl_preview(body: CrawlBody, request: Request):
    if body.session_id:
        _assert_session_access(body.session_id, request)
    try:
        result = crawl_catalog_url(body.url)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Erro no crawler: {exc}") from exc
    if body.session_id:
        attach_crawler_capture(body.session_id, result)
    return result


@router.post("/{session_id}/ingest-document")
def ingest_document(session_id: str, body: IngestDocumentBody, request: Request):
    """Document-text counterpart to /crawl-preview: extracts catalog
    candidates from a pasted/uploaded document (instead of a crawled URL)
    and attaches them the same way, so the existing 'build the whole tree'
    flow (build_full_tree_plan_from_session) picks them up unchanged. Every
    candidate lands in the plan as status='pendente_validacao' -- nothing
    from a bulk document reaches a publishable GraphBundle without the
    operator explicitly confirming it first."""
    _assert_session_access(session_id, request)
    from services.document_candidate_extractor import extract_candidates_from_document

    if not body.text.strip():
        raise HTTPException(400, {"error": "Documento vazio."})
    result = extract_candidates_from_document(body.text, model=body.model)
    attach_crawler_capture(session_id, result)
    return result


@router.post("/upload")
async def upload_file(
    request: Request,
    session_id: str = Form(...),
    message: str = Form(""),
    file: UploadFile = File(...),
):
    session_for_access = _assert_session_access(session_id, request)
    openai_api_key = integration_service.get_enabled_user_secret(session_for_access.get("user_id") or "", "openai")
    upload_started = time.perf_counter()
    last_mark = upload_started
    timings_ms: dict[str, int] = {}

    def mark_timing(name: str) -> None:
        nonlocal last_mark
        now = time.perf_counter()
        timings_ms[name] = int((now - last_mark) * 1000)
        last_mark = now

    content = await file.read()
    mark_timing("read")
    fname = file.filename or "upload"
    ext = ("." + fname.rsplit(".", 1)[-1].lower()) if "." in fname else ""

    storage_path: Optional[str] = None
    file_url: Optional[str] = None
    asset_id: Optional[str] = None
    asset_reading_summary: Optional[dict] = None

    # Resolve persona from the session for asset row tagging.
    session_persona_id: Optional[str] = None
    session_persona_slug: Optional[str] = None
    try:
        sess = get_session(session_id)
        if sess:
            mission = sess.get("mission_state") or {}
            session_persona_id = mission.get("persona_id") or sess.get("persona_id")
            session_persona_slug = mission.get("persona_slug") or sess.get("persona_slug")
            if not session_persona_slug:
                session_persona_slug = (sess.get("classification") or {}).get("persona_slug")
        if session_persona_slug and not session_persona_id:
            from services import supabase_client
            persona = supabase_client.get_persona(session_persona_slug) or {}
            session_persona_id = persona.get("id")
            session_persona_slug = persona.get("slug") or session_persona_slug
            if sess and session_persona_id:
                sess["persona_id"] = session_persona_id
                sess["persona_slug"] = session_persona_slug
                mission = sess.setdefault("mission_state", {})
                mission["persona_id"] = session_persona_id
                mission["persona_slug"] = session_persona_slug
    except Exception:
        pass
    if not session_persona_id:
        prune_unpersisted_asset_readings(get_session(session_id) or {})
        return {
            "ok": False,
            "error_code": "ASSET_PERSONA_REQUIRED",
            "message": "Para persistir um asset, a sessao precisa estar vinculada a uma persona. A sessao foi reiniciada para nao manter leitura sem grafo.",
            "reset_session": True,
        }

    # 1) Upload to Storage. If this fails, do not attach the reading to Sofia:
    # local-only readings make the session remember files that do not exist in
    # the assets system.
    try:
        from services.supabase_client import upload_to_storage, insert_kb_intake
        storage_bucket = "assets-raw"
        storage_owner = session_persona_id or session_id
        storage_path = f"{storage_owner}/{fname}"
        file_url = upload_to_storage(
            storage_bucket,
            storage_path,
            content,
            file.content_type or "application/octet-stream",
        )
        insert_kb_intake({
            "filename": fname,
            "file_path": f"{storage_bucket}:{storage_path}",
            "persona_id": session_persona_id,
            "status": "pending",
        })
        mark_timing("storage")
    except Exception as exc:
        from services import sre_logger
        sre_logger.error("kb_intake_upload", f"storage/db write failed: {exc}", exc)
        prune_unpersisted_asset_readings(get_session(session_id) or {})
        return {
            "ok": False,
            "error_code": "ASSET_STORAGE_FAILED",
            "message": "Nao consegui persistir a imagem no Storage. A sessao foi reiniciada para nao manter uma leitura sem arquivo.",
            "detail": str(exc)[:300],
            "reset_session": True,
        }

    # 2) Run the asset pipeline (classifier + OCR + fallbacks) for any uploaded file.
    bundle = None
    try:
        from services.asset_pipeline import run_pipeline, AssetPipelineContext
        ctx = AssetPipelineContext(
            persona_id=session_persona_id,
            persona_slug=session_persona_slug,
            session_id=session_id,
            upload_context="sofia_chat",
            original_filename=fname,
            mime=file.content_type or "",
            branch_hint=None,
            branch_label=None,
            openai_api_key=openai_api_key,
        )
        bundle = run_pipeline(content, ctx)
        mark_timing("asset_pipeline")
    except Exception as exc:
        from services import sre_logger
        sre_logger.warn("kb_intake_upload", f"asset pipeline skipped: {exc}", exc)
        prune_unpersisted_asset_readings(get_session(session_id) or {})
        return {
            "ok": False,
            "error_code": "ASSET_PIPELINE_FAILED",
            "message": "A imagem foi enviada ao Storage, mas nao consegui registrar a leitura nos assets. A sessao foi reiniciada para nao manter contexto incompleto.",
            "detail": str(exc)[:300],
            "reset_session": True,
        }

    # 3) Persist public.assets + asset_readings for the chat upload too.
    if bundle is not None:
        try:
            from services import supabase_client
            asset_type = _bundle_to_asset_type(bundle.classification.kind)
            asset_row = supabase_client.insert_asset({
                "persona_id": session_persona_id,
                "type": asset_type,
                "name": bundle.rename.title or fname,
                "url": file_url,
                "metadata": {
                    "session_id": session_id,
                    "persona_slug": session_persona_slug,
                    "original_filename": fname,
                    "preview_path": None,
                    "reading_status": bundle.reading_status,
                    "extracted_text": (bundle.extracted_text or "")[:8000],
                    "visual_summary": bundle.visual_summary or "",
                    "video_reading_mocked": bool(bundle.video_mock),
                    "upload_context": "sofia_chat",
                    "validation_status": "context_only",
                    "kind": bundle.classification.kind,
                    "engine": (bundle.ocr.engine if bundle.ocr else None),
                    "ai_fallback_used": bundle.ai_fallback is not None and not (bundle.ai_fallback.error or ""),
                },
                "source": "upload",
                "storage_bucket": storage_bucket,
                "storage_path": storage_path,
                "mime_type": file.content_type or "",
                "file_size": len(content),
                "original_filename": fname,
                "status": "ready" if bundle.reading_status in ("completed", "mocked") else "reading",
                "upload_context": "sofia_chat",
            })
            asset_id = (asset_row or {}).get("id")
            if asset_id:
                for row in bundle.rows_to_persist:
                    payload = dict(row)
                    payload["asset_id"] = asset_id
                    try:
                        supabase_client.insert_asset_reading(payload)
                    except Exception as exc2:
                        from services import sre_logger
                        sre_logger.warn("kb_intake_upload", f"asset_reading insert failed: {exc2}", exc2)
            mark_timing("asset_db")
        except Exception as exc:
            from services import sre_logger
            sre_logger.warn("kb_intake_upload", f"asset insert skipped: {exc}", exc)

        if not asset_id:
            prune_unpersisted_asset_readings(get_session(session_id) or {})
            return {
                "ok": False,
                "error_code": "ASSET_DB_FAILED",
                "message": "A imagem foi enviada ao Storage, mas nao foi persistida no sistema de assets. A sessao foi reiniciada para evitar leitura fantasma.",
                "reset_session": True,
            }

        asset_reading_summary = bundle.to_summary()
        # 4) Attach reading to the session so Sofia sees it as context.
        try:
            attach_reading(
                session_id,
                asset_reading_summary,
                file_meta={
                    "filename": fname,
                    "size": len(content),
                    "mime": file.content_type or "",
                    "url": file_url,
                    "asset_id": asset_id,
                    "storage_bucket": storage_bucket,
                    "storage_path": storage_path,
                },
            )
            mark_timing("attach_reading")
        except Exception as exc:
            from services import sre_logger
            sre_logger.warn("kb_intake_upload", f"attach_reading skipped: {exc}", exc)

    # 5) Hand the file to chat() (existing behavior) so the assistant can react.
    file_info = {
        "filename": fname,
        "size": len(content),
        "content_type": file.content_type or "",
        "ext": ext,
        "bytes": content,
        "asset_reading": asset_reading_summary,
    }
    try:
        result = chat(session_id, message, file_info=file_info)
        mark_timing("chat_after_upload")
    except Exception as exc:
        import traceback as _tb
        tb_text = _tb.format_exc()
        try:
            from services import sre_logger
            sre_logger.error(
                "kb_intake_upload",
                f"unhandled in chat(file_info) session={(session_id or '')[:8]}: {exc}",
                exc,
            )
        except Exception:
            pass
        try:
            session = get_session(session_id)
        except Exception:
            session = None
        return {
            "ok": False,
            "error_code": "INTERNAL_ERROR",
            "exception_type": type(exc).__name__,
            "message": "Nao consegui processar o arquivo agora. Mantive sua configuracao e voce pode tentar novamente.",
            "detail": str(exc)[:300],
            "traceback_tail": tb_text.splitlines()[-12:],
            "state": (session or {}).get("mission_state") if session else None,
            "asset_reading": asset_reading_summary,
            "asset_id": asset_id,
        }
    if file_url:
        result["file_url"] = file_url
        result["storage_path"] = storage_path
    if asset_id:
        result["asset_id"] = asset_id
    if asset_reading_summary:
        result["asset_reading"] = asset_reading_summary
    timings_ms["total"] = int((time.perf_counter() - upload_started) * 1000)
    try:
        from services import sre_logger
        sre_logger.info(
            "kb_intake_upload",
            f"upload timings session={(session_id or '')[:8]} asset={asset_id or '-'} timings_ms={timings_ms}",
        )
    except Exception:
        pass
    if os.environ.get("ENV", "").lower() in {"dev", "development", "local"} or os.environ.get("DEBUG_TIMINGS"):
        result["timings_ms"] = timings_ms
    return result


def _bundle_to_asset_type(kind: str) -> str:
    if kind.startswith("image"):
        return "image"
    if kind == "pdf":
        return "pdf"
    if kind == "video":
        return "video"
    if kind in ("text", "markdown"):
        return "text"
    return "image"


@router.get("/session/{session_id}")
def get_session_info(session_id: str, request: Request):
    session = _assert_session_access(session_id, request)
    prune_unpersisted_asset_readings(session)
    live_state = _session_public_state(session)
    return {
        "id": session["id"],
        "stage": session["stage"],
        "model": session["model"],
        "classification": {k: v for k, v in session["classification"].items() if k != "file_bytes"},
        "message_count": len(session["messages"]),
        "state": session.get("mission_state"),
        "resumed_from_session_id": session.get("resumed_from_session_id"),
        "resume_source": session.get("resume_source"),
        "resume_summary": session.get("resume_summary"),
        **live_state,
    }


@router.patch("/session/{session_id}/plan")
def patch_session_plan(session_id: str, body: PlanUpdateBody, request: Request):
    _assert_session_access(session_id, request)
    result = update_session_plan(
        session_id,
        body.knowledge_plan,
        status=body.status,
        source="kb-intake.sidebar",
        last_change=body.last_change or "frontend plan sync",
    )
    if result.get("ok") is False:
        raise HTTPException(400, result)
    return result


@router.post("/save")
def save_knowledge(body: SaveBody, request: Request):
    _assert_session_access(body.session_id, request)
    try:
        result = save(body.session_id, body.content, body.plan_override)
        # Validate serialization here (inside the try) so a non-serializable
        # field in the success body OR the HTTPException(400, result) detail
        # below cannot escape as a raw 500 during FastAPI response encoding.
        _assert_json_serializable(result)
    except Exception as exc:
        # Safety net: surface unhandled exceptions in the response body so the
        # frontend (and the operator) can see the real cause without digging
        # through stderr. Remove once the save path is stable.
        import traceback as _tb
        tb_text = _tb.format_exc()
        try:
            from services import sre_logger
            sre_logger.error("kb_intake_save", f"unhandled in save(): {exc}", exc)
        except Exception:
            pass
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Unhandled exception in save(): {exc}",
                "exception_type": type(exc).__name__,
                "traceback": tb_text.splitlines()[-20:],
            },
        )

    if "error" in result:
        try:
            from services import sre_logger
            sre_logger.warn(
                "kb_intake_save",
                f"save rejected session={body.session_id[:8]} error={result.get('error')} classification={result.get('classification')}",
            )
        except Exception:
            pass
        raise HTTPException(400, result)
    return result


@router.post("/{session_id}/approve-publication")
def approve_publication(session_id: str, body: ApprovePublicationBody, request: Request):
    """Human approval gate for a Sofia-built GraphBundle staged in
    save()/_save_via_graph_bundle (session["pending_graph_bundle"] +
    session["pending_publication_plan"], stage="awaiting_publication_approval").

    Never called implicitly -- save() stops short of staging/activating a
    GraphBundle precisely so this checksum-approved call is the only way a
    Sofia-authored persona ever reaches production, mirroring the manual
    graph-publisher review used to activate Tock Fatal's graph this session
    and api/scripts/publish_graph_bundle.py's --apply/--activate contract.
    """
    user = auth_service.current_user(request)
    session = _assert_session_access(session_id, request)
    if session.get("stage") != "awaiting_publication_approval":
        raise HTTPException(409, {
            "error": "Sessao nao tem uma publicacao pendente de aprovacao.",
            "stage": session.get("stage"),
        })
    bundle = session.get("pending_graph_bundle")
    plan = session.get("pending_publication_plan") or {}
    if not bundle or not plan:
        raise HTTPException(409, {"error": "Sessao nao tem GraphBundle/plano pendente."})
    if plan.get("breaking_contract_changes") and not body.acknowledge_breaking_changes:
        raise HTTPException(400, {
            "error": (
                "Este plano tem mudancas de ruptura -- confirme "
                "acknowledge_breaking_changes=true depois de revisar."
            ),
            "breaking_contract_changes": plan["breaking_contract_changes"],
        })
    actor = str(user.get("email") or user.get("id") or "kb-intake-operator")
    try:
        staged = graph_bundle_publisher.stage_bundle(
            bundle,
            approved_draft_checksum=body.approved_draft_checksum,
            actor=actor,
        )
    except graph_bundle_publisher.GraphBundlePublishError as exc:
        raise HTTPException(400, {"error": str(exc), "error_code": "GRAPH_BUNDLE_STAGE_FAILED"})
    publication = staged.get("publication") or {}
    result: dict[str, Any] = {
        "ok": True,
        "staged": True,
        "activated": False,
        "publication_id": publication.get("id"),
        "version": publication.get("version"),
        "runtime_checksum": publication.get("checksum"),
    }
    if body.activate:
        try:
            activation = graph_bundle_publisher.activate_staged_bundle(
                bundle,
                publication_id=str(publication.get("id")),
                approved_draft_checksum=body.approved_draft_checksum,
                approved_runtime_checksum=body.approved_runtime_checksum,
                actor=actor,
            )
        except graph_bundle_publisher.GraphBundlePublishError as exc:
            raise HTTPException(400, {"error": str(exc), "error_code": "GRAPH_BUNDLE_ACTIVATE_FAILED"})
        result["activated"] = True
        result["activation"] = activation.get("activation")
    session["stage"] = "done"
    session["status"] = "saved" if body.activate else "staged"
    session.pop("pending_graph_bundle", None)
    _save_session(session)
    return result

