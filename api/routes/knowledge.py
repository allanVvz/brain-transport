import asyncio
import hashlib
import mimetypes
import os
from datetime import datetime, timezone
from fastapi import APIRouter, Query, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from typing import Optional
from services import auth_service, supabase_client, knowledge_graph, knowledge_lifecycle
from services import context_cards as context_cards_service
from services import approved_knowledge_snapshots
from services import graph_document_publisher, graph_json_v2_store
from services import graph_json_v21_adapter
from services import graph_context_resolver_v2
from schemas.graph_json_v2 import Edge, GraphJson
from services import integration_service, product_import_service
from services.knowledge_rag_backfill import backfill_knowledge_rag
from services.knowledge_rag_intake import process_intake, process_intake_plan
from services.vault_sync import run_sync, scan_vault
from services.event_emitter import emit
from services.audit_helpers import current_actor, summarize_diff
from services import knowledge_taxonomy
from core.landing_slots import LandingSlot, edge_metadata_for_slot, slot_config

VAULT_SOURCE_MODE = os.environ.get("VAULT_SOURCE_MODE")
OBSIDIAN_LOCAL_PATH = os.environ.get("OBSIDIAN_LOCAL_PATH")

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


class ResolveContextBody(BaseModel):
    persona_slug: str
    destination_id: str
    graph_version: int
    intent: str = "product_interest"
    query: str = ""
    seed_refs: list[str] = Field(default_factory=list)
    max_nodes: int = 24
    max_tokens: int = 8000


class PublishContextCardBody(BaseModel):
    persona_slug: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1, max_length=30000)
    expected_version: int = Field(..., ge=1)
    reason: str = Field(..., min_length=3, max_length=1000)
    idempotency_key: Optional[str] = Field(None, min_length=1, max_length=200)


@router.post("/context/resolve")
def resolve_graph_context(body: ResolveContextBody, request: Request):
    persona = supabase_client.get_persona(body.persona_slug)
    if not persona:
        raise HTTPException(404, f"Persona not found: {body.persona_slug}")
    auth_service.assert_persona_access(
        request,
        persona_id=persona.get("id"),
        persona_slug=body.persona_slug,
    )
    try:
        return graph_context_resolver_v2.resolve_context(
            persona_slug=body.persona_slug,
            destination_id=body.destination_id,
            graph_version=body.graph_version,
            intent=body.intent,
            query=body.query,
            seed_refs=body.seed_refs,
            max_nodes=max(1, min(body.max_nodes, 50)),
            max_tokens=max(256, min(body.max_tokens, 32000)),
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc


@router.get("/taxonomy")
def get_canonical_taxonomy(canonical_only: bool = False):
    """Returns the canonical fractal graph taxonomy.

    Single source of truth used by the dashboard, the canonical middleware
    and Sofia's tools. `canonical_only=true` filters out legacy aliases.
    """
    snapshot = knowledge_taxonomy.taxonomy_snapshot()
    if canonical_only:
        snapshot["node_types"] = [n for n in snapshot["node_types"] if n.get("canonical")]
        snapshot["relations"] = [r for r in snapshot["relations"] if r.get("canonical")]
    return snapshot


class RagIntakeBody(BaseModel):
    raw_text: str
    persona_id: Optional[str] = None
    persona_slug: Optional[str] = None
    source: str = "manual"
    source_ref: Optional[str] = None
    title: Optional[str] = None
    content_type: Optional[str] = None
    tags: list[str] = []
    metadata: dict = {}
    submitted_by: Optional[str] = None
    validate: bool = False
    parent_node_id: Optional[str] = None
    parent_relation_type: str = "manual"


class RagBackfillBody(BaseModel):
    persona_id: Optional[str] = None
    persona_slug: Optional[str] = None
    include_vault: bool = True
    # This is no longer used, vault source is configured by env vars
    # vault_path: Optional[str] = None 
    limit_items: int = 5000
    limit_nodes: int = 5000


class RagIntakePlanBody(BaseModel):
    persona_id: Optional[str] = None
    persona_slug: Optional[str] = None
    run_token: Optional[str] = None
    entries: list[dict]
    links: list[dict] = []
    source: str = "plan"
    source_ref: Optional[str] = None
    submitted_by: Optional[str] = None
    validate: bool = True


class ProductBody(BaseModel):
    persona_id: Optional[str] = None
    persona_slug: Optional[str] = None
    slug: Optional[str] = None
    title: str
    summary: Optional[str] = None
    tags: list[str] = []
    status: str = "pending_validation"
    collection_slug: Optional[str] = None
    category_slug: Optional[str] = None
    metadata: dict = {}


class ProductPatchBody(BaseModel):
    title: Optional[str] = None
    summary: Optional[str] = None
    tags: Optional[list[str]] = None
    metadata: Optional[dict] = None
    status: Optional[str] = None


class LinkAssetBody(BaseModel):
    asset_id: Optional[str] = None
    asset_node_id: Optional[str] = None
    relation_type: str = "product_image"
    metadata: dict = {}


class SofiaSuggestBody(BaseModel):
    limit: int = 12
    min_score: float = 0.15


@router.post("/intake")
def intake_rag_knowledge(body: RagIntakeBody, request: Request):
    if not body.raw_text.strip():
        raise HTTPException(400, "raw_text is required")
    if not body.persona_id and not body.persona_slug:
        raise HTTPException(400, "persona_id or persona_slug is required")
    auth_service.assert_persona_access(request, persona_id=body.persona_id, persona_slug=body.persona_slug)
    try:
        result = process_intake(**body.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Knowledge intake failed: {exc}") from exc

    rag_entry = result.get("rag_entry") or {}
    emit(
        "knowledge_rag_intake_created",
        entity_type="knowledge_rag_entry",
        entity_id=rag_entry.get("id"),
        persona_id=rag_entry.get("persona_id"),
        payload={
            "title": rag_entry.get("title"),
            "content_type": rag_entry.get("content_type"),
            "status": rag_entry.get("status"),
        },
    )
    return result


@router.post("/intake/plan")
def intake_rag_knowledge_plan(body: RagIntakePlanBody, request: Request):
    if not body.entries:
        raise HTTPException(400, "entries is required")
    if not body.persona_id and not body.persona_slug:
        raise HTTPException(400, "persona_id or persona_slug is required")
    auth_service.assert_persona_access(request, persona_id=body.persona_id, persona_slug=body.persona_slug)
    try:
        result = process_intake_plan(**body.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Knowledge plan intake failed: {exc}") from exc

    emit(
        "knowledge_rag_plan_intake_created",
        entity_type="knowledge_rag_plan",
        entity_id=result.get("run_token"),
        persona_id=(result.get("persona") or {}).get("id"),
        payload={
            "entries_created": result.get("entries_created"),
            "nodes_created": result.get("nodes_created"),
            "main_edges": result.get("main_edges"),
            "auxiliary_edges": result.get("auxiliary_edges"),
        },
    )
    return result

# â”€â”€ Vault Sync â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.post("/rag/backfill")
async def backfill_rag_knowledge(body: RagBackfillBody):
    """Reprocess legacy knowledge into knowledge_rag_entries/chunks/links."""
    if body.persona_id and body.persona_slug:
        raise HTTPException(400, "Use persona_id or persona_slug, not both")
    try:
        # Pass a dictionary without vault_path to the backfill function
        dump = body.model_dump()
        dump.pop("vault_path", None) 
        result = await asyncio.to_thread(backfill_knowledge_rag, **dump)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"RAG backfill failed: {exc}") from exc
    emit(
        "knowledge_rag_backfill_completed",
        entity_type="knowledge_rag_backfill",
        persona_id=body.persona_id,
        payload=result,
    )
    return result


@router.post("/import-vault")
async def import_vault(persona: str = Query(None)):
    emit("vault_sync_started", entity_type="sync", payload={})
    result = await asyncio.to_thread(run_sync, persona_filter=persona)
    if "error" in result:
        emit("vault_sync_failed", payload=result)
        raise HTTPException(400, result["error"])
    return result


@router.post("/sync", deprecated=True)
async def trigger_sync(persona: str = Query(None)):
    emit(
        "deprecated_route_used",
        entity_type="route",
        entity_id="/knowledge/sync",
        payload={"replacement": "/knowledge/import-vault"},
        level="warn",
    )
    return await import_vault(persona)


@router.get("/import-vault/preview")
async def preview_import_vault():
    result = await asyncio.to_thread(scan_vault)
    return result


@router.get("/sync/preview", deprecated=True)
async def preview_sync():
    emit("deprecated_route_used", entity_type="route", entity_id="/knowledge/sync/preview", payload={"replacement": "/knowledge/import-vault/preview"}, level="warn")
    return await preview_import_vault()


@router.get("/import-vault/runs")
def list_import_vault_runs(limit: int = 20):
    return supabase_client.get_sync_runs(limit)


@router.get("/sync/runs", deprecated=True)
def list_sync_runs(limit: int = 20):
    emit("deprecated_route_used", entity_type="route", entity_id="/knowledge/sync/runs", payload={"replacement": "/knowledge/import-vault/runs"}, level="warn")
    return list_import_vault_runs(limit)


@router.get("/import-vault/runs/{run_id}/logs")
def get_import_vault_logs(run_id: str, limit: int = 200):
    return supabase_client.get_sync_logs(run_id, limit)


@router.get("/sync/runs/{run_id}/logs", deprecated=True)
def get_sync_logs(run_id: str, limit: int = 200):
    emit("deprecated_route_used", entity_type="route", entity_id="/knowledge/sync/runs/{run_id}/logs", payload={"replacement": "/knowledge/import-vault/runs/{run_id}/logs"}, level="warn")
    return get_import_vault_logs(run_id, limit)


# â”€â”€ File serve (for asset preview) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.get("/file")
def serve_vault_file(path: str):
    """Serve a file from Storage (`bucket:path`) or the local vault."""
    if ":" in path and not path.startswith(("/", "\\")):
        bucket, object_path = path.split(":", 1)
        bucket = bucket.strip()
        object_path = object_path.strip().lstrip("/\\")
        allowed_buckets = {"assets-raw", "assets-derived", "knowledge"}
        if bucket not in allowed_buckets or not object_path or ".." in object_path.replace("\\", "/").split("/"):
            raise HTTPException(403, "Access denied")
        try:
            data = supabase_client.download_from_storage(bucket, object_path)
        except Exception:
            raise HTTPException(404, "File not found")
        media_type = mimetypes.guess_type(object_path)[0] or "application/octet-stream"
        return Response(
            content=data,
            media_type=media_type,
            headers={"Cache-Control": "private, max-age=300"},
        )

    """Serve a file from the vault. Only available in local mode."""
    if VAULT_SOURCE_MODE != "local":
        raise HTTPException(
            status_code=501,
            detail="File serving from vault is only supported in VAULT_SOURCE_MODE='local'."
        )
    if not OBSIDIAN_LOCAL_PATH:
        raise HTTPException(status_code=500, detail="OBSIDIAN_LOCAL_PATH is not set.")

    from pathlib import Path
    vault_root = Path(OBSIDIAN_LOCAL_PATH).resolve()
    requested = (vault_root / path).resolve()
    # Security: prevent path traversal
    if not str(requested).startswith(str(vault_root)):
        raise HTTPException(403, "Access denied")
    if not requested.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(requested))


# â”€â”€ Knowledge Queue â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Statuses that require human attention before content can be used
ATTENTION_STATUSES = ["needs_persona", "needs_category", "pending"]

# Fixed paths MUST be registered before parameterised /{item_id}
@router.get("/queue")
def list_queue(
    request: Request,
    status: str = Query("pending"),
    persona_id: str = Query(None),
    content_type: str = Query(None),
    limit: int = 100,
    offset: int = 0,
):
    if persona_id:
        auth_service.assert_persona_access(request, persona_id=persona_id)
    elif not auth_service.is_admin(auth_service.current_user(request)):
        rows: list[dict] = []
        for pid in auth_service.allowed_persona_ids(request):
            rows.extend(list_queue(request, status=status, persona_id=pid, content_type=content_type, limit=limit, offset=offset))
        return rows[:limit]
    # "attention" is a virtual combined filter
    if status == "attention":
        return supabase_client.get_knowledge_items_multi(
            statuses=ATTENTION_STATUSES,
            persona_id=persona_id,
            content_type=content_type,
            limit=limit,
            offset=offset,
        )
    return supabase_client.get_knowledge_items(
        status=status,
        persona_id=persona_id,
        content_type=content_type,
        limit=limit,
        offset=offset,
    )


@router.get("/queue/counts")
def queue_counts(request: Request, persona_id: str = Query(None)):
    if persona_id:
        auth_service.assert_persona_access(request, persona_id=persona_id)
    elif not auth_service.is_admin(auth_service.current_user(request)):
        combined = {"by_status": {}, "total": 0}
        for pid in auth_service.allowed_persona_ids(request):
            partial = queue_counts(request, persona_id=pid)
            combined["total"] += partial.get("total", 0)
            for key, value in (partial.get("by_status") or {}).items():
                combined["by_status"][key] = combined["by_status"].get(key, 0) + value
        return combined
    counts = supabase_client.get_knowledge_item_counts(persona_id=persona_id)
    bs = counts.get("by_status", {})
    counts["by_status"]["attention"] = sum(
        bs.get(s, 0) for s in ATTENTION_STATUSES
    )
    return counts


@router.get("/gallery-assets")
def gallery_assets(request: Request, persona_id: str = Query(None), limit: int = Query(250, ge=1, le=500)):
    if persona_id:
        auth_service.assert_persona_access(request, persona_id=persona_id)
    elif not auth_service.is_admin(auth_service.current_user(request)):
        rows: list[dict] = []
        for pid in auth_service.allowed_persona_ids(request):
            rows.extend(supabase_client.list_gallery_assets(persona_id=pid, limit=limit))
        return rows[:limit]
    return supabase_client.list_gallery_assets(persona_id=persona_id, limit=limit)


class GenerateFaqsBody(BaseModel):
    max_questions: int = 8


@router.post("/graph/{node_id}/generate-faqs")
def generate_faqs_for_graph_node(node_id: str, body: GenerateFaqsBody, request: Request):
    """Sidebar action on the Graph screen: generate real FAQ content for an
    already-published node's branch, outside any Sofia chat session.

    Builds the same GraphBundle + PublicationPlan shape a Sofia session
    would produce, and stores it as a synthetic kb-intake session so the
    existing `POST /kb-intake/{session_id}/approve-publication` gate (same
    checksum/breaking-change discipline, same dashboard approval flow) is
    the only path that ever activates it -- this route only proposes."""
    from services import faq_bulk_generator, graph_bundle, graph_compiler_v3
    from services.kb_intake_service import _current_persona_base_bundle, _save_session
    import uuid as _uuid

    node = supabase_client.get_knowledge_node(node_id)
    if not node:
        raise HTTPException(404, "Node not found")
    persona_id = str(node.get("persona_id") or "")
    if not persona_id:
        raise HTTPException(400, "Node has no persona_id")
    auth_service.assert_persona_access(request, persona_id=persona_id)
    persona = supabase_client.get_persona_by_id(persona_id)
    if not persona:
        raise HTTPException(404, "Persona not found")
    persona_slug = str(persona.get("slug") or "")

    node_rows, edge_rows = supabase_client.list_all_knowledge_graph(persona_id=persona_id, limit_nodes=10000)
    chain = faq_bulk_generator.build_chain_from_live_graph(node_rows, edge_rows, node_id)
    if not chain:
        raise HTTPException(400, "Could not resolve this node's branch in the live graph")
    pairs = faq_bulk_generator.generate_faqs_for_chain(chain, max_questions=body.max_questions)
    if not pairs:
        return {"ok": False, "faqs": [], "error": "FAQ generation produced no usable output"}

    base = _current_persona_base_bundle(persona_id, persona_slug)
    if base is None:
        raise HTTPException(409, "Persona has no live GraphBundle-materialized graph yet")
    metadata = (node.get("metadata") or {})
    parent_bundle_id = str(metadata.get("graph_json_node_id") or f"{node.get('node_type')}:{node.get('slug')}")
    base_node_ids = {n["id"] for n in base["nodes"]}
    new_nodes = []
    new_edges = []
    for i, pair in enumerate(pairs, start=1):
        faq_id = f"faq:{node.get('slug')}-auto-{i}"
        suffix = 1
        while faq_id in base_node_ids:
            suffix += 1
            faq_id = f"faq:{node.get('slug')}-auto-{i}-{suffix}"
        base_node_ids.add(faq_id)
        new_nodes.append({
            "id": faq_id, "node_type": "faq", "slug": faq_id.split(":", 1)[1],
            "title": pair["question"], "summary": pair["answer"], "tags": ["faq", "auto-from-branch"],
            # Unlike Sofia's chat path (where a "pendente_validacao" entry is
            # held back from the bundle until the operator confirms it in a
            # follow-up turn), this route has no such follow-up turn -- the
            # `faqs` list in the response IS the review surface, and
            # approve-publication is the confirmation gate. Marking these
            # "pending_validation" here would make build_publication_plan
            # reject the whole bundle with no way to ever un-block it.
            "status": "validated",
            "data": {"question": pair["question"], "answer": pair["answer"], "source": "knowledge_graph_sidebar_generate_faqs"},
        })
        new_edges.append({
            "id": f"edge:{parent_bundle_id}->{faq_id}", "source": parent_bundle_id, "target": faq_id,
            "relation_type": "contains", "weight": 1.0, "metadata": {},
        })

    bundle = {
        "bundle_version": "1.0",
        "persona": {"id": persona_id, "slug": persona_slug},
        "metadata": {
            "purpose": f"faq_bulk_generation_{persona_slug}",
            "source": "knowledge_graph_sidebar_generate_faqs",
            "publication_allowed": True,
            "embedding_profile": {
                "embedding_provider": "local",
                "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "embedding_dimension": graph_compiler_v3.EMBEDDING_DIMENSION,
            },
        },
        "nodes": base["nodes"] + new_nodes,
        "edges": base["edges"] + new_edges,
    }
    from services.graph_bundle_adapter import ensure_branch_reachability
    bundle = ensure_branch_reachability(bundle)
    current_document = graph_compiler_v3.compile_graph(
        persona=persona, node_rows=node_rows, edge_rows=edge_rows,
        embedding_profile={
            "embedding_provider": "local",
            "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "embedding_dimension": graph_compiler_v3.EMBEDDING_DIMENSION,
        },
    )
    plan = graph_bundle.build_publication_plan(bundle, current_document=current_document, next_version=1)

    user = auth_service.current_user(request)
    session_id = f"faq-bulk-{_uuid.uuid4().hex}"
    session = {
        "id": session_id, "user_id": user.get("id"), "persona_id": persona_id, "persona_slug": persona_slug,
        "stage": "awaiting_publication_approval" if not plan.get("validation_errors") else "blocked",
        "status": "pending_approval",
        "pending_graph_bundle": bundle,
        "pending_publication_plan": plan,
        "source": "knowledge_graph_sidebar_generate_faqs",
    }
    _save_session(session)

    from services.graph_bundle_error_translations import translate_errors

    return {
        "ok": not plan.get("validation_errors"),
        "session_id": session_id,
        "faqs": pairs,
        "publication_plan": plan,
        "translated_errors": translate_errors(plan.get("validation_errors") or []),
    }


@router.get("/queue/{item_id}")
def get_queue_item(item_id: str, request: Request):
    try:
        item = supabase_client.get_knowledge_item(item_id)
    except Exception as exc:
        raise HTTPException(502, f"Database error: {exc}") from exc
    if not item:
        raise HTTPException(404, "Item not found")
    if item.get("persona_id"):
        auth_service.assert_persona_access(request, persona_id=item.get("persona_id"))
    return item


class ItemUpdate(BaseModel):
    persona_id: Optional[str] = None
    content_type: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    status: Optional[str] = None
    rejected_reason: Optional[str] = None
    tags: Optional[list] = None
    agent_visibility: Optional[list] = None
    asset_type: Optional[str] = None
    asset_function: Optional[str] = None


_FAQ_REVERT_STATUSES = {"approved", "embedded"}
_FAQ_CONTENT_FIELDS = ("title", "content")


def _should_revert_faq_to_draft(existing: Optional[dict], data: dict) -> bool:
    """An already-approved/embedded FAQ whose content is edited must go back to
    draft (rascunho) so it is re-approved and re-published before it can live in
    Embedded again. A patch that explicitly sets `status` is left untouched."""
    if not existing or "status" in data:
        return False
    if str(existing.get("content_type") or "").lower() != "faq":
        return False
    if str(existing.get("status") or "").lower() not in _FAQ_REVERT_STATUSES:
        return False
    return any(
        field in data and data[field] is not None and data[field] != existing.get(field)
        for field in _FAQ_CONTENT_FIELDS
    )


@router.patch("/queue/{item_id}")
def update_queue_item(item_id: str, body: ItemUpdate, request: Request):
    data = body.model_dump(exclude_none=True)
    if not data:
        raise HTTPException(400, "Nothing to update")
    try:
        # Auto-upgrade status based on what's being filled in
        cached_item: Optional[dict] = None
        existing = supabase_client.get_knowledge_item(item_id)
        if existing and existing.get("persona_id"):
            auth_service.assert_persona_access(request, persona_id=existing.get("persona_id"))
        if data.get("persona_id"):
            auth_service.assert_persona_access(request, persona_id=data.get("persona_id"))

        if "persona_id" in data:
            cached_item = existing
            if cached_item and cached_item.get("status") == "needs_persona":
                ct = data.get("content_type") or cached_item.get("content_type", "other")
                data["status"] = "pending"

        if "content_type" in data and data["content_type"] != "other":
            if cached_item is None:
                cached_item = supabase_client.get_knowledge_item(item_id)
            if cached_item and cached_item.get("status") == "needs_category":
                data["status"] = "pending"

        # Editing an approved/embedded FAQ document withdraws it from Embedded and
        # sends it back to draft until it is re-approved and re-published.
        reverted_faq = _should_revert_faq_to_draft(existing, data)
        if reverted_faq:
            data["status"] = "pending"
            data["curation_status"] = "draft"

        supabase_client.update_knowledge_item(item_id, data)
        if reverted_faq:
            try:
                supabase_client.withdraw_faq_from_embedded(item_id)
                # The withdrawn FAQ must drop out of the Embedded body too.
                from services import embedded_markdown
                embedded_markdown.rebuild_embedded_markdown((existing or {}).get("persona_id"))
            except Exception as exc:
                emit(
                    "faq_withdraw_from_embedded_failed",
                    entity_type="knowledge_item",
                    entity_id=item_id,
                    persona_id=(existing or {}).get("persona_id"),
                    payload={"error": str(exc)},
                    level="warn",
                    source="routes.knowledge",
                )
        updated = supabase_client.get_knowledge_item(item_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Database error: {exc}") from exc

    if not updated:
        raise HTTPException(404, "Item not found after update")

    before_view = {k: (existing or {}).get(k) for k in data.keys()}
    after_view = {k: (updated or {}).get(k) for k in data.keys()}
    emit(
        "knowledge_item_updated",
        entity_type="knowledge_item",
        entity_id=item_id,
        persona_id=(updated or {}).get("persona_id") or (existing or {}).get("persona_id"),
        payload={
            "actor": current_actor(request),
            "before": before_view,
            "after": after_view,
            "diff": summarize_diff(before_view, after_view),
            "context": {"content_type": (updated or {}).get("content_type")},
        },
        source="routes.knowledge",
    )
    return updated


class ApproveBody(BaseModel):
    promote_to_kb: bool = False
    # Deprecated compatibility input. Golden Dataset visibility is determined
    # exclusively by persona authorization.
    agent_visibility: Optional[list] = None


def _ensure_promotion_evidence(evidence: dict, *, require_embed: bool) -> None:
    if not evidence.get("knowledge_item_id"):
        raise HTTPException(502, "Approve/promote failed: missing knowledge_item_id confirmation")
    if not evidence.get("knowledge_node_id"):
        raise HTTPException(502, "Approve/promote failed: missing knowledge_node_id confirmation")
    if require_embed and not evidence.get("kb_entry_id"):
        raise HTTPException(502, "Approve/promote failed: missing kb_entry_id confirmation")
    if require_embed and not evidence.get("embedded_edge_id"):
        raise HTTPException(502, "Approve/promote failed: missing embedded_edge_id confirmation")


@router.post("/queue/{item_id}/approve")
def approve_item(item_id: str, request: Request, body: ApproveBody = ApproveBody()):
    try:
        item = supabase_client.get_knowledge_item(item_id)
        if not item:
            raise HTTPException(404, "Item not found")
        if not item.get("persona_id"):
            raise HTTPException(400, "Assign a persona before approving")
        auth_service.assert_persona_access(request, persona_id=item.get("persona_id"))
        result = knowledge_lifecycle.promote_knowledge_item(
            item_id,
            promote_to_kb=False,
            agent_visibility=None,
            approval_mode="manual_validation",
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Approve/promote failed: {exc}") from exc

    updated_item = result.get("item") or supabase_client.get_knowledge_item(item_id)
    evidence = result.get("evidence") or {}
    _ensure_promotion_evidence(evidence, require_embed=body.promote_to_kb)
    publication_evidence = {}
    try:
        user = auth_service.current_user(request)
        if str(updated_item.get("content_type") or "").lower() == "faq":
            persona = supabase_client.get_persona_by_id(updated_item.get("persona_id")) or {}
            persona_slug = persona.get("slug")
            current_doc = graph_json_v2_store.load_current(persona_slug) if persona_slug else None
            if current_doc is None:
                raise RuntimeError("FAQ approval requires a published canonical Graph JSON")
            current_version, graph = current_doc
            next_graph = GraphJson.model_validate(graph.model_dump())
            semantic_node = supabase_client.get_knowledge_node(evidence.get("knowledge_node_id")) or {}
            faq_doc = next(
                (
                    node
                    for node in next_graph.nodes
                    if node.node_type == "faq"
                    and (
                        str((node.data or {}).get("knowledge_item_id") or "") == item_id
                        or str((node.data or {}).get("knowledge_node_id") or "") == str(evidence.get("knowledge_node_id") or "")
                        or (
                            semantic_node.get("slug")
                            and node.slug == semantic_node.get("slug")
                        )
                    )
                ),
                None,
            )
            embedded_doc = next((node for node in next_graph.nodes if node.node_type == "embedded"), None)
            if not faq_doc or not embedded_doc:
                raise RuntimeError("Canonical FAQ or Embedded node was not found")
            faq_doc.data = {**(faq_doc.data or {}), "status": "approved", "validation_status": "approved"}
            next_graph.edges = [
                edge
                for edge in next_graph.edges
                if not (edge.source == faq_doc.id and edge.target == embedded_doc.id)
            ]
            next_graph.edges.append(
                Edge(
                    id=f"edge:faq-embedded:{faq_doc.slug}",
                    source=faq_doc.id,
                    target=embedded_doc.id,
                    relation="visible_to_agent",
                    primary_tree=False,
                    metadata={"created_from": "atomic_faq_approval", "active": True},
                )
            )
            canonical_publication = graph_document_publisher.publish(
                graph=next_graph,
                persona_slug=persona_slug,
                brand_slug=next_graph.brand_slug,
                source="knowledge.queue.approve",
                published_by=user.get("id"),
                expected_version=current_version,
                idempotency_key=f"faq-approval:{item_id}:{current_version}",
            )
            faq_publications = (canonical_publication.get("projections") or {}).get("faq_publications") or []
            publication_evidence = {
                **(faq_publications[0] if faq_publications else {}),
                "graph_version": canonical_publication.get("version"),
                "graph_checksum": canonical_publication.get("checksum"),
            }
        else:
            publication_evidence = approved_knowledge_snapshots.publish_approved_node(
                evidence.get("knowledge_node_id"),
                approved_by=user.get("id"),
                require_rag_for_faq=False,
            )
    except Exception as exc:
        # Approval is atomic from the operator's perspective: incomplete
        # snapshot/RAG/Embedded publication returns the FAQ to validation.
        try:
            supabase_client.update_knowledge_item(
                item_id,
                {"status": "pending", "curation_status": "draft"},
            )
            if evidence.get("knowledge_node_id"):
                supabase_client.update_knowledge_node(
                    evidence["knowledge_node_id"],
                    {"status": "pending"},
                    mark_related_faqs=False,
                )
            supabase_client.withdraw_faq_from_embedded(item_id)
        except Exception:
            pass
        partial_ids = {
            "knowledge_item_id": evidence.get("knowledge_item_id"),
            "source_node_id": evidence.get("knowledge_node_id"),
            "rag_entry_id": evidence.get("knowledge_rag_entry_id"),
            "rag_chunk_ids": evidence.get("knowledge_rag_chunk_ids") or [],
        }
        emit("item_approved_snapshot_failed", entity_type="knowledge_item", entity_id=item_id,
             persona_id=updated_item.get("persona_id"),
             payload={"title": updated_item.get("title"), "error": str(exc), "partial_ids": partial_ids})
        raise HTTPException(
            502,
            {
                "code": "FAQ_APPROVAL_ATOMIC_PUBLICATION_FAILED",
                "stage": "approved_snapshot_publication",
                "error": str(exc),
                "partial_ids": partial_ids,
            },
        ) from exc
    emit("item_approved", entity_type="knowledge_item", entity_id=item_id,
         persona_id=updated_item.get("persona_id"),
         payload={"title": updated_item.get("title"), "content_type": updated_item.get("content_type"),
                  "promoted_to_kb": body.promote_to_kb, "evidence": evidence,
                  "publication_evidence": publication_evidence})

    merged_evidence = {**evidence, **publication_evidence}
    return {
        "ok": True,
        "success": True,
        "item": updated_item,
        "evidence": merged_evidence,
        **publication_evidence,
    }


class RejectBody(BaseModel):
    reason: str = ""


@router.post("/queue/{item_id}/reject")
def reject_item(item_id: str, request: Request, body: RejectBody = RejectBody()):
    item = supabase_client.get_knowledge_item(item_id)
    if not item:
        raise HTTPException(404, "Item not found")
    if item.get("persona_id"):
        auth_service.assert_persona_access(request, persona_id=item.get("persona_id"))
    supabase_client.update_knowledge_item(item_id, {
        "status": "rejected",
        "rejected_reason": body.reason,
    })
    emit("item_rejected", entity_type="knowledge_item", entity_id=item_id,
         persona_id=item.get("persona_id"),
         payload={"title": item.get("title"), "reason": body.reason})
    return {"ok": True}


@router.delete("/queue/{item_id}")
def delete_queue_item(item_id: str, request: Request):
    try:
        item = supabase_client.get_knowledge_item(item_id)
        if not item:
            raise HTTPException(404, "Item not found")
        if item.get("persona_id"):
            auth_service.assert_persona_access(request, persona_id=item.get("persona_id"))
        evidence = knowledge_lifecycle.delete_knowledge_item_cascade(item_id)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Delete failed: {exc}") from exc
    emit(
        "knowledge_item_deleted",
        entity_type="knowledge_item",
        entity_id=item_id,
        persona_id=item.get("persona_id"),
        payload=evidence,
    )
    return {"ok": True, "evidence": evidence}


@router.post("/queue/{item_id}/to-kb")
def promote_to_kb(item_id: str, request: Request):
    item = supabase_client.get_knowledge_item(item_id)
    if not item:
        raise HTTPException(404, "Item not found")
    if item.get("persona_id"):
        auth_service.assert_persona_access(request, persona_id=item.get("persona_id"))
    raise HTTPException(
        409,
        "Publication now happens only through the graph: approve the FAQ first, then connect it to Embedded to publish it to the Golden Dataset.",
    )


# â”€â”€ Upload / Knowledge Intake â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class UploadTextBody(BaseModel):
    title: str
    content: str
    persona_id: Optional[str] = None
    content_type: str = "other"
    metadata: dict = {}


@router.post("/upload/text")
def upload_text(body: UploadTextBody, request: Request):
    if body.persona_id:
        auth_service.assert_persona_access(request, persona_id=body.persona_id)
    source = supabase_client.get_or_create_manual_source()
    status = "pending"
    item = supabase_client.insert_knowledge_item({
        "persona_id": body.persona_id,
        "source_id": source["id"],
        "status": status,
        "content_type": body.content_type,
        "title": body.title,
        "content": body.content,
        "metadata": body.metadata,
        "file_type": "text",
    })
    if item:
        knowledge_graph.bootstrap_from_item(
            item,
            frontmatter=body.metadata or {},
            body=body.content,
            persona_id=body.persona_id,
            source_table="knowledge_items",
        )
    emit("upload_received", entity_type="knowledge_item", entity_id=item["id"],
         persona_id=body.persona_id,
         payload={"title": body.title, "content_type": body.content_type})
    return item


@router.post("/upload/file")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    persona_id: str = Form(None),
    content_type: str = Form("other"),
):
    if persona_id:
        auth_service.assert_persona_access(request, persona_id=persona_id)
    content_bytes = await file.read()
    filename = file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mime = (file.content_type or "").lower()
    text_like = ext in {"txt", "md", "markdown", "json", "csv", "yaml", "yml"} or mime.startswith("text/") or mime in {"application/json", "application/x-yaml"}
    if not text_like:
        raise HTTPException(
            415,
            {
                "error": "binary_upload_unsupported",
                "message": (
                    "Este endpoint aceita somente arquivos de texto. "
                    "Para imagens, PDFs ou videos use /assets/upload "
                    "(card ASSET na aba 'Asset visual / Outro')."
                ),
                "use_endpoint": "/assets/upload",
                "filename": filename,
            },
        )
    text = ""
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = content_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if not text.strip() and content_bytes:
        raise HTTPException(
            415,
            {
                "error": "text_decode_failed",
                "message": "Nao consegui ler o arquivo como texto. Envie TXT, MD, JSON, CSV ou use /assets/upload para arquivos binarios.",
                "filename": filename,
            },
        )

    source = supabase_client.get_or_create_manual_source()
    status = "pending"
    item = supabase_client.insert_knowledge_item({
        "persona_id": persona_id,
        "source_id": source["id"],
        "status": status,
        "content_type": content_type,
        "title": filename,
        "content": text[:8000],
        "file_type": ext or "txt",
        "metadata": {"original_filename": filename, "mime": mime},
    })
    if item:
        knowledge_graph.bootstrap_from_item(
            item,
            frontmatter={"original_filename": filename},
            body=text,
            persona_id=persona_id,
            source_table="knowledge_items",
        )
    emit("upload_received", entity_type="knowledge_item", entity_id=item["id"],
         persona_id=persona_id, payload={"filename": file.filename})
    return item


# â”€â”€ KB Entries (Vault) â€” single-item CRUD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.get("/kb/{entry_id}")
def get_kb_entry(entry_id: str, request: Request):
    entry = supabase_client.get_kb_entry(entry_id)
    if not entry:
        raise HTTPException(404, "Entry not found")
    if entry.get("persona_id"):
        auth_service.assert_persona_access(request, persona_id=entry.get("persona_id"))
    return entry


class KbEntryUpdate(BaseModel):
    titulo: Optional[str] = None
    conteudo: Optional[str] = None
    tipo: Optional[str] = None
    categoria: Optional[str] = None
    tags: Optional[list] = None
    status: Optional[str] = None
    agent_visibility: Optional[list] = None


@router.patch("/kb/{entry_id}")
def update_kb_entry(entry_id: str, body: KbEntryUpdate, request: Request):
    data = body.model_dump(exclude_none=True)
    if not data:
        raise HTTPException(400, "Nothing to update")
    entry = supabase_client.get_kb_entry(entry_id)
    if not entry:
        raise HTTPException(404, "Entry not found")
    if entry.get("persona_id"):
        auth_service.assert_persona_access(request, persona_id=entry.get("persona_id"))
    supabase_client.update_kb_entry(entry_id, data)
    entry = supabase_client.get_kb_entry(entry_id)
    emit("kb_entry_updated", entity_type="kb_entry", entity_id=entry_id,
         persona_id=entry.get("persona_id"),
         payload={"titulo": entry.get("titulo"), "fields": list(data.keys())})
    return entry


@router.post("/kb/{entry_id}/validate")
def validate_kb_entry(entry_id: str, request: Request):
    entry = supabase_client.get_kb_entry(entry_id)
    if not entry:
        raise HTTPException(404, "Entry not found")
    if entry.get("persona_id"):
        auth_service.assert_persona_access(request, persona_id=entry.get("persona_id"))
    supabase_client.update_kb_entry(entry_id, {"status": "ATIVO"})
    emit("kb_entry_validated", entity_type="kb_entry", entity_id=entry_id,
         persona_id=entry.get("persona_id"),
         payload={"titulo": entry.get("titulo")})
    return {"ok": True, "status": "ATIVO"}


# â”€â”€ Workflow Bindings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.get("/bindings")
def list_bindings(request: Request, persona_id: str = Query(None)):
    if persona_id:
        auth_service.assert_persona_access(request, persona_id=persona_id)
    return supabase_client.get_workflow_bindings(persona_id)


class BindingBody(BaseModel):
    persona_id: str
    workflow_name: str
    n8n_workflow_id: Optional[str] = None
    whatsapp_number: Optional[str] = None
    active: bool = True


@router.post("/bindings")
def create_binding(body: BindingBody, request: Request):
    if not auth_service.is_admin(auth_service.current_user(request)):
        raise HTTPException(403, "Apenas admin pode criar bindings")
    return supabase_client.upsert_workflow_binding(body.model_dump())


# Products / product collections live in knowledge_nodes and knowledge_edges.
def _slugify_local(value: str) -> str:
    return supabase_client._slugify(value)


def _resolve_persona_scope(request: Request, persona_id: Optional[str] = None, persona_slug: Optional[str] = None) -> tuple[Optional[str], Optional[dict]]:
    if persona_id:
        persona = supabase_client.get_persona_by_id(persona_id)
        if not persona:
            raise HTTPException(404, "Persona not found")
        auth_service.assert_persona_access(request, persona_id=persona_id)
        return persona_id, persona
    if persona_slug:
        persona = supabase_client.get_persona(persona_slug)
        if not persona:
            raise HTTPException(404, "Persona not found")
        auth_service.assert_persona_access(request, persona_id=persona.get("id"), persona_slug=persona_slug)
        return persona.get("id"), persona
    if auth_service.is_admin(auth_service.current_user(request)):
        return None, None
    allowed = auth_service.allowed_persona_ids(request)
    if len(allowed) == 1:
        return allowed[0], supabase_client.get_persona_by_id(allowed[0])
    raise HTTPException(400, "persona_id or persona_slug is required")


def _compose_product_markdown(product: dict, category: Optional[dict], collection: Optional[dict], assets: list[dict]) -> str:
    metadata = product.get("metadata") or {}
    tags = product.get("tags") or []
    lines = [f"# {product.get('title') or product.get('slug')}", "", product.get("summary") or "", "", "## Tags"]
    lines.extend([f"- {tag}" for tag in tags] or ["-"])
    lines.extend(["", "## Categoria", (category or {}).get("title") or metadata.get("category_slug") or "-", "", "## Colecao", (collection or {}).get("title") or metadata.get("collection_slug") or "-", "", "## Assets"])
    lines.extend([f"- {a.get('title') or a.get('slug')} ({a.get('slug') or a.get('id')})" for a in assets] or ["-"])
    lines.extend(["", "## Metadata", "```json", __import__("json").dumps(metadata, ensure_ascii=False, indent=2), "```"])
    return "\n".join(lines).strip() + "\n"


def _decorate_products(products: list[dict]) -> list[dict]:
    product_ids = [p["id"] for p in products if p.get("id")]
    edges = supabase_client.list_edges_for_nodes(product_ids, relation_types=["category_has_product", "in_category", "part_of_collection", "product_image", "product_has_asset"])
    related_ids = set(product_ids)
    for edge in edges:
        related_ids.add(edge.get("source_node_id"))
        related_ids.add(edge.get("target_node_id"))
    nodes_by_id = {row["id"]: row for row in supabase_client.list_knowledge_nodes_by_ids(list(related_ids)) if row.get("id")}
    out = []
    for product in products:
        pid = product.get("id")
        category = None
        collection = None
        assets = []
        product_edges = []
        for edge in edges:
            if edge.get("source_node_id") != pid and edge.get("target_node_id") != pid:
                continue
            product_edges.append(edge)
            rt = edge.get("relation_type")
            other_id = edge.get("source_node_id") if edge.get("target_node_id") == pid else edge.get("target_node_id")
            other = nodes_by_id.get(other_id) or {}
            if rt in {"category_has_product", "in_category"} and other.get("node_type") == "category":
                category = other
            elif rt == "part_of_collection" and other.get("node_type") == "product_collection":
                collection = other
            elif rt in {"product_image", "product_has_asset"} and other.get("node_type") == "asset":
                assets.append(other)
        metadata = product.get("metadata") or {}
        first_asset_meta = (assets[0].get("metadata") or {}) if assets else {}
        out.append({
            **product,
            "collection": collection,
            "category": category,
            "assets": assets,
            "edges": product_edges,
            "thumbnail": first_asset_meta.get("url") or first_asset_meta.get("file_path"),
            "markdown": _compose_product_markdown(product, category, collection, assets),
            "collection_slug": metadata.get("collection_slug") or (collection or {}).get("slug"),
            "category_slug": metadata.get("category_slug") or (category or {}).get("slug"),
        })
    return out


@router.get("/product-collections")
def list_product_collections(request: Request, persona_slug: Optional[str] = Query(None), persona_id: Optional[str] = Query(None)):
    resolved_persona_id, _ = _resolve_persona_scope(request, persona_id=persona_id, persona_slug=persona_slug)
    return supabase_client.list_product_collection_nodes(persona_id=resolved_persona_id, node_type="product_collection")


@router.get("/categories")
def list_product_categories(request: Request, persona_slug: Optional[str] = Query(None), persona_id: Optional[str] = Query(None), collection_slug: Optional[str] = Query(None)):
    resolved_persona_id, _ = _resolve_persona_scope(request, persona_id=persona_id, persona_slug=persona_slug)
    rows = supabase_client.list_product_collection_nodes(persona_id=resolved_persona_id, node_type="category")
    if collection_slug:
        rows = [row for row in rows if (row.get("metadata") or {}).get("collection_slug") == collection_slug]
    return rows


@router.get("/products")
def list_products(request: Request, persona_slug: Optional[str] = Query(None), persona_id: Optional[str] = Query(None), collection_slug: Optional[str] = Query(None), category_slug: Optional[str] = Query(None), status: Optional[str] = Query(None)):
    resolved_persona_id, _ = _resolve_persona_scope(request, persona_id=persona_id, persona_slug=persona_slug)
    products = supabase_client.list_product_nodes(persona_id=resolved_persona_id, collection_slug=collection_slug, category_slug=category_slug, status=status)
    return _decorate_products(products)


@router.post("/products")
def create_product(body: ProductBody, request: Request):
    persona_id, persona = _resolve_persona_scope(request, persona_id=body.persona_id, persona_slug=body.persona_slug)
    if not persona_id:
        raise HTTPException(400, "persona_id or persona_slug is required")
    slug = _slugify_local(body.slug or body.title)
    metadata = {**(body.metadata or {}), "persona_slug": (persona or {}).get("slug"), "collection_slug": body.collection_slug, "category_slug": body.category_slug, "source": (body.metadata or {}).get("source") or "manual", "open_url": f"/marketing/produtos?product={slug}"}
    metadata = {k: v for k, v in metadata.items() if v is not None}
    product = supabase_client.upsert_knowledge_node({"persona_id": persona_id, "node_type": "product", "slug": slug, "title": body.title, "summary": body.summary, "tags": body.tags or [], "metadata": metadata, "status": body.status or "pending_validation"})
    if not product:
        raise HTTPException(502, "Could not create product node")
    if body.category_slug:
        category = supabase_client.get_knowledge_node_by_slug(body.category_slug, persona_id=persona_id, node_type="category")
        if category:
            supabase_client.upsert_knowledge_edge(category["id"], product["id"], "category_has_product", persona_id=persona_id, weight=0.86, metadata={"primary_tree": True, "created_from": "products_api"})
    if body.collection_slug:
        collection = supabase_client.get_knowledge_node_by_slug(body.collection_slug, persona_id=persona_id, node_type="product_collection")
        if collection:
            supabase_client.upsert_knowledge_edge(product["id"], collection["id"], "part_of_collection", persona_id=persona_id, weight=0.7, metadata={"primary_tree": False, "created_from": "products_api"})
    emit("product_node_created", entity_type="knowledge_node", entity_id=product.get("id"), persona_id=persona_id, payload={"slug": slug, "title": body.title})
    return _decorate_products([product])[0]


class ProductImportPreviewBody(BaseModel):
    provider: str
    persona_id: Optional[str] = None
    persona_slug: Optional[str] = None
    config: Optional[dict] = None


@router.post("/products/import/preview")
def import_products_preview_route(body: ProductImportPreviewBody, request: Request):
    """Crawl/list products WITHOUT importing, grouped by collection.

    Powers the audit screen so the operator can toggle collections/products
    before confirming. No nodes are created."""
    persona_id, _persona = _resolve_persona_scope(request, persona_id=body.persona_id, persona_slug=body.persona_slug)
    if not persona_id:
        raise HTTPException(400, "persona_id or persona_slug is required")
    provider = (body.provider or "").strip().lower()
    config = dict(body.config or {})
    try:
        return product_import_service.preview_products(provider=provider, config=config)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class ProductImportBody(BaseModel):
    provider: str
    persona_id: Optional[str] = None
    persona_slug: Optional[str] = None
    config: Optional[dict] = None
    items: Optional[list[dict]] = None
    download_images: Optional[bool] = None


@router.post("/products/import")
def import_products_route(body: ProductImportBody, request: Request):
    """Import products from Meta / Shopify / Scraper(mock) into pending nodes.

    The response NEVER includes credentials. Meta credentials are read
    decrypted from the per-user integration (configured in Tools). When `items`
    is provided (audit confirmation), only those products are imported."""
    persona_id, persona = _resolve_persona_scope(request, persona_id=body.persona_id, persona_slug=body.persona_slug)
    if not persona_id:
        raise HTTPException(400, "persona_id or persona_slug is required")
    provider = (body.provider or "").strip().lower()
    config = dict(body.config or {})
    if provider == "meta" and not body.items:
        user = auth_service.current_user(request) or {}
        config = {**config, **integration_service.get_meta_credentials(user.get("id") or "")}
    try:
        result = product_import_service.import_products(
            provider=provider,
            persona_id=persona_id,
            persona_slug=(persona or {}).get("slug"),
            config=config,
            items=body.items,
            download_images=bool(body.download_images),
        )
    except integration_service.IntegrationValidationError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    emit("products_imported", entity_type="persona", entity_id=persona_id, persona_id=persona_id,
         payload={"provider": provider, "created": result["created"], "updated": result["updated"],
                  "skipped": result["skipped"], "images_downloaded": result.get("images_downloaded", 0)})
    return result


@router.post("/products/import/csv")
async def import_products_csv_route(
    request: Request,
    file: UploadFile = File(...),
    persona_id: Optional[str] = Form(None),
    persona_slug: Optional[str] = Form(None),
):
    resolved_persona_id, persona = _resolve_persona_scope(request, persona_id=persona_id, persona_slug=persona_slug)
    if not resolved_persona_id:
        raise HTTPException(400, "persona_id or persona_slug is required")
    content = await file.read()
    try:
        result = product_import_service.import_products(
            provider="csv",
            persona_id=resolved_persona_id,
            persona_slug=(persona or {}).get("slug"),
            file_bytes=content,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    emit("products_imported", entity_type="persona", entity_id=resolved_persona_id, persona_id=resolved_persona_id,
         payload={"provider": "csv", "created": result["created"], "updated": result["updated"], "skipped": result["skipped"]})
    return result


@router.get("/products/{slug}")
def get_product(slug: str, request: Request, persona_slug: Optional[str] = Query(None), persona_id: Optional[str] = Query(None)):
    resolved_persona_id, _ = _resolve_persona_scope(request, persona_id=persona_id, persona_slug=persona_slug)
    product = supabase_client.get_knowledge_node_by_slug(slug, persona_id=resolved_persona_id, node_type="product")
    if not product:
        raise HTTPException(404, "Product not found")
    if product.get("persona_id"):
        auth_service.assert_persona_access(request, persona_id=product.get("persona_id"))
    return _decorate_products([product])[0]


@router.patch("/products/{slug}")
def update_product(slug: str, body: ProductPatchBody, request: Request, persona_slug: Optional[str] = Query(None), persona_id: Optional[str] = Query(None)):
    resolved_persona_id, _ = _resolve_persona_scope(request, persona_id=persona_id, persona_slug=persona_slug)
    product = supabase_client.get_knowledge_node_by_slug(slug, persona_id=resolved_persona_id, node_type="product")
    if not product:
        raise HTTPException(404, "Product not found")
    auth_service.assert_persona_access(request, persona_id=product.get("persona_id"))
    patch = body.model_dump(exclude_none=True)
    if "metadata" in patch:
        patch["metadata"] = {**(product.get("metadata") or {}), **(patch["metadata"] or {})}
    before_view = {k: product.get(k) for k in patch.keys()}
    updated = supabase_client.update_knowledge_node(product["id"], patch)
    after_view = {k: (updated or {**product, **patch}).get(k) for k in patch.keys()}
    emit(
        "product_node_updated",
        entity_type="knowledge_node",
        entity_id=product.get("id"),
        persona_id=product.get("persona_id"),
        payload={
            "actor": current_actor(request),
            "before": before_view,
            "after": after_view,
            "diff": summarize_diff(before_view, after_view),
            "context": {"slug": slug},
        },
        source="routes.knowledge",
    )
    return _decorate_products([updated or {**product, **patch}])[0]


@router.post("/products/{slug}/approve")
def approve_product(slug: str, request: Request, persona_slug: Optional[str] = Query(None), persona_id: Optional[str] = Query(None)):
    resolved_persona_id, _ = _resolve_persona_scope(request, persona_id=persona_id, persona_slug=persona_slug)
    product = supabase_client.get_knowledge_node_by_slug(slug, persona_id=resolved_persona_id, node_type="product")
    if not product:
        raise HTTPException(404, "Product not found")
    auth_service.assert_persona_access(request, persona_id=product.get("persona_id"))
    metadata = {**(product.get("metadata") or {}), "validated_at": datetime.now(timezone.utc).isoformat(), "validated_by": (auth_service.current_user(request) or {}).get("id")}
    updated = supabase_client.update_knowledge_node(
        product["id"],
        {"status": "validated", "metadata": metadata},
        mark_related_faqs=False,
    )
    emit("product_node_approved", entity_type="knowledge_node", entity_id=product.get("id"), persona_id=product.get("persona_id"), payload={"slug": slug})
    return {"ok": True, "product": _decorate_products([updated or product])[0]}


def _asset_node_from_link_body(body: LinkAssetBody, persona_id: str) -> Optional[dict]:
    if body.asset_node_id:
        return supabase_client.get_knowledge_node(body.asset_node_id[3:] if body.asset_node_id.startswith("gn:") else body.asset_node_id)
    if body.asset_id:
        return supabase_client.get_knowledge_node_for_source("assets", body.asset_id, persona_id=persona_id)
    return None


@router.post("/products/{slug}/link-asset")
def link_product_asset(slug: str, body: LinkAssetBody, request: Request, persona_slug: Optional[str] = Query(None), persona_id: Optional[str] = Query(None)):
    resolved_persona_id, _ = _resolve_persona_scope(request, persona_id=persona_id, persona_slug=persona_slug)
    product = supabase_client.get_knowledge_node_by_slug(slug, persona_id=resolved_persona_id, node_type="product")
    if not product:
        raise HTTPException(404, "Product not found")
    auth_service.assert_persona_access(request, persona_id=product.get("persona_id"))
    asset_node = _asset_node_from_link_body(body, product.get("persona_id"))
    if not asset_node or asset_node.get("node_type") != "asset":
        raise HTTPException(404, "Asset node not found")
    slot_meta = edge_metadata_for_slot(LandingSlot.PRODUCT_IMAGE, label=product.get("title") or product.get("slug"))
    binding = dict(slot_meta.get("page_binding") or {})
    binding["slot_key"] = f"{LandingSlot.PRODUCT_IMAGE.value}:{product['slug']}"
    binding["target_slug"] = product["slug"]
    slot_meta["page_binding"] = binding
    removed_edge_ids: list[str] = []
    for edge in supabase_client.list_edges_for_nodes([product["id"]], relation_types=["product_image", "product_has_asset"], limit=1000):
        if edge.get("source_node_id") == product["id"] or edge.get("target_node_id") == product["id"]:
            if supabase_client.delete_knowledge_edge(edge.get("id")):
                removed_edge_ids.append(edge.get("id"))
    metadata = {
        **(body.metadata or {}),
        **slot_meta,
        "status": "active",
        "proposed_by": (body.metadata or {}).get("proposed_by") or "manual",
        "created_from": "product_link_asset",
        "primary_tree": True,
        "direction": "product_to_asset",
        "parent_slug": product.get("slug"),
        "parent_type": "product",
    }
    edge = supabase_client.upsert_knowledge_edge(product["id"], asset_node["id"], body.relation_type or "product_image", persona_id=product.get("persona_id"), weight=0.85, metadata=metadata)
    emit(
        "product_asset_linked",
        entity_type="knowledge_edge",
        entity_id=(edge or {}).get("id") if isinstance(edge, dict) else None,
        persona_id=product.get("persona_id"),
        payload={
            "product_slug": product.get("slug"),
            "product_node_id": product.get("id"),
            "asset_node_id": asset_node.get("id"),
            "asset_id": body.asset_id,
            "relation_type": body.relation_type or "product_image",
            "slot_key": binding.get("slot_key"),
            "removed_previous_edge_ids": removed_edge_ids,
        },
        source="routes.knowledge",
    )
    asset = supabase_client.get_asset(body.asset_id) if body.asset_id else None
    if asset:
        asset_metadata = {
            **(asset.get("metadata") or {}),
            "asset_function": slot_config(LandingSlot.PRODUCT_IMAGE)["asset_function"],
            "asset_type": asset.get("type") or (asset.get("metadata") or {}).get("asset_type") or "image",
            "parent_node_id": product["id"],
            "parent_edge_id": edge.get("id") if isinstance(edge, dict) else None,
        }
        supabase_client.update_asset(body.asset_id, {"metadata": {k: v for k, v in asset_metadata.items() if v is not None}})
        try:
            supabase_client.update_knowledge_node(asset_node["id"], {"metadata": {**(asset_node.get("metadata") or {}), **asset_metadata}})
        except Exception:
            pass
    return {"ok": True, "edge": edge, "product": _decorate_products([product])[0]}


@router.post("/products/{slug}/sofia-suggest-images")
def sofia_suggest_product_images(slug: str, body: SofiaSuggestBody, request: Request, persona_slug: Optional[str] = Query(None), persona_id: Optional[str] = Query(None)):
    resolved_persona_id, _ = _resolve_persona_scope(request, persona_id=persona_id, persona_slug=persona_slug)
    product = supabase_client.get_knowledge_node_by_slug(slug, persona_id=resolved_persona_id, node_type="product")
    if not product:
        raise HTTPException(404, "Product not found")
    auth_service.assert_persona_access(request, persona_id=product.get("persona_id"))
    assets = supabase_client.list_product_collection_nodes(persona_id=product.get("persona_id"), node_type="asset", limit=500)
    existing_edges = supabase_client.list_edges_for_nodes([product["id"]], relation_types=["product_image", "product_has_asset"], limit=1000)
    linked_asset_ids = {edge.get("source_node_id") if edge.get("target_node_id") == product["id"] else edge.get("target_node_id") for edge in existing_edges}
    product_terms = {str(term).lower() for term in [product.get("slug"), product.get("title"), *(product.get("tags") or [])] if term}
    suggestions = []
    for asset in assets:
        if asset.get("id") in linked_asset_ids:
            continue
        haystack = " ".join([str(asset.get("slug") or ""), str(asset.get("title") or ""), " ".join(asset.get("tags") or []), str((asset.get("metadata") or {}).get("original_filename") or ""), str((asset.get("metadata") or {}).get("visual_summary") or "")]).lower()
        score = sum(1 for term in product_terms if term and term in haystack) / max(1, len(product_terms))
        if score < body.min_score:
            continue
        edge = supabase_client.upsert_knowledge_edge(asset["id"], product["id"], "product_image", persona_id=product.get("persona_id"), weight=max(0.4, min(0.95, score)), metadata={"status": "pending_validation", "proposed_by": "sofia", "created_from": "sofia_suggest_images", "score": score, "primary_tree": False})
        suggestions.append({"asset": asset, "edge": edge, "score": score})
        if len(suggestions) >= body.limit:
            break
    return {"ok": True, "suggestions": suggestions}


# â”€â”€ Knowledge Graph rebuild (admin) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.post("/graph/rebuild")
def rebuild_graph(request: Request, persona_slug: Optional[str] = Query(None)):
    if not auth_service.is_admin(auth_service.current_user(request)):
        raise HTTPException(403, "Apenas admin pode reconstruir o grafo")
    """Reprocessa knowledge_items + kb_entries existentes para popular
    knowledge_nodes / knowledge_edges (migration 008).

    Use apÃ³s aplicar 008 ou quando o grafo divergir das tabelas-fonte.

    Quando `persona_slug` Ã© informado, escopa pra essa persona; senÃ£o,
    roda globalmente (cuidado em prod com muitos clientes).

    Resposta:
      {persona_slug, persona_id, counts: {items_seen, items_mirrored,
       items_skipped, kb_seen, kb_mirrored, kb_skipped, errors[]}}
    """
    persona_id: Optional[str] = None
    if persona_slug:
        persona = supabase_client.get_persona(persona_slug)
        if not persona:
            raise HTTPException(404, f"Persona not found: {persona_slug}")
        persona_id = persona.get("id")

    counts = knowledge_graph.rebuild_graph_for_persona(persona_id)
    return {
        "persona_slug": persona_slug,
        "persona_id": persona_id,
        "counts": counts,
    }


# â”€â”€ Chat Context (semantic graph + KB fallback) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.get("/catalog")
def knowledge_catalog(
    request: Request,
    persona_id: Optional[str] = Query(None),
    persona_slug: Optional[str] = Query(None),
):
    from services import knowledge_catalog as catalog_service

    personas = supabase_client.get_personas() or []
    if persona_id or persona_slug:
        persona = next(
            (
                row for row in personas
                if (persona_id and row.get("id") == persona_id)
                or (persona_slug and row.get("slug") == persona_slug)
            ),
            None,
        )
        if not persona:
            raise HTTPException(404, "Persona not found")
        auth_service.assert_persona_access(
            request,
            persona_id=persona.get("id"),
            persona_slug=persona.get("slug"),
        )
        personas = [persona]
    elif not auth_service.is_admin(auth_service.current_user(request)):
        allowed = set(auth_service.allowed_persona_ids(request))
        personas = [row for row in personas if row.get("id") in allowed]

    catalogs = []
    for persona in personas:
        catalog = catalog_service.load_catalog(
            persona_slug=persona.get("slug"),
            persona_id=persona.get("id"),
            persona_name=persona.get("name"),
        )
        if catalog:
            catalogs.append(catalog)
    return {
        "catalogs": catalogs,
        "persona_count": len(catalogs),
        "document_count": sum(
            int((item.get("graph") or {}).get("document_count") or 0)
            for item in catalogs
        ),
    }


@router.get("/chat-context")
def chat_context(
    request: Request,
    lead_ref: int = Query(...),
    persona_id: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
    response_message_id: Optional[str] = Query(None),
    limit: int = Query(12, le=50),
):
    """Knowledge bundle for the messages sidebar.

    Resolves products/campaigns/assets/FAQs related to a lead's recent
    conversation (or to an explicit `q`). Falls back gracefully when the
    semantic graph has no data â€” always returns the same response shape.
    """
    if persona_id:
        auth_service.assert_persona_access(request, persona_id=persona_id)
    else:
        scoped_lead = supabase_client.get_lead_by_ref(lead_ref) or {}
        if scoped_lead.get("persona_id"):
            auth_service.assert_persona_access(
                request, persona_id=str(scoped_lead["persona_id"])
            )
    context = knowledge_graph.get_chat_context(
        lead_ref=lead_ref,
        persona_id=persona_id,
        user_text=q,
        limit=limit,
    )
    resolved_persona_id = str(context.get("persona_id") or persona_id or "")
    if not resolved_persona_id:
        raise HTTPException(403, "Lead sem persona autorizada.")
    auth_service.assert_persona_access(request, persona_id=resolved_persona_id)
    personas = supabase_client.get_personas() or []
    persona = next((row for row in personas if str(row.get("id")) == resolved_persona_id), None)
    if not persona:
        raise HTTPException(404, "Persona not found")
    messages = supabase_client.get_messages(str(lead_ref), limit=500) if lead_ref else []
    try:
        projection_nodes, _projection_edges = supabase_client.list_all_knowledge_graph(
            persona_id=resolved_persona_id,
            limit_nodes=5000,
        )
    except Exception:
        projection_nodes = list(context.get("nodes") or [])
    try:
        turn = context_cards_service.response_context(
            persona_slug=str(persona.get("slug")),
            persona_id=resolved_persona_id,
            lead_ref=int(lead_ref),
            messages=messages,
            response_message_id=response_message_id,
            query=q or "",
            projection_nodes=projection_nodes,
            limit=limit,
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {**knowledge_graph.with_operator_context(context, limit=limit), **turn}


@router.post("/context-cards/{node_id}/publish")
def publish_context_card(
    node_id: str,
    body: PublishContextCardBody,
    request: Request,
):
    """Edit, approve, project and activate one canonical card in one commit."""
    persona = supabase_client.get_persona(body.persona_slug)
    if not persona:
        raise HTTPException(404, f"Persona not found: {body.persona_slug}")
    user = auth_service.current_user(request)
    if not auth_service.is_admin(user):
        allowed = next(
            (
                row for row in auth_service.allowed_access(request)
                if row.get("persona_slug") == body.persona_slug and row.get("can_edit")
            ),
            None,
        )
        if not allowed:
            raise HTTPException(403, "Acesso de edicao negado para esta persona.")
    current = graph_json_v2_store.load_current(body.persona_slug)
    if not current:
        raise HTTPException(404, "Published Graph JSON not found")
    current_version, graph = current
    graph = graph_json_v21_adapter.upgrade_to_v21(graph)
    node = next((item for item in graph.nodes if item.id == node_id), None)
    if not node or node.node_class != "knowledge":
        raise HTTPException(404, "Context card node not found")
    content = body.content.strip()
    if not content:
        raise HTTPException(422, "Card content cannot be blank")
    if node.node_type == "faq":
        patch = {
            "spec.answer": content,
            "data.answer": content,
            "data.content": content,
        }
    else:
        patch = {
            "spec.summary": content,
            "data.summary": content,
            "data.content": content,
        }
    digest = hashlib.sha256(f"{content}\n{body.reason}".encode("utf-8")).hexdigest()[:20]
    key = body.idempotency_key or f"context-card:{node_id}:v{body.expected_version}:{digest}"
    try:
        next_graph = graph_document_publisher.apply_operations(
            graph,
            [
                {"op": "update_node", "node_id": node_id, "patch": patch},
                {
                    "op": "approve_node", "node_id": node_id,
                    "approved_by": (user or {}).get("id"), "reason": body.reason,
                },
            ],
        )
        result = graph_document_publisher.commit(
            graph=next_graph,
            persona_slug=body.persona_slug,
            brand_slug=graph.brand_slug,
            source="knowledge.context_card.publish",
            reason=body.reason,
            published_by=(user or {}).get("id"),
            expected_version=body.expected_version,
            idempotency_key=key,
        )
    except graph_document_publisher.VersionConflict as exc:
        context_cards_service.emit_metric(
            "knowledge_context.version_conflict",
            persona_id=persona.get("id"), lead_ref=None,
            payload={"node_id": node_id, "expected": exc.expected, "current": exc.current},
        )
        raise HTTPException(
            409,
            {"code": "GRAPH_VERSION_CONFLICT", "expected_version": exc.expected, "current_version": exc.current},
        ) from exc
    except graph_document_publisher.GraphValidationError as exc:
        raise HTTPException(422, {"code": "GRAPH_VALIDATION_FAILED", "errors": exc.errors}) from exc
    except graph_document_publisher.ProjectionFailed as exc:
        raise HTTPException(502, {"code": "GRAPH_PROJECTION_FAILED", "graph_version": exc.version, "error": str(exc)}) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    activated = context_cards_service.current_graph(body.persona_slug)
    version, checksum, active_graph = activated
    card = context_cards_service.cards_for_ids(
        graph=active_graph, graph_version=version, graph_checksum=checksum,
        ids=[node_id],
    )
    context_cards_service.emit_metric(
        "knowledge_context.card_published",
        persona_id=persona.get("id"), lead_ref=None,
        payload={
            "node_id": node_id, "author": (user or {}).get("id"),
            "reason": body.reason, "previous_version": current_version,
            "new_version": version,
        },
    )
    return {**result, "card": card[0].model_dump(mode="json") if card else None}


# â”€â”€ Published Graph JSON v2 context â”€â”€â”€

@router.get("/context/{persona_slug}")
def get_kb_context(persona_slug: str, request: Request):
    """Return approved context from the published Graph JSON v2."""
    persona = supabase_client.get_persona(persona_slug)
    if not persona:
        raise HTTPException(404, f"Persona not found: {persona_slug}")
    auth_service.assert_persona_access(request, persona_id=persona.get("id"), persona_slug=persona_slug)
    current = graph_json_v2_store.load_current(persona_slug)
    if not current:
        return {"persona_slug": persona_slug, "context": "", "graph_version": None}
    version, graph = current
    event = graph_json_v2_store.latest_event(persona_slug) or {}
    lines: list[str] = []
    for node in graph.nodes:
        data = node.data or {}
        if data.get("active", True) is False:
            continue
        if str(data.get("status") or "").lower() not in {
            "approved", "validated", "active", "ativo"
        }:
            continue
        lines.append(
            f"\n### {node.node_type.upper()} Â· {node.id}\n"
            f"{str(data.get('markdown') or node.label)[:800]}"
        )
    return {
        "persona_slug": persona_slug,
        "graph_version": version,
        "graph_checksum": (event.get("payload") or {}).get("checksum"),
        "context": "\n".join(lines),
    }


# â”€â”€ Brand Profiles â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.get("/brand/{persona_id}")
def get_brand(persona_id: str, request: Request):
    auth_service.assert_persona_access(request, persona_id=persona_id)
    return supabase_client.get_brand_profile(persona_id) or {}


@router.put("/brand/{persona_id}")
def upsert_brand(persona_id: str, body: dict, request: Request):
    auth_service.assert_persona_access(request, persona_id=persona_id)
    before = supabase_client.get_brand_profile(persona_id) or {}
    updated = supabase_client.upsert_brand_profile({"persona_id": persona_id, **body})
    after = updated or {**before, **(body or {})}
    keys = set((body or {}).keys()) | set(before.keys() if isinstance(before, dict) else [])
    before_view = {k: before.get(k) if isinstance(before, dict) else None for k in keys}
    after_view = {k: after.get(k) for k in keys}
    emit(
        "brand_profile_updated",
        entity_type="brand_profile",
        entity_id=persona_id,
        persona_id=persona_id,
        payload={
            "actor": current_actor(request),
            "before": before_view,
            "after": after_view,
            "diff": summarize_diff(before_view, after_view),
        },
        source="routes.knowledge",
    )
    return updated

