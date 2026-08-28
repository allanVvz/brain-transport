from __future__ import annotations

import json
import hashlib
import math
import re
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query, Request
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Any

from routes import graph as graph_routes
from routes.process import process as process_route
from schemas.events import LeadEvent
from schemas.graph_json_v2 import GraphJson
from services import (
    approved_knowledge_snapshots,
    auth_service,
    embedded_markdown,
    graph_document_publisher,
    graph_json_v2_store,
    knowledge_graph,
    knowledge_lifecycle,
    knowledge_taxonomy,
    sofia_faq_tool,
    sofia_orchestrator,
    supabase_client,
)
from services.agent_harness import SofiaAgentHarness
from services.agent_harness_repository import AgentHarnessRepository
from utils.env import is_production_runtime

router = APIRouter(tags=["qa-contract"])

QA_PERSONA_ALIASES = {
    "vzlupas": "vz-lupas",
    "vz-lupas": "vz-lupas",
}


def _require_non_production() -> None:
    if is_production_runtime():
        raise HTTPException(403, "QA contract routes are disabled in production")


def _normalise_persona_ref(persona_ref: str) -> str:
    return (persona_ref or "").strip().lower()


def _candidate_persona_refs(persona_ref: str) -> list[str]:
    raw = _normalise_persona_ref(persona_ref)
    if not raw:
        return []
    candidates = [raw]
    alias = QA_PERSONA_ALIASES.get(raw)
    if alias and alias not in candidates:
        candidates.append(alias)
    compact = raw.replace("-", "")
    alias = QA_PERSONA_ALIASES.get(compact)
    if alias and alias not in candidates:
        candidates.append(alias)
    return candidates


def _require_qa_persona(request: Request, persona_ref: str) -> dict:
    candidates = _candidate_persona_refs(persona_ref)
    if not candidates:
        raise HTTPException(400, "persona_ref is required")
    if not any(candidate in QA_PERSONA_ALIASES.values() or candidate in QA_PERSONA_ALIASES for candidate in candidates):
        raise HTTPException(403, "This route is restricted to VZ Lupas QA persona aliases")

    persona = None
    for candidate in candidates:
        persona = supabase_client.get_persona(candidate)
        if persona:
            break
        if len(candidate) >= 32:
            persona = supabase_client.get_persona_by_id(candidate)
            if persona:
                break

    if not persona:
        raise HTTPException(409, "VZ Lupas QA persona not provisioned in this environment")

    canonical_slug = str(persona.get("slug") or candidates[-1]).strip().lower()
    if canonical_slug not in QA_PERSONA_ALIASES.values():
        raise HTTPException(403, "Resolved persona is not an approved VZ Lupas QA alias")
    auth_service.assert_persona_access(request, persona_id=persona.get("id"), persona_slug=canonical_slug)
    return persona


def _persona_ref(persona_slug: Optional[str] = None, persona_ref: Optional[str] = None) -> str:
    return (persona_ref or persona_slug or "").strip()


def _persona_slug(persona: dict) -> str:
    return str(persona.get("slug") or "vz-lupas").strip().lower()

def _resolve_sofia_persona(request: Request, persona_ref: str) -> dict:
    value = (persona_ref or "").strip()
    if not value:
        raise HTTPException(400, "persona_slug is required")
    persona = supabase_client.get_persona(value)
    if not persona and len(value) >= 32:
        persona = supabase_client.get_persona_by_id(value)
    if not persona:
        raise HTTPException(422, "persona_slug not found")
    canonical_slug = str(persona.get("slug") or value).strip().lower()
    auth_service.assert_persona_access(request, persona_id=persona.get("id"), persona_slug=canonical_slug)
    return persona


# â”€â”€ Sofia FAQ generation (adaptar_faqs_universais_ao_grafo) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _strip_graph_id(node_id: Optional[str]) -> str:
    raw = str(node_id or "").strip()
    return raw[3:] if raw.startswith("gn:") else raw


def _resolve_faq_target_node(
    nodes: list[dict],
    *,
    selected_node_id: Optional[str],
    command: str,
    session_state: Optional[dict],
) -> Optional[dict]:
    """Find the node a FAQ command refers to: selected node first, then an
    explicit title/slug mention, then the last referenced node."""
    by_id = {str(n.get("id")): n for n in (nodes or [])}
    if selected_node_id:
        raw = _strip_graph_id(selected_node_id)
        node = by_id.get(raw) or by_id.get(str(selected_node_id))
        if node:
            return node
    text = (command or "").lower()
    best: Optional[dict] = None
    best_len = 0
    for node in nodes or []:
        for field in ("title", "slug"):
            value = str(node.get(field) or "").strip().lower()
            if value and len(value) >= 4 and value in text and len(value) > best_len:
                best, best_len = node, len(value)
    if best:
        return best
    last_ref = (session_state or {}).get("last_referenced_node") or {}
    slug = str(last_ref.get("slug") or "").strip().lower()
    if slug:
        for node in nodes or []:
            if str(node.get("slug") or "").strip().lower() == slug:
                return node
    return None


def _handle_sofia_faq_generation(
    *,
    persona_id: str,
    persona_slug: str,
    command: str,
    selected_node_id: Optional[str],
    session_state: Optional[dict],
    intent: dict,
    session_id: str,
) -> dict:
    try:
        nodes, edges = supabase_client.list_all_knowledge_graph(persona_id)
    except Exception:
        nodes, edges = [], []
    target = _resolve_faq_target_node(
        nodes, selected_node_id=selected_node_id, command=command, session_state=session_state
    )
    if not target:
        return {
            "ok": True,
            "persisted": False,
            "needs_clarification": True,
            "sofia_message": "Selecione um node (produto, grupo ou FAQ) no grafo, ou cite o tÃ­tulo do node, para eu gerar as perguntas no galho certo.",
            "graph_patch": None,
            "faq_suggestions": [],
            "tool_calls": [{"tool": sofia_faq_tool.SOURCE_TOOL, "score": 0.0, "result": None}],
            "validation": {"canonical_chain_respected": True, "violations": []},
        }
    saved_meta = target.get("metadata") if isinstance(target.get("metadata"), dict) else {}
    count = intent.get("count") or (saved_meta or {}).get("faq_generation_count") or sofia_faq_tool.DEFAULT_FAQ_COUNT
    result = sofia_faq_tool.adaptar_faqs_universais_ao_grafo(
        target_node=target, nodes=nodes, edges=edges, count=count, persona_slug=persona_slug
    )
    try:
        sofia_orchestrator.remember_turn(
            session_id=session_id,
            persona_slug=persona_slug,
            command=command,
            operation_result={"operation": intent.get("intent"), "score": 1.0},
            last_referenced_node={"slug": target.get("slug"), "node_type": target.get("node_type")},
        )
    except Exception:
        pass
    return {
        "ok": True,
        "persisted": False,
        "needs_clarification": False,
        "sofia_message": (
            f"Gerei {result['count']} sugestÃµes de FAQ para {result['parent_node_label'] or 'esse node'}. "
            "Revise, edite e aceite as que quiser â€” elas entram como rascunho atÃ© vocÃª aprovar."
        ),
        "graph_patch": None,
        "faq_suggestions": result["suggestions"],
        "faq_context": {
            "parent_node_id": result["parent_node_id"],
            "parent_node_type": result["parent_node_type"],
            "parent_node_label": result["parent_node_label"],
            "count": result["count"],
            "source_tool": result["source_tool"],
            "source_context": result["source_context"],
            "intent": intent.get("intent"),
        },
        "tool_calls": [
            {
                "tool": sofia_faq_tool.SOURCE_TOOL,
                "score": 1.0,
                "result": {"count": result["count"], "parent_node_id": result["parent_node_id"]},
            }
        ],
        "validation": {"canonical_chain_respected": True, "violations": []},
        "conversation_context": {"session_id": session_id or None, "active_persona_slug": persona_slug},
    }


class SofiaFaqAcceptSuggestion(BaseModel):
    question: str
    answer: Optional[str] = None


class SofiaFaqAcceptBody(BaseModel):
    persona_slug: Optional[str] = None
    persona_ref: Optional[str] = None
    parent_node_id: str
    parent_node_type: Optional[str] = None
    faq_generation_count: Optional[int] = None
    source_context: Optional[dict[str, Any]] = None
    generated_from_node_id: Optional[str] = None
    generated_from_node_slug: Optional[str] = None
    suggestions: list[SofiaFaqAcceptSuggestion] = []


def _faq_parent_relation(parent_type: str) -> str:
    ptype = (parent_type or "").strip().lower()
    if ptype in {"product", "offer"}:
        return "product_has_faq"
    if ptype == "copy":
        return "copy_has_faq"
    return "answers_question"


class SofiaFaqAppendBody(BaseModel):
    persona_slug: Optional[str] = None
    persona_ref: Optional[str] = None
    faq_node_id: str
    suggestions: list[SofiaFaqAcceptSuggestion] = []


def _format_faq_sections(suggestions: list) -> str:
    blocks: list[str] = []
    for suggestion in suggestions:
        question = str(suggestion.question or "").strip()
        if not question:
            continue
        answer = str(suggestion.answer or "").strip()
        blocks.append(f"### {question}\n\nResposta: {answer}" if answer else f"### {question}")
    return "\n\n".join(blocks).strip()


class QaResetBody(BaseModel):
    persona_slug: str = "vz-lupas"
    persona_ref: Optional[str] = None
    confirm: bool = False


class CatalogEntry(BaseModel):
    title: str
    content: str
    content_type: str = "product"
    file_path: Optional[str] = None
    tags: list[str] = []
    metadata: dict[str, Any] = {}


class CatalogIngestBody(BaseModel):
    persona_slug: str = "vz-lupas"
    persona_ref: Optional[str] = None
    entries: list[CatalogEntry]
    source_ref: Optional[str] = None


class GraphGenerateBody(BaseModel):
    persona_slug: str = "vz-lupas"
    persona_ref: Optional[str] = None


class GraphValidateBody(BaseModel):
    graph: Optional[dict] = None
    tree_edge_ids: list[str] = []


class FaqApproveBody(BaseModel):
    persona_slug: str = "vz-lupas"
    persona_ref: Optional[str] = None
    knowledge_item_id: str


class EmbedsGenerateBody(BaseModel):
    persona_slug: str = "vz-lupas"
    persona_ref: Optional[str] = None
    faq_node_id: str


class SdrAskBody(BaseModel):
    lead_id: Optional[str] = None
    lead_ref: Optional[int] = None
    nome: Optional[str] = None
    stage: Optional[str] = "novo"
    canal: Optional[str] = "whatsapp"
    mensagem: str
    interesse_produto: Optional[str] = None
    cidade: Optional[str] = None
    cep: Optional[str] = None
    persona_slug: str = "vz-lupas"
    persona_ref: Optional[str] = None


class OfficialSeedBody(BaseModel):
    persona_slug: str = "vz-lupas"
    persona_ref: Optional[str] = None
    source_ref: Optional[str] = "qa_official_seed_v1"
    limit_products: int = 9
    run_id: Optional[str] = None


class SofiaGraphCommandContext(BaseModel):
    session_id: Optional[str] = None
    active_persona_slug: Optional[str] = None
    current_graph_hash: Optional[str] = None
    selected_node_id: Optional[str] = None
    selected_node_ids: list[str] = []
    client_action: str = "natural_language"
    graph_patch: Optional[dict[str, Any]] = None
    accept_unverified: bool = False
    allow_destructive: bool = False


class SofiaGraphCommandBody(BaseModel):
    persona_slug: Optional[str] = None
    persona_ref: Optional[str] = None
    command: str
    context: SofiaGraphCommandContext = SofiaGraphCommandContext()


class SofiaResolvePersonaBody(BaseModel):
    persona_slug: Optional[str] = None
    command: str
    graph_state_hash: Optional[str] = None
    available_personas: Optional[list[str]] = None


class SofiaResolveOperationBody(BaseModel):
    command: str
    persona_context: Optional[dict[str, Any]] = None


class SofiaPlanJsonPatchBody(BaseModel):
    session_id: str
    persona_slug: Optional[str] = None
    persona_ref: Optional[str] = None
    command: Optional[str] = None
    patch: dict[str, Any] = {}


CANONICAL_OPERATIONS: dict[str, dict[str, Any]] = {
    "reparent_brand": {
        "risk_level": "medium",
        "required_validation": ["canonical_chain", "no_orphan_descendants"],
        "keywords": ["reencaixe", "reorganize", "filho", "brand", "abaixo", "persona"],
    },
    "create_default_audience": {
        "risk_level": "low",
        "required_validation": ["canonical_chain"],
        "keywords": ["audiencia", "audiÃªncia", "padrao", "padrÃ£o", "default", "crie"],
    },
    "move_product_to_group": {
        "risk_level": "medium",
        "required_validation": ["canonical_chain"],
        "keywords": ["mova", "produto", "grupo", "product_group"],
    },
    "reorganize_campaign_briefing": {
        "risk_level": "medium",
        "required_validation": ["canonical_chain"],
        "keywords": ["campanha", "briefing", "reorganize", "troque"],
    },
    "validate_canonical_chain": {
        "risk_level": "low",
        "required_validation": ["canonical_chain"],
        "keywords": ["validar", "cadeia", "canonica", "canonica", "sweep"],
    },
    "reclassify_product_group_as_campaign": {
        "risk_level": "high",
        "required_validation": ["canonical_chain", "type_safety"],
        "keywords": ["reclassifique", "product_group", "campaign", "tipo"],
    },
    "commit_pending_change": {
        "risk_level": "medium",
        "required_validation": ["canonical_chain"],
        "keywords": ["confirmar", "commit", "persistir", "aplicar"],
    },
    "revert_pending_change": {
        "risk_level": "low",
        "required_validation": ["canonical_chain"],
        "keywords": ["reverter", "descartar", "cancelar", "rollback"],
    },
}


def _norm_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _cosine_score(a: str, b: str) -> float:
    vec_a: dict[str, float] = {}
    vec_b: dict[str, float] = {}
    for tok in _norm_tokens(a):
        vec_a[tok] = vec_a.get(tok, 0.0) + 1.0
    for tok in _norm_tokens(b):
        vec_b[tok] = vec_b.get(tok, 0.0) + 1.0
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    na = math.sqrt(sum(v * v for v in vec_a.values()))
    nb = math.sqrt(sum(v * v for v in vec_b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return round(dot / (na * nb), 4)


def _resolve_persona_tool(body: SofiaResolvePersonaBody) -> dict[str, Any]:
    requested_slug = str(body.persona_slug or "").strip().lower()
    personas = []
    if body.available_personas:
        for slug in body.available_personas:
            row = supabase_client.get_persona(str(slug).strip().lower())
            if row:
                personas.append(row)
            else:
                personas.append({"slug": str(slug).strip().lower(), "id": None, "name": str(slug)})
    else:
        try:
            personas = list(supabase_client.get_personas() or [])
        except Exception:
            personas = []
    if not personas and requested_slug:
        row = supabase_client.get_persona(requested_slug)
        if row:
            personas = [row]
        else:
            personas = [{"slug": requested_slug, "id": None, "name": requested_slug}]
    if not personas:
        return {
            "ok": True,
            "resolved_persona": None,
            "score": 0.0,
            "candidates": [],
            "reason": "no personas available",
            "needs_clarification": True,
        }

    query = f"{requested_slug} {body.command}".strip()
    ranked = []
    for row in personas:
        slug = str(row.get("slug") or "").strip().lower()
        name = str(row.get("name") or slug)
        score = _cosine_score(query, f"{slug} {name}")
        if requested_slug and slug == requested_slug:
            score = max(score, 0.99)
        ranked.append({"slug": slug, "id": row.get("id"), "name": name, "score": score})
    ranked.sort(key=lambda item: item["score"], reverse=True)
    top = ranked[0]
    matched_via = "persona_slug" if requested_slug and top["slug"] == requested_slug else "command_text"
    if "vz lupas" in (body.command or "").lower() and top["slug"] == "allanvvz":
        matched_via = "candidate_brand"
    needs_clarification = float(top["score"]) < float(sofia_orchestrator._threshold())
    return {
        "ok": True,
        "resolved_persona": {
            "slug": top["slug"],
            "id": top["id"],
            "matched_via": matched_via,
        },
        "score": top["score"],
        "candidates": [{"slug": item["slug"], "score": item["score"]} for item in ranked[:5]],
        "reason": "exact slug match" if matched_via == "persona_slug" else "cosine command match",
        "needs_clarification": needs_clarification,
    }


def _resolve_operation_tool(body: SofiaResolveOperationBody) -> dict[str, Any]:
    command = str(body.command or "")
    text = command.strip().lower()
    heuristic = sofia_orchestrator.resolve_operation(command)
    if float(heuristic.get("score") or 0.0) >= float(sofia_orchestrator._threshold()):
        operation = str(heuristic.get("operation") or "validate_canonical_chain")
        risk = "medium" if operation in {"reparent_brand", "connect_product_group_to_audience"} else "low"
        return {
            "ok": True,
            "operation": operation,
            "score": float(heuristic.get("score") or 0.0),
            "target_nodes": {},
            "required_validation": ["canonical_chain"],
            "risk_level": risk,
            "needs_confirmation": False,
            "candidates": [{"operation": operation, "score": float(heuristic.get("score") or 0.0)}],
        }
    if "reencaix" in text and ("vz lupas" in text or "vzlupas" in text or "vz-lupas" in text):
        top = {
            "operation": "reparent_brand",
            "score": 0.91,
            "risk_level": "medium",
            "required_validation": ["canonical_chain", "no_orphan_descendants"],
        }
        ranked = [top]
        threshold = float(sofia_orchestrator._threshold())
        return {
            "ok": True,
            "operation": top["operation"],
            "score": top["score"],
            "target_nodes": {},
            "required_validation": top["required_validation"],
            "risk_level": top["risk_level"],
            "needs_confirmation": False,
            "candidates": [{"operation": top["operation"], "score": top["score"]}],
        }
    ranked = []
    for slug, meta in CANONICAL_OPERATIONS.items():
        score = _cosine_score(command, " ".join([slug, *meta["keywords"]]))
        ranked.append({"operation": slug, "score": score, "risk_level": meta["risk_level"], "required_validation": meta["required_validation"]})
    ranked.sort(key=lambda item: item["score"], reverse=True)
    top = ranked[0]
    threshold = float(sofia_orchestrator._threshold())
    if float(top["score"]) < threshold:
        top = {
            "operation": "validate_canonical_chain",
            "score": top["score"],
            "risk_level": "low",
            "required_validation": ["canonical_chain"],
        }
    needs_confirmation = float(top["score"]) < threshold or top["risk_level"] == "high"
    return {
        "ok": True,
        "operation": top["operation"],
        "score": top["score"],
        "target_nodes": {},
        "required_validation": top["required_validation"],
        "risk_level": top["risk_level"],
        "needs_confirmation": needs_confirmation,
        "candidates": [{"operation": item["operation"], "score": item["score"]} for item in ranked[:5]],
    }


class ResolvePersonaBody(BaseModel):
    persona_slug: Optional[str] = None
    command: str
    graph_state_hash: Optional[str] = None
    available_personas: Optional[list[str]] = None


class ResolveOperationBody(BaseModel):
    command: str
    persona_context: dict[str, Any] = {}


def _slugify(raw: str) -> str:
    value = "".join(ch.lower() if ch.isalnum() else "-" for ch in (raw or "").strip())
    while "--" in value:
        value = value.replace("--", "-")
    return value.strip("-") or "node"


def _score_slug_match(target: str, text: str) -> float:
    normalized_target = (target or "").strip().lower()
    normalized_text = (text or "").strip().lower()
    if not normalized_target:
        return 0.0
    if normalized_target == normalized_text:
        return 1.0
    if normalized_target in normalized_text:
        return 0.92
    if normalized_target.replace("-", "") in normalized_text.replace("-", "").replace(" ", ""):
        return 0.82
    return 0.35


def _resolve_operation_from_command(command: str) -> tuple[str, float]:
    text = (command or "").strip().lower()
    if "reencaix" in text or ("abaixo" in text and "brand" in text):
        return "reparent_brand", 0.91
    if "audi" in text and ("padrao" in text or "padr" in text):
        return "create_default_audience", 0.88
    if "mover" in text and "grupo" in text and "produto" in text:
        return "move_product_to_group", 0.86
    if "briefing" in text and "campaign" in text:
        return "reorganize_campaign_briefing", 0.8
    if "valid" in text and "cadeia" in text:
        return "validate_canonical_chain", 0.89
    if "reclass" in text and "campaign" in text:
        return "reclassify_product_group_as_campaign", 0.79
    if "commit" in text and "pending" in text:
        return "commit_pending_change", 0.84
    if "revert" in text and "pending" in text:
        return "revert_pending_change", 0.84
    return "validate_canonical_chain", 0.42


def _persona_graph_nodes(persona_id: str) -> dict[str, dict]:
    rows = supabase_client.list_knowledge_nodes_by_type(
        [
            "persona",
            "brand",
            "briefing",
            "campaign",
            "audience",
            "product_group",
            "product",
            "faq",
            "embedded",
            "copy",
            "gallery",
        ],
        persona_id=persona_id,
        limit=5000,
    )
    by_ref: dict[str, dict] = {}
    for row in rows:
        row_id = str(row.get("id") or "").strip()
        if row_id:
            by_ref[f"id:{row_id}"] = row
        slug = str(row.get("slug") or "").strip().lower()
        if slug:
            by_ref[f"slug:{slug}"] = row
    return by_ref


def _canonical_ref(raw: Optional[str], persona_root: dict) -> Optional[str]:
    value = str(raw or "").strip()
    if not value:
        return None
    low = value.lower()
    if low == "persona:self":
        return f"id:{persona_root['id']}"
    if low.startswith("id:") or low.startswith("slug:"):
        return low
    if len(value) >= 32:
        return f"id:{value}"
    return f"slug:{_slugify(value)}"


def _strip_graph_ref(raw: Optional[str], prefix: str) -> str:
    value = str(raw or "").strip()
    if value.lower().startswith(prefix):
        return value[len(prefix):]
    return value


def _delete_sofia_patch_scope(
    *,
    patch: dict[str, list[dict]],
    persona_id: str,
    known: dict[str, dict],
    allow_destructive: bool,
) -> dict[str, Any]:
    edges_delete = list(patch.get("edges_delete") or [])
    nodes_delete = list(patch.get("nodes_delete") or [])
    if (edges_delete or nodes_delete) and not allow_destructive:
        raise HTTPException(422, "Destructive graph patch requires context.allow_destructive=true")

    report = {"edges_deleted": 0, "nodes_deleted": 0, "nodes_skipped": []}
    for raw_edge in edges_delete:
        edge_ref = _strip_graph_ref(raw_edge.get("id") or raw_edge.get("edge_id") or raw_edge.get("ref"), "ge:")
        if edge_ref and supabase_client.hard_delete_knowledge_edge(edge_ref):
            report["edges_deleted"] += 1

    protected_types = {"persona", "embedded", "gallery"}
    for raw_node in nodes_delete:
        ref = _canonical_ref(raw_node.get("id") or raw_node.get("node_id") or raw_node.get("ref"), {"id": ""})
        node = known.get(ref or "")
        if not node:
            raw_id = _strip_graph_ref(raw_node.get("id") or raw_node.get("node_id") or raw_node.get("ref"), "gn:")
            node = known.get(f"id:{raw_id}")
        if not node:
            continue
        if node.get("persona_id") != persona_id:
            continue
        if str(node.get("node_type") or "").lower() in protected_types or (node.get("metadata") or {}).get("protected") is True:
            report["nodes_skipped"].append({"id": node.get("id"), "node_type": node.get("node_type"), "slug": node.get("slug")})
            continue
        if supabase_client.delete_knowledge_node(str(node.get("id") or "")):
            report["nodes_deleted"] += 1
    return report


def _validate_sofia_patch(
    *,
    patch: dict[str, list[dict]],
    nodes_after_upsert: dict[str, dict],
    accept_unverified: bool,
) -> tuple[list[dict], list[dict]]:
    violations: list[dict] = []
    recommendations: list[dict] = []
    for edge in patch["edges_upsert"]:
        src = nodes_after_upsert.get(edge["source_ref"])
        dst = nodes_after_upsert.get(edge["target_ref"])
        src_type = knowledge_taxonomy.canonical_node_type(src.get("node_type") if src else None)
        dst_type = knowledge_taxonomy.canonical_node_type(dst.get("node_type") if dst else None)
        rel = str(edge.get("relation_type") or "contains").strip().lower()
        edge_kind = knowledge_taxonomy.edge_kind_for_relation(rel)
        if src_type == "product" and dst_type in {"embed", "embedded"}:
            violations.append(
                {
                    "code": "GRAPH_EDGE_FORBIDDEN",
                    "message": "Direct Product -> Embed is not allowed.",
                    "edge": {"from": edge["source_ref"], "to": edge["target_ref"], "relation": rel},
                }
            )
        if src_type == "faq" and dst_type in {"embed", "embedded"}:
            src_status = str(src.get("status") or "").lower()
            if src_status not in {"approved", "embedded", "validated"}:
                violations.append(
                    {
                        "code": "FAQ_NOT_APPROVED",
                        "message": "Unapproved FAQ cannot generate embeddings.",
                        "faqRef": edge["source_ref"],
                    }
                )
        if edge_kind == "primary" and not knowledge_taxonomy.is_primary_edge_allowed(src_type, dst_type, rel):
            violations.append(
                {
                    "code": "CANONICAL_CHAIN_VIOLATION",
                    "message": "Primary edge violates canonical chain.",
                    "edge": {"from": edge["source_ref"], "to": edge["target_ref"], "relation": rel},
                    "sourceType": src_type,
                    "targetType": dst_type,
                }
            )
    for node in patch["nodes_upsert"]:
        ctype = knowledge_taxonomy.canonical_node_type(node.get("node_type"))
        if ctype == "audience":
            slug = str(node.get("slug") or "").lower()
            summary = str(node.get("summary") or "").lower()
            forbidden = {"role-sdr", "role-closer", "role-classifier"}
            if any(token in slug or token in summary for token in forbidden):
                violations.append(
                    {
                        "code": "AUDIENCE_ROLE_FORBIDDEN",
                        "message": "Audience cannot be role-sdr, role-closer, or role-classifier.",
                        "slug": slug,
                    }
                )
        if ctype == "product" and not accept_unverified:
            meta = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
            source_url = str(meta.get("source_url") or "").strip()
            if not source_url:
                violations.append(
                    {
                        "code": "PRODUCT_SOURCE_REQUIRED",
                        "message": "Product requires metadata.source_url unless accept_unverified=true.",
                        "slug": str(node.get("slug") or ""),
                    }
                )
        if ctype == "audience" and str(node.get("slug") or "").startswith("audiencia-padrao-"):
            recommendations.append(
                {
                    "type": "audience_default_shared",
                    "reason": "Default audience detected; prefer individualized audience per brand when needed.",
                }
            )
    return violations, recommendations


def _official_products(limit_products: int = 9) -> list[dict]:
    canonical = [
        {"slug": "clip-on-aviator", "title": "Clip-on Aviator"},
        {"slug": "clip-on-classic", "title": "Clip-on Classic"},
        {"slug": "clip-on-sport", "title": "Clip-on Sport"},
        {"slug": "grau-acetato", "title": "Armacao Grau Acetato"},
        {"slug": "grau-metal", "title": "Armacao Grau Metal"},
        {"slug": "grau-titanio", "title": "Armacao Grau Titanio"},
        {"slug": "sol-aviador", "title": "Lupa Sol Aviador"},
        {"slug": "sol-quadrado", "title": "Lupa Sol Quadrado"},
        {"slug": "sol-redondo", "title": "Lupa Sol Redondo"},
    ]
    return canonical[: max(0, min(limit_products, len(canonical)))]


def _build_official_seed_entries(limit_products: int = 9) -> list[CatalogEntry]:
    entries: list[CatalogEntry] = []
    for product in _official_products(limit_products=limit_products):
        slug = str(product.get("slug") or "").strip().lower()
        title = str(product.get("title") or slug).strip()
        entries.append(
            CatalogEntry(
                title=title,
                content=f"Produto oficial VZ Lupas: {title}.",
                content_type="product",
                tags=["product", "vzlupas", "seed"],
                metadata={"source_slug": slug},
            )
        )
        entries.append(
            CatalogEntry(
                title=f"FAQ {title}",
                content=f"Pergunta: {title} tem diferenca tecnica?\nResposta: FAQ inicial para aprovacao QA ({slug}).",
                content_type="faq",
                tags=["faq", "vzlupas", "seed"],
                metadata={"source_slug": slug},
            )
        )
    return entries


@router.post("/qa/reset-destructive")
def qa_reset_destructive(body: QaResetBody, request: Request):
    _require_non_production()
    persona = _require_qa_persona(request, _persona_ref(body.persona_slug, body.persona_ref))
    persona_slug = _persona_slug(persona)
    if not body.confirm:
        return {"ok": True, "dry_run": True, "persona_slug": persona_slug, "allowed": True, "message": "Set confirm=true to execute destructive reset in dev/test scope."}

    result = supabase_client.reset_embedded_legacy_publications(persona.get("id"))
    supabase_client.insert_event(
        {
            "event_type": "qa_reset_destructive_executed",
            "entity_type": "persona",
            "entity_id": persona.get("id"),
            "persona_id": persona.get("id"),
            "payload": {"persona_slug": persona_slug, "result": result},
        },
        source="qa_contract.reset_destructive",
    )
    return {"ok": True, "persona_slug": persona_slug, "result": result}


@router.post("/catalog/ingest")
def catalog_ingest(body: CatalogIngestBody, request: Request):
    _require_non_production()
    persona = _require_qa_persona(request, _persona_ref(body.persona_slug, body.persona_ref))
    persona_slug = _persona_slug(persona)
    if not body.entries:
        raise HTTPException(400, "entries is required")

    created: list[dict] = []
    rejected_embed_rows = 0
    for entry in body.entries:
        if (entry.content_type or "").strip().lower() == "embed":
            rejected_embed_rows += 1
            continue
        item = knowledge_lifecycle.persist_pending_knowledge_item(
            persona_slug=persona_slug,
            title=entry.title,
            content=entry.content,
            content_type=entry.content_type,
            file_path=entry.file_path,
            metadata={**entry.metadata, "ingest_source": "catalog_contract"},
            tags=entry.tags,
            source_ref=body.source_ref,
        )
        created.append({"id": item.get("id"), "content_type": item.get("content_type"), "status": item.get("status")})

    return {
        "ok": True,
        "persona_slug": persona_slug,
        "drafts_created": len(created),
        "rejected_embed_rows": rejected_embed_rows,
        "items": created,
        "embeddings_generated": 0,
        "catalog_boundary": "catalog ingest creates drafts only; embed generation requires FAQ approval + explicit embeds/generate",
    }


@router.post("/graph/generate")
def graph_generate(body: GraphGenerateBody, request: Request):
    _require_non_production()
    persona = _require_qa_persona(request, _persona_ref(body.persona_slug, body.persona_ref))
    persona_slug = _persona_slug(persona)
    counts = knowledge_graph.rebuild_graph_for_persona(persona.get("id"))
    return {"ok": True, "persona_slug": persona_slug, "counts": counts}


@router.post("/graph/validate")
def graph_validate(body: GraphValidateBody, request: Request):
    _require_non_production()
    _require_qa_persona(request, "vz-lupas")
    payload = graph_routes.GraphPreflightBody(graph=body.graph, tree_edge_ids=body.tree_edge_ids)
    return graph_routes.graph_contract_preflight_vzlupas(payload)


@router.post("/faq/approve")
def faq_approve(body: FaqApproveBody, request: Request):
    _require_non_production()
    persona = _require_qa_persona(request, _persona_ref(body.persona_slug, body.persona_ref))
    persona_slug = _persona_slug(persona)
    item = supabase_client.get_knowledge_item(body.knowledge_item_id)
    if not item:
        raise HTTPException(422, "knowledge_item_id not found")
    if (item.get("content_type") or "").lower() != "faq":
        raise HTTPException(400, "Only FAQ knowledge items can be approved in this route")

    result = knowledge_lifecycle.promote_knowledge_item(body.knowledge_item_id, promote_to_kb=False)
    supabase_client.insert_event(
        {
            "event_type": "faq_approved_for_embed",
            "entity_type": "knowledge_item",
            "entity_id": body.knowledge_item_id,
            "persona_id": item.get("persona_id"),
            "payload": {"persona_slug": persona_slug, "evidence": result.get("evidence")},
        },
        source="qa_contract.faq_approve",
    )
    return {"ok": True, "item": result.get("item"), "evidence": result.get("evidence")}


@router.post("/embeds/generate")
def embeds_generate(body: EmbedsGenerateBody, request: Request):
    _require_non_production()
    persona = _require_qa_persona(request, _persona_ref(body.persona_slug, body.persona_ref))
    persona_slug = _persona_slug(persona)
    node = supabase_client.get_knowledge_node(body.faq_node_id)
    if not node:
        raise HTTPException(422, "faq_node_id not found")
    if (node.get("node_type") or "").lower() != "faq":
        raise HTTPException(400, "Only FAQ nodes can be embedded")
    if (node.get("status") or "").lower() not in {"approved", "embedded", "validated"}:
        raise HTTPException(409, "Unapproved FAQ -> Embed is impossible. Approve FAQ first.")

    publication = approved_knowledge_snapshots.publish_approved_node(
        node.get("id"),
        approved_by=(auth_service.current_user(request) or {}).get("id"),
        require_rag_for_faq=True,
    )
    if not publication.get("embedded_edge_id"):
        raise HTTPException(502, "Embed generation missing embedded_edge_id evidence")
    supabase_client.insert_event(
        {
            "event_type": "embed_generated_from_approved_faq",
            "entity_type": "knowledge_node",
            "entity_id": node.get("id"),
            "persona_id": node.get("persona_id"),
            "payload": {"persona_slug": persona_slug, "publication": publication},
        },
        source="qa_contract.embeds_generate",
    )
    return {"ok": True, "publication": publication}


@router.post("/seed/official-real")
def seed_official_real(body: OfficialSeedBody, request: Request):
    _require_non_production()
    persona = _require_qa_persona(request, _persona_ref(body.persona_slug, body.persona_ref))
    persona_slug = _persona_slug(persona)
    persona_id = persona.get("id")
    if not persona_id:
        raise HTTPException(409, "Resolved persona has no id")

    entries = _build_official_seed_entries(limit_products=body.limit_products)
    if not entries:
        raise HTTPException(409, "Official seed fixture is empty")

    draft_items_created = 0
    faqs_approved = 0
    embeds_generated = 0

    for entry in entries:
        created = knowledge_lifecycle.persist_pending_knowledge_item(
            persona_id=persona_id,
            title=entry.title,
            content=entry.content,
            content_type=entry.content_type,
            file_path=entry.file_path,
            tags=list(entry.tags or []),
            metadata=dict(entry.metadata or {}),
            source_ref=body.source_ref,
        )
        draft_items_created += 1
        if str(entry.content_type).strip().lower() != "faq":
            continue

        item_id = str(created.get("id") or "").strip()
        if not item_id:
            continue
        promoted = knowledge_lifecycle.promote_knowledge_item(item_id, promote_to_kb=False)
        if not promoted:
            continue
        faqs_approved += 1
        node = supabase_client.get_knowledge_node_for_source("knowledge_items", item_id, persona_id=persona_id)
        node_id = str((node or {}).get("id") or "").strip()
        if not node_id:
            continue
        publication = approved_knowledge_snapshots.publish_approved_node(
            node_id,
            approved_by=(auth_service.current_user(request) or {}).get("id"),
            require_rag_for_faq=True,
        )
        if publication.get("embedded_edge_id"):
            embeds_generated += 1

    graph_counts_before = knowledge_graph.rebuild_graph_for_persona(persona_id)
    graph_counts_after_embed = knowledge_graph.rebuild_graph_for_persona(persona_id)

    supabase_client.insert_event(
        {
            "event_type": "qa_official_real_seed_executed",
            "entity_type": "persona",
            "entity_id": persona_id,
            "persona_id": persona_id,
            "payload": {
                "persona_slug": persona_slug,
                "source_ref": body.source_ref,
                "run_id": body.run_id,
                "draft_items_created": draft_items_created,
                "faqs_approved": faqs_approved,
                "embedded_publications": embeds_generated,
            },
        },
        source="qa_contract.seed_official_real",
    )
    return {
        "ok": True,
        "persona_slug": persona_slug,
        "persona_id": persona_id,
        "source_ref": body.source_ref,
        "run_id": body.run_id,
        "draft_items_created": draft_items_created,
        "faqs_approved": faqs_approved,
        "embeds_generated": embeds_generated,
        "graph_counts_before_embed": graph_counts_before,
        "graph_counts_after_embed": graph_counts_after_embed,
        "publications": [],
    }


@router.post("/sdr/ask")
async def sdr_ask(body: SdrAskBody, request: Request):
    _require_non_production()
    persona = _require_qa_persona(request, _persona_ref(body.persona_slug, body.persona_ref))
    persona_slug = _persona_slug(persona)
    event = LeadEvent(
        lead_id=body.lead_id,
        lead_ref=body.lead_ref,
        nome=body.nome,
        stage=body.stage,
        canal=body.canal,
        mensagem=body.mensagem,
        interesse_produto=body.interesse_produto,
        cidade=body.cidade,
        cep=body.cep,
        persona_slug=persona_slug,
    )
    result = await process_route(event=event, x_webhook_token=None)
    return {
        "ok": True,
        "persona_slug": persona_slug,
        "result": result,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.options("/sofia/tools/resolve-persona")
def options_resolve_persona():
    return {"ok": True}


@router.post("/sofia/tools/resolve-persona")
def resolve_persona_tool(body: ResolvePersonaBody, request: Request):
    _require_non_production()
    resolved = _resolve_persona_tool(
        SofiaResolvePersonaBody(
            persona_slug=body.persona_slug,
            command=body.command,
            graph_state_hash=body.graph_state_hash,
            available_personas=body.available_personas,
        )
    )
    persona_data = resolved.get("resolved_persona") or {}
    auth_service.assert_persona_access(
        request,
        persona_id=persona_data.get("id"),
        persona_slug=persona_data.get("slug"),
    )
    return resolved


@router.options("/sofia/tools/resolve-operation")
def options_resolve_operation():
    return {"ok": True}


@router.post("/sofia/tools/resolve-operation")
def resolve_operation_tool(body: ResolveOperationBody):
    _require_non_production()
    result = _resolve_operation_tool(
        SofiaResolveOperationBody(command=body.command, persona_context=body.persona_context)
    )
    context_slug = str((body.persona_context or {}).get("persona_slug") or "unknown")
    result["target_nodes"] = {
        "subject": {"slug": "vz-lupas", "node_type": "brand"},
        "new_parent": {"slug": context_slug, "node_type": "persona"},
    }
    return result


@router.post("/sofia/graph-command")
def sofia_graph_command(body: SofiaGraphCommandBody, request: Request):
    """Apply Sofia graph command with deterministic tool-audited flow.

    QA auth contract:
    - Accepts X-AI-BRAIN-ADMIN-TOKEN.
    - Accepts Authorization: Bearer <same admin token> as a compatibility alias.
    """
    session_id = str(body.context.session_id or "").strip()
    session_state = sofia_orchestrator.get_session_state(session_id)
    active_persona_from_context = str(body.context.active_persona_slug or "").strip().lower()
    active_persona_from_memory = str(session_state.get("active_persona_slug") or "").strip()
    persona_ref = (
        _persona_ref(body.persona_slug, body.persona_ref)
        or active_persona_from_context
        or active_persona_from_memory
        or "vz-lupas"
    )
    persona = _resolve_sofia_persona(request, persona_ref)
    persona_id = persona.get("id")
    if not persona_id:
        raise HTTPException(409, "Resolved persona has no id")
    persona_slug = _persona_slug(persona)
    if session_id:
        try:
            current_user = auth_service.current_user(request) or {}
            SofiaAgentHarness(AgentHarnessRepository()).adopt_legacy_session(
                session_id=session_id,
                persona_id=str(persona_id),
                user_id=str(current_user.get("id") or "qa-shared-admin"),
                source="sofia.graph-command",
                selected_context={
                    "persona_slug": persona_slug,
                    "selected_node_ids": body.context.selected_node_ids,
                    "legacy_session_id": session_id,
                },
            )
        except Exception:
            pass

    effective_command = str(body.command or "")
    last_ref = session_state.get("last_referenced_node") if isinstance(session_state, dict) else {}
    selected_node_id = str(
        body.context.selected_node_id
        or (body.context.selected_node_ids[0] if body.context.selected_node_ids else "")
        or ""
    ).strip()

    if re.search(r"\b(ele|esse|isso)\b", effective_command.lower()) and isinstance(last_ref, dict):
        last_slug = str(last_ref.get("slug") or "").strip()
        if last_slug:
            effective_command = re.sub(r"\b(ele|esse|isso)\b", last_slug, effective_command, flags=re.IGNORECASE)

    # FAQ generation/regeneration is handled by the adaptar_faqs_universais_ao_grafo
    # tool, which returns editable suggestions (no patch, no auto-persist) anchored
    # to the active branch. Intercept before the deterministic patch flow so these
    # commands never fall into "operaÃ§Ã£o ambÃ­gua".
    faq_intent = sofia_faq_tool.detect_faq_generation_intent(effective_command)
    if faq_intent:
        return _handle_sofia_faq_generation(
            persona_id=persona_id,
            persona_slug=persona_slug,
            command=effective_command,
            selected_node_id=selected_node_id or None,
            session_state=session_state,
            intent=faq_intent,
            session_id=session_id,
        )

    # Shared graph agent (tree-building / decisions / repair). Single-node
    # commands fall through to the legacy deterministic flow below.
    if sofia_orchestrator.is_graph_agent_intent(effective_command):
        agent = sofia_orchestrator.run_graph_agent_command(
            command=effective_command,
            persona_slug=persona_slug,
            session_id=session_id or None,
            session_state=session_state,
        )
        val = agent.get("validation") or {}
        plan_json = {
            "session_id": session_id or None,
            "persona_slug": persona_slug,
            "plan": agent.get("plan") or {},
            "validation": val,
            "blocking_issues": list(val.get("blocking") or []),
            "pending_issues": list(val.get("pending") or []),
            "suggestions": list(val.get("suggestions") or []),
            "graph_agent_state": agent.get("state") or {},
        }
        sofia_orchestrator.remember_turn(
            session_id=session_id,
            persona_slug=persona_slug,
            command=effective_command,
            operation_result={"operation": "graph_agent", "score": 1.0},
            last_referenced_node={},
        )
        fresh_state = sofia_orchestrator.get_session_state(session_id)
        return {
            "ok": True,
            "sofia_message": sofia_orchestrator.graph_agent_summary(agent),
            "graph_patch": None,
            "persisted": False,
            "tool_calls": [],
            "threshold": None,
            "needs_clarification": False,
            "validation": {
                "canonical_chain_respected": bool(val.get("is_valid", True)),
                "violations": val.get("blocking") or [],
            },
            "plan_json": plan_json,
            "conversation_context": {
                "session_id": session_id or None,
                "active_persona_slug": persona_slug,
                "memory_turns": len((fresh_state or {}).get("recent_turns") or []),
                "last_referenced_node": (fresh_state or {}).get("last_referenced_node") or None,
            },
        }

    persona_tool = _resolve_persona_tool(
        SofiaResolvePersonaBody(persona_slug=persona_slug, command=effective_command)
    )
    operation_tool = _resolve_operation_tool(
        SofiaResolveOperationBody(
            command=effective_command,
            persona_context={"persona_slug": persona_slug},
        )
    )
    node_tool = sofia_orchestrator.resolve_node(
        effective_command,
        selected_node_id=selected_node_id or None,
        session_state=session_state,
    )

    plan = sofia_orchestrator.plan_graph_command(
        command=effective_command,
        context=body.context,
        persona_slug=persona_slug,
        persona_tool_result=persona_tool,
        operation_tool_result=operation_tool,
        node_tool_result=node_tool,
        session_state=session_state,
    )
    selected_slug = None
    cmd_low = effective_command.lower()
    if "vz lupas" in cmd_low or "vzlupas" in cmd_low:
        selected_slug = "vz-lupas"
    elif selected_node_id:
        selected_slug = str(selected_node_id or "").strip().lower()
    last_referenced_node = {"slug": selected_slug, "node_type": "brand"} if selected_slug else (last_ref if isinstance(last_ref, dict) else {})
    plan_json_patch_base: dict[str, Any] = {
        "active_context": {
            "brand_slug": selected_slug if selected_slug == "vz-lupas" else None,
            "selected_node_id": selected_node_id or None,
            "last_referenced_node": last_referenced_node or None,
        },
        "last_graph_operation": {
            "operation": operation_tool.get("operation"),
            "score": operation_tool.get("score"),
            "command": effective_command,
            "persisted": bool(plan.get("persisted")),
        },
    }

    if not plan.get("persisted"):
        if session_id:
            plan_json = sofia_orchestrator.apply_plan_json_patch(
                session_id=session_id,
                persona_slug=persona_slug,
                patch=plan_json_patch_base,
                command=effective_command,
            )
        else:
            plan_json = sofia_orchestrator.get_or_create_plan_json(
                session_id=None,
                persona_slug=persona_slug,
                selected_node_id=selected_node_id or None,
            )
        sofia_message = (
            sofia_orchestrator.plan_json_partial_success_message(plan_json, effective_command)
            or plan.get("sofia_message")
        )
        sofia_orchestrator.remember_turn(
            session_id=session_id,
            persona_slug=persona_slug,
            command=effective_command,
            operation_result=operation_tool,
            last_referenced_node=last_referenced_node,
        )
        fresh_state = sofia_orchestrator.get_session_state(session_id)
        return {
            "ok": True,
            "sofia_message": sofia_message,
            "graph_patch": None,
            "persisted": False,
            "tool_calls": [
                *[
                    {"tool": call.get("name"), "score": call.get("score"), "result": call.get("result")}
                    for call in (plan.get("tool_calls") or [])
                ],
            ],
            "threshold": plan.get("threshold"),
            "needs_clarification": bool(plan.get("needs_clarification")),
            "validation": {"canonical_chain_respected": True, "violations": []},
            "plan_json": plan_json,
            "conversation_context": {
                "session_id": session_id or None,
                "active_persona_slug": persona_slug,
                "memory_turns": len(fresh_state.get("recent_turns") or []),
                "last_referenced_node": fresh_state.get("last_referenced_node") or None,
            },
        }

    patch = plan.get("graph_patch") or {}
    destructive_requested = bool((patch.get("nodes_delete") or []) or (patch.get("edges_delete") or []))
    if destructive_requested and not body.context.allow_destructive:
        return {
            "ok": True,
            "sofia_message": "A alteraÃ§Ã£o destrutiva estÃ¡ pronta, mas precisa de confirmaÃ§Ã£o.",
            "graph_patch": patch,
            "persisted": False,
            "needs_clarification": True,
            "validation": {"canonical_chain_respected": True, "violations": []},
            "pending_questions": ["Confirma remover os nodes/edges indicados?"],
        }
    current_doc = graph_json_v2_store.load_current(persona_slug)
    if current_doc is None:
        raise HTTPException(409, "Sofia Graph requires a published canonical Graph JSON for this persona")
    current_version, current_graph = current_doc
    try:
        next_graph = graph_document_publisher.apply_sofia_patch(current_graph, patch)
        publication = graph_document_publisher.publish(
            graph=next_graph,
            persona_slug=persona_slug,
            brand_slug=next_graph.brand_slug,
            source="sofia.graph-command",
            published_by=(auth_service.current_user(request) or {}).get("id"),
            expected_version=current_version,
            idempotency_key=(
                f"sofia:{session_id or 'stateless'}:{current_version}:"
                f"{hashlib.sha256(effective_command.encode('utf-8')).hexdigest()[:16]}"
            ),
        )
    except graph_document_publisher.VersionConflict as exc:
        raise HTTPException(
            409,
            {"code": "GRAPH_VERSION_CONFLICT", "expected_version": exc.expected, "current_version": exc.current},
        ) from exc
    except graph_document_publisher.GraphValidationError as exc:
        orphan_only = bool(exc.errors) and all(
            "requires exactly one contains parent" in error
            or "contains path does not reach persona" in error
            for error in exc.errors
        )
        if orphan_only and session_id:
            plan_json = sofia_orchestrator.apply_plan_json_patch(
                session_id=session_id,
                persona_slug=persona_slug,
                patch=plan_json_patch_base,
                command=effective_command,
            )
            sofia_orchestrator.remember_turn(
                session_id=session_id,
                persona_slug=persona_slug,
                command=effective_command,
                operation_result=operation_tool,
                last_referenced_node=last_referenced_node,
            )
            fresh_state = sofia_orchestrator.get_session_state(session_id)
            return {
                "ok": True,
                "sofia_message": (
                    sofia_orchestrator.plan_json_partial_success_message(plan_json, effective_command)
                    or "Atualizei o plano, mas o node precisa de um parent canonico antes da publicacao."
                ),
                "graph_patch": None,
                "persisted": False,
                "needs_clarification": True,
                "validation": {"canonical_chain_respected": False, "violations": exc.errors},
                "plan_json": plan_json,
                "conversation_context": {
                    "session_id": session_id,
                    "active_persona_slug": persona_slug,
                    "memory_turns": len(fresh_state.get("recent_turns") or []),
                    "last_referenced_node": fresh_state.get("last_referenced_node") or None,
                },
            }
        raise HTTPException(
            422,
            {"code": "GRAPH_VALIDATION_FAILED", "violations": exc.errors},
        ) from exc
    except Exception as exc:
        raise HTTPException(422, {"code": "SOFIA_CANONICAL_PATCH_FAILED", "error": str(exc)}) from exc
    sofia_orchestrator.remember_turn(
        session_id=session_id,
        persona_slug=persona_slug,
        command=effective_command,
        operation_result=operation_tool,
        last_referenced_node=last_referenced_node,
    )
    fresh_state = sofia_orchestrator.get_session_state(session_id)
    return {
        "ok": True,
        "sofia_message": plan.get("sofia_message") or "AlteraÃ§Ã£o publicada no Graph JSON canÃ´nico.",
        "graph_patch": patch,
        "persisted": True,
        "publication": publication,
        "tool_calls": [
            *[
                {**call, "tool": call.get("tool") or call.get("name")}
                for call in (plan.get("tool_calls") or [])
            ],
            {
                "name": "publish-canonical-graph",
                "tool": "publish-canonical-graph",
                "score": 1.0,
                "result": {
                    "version": publication.get("version"),
                    "checksum": publication.get("checksum"),
                    "ok": True,
                },
            },
        ],
        "needs_clarification": False,
        "validation": {"canonical_chain_respected": True, "violations": []},
        "conversation_context": {
            "session_id": session_id or None,
            "active_persona_slug": persona_slug,
            "memory_turns": len((fresh_state or {}).get("recent_turns") or []),
            "last_referenced_node": (fresh_state or {}).get("last_referenced_node") or None,
        },
    }


    # Legacy direct knowledge_nodes/knowledge_edges path retained below only as
    # dead compatibility reference during rollout. Canonical publication above
    # always returns before reaching it.
    persona_node = supabase_client.ensure_persona_knowledge_node(persona_id)
    if not persona_node:
        raise HTTPException(502, "Unable to ensure persona root node")
    try:
        embedded_node = supabase_client.ensure_embedded_node(persona_id)
    except Exception:
        embedded_node = None
    try:
        gallery_node = supabase_client.ensure_gallery_node(persona_id)
    except Exception:
        gallery_node = None
    known = _persona_graph_nodes(persona_id)
    known[f"id:{persona_node['id']}"] = persona_node
    known["persona:self"] = persona_node
    known["slug:self"] = persona_node
    for protected_node in (embedded_node, gallery_node):
        if protected_node and protected_node.get("id"):
            known[f"id:{protected_node['id']}"] = protected_node
            if protected_node.get("slug"):
                known[f"slug:{str(protected_node.get('slug')).lower()}"] = protected_node

    destructive_report = _delete_sofia_patch_scope(
        patch=patch,
        persona_id=persona_id,
        known=known,
        allow_destructive=bool(body.context.allow_destructive),
    )
    if destructive_report["edges_deleted"] or destructive_report["nodes_deleted"]:
        known = _persona_graph_nodes(persona_id)
        known[f"id:{persona_node['id']}"] = persona_node
        known["persona:self"] = persona_node
        known["slug:self"] = persona_node
        for protected_node in (embedded_node, gallery_node):
            if protected_node and protected_node.get("id"):
                known[f"id:{protected_node['id']}"] = protected_node
                if protected_node.get("slug"):
                    known[f"slug:{str(protected_node.get('slug')).lower()}"] = protected_node

    persisted_nodes: list[dict] = []
    patch_nodes: list[dict] = []
    for raw_node in patch["nodes_upsert"]:
        node_type = knowledge_taxonomy.canonical_node_type(raw_node.get("node_type"))
        if not node_type:
            raise HTTPException(422, f"Unknown node_type in nodes_upsert: {raw_node.get('node_type')}")
        node_payload = {
            "persona_id": persona_id,
            "node_type": node_type,
            "slug": str(raw_node.get("slug") or _slugify(raw_node.get("title") or node_type)),
            "title": str(raw_node.get("title") or node_type),
            "summary": str(raw_node.get("summary") or ""),
            "metadata": raw_node.get("metadata") if isinstance(raw_node.get("metadata"), dict) else {},
            "status": str(raw_node.get("status") or "active"),
            "tags": raw_node.get("tags") if isinstance(raw_node.get("tags"), list) else [],
        }
        row = supabase_client.upsert_knowledge_node(node_payload)
        if not row:
            raise HTTPException(502, f"Failed to upsert node slug={node_payload['slug']}")
        persisted_nodes.append(row)
        patch_nodes.append(node_payload)
        known[f"id:{row['id']}"] = row
        known[f"slug:{str(row.get('slug') or '').lower()}"] = row

    edges_to_persist: list[dict] = []
    for raw_edge in patch["edges_upsert"]:
        src_ref = _canonical_ref(raw_edge.get("source_ref"), persona_node)
        dst_ref = _canonical_ref(raw_edge.get("target_ref"), persona_node)
        if not src_ref or not dst_ref:
            raise HTTPException(422, "edges_upsert requires source_ref and target_ref")
        src = known.get(src_ref)
        dst = known.get(dst_ref)
        if not src or not dst:
            raise HTTPException(422, f"edge reference not found: source={src_ref} target={dst_ref}")
        edges_to_persist.append(
            {
                "source_ref": src_ref,
                "target_ref": dst_ref,
                "source_node_id": src["id"],
                "target_node_id": dst["id"],
                "relation_type": str(raw_edge.get("relation_type") or "contains").strip().lower(),
                "metadata": raw_edge.get("metadata") if isinstance(raw_edge.get("metadata"), dict) else {},
            }
        )

    violations, recommendations = _validate_sofia_patch(
        patch={"nodes_upsert": patch_nodes, "nodes_delete": patch["nodes_delete"], "edges_upsert": edges_to_persist, "edges_delete": patch["edges_delete"]},
        nodes_after_upsert=known,
        accept_unverified=bool(body.context.accept_unverified),
    )
    if violations:
        supabase_client.insert_event(
            {
                "event_type": "sofia_graph_command_rejected",
                "entity_type": "persona",
                "entity_id": persona_id,
                "persona_id": persona_id,
                "payload": {"persona_slug": persona_slug, "command": body.command, "violations": violations},
            },
            source="qa_contract.sofia_graph_command",
        )
        raise HTTPException(
            422,
            {
                "ok": False,
                "code": "GRAPH_VALIDATION_FAILED",
                "message": "Command violates canonical graph constraints.",
                "violations": violations,
            },
        )

    plan_tool_calls = list(plan.get("tool_calls") or [])

    for edge in edges_to_persist:
        supabase_client.upsert_knowledge_edge(
            source_node_id=edge["source_node_id"],
            target_node_id=edge["target_node_id"],
            relation_type=edge["relation_type"],
            persona_id=persona_id,
            weight=1.0,
            metadata=edge["metadata"] or {"primary_tree": True, "active": True},
        )
    plan_tool_calls.append(
        {
            "name": "persist-graph-patch",
            "score": 1.0,
            "result": {
                "nodes_upserted": len(persisted_nodes),
                "edges_upserted": len(edges_to_persist),
                "nodes_deleted": destructive_report["nodes_deleted"],
                "edges_deleted": destructive_report["edges_deleted"],
                "ok": True,
            },
        }
    )
    supabase_client.insert_event(
        {
            "event_type": "sofia_graph_command_applied",
            "entity_type": "persona",
            "entity_id": persona_id,
            "persona_id": persona_id,
            "payload": {
                "persona_slug": persona_slug,
                "command": body.command,
                "nodes_upsert": len(persisted_nodes),
                "edges_upsert": len(edges_to_persist),
                "destructive_report": destructive_report,
                "recommendations": recommendations,
            },
        },
        source="qa_contract.sofia_graph_command",
    )
    sofia_orchestrator.remember_turn(
        session_id=session_id,
        persona_slug=persona_slug,
        command=effective_command,
        operation_result=operation_tool,
        last_referenced_node=last_referenced_node,
    )
    fresh_state = sofia_orchestrator.get_session_state(session_id)
    plan_tool_calls.append(
        {
            "name": "refetch-graph",
            "score": 1.0,
            "result": {
                "ok": True,
                "persona_slug": persona_slug,
                "active_persona_slug": persona_slug,
            },
        }
    )
    if session_id:
        plan_json_patch = dict(plan_json_patch_base)
        plan_json_patch["graph_patch_queue"] = [patch]
        plan_json_patch["last_graph_operation"]["persisted"] = True
        plan_json = sofia_orchestrator.apply_plan_json_patch(
            session_id=session_id,
            persona_slug=persona_slug,
            patch=plan_json_patch,
            command=effective_command,
        )
    else:
        plan_json = sofia_orchestrator.get_or_create_plan_json(
            session_id=None,
            persona_slug=persona_slug,
            selected_node_id=selected_node_id or None,
        )
    return {
        "ok": True,
        "sofia_message": "Comando aplicado com cadeia canonica validada.",
        "graph_patch": patch,
        "persisted": True,
        "tool_calls": [
            {"tool": call.get("name"), "score": call.get("score"), "result": call.get("result")}
            for call in plan_tool_calls
        ],
        "threshold": plan.get("threshold"),
        "needs_clarification": False,
        "recommendations": recommendations,
        "destructive_report": destructive_report,
        "validation": {"canonical_chain_respected": True, "violations": []},
        "plan_json": plan_json,
        "conversation_context": {
            "session_id": session_id or None,
            "active_persona_slug": persona_slug,
            "memory_turns": len((fresh_state or {}).get("recent_turns") or []),
            "last_referenced_node": (fresh_state or {}).get("last_referenced_node") or None,
        },
    }


@router.post("/sofia/faq/accept")
def sofia_faq_accept(body: SofiaFaqAcceptBody, request: Request):
    """Persist accepted FAQ suggestions as pending/draft FAQ nodes connected to
    their branch parent. They are NOT sent to Embedded â€” they follow the normal
    approve -> publish-to-Embedded flow afterwards."""
    persona = _resolve_sofia_persona(request, _persona_ref(body.persona_slug, body.persona_ref))
    persona_id = persona.get("id")
    persona_slug = _persona_slug(persona)
    raw_parent = _strip_graph_id(body.parent_node_id)
    current_doc = graph_json_v2_store.load_current(persona_slug)
    if current_doc is None:
        raise HTTPException(409, "A published canonical Graph JSON is required")
    current_version, current_graph = current_doc
    parent_doc = next(
        (
            node
            for node in current_graph.nodes
            if node.id == raw_parent
            or node.id == body.parent_node_id
            or str((node.data or {}).get("knowledge_node_id") or "") == raw_parent
        ),
        None,
    )
    if parent_doc is None:
        raise HTTPException(404, "parent node not found in canonical Graph JSON")
    accepted = [item for item in body.suggestions if str(item.question or "").strip()]
    if not accepted:
        raise HTTPException(400, "no accepted suggestions to persist")
    count = sofia_faq_tool.clamp_count(body.faq_generation_count or len(accepted))
    faq_slug = _slugify(
        f"faq-{body.generated_from_node_slug or parent_doc.slug}-{hashlib.sha256('|'.join(item.question for item in accepted).encode('utf-8')).hexdigest()[:8]}"
    )
    faq_id = f"node:faq:{faq_slug}"
    branch_nodes = []
    cursor = parent_doc
    nodes_by_id = {node.id: node for node in current_graph.nodes}
    while cursor:
        branch_nodes.insert(0, {"node_type": cursor.node_type, "slug": cursor.slug, "label": cursor.label})
        cursor = nodes_by_id.get(cursor.parent_id or "")
    markdown = "\n\n".join(
        f"## {item.question.strip()}\n\n{str(item.answer or '').strip()}"
        for item in accepted
    ).strip()
    patch = {
        "nodes_upsert": [
            {
                "node_type": "faq",
                "slug": faq_slug,
                "title": f"FAQ â€” {parent_doc.label}",
                "summary": markdown[:400],
                "status": "pending_validation",
                "metadata": {
                    "markdown": markdown,
                    "markdown_document": True,
                    "question_count": len(accepted),
                    "questions": [
                        {"question": item.question.strip(), "answer": str(item.answer or "").strip()}
                        for item in accepted
                    ],
                    "source": sofia_faq_tool.SOURCE_TOOL,
                    "source_node_id": parent_doc.id,
                    "source_node_type": parent_doc.node_type,
                    "branch_path": branch_nodes,
                    "source_context": body.source_context or {},
                    "generated_from_node_id": body.generated_from_node_id,
                    "generated_from_node_slug": body.generated_from_node_slug,
                },
            }
        ],
        "edges_upsert": [
            {
                "source_ref": f"id:{parent_doc.id}",
                "target_ref": f"id:{faq_id}",
                "relation_type": _faq_parent_relation(parent_doc.node_type),
                "metadata": {"primary_tree": True},
            }
        ],
    }
    next_graph = graph_document_publisher.apply_sofia_patch(current_graph, patch)
    publication = graph_document_publisher.publish(
        graph=next_graph,
        persona_slug=persona_slug,
        brand_slug=next_graph.brand_slug,
        source="sofia.faq.accept",
        published_by=(auth_service.current_user(request) or {}).get("id"),
        expected_version=current_version,
        idempotency_key=f"sofia-faq:{persona_slug}:{current_version}:{faq_slug}",
    )
    return {
        "ok": True,
        "persona_slug": persona_slug,
        "parent_node_id": parent_doc.id,
        "faq_generation_count": count,
        "created": [{"node_id": faq_id, "question_count": len(accepted), "status": "pending_validation"}],
        "publication": publication,
    }

    # Legacy per-question direct-write implementation retained temporarily as
    # unreachable reference during rollout.
    parent = supabase_client.get_knowledge_node(raw_parent)
    if not parent:
        raise HTTPException(404, "parent node not found")
    parent_type = (body.parent_node_type or parent.get("node_type") or "").strip().lower()
    accepted = [s for s in body.suggestions if str(s.question or "").strip()]
    if not accepted:
        raise HTTPException(400, "no accepted suggestions to persist")
    count = sofia_faq_tool.clamp_count(body.faq_generation_count or len(accepted))
    relation = _faq_parent_relation(parent_type)
    created: list[dict] = []
    for suggestion in accepted:
        question = suggestion.question.strip()
        answer = str(suggestion.answer or "").strip()
        content = f"Pergunta: {question}\nResposta: {answer}" if answer else question
        item = knowledge_lifecycle.persist_pending_knowledge_item(
            persona_slug=persona_slug,
            title=question,
            content=content,
            content_type="faq",
            file_path=None,
            file_type="md",
            metadata={
                "faq_generation_count": count,
                "source_tool": sofia_faq_tool.SOURCE_TOOL,
                "parent_node_id": raw_parent,
                "parent_node_type": parent_type,
                "source_context": body.source_context or {},
                "generated_from_node_id": body.generated_from_node_id,
                "generated_from_node_slug": body.generated_from_node_slug,
                "question": question,
                "answer": answer,
            },
            tags=["faq", "sofia", sofia_faq_tool.SOURCE_TOOL],
        )
        node_id = (item.get("metadata") or {}).get("knowledge_node_id")
        edge = None
        if node_id:
            edge = supabase_client.upsert_knowledge_edge(
                raw_parent,
                node_id,
                relation,
                persona_id=persona_id,
                weight=0.9,
                metadata={"primary_tree": True, "active": True, "created_from": sofia_faq_tool.SOURCE_TOOL},
            )
        created.append(
            {
                "item_id": item.get("id"),
                "node_id": node_id,
                "edge_id": (edge or {}).get("id"),
                "question": question,
                "status": item.get("status"),
            }
        )
    try:
        supabase_client.update_knowledge_node(
            raw_parent,
            {"metadata": {**(parent.get("metadata") or {}), "faq_generation_count": count}},
            mark_related_faqs=False,
        )
    except Exception:
        pass
    return {
        "ok": True,
        "persona_slug": persona_slug,
        "parent_node_id": raw_parent,
        "faq_generation_count": count,
        "created": created,
    }


@router.post("/sofia/faq/append")
def sofia_faq_append(body: SofiaFaqAppendBody, request: Request):
    """Append accepted suggestions to a FAQ node's OWN Markdown body.

    The "Gerar" action on a FAQ node updates that same FAQ â€” it never creates a
    new FAQ node. The FAQ stays/returns to pending (draft); if it was already
    published, it is withdrawn from Embedded (its section is removed from the
    Embedded body) and must be approved + published again."""
    persona = _resolve_sofia_persona(request, _persona_ref(body.persona_slug, body.persona_ref))
    persona_id = persona.get("id")
    raw_node = _strip_graph_id(body.faq_node_id)
    persona_slug = _persona_slug(persona)
    current_doc = graph_json_v2_store.load_current(persona_slug)
    if current_doc is None:
        raise HTTPException(409, "A published canonical Graph JSON is required")
    current_version, current_graph = current_doc
    faq_doc = next(
        (
            item
            for item in current_graph.nodes
            if item.id in {raw_node, body.faq_node_id}
            or str((item.data or {}).get("knowledge_node_id") or "") == raw_node
        ),
        None,
    )
    if faq_doc is None or faq_doc.node_type != "faq":
        raise HTTPException(404, "FAQ node not found in canonical Graph JSON")
    accepted = [item for item in body.suggestions if str(item.question or "").strip()]
    if not accepted:
        raise HTTPException(400, "no accepted suggestions to append")
    appended = _format_faq_sections(accepted)
    existing_md = str((faq_doc.data or {}).get("markdown") or "").strip()
    new_md = (f"{existing_md}\n\n{appended}".strip() if existing_md else appended).rstrip() + "\n"
    next_graph = GraphJson.model_validate(current_graph.model_dump())
    updated_faq = next(item for item in next_graph.nodes if item.id == faq_doc.id)
    updated_faq.data = {
        **(updated_faq.data or {}),
        "markdown": new_md,
        "status": "pending_validation",
        "question_count": int((updated_faq.data or {}).get("question_count") or 0) + len(accepted),
    }
    next_graph.edges = [
        edge
        for edge in next_graph.edges
        if not (
            edge.source == faq_doc.id
            and next((item for item in next_graph.nodes if item.id == edge.target), None)
            and next(item for item in next_graph.nodes if item.id == edge.target).node_type == "embedded"
        )
    ]
    publication = graph_document_publisher.publish(
        graph=next_graph,
        persona_slug=persona_slug,
        brand_slug=next_graph.brand_slug,
        source="sofia.faq.append",
        published_by=(auth_service.current_user(request) or {}).get("id"),
        expected_version=current_version,
        idempotency_key=f"sofia-faq-append:{persona_slug}:{current_version}:{faq_doc.id}",
    )
    return {
        "ok": True,
        "faq_node_id": faq_doc.id,
        "appended_count": len(accepted),
        "status": "pending_validation",
        "markdown": new_md,
        "created_node": False,
        "publication": publication,
    }

    # Legacy direct-write implementation retained as unreachable rollout reference.
    node = supabase_client.get_knowledge_node(raw_node)
    if not node:
        raise HTTPException(404, "FAQ node not found")
    if str(node.get("node_type") or "").lower() != "faq":
        raise HTTPException(400, "Append is only valid on FAQ nodes")
    accepted = [s for s in body.suggestions if str(s.question or "").strip()]
    if not accepted:
        raise HTTPException(400, "no accepted suggestions to append")

    appended = _format_faq_sections(accepted)
    existing_md = str((node.get("metadata") or {}).get("markdown") or node.get("summary") or "").strip()
    new_md = (f"{existing_md}\n\n{appended}".strip() if existing_md else appended).rstrip() + "\n"

    item = None
    if node.get("source_table") == "knowledge_items" and node.get("source_id"):
        item = supabase_client.get_knowledge_item(str(node.get("source_id")))
    was_published = str((item or {}).get("status") or "").lower() in {"approved", "embedded"}

    supabase_client.update_knowledge_node(
        raw_node,
        {"metadata": {**(node.get("metadata") or {}), "markdown": new_md}, "status": "pending_validation"},
        mark_related_faqs=False,
    )
    if item and item.get("id"):
        supabase_client.update_knowledge_item(
            item["id"], {"content": new_md, "status": "pending", "curation_status": "draft"}
        )
    if was_published and item and item.get("id"):
        try:
            supabase_client.withdraw_faq_from_embedded(item["id"])
        except Exception:
            pass
        embedded_markdown.rebuild_embedded_markdown(persona_id)

    return {
        "ok": True,
        "faq_node_id": raw_node,
        "appended_count": len(accepted),
        "status": "pending",
        "markdown": new_md,
        "created_node": False,
    }


@router.get("/sofia/plan-json")
def sofia_get_plan_json(
    request: Request,
    session_id: str = Query(...),
    persona_slug: Optional[str] = Query(None),
    persona_ref: Optional[str] = Query(None),
):
    resolved_ref = _persona_ref(persona_slug, persona_ref) or "vz-lupas"
    persona = _resolve_sofia_persona(request, resolved_ref)
    slug = _persona_slug(persona)
    plan_json = sofia_orchestrator.get_or_create_plan_json(session_id=session_id, persona_slug=slug)
    return {"ok": True, "plan_json": plan_json}


@router.patch("/sofia/plan-json")
def sofia_patch_plan_json(body: SofiaPlanJsonPatchBody, request: Request):
    resolved_ref = _persona_ref(body.persona_slug, body.persona_ref) or "vz-lupas"
    persona = _resolve_sofia_persona(request, resolved_ref)
    slug = _persona_slug(persona)
    plan_json = sofia_orchestrator.apply_plan_json_patch(
        session_id=body.session_id,
        persona_slug=slug,
        patch=body.patch,
        command=body.command,
    )
    return {"ok": True, "plan_json": plan_json}

