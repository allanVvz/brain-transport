# -*- coding: utf-8 -*-
"""
WA Validator Service — generates test scripts from KB, tracks validation sessions,
and analyses conversation gaps to feed back into KB Intake.
"""

import asyncio
import difflib
import hashlib
import httpx
import json
import logging
import os
import random
import re
import subprocess
import threading
import time
import traceback
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from services import (
    agents_service,
    conversation_repetition,
    conversation_runtime,
    graph_agent_runtime_v3,
    graph_compiler_v3,
    graph_json_v2_store,
    graph_proof_checker_v3,
    media_ingest,
    n8n_client,
    supabase_client,
    validator_sofia_insights,
)

logger = logging.getLogger("wa_validator_service")

# Flows that collect the customer's name as one of their scripted steps --
# the only ones "known_name" initial state can meaningfully apply to (a
# flow with no name-collection step has nothing to pre-seed or omit).
_NAME_COLLECTING_FLOWS = {
    "sdr_qualificacao_carro", "sdr_troca_servico", "sdr_multiplos_servicos",
    "sdr_reativacao_pos_handoff",
}

_MODEL_DEFAULT = "none"
AVAILABLE_MODELS = {
    "none": "Determinístico — sem modelo",
}
_API_DIR = Path(__file__).resolve().parents[1]
_ROOT_DIR = Path(__file__).resolve().parents[2]
_WA_EXECUTOR = _ROOT_DIR / "dashboard" / "scripts" / "wa-validator.mjs"
_WA_NODE = os.environ.get("WA_VALIDATOR_NODE", "node")
_WA_RUNTIME = _ROOT_DIR / ".runtime" / "wa-validator"
_WA_PROFILE = Path(
    os.environ.get(
        "WA_VALIDATOR_PROFILE",
        str(_ROOT_DIR / ".runtime" / "wa-validator-profile"),
    )
)
_WA_ARTIFACTS = _ROOT_DIR / "test-artifacts" / "wa-validator"
_WA_RUNNER_URL = (os.environ.get("WA_VALIDATOR_RUNNER_URL") or "").rstrip("/")
_BRAIN_API_URL = os.environ.get("BRAIN_API_URL", "http://localhost:8080")
_CUSTOMER_PROFILES_PATH = _API_DIR / "evaluation" / "wa_validator_customer_profiles.json"
_SDR_FLOW_CORPUS_PATH = _API_DIR / "evaluation" / "sdr_flow_cases.json"

# WA Validator sessions live in Supabase, not a plain in-process dict.
# Confirmed live 2026-08-08: production runs GUNICORN_WORKERS=2, so a
# "generate script" request and the following "run" request had roughly a
# coin-flip chance of landing on different worker processes -- the second
# worker's in-memory dict never had the session the first one created,
# producing "Sessão não encontrada" for a session that genuinely existed,
# just in the other process's memory.


def _session_get(session_id: str) -> Optional[dict]:
    return supabase_client.get_wa_validator_session(session_id)


def _session_create(
    session_id: str, data: dict, *, persona_slug: Optional[str] = None, flow_id: Optional[str] = None
) -> None:
    supabase_client.upsert_wa_validator_session(session_id, data, persona_slug=persona_slug, flow_id=flow_id)


def _session_update(session_id: str, **fields) -> dict:
    """Read-modify-write merge patch.

    Best-effort, not transactional -- acceptable for this ephemeral
    test-tooling data, where each session_id has at most one active writer
    in practice (a single validator run driving its own session).
    """
    current = supabase_client.get_wa_validator_session(session_id) or {}
    current.update(fields)
    current["updated_at"] = datetime.now(timezone.utc).isoformat()
    supabase_client.upsert_wa_validator_session(session_id, current)
    return current


def _session_list(
    *, persona_slug: str | None = None, since_hours: int | None = None, limit: int = 100,
) -> list[dict]:
    return supabase_client.list_wa_validator_sessions(
        limit=limit, persona_slug=persona_slug, since_hours=since_hours,
    )

# ── Bot registry ───────────────────────────────────────────────────────────────
_BOT_REGISTRY: list[dict] = []

_KNOWN_CONVERSATION_MODES = {"deterministic", "n8n_agents", "orquestrador"}


def _resolve_conversation_mode(
    persona_id: str | None, routing: dict, active_binding: dict | None = None,
) -> str:
    """The routing switch that actually governs live dispatch.

    Confirmed live 2026-08-08: this previously read only the legacy
    persona.process_mode column, so the validator tested a different engine
    (n8n_agents) than the one actually handling real WhatsApp traffic
    (deterministic, per the active binding's decision_owner). A validation
    session can otherwise fail every step because it POSTs to an n8n
    webhook nobody maintains, while real customers were being served by the
    deterministic graph_agent_runtime_v3 pipeline the whole time. Mirrors
    routes.personas._mask_routing's resolution exactly, so the validator and
    the settings UI never disagree about which engine is live.
    """
    process_mode = routing.get("process_mode") or "internal"
    fallback = "n8n_agents" if process_mode == "n8n" else "deterministic"
    if not persona_id:
        return fallback
    if active_binding is None:
        active_binding = next(
            (
                row
                for row in supabase_client.get_workflow_bindings(persona_id)
                if row.get("active")
            ),
            None,
        )
    decision_owner = ((active_binding or {}).get("metadata") or {}).get("decision_owner")
    return decision_owner if decision_owner in _KNOWN_CONVERSATION_MODES else fallback


def bots(allowed_persona_ids: set[str] | None = None) -> list:
    """Return available bots: static registry + any active personas not yet listed."""
    result = [
        row for row in _BOT_REGISTRY
        if allowed_persona_ids is None or str(row.get("persona_id") or "") in allowed_persona_ids
    ]
    registered = {b["persona_slug"] for b in _BOT_REGISTRY}
    try:
        for p in supabase_client.get_personas():
            if allowed_persona_ids is not None and str(p.get("id") or "") not in allowed_persona_ids:
                continue
            slug = p.get("slug", "")
            if slug and slug not in registered:
                agent_name = p.get("name", slug)
                agent_slug = "agent"
                agent_node = None
                try:
                    current = graph_json_v2_store.load_current(slug)
                    graph = current[1] if current else None
                    agent_node = next(
                        (
                            node
                            for node in (graph.nodes if graph else [])
                            if (node.data or {}).get("metadata", {}).get(
                                "agent_slug"
                            )
                        ),
                        None,
                    )
                    if agent_node:
                        agent_name = agent_node.label
                        agent_slug = str(
                            (agent_node.data or {}).get("metadata", {}).get(
                                "agent_slug"
                            )
                            or agent_node.slug
                        )
                except Exception:
                    pass
                result.append({
                    "id": slug,
                    "bot_name": agent_name,
                    "agent_slug": agent_slug,
                    "whatsapp_phone": (
                        (agent_node.data or {}).get("metadata", {}).get(
                            "whatsapp_phone"
                        )
                        if agent_node
                        else None
                    ),
                    "label": f"{agent_name} — {p.get('name', slug)}",
                    "persona_slug": slug,
                    "persona_id": p.get("id"),
                    "description": p.get("description", ""),
                })
                registered.add(slug)
    except Exception as exc:
        raise RuntimeError("Não foi possível carregar as personas do WA Validator") from exc
    return result


def bootstrap(persona_slug: str) -> dict:
    """One consistent, non-secret validator snapshot for the selected persona."""
    persona = supabase_client.get_persona(persona_slug) or {}
    if not persona:
        raise ValueError("Persona não encontrada")
    persona_id = str(persona.get("id") or "")
    routing = supabase_client.get_persona_routing(persona_slug) or {}
    bindings = supabase_client.get_workflow_bindings(persona_id)
    active_binding = next((row for row in bindings if row.get("active")), None) or {}
    metadata = active_binding.get("metadata") or {}
    # The active v3 publication is the runtime authority and is already scoped
    # to this persona.  Reading the legacy Graph JSON v2 object from Storage on
    # every bootstrap was measured at 2.3-2.8s by itself in production, while
    # every other scoped query together took only tens of milliseconds.  Use
    # the compiled publication's persona node when available; retain the v2
    # fallback for personas that have not published v3 yet.
    publication = supabase_client.get_active_graph_publication(persona_id) or {}
    document = publication.get("document_json") or {}
    published_persona_node = next(
        (
            node for node in (document.get("node_by_id") or {}).values()
            if str(node.get("node_type") or "").lower() == "persona"
        ),
        None,
    )
    if published_persona_node:
        agent_data = published_persona_node.get("data") or {}
        agent_metadata = agent_data.get("metadata") or {}
        agent_name = str(
            published_persona_node.get("title")
            or persona.get("name")
            or persona_slug
        )
        business_model = str(agent_data.get("business_model") or "").strip() or None
    else:
        _version, _checksum, graph = _published_graph(persona_slug)
        agent_node = next(
            (
                node for node in graph.nodes
                if str(getattr(node, "node_type", "") or "").lower() == "persona"
            ),
            None,
        )
        agent_data = (agent_node.data or {}) if agent_node else {}
        agent_metadata = agent_data.get("metadata") or {}
        agent_name = (
            agent_node.label if agent_node else persona.get("name", persona_slug)
        )
        business_model = conversation_runtime._business_model(graph)
    persona_config = persona.get("config") or {}
    resolved_agent_slug = str(
        agent_metadata.get("agent_slug")
        or agent_data.get("agent_slug")
        or persona_config.get("agent_slug")
        or ((persona_config.get("agent") or {}).get("slug") if isinstance(persona_config.get("agent"), dict) else "")
        or "agent"
    )
    compatible_flows = _flows_for_business_model(business_model)
    return {
        "persona": {"id": persona_id, "slug": persona_slug, "name": persona.get("name")},
        "routing": {
            "conversation_mode": _resolve_conversation_mode(
                persona_id, routing, active_binding,
            ),
            "process_mode": routing.get("process_mode"),
        },
        "binding": {
            "id": active_binding.get("id"),
            "provider": active_binding.get("provider"),
            "connection_status": active_binding.get("connection_status"),
            "decision_owner": metadata.get("decision_owner"),
            "pipeline_contract": metadata.get("pipeline_contract"),
        },
        "bot": {
            "id": persona_slug,
            "bot_name": agent_name,
            "agent_slug": resolved_agent_slug,
            "whatsapp_phone": agent_metadata.get("whatsapp_phone"),
            "label": f"{agent_name} — {persona.get('name', persona_slug)}",
            "persona_slug": persona_slug,
            "persona_id": persona_id,
            "description": persona.get("description", ""),
        },
        "flows": compatible_flows,
        "sessions": list_sessions(
            persona_slug=persona_slug, since_hours=12, limit=25,
        ),
    }


_FLOWS = {
    "compra_simples": "Fluxo de compra simples: cliente pergunta sobre produto, recebe info, confirma compra.",
    "duvida_frete": "Fluxo de dúvida sobre frete/entrega: cliente pergunta prazo e valor de frete.",
    "saudacao_despedida": "Fluxo básico: saudação, pergunta simples, despedida.",
    "produto_especifico": "Fluxo de produto específico: cliente nomeia produto, bot responde com detalhes e CTA.",
    "reclamacao": "Fluxo de reclamação/insatisfação: cliente reclama, bot reconhece e escalona.",
    "atendente_humano": "Pedido explícito de atendente com handoff.",
    "produto_inexistente": "Produto inexistente após uma pergunta de esclarecimento.",
    "sem_evidencia": "Pergunta comercial sem evidência aprovada no grafo.",
    "produto_ambiguo": "Nome de produto ambíguo que exige esclarecimento.",
    "mensagem_duplicada": "Retry idempotente da mesma mensagem Meta.",
    "estagio_monotonic": "Mensagem curta não pode rebaixar o estágio.",
    "classifier_failure": "Falha do classificador determinístico gera handoff.",
    "invalid_decision_schema": "Saída fora do contrato gera handoff.",
    "delivery_callback": "Callbacks sent/delivered/read/failed são reconciliados.",
    "sdr_qualificacao_carro": (
        "Agendamento orientado pelo grafo: acompanha os campos publicados, "
        "responde interrupções e conclui somente no handoff autorizado."
    ),
    "sdr_troca_servico": (
        "Troca de ramo orientada pelo grafo: preserva facts compatíveis e "
        "recalcula somente os campos realmente faltantes."
    ),
    "sdr_multiplos_servicos": (
        "Múltiplos serviços na mesma conversa: adiciona um segundo ramo sem "
        "perder o primeiro e conclui somente quando o conjunto estiver completo."
    ),
    "sdr_reativacao_pos_handoff": (
        "Qualifica e confirma um pedido, reativa somente o lead sintético e "
        "prova que Oi/Oii permanecem saudações sem reiniciar a coleta."
    ),
    "sdr_sales_retail": (
        "Qualificação sales orientada pelo grafo para uso próprio/varejo."
    ),
    "sdr_sales_reseller": (
        "Qualificação sales orientada pelo grafo para atacado/revenda."
    ),
    "sdr_sales_branch_switch": (
        "Troca entre audiências sales, preservando apenas fatos compatíveis."
    ),
    "sdr_sales_knowledge_gap": (
        "Pergunta comercial sem fonte deve ser deferida sem invenção."
    ),
}

# A fixed name (previously always "Allan") made every validator run
# indistinguishable and never exercised the fact-extraction path for any
# other name. One is picked per generate_script() call so consecutive runs
# actually vary, covering both genders instead of skewing the sample.
_CLIENT_NAMES = [
    "Allan", "Bruno", "Carlos", "Diego", "Eduardo", "Fabio", "Gustavo",
    "Marcos", "Rafael", "Thiago",
    "Ana", "Beatriz", "Camila", "Fernanda", "Gabriela", "Helena", "Isabela",
    "Juliana", "Larissa", "Patricia",
]

# Which business_model(s) (persona_node.data.business_model, same field
# services.conversation_runtime._business_model reads) a flow's scripted
# messages actually make sense for. Confirmed live 2026-08-08: running
# "compra_simples" (asks price/quantity of a "produto") against an
# appointment persona with no product nodes produced a
# looping, self-contradicting conversation -- not a pipeline bug, a
# nonsensical test. Flows absent here (or mapped to an empty set) are
# treated as valid for any business model.
_FLOW_BUSINESS_MODELS: dict[str, set[str]] = {
    "compra_simples": {"sales"},
    "duvida_frete": {"sales"},
    "produto_especifico": {"sales"},
    "produto_inexistente": {"sales"},
    "produto_ambiguo": {"sales"},
    "mensagem_duplicada": {"sales"},
    "estagio_monotonic": {"sales"},
    "sdr_qualificacao_carro": {"appointment"},
    "sdr_troca_servico": {"appointment"},
    "sdr_multiplos_servicos": {"appointment"},
    "sdr_reativacao_pos_handoff": {"appointment"},
    "sdr_sales_retail": {"sales"},
    "sdr_sales_reseller": {"sales"},
    "sdr_sales_branch_switch": {"sales"},
    "sdr_sales_knowledge_gap": {"sales"},
}


def _extract_json(text: str) -> dict:
    """Parse JSON from Claude response regardless of markdown fences or surrounding text."""
    text = text.strip()
    # Find the outermost { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    # Fallback: try to parse as-is (handles plain JSON with no fences)
    return json.loads(text)


def _published_graph(persona_slug: str):
    current = graph_json_v2_store.load_current(persona_slug)
    if not current:
        raise ValueError(
            f"Nenhum Graph JSON v2 publicado para {persona_slug}"
        )
    version, graph = current
    event = graph_json_v2_store.latest_event(persona_slug) or {}
    checksum = (
        (event.get("payload") or {}).get("checksum")
        or graph_json_v2_store.checksum_graph(graph)
    )
    if graph.status != "published" or not graph.validation.is_valid:
        raise ValueError("Graph JSON v2 publicado não está válido")
    return version, checksum, graph


def _build_graph_context(persona_slug: str) -> tuple[str, int, str, object]:
    if graph_json_v2_store.load_current(persona_slug):
        version, checksum, graph = _published_graph(persona_slug)
    else:
        persona = supabase_client.get_persona(persona_slug) or {}
        publication = supabase_client.get_active_graph_publication(
            str(persona.get("id") or "")
        )
        if not publication:
            raise ValueError(
                f"Nenhum Graph JSON v2 ou GraphRAG v3 publicado para {persona_slug}"
            )
        document = publication.get("document_json") or {}
        document_without_checksum = dict(document) if isinstance(document, dict) else {}
        document_checksum = str(document_without_checksum.pop("checksum", ""))
        publication_checksum = str(publication.get("checksum") or "")
        document_persona = document.get("persona") if isinstance(document, dict) else {}
        document_nodes = document.get("nodes") if isinstance(document, dict) else None
        if (
            str(publication.get("status") or "") != "active"
            or not document_checksum
            or document_checksum != publication_checksum
            or graph_compiler_v3.canonical_checksum(document_without_checksum)
            != document_checksum
            or not isinstance(document_persona, dict)
            or str(document_persona.get("id") or "") != str(persona.get("id") or "")
            or str(document_persona.get("slug") or "") != persona_slug
            or not isinstance(document_nodes, list)
            or not document_nodes
        ):
            raise ValueError("Publicação GraphRAG v3 ativa está inconsistente")
        graph = SimpleNamespace(
            nodes=[
                SimpleNamespace(
                    id=str(node.get("id") or ""),
                    node_type=str(node.get("node_type") or "knowledge"),
                    slug=str(node.get("slug") or node.get("id") or ""),
                    label=str(node.get("title") or node.get("slug") or ""),
                    data={
                        **dict(node.get("data") or {}),
                        "status": str(node.get("status") or ""),
                    },
                )
                for node in document_nodes
            ],
            status="published",
            validation=SimpleNamespace(is_valid=True),
        )
        version = int(publication["version"])
        checksum = str(publication["checksum"])
    lines: list[str] = []
    for node in graph.nodes:
        data = node.data or {}
        if data.get("active", True) is False:
            continue
        if str(data.get("status") or "").lower() not in {
            "approved",
            "validated",
            "active",
            "ativo",
        }:
            continue
        lines.append(
            f"[{node.id}] {node.node_type}/{node.slug}: "
            f"{node.label} {str(data.get('markdown') or '')[:300]}"
        )
    return "\n".join(lines), version, str(checksum), graph


def _customer_profile(business_model: str) -> dict:
    try:
        payload = json.loads(_CUSTOMER_PROFILES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Perfil de cliente do WA Validator inválido: {exc}") from exc
    profile = payload.get(business_model)
    if not isinstance(profile, dict):
        raise ValueError(
            f"WA Validator não possui perfil de cliente para business_model={business_model}"
        )
    return profile


def _sdr_flow_corpus() -> dict:
    try:
        payload = json.loads(_SDR_FLOW_CORPUS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Corpus SDR compartilhado inválido: {exc}") from exc
    if payload.get("version") != 1 or not isinstance(payload.get("cases"), list):
        raise ValueError("Corpus SDR compartilhado não possui contrato versionado")
    return payload


def _appointment_branch_candidates(document: dict) -> list[tuple[str, dict, dict]]:
    candidates: list[tuple[str, dict, dict]] = []
    for anchor in document.get("branch_anchors") or []:
        contract = (document.get("branch_contracts") or {}).get(anchor) or {}
        node = (document.get("node_by_id") or {}).get(anchor) or {}
        if contract.get("fields") and node:
            candidates.append((str(anchor), node, contract))
    if not candidates:
        raise ValueError("Publicação ativa não possui branch de qualificação testável")
    max_fields = max(len(contract.get("fields") or []) for _, _, contract in candidates)
    # Avoid selecting a tiny exception/handoff branch when full qualification
    # branches exist, while staying generic for graphs with a single branch.
    floor = max(1, max_fields - 2)
    return [row for row in candidates if len(row[2].get("fields") or []) >= floor]


def _branch_identity_field(contract: dict, anchor: str) -> dict | None:
    return next(
        (
            field for field in contract.get("fields") or []
            if str(field.get("owner_node_id") or "") == anchor
            and field.get("branch_selection_field") is True
        ),
        next(
            (
                field for field in contract.get("fields") or []
                if str(field.get("owner_node_id") or "") == anchor
            ),
            None,
        ),
    )


def _graph_doubt(
    document: dict, contract: dict, branch_anchor_node_id: str | None = None,
) -> dict | None:
    qualification_questions = set((contract.get("questions") or {}).keys())
    anchor = str(
        branch_anchor_node_id or contract.get("branch_anchor_node_id") or ""
    )
    branch = (document.get("node_by_id") or {}).get(anchor) or {}
    title = str(branch.get("title") or branch.get("slug") or "").strip()
    for node_id in contract.get("closure_node_ids") or []:
        if node_id in qualification_questions:
            continue
        node = (document.get("node_by_id") or {}).get(node_id) or {}
        if str(node.get("node_type") or "").lower() != "faq":
            continue
        data = node.get("data") or {}
        question = str(data.get("question") or node.get("title") or "").strip()
        if question:
            expected = [node_id]
            normalized_question = _semantic_fold(question)
            normalized_title = _semantic_fold(title)
            existence_verbs = (" fazem ", " oferecem ", " realizam ", " trabalham com ")
            operational_terms = (" horario ", " agenda ", " data ", " dia ", " amanha ")
            padded = f" {normalized_question} "
            if (
                anchor and normalized_title and normalized_title in normalized_question
                and any(verb in padded for verb in existence_verbs)
                and not any(term in padded for term in operational_terms)
            ):
                expected.append(anchor)
            return {"text": question, "expected_evidence_node_ids": expected}
    if anchor and title:
        return {
            "text": f"Vocês oferecem {title}?",
            "expected_evidence_node_ids": [anchor],
        }
    return None


def _semantic_appointment_script(
    *, publication: dict, flow_id: str, initial_state: str,
) -> dict:
    document = publication.get("document_json") or {}
    branches = _appointment_branch_candidates(document)
    anchor, branch_node, contract = random.choice(branches)
    identity_field = _branch_identity_field(contract, anchor)
    if not identity_field:
        raise ValueError(
            f"Branch {anchor} não declara um campo de identidade pertencente ao próprio branch"
        )
    profile = _customer_profile("appointment")
    configured_answers = profile.get("answers") or {}
    answers: dict[str, dict] = {}
    missing_examples: list[str] = []
    identity_key = str(identity_field.get("key") or "")
    identity_value = str(branch_node.get("slug") or branch_node.get("title") or anchor)
    for field in contract.get("fields") or []:
        key = str(field.get("key") or "")
        if not key or key == identity_key:
            continue
        answer = configured_answers.get(key)
        if not isinstance(answer, dict) or not str(answer.get("text") or "").strip():
            missing_examples.append(key)
            continue
        answers[key] = {
            "text": str(answer["text"]),
            "value": answer.get("value"),
        }
    if missing_examples:
        raise ValueError(
            "Perfil de cliente do WA Validator não cobre os campos publicados: "
            + ", ".join(sorted(missing_examples))
        )
    branch_title = str(branch_node.get("title") or branch_node.get("slug") or anchor)
    opening = {
        "text": f"Olá! Tenho interesse em {branch_title}.",
        "intended_facts": {identity_key: identity_value},
        "expected_branch_node_id": anchor,
    }
    switch = None
    driver_questions = dict(contract.get("questions") or {})
    driver_required_fields = [
        str(field.get("key") or "") for field in contract.get("fields") or []
    ]
    if flow_id in {"sdr_troca_servico", "sdr_multiplos_servicos"} and len(branches) > 1:
        alternatives = [row for row in branches if row[0] != anchor]
        second_anchor, second_node, second_contract = random.choice(alternatives)
        second_identity = _branch_identity_field(second_contract, second_anchor)
        if second_identity:
            second_missing: list[str] = []
            for field in second_contract.get("fields") or []:
                key = str(field.get("key") or "")
                if not key or key == str(second_identity.get("key") or ""):
                    continue
                answer = configured_answers.get(key)
                if not isinstance(answer, dict) or not str(answer.get("text") or "").strip():
                    second_missing.append(key)
                    continue
                answers.setdefault(key, {
                    "text": str(answer["text"]),
                    "value": answer.get("value"),
                })
            if second_missing:
                raise ValueError(
                    "Perfil de cliente do WA Validator não cobre os campos publicados: "
                    + ", ".join(sorted(second_missing))
                )
            driver_questions.update(second_contract.get("questions") or {})
            driver_required_fields.extend(
                str(field.get("key") or "")
                for field in second_contract.get("fields") or []
                if str(field.get("key") or "") not in driver_required_fields
            )
            second_title = str(second_node.get("title") or second_node.get("slug") or second_anchor)
            additive = flow_id == "sdr_multiplos_servicos"
            switch = {
                "after_answered_fields": 2,
                "text": (
                    f"Também quero {second_title}."
                    if additive else f"Na verdade, prefiro {second_title}."
                ),
                "intended_facts": {
                    str(second_identity.get("key") or identity_key): str(
                        second_node.get("slug") or second_node.get("title") or second_anchor
                    )
                },
                "expected_branch_node_id": second_anchor,
                "expected_active_branch_node_ids": (
                    [anchor, second_anchor] if additive else [second_anchor]
                ),
            }
    doubt = _graph_doubt(document, contract, anchor)
    if not doubt:
        raise ValueError(
            f"Branch {anchor} não possui FAQ publicada para testar dúvida/interrupção"
        )
    driver = {
        "mode": "semantic_graph_v1",
        "opening": opening,
        "answers": answers,
        "required_fields": driver_required_fields,
        "questions": driver_questions,
        "branch_anchor_node_id": anchor,
        "max_turns": sum(
            len(item_contract.get("fields") or [])
            for _item_anchor, _item_node, item_contract in branches[:2]
        ) + 6,
        "expected_handoff": True,
        "switch": switch,
        "doubt": doubt,
        "interruption_after_answered_fields": max(
            0, len(driver_required_fields) - 1,
        ),
        "second_ignore": (
            {"text": "Podemos seguir sem essa informação?"}
            if flow_id == "sdr_multiplos_servicos" else None
        ),
        "confirmation": {"text": "Sim"},
        "question_repetition": dict(
            (contract.get("conversation_policy") or {}).get("question_repetition") or {}
        ),
        "initial_known_fields": (
            ["nome_cliente"] if initial_state == "known_name" else []
        ),
        "regression_corpus_ids": [
            str(item.get("id")) for item in _sdr_flow_corpus().get("cases") or []
            if item.get("id")
        ],
        "post_handoff_greetings": (
            [
                {
                    "text": str(item["message"]),
                    "kind": "post_handoff_greeting",
                    "intended_facts": {},
                    "resume_after_handoff": index == 0,
                }
                for index, item in enumerate(_sdr_flow_corpus().get("cases") or [])
                if item.get("id") in {
                    "greeting_after_handoff_oi", "greeting_after_handoff_oii",
                }
            ]
            if flow_id == "sdr_reativacao_pos_handoff" else []
        ),
    }
    known_name = (configured_answers.get("nome_cliente") or {}).get("value")
    return {
        "flow_description": _FLOWS.get(flow_id, flow_id),
        "expected_knowledge": [
            f"graph:{publication.get('version')}:{publication.get('checksum')}"
        ],
        "steps": [opening],
        "driver": driver,
        "expected_dialogue": {
            "known_name": known_name,
            "known_service": identity_value,
            "client_name_omitted": initial_state == "known_name",
        },
    }


def _semantic_sales_script(*, publication: dict, flow_id: str) -> dict:
    """Build a graph-driven sales customer without fixed products or prices."""
    document = publication.get("document_json") or {}
    branches = _appointment_branch_candidates(document)
    preferred_terms = (
        ("revenda", "atacado")
        if flow_id in {"sdr_sales_reseller", "sdr_sales_branch_switch"}
        else ("varejo", "uso-proprio", "uso próprio")
    )

    def branch_text(row: tuple[str, dict, dict]) -> str:
        anchor, node, _contract = row
        return _semantic_fold(
            " ".join([
                anchor,
                str(node.get("slug") or ""),
                str(node.get("title") or ""),
                " ".join(str(value) for value in (node.get("data") or {}).get("aliases") or []),
            ])
        )

    selected = next(
        (row for row in branches if any(_semantic_fold(term) in branch_text(row) for term in preferred_terms)),
        branches[0],
    )
    anchor, branch_node, contract = selected
    identity_field = _branch_identity_field(contract, anchor)
    if not identity_field:
        raise ValueError(f"Branch {anchor} não declara branch selector")
    identity_key = str(identity_field.get("key") or "")
    identity_value = str(branch_node.get("slug") or branch_node.get("title") or anchor)
    profile = _customer_profile("sales")
    configured_answers = profile.get("answers") or {}
    answers: dict[str, dict] = {}
    missing_examples: list[str] = []
    for field in contract.get("fields") or []:
        key = str(field.get("key") or "")
        if not key or key == identity_key:
            continue
        answer = configured_answers.get(key)
        if not isinstance(answer, dict) or not str(answer.get("text") or "").strip():
            missing_examples.append(key)
            continue
        answers[key] = {"text": str(answer["text"]), "value": answer.get("value")}
    if missing_examples:
        raise ValueError(
            "Perfil sales do WA Validator não cobre os campos publicados: "
            + ", ".join(sorted(missing_examples))
        )

    openings = profile.get("openings") or {}
    opening_text = str(openings.get(flow_id) or "").strip()
    if not opening_text:
        opening_text = f"Olá! Tenho interesse em {branch_node.get('title') or identity_value}."
    opening = {
        "text": opening_text,
        "intended_facts": {identity_key: identity_value},
        "expected_branch_node_id": anchor,
    }
    required_fields = [
        str(field.get("key") or "") for field in contract.get("fields") or []
        if str(field.get("key") or "")
    ]
    driver_questions = dict(contract.get("questions") or {})
    doubt = {
        "text": str(profile.get("unsupported_commercial_question") or "Qual é o preço?"),
        "expected_evidence_node_ids": [],
        "forbidden_claim_patterns": [
            r"R\$\s*\d", r"\bestoque\s+(?:disponível|garantido)\b",
            r"\bentrega\s+em\s+\d+", r"\bpedido\s+mínimo\s+(?:é|de)\b",
        ],
    }
    switch = None
    if flow_id == "sdr_sales_branch_switch" and len(branches) > 1:
        alternative = next((row for row in branches if row[0] != anchor), None)
        if alternative:
            second_anchor, second_node, second_contract = alternative
            second_identity = _branch_identity_field(second_contract, second_anchor)
            if second_identity:
                switch = {
                    "after_answered_fields": 1,
                    "text": str(profile.get("branch_switch_text") or "Na verdade, é para uso próprio."),
                    "intended_facts": {
                        str(second_identity.get("key") or identity_key): str(
                            second_node.get("slug") or second_node.get("title") or second_anchor
                        )
                    },
                    "expected_branch_node_id": second_anchor,
                    "expected_active_branch_node_ids": [second_anchor],
                }
                for field in second_contract.get("fields") or []:
                    key = str(field.get("key") or "")
                    if key and key != str(second_identity.get("key") or ""):
                        answer = configured_answers.get(key)
                        if isinstance(answer, dict) and str(answer.get("text") or "").strip():
                            answers.setdefault(key, {
                                "text": str(answer["text"]), "value": answer.get("value")
                            })
                for key, question in (second_contract.get("questions") or {}).items():
                    driver_questions.setdefault(key, question)
                for field in second_contract.get("fields") or []:
                    key = str(field.get("key") or "")
                    if key and key not in required_fields:
                        required_fields.append(key)

    driver = {
        "mode": "semantic_graph_v1",
        "opening": opening,
        "answers": answers,
        "required_fields": required_fields,
        "questions": driver_questions,
        "branch_anchor_node_id": anchor,
        "max_turns": len(required_fields) + 7,
        "expected_handoff": True,
        "switch": switch,
        "doubt": doubt,
        "interruption_after_answered_fields": max(1, len(required_fields) - 1),
        "confirmation": {"text": "Sim"},
        "question_repetition": dict(
            (contract.get("conversation_policy") or {}).get("question_repetition") or {}
        ),
    }
    return {
        "flow_description": _FLOWS.get(flow_id, flow_id),
        "expected_knowledge": [
            f"graph:{publication.get('version')}:{publication.get('checksum')}"
        ],
        "steps": [opening],
        "driver": driver,
        "expected_dialogue": {
            "branch_anchor_node_id": anchor,
            "unsupported_claims_forbidden": True,
        },
    }


def _deterministic_script(
    flow_id: str,
    *,
    product: object | None,
    graph_version: int,
    graph_checksum: str,
    omit_client_name: bool = False,
) -> dict:
    product_id = getattr(product, "id", None)
    product_data = getattr(product, "data", {}) or {}
    product_metadata = product_data.get("metadata") or {}
    product_name = (
        product_metadata.get("display_name")
        or getattr(product, "label", None)
        or "o primeiro produto ativo"
    )
    price = product_data.get("price") or product_metadata.get("price") or {}
    unit_price = (
        float(price.get("amount"))
        if isinstance(price, dict) and price.get("amount") is not None
        else None
    )
    common_expected = [f"graph:{graph_version}:{graph_checksum}"]
    if product_id:
        common_expected.insert(0, f"evidence:{product_id}")
    client_name = random.choice(_CLIENT_NAMES)
    scenarios = {
        "compra_simples": [
            "Oi",
            f"Quanto custa {product_name}?",
            "quero 2",
            "mude para 3",
            "qual o total?",
            client_name,
            "Rua QA, 100, Canoas",
            "Sim",
        ],
        "saudacao_despedida": ["Oi", "Quais categorias estão disponíveis?", "Obrigado"],
        "produto_especifico": [f"Vocês têm {product_name}?", "quanto custa?", "quero 2"],
        "duvida_frete": ["Oi", "Como funciona a entrega?"],
        "reclamacao": ["Estou com um problema e quero fazer uma reclamação"],
        "atendente_humano": ["Quero falar com um atendente humano"],
        "produto_inexistente": ["Vocês têm o produto QA inexistente?", "É esse mesmo"],
        "sem_evidencia": ["Qual é uma condição comercial que não está documentada?"],
        "produto_ambiguo": ["Quero aquele produto", "Não sei o nome completo"],
        "mensagem_duplicada": [f"Quanto custa {product_name}?", f"Quanto custa {product_name}?"],
        "estagio_monotonic": [f"Quero 2 de {product_name}", "Oi"],
        "classifier_failure": ["[QA_CLASSIFIER_FAILURE]"],
        "invalid_decision_schema": ["[QA_INVALID_DECISION_SCHEMA]"],
        "delivery_callback": [f"Quero 1 de {product_name}"],
    }
    messages = scenarios.get(flow_id)
    if not messages:
        raise ValueError(f"Fluxo determinístico desconhecido: {flow_id}")
    expected_dialogue = {
        "product_name": product_name,
        "unit_price": unit_price,
        "final_quantity": 3 if flow_id == "compra_simples" else None,
        "final_total": (
            round(unit_price * 3, 2)
            if unit_price is not None and flow_id == "compra_simples"
            else None
        ),
        "forbidden_terms": [],
    }
    return {
        "flow_description": _FLOWS.get(flow_id, flow_id),
        "expected_knowledge": common_expected,
        "steps": [{"text": text, "wait": 10} for text in messages],
        "expected_dialogue": expected_dialogue,
    }


def _resolve_initial_state(requested: str | None, flow_id: str) -> str:
    """Normalize the requested initial state to "cold" or "known_name".

    "random" and anything not applicable to this flow (no name-collection
    step) resolve to "cold" -- there is nothing to pre-seed or omit for a
    flow that never asks for the name in the first place.
    """
    if flow_id not in _NAME_COLLECTING_FLOWS:
        return "cold"
    if requested == "random":
        return random.choice(["cold", "known_name"])
    if requested == "known_name":
        return "known_name"
    return "cold"


def generate_script(
    persona_slug: str,
    flow_id: str,
    target_contact: str,
    model: str = _MODEL_DEFAULT,
    initial_state: str | None = None,
) -> dict:
    persona = supabase_client.get_persona(persona_slug)
    if not persona:
        raise ValueError(f"Persona não encontrada: {persona_slug}")

    persona_id = persona["id"]
    persona_name = persona.get("name", persona_slug)
    if not str(target_contact or "").strip():
        raise ValueError("Contato WhatsApp inválido")
    kb_ctx, graph_version, graph_checksum, graph = _build_graph_context(
        persona_slug
    )
    # Confirmed live 2026-08-09: for any persona actually running
    # graph_agent_runtime_v3, every real turn reports the v3
    # compiler's own publication version/checksum (graph_agent_runtime_v3.
    # build_context() reads it from graph_publications, never from the
    # legacy v2.1 store) -- an entirely separate counter from the v2.1
    # store version above. Baking the v2.1 version into "expected
    # knowledge" and the script label meant analyze_gaps() compared two
    # counters that were never the same thing by design, always reporting
    # a false "high" severity graph-lineage gap for these personas and
    # dragging overall_score down over nothing. When an active v3
    # publication exists, it -- not the v2.1 store -- is what a real turn
    # will actually report, so it is the correct "expected" baseline.
    v3_publication = supabase_client.get_active_graph_publication(persona_id)
    if v3_publication:
        graph_version = int(v3_publication["version"])
        graph_checksum = str(v3_publication["checksum"])
    business_model = conversation_runtime._business_model(graph)
    flow_models = _FLOW_BUSINESS_MODELS.get(flow_id)
    if flow_models and business_model not in flow_models:
        raise ValueError(
            f"Fluxo '{flow_id}' não é válido para uma persona "
            f"'{business_model}' (requer {sorted(flow_models)})"
        )
    resolved_initial_state = _resolve_initial_state(initial_state, flow_id)
    agent_node = next(
        (
            node
            for node in graph.nodes
            if (node.data or {}).get("metadata", {}).get("agent_slug")
        ),
        None,
    )
    agent_metadata = (agent_node.data or {}).get("metadata", {}) if agent_node else {}
    agent_slug = str(agent_metadata.get("agent_slug") or "agent")
    target_phone = str(agent_metadata.get("whatsapp_phone") or "").strip()
    primary_product = next(
        (
            node
            for node in graph.nodes
            if node.node_type == "product"
            and (node.data or {}).get("metadata", {}).get("qa_primary") is True
        ),
        None,
    )
    semantic_appointment_flow = flow_id in {
        "sdr_qualificacao_carro", "sdr_troca_servico", "sdr_multiplos_servicos",
        "sdr_reativacao_pos_handoff",
    }
    semantic_sales_flow = flow_id in {
        "sdr_sales_retail", "sdr_sales_reseller", "sdr_sales_branch_switch",
        "sdr_sales_knowledge_gap",
    }
    if business_model == "appointment" and semantic_appointment_flow and not v3_publication:
        raise ValueError(
            "Fluxos de agendamento exigem publicação Graph v3 ativa; "
            "o roteiro legacy com conteúdo fixo foi removido."
        )
    if business_model == "appointment" and v3_publication and semantic_appointment_flow:
        script_data = _semantic_appointment_script(
            publication=v3_publication,
            flow_id=flow_id,
            initial_state=resolved_initial_state,
        )
    elif business_model == "sales" and semantic_sales_flow and not v3_publication:
        raise ValueError(
            "Fluxos sales semânticos exigem publicação Graph v3 ativa; "
            "o roteiro fixo não é evidência conversacional."
        )
    elif business_model == "sales" and v3_publication and semantic_sales_flow:
        script_data = _semantic_sales_script(
            publication=v3_publication,
            flow_id=flow_id,
        )
    else:
        script_data = _deterministic_script(
            flow_id,
            product=primary_product,
            graph_version=graph_version,
            graph_checksum=graph_checksum,
            omit_client_name=(resolved_initial_state == "known_name"),
        )
    routing = supabase_client.get_persona_routing(persona_slug) or {}
    conversation_mode = _resolve_conversation_mode(persona_id, routing)

    session_id = str(uuid.uuid4())
    script = {
        "meta": {
            "persona": persona_slug,
            "persona_name": persona_name,
            "flow": flow_id,
            "session_id": session_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": "none",
            "classifier": (
                "semantic_graph_v1"
                if (script_data.get("driver") or {}).get("mode") == "semantic_graph_v1"
                else "deterministic_v1"
            ),
            "conversation_mode": conversation_mode,
            "pipeline_contract": (
                "conversation_v3"
                if (script_data.get("driver") or {}).get("mode") == "semantic_graph_v1"
                else "conversation_v1"
            ),
            "agent_slug": agent_slug,
            "graph_version": graph_version,
            "graph_checksum": graph_checksum,
            "initial_state": resolved_initial_state,
        },
        "target": target_contact,
        "target_phone": target_phone or None,
        "flow_description": script_data.get("flow_description", ""),
        "expected_knowledge": script_data.get("expected_knowledge", []),
        "expected_dialogue": script_data.get("expected_dialogue", {}),
        "steps": script_data.get("steps", []),
        "driver": script_data.get("driver"),
    }

    _session_create(session_id, {
        "id": session_id,
        "persona_slug": persona_slug,
        "flow_id": flow_id,
        "status": "ready",
        "script": script,
        "output": None,
        "insights": None,
        "pid": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, persona_slug=persona_slug, flow_id=flow_id)

    supabase_client.insert_event({
        "event_type": "wa_validator_script_generated",
        "payload": {
            "session_id": session_id,
            "persona_slug": persona_slug,
            "flow_id": flow_id,
            "n_steps": len(script["steps"]),
        },
    })

    return {"session_id": session_id, "script": script}


def run_session(session_id: str) -> dict:
    claimed = supabase_client.claim_wa_validator_session(session_id)
    if not claimed.get("claimed"):
        state = str(claimed.get("state") or "unknown")
        raise ValueError(f"Sessão não pode ser reexecutada no estado {state}")
    session = claimed.get("session") or _session_get(session_id) or {}
    if _WA_RUNNER_URL:
        token = (os.environ.get("AI_BRAIN_WEBHOOK_TOKEN") or "").strip()
        response = httpx.post(
            f"{_WA_RUNNER_URL}/run",
            json={"session_id": session_id, "script": session["script"]},
            headers={"X-Webhook-Token": token},
            timeout=15,
        )
        response.raise_for_status()
        _session_update(session_id, status="starting", runner=_WA_RUNNER_URL)
        return get_session(session_id)

    runtime_dir = _WA_RUNTIME / session_id
    runtime_dir.mkdir(parents=True, exist_ok=True)
    script_path = runtime_dir / "script.json"
    output_path = runtime_dir / "output.json"
    artifact_dir = _WA_ARTIFACTS / session_id
    script_path.write_text(
        json.dumps(session["script"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    def _run():
        import logging as _logging
        _log = _logging.getLogger("wa_validator_service")
        try:
            cmd = [
                _WA_NODE,
                str(_WA_EXECUTOR),
                "--script",
                str(script_path),
                "--output",
                str(output_path),
                "--profile",
                str(_WA_PROFILE),
                "--artifacts",
                str(artifact_dir),
            ]
            _log.info("Iniciando subprocess: %s", " ".join(cmd))
            proc = subprocess.Popen(
                cmd,
                cwd=_ROOT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            _session_update(session_id, pid=proc.pid, status="running")

            stdout_data, _ = proc.communicate()

            if stdout_data:
                _log.info("wa_validator stdout [%s]:\n%s", session_id[:8], stdout_data[-3000:])

            output = {}
            if output_path.exists():
                output = json.loads(output_path.read_text(encoding="utf-8"))

            if not output and proc.returncode != 0:
                final_status = "error"
                error_msg = f"Processo encerrou com código {proc.returncode}.\nLog:\n{stdout_data[-2000:] if stdout_data else '(sem saída)'}"
            else:
                final_status = output.get("status", "done")
                error_msg = output.get("error", "")

            _session_update(
                session_id, output=output, status=final_status, error=error_msg,
                log=stdout_data[-4000:] if stdout_data else "",
                output_path=str(output_path), artifact_dir=str(artifact_dir),
            )

            supabase_client.insert_event({
                "event_type": "wa_validator_session_done",
                "payload": {
                    "session_id": session_id,
                    "status": final_status,
                    "returncode": proc.returncode,
                    "n_turns": len(output.get("conversation", [])),
                },
            })

        except Exception as e:
            _session_update(session_id, status="error", error=str(e))

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    _session_update(session_id, status="starting")

    return get_session(session_id)


def get_session(session_id: str) -> dict:
    session = _session_get(session_id)
    if not session:
        raise ValueError(f"Sessão não encontrada: {session_id}")
    if _WA_RUNNER_URL and session.get("runner"):
        try:
            token = (os.environ.get("AI_BRAIN_WEBHOOK_TOKEN") or "").strip()
            response = httpx.get(
                f"{_WA_RUNNER_URL}/sessions/{session_id}",
                headers={"X-Webhook-Token": token},
                timeout=5,
            )
            response.raise_for_status()
            output = response.json()
            session = _session_update(session_id, output=output, status=output.get("status", "running"))
        except Exception as exc:
            session = _session_update(session_id, runner_error=str(exc))

    output_path = _WA_RUNTIME / session_id / "output.json"
    if session["status"] in ("running", "starting") and output_path.exists():
        try:
            partial = json.loads(output_path.read_text(encoding="utf-8"))
            session = _session_update(session_id, output=partial)
        except Exception:
            pass

    return session


def store_validation_media(
    session_id: str,
    *,
    filename: str,
    content_type: str,
    content: bytes,
    idempotency_key: str,
) -> dict:
    """Persist an inbound media fixture for a direct-validator conversation.

    This is deliberately not a provider webhook and does not enqueue a
    ``lead_buffer`` row.  It gives the browser E2E a safe way to prove private
    storage and CRM rendering without touching a real WhatsApp account or
    triggering an agent decision/outbound.
    """
    session = get_session(session_id)
    if str(session.get("status") or "") != "done":
        raise ValueError("A sessao direta precisa terminar antes do upload de midia.")
    lead_ref = session.get("lead_ref")
    if not lead_ref:
        raise ValueError("Execute a sessao direta antes de anexar uma midia de teste.")

    persona_slug = str(session.get("persona_slug") or "")
    persona = supabase_client.get_persona(persona_slug) or {}
    persona_id = str(persona.get("id") or "")
    lead = supabase_client.get_lead_by_ref(int(lead_ref)) or {}
    validation = (lead.get("metadata") or {}).get("validation") or {}
    if not persona_id or str(validation.get("session_id") or "") != session_id:
        raise ValueError("A sessao nao esta vinculada a um lead sintetico valido.")

    safe_name = Path(str(filename or "arquivo").replace("\\", "/")).name[:180]
    mime = str(content_type or "application/octet-stream").split(";", 1)[0].lower()
    if not content or len(content) > 20 * 1024 * 1024:
        raise ValueError("O arquivo deve ter entre 1 byte e 20 MB.")
    if mime == "image/png" and not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("O conteudo nao corresponde a um PNG valido.")
    if mime == "image/jpeg" and not content.startswith(b"\xff\xd8\xff"):
        raise ValueError("O conteudo nao corresponde a um JPEG valido.")
    if mime == "image/webp" and not (content.startswith(b"RIFF") and content[8:12] == b"WEBP"):
        raise ValueError("O conteudo nao corresponde a um WebP valido.")
    if mime == "application/pdf" and not content.startswith(b"%PDF-"):
        raise ValueError("O conteudo nao corresponde a um PDF valido.")
    if mime not in {"image/png", "image/jpeg", "image/webp", "application/pdf"}:
        raise ValueError("Formato de teste nao suportado. Use PNG, JPEG, WebP ou PDF.")
    if not re.fullmatch(r"[A-Za-z0-9._:-]{8,160}", idempotency_key or ""):
        raise ValueError("Chave de idempotencia invalida.")

    checksum = hashlib.sha256(content).hexdigest()
    client = supabase_client.get_client()
    existing_rows = (
        client.table("assets")
        .select("*")
        .eq("persona_id", persona_id)
        .eq("lead_id", int(lead_ref))
        .eq("upload_context", "whatsapp_inbound")
        .contains("metadata", {"validator_media": {"idempotency_key": idempotency_key}})
        .limit(1)
        .execute()
    ).data or []
    asset = existing_rows[0] if existing_rows else None
    if asset:
        recorded = ((asset.get("metadata") or {}).get("validator_media") or {}).get("sha256")
        if recorded and recorded != checksum:
            raise ValueError("A chave de idempotencia ja foi usada por outro arquivo.")

    kind = "image" if mime.startswith("image/") else "document"
    descriptor = {
        "kind": kind,
        "mime": mime,
        "filename": safe_name,
        "size": len(content),
        "reading_status": "completed",
        "validator_direct": True,
    }
    attribution = media_ingest.resolve_campaign_attribution(persona_id, int(lead_ref))
    if not asset:
        asset = supabase_client.insert_asset({
            "persona_id": persona_id,
            "lead_id": int(lead_ref),
            "campaign_id": attribution.get("campaign_id"),
            "campaign_recipient_id": attribution.get("campaign_recipient_id"),
            "type": "image" if kind == "image" else "pdf",
            "name": safe_name,
            "source": "whatsapp",
            "upload_context": "whatsapp_inbound",
            "status": "reading",
            "mime_type": mime,
            "file_size": len(content),
            "original_filename": safe_name,
            "metadata": {
                "media": descriptor,
                "direction": "inbound",
                "reading_status": "completed",
                "validation_status": "not_applicable",
                "upload_context": "whatsapp_inbound",
                "rag_eligible": False,
                "validator_media": {
                    "session_id": session_id,
                    "idempotency_key": idempotency_key,
                    "sha256": checksum,
                },
            },
        })

    asset_id = str(asset.get("id") or "")
    if not asset_id:
        raise RuntimeError("Nao foi possivel registrar o asset de validacao.")
    extension = ".pdf" if mime == "application/pdf" else {
        "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp",
    }[mime]
    storage_path = f"{persona_id}/{lead_ref}/{asset_id}-validator{extension}"
    supabase_client.upload_to_storage(
        supabase_client.WHATSAPP_MEDIA_BUCKET, storage_path, content, mime,
    )
    asset = supabase_client.update_asset(asset_id, {
        "status": "ready",
        "storage_bucket": supabase_client.WHATSAPP_MEDIA_BUCKET,
        "storage_path": storage_path,
        "file_size": len(content),
    }) or asset

    external_id = f"validator-media:{session_id}:{idempotency_key}"
    text = f"[imagem de teste recebida: {safe_name}]" if kind == "image" else f"[documento: {safe_name}]"
    supabase_client.insert_message({
        "lead_id": int(lead_ref),
        "role": "user",
        "content": text,
        "direction": "inbound",
        "status": "delivered",
        "channel": "whatsapp",
        "sender_id": external_id,
        "external_message_id": external_id,
        "channel_binding_id": session.get("channel_binding_id"),
        "correlation_id": external_id,
        "metadata": {
            "asset_id": asset_id,
            "media": descriptor,
            "validation": {"is_validation": True, "session_id": session_id},
        },
    })
    message_rows = (
        client.table("messages")
        .select("id,external_message_id,direction,created_at")
        .eq("lead_id", int(lead_ref))
        .eq("external_message_id", external_id)
        .limit(1)
        .execute()
    ).data or []
    message = message_rows[0] if message_rows else {}
    if message.get("id") and not asset.get("message_id"):
        asset = supabase_client.update_asset(asset_id, {"message_id": message["id"]}) or asset

    graph_attachment: dict[str, Any]
    try:
        from services import conversation_graph
        graph_attachment = conversation_graph.attach_inbound_asset(asset_id)
    except Exception as exc:
        logger.warning("validator media graph attach skipped asset=%s: %s", asset_id, exc)
        graph_attachment = {
            "attached": False,
            "status": "error",
            "reason": type(exc).__name__,
        }

    return {
        "session_id": session_id,
        "lead_ref": int(lead_ref),
        "asset": {
            "id": asset_id,
            "filename": safe_name,
            "mime_type": mime,
            "file_size": len(content),
            "sha256": checksum,
            "status": "ready",
            "media_url": f"/assets/{asset_id}/media",
        },
        "message": message,
        "graph_attachment": graph_attachment,
        "idempotent": bool(existing_rows),
        "outbound_enqueued": False,
    }


def list_sessions(
    *, persona_slug: str | None = None, since_hours: int | None = None, limit: int = 100,
) -> list:
    return sorted(
        _session_list(
            persona_slug=persona_slug, since_hours=since_hours, limit=limit,
        ),
        key=lambda x: x["created_at"], reverse=True,
    )


def analyze_gaps(session_id: str, model: str = _MODEL_DEFAULT) -> dict:
    session = get_session(session_id)
    output = session.get("output") or {}
    conversation = output.get("conversation", [])
    script = session.get("script", {})
    expected = script.get("expected_knowledge", [])
    persona_slug = session.get("persona_slug", "")

    _EMPTY_INSIGHTS = {
        "demonstrated": [], "gaps": [], "recommendations": [],
        "overall_score": 0, "summary": "",
    }

    if not conversation:
        return {**_EMPTY_INSIGHTS, "summary": "Sem conversa para analisar.", "session_id": session_id}

    bot_turns = [turn for turn in conversation if turn.get("role") == "bot"]
    failures = [
        turn
        for turn in bot_turns
        if turn.get("timeout")
        or turn.get("error")
        or str(turn.get("text") or "").startswith("(erro:")
    ]
    semantic_mode = (script.get("driver") or {}).get("mode") == "semantic_graph_v1"
    if semantic_mode:
        semantic_turns = [
            (index, turn.get("semantic_audit"))
            for index, turn in enumerate(bot_turns)
        ]
        semantic_gaps: list[dict] = []
        passed_criteria = 0
        total_criteria = 0
        for index, audit in semantic_turns:
            if not isinstance(audit, dict):
                total_criteria += 1
                semantic_gaps.append({
                    "topic": "missing_semantic_turn_audit",
                    "evidence": f"Turno {index + 1} não possui auditoria semântica.",
                    "priority": "high",
                })
                continue
            criteria = audit.get("criteria") or {}
            total_criteria += len(criteria)
            passed_criteria += sum(value is True for value in criteria.values())
            for failure in audit.get("failures") or []:
                semantic_gaps.append({
                    "topic": str(failure),
                    "evidence": f"Critério conversacional falhou no turno {index + 1}.",
                    "priority": "high",
                })
        expected_lineages = {
            item.split(":", 1)[1]
            for item in expected
            if str(item).startswith("graph:")
        }
        actual_lineages = {
            f"{turn.get('graph_version')}:{turn.get('graph_checksum')}"
            for turn in bot_turns
            if turn.get("graph_version") and not turn.get("error")
        }
        lineage_pass = not expected_lineages or actual_lineages == expected_lineages
        total_criteria += 1
        passed_criteria += int(lineage_pass)
        if not lineage_pass:
            semantic_gaps.append({
                "topic": "graph_lineage_mismatch",
                "evidence": (
                    f"Runtime reportou {sorted(actual_lineages)}; "
                    f"esperado {sorted(expected_lineages)}."
                ),
                "priority": "high",
            })
        if failures:
            semantic_gaps.append({
                "topic": "transport_or_reply",
                "evidence": f"{len(failures)} turno(s) sem resposta válida.",
                "priority": "high",
            })
        output_quality_confirmed = output.get("quality_pass") is True
        if not output_quality_confirmed:
            semantic_gaps.append({
                "topic": "semantic_run_not_completed",
                "evidence": "O executor não concluiu todos os turnos com quality_pass=true.",
                "priority": "high",
            })
        technical_pass = bool(
            not failures and output.get("technical_pass") is True
        )
        quality_pass = bool(
            bot_turns
            and technical_pass
            and not semantic_gaps
            and output_quality_confirmed
        )
        conversational_score = round(100 * passed_criteria / max(1, total_criteria))
        # A partial run can demonstrate useful individual criteria, but it is
        # not acceptance evidence.  Keep the partial diagnostic score separate
        # and make the prominent overall score a hard completion gate so the
        # UI can never display 100 for a stopped/error session.
        acceptance_score = conversational_score if quality_pass else 0
        insights = {
            **_EMPTY_INSIGHTS,
            "demonstrated": [
                "technical_turn_invariants",
                *(["graph_lineage"] if lineage_pass else []),
                *(["dynamic_dialogue_quality"] if quality_pass else []),
            ],
            "gaps": semantic_gaps,
            "recommendations": (
                ["Interromper no primeiro critério semântico reprovado e corrigir a origem no grafo/runtime."]
                if semantic_gaps else []
            ),
            "overall_score": acceptance_score,
            "conversational_quality_score": conversational_score,
            "technical_pass": technical_pass,
            "quality_pass": quality_pass,
            "quality_scope": "semantic_graph_v1",
            "summary": (
                "Qualidade conversacional aprovada turno a turno."
                if quality_pass
                else f"Qualidade conversacional reprovada com {len(semantic_gaps)} gap(s)."
            ),
            "session_id": session_id,
            "persona_slug": persona_slug,
            "analyzer": "semantic_graph_v1",
        }
        insights["sofia_review"] = validator_sofia_insights.build_sofia_review(
            persona_slug=persona_slug,
            session_id=session_id,
            gaps=semantic_gaps,
        )
        _session_update(session_id, insights=insights)
        supabase_client.insert_event({
            "event_type": "wa_validator_gaps_analyzed",
            "payload": {
                "session_id": session_id,
                "persona_slug": persona_slug,
                "n_gaps": len(semantic_gaps),
                "score": acceptance_score,
                "quality_pass": quality_pass,
            },
        })
        return insights
    evidence_used = {
        str(node_id)
        for turn in bot_turns
        for node_id in (turn.get("evidence_node_ids") or [])
    }
    demonstrated: list[str] = []
    gaps: list[dict] = []
    for item in expected:
        if item.startswith("evidence:"):
            node_id = item.split(":", 1)[1]
            if node_id in evidence_used:
                demonstrated.append(item)
            else:
                gaps.append({
                    "topic": item,
                    "evidence": "Node não apareceu na evidência registrada.",
                    "priority": "high",
                })
        elif item.startswith("graph:"):
            expected_lineage = item.split(":", 1)[1]
            successful_turns = [turn for turn in bot_turns if turn not in failures]
            actual_lineages = {
                f"{turn.get('graph_version')}:{turn.get('graph_checksum')}"
                for turn in successful_turns
                if turn.get("graph_version")
            }
            if actual_lineages == {expected_lineage}:
                demonstrated.append(item)
            else:
                gaps.append({
                    "topic": item,
                    "evidence": (
                        "Nenhuma resposta bem-sucedida confirmou a versão do grafo."
                        if not actual_lineages
                        else f"Runtime reportou {sorted(actual_lineages)}, esperado {expected_lineage}."
                    ),
                    "priority": "high",
                })
    if failures:
        gaps.append({
            "topic": "transport_or_reply",
            "evidence": f"{len(failures)} turno(s) sem resposta válida.",
            "priority": "high",
        })
    response_ratio = (len(bot_turns) - len(failures)) / max(
        1,
        len([turn for turn in conversation if turn.get("role") == "validator"]),
    )
    evidence_ratio = len(demonstrated) / max(1, len(expected))

    technical_score = round(max(0, min(100, (response_ratio * 50) + (evidence_ratio * 50))))
    gaps.append({
        "topic": "conversational_quality_not_evaluated",
        "evidence": (
            "Este roteiro não foi dirigido pela pergunta real do agente; "
            "o resultado prova apenas transporte, lineage e contagens técnicas."
        ),
        "priority": "high",
    })
    technical_pass = bool(
        not failures and output.get("technical_pass") is not False
    )
    insights = {
        **_EMPTY_INSIGHTS,
        "demonstrated": demonstrated,
        "gaps": gaps,
        "recommendations": [
            "Corrigir somente a origem Markdown/Graph indicada pela evidência."
        ] if gaps else [],
        "overall_score": 0,
        "technical_score": technical_score,
        "conversational_quality_score": None,
        "technical_pass": technical_pass,
        "quality_pass": False,
        "quality_scope": "technical_only",
        "summary": (
            "Validação técnica concluída; qualidade conversacional não avaliada. "
            f"{len(failures)} falha(s) técnica(s) e {len(gaps)} gap(s)."
        ),
        "session_id": session_id,
        "persona_slug": persona_slug,
        "analyzer": "deterministic_v1",
    }
    insights["sofia_review"] = validator_sofia_insights.build_sofia_review(
        persona_slug=persona_slug,
        session_id=session_id,
        gaps=gaps,
    )

    _session_update(session_id, insights=insights)

    supabase_client.insert_event({
        "event_type": "wa_validator_gaps_analyzed",
        "payload": {
            "session_id": session_id,
            "persona_slug": persona_slug,
            "n_gaps": len(insights.get("gaps", [])),
            "score": insights.get("overall_score"),
            "technical_score": technical_score,
            "quality_pass": False,
        },
    })

    return insights


async def _wait_for_reply_delivered(
    lead_ref: int,
    *,
    outbound_message_id: str,
    expected_reply: str,
    max_wait_s: float,
    poll_interval_s: float = 1.0,
) -> dict:
    """Return the exact persisted outbound produced by the audited inbound.

    Confirmed live 2026-08-08: scripted steps advanced on a fixed sleep
    (capped at 3s) regardless of how long the real pipeline took, so on a
    slower turn (branch retrieval + agentic model + possible repair round)
    the next script message could go out before the previous one's reply
    had landed. Several inbound messages for the same lead in flight at
    once then raced graph_agent_runtime_v3's optimistic ledger lock
    (commit_graph_turn_v3's expected_revision check) -- every turn that
    lost that race silently produced no reply at all, while the customer
    message itself was still persisted, leaving several client turns in a
    row with no bot reply between them. Polling for an actual new message
    to land (up to max_wait_s) makes the validator behave like a real
    customer -- one message, wait for the reply, then the next -- instead
    of outrunning the pipeline. Generic: applies to every persona/flow the
    validator runs, regardless of persona.
    """
    if not outbound_message_id:
        raise RuntimeError("Turno não retornou a identidade canônica do outbound")
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        messages = supabase_client.get_messages(str(lead_ref), limit=200) or []
        for message in messages:
            identity = str(
                message.get("message_id")
                or message.get("external_message_id")
                or message.get("correlation_id")
                or ""
            )
            if identity != outbound_message_id:
                continue
            if str(message.get("direction") or "").lower() != "outbound":
                continue
            persisted = str(message.get("content") or message.get("text") or "").strip()
            if expected_reply.strip() and persisted != expected_reply.strip():
                raise RuntimeError(
                    "Outbound persistido não corresponde à resposta auditada do turno"
                )
            return message
        await asyncio.sleep(poll_interval_s)
    raise TimeoutError(
        f"Outbound {outbound_message_id} não foi persistido no destino dentro de {max_wait_s:.0f}s"
    )


async def _wait_for_turn_audit_v3(
    inbound_buffer_id: str,
    *,
    max_wait_s: float,
    poll_interval_s: float = 1.0,
) -> dict:
    """Wait until a quiet-burst turn has its canonical proof and commit.

    The n8n webhook may acknowledge an inbound while the four-second quiet
    burst window is still open. Treating that acknowledgement as the final
    result made the validator audit the row before the worker committed its
    decision/proof, producing a false ``decision_count=0`` failure even
    though the canonical commit arrived seconds later.
    """
    deadline = time.monotonic() + max_wait_s
    audit: dict = {}
    while time.monotonic() < deadline:
        audit = supabase_client.audit_conversation_turn_v3(inbound_buffer_id)
        if (
            int(audit.get("inbound_count") or 0) == 1
            and int(audit.get("decision_count") or 0) == 1
            and int(audit.get("proof_count") or 0) == 1
            and audit.get("commit_state") == "completed"
        ):
            return audit
        await asyncio.sleep(poll_interval_s)
    return audit


def _seed_known_name(*, persona: dict, lead_ref: int, client_name: str) -> None:
    """Pre-seed nome_cliente as already-known before any script step runs.

    Tests the exact bug class fixed 2026-08-09 (a resolved field being
    re-asked/lost): a script generated with initial_state="known_name"
    never sends the customer's name at all, so if the agent asks for it
    anyway, that is a real regression, not scripted repetition. Reuses
    commit_graph_turn_v3 -- the same RPC every real turn commits through --
    instead of writing to conversation_ledgers/conversation_facts directly,
    so the seeded lead's state is indistinguishable from one that reached
    the same fact organically. Only meaningful for graph_agent_runtime_v3
    personas; quietly no-ops for any other runtime, where
    "known_name" behaves the same as "cold" for now.
    """
    if not client_name:
        return
    persona_id = str(persona.get("id") or "")
    if not persona_id:
        return
    publication = supabase_client.get_active_graph_publication(persona_id)
    if not publication:
        return
    document = publication.get("document_json") or {}
    owner_node_id = next(
        (
            field.get("owner_node_id")
            for contract in (document.get("branch_contracts") or {}).values()
            for field in contract.get("fields") or []
            if field.get("key") == "nome_cliente" and field.get("owner_node_id")
        ),
        None,
    )
    if not owner_node_id:
        return
    ledger = supabase_client.get_conversation_ledger(persona_id, lead_ref) or {}
    existing = (ledger.get("facts") or {}).get("nome_cliente") or {}
    if existing.get("status") == "known" and str(existing.get("value") or "").strip():
        return
    supabase_client.commit_graph_turn_v3(
        p_canonical_inbound_id=f"validator-seed:{lead_ref}",
        p_persona_id=persona_id, p_lead_ref=lead_ref,
        p_publication_id=publication["id"], p_graph_checksum=publication["checksum"],
        p_active_branch_node_id=None, p_asked_question_node_ids=[],
        p_expected_revision=int(ledger.get("revision") or 0),
        p_facts=[{
            "field_key": "nome_cliente", "owner_node_id": owner_node_id,
            "status": "known", "value": client_name,
            "source_message_id": f"validator-seed:{lead_ref}",
            "evidence_span": "", "confidence": 1.0,
        }],
        p_retrieval_trace={}, p_model_proposal={}, p_proof_result={}, p_repair_result={},
        p_final_decision={},
    )


def _semantic_fold(value: object) -> str:
    folded = unicodedata.normalize("NFKD", str(value or "").casefold())
    ascii_text = "".join(char for char in folded if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text))


def _semantic_similarity(left: object, right: object) -> float:
    return difflib.SequenceMatcher(
        None, _semantic_fold(left), _semantic_fold(right), autojunk=False,
    ).ratio()


def _fact_matches_expected(fact: dict | None, expected: object) -> bool:
    if not fact or fact.get("status") != "known":
        return False
    actual = fact.get("value", fact.get("value_json"))
    if isinstance(expected, bool):
        if isinstance(actual, bool):
            return actual is expected
        folded = _semantic_fold(actual)
        normalized = (
            True if folded in {"1", "sim", "true", "yes"}
            else False if folded in {"0", "nao", "false", "no"}
            else None
        )
        return normalized is expected
    return _semantic_fold(actual) == _semantic_fold(expected)


def _accepted_fact_matches_intent(
    fact: dict | None, expected: object, customer_text: str,
) -> bool:
    """Accept graph-proved string normalization without weakening lineage.

    The proof checker already enforces declared field/owner, literal evidence
    span, status and value schema.  Validator examples describe customer
    intent, not a required normalized storage representation, so a model may
    validly store ``bancos manchados...`` for the literal ``Os bancos estão
    manchados...``. Canonical boolean strings are equivalent to JSON booleans;
    numbers remain exact.
    """
    if _fact_matches_expected(fact, expected):
        return True
    if not fact or fact.get("status") != "known" or not isinstance(expected, str):
        return False
    evidence_span = str(fact.get("evidence_span") or "").strip()
    return bool(
        evidence_span
        and _semantic_fold(evidence_span) in _semantic_fold(customer_text)
    )


def _active_validator_contract(
    graph_document: dict, active_branch_node_ids: list[str],
) -> dict:
    """Project the published field/question contract for every active branch."""
    fields: list[dict] = []
    seen_fields: set[tuple[str, str]] = set()
    questions: dict[str, dict] = {}
    for anchor in dict.fromkeys(str(value) for value in active_branch_node_ids if value):
        branch = (graph_document.get("branch_contracts") or {}).get(anchor) or {}
        questions.update(branch.get("questions") or {})
        for field in branch.get("fields") or []:
            identity = (
                str(field.get("key") or ""),
                str(field.get("owner_node_id") or ""),
            )
            if identity in seen_fields:
                continue
            seen_fields.add(identity)
            fields.append(field)
    persona = next(
        (
            node for node in (graph_document.get("node_by_id") or {}).values()
            if str(node.get("node_type") or "") == "persona"
        ),
        {},
    )
    persona_data = persona.get("data") or {}
    first_contract = next(
        (
            (graph_document.get("branch_contracts") or {}).get(anchor) or {}
            for anchor in active_branch_node_ids
            if (graph_document.get("branch_contracts") or {}).get(anchor)
        ),
        {},
    )
    return {
        "fields": fields,
        "questions": questions,
        "conversation_policy": (
            first_contract.get("conversation_policy")
            or persona_data.get("conversation_policy")
            or {}
        ),
        "field_labels": (
            first_contract.get("field_labels")
            or ((persona_data.get("appointment_policy") or {}).get("field_labels") or {})
        ),
    }


def _semantic_turn_audit(
    *,
    customer_step: dict,
    turn: dict,
    proof_record: dict,
    ledger_before: dict,
    ledger_after: dict,
    contract: dict,
    recent_replies: list[str],
    previous_question_node_id: str | None,
    expected_handoff: bool = False,
) -> dict:
    proof = proof_record.get("proof_result") or {}
    decision = proof_record.get("final_decision") or {}
    reply = str(turn.get("text") or "").strip()
    intended = customer_step.get("intended_facts") or {}
    accepted_by_key: dict[str, list[dict]] = {}
    for fact in proof.get("accepted_facts") or []:
        accepted_by_key.setdefault(str(fact.get("field_key") or ""), []).append(fact)
    # A service switch legitimately commits two facts for the same selector:
    # the new branch as ``known`` and the previous branch as ``declined``.
    # Keep the positive candidate as the representative fact so the old
    # branch's tombstone cannot turn a proved switch into a Validator-only
    # ``all_intended_facts_extracted`` failure.
    accepted_status_rank = {
        "known": 5,
        "needs_confirmation": 4,
        "unknown": 3,
        "invalid": 2,
        "declined": 1,
    }
    accepted = {
        key: max(
            facts,
            key=lambda fact: accepted_status_rank.get(str(fact.get("status") or ""), 0),
        )
        for key, facts in accepted_by_key.items()
        if key and facts
    }
    facts_after = ledger_after.get("facts") or {}
    facts_by_key = ledger_after.get("facts_by_key") or {
        key: [value] for key, value in facts_after.items()
    }
    missing = [str(value) for value in proof.get("missing_fields") or []]
    question_id = str(proof.get("next_question_node_id") or "") or None
    first_missing = missing[0] if missing else None
    first_field = next(
        (
            field for field in contract.get("fields") or []
            if str(field.get("key") or "") in missing
            and not any(
                fact.get("status") == "unknown"
                and str(fact.get("owner_node_id") or "")
                == str(field.get("owner_node_id") or "")
                for fact in facts_by_key.get(str(field.get("key") or "")) or []
            )
        ),
        None,
    )
    first_askable = str((first_field or {}).get("key") or "") or None
    expected_question_id = str((first_field or {}).get("question_node_id") or "") or None
    question = (contract.get("questions") or {}).get(question_id or "") or {}
    previous_question = (
        (contract.get("questions") or {}).get(previous_question_node_id or "") or {}
    )
    previous_asked_field = str(previous_question.get("field_key") or "") or None
    question_text = str(question.get("text") or "").strip()
    asked_field = str(question.get("field_key") or "") or None
    declarative_parts = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", reply)
        if sentence.strip() and "?" not in sentence
    ]
    expected_evidence = set(customer_step.get("expected_evidence_node_ids") or [])
    actual_evidence = set(turn.get("evidence_node_ids") or []) | set(
        decision.get("evidence_node_ids") or []
    )
    asked_owner = str(question.get("owner_node_id") or (first_field or {}).get("owner_node_id") or "")
    asked_fact_already_known = any(
        fact.get("status") == "known"
        and (not asked_owner or str(fact.get("owner_node_id") or "") == asked_owner)
        for fact in facts_by_key.get(asked_field or "") or []
    )
    first_question_offset = reply.find("?")
    acknowledgement_before_question = any(
        reply.find(part) < first_question_offset
        for part in declarative_parts
        if first_question_offset >= 0
    )
    qualification_complete = bool(proof.get("qualification_complete"))
    confirmation_pending = proof.get("confirmation_state") == "awaiting_confirmation"
    collection_complete = bool(proof.get("collection_complete"))
    consumed_service_values = {
        _semantic_fold(str(span.get("text") or ""))
        for span in proof.get("consumed_service_spans") or []
        if str(span.get("text") or "").strip()
    }
    branch_selection_keys = {
        str(field.get("key") or "")
        for field in contract.get("fields") or []
        if field.get("branch_selection_field") is True
    }
    pending_confirmation = (
        proof.get("pending_confirmation")
        if isinstance(proof.get("pending_confirmation"), dict)
        else {}
    )
    pending_confirmation_branch = str(
        pending_confirmation.get("branch_anchor_node_id") or ""
    )
    pending_confirmation_candidate = str(
        pending_confirmation.get("candidate") or ""
    )

    def intended_fact_is_pending_branch_confirmation(key: str, value: object) -> bool:
        """Accept a graph-proved service candidate until the customer confirms it.

        A branch selector deliberately persists only after the next explicit
        confirmation.  The previous Validator required the active branch and
        a ledger fact on the candidate-detection turn, so it stopped before it
        could send that confirmation even when the proof had already bound the
        exact catalog candidate to the expected graph branch.
        """
        fact = accepted.get(key) or {}
        expected_branch = str(customer_step.get("expected_branch_node_id") or "")
        return bool(
            key in branch_selection_keys
            and fact.get("status") == "needs_confirmation"
            and pending_confirmation.get("kind") == "service"
            and pending_confirmation_branch
            and pending_confirmation_branch == expected_branch
            and str(fact.get("owner_node_id") or "") == expected_branch
            and _semantic_fold(pending_confirmation_candidate)
            == _semantic_fold(str(value or ""))
            and _semantic_fold(str(fact.get("evidence_span") or ""))
            in _semantic_fold(str(customer_step.get("text") or ""))
            and bool(
                _semantic_fold(str(fact.get("evidence_span") or ""))
            )
        )

    accepted_all = all(
        (
            key in accepted
            and _accepted_fact_matches_intent(
                accepted.get(key), value, str(customer_step.get("text") or ""),
            )
            and any(
                _fact_matches_expected(
                    current,
                    accepted[key].get("value", accepted[key].get("value_json")),
                )
                and (
                    not accepted[key].get("owner_node_id")
                    or str(current.get("owner_node_id") or "")
                    == str(accepted[key].get("owner_node_id") or "")
                )
                for current in facts_by_key.get(key) or []
            )
        )
        or intended_fact_is_pending_branch_confirmation(key, value)
        for key, value in intended.items()
    )
    service_value_not_reused_as_field = not any(
        str(fact.get("field_key") or "") not in branch_selection_keys
        and (
            _semantic_fold(str(fact.get("value") or "")) in consumed_service_values
            or _semantic_fold(str(fact.get("evidence_span") or "")) in consumed_service_values
        )
        for fact in proof.get("accepted_facts") or []
    )
    resolved_operations = (
        (proof.get("service_resolution") or {}).get("operations") or []
    )
    applied_operations = (
        proof.get("applied_service_operations")
        if isinstance(proof.get("applied_service_operations"), list)
        else proof.get("service_operations") or []
    )

    def operation_signature(operation: dict) -> tuple[str, str, str, str, str]:
        return (
            str(operation.get("action") or ""),
            str(operation.get("branch_anchor_node_id") or ""),
            str(operation.get("branch_path_checksum") or ""),
            _semantic_fold(str(operation.get("evidence_span") or "")),
            str(operation.get("evidence_type") or ""),
        )

    resolved_signatures = [operation_signature(item) for item in resolved_operations]
    applied_signatures = [operation_signature(item) for item in applied_operations]
    deterministic_field_confirmation = (
        proof.get("mode") == "deterministic_field_confirmation"
    )
    operations_match_resolution = (
        deterministic_field_confirmation
        or applied_signatures == resolved_signatures
    )
    consumed_spans = proof.get("consumed_service_spans") or []
    operations_have_authorized_evidence = all(
        str(operation.get("evidence_type") or "")
        in {"exact_catalog", "confirmed_candidate", "explicit_change"}
        and (
            str(operation.get("evidence_type") or "") != "explicit_change"
            or str(operation.get("action") or "") == "drop"
        )
        and any(
            _semantic_fold(str(span.get("text") or ""))
            == _semantic_fold(str(operation.get("evidence_span") or ""))
            and str(span.get("evidence_type") or "")
            == str(operation.get("evidence_type") or "")
            for span in consumed_spans
        )
        for operation in applied_operations
    )
    customer_text = str(customer_step.get("text") or "")
    consumed_intervals = [
        (int(span.get("start") or 0), int(span.get("end") or 0))
        for span in consumed_spans
        if int(span.get("end") or 0) > int(span.get("start") or 0)
    ]
    field_evidence_disjoint = True
    for fact in proof.get("accepted_facts") or []:
        if str(fact.get("field_key") or "") in branch_selection_keys:
            continue
        evidence = str(fact.get("evidence_span") or "")
        starts = [match.start() for match in re.finditer(re.escape(evidence), customer_text)] if evidence else []
        if any(
            start < service_end and service_start < start + len(evidence)
            for start in starts
            for service_start, service_end in consumed_intervals
        ):
            field_evidence_disjoint = False
            break
    focused_id = str(ledger_after.get("active_branch_node_id") or "") or None
    active_ids = [
        str(value) for value in ledger_after.get("active_branch_node_ids") or [] if value
    ]
    if not active_ids and focused_id:
        active_ids = [focused_id]
    branch_focus_invariant = (
        (not active_ids and focused_id is None) or focused_id in set(active_ids)
    )
    handoff_observed = bool(
        turn.get("handoff")
        or proof.get("handoff_requested")
        or decision.get("handoff_requested")
        or str(turn.get("route") or "").upper() == "HUMAN"
    )
    terminal_intent = None
    if handoff_observed:
        terminal_intent = (
            "qualification_complete" if qualification_complete
            else "qualification_incomplete"
        )
    forbidden_claim_patterns = [
        str(pattern) for pattern in customer_step.get("forbidden_claim_patterns") or []
        if str(pattern)
    ]
    unsupported_claim_not_invented = not any(
        re.search(pattern, reply, flags=re.IGNORECASE)
        for pattern in forbidden_claim_patterns
    )
    repetition_policy = (
        (contract.get("conversation_policy") or {}).get("question_repetition") or {}
    )
    repetition = conversation_repetition.assess_repetition(
        current_reply=reply,
        recent_replies=recent_replies[-4:],
        question_node_id=question_id,
        question_text=question_text,
        asked_question_node_ids=ledger_before.get("asked_question_node_ids") or [],
        max_attempts=repetition_policy.get("max_attempts", 0),
        field_pending=bool(first_askable and question_id == expected_question_id),
        terminal_intent=terminal_intent,
        previous_terminal_intent=str(
            ((ledger_before.get("terminal_handoff") or {}).get("intent") or "")
        ) or None,
    )
    repetition_failures = set(repetition["failures"])
    criteria = {
        "intent_identified": bool(decision.get("intent")),
        "doubt_answered_first": (
            not expected_evidence
            or (
                bool(expected_evidence & actual_evidence)
                and bool(declarative_parts)
                and (first_question_offset < 0 or acknowledgement_before_question)
            )
        ),
        "unsupported_claim_not_invented": unsupported_claim_not_invented,
        "all_intended_facts_extracted": accepted_all,
        "service_value_not_reused_as_field": service_value_not_reused_as_field,
        "service_operations_match_resolution": operations_match_resolution,
        "service_operations_have_authorized_evidence": operations_have_authorized_evidence,
        "service_field_evidence_disjoint": field_evidence_disjoint,
        "branch_focus_invariant": branch_focus_invariant,
        "field_confirmation_precedes_final_confirmation": not (
            proof.get("pending_confirmation")
            and proof.get("explicit_confirmation") is True
        ),
        "received_content_acknowledged": not intended or bool(declarative_parts),
        "first_missing_field_only": (
            (not missing and question_id is None)
            or (
                proof.get("confirmation_state") == "field_confirmation"
                and proof.get("pending_confirmation")
                and question_id is None
            )
            or (
                handoff_observed
                and question_id is None
                and first_askable is None
            )
            or (
                question_id == expected_question_id
                and bool(question_text)
                and graph_proof_checker_v3._question_already_asked(question_text, reply)
            )
        ),
        "known_fact_not_reasked": not asked_fact_already_known,
        "reply_not_repeated": not bool(
            repetition_failures & {"semantic_repetition", "terminal_repetition"}
        ),
        "question_repetition_budget": (
            "question_attempt_budget_exceeded" not in repetition_failures
        ),
        "contextual_retry_valid": not bool(
            repetition_failures & {"contextual_bridge_required", "question_field_not_pending"}
        ),
        "terminal_not_repeated": "terminal_repetition" not in repetition_failures,
        "model_reconciled_without_fallback": (
            # A valid, complete proof intentionally resolves to the published
            # deterministic completion reply. That is not a model-repair
            # failure and must terminate the Validator immediately.
            (
                qualification_complete
                and not missing
                and question_id is None
                and proof.get("valid") is not False
            )
            or (
                proof.get("fallback_used") is not True
                and not (proof.get("model_proposal_errors") or [])
            )
        ),
        "expected_branch_persisted": (
            not customer_step.get("expected_branch_node_id")
            or (
                str(ledger_after.get("active_branch_node_id") or "")
                == str(customer_step.get("expected_branch_node_id"))
            )
            or (
                len(ledger_after.get("active_branch_node_ids") or []) > 1
                and str(customer_step.get("expected_branch_node_id"))
                in set(ledger_after.get("active_branch_node_ids") or [])
            )
            or (
                proof.get("confirmation_state") == "field_confirmation"
                and pending_confirmation.get("kind") == "service"
                and pending_confirmation_branch
                == str(customer_step.get("expected_branch_node_id"))
            )
        ),
        "expected_active_branches_persisted": (
            not customer_step.get("expected_active_branch_node_ids")
            or set(
                ledger_after.get("active_branch_node_ids")
                or (
                    [ledger_after.get("active_branch_node_id")]
                    if ledger_after.get("active_branch_node_id")
                    else []
                )
            )
            == set(customer_step.get("expected_active_branch_node_ids") or [])
        ),
        "question_advanced": (
            not previous_question_node_id
            or previous_question_node_id != question_id
            # A service switch/add can legitimately leave the outstanding
            # shared-field question unchanged: the customer changed the
            # branch but still has not answered that question.  Require
            # advancement only when this turn intended to answer the field
            # that was actually asked previously.
            or previous_asked_field not in intended
        ),
        "handoff_only_after_completion": (
            not handoff_observed
            or qualification_complete
            or bool(missing and first_askable is None)
            or collection_complete
        ),
        "expected_handoff_reached": (
            not expected_handoff
            or first_askable is not None
            or confirmation_pending
            or handoff_observed
        ),
    }
    failures = [name for name, passed in criteria.items() if not passed]
    return {
        "passed": not failures,
        "criteria": criteria,
        "failures": failures,
        "asked_field": asked_field,
        "next_question_node_id": question_id,
        "first_missing_field": first_missing,
        "missing_fields": missing,
        "accepted_fact_keys": sorted(accepted),
        "intended_fact_keys": sorted(str(key) for key in intended),
        "previous_ledger_revision": ledger_before.get("revision"),
        "ledger_revision": ledger_after.get("revision"),
        "qualification_complete": qualification_complete,
        "handoff_observed": handoff_observed,
        "repetition_audit": repetition,
    }


def _post_handoff_greeting_audit(
    *, customer_step: dict, turn: dict, proof_record: dict,
    ledger_before: dict, ledger_after: dict, journey_after: dict,
) -> dict:
    """Prove the original Oi/Oii regression on a reactivated synthetic lead."""
    proof = proof_record.get("proof_result") or {}
    decision = proof_record.get("final_decision") or {}
    accepted_service = [
        fact for fact in proof.get("accepted_facts") or []
        if str(fact.get("field_key") or "") == "servico"
    ]
    service_before = (ledger_before.get("facts") or {}).get("servico") or {}
    service_after = (ledger_after.get("facts") or {}).get("servico") or {}
    message = str(customer_step.get("text") or "")
    handoff_observed = bool(
        turn.get("handoff")
        or proof.get("handoff_requested")
        or decision.get("handoff_requested")
        or str(turn.get("route") or "").upper() == "HUMAN"
    )
    criteria = {
        "greeting_intent_current_turn": (
            str(decision.get("intent") or turn.get("intent") or "") == "greeting"
            and (proof.get("intent_audit") or {}).get("greeting") is True
        ),
        "greeting_not_extracted_as_service": not accepted_service,
        "service_fact_preserved": bool(
            service_before
            and service_after.get("status") == "known"
            and service_after.get("value") == service_before.get("value")
            and service_after.get("owner_node_id") == service_before.get("owner_node_id")
            and str(service_after.get("value") or "").casefold() != message.casefold()
        ),
        "service_resolution_remains_referential": (
            (proof.get("service_resolution") or {}).get("resolved") is True
        ),
        "no_service_or_field_requestion": not proof.get("next_question_node_id"),
        "support_reply_emitted": bool(str(turn.get("text") or "").strip())
        and not turn.get("timeout"),
        "no_new_handoff": not handoff_observed,
        "support_route_is_sdr": str(turn.get("route") or "").upper() == "SDR",
        "journey_remains_handed_off": str(journey_after.get("state") or "")
        == "handed_off",
    }
    failures = [name for name, passed in criteria.items() if not passed]
    return {
        "passed": not failures,
        "criteria": criteria,
        "failures": failures,
        "asked_field": None,
        "next_question_node_id": proof.get("next_question_node_id"),
        "missing_fields": proof.get("missing_fields") or [],
        "accepted_fact_keys": sorted(
            str(fact.get("field_key") or "")
            for fact in proof.get("accepted_facts") or []
        ),
        "qualification_complete": True,
        "handoff_observed": False,
        "post_handoff_support": True,
    }


def _semantic_failure_records(
    *,
    conversation: list[dict],
    turn_index: int,
    audit: dict,
    session_id: str,
    persona_slug: str,
    buffer_id: str,
    external_message_id: str,
    correlation_id: str,
    journey_state: str | None = None,
) -> tuple[str, dict, dict]:
    """Build the terminal quality-failure result and its non-secret event."""
    failure = "semantic_turn_failed:" + ",".join(audit.get("failures") or [])
    output = {
        "conversation": conversation,
        "status": "error",
        "technical_pass": True,
        "quality_pass": False,
        "failed_turn": turn_index,
        "failure": failure,
    }
    event = {
        "event_type": "wa_validator_semantic_failed",
        "payload": {
            "session_id": session_id,
            "persona_slug": persona_slug,
            "turn": turn_index,
            "failures": audit.get("failures") or [],
            "criteria": audit.get("criteria") or {},
            "repetition_audit": audit.get("repetition_audit") or {},
            "canonical_inbound_id": buffer_id,
            "canonical_inbound": {
                "buffer_id": buffer_id,
                "external_message_id": external_message_id,
                "correlation_id": correlation_id,
            },
            "journey_state": journey_state,
        },
    }
    return failure, output, event


def enqueue_session_direct(session_id: str) -> dict:
    """Queue a direct run without executing conversation work in the API."""
    queued = supabase_client.enqueue_wa_validator_session(session_id, "direct")
    if not queued.get("queued"):
        state = str(queued.get("state") or "unknown")
        raise ValueError(f"Sessão não pode ser enfileirada no estado {state}")
    return queued.get("session") or get_session(session_id)


def _next_semantic_driver_step(
    *,
    driver: dict,
    state: dict,
    asked_field: str,
    answered_fields: set[str],
    active_anchor: str,
    expected_active_branches: list[str],
    qualification_complete: bool = False,
) -> dict | None:
    """Select the next synthetic customer turn for a semantic validation.

    The standard appointment driver answers graph fields until one remains,
    sends a pure interruption, verifies one contextual resumption, then
    ignores that field again to require an incomplete terminal handoff.
    """
    if qualification_complete and not state.get("confirmation_sent"):
        confirmation = driver.get("confirmation") or {}
        text = str(confirmation.get("text") or "").strip()
        if text:
            state["confirmation_sent"] = True
            return {
                "text": text,
                "kind": "explicit_confirmation",
                "intended_facts": {},
                "expected_branch_node_id": active_anchor,
                "expected_active_branch_node_ids": list(expected_active_branches),
            }

    second_ignore = driver.get("second_ignore")
    if (
        state.get("doubt_sent")
        and not state.get("second_ignore_sent")
        and asked_field
        and asked_field == state.get("interrupted_field")
        and isinstance(second_ignore, dict)
        and str(second_ignore.get("text") or "").strip()
    ):
        state["second_ignore_sent"] = True
        return {
            **second_ignore,
            "kind": "ignored_again",
            "intended_facts": {},
            "expected_branch_node_id": active_anchor,
            "expected_active_branch_node_ids": list(expected_active_branches),
        }

    doubt = driver.get("doubt")
    interruption_threshold = int(driver.get("interruption_after_answered_fields") or 0)
    if (
        isinstance(doubt, dict)
        and not state.get("doubt_sent")
        and len(answered_fields) >= interruption_threshold
    ):
        state["doubt_sent"] = True
        state["interrupted_field"] = asked_field
        return {
            **doubt,
            "kind": "doubt",
            "intended_facts": {},
            "expected_branch_node_id": active_anchor,
        }

    switch = driver.get("switch")
    if (
        isinstance(switch, dict)
        and not state.get("switch_sent")
        and len(answered_fields) >= int(switch.get("after_answered_fields") or 0)
    ):
        state["switch_sent"] = True
        return {**switch, "kind": "branch_switch"}

    deferred = driver.get("deferred_answer")
    if isinstance(deferred, dict):
        deferred_field = str(deferred.get("field") or "")
        defer_text = str(deferred.get("defer_text") or "").strip()
        if (
            deferred_field
            and asked_field == deferred_field
            and defer_text
            and not state.get("deferred_sent")
        ):
            state["deferred_sent"] = True
            return {
                "text": defer_text,
                "kind": "field_deferred",
                "intended_facts": {},
                "expected_branch_node_id": active_anchor,
                "expected_active_branch_node_ids": list(expected_active_branches),
            }

        later_text = str(deferred.get("later_text") or "").strip()
        if (
            state.get("deferred_sent")
            and not state.get("loose_later_sent")
            and deferred_field
            and asked_field
            and asked_field != deferred_field
            and later_text
        ):
            state["loose_later_sent"] = True
            return {
                "text": later_text,
                "kind": "loose_field_answer",
                "intended_facts": {deferred_field: deferred.get("later_value")},
                "expected_branch_node_id": active_anchor,
                "expected_active_branch_node_ids": list(expected_active_branches),
            }

    answer = (driver.get("answers") or {}).get(asked_field)
    if isinstance(answer, dict) and str(answer.get("text") or "").strip():
        return {
            "text": str(answer["text"]),
            "kind": "field_answer",
            "intended_facts": {asked_field: answer.get("value")},
            "expected_branch_node_id": active_anchor,
            "expected_active_branch_node_ids": list(expected_active_branches),
        }
    return None


def cleanup_expired_artifacts(*, hours: int = 12, dry_run: bool = True) -> dict:
    return supabase_client.cleanup_wa_validator_artifacts(
        hours=hours, dry_run=dry_run,
    )


def mark_session_execution_error(session_id: str, exc: Exception) -> dict:
    return _session_update(
        session_id,
        status="error",
        error=str(exc),
        output={"status": "error", "failure": "validator_worker_error"},
    )


async def run_session_direct(
    session_id: str, *, claimed_session: dict | None = None,
) -> dict:
    """Execute through the selected mode using the conversation_v1 contract."""
    if claimed_session is None:
        claimed = supabase_client.claim_wa_validator_session(session_id)
        if not claimed.get("claimed"):
            state = str(claimed.get("state") or "unknown")
            raise ValueError(f"Sessão não pode ser reexecutada no estado {state}")
        session = claimed.get("session") or _session_get(session_id) or {}
    else:
        session = claimed_session

    script = session.get("script", {})
    persona_slug = session.get("persona_slug", "")
    persona = supabase_client.get_persona(persona_slug) or {}
    routing = supabase_client.get_persona_routing(persona_slug) or {}
    conversation_mode = _resolve_conversation_mode(persona.get("id"), routing)
    bindings = supabase_client.get_workflow_bindings(persona.get("id"))
    binding = next(
        (
            row
            for row in bindings
            if row.get("active", True)
            and (row.get("metadata") or {}).get("decision_owner")
            in {"n8n_hybrid", "n8n_agents"}
        ),
        None,
    )
    binding_metadata = (binding or {}).get("metadata") or {}
    workflow_url = str(
        binding_metadata.get("conversation_webhook_url")
        or binding_metadata.get("webhook_url")
        or os.environ.get("N8N_CONVERSATION_TEST_URL")
        or ""
    ).strip()
    if conversation_mode == "n8n_agents" and not workflow_url:
        raise ValueError("Workflow n8n_agents ativo não configurado")
    driver = script.get("driver") or {}
    semantic_mode = driver.get("mode") == "semantic_graph_v1"
    pipeline_contract = (
        str(binding_metadata.get("pipeline_contract") or "conversation_v3")
        if conversation_mode == "n8n_agents"
        else "conversation_v1"
    )
    if semantic_mode and pipeline_contract != "conversation_v3":
        raise ValueError(
            "Driver semântico exige pipeline_contract=conversation_v3 para "
            "provar commit, proof e outbox por turno"
        )
    flow_id = session.get("flow_id", "")
    graph_version = script.get("meta", {}).get("graph_version")
    # Flow name + graph version it was generated against (not just the
    # persona slug) so two validation leads for the same persona are
    # distinguishable at a glance, and a stale run against an old graph
    # version is obvious from the name alone.
    lead_label = f"{flow_id} v{graph_version}" if graph_version is not None else flow_id
    lead = supabase_client.ensure_lead_for_persona(
        lead_id=f"validator_{session_id[:8]}",
        persona_slug_or_id=persona_slug,
        nome=lead_label,
        stage="novo",
        canal="whatsapp",
    ) or {}
    lead_ref = lead.get("id")
    if not lead_ref:
        raise ValueError("Não foi possível criar o lead de validação")
    supabase_client.update_lead(
        lead_ref,
        {
            "metadata": {
                **dict(lead.get("metadata") or {}),
                "validation": {
                    "is_validation": True,
                    "source": "webscraping",
                    "run_id": session_id,
                    "session_id": session_id,
                },
            }
        },
    )
    channel_binding_id = (supabase_client.get_lead_by_ref(lead_ref) or {}).get(
        "channel_binding_id"
    )
    if not channel_binding_id:
        raise ValueError(
            f"Lead de validação para {persona_slug} não recebeu channel_binding_id "
            "automático; configure um workflow_binding ativo para a persona."
        )
    # Keep the exact synthetic conversation identity on the session. Follow-up
    # QA actions must never guess a lead from its display name.
    _session_update(
        session_id,
        lead_ref=int(lead_ref),
        channel_binding_id=str(channel_binding_id),
    )
    initial_state = script.get("meta", {}).get("initial_state", "cold")
    if initial_state == "known_name":
        _seed_known_name(
            persona=persona, lead_ref=int(lead_ref),
            client_name=str(script.get("expected_dialogue", {}).get("known_name") or ""),
        )
    publication: dict = {}
    graph_document: dict = {}
    if semantic_mode:
        publication = supabase_client.get_active_graph_publication(str(persona.get("id") or "")) or {}
        graph_document = publication.get("document_json") or {}
        if not publication or not graph_document:
            raise ValueError("Driver semântico exige publicação Graph v3 ativa")
        expected_version = script.get("meta", {}).get("graph_version")
        expected_checksum = str(script.get("meta", {}).get("graph_checksum") or "")
        if expected_version is not None and int(publication.get("version") or 0) != int(expected_version):
            raise ValueError("Publicação do grafo mudou depois da geração do roteiro")
        if expected_checksum and str(publication.get("checksum") or "") != expected_checksum:
            raise ValueError("Checksum do grafo mudou depois da geração do roteiro")
        opening = driver.get("opening")
        if not isinstance(opening, dict) or not str(opening.get("text") or "").strip():
            raise ValueError("Driver semântico não possui abertura válida")
        steps = [opening]
    else:
        steps = script.get("steps", [])

    _session_update(session_id, status="running", output={"conversation": [], "status": "running"})

    conversation: list[dict] = []

    async def _do_run() -> None:
        import logging as _logging
        _log = _logging.getLogger("wa_validator_service.direct")
        try:
            token = (os.environ.get("AI_BRAIN_WEBHOOK_TOKEN") or "").strip()
            step_queue = list(steps)
            max_turns = int(driver.get("max_turns") or len(step_queue) or 1)
            recent_replies: list[str] = []
            previous_question_node_id: str | None = None
            answered_fields: set[str] = {
                str(value) for value in driver.get("initial_known_fields") or []
            } | {
                str(value) for value in (driver.get("opening") or {}).get("intended_facts") or {}
            }
            driver_state = {
                "doubt_sent": False,
                "switch_sent": False,
                "deferred_sent": False,
                "loose_later_sent": False,
                "post_handoff_started": False,
            }
            expected_active_branches: list[str] = []
            semantic_complete = False
            i = 0
            while step_queue and i < max_turns:
                step = step_queue.pop(0)
                if "backdate_hours" in step:
                    hours = float(step["backdate_hours"])
                    shifted = supabase_client.backdate_lead_messages(int(lead_ref), hours)
                    conversation.append({
                        "role": "system",
                        "text": f"[backdated {shifted} message(s) by {hours}h]",
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "message_id": f"validator:{session_id}:{i}:backdate",
                    })
                    _session_update(session_id, output={"conversation": list(conversation), "status": "running"})
                    i += 1
                    continue
                if step.get("resume_after_handoff"):
                    if not agents_service.resume_lead(int(lead_ref)):
                        raise RuntimeError("synthetic lead could not be resumed")
                    resumed = supabase_client.get_lead_by_ref(int(lead_ref)) or {}
                    if str(resumed.get("handoff_level") or "none") != "none":
                        raise RuntimeError("synthetic lead remained paused after resume")
                    conversation.append({
                        "role": "system",
                        "text": "[synthetic lead reactivated after handoff]",
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "message_id": f"validator:{session_id}:{i}:resume",
                    })
                text = step.get("text", "")
                configured_wait = float(step.get("wait", 10) or 10)
                ts_now = datetime.now(timezone.utc).isoformat()
                message_id = f"validator:{session_id}:{i}"
                correlation_id = f"validator:{session_id}:{i}"
                ledger_before = (
                    supabase_client.get_conversation_ledger(str(persona.get("id") or ""), int(lead_ref))
                    or {"revision": 0, "facts": {}, "active_branch_node_id": None}
                )
                conversation.append({
                    "role": "validator",
                    "text": text,
                    "ts": ts_now,
                    "message_id": message_id,
                })
                # commit() claims the inbound message through the same
                # durable lead_buffer row the real inbound webhooks create
                # (claim_conversation_commit requires it to already exist),
                # so the validator must go through the same atomic enqueue
                # used in production rather than a bare message insert.
                #
                # A direct validation turn is invoked synchronously below;
                # it must therefore be inert to the transport worker.  If it
                # starts as ``buffered``, dispatch can claim it concurrently
                # and the next validator turn also sees it as an unconsumed
                # burst sibling, producing ``burst_superseded``.  Keep the
                # canonical evidence row in a non-dispatchable state until
                # the v3 audit proves the commit, then terminalize it below.
                envelope = supabase_client.enqueue_whatsapp_envelope(
                    buffer={
                        "persona_id": persona.get("id"),
                        "lead_ref": lead_ref,
                        "channel_binding_id": channel_binding_id,
                        "whatsapp_phone_number_id": None,
                        "external_message_id": message_id,
                        "direction": "inbound",
                        "payload": {"text": text, "sender": "wa-validator"},
                        "status": "waiting_human",
                        "batch_key": f"{persona.get('id')}:{lead_ref}",
                        "idempotency_key": f"inbound:wa-validator:{message_id}",
                        "correlation_id": correlation_id,
                    },
                    message={
                        "lead_id": lead_ref,
                        "role": "user",
                        "content": text,
                        "direction": "inbound",
                        "status": "buffered",
                        "channel": "whatsapp",
                        "sender_id": "wa-validator",
                        "external_message_id": message_id,
                        "channel_binding_id": channel_binding_id,
                        "correlation_id": correlation_id,
                        "metadata": {"provider": "wa-validator"},
                        "created_at": ts_now,
                    },
                )
                buffer_uuid = str(envelope.get("buffer_id") or "")
                _session_update(session_id, output={"conversation": list(conversation), "status": "running"})
                try:
                    # Confirmed live 2026-08-08: hardcoding "conversation_v1"
                    # here got n8n_agents steps rejected by the
                    # workflow's own "pipeline contract mismatch" guard --
                    # the deployed workflow expects whatever the binding
                    # itself declares, exactly like real dispatch
                    # (workers.whatsapp_dispatch_worker) already resolves it.
                    event = {
                            "persona_slug": persona_slug,
                            "lead_ref": lead_ref,
                            "buffer_id": buffer_uuid,
                            "external_message_id": message_id,
                            "correlation_id": correlation_id,
                            "phone_number_id": None,
                            "channel_binding_id": channel_binding_id,
                            "message": text,
                            "pipeline_contract": pipeline_contract,
                            "decision_owner": conversation_mode,
                        }
                    if conversation_mode == "n8n_agents":
                        # Reuse the exact same client the real dispatch
                        # worker uses (workers.whatsapp_dispatch_worker)
                        # instead of a hand-rolled httpx call -- the
                        # previous inline version built headers = {} even
                        # when a token was configured, so it silently
                        # never sent X-Webhook-Token at all.
                        status, body = await asyncio.to_thread(
                            n8n_client.send_to_webhook,
                            workflow_url,
                            event,
                            secret=token or None,
                            timeout=45.0,
                            # send_to_webhook truncates its return value to
                            # this many chars -- workers.whatsapp_dispatch_
                            # worker only wants a short log preview, but this
                            # caller parses the WHOLE body as JSON. Confirmed
                            # live 2026-08-08: reusing the worker's 65_536
                            # preview limit here truncated real
                            # replies mid-string, turning a working turn into
                            # a bogus JSONDecodeError. Large enough that no
                            # realistic conversation turn is ever cut off.
                            response_limit=5_000_000,
                        )
                        if status >= 400:
                            raise RuntimeError(
                                f"n8n webhook returned HTTP {status}: {body[:500]}"
                            )
                        if not body.strip():
                            raise RuntimeError(
                                f"n8n webhook returned HTTP {status} with an "
                                "empty body -- the workflow likely isn't "
                                "reaching its Respond to Webhook node"
                            )
                        data = json.loads(body)
                    else:
                        data = conversation_runtime.execute_pipeline(
                            persona_slug=persona_slug,
                            lead_ref=int(lead_ref),
                            message=text,
                            message_id=message_id,
                            correlation_id=correlation_id,
                            phone_number_id=None,
                            channel_binding_id=channel_binding_id,
                            inbound_buffer_id=event["buffer_id"],
                        )
                    turn_audit: dict | None = None
                    if pipeline_contract == "conversation_v3":
                        turn_audit = await _wait_for_turn_audit_v3(
                            buffer_uuid,
                            max_wait_s=max(configured_wait, 45.0),
                        )
                        # When n8n acknowledged during the quiet-burst
                        # window, use the canonical committed result rather
                        # than the early acknowledgement body for dialogue
                        # and proof validation.
                        committed_buffer = (
                            supabase_client.get_whatsapp_buffer_by_idempotency(
                                f"inbound:wa-validator:{message_id}"
                            )
                            or {}
                        )
                        committed_result = (
                            ((committed_buffer.get("payload") or {}).get("conversation_commit") or {})
                            .get("result")
                        )
                        if isinstance(committed_result, dict) and committed_result:
                            data = committed_result

                    reply: str = data.get("reply_text") or ""
                    turn: dict = {
                        "role": "bot",
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "agent": script.get("meta", {}).get("agent_slug"),
                        "route": data.get("route"),
                        "intent": data.get("intent"),
                        "handoff": data.get("handoff"),
                        "message_id": data.get("message_id"),
                        "classifier": data.get("classifier"),
                        "conversation_mode": conversation_mode,
                        "pipeline_contract": data.get("pipeline_contract")
                        or pipeline_contract,
                        "evidence_node_ids": data.get("evidence_node_ids")
                        or [],
                        "graph_version": data.get("graph_version")
                        or script.get("meta", {}).get("graph_version"),
                        "graph_checksum": data.get("graph_checksum")
                        or script.get("meta", {}).get("graph_checksum"),
                    }
                    if reply:
                        turn["text"] = reply
                    else:
                        turn["text"] = "(sem resposta — agente não gerou reply)"
                        turn["timeout"] = True
                    if pipeline_contract == "conversation_v3":
                        audit = turn_audit or supabase_client.audit_conversation_turn_v3(buffer_uuid)
                        invariant_errors: list[str] = []
                        for key, expected in (
                            ("inbound_count", 1), ("decision_count", 1),
                            ("proof_count", 1), ("valid_proof_count", 1),
                        ):
                            if int(audit.get(key) or 0) != expected:
                                invariant_errors.append(f"{key}={audit.get(key)}")
                        if int(audit.get("outbound_count") or 0) > 1:
                            invariant_errors.append(f"outbound_count={audit.get('outbound_count')}")
                        if audit.get("outbound_released_after_proof") is not True:
                            invariant_errors.append("outbound_released_before_proof")
                        if audit.get("commit_state") != "completed":
                            invariant_errors.append(f"commit_state={audit.get('commit_state')}")
                        prompt_tokens = max(
                            int(audit.get("prompt_tokens") or 0),
                            int(audit.get("prompt_estimated_tokens") or 0),
                        )
                        # Token usage is aggregated across proposal and repair
                        # calls. Keep the 24k ceiling per model call instead of
                        # rejecting a valid repaired turn on its summed usage.
                        model_calls = max(1, int(audit.get("model_calls") or 0))
                        if prompt_tokens > 24_000 * model_calls:
                            invariant_errors.append(f"prompt_tokens={prompt_tokens}")
                        if audit.get("deterministic_branch_match") and int(audit.get("model_calls") or 0) > 1:
                            invariant_errors.append(f"model_calls={audit.get('model_calls')}")
                        turn["turn_audit"] = audit
                        if invariant_errors:
                            raise RuntimeError(
                                "WA Validator turn invariant failed: "
                                + ", ".join(invariant_errors)
                            )
                        # The direct/internal driver, not the WhatsApp
                        # dispatch worker, consumed this canonical inbound.
                        # Only mark it terminal after the decision, proof and
                        # outbox invariants above all passed.
                        supabase_client.complete_whatsapp_buffer(buffer_uuid, "sent")
                except Exception as exc:
                    tb = traceback.format_exc()
                    _log.error(
                        "Step %d %s pipeline failed:\n%s", i, conversation_mode, tb
                    )
                    turn = {
                        "role": "bot",
                        "text": f"(erro: {exc})",
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "error": True,
                        "error_detail": tb,
                    }

                conversation.append(turn)
                _session_update(session_id, output={"conversation": list(conversation), "status": "running"})

                if turn.get("error"):
                    failure_output = {
                        "conversation": conversation,
                        "status": "error",
                        "technical_pass": False,
                        "quality_pass": False if semantic_mode else None,
                        "failed_turn": i,
                        "failure": "pipeline_error",
                    }
                    _session_update(
                        session_id, status="error", output=failure_output,
                        error=str(turn.get("text") or "pipeline_error"),
                    )
                    return

                await _wait_for_reply_delivered(
                    int(lead_ref),
                    outbound_message_id=str(turn.get("message_id") or ""),
                    expected_reply=str(turn.get("text") or ""),
                    max_wait_s=max(configured_wait, 20.0),
                )

                if semantic_mode:
                    proof_record = supabase_client.get_conversation_turn_proof(buffer_uuid) or {}
                    ledger_after = (
                        supabase_client.get_conversation_ledger(
                            str(persona.get("id") or ""), int(lead_ref),
                        )
                        or {}
                    )
                    journey_after = supabase_client.get_current_conversation_journey(
                        str(persona.get("id") or ""), int(lead_ref),
                    ) or {}
                    turn["journey_state"] = journey_after.get("state")
                    active_anchor = str(
                        ledger_after.get("active_branch_node_id")
                        or step.get("expected_branch_node_id")
                        or ""
                    )
                    active_anchors = list(dict.fromkeys([
                        active_anchor,
                        *[
                            str(value) for value in
                            (ledger_after.get("active_branch_node_ids") or [])
                            if value
                        ],
                    ]))
                    contract = _active_validator_contract(
                        graph_document, active_anchors,
                    )
                    if not proof_record or not ledger_after or not contract:
                        audit = {
                            "passed": False,
                            "criteria": {},
                            "failures": ["missing_semantic_commit_evidence"],
                            "qualification_complete": False,
                        }
                    elif step.get("kind") == "post_handoff_greeting":
                        audit = _post_handoff_greeting_audit(
                            customer_step=step,
                            turn=turn,
                            proof_record=proof_record,
                            ledger_before=ledger_before,
                            ledger_after=ledger_after,
                            journey_after=journey_after,
                        )
                    else:
                        audit = _semantic_turn_audit(
                            customer_step=step,
                            turn=turn,
                            proof_record=proof_record,
                            ledger_before=ledger_before,
                            ledger_after=ledger_after,
                            contract=contract,
                            recent_replies=recent_replies,
                            previous_question_node_id=previous_question_node_id,
                            expected_handoff=bool(driver.get("expected_handoff")),
                        )
                    turn["semantic_audit"] = audit
                    _session_update(
                        session_id,
                        output={
                            "conversation": list(conversation),
                            "status": "running",
                            "technical_pass": True,
                            "quality_pass": False,
                        },
                    )
                    if not audit.get("passed"):
                        failure, failure_output, failure_event = _semantic_failure_records(
                            conversation=conversation,
                            turn_index=i,
                            audit=audit,
                            session_id=session_id,
                            persona_slug=persona_slug,
                            buffer_id=buffer_uuid,
                            external_message_id=message_id,
                            correlation_id=correlation_id,
                            journey_state=str(journey_after.get("state") or "") or None,
                        )
                        _session_update(
                            session_id, status="error", output=failure_output,
                            error=failure,
                        )
                        supabase_client.insert_event(failure_event)
                        return

                    recent_replies.append(str(turn.get("text") or ""))
                    previous_question_node_id = audit.get("next_question_node_id")
                    if step.get("kind") == "post_handoff_greeting":
                        if not any(
                            queued.get("kind") == "post_handoff_greeting"
                            for queued in step_queue
                        ):
                            semantic_complete = True
                            break
                        i += 1
                        continue
                    if step.get("expected_active_branch_node_ids"):
                        expected_active_branches = [
                            str(value)
                            for value in step.get("expected_active_branch_node_ids") or []
                        ]
                    if audit.get("handoff_observed"):
                        post_handoff = list(driver.get("post_handoff_greetings") or [])
                        if post_handoff and not driver_state["post_handoff_started"]:
                            driver_state["post_handoff_started"] = True
                            step_queue.extend(post_handoff)
                            i += 1
                            continue
                        semantic_complete = True
                        break

                    if step.get("kind") in {
                        "field_answer", "loose_field_answer", "doubt_with_field_answer",
                    }:
                        answered_fields.update(
                            str(key) for key in (step.get("intended_facts") or {})
                        )
                    asked_field = str(audit.get("asked_field") or "")
                    next_step = _next_semantic_driver_step(
                        driver=driver,
                        state=driver_state,
                        asked_field=asked_field,
                        answered_fields=answered_fields,
                        active_anchor=active_anchor,
                        expected_active_branches=expected_active_branches,
                        qualification_complete=bool(audit.get("qualification_complete")),
                    )
                    if not next_step:
                        failure = f"script_question_mismatch:{asked_field or 'unknown'}"
                        failure_output = {
                            "conversation": conversation,
                            "status": "error",
                            "technical_pass": True,
                            "quality_pass": False,
                            "failed_turn": i,
                            "failure": failure,
                        }
                        _session_update(
                            session_id, status="error", output=failure_output,
                            error=failure,
                        )
                        return
                    step_queue.append(next_step)

                i += 1

            if semantic_mode and not semantic_complete:
                failure = "semantic_driver_exhausted_before_terminal_handoff"
                final_output = {
                    "conversation": conversation,
                    "status": "error",
                    "technical_pass": True,
                    "quality_pass": False,
                    "failure": failure,
                }
                _session_update(session_id, status="error", output=final_output, error=failure)
                return

            final_output = {
                "conversation": conversation,
                "status": "done",
                "technical_pass": True,
                "quality_pass": True if semantic_mode else None,
                "quality_scope": "semantic_graph_v1" if semantic_mode else "technical_only",
            }
            _session_update(session_id, status="done", output=final_output)

            supabase_client.insert_event({
                "event_type": "wa_validator_direct_done",
                "payload": {
                    "session_id": session_id,
                    "persona_slug": persona_slug,
                    "n_turns": len(conversation),
                    "graph_version": script.get("meta", {}).get("graph_version"),
                    "graph_checksum": script.get("meta", {}).get("graph_checksum"),
                    "conversation_mode": conversation_mode,
                    "classifier": "semantic_graph_v1" if semantic_mode else "deterministic_v1",
                    "quality_pass": True if semantic_mode else None,
                    "pipeline_contract": pipeline_contract,
                },
            })

        except Exception as exc:
            _session_update(
                session_id,
                status="error",
                error=str(exc),
                output={
                    "conversation": conversation,
                    "status": "error",
                    "technical_pass": False,
                    "quality_pass": False if semantic_mode else None,
                    "failure": "validator_execution_error",
                },
            )

    task = asyncio.create_task(_do_run())
    # CLI and container QA runs have no long-lived ASGI loop after the command
    # returns. Opt in to awaiting the deterministic scenario there, while the
    # API route retains its non-blocking behaviour by default.
    if claimed_session is not None or os.environ.get(
        "WA_VALIDATOR_DIRECT_WAIT", ""
    ).strip().lower() in {"1", "true", "yes"}:
        await task
    return get_session(session_id)


def _persona_business_model(persona_slug: str) -> str | None:
    """The persona's declared business_model, or None if it can't be resolved.

    Reuses conversation_runtime._business_model -- the same field the
    runtime itself gates appointment vs. sales behavior on -- rather than
    re-deriving it here. None (not a default) lets callers fail open when
    the graph isn't loadable, since this is a UX filter, not a security
    boundary.
    """
    try:
        _version, _checksum, graph = _published_graph(persona_slug)
        return conversation_runtime._business_model(graph)
    except Exception:
        return None


def flows(persona_slug: str | None = None) -> list:
    business_model = _persona_business_model(persona_slug) if persona_slug else None
    return _flows_for_business_model(business_model)


def _flows_for_business_model(business_model: str | None) -> list:
    return [
        {"id": k, "label": v.split(":")[0]}
        for k, v in _FLOWS.items()
        if business_model is None
        or not _FLOW_BUSINESS_MODELS.get(k)
        or business_model in _FLOW_BUSINESS_MODELS[k]
    ]

