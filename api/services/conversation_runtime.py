"""Graph-backed Vitoria conversation runtime used by persona n8n workflows."""
from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from typing import Any, Literal

logger = logging.getLogger("conversation_runtime")

from schemas.conversation import (
    AgentResponse,
    CartAction,
    ConversationContext,
    ConversationDecision,
    ConversationProposal,
    ConversationRoute,
)
from services import (
    context_cards as context_cards_service,
    graph_agent_runtime_v3,
    graph_conversation_contract,
    graph_json_v2_store,
    lead_qualification,
    model_pricing,
    supabase_client,
    whatsapp_outbox,
)
from services.model_router import get_router
from services.deterministic_sdr import (
    DeterministicSDR,
    _quantity,
    catalog_from_graph,
)
from services.deterministic_appointment import DeterministicAppointment


def emit_turn_event(
    *,
    agent_name: str,
    trace_id: str | None,
    lead_ref: int | str | None,
    persona_id: str | None = None,
    status: str = "success",
    model_used: str | None = None,
    token_input: int | None = None,
    token_output: int | None = None,
    latency_ms: int | None = None,
    error_msg: str | None = None,
    input_data: dict[str, Any] | None = None,
    output_data: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write one observability event (one agent_logs row) for one step of a
    conversation turn -- context/decide/repair/commit/error, all sharing the
    same trace_id (the turn's inbound_buffer_id) so a single SQL query can
    reconstruct the whole turn. Best-effort: never raises, never blocks the
    reply -- a lost observability row is acceptable, a broken reply is not.

    Written with `metadata` shaped so trace_id/conversation_id/
    n8n_execution_id/cost_usd/quality_score are picked up by the generated
    columns added in migration 106, without needing a schema change here.
    """
    if not trace_id:
        return
    try:
        supabase_client.insert_agent_log(
            {
                "lead_id": str(lead_ref) if lead_ref is not None else None,
                "persona_id": persona_id,
                "agent_type": agent_name,
                "action": f"[{'ERROR' if status == 'error' else 'INFO'}] {agent_name}",
                "decision": error_msg or "",
                "input": input_data or {},
                "output": output_data or {},
                "latency_ms": latency_ms,
                "model_used": model_used,
                "token_input": token_input,
                "token_output": token_output,
                "metadata": {
                    "level": "ERROR" if status == "error" else "INFO",
                    "component": "conversation_runtime",
                    "trace_id": trace_id,
                    "conversation_id": lead_ref,
                    **(metadata or {}),
                },
            }
        )
    except Exception:  # noqa: BLE001 â€” observability must never break the turn
        logger.warning(
            "emit_turn_event failed agent_name=%s trace_id=%s", agent_name, trace_id,
            exc_info=True,
        )


STAGES = ("novo", "contatado", "engajado", "qualificado", "oportunidade")
INTENT_ACTIONS = {
    "add_item": CartAction.ADD_ITEM,
    "change_quantity": CartAction.CHANGE_QUANTITY,
    "remove_item": CartAction.REMOVE_ITEM,
    "consult_price": CartAction.SHOW_TOTAL,
    "confirm_order": CartAction.CONFIRM_ORDER,
    "deny_order": CartAction.CANCEL_ORDER,
}
CLOSER_INTENTS = {
    "add_item",
    "change_quantity",
    "remove_item",
    "provide_name",
    "provide_address",
    "confirm_order",
    "deny_order",
}


class PublishedGraphUnavailable(RuntimeError):
    pass


class ConversationCommitFailed(RuntimeError):
    def __init__(self, *, failed_step: str, inbound_id: str | None,
                 correlation_id: str | None = None, cause: Exception):
        self.failed_step = failed_step
        self.inbound_id = inbound_id
        self.postgres_code = str(getattr(cause, "code", "") or "") or None
        self.constraint = str(getattr(cause, "constraint_name", "") or "") or None
        self.correlation_id = correlation_id
        super().__init__(f"conversation commit failed at {failed_step}")

    def canonical_result(self) -> dict[str, Any]:
        return {
            "ok": False,
            "technical_failure": True,
            "failed_step": self.failed_step,
            "postgres_code": self.postgres_code,
            "constraint": self.constraint,
            "canonical_inbound_id": self.inbound_id,
            "inbound_id": self.inbound_id,
            "correlation_id": self.correlation_id,
            "commit_state": "failed",
        }


def _commit_graph_turn_and_outbox_or_raise(
    *, inbound_id: str | None, correlation_id: str, **payload: Any,
) -> dict:
    try:
        return supabase_client.commit_graph_turn_and_outbox_v4(**payload)
    except Exception as exc:
        logger.exception(
            "atomic conversation commit failed step=commit_graph_turn_and_outbox_v4 "
            "inbound_id=%s correlation_id=%s",
            inbound_id,
            correlation_id,
        )
        raise ConversationCommitFailed(
            failed_step="commit_graph_turn_and_outbox_v4",
            inbound_id=inbound_id,
            correlation_id=correlation_id,
            cause=exc,
        ) from exc


class ModelDecisionError(RuntimeError):
    pass


def _json_object(raw: str) -> dict[str, Any]:
    start = (raw or "").find("{")
    end = (raw or "").rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model output does not contain a JSON object")
    parsed = json.loads(raw[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("model output must be an object")
    return parsed


def strict_model_decision(
    context: ConversationContext,
    *,
    router: Any | None = None,
    model: str | None = None,
) -> ConversationDecision:
    """Validate strict model JSON, allowing exactly one correction attempt."""
    router = router or get_router()
    model = model or os.environ.get("CONVERSATION_CLASSIFIER_MODEL") or ""
    if not model:
        raise ModelDecisionError("conversation classifier model is not configured")
    evidence = [
        {
            "id": node.get("id"),
            "node_type": node.get("node_type"),
            "slug": node.get("slug"),
            "title": node.get("title"),
        }
        for node in context.rag_nodes
    ]
    schema = ConversationDecision.model_json_schema()
    prompt = (
        "Classifique a conversa usando apenas os IDs de evidÃªncia fornecidos. "
        "NÃ£o calcule preÃ§o ou total e nÃ£o altere o carrinho. Responda somente "
        "com JSON compatÃ­vel com o schema.\n"
        f"SCHEMA={json.dumps(schema, ensure_ascii=False)}\n"
        f"MESSAGES={json.dumps(context.messages[-20:], ensure_ascii=False)}\n"
        f"CART={json.dumps(context.cart, ensure_ascii=False)}\n"
        f"EVIDENCE={json.dumps(evidence, ensure_ascii=False)}"
    )
    error = ""
    for attempt in range(2):
        request = prompt
        if attempt:
            request += (
                "\nA saÃ­da anterior foi invÃ¡lida. Corrija uma Ãºnica vez. "
                f"ERRO={error}"
            )
        try:
            raw = router.chat(model, request, max_tokens=700)
            decision = ConversationDecision.model_validate(_json_object(raw))
            allowed_ids = {
                str(node.get("id")) for node in context.rag_nodes if node.get("id")
            }
            if not set(decision.evidence_node_ids).issubset(allowed_ids):
                raise ValueError("model referenced evidence outside context")
            return decision
        except Exception as exc:
            error = str(exc)[:1000]
    raise ModelDecisionError(error or "invalid model decision")


def _norm(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    normalized = "".join(
        character for character in normalized
        if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9]+", " ", normalized.lower()).strip()


def _node_path(graph: Any, node_id: str) -> list[str]:
    by_id = {node.id: node for node in graph.nodes}
    result: list[str] = []
    current = by_id.get(node_id)
    seen: set[str] = set()
    while current and current.id not in seen:
        seen.add(current.id)
        result.insert(0, current.id)
        current = by_id.get(current.parent_id or "")
    return result


def _current_graph(persona_slug: str) -> tuple[int, str, Any]:
    current = graph_json_v2_store.load_current(persona_slug)
    if not current:
        raise PublishedGraphUnavailable(
            f"No published Graph JSON v2 for {persona_slug}"
        )
    version, graph = current
    graph.graph_version = version
    event = graph_json_v2_store.latest_event(persona_slug) or {}
    checksum = str(
        ((event.get("payload") or {}).get("checksum"))
        or graph_json_v2_store.checksum_graph(graph)
    )
    if graph.status != "published" or not graph.validation.is_valid:
        raise PublishedGraphUnavailable(
            f"Published graph is not valid for {persona_slug}"
        )
    return version, checksum, graph


def _business_model(graph: Any) -> str:
    persona = next(
        (node for node in graph.nodes if node.node_type == "persona"),
        None,
    )
    data = (persona.data or {}) if persona else {}
    metadata = data.get("metadata") or {}
    return str(
        data.get("business_model")
        or metadata.get("business_model")
        or "sales"
    ).strip().lower()


def _appointment_policy(graph: Any) -> dict[str, Any]:
    persona = next(
        (node for node in graph.nodes if node.node_type == "persona"),
        None,
    )
    data = (persona.data or {}) if persona else {}
    return dict(data.get("appointment_policy") or {})


# Non-negotiable regardless of persona content: the n8n agentic reply must
# never be the sole authority on a price or schedule confirmation.
# conversation_runtime.commit()'s _reply_confirms_price_or_schedule guard
# already enforces this server-side independent of what the agent
# produces â€” this instruction is a first line of defense, not the only one.
_AGENTIC_SAFETY_INSTRUCTIONS = (
    "Nunca confirme preco final, disponibilidade, data ou horario sem uma "
    "regra e evidencia publicadas que autorizem isso. Responda apenas com "
    "base no pacote aprovado do galho ativo e cite somente IDs fornecidos. "
    "Retorne uma proposta JSON com branch_anchor_node_id, "
    "branch_path_checksum, extracted_facts, next_question_node_id, "
    "cited_node_ids, reply, qualification_complete e handoff_requested. "
    "Cada fato usa field_key, value, status, source_message_id e "
    "owner_node_id. Os status permitidos sao known, unknown, declined, "
    "needs_confirmation e invalid. Quando a pessoa disser que nao sabe, "
    "nao possui ou nao consegue informar, use unknown com value null; "
    "nunca salve essa frase como valor conhecido. Leia a mensagem inteira "
    "e capture tambem informacoes mencionadas de passagem. Use "
    "extracted_facts, que substitui o contrato legado extracted_fields. "
    "Nao invente campos, "
    "nodes, perguntas, produtos, regras ou evidencias."
)


# tone/rule nodes are concatenated into every turn's system prompt
# unconditionally (no per-query relevance filtering, unlike the RAG card
# budgets in context_cards.py/graph_agent_runtime_v3.py) -- this budget
# stops that block from growing without bound as operators publish more
# tone/rule nodes. ~4 chars/token, matching the estimator used elsewhere
# in this codebase (context_cards.py token_count).
TONE_RULE_TOKEN_BUDGET = 1500


def _cap_blocks_by_token_budget(blocks: list[str], budget: int, *, label: str) -> str:
    kept: list[str] = []
    used_tokens = 0
    dropped = 0
    for block in blocks:
        estimated = max(1, len(block) // 4)
        if kept and used_tokens + estimated > budget:
            dropped += 1
            continue
        kept.append(block)
        used_tokens += estimated
    if dropped:
        logger.warning(
            "build_system_prompt: dropped %d %s block(s) past %d-token budget",
            dropped, label, budget,
        )
    return "\n\n".join(kept).strip()


def build_system_prompt(graph: Any, cards: list[Any] | None = None) -> str:
    """Compose this persona's n8n agentic system prompt from its own graph.

    Reads generic node types (persona, tone, rule) present in any persona
    graph. A persona with no tone/rule nodes still gets a valid, if minimal,
    prompt. Replaces a static prompt formerly embedded in a persona export and
    reused verbatim by
    every persona on the agentic template.
    """
    persona_node = next(
        (node for node in graph.nodes if node.node_type == "persona"),
        None,
    )
    persona_data = (persona_node.data or {}) if persona_node else {}
    persona_name = (persona_node.label if persona_node else None) or "a empresa"
    summary = str(persona_data.get("summary") or "").strip()

    if cards is not None:
        persona_name = "a empresa"
        summary = ""
        tone_text = _cap_blocks_by_token_budget(
            [str(card.rendered_content).strip() for card in cards if card.node_type == "tone"],
            TONE_RULE_TOKEN_BUDGET, label="tone",
        )
        rule_text = _cap_blocks_by_token_budget(
            [str(card.rendered_content).strip() for card in cards if card.node_type == "rule"],
            TONE_RULE_TOKEN_BUDGET, label="rule",
        )
        persona_card = next((card for card in cards if card.node_type == "persona"), None)
        if persona_card:
            persona_name = persona_card.title
            summary = persona_card.rendered_content
    else:
        tone_text = _cap_blocks_by_token_budget(
            [
                str(node.data.get("markdown") or node.data.get("summary") or "").strip()
                for node in graph.nodes
                if node.node_type == "tone" and ((node.data or {}).get("markdown") or (node.data or {}).get("summary"))
            ],
            TONE_RULE_TOKEN_BUDGET, label="tone",
        )
        rule_text = _cap_blocks_by_token_budget(
            [
                str(node.data.get("markdown") or node.data.get("summary") or "").strip()
                for node in graph.nodes
                if node.node_type == "rule" and ((node.data or {}).get("markdown") or (node.data or {}).get("summary"))
            ],
            TONE_RULE_TOKEN_BUDGET, label="rule",
        )

    lines = [
        f"Voce e o agente de atendimento (SDR) de {persona_name}.",
    ]
    if summary:
        lines.append(summary)
    if tone_text:
        lines.append("Tom de voz:\n" + tone_text)
    if rule_text:
        lines.append("Regras operacionais e comerciais:\n" + rule_text)
    lines.append(_AGENTIC_SAFETY_INSTRUCTIONS)
    if graph_conversation_contract.price_disclosure_is_human_only(
        _appointment_policy(graph)
    ):
        # Only when the persona's own graph declares it: other personas
        # legitimately quote published prices and must keep doing so.
        lines.append(
            "Nunca informe preco, faixa de preco, valor aproximado, "
            "parcelamento ou desconto de qualquer servico, nem mesmo uma "
            "estimativa ou um 'a partir de'. O valor e sempre definido por "
            "uma pessoa da equipe. Quando perguntarem sobre valores, colete "
            "as informacoes necessarias e encaminhe a conversa para o "
            "atendimento humano."
        )
    lines.append(
        "Enquanto estiver ativo, responda o que voce consegue com base no "
        "conhecimento aprovado e conduza a conversa fazendo perguntas de "
        "qualificacao adequadas, como um bom vendedor faria â€” nao apenas "
        "responda e espere a proxima pergunta. Va com calma: peca apenas "
        "UMA informacao pendente por mensagem (no maximo duas, e so se "
        "forem intimamente relacionadas, como marca e modelo do mesmo "
        "item). Nunca liste tres ou mais perguntas na mesma mensagem, "
        "principalmente na primeira interacao com o cliente â€” isso "
        "cansa e afasta. Deixe a conversa fluir por varias mensagens "
        "curtas, como uma pessoa conversando de verdade, nao um "
        "formulario."
    )
    return "\n\n".join(lines)


def _commercial_note_fields(context: ConversationContext) -> list[str]:
    """Field names this persona wants mirrored into leads.metadata.commercial_note.

    Declared per persona in its graph's persona-node data
    (`commercial_note_fields: [...]`), not hardcoded here â€” any business
    model, any engine. commit() only has `context.rag_nodes` (flattened
    node dicts), not the full GraphJson, so this reads from there instead
    of `_appointment_policy`.
    """
    persona_node = next(
        (item for item in context.rag_nodes if item.get("node_type") == "persona"),
        None,
    )
    data = (persona_node or {}).get("data") or {}
    fields = data.get("commercial_note_fields") or []
    return [str(field) for field in fields if field]


def _merge_extracted_fields(cart_state: dict[str, Any], extracted_fields: dict[str, str]) -> dict[str, Any]:
    """Apply agentic-flow field extraction into appointment_request, safely.

    Confirmed live 2026-08-01: the deterministic engine's own extraction
    (deterministic_appointment._collect) only accepts a value for whichever
    field is next in missing_fields, and treats the *entire* message as
    that field's value. A customer answering several fields in one
    natural sentence containing multiple qualification facts only ever
    fills the first â€” the agent's reply claimed the rest was "anotado"
    when it never reached appointment_request at all.

    The agentic engine (DeepSeek) reads the whole message and the list of
    missing fields, so it can extract more than one at once â€” but it must
    never be trusted blindly: only fields genuinely still in
    missing_fields get accepted, and an already-filled field is never
    overwritten by agent output, exactly mirroring _collect()'s own
    `request.get(expected)` guard.
    """
    if not extracted_fields:
        return cart_state
    missing = list(cart_state.get("missing_fields") or [])
    if not missing:
        return cart_state
    request = dict(cart_state.get("appointment_request") or {})
    still_missing = []
    for field in missing:
        value = extracted_fields.get(field)
        if value and str(value).strip() and not request.get(field):
            request[field] = str(value).strip()
        else:
            still_missing.append(field)
    return {**cart_state, "appointment_request": request, "missing_fields": still_missing}


def _resolve_identified_service(
    extracted_fields: dict[str, str],
    identified_service_slug: str | None,
    available_services: list[dict[str, str]],
) -> dict[str, str]:
    """Fold an AI-inferred service into extracted_fields, safely.

    Confirmed live 2026-08-01: a customer describing a symptom ("risco
    fundo na porta") got a reply that correctly recommended "chapeaÃ§Ã£o",
    but "servico" stayed in missing_fields forever â€” the customer never
    said the word themselves, so the extraction contract (which only
    covers fields the customer explicitly answered) never captured it.
    identified_service_slug lets the model report what it inferred, but
    it's only trusted when it matches a real slug from this persona's own
    catalog â€” never an invented one â€” and never overrides a value the
    customer (or a prior turn) already provided.
    """
    if not identified_service_slug or "servico" in extracted_fields:
        return extracted_fields
    valid_slugs = {svc.get("slug") for svc in available_services}
    if identified_service_slug not in valid_slugs:
        return extracted_fields
    return {**extracted_fields, "servico": identified_service_slug}


def _next_field_question(cart_state: dict[str, Any], context: ConversationContext) -> str | None:
    """The question for whatever field is next in missing_fields.

    Read from this persona's own graph data (appointment_policy.field_
    questions), same source _commercial_note_fields uses â€” no hardcoded
    field names or questions here, any persona with an appointment_policy.
    """
    if cart_state.get("business_model") != "appointment":
        return None
    missing = cart_state.get("missing_fields") or []
    if not missing:
        return None
    persona_node = next(
        (item for item in context.rag_nodes if item.get("node_type") == "persona"),
        None,
    )
    field_questions = (
        (persona_node or {}).get("data", {}).get("appointment_policy", {}).get("field_questions")
        or {}
    )
    question = field_questions.get(missing[0])
    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            "published graph appointment_policy.field_questions is missing "
            f"required field {missing[0]}"
        )
    return question.strip()


def _ensure_trailing_question(reply_text: str | None, cart_state: dict[str, Any], context: ConversationContext) -> str | None:
    """Never leave the customer without a next step while info is missing.

    Confirmed live 2026-08-01: the model's candidate correctly asked the
    graph-defined next question, but got discarded by the unsafe-price
    filter and fell back to the deterministic engine's plain price-fact
    reply â€” which never asks anything, silently dropping the qualification
    flow. Whatever reply text ends up used (agentic or deterministic
    fallback), if it doesn't already end in a question and fields are
    still missing, this appends the graph's own next question for it â€”
    a structural guarantee, not just a prompt instruction the model (or a
    safety fallback) can skip.
    """
    if reply_text is None:
        return None
    text = reply_text.strip()
    if text.endswith("?"):
        return reply_text
    question = _next_field_question(cart_state, context)
    if not question:
        return reply_text
    if not text:
        return question
    return f"{text} {question}"


def _approved_faq_match(graph: Any, message: str) -> Any | None:
    """Match only an exact, validated FAQ question.

    Exact normalization keeps graph answers authoritative without introducing
    fuzzy guesses. FAQ interruptions must not discard an in-progress
    appointment request; the caller preserves the cart state unchanged.
    """
    query = _norm(message)
    if not query:
        return None
    for node in graph.nodes:
        if node.node_type != "faq":
            continue
        data = node.data or {}
        if str(data.get("status") or "").lower() not in {
            "approved",
            "validated",
            "active",
            "ativo",
        }:
            continue
        questions = (
            str(data.get("question") or ""),
            str(node.label or ""),
        )
        if any(query == _norm(question) for question in questions if question):
            return node
    return None


def _relevant_nodes(graph: Any, query: str, *, limit: int = 30) -> list[Any]:
    terms = {term for term in _norm(query).split() if len(term) > 1}
    mandatory = {"persona", "brand", "tone", "rule", "briefing"}
    scored: list[tuple[int, Any]] = []
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
        haystack = _norm(
            " ".join(
                (
                    node.slug,
                    node.label,
                    str(data.get("markdown") or ""),
                    " ".join(str(alias) for alias in data.get("aliases") or []),
                )
            )
        )
        score = sum(1 for term in terms if term in haystack)
        if node.node_type in mandatory:
            score += 100
        if score:
            scored.append((score, node))
    scored.sort(key=lambda item: (item[0], item[1].id), reverse=True)
    selected: dict[str, Any] = {
        node.id: node for _, node in scored[:limit]
    }
    by_id = {node.id: node for node in graph.nodes}
    for node in list(selected.values()):
        for path_id in _node_path(graph, node.id):
            if path_id in by_id:
                selected[path_id] = by_id[path_id]
    return list(selected.values())


def build_context(
    *,
    persona_slug: str,
    lead_ref: int,
    message: str,
    message_id: str | None = None,
    trace_id: str | None = None,
) -> ConversationContext:
    _turn_started_at = time.monotonic()
    lead_binding_probe = supabase_client.get_lead_by_ref(lead_ref) or {}
    binding_probe = supabase_client.get_workflow_binding_by_id(
        lead_binding_probe.get("channel_binding_id")
    )
    binding_probe_metadata = (binding_probe or {}).get("metadata") or {}
    if (
        not graph_agent_runtime_v3.binding_uses_v3(binding_probe)
        and binding_probe_metadata.get("shadow_runtime_version")
        == graph_agent_runtime_v3.RUNTIME_VERSION
    ):
        try:
            shadow = graph_agent_runtime_v3.build_context(
                persona_slug=persona_slug,
                lead_ref=lead_ref,
                message=message,
                message_id=message_id,
            )
            supabase_client.insert_event(
                {
                    "event_type": "conversation.graph_v3_shadow_context",
                    "entity_type": "lead",
                    "entity_id": str(lead_ref),
                    "persona_id": lead_binding_probe.get("persona_id"),
                    "payload": {
                        "runtime_version": graph_agent_runtime_v3.RUNTIME_VERSION,
                        "publication_id": shadow.publication_id,
                        "graph_checksum": shadow.graph_checksum,
                        "retrieval_trace": shadow.retrieval_trace,
                        "outbound_suppressed": True,
                        "trace_id": trace_id,
                    },
                },
                source="conversation_runtime.shadow",
            )
        except Exception as exc:  # shadow must never change the live decision
            supabase_client.insert_event(
                {
                    "event_type": "conversation.graph_v3_shadow_failed",
                    "entity_type": "lead",
                    "entity_id": str(lead_ref),
                    "persona_id": lead_binding_probe.get("persona_id"),
                    "payload": {"error": str(exc)[:1000], "outbound_suppressed": True, "trace_id": trace_id},
                },
                level="warning",
                source="conversation_runtime.shadow",
            )
    if graph_agent_runtime_v3.binding_uses_v3(binding_probe):
        try:
            v3_context = graph_agent_runtime_v3.build_context(
                persona_slug=persona_slug,
                lead_ref=lead_ref,
                message=message,
                message_id=message_id,
            )
        except RuntimeError as exc:
            raise PublishedGraphUnavailable(str(exc)) from exc
        emit_turn_event(
            agent_name="conversation.context_built",
            trace_id=trace_id,
            lead_ref=lead_ref,
            persona_id=lead_binding_probe.get("persona_id"),
            latency_ms=int((time.monotonic() - _turn_started_at) * 1000),
            output_data={
                "graph_version": v3_context.graph_version,
                "graph_checksum": v3_context.graph_checksum,
                "context_cards": [
                    {"id": card.id, "node_type": card.node_type, "title": card.title}
                    for card in v3_context.context_cards
                ],
                "runtime_version": v3_context.runtime_version,
            },
            metadata={"persona_slug": persona_slug, "message_id": message_id},
        )
        return v3_context
    version, checksum, graph = _current_graph(persona_slug)
    lead = supabase_client.get_lead_by_ref(lead_ref) or {}
    persona = supabase_client.get_persona(persona_slug) or {}
    if lead.get("persona_id") and persona.get("id") and persona["id"] != lead["persona_id"]:
        raise PermissionError("lead does not belong to requested persona")
    messages = supabase_client.get_messages(str(lead_ref), limit=20) or []
    if message and not any(
        str(row.get("message_id") or row.get("external_message_id") or "")
        == str(message_id or "")
        for row in messages
    ):
        messages.append(
            {
                "message_id": message_id,
                "sender_type": "lead",
                "role": "user",
                "texto": message,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    messages = messages[-20:]
    lead_metadata = lead.get("metadata") or {}
    cart = dict(
        lead_metadata.get("conversation_state")
        or lead_metadata.get("vitoria_state")
        or {}
    )
    prior_runtime = lead_metadata.get("conversation_runtime") or {}
    prior_version = prior_runtime.get("graph_version")
    prior_checksum = prior_runtime.get("graph_checksum")
    # Graph-backed selections must be revalidated on a new publication, but
    # customer-provided facts are not graph evidence and must survive.  The
    # decision engine receives an explicit marker and applies business-model
    # policy: appointment flows can safely rebind their service against the
    # current catalog, while transactional carts retain the conservative
    # handoff behavior below.
    try:
        graph_changed = bool(prior_version) and (
            int(prior_version) != version
            or str(prior_checksum or "") != checksum
        )
    except (TypeError, ValueError):
        graph_changed = True
    if graph_changed:
        cart["_graph_changed"] = True
        cart["_previous_graph_version"] = prior_version
        cart["_previous_graph_checksum"] = prior_checksum
    cart.setdefault("_lead_stage", lead.get("stage") or "novo")
    previous_branch_id = str(cart.get("active_branch_node_id") or "") or None
    branch_resolution = graph_conversation_contract.resolve_branch_anchor(
        graph,
        message,
        previous_anchor_node_id=previous_branch_id,
    )
    active_branch_id = branch_resolution["branch_anchor_node_id"]
    contract = graph_conversation_contract.compile_branch_contract(
        graph, active_branch_id
    )
    branch_ids = graph_conversation_contract.branch_closure(graph, active_branch_id)
    active_coordinate = (
        graph_conversation_contract.coordinate_for_node(
            graph, active_branch_id, graph_version=version
        )
        if active_branch_id else {}
    )
    if branch_resolution["branch_changed"]:
        allowed_fields = set(contract.get("required_fields") or [])
        cart["facts"] = {
            key: value for key, value in (cart.get("facts") or {}).items()
            if key in allowed_fields
        }
        cart["appointment_request"] = {
            key: value for key, value in (cart.get("appointment_request") or {}).items()
            if key in allowed_fields
        }
        cart["asked_question_node_ids"] = []
    if active_branch_id:
        cart["active_branch_node_id"] = active_branch_id
        cart["active_path_checksum"] = active_coordinate.get("path_checksum")
        branch_node = next(node for node in graph.nodes if node.id == active_branch_id)
        # Compatibility hint for the legacy deterministic fallback.  The
        # graph contract itself never assumes the name of a branch field.
        cart["_identified_service_slug"] = branch_node.slug
    ledger = graph_conversation_contract.ledger_from_state(cart, contract)
    unresolved_fields = graph_conversation_contract.missing_fields(ledger, contract)
    # Golden Dataset RAG (knowledge_rag_entries/knowledge_rag_chunks,
    # persona-scoped, approved/validated only) â€” a real retrieval layer
    # distinct from rag_nodes' in-memory graph-node keyword filter above.
    # Built for the n8n agentic flow (deterministic ignores this field and
    # keeps using rag_nodes/rag_paths unchanged).
    try:
        rag_chunks = supabase_client.search_active_rag_chunks(
            persona_id=str(persona.get("id") or lead.get("persona_id") or ""),
            query=message,
            limit=12,
            branch_anchor_node_id=active_branch_id,
            allowed_node_ids=sorted(branch_ids),
            active_path_node_ids=active_coordinate.get("path_node_ids") or [],
            unresolved_fields=unresolved_fields,
            graph_version=version,
        )
    except Exception as exc:  # noqa: BLE001 â€” RAG is best-effort context, never fatal
        logger.warning("search_active_rag_chunks failed: %s", exc)
        rag_chunks = []
    try:
        system_prompt = build_system_prompt(graph)
    except Exception as exc:  # noqa: BLE001 â€” the deterministic path never needs this
        logger.warning("build_system_prompt failed: %s", exc)
        system_prompt = ""
    available_services = [
        {"slug": node.slug, "label": node.label}
        for node in graph.nodes
        if node.node_type in ("product", "service") and node.slug
    ]
    agent_slug = next(
        (
            str((node.data or {}).get("metadata", {}).get("agent_slug"))
            for node in graph.nodes
            if (node.data or {}).get("metadata", {}).get("agent_slug")
        ),
        str(
            (
                next(
                    (
                        (node.data or {}).get("metadata", {})
                        for node in graph.nodes
                        if node.node_type == "persona"
                    ),
                    {},
                )
            ).get("public_agent_slug")
            or "agent"
        ),
    )
    try:
        projection_nodes, _projection_edges = supabase_client.list_all_knowledge_graph(
            persona_id=str(persona.get("id") or lead.get("persona_id") or ""),
            limit_nodes=5000,
        )
    except Exception as exc:  # projection UUIDs are optional trace metadata
        logger.warning("list_all_knowledge_graph failed: %s", exc)
        projection_nodes = []
    cards = context_cards_service.resolve_cards(
        graph=graph,
        graph_version=version,
        graph_checksum=checksum,
        query=message,
        rag_chunks=rag_chunks,
        projection_nodes=projection_nodes,
        branch_node_ids=sorted(branch_ids),
        active_path_node_ids=active_coordinate.get("path_node_ids") or [],
        unresolved_fields=unresolved_fields,
        max_cards=24,
        max_tokens=8000,
    )
    if not cards:
        context_cards_service.emit_metric(
            "knowledge_context.empty",
            persona_id=persona.get("id"),
            lead_ref=lead_ref,
            payload={"persona_slug": persona_slug, "graph_version": version},
        )
    graph_nodes_by_id = {node.id: node for node in graph.nodes}
    nodes = [graph_nodes_by_id[card.id] for card in cards if card.id in graph_nodes_by_id]
    try:
        system_prompt = build_system_prompt(graph, cards)
    except Exception as exc:
        logger.warning("build_system_prompt from context cards failed: %s", exc)
    emit_turn_event(
        agent_name="conversation.context_built",
        trace_id=trace_id,
        lead_ref=lead_ref,
        persona_id=persona.get("id"),
        latency_ms=int((time.monotonic() - _turn_started_at) * 1000),
        output_data={
            "graph_version": version,
            "graph_checksum": checksum,
            "context_cards": [{"id": card.id, "node_type": card.node_type, "title": card.title} for card in cards],
            "runtime_version": (binding_probe or {}).get("metadata", {}).get("runtime_version"),
        },
        metadata={"persona_slug": persona_slug, "message_id": message_id},
    )
    return ConversationContext(
        persona_slug=persona_slug,
        agent_slug=agent_slug,
        graph_version=version,
        graph_checksum=checksum,
        messages=messages,
        cart=cart,
        rag_nodes=[
            {
                "id": node.id,
                "node_type": node.node_type,
                "slug": node.slug,
                "title": node.label,
                "status": (node.data or {}).get("status"),
                "source": (node.data or {}).get("source"),
                "data": node.data or {},
            }
            for node in nodes
        ],
        rag_paths=[_node_path(graph, node.id) for node in nodes],
        rag_chunks=rag_chunks,
        context_cards=cards,
        system_prompt=system_prompt,
        available_services=available_services,
        active_branch_node_id=active_branch_id,
        active_path_checksum=active_coordinate.get("path_checksum"),
        branch_node_ids=sorted(branch_ids),
        graph_contract=contract,
    )


def _message(context: ConversationContext) -> str:
    for message in reversed(context.messages):
        role = str(
            message.get("role") or message.get("sender_type") or ""
        ).lower()
        if role in {"user", "lead", "client", "cliente"}:
            return str(
                message.get("texto")
                or message.get("content")
                or message.get("text")
                or ""
            )
    return ""


def _observation_value(value: Any, *, limit: int = 160) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split()).strip(" .,")
    return normalized[:limit] if len(normalized) >= 2 else None


def _apply_model_fields(
    state: dict[str, Any],
    observation: dict[str, Any] | None,
    *,
    business_model: str,
    graph: Any | None = None,
) -> dict[str, Any]:
    """Append model-extracted facts without granting it business authority."""
    fields = (observation or {}).get("fields")
    identified_service_raw = (observation or {}).get("identified_service_slug")
    if not isinstance(fields, dict) and not identified_service_raw:
        return state
    if not isinstance(fields, dict):
        fields = {}
    merged = dict(state)
    if business_model == "appointment":
        if graph is None:
            raise ValueError("appointment model fields require published graph policy")
        request = dict(merged.get("appointment_request") or {})
        identified_service = _observation_value(identified_service_raw)
        service_nodes = {
            node.slug: node
            for node in graph.nodes
            if node.node_type in {"product", "service"} and node.slug
        }
        if identified_service in service_nodes:
            # A service identified from the current inbound turn is allowed to
            # replace a historical choice.  The deterministic appointment
            # engine will recompute required fields from that service node.
            request["servico"] = identified_service
            merged["_identified_service_slug"] = identified_service

        selected_service = service_nodes.get(str(request.get("servico") or ""))
        booking = (
            (selected_service.data or {}).get("booking")
            if selected_service is not None
            else {}
        )
        policy_required = _appointment_policy(graph).get("required_fields") or []
        required_fields = (
            booking.get("required_fields")
            if isinstance(booking, dict) and booking.get("required_fields")
            else policy_required
        )
        # Graph-required fields are the extraction allowlist. This also works
        # on the first qualification turn, before missing_fields has ever been
        # persisted, without embedding any field names in the backend.
        for field in required_fields:
            field = str(field).strip()
            value = _observation_value(fields.get(field)) if field else None
            if value and not request.get(field):
                request[field] = value
        merged["appointment_request"] = request
        return merged

    name = _observation_value(fields.get("customer_name"))
    if name and not merged.get("customer_name"):
        merged["customer_name"] = name
    address = fields.get("delivery_address")
    if isinstance(address, dict):
        current_address = dict(merged.get("address") or {})
        for field in (
            "street",
            "number",
            "complement",
            "neighborhood",
            "city",
            "state",
            "zip",
        ):
            value = _observation_value(address.get(field), limit=100)
            if value and not current_address.get(field):
                current_address[field] = value
        if current_address:
            merged["address"] = current_address
    return merged


def _proposal_from_observation(
    observation: dict[str, Any] | None,
) -> ConversationProposal | None:
    if not isinstance(observation, dict):
        return None
    raw = observation.get("proposal")
    if not isinstance(raw, dict) and "branch_anchor_node_id" in observation:
        raw = observation
    if not isinstance(raw, dict):
        return None
    normalized = dict(raw)
    normalized_facts = []
    for fact in normalized.get("extracted_facts") or []:
        if not isinstance(fact, dict):
            continue
        candidate = dict(fact)
        if candidate.get("status") not in graph_conversation_contract.FACT_STATUSES:
            candidate["status"] = "invalid"
            candidate["value"] = None
        normalized_facts.append(candidate)
    normalized["extracted_facts"] = normalized_facts
    return ConversationProposal.model_validate(normalized)


def _state_from_ledger(
    state: dict[str, Any],
    ledger: dict[str, Any],
    contract: dict[str, Any],
    missing: list[str],
) -> dict[str, Any]:
    """Persist a factual ledger while preserving the legacy read projection."""
    next_state = {
        **state,
        "business_model": "appointment",
        "active_branch_node_id": ledger.get("active_branch_node_id"),
        "active_path_checksum": ledger.get("active_path_checksum"),
        "facts": ledger.get("facts") or {},
        "asked_question_node_ids": ledger.get("asked_question_node_ids") or [],
        "missing_fields": missing,
        "conversation_state": "collecting" if missing else "qualified",
    }
    declared = {field["key"] for field in contract.get("fields") or []}
    request = {
        key: fact.get("value")
        for key, fact in (ledger.get("facts") or {}).items()
        if key in declared
        and isinstance(fact, dict)
        and fact.get("status") == "known"
        and fact.get("value") not in (None, "")
    }
    next_state["appointment_request"] = request
    next_state.pop("_identified_service_slug", None)
    return next_state


def _proposal_repair_cards(
    context: ConversationContext,
    graph: Any,
    node_ids: list[str],
) -> list[dict[str, Any]]:
    if not node_ids:
        return []
    existing = {card.id for card in context.context_cards}
    requested = [node_id for node_id in node_ids if node_id not in existing]
    cards = context_cards_service.cards_for_ids(
        graph=graph,
        graph_version=context.graph_version,
        graph_checksum=context.graph_checksum,
        ids=requested,
    )
    return [card.model_dump(mode="json") for card in cards]


def _decide_appointment_proposal(
    context: ConversationContext,
    graph: Any,
    *,
    state: dict[str, Any],
    current_stage: str,
    proposal: ConversationProposal,
    observation: dict[str, Any],
) -> tuple[ConversationDecision, AgentResponse]:
    """Validate the model's conversation choice as a graph proof."""
    by_id = {node.id: node for node in graph.nodes}
    proposed_branch = by_id.get(proposal.branch_anchor_node_id)
    branch_id = (
        proposed_branch.id
        if proposed_branch is not None
        and proposed_branch.node_type in graph_conversation_contract.BRANCH_TYPES
        else context.active_branch_node_id
    )
    contract = graph_conversation_contract.compile_branch_contract(graph, branch_id)
    ledger = graph_conversation_contract.ledger_from_state(state, contract)
    if ledger.get("active_branch_node_id") != branch_id:
        declared = {field["key"] for field in contract.get("fields") or []}
        ledger["facts"] = {
            key: fact for key, fact in (ledger.get("facts") or {}).items()
            if key in declared
        }
        ledger["asked_question_node_ids"] = []
    ledger["active_branch_node_id"] = branch_id
    ledger["active_path_checksum"] = contract.get("branch_path_checksum")
    package_ids = {card.id for card in context.context_cards}
    repair_context_ids = {
        str(item) for item in observation.get("repair_context_node_ids") or []
        if item
    }
    proof = graph_conversation_contract.check_proposal(
        graph=graph,
        contract=contract,
        ledger=ledger,
        proposal=proposal.model_dump(mode="json"),
        package_node_ids=package_ids,
        repair_context_node_ids=repair_context_ids,
    )
    checked_ledger = proof["ledger"]
    checked_ledger["active_branch_node_id"] = branch_id
    checked_ledger["active_path_checksum"] = contract.get("branch_path_checksum")
    missing = list(proof["missing_fields"])
    next_state = _state_from_ledger(state, checked_ledger, contract, missing)
    repair_attempt = int(observation.get("repair_attempt") or 0)
    if proof["valid"]:
        next_question_id = proof.get("next_question_node_id")
        if next_question_id:
            asked = list(next_state.get("asked_question_node_ids") or [])
            if next_question_id not in asked:
                asked.append(next_question_id)
            next_state["asked_question_node_ids"] = asked
        complete = not missing
        handoff = bool(complete and contract.get("confirmation_required"))
        if handoff:
            next_state["conversation_state"] = "handoff"
        evidence = list(dict.fromkeys([
            *proposal.cited_node_ids,
            *(
                graph_conversation_contract.coordinate_for_node(graph, branch_id)["path_node_ids"]
                if branch_id else []
            ),
            *([next_question_id] if next_question_id else []),
        ]))
        intent = "complete_qualification" if complete else "collect_qualification"
        route = ConversationRoute.HUMAN if handoff else ConversationRoute.SDR
        decision = ConversationDecision(
            classifier="graph_proof_checker_v1",
            intent=intent,
            route=route,
            confidence=1.0,
            product_slug=proposed_branch.slug if proposed_branch else None,
            lead_stage=_appointment_stage(current_stage, "complete_booking_request" if complete else "request_service"),
            handoff_reason="graph_confirmation_required" if handoff else None,
            evidence_node_ids=evidence,
        )
        return decision, AgentResponse(
            reply_text=proposal.reply,
            role=route,
            evidence_node_ids=evidence,
            cart_state=next_state,
            handoff_required=handoff,
            proposal=proposal,
            proof=proof,
        )

    repair_cards = _proposal_repair_cards(
        context, graph, proof.get("repair_node_ids") or []
    )
    if proof.get("repair_required") and repair_attempt < 1 and repair_cards:
        decision = ConversationDecision(
            classifier="graph_proof_checker_v1",
            intent="repair_retrieval",
            route=ConversationRoute.SDR,
            confidence=0.0,
            lead_stage=current_stage,
            evidence_node_ids=[],
        )
        return decision, AgentResponse(
            reply_text=None,
            role=ConversationRoute.SDR,
            evidence_node_ids=[],
            cart_state=next_state,
            handoff_required=False,
            proposal=proposal,
            proof=proof,
            repair_context_cards=repair_cards,
        )

    question_id, question = graph_conversation_contract.fallback_question(
        contract, missing
    )
    if question_id:
        asked = list(next_state.get("asked_question_node_ids") or [])
        if question_id not in asked:
            asked.append(question_id)
        next_state["asked_question_node_ids"] = asked
    evidence = [item for item in [branch_id, question_id] if item]
    fallback_proof = {**proof, "repair_required": False, "fallback_used": True}
    decision = ConversationDecision(
        classifier="graph_proof_checker_v1",
        intent="technical_proof_fallback",
        route=ConversationRoute.SDR,
        confidence=0.0,
        product_slug=proposed_branch.slug if proposed_branch else None,
        lead_stage=current_stage,
        evidence_node_ids=evidence,
    )
    return decision, AgentResponse(
        reply_text=question,
        role=ConversationRoute.SDR,
        evidence_node_ids=evidence,
        cart_state=next_state,
        handoff_required=False,
        proposal=proposal,
        proof=fallback_proof,
        repair_context_cards=repair_cards,
    )


def _stage(current: str, intent: str, route: ConversationRoute) -> str:
    try:
        index = STAGES.index(current)
    except ValueError:
        index = 0
    target = {
        ConversationRoute.SDR: 1,
        ConversationRoute.CLOSER: 3,
        ConversationRoute.HUMAN: index,
    }[route]
    if intent in {"consult_product", "consult_price", "consult_category"}:
        target = max(target, 2)
    if intent in {"provide_name", "provide_address", "confirm_order"}:
        target = 4
    return STAGES[max(index, target)]


def _appointment_stage(current: str, intent: str) -> str:
    try:
        index = STAGES.index(current)
    except ValueError:
        index = 0
    if intent in {"greeting", "list_services"}:
        target = 1
    elif intent in {"consult_service", "consult_price"}:
        target = 2
    elif intent in {
        "request_quote",
        "request_booking",
        "request_service",
        "provide_vehicle",
        "provide_condition",
        "provide_date",
        "provide_time_window",
    }:
        target = 3
    elif intent == "complete_booking_request":
        target = 4
    else:
        target = index
    return STAGES[max(index, target)]


def _appointment_evidence(
    graph: Any,
    *,
    product_slug: str | None,
    include_all_products: bool = False,
) -> list[str]:
    by_id = {node.id: node for node in graph.nodes}
    evidence: list[str] = []
    product_node = next(
        (
            node
            for node in graph.nodes
            if node.node_type == "product" and node.slug == product_slug
        ),
        None,
    )
    product_nodes = (
        [product_node]
        if product_node
        else [node for node in graph.nodes if node.node_type == "product"]
        if include_all_products
        else []
    )
    for selected_product in product_nodes:
        evidence.extend(_node_path(graph, selected_product.id))
        evidence.append(selected_product.id)
        for node in graph.nodes:
            data = node.data or {}
            if (
                node.node_type == "faq"
                and data.get("source_node_id") == selected_product.id
            ):
                evidence.append(node.id)
    for node in graph.nodes:
        if node.node_type in {"persona", "brand", "briefing", "rule", "tone"}:
            status = str((node.data or {}).get("status") or "").lower()
            if status in {"approved", "validated", "active", "ativo"}:
                evidence.extend(_node_path(graph, node.id))
                evidence.append(node.id)
    return [node_id for node_id in dict.fromkeys(evidence) if node_id in by_id]


def _decide_appointment(
    context: ConversationContext,
    graph: Any,
    *,
    graph_changed: bool,
    current_stage: str,
    state: dict[str, Any],
    message: str,
) -> tuple[ConversationDecision, AgentResponse]:
    if graph_changed:
        # Appointment state contains customer facts and an unconfirmed request,
        # not a committed transaction. Re-evaluate it against the current graph
        # instead of erasing it or pausing the agent.
        state.pop("_previous_graph_version", None)
        state.pop("_previous_graph_checksum", None)

    faq = _approved_faq_match(graph, message)
    if faq:
        faq_data = faq.data or {}
        answer = str(faq_data.get("answer") or "").strip()
        if answer:
            evidence = list(dict.fromkeys([*_node_path(graph, faq.id), faq.id]))
            state["clarification_attempts"] = 0
            # A published FAQ may declare that its own answer is only the
            # opening line and a person has to take over (knowledge-gap FAQs).
            faq_handoff = bool(faq_data.get("handoff_required"))
            route = ConversationRoute.HUMAN if faq_handoff else ConversationRoute.SDR
            decision = ConversationDecision(
                classifier="deterministic_appointment_v1",
                intent="answer_faq",
                route=route,
                confidence=1.0,
                lead_stage=current_stage,
                handoff_reason="faq_requires_human" if faq_handoff else None,
                evidence_node_ids=evidence,
            )
            return decision, AgentResponse(
                reply_text=answer,
                role=route,
                evidence_node_ids=evidence,
                cart_state=state,
                handoff_required=faq_handoff,
            )

    catalog = catalog_from_graph(graph)
    identified_service_slug = state.pop("_identified_service_slug", None)
    result = DeterministicAppointment(
        catalog,
        policy=_appointment_policy(graph),
    ).handle(
        message,
        state=state,
        identified_service_slug=identified_service_slug,
    )
    evidence = _appointment_evidence(
        graph,
        product_slug=result.product.slug if result.product else None,
        include_all_products=result.intent == "list_services",
    )
    route = ConversationRoute.HUMAN if result.handoff else ConversationRoute.SDR
    confidence = 1.0 if result.intent != "ununderstood" else (0.0 if result.handoff else 0.7)
    if not evidence:
        route = ConversationRoute.HUMAN
        confidence = 0.0
        result.handoff = True
        result.handoff_reason = "missing_approved_evidence"
        result.reply = None
        result.state["conversation_state"] = "handoff"
    decision = ConversationDecision(
        classifier="deterministic_appointment_v1",
        intent=result.intent,
        route=route,
        confidence=confidence,
        product_slug=result.product.slug if result.product else None,
        lead_stage=_appointment_stage(current_stage, result.intent),
        handoff_reason=result.handoff_reason,
        evidence_node_ids=evidence,
    )
    response = AgentResponse(
        reply_text=result.reply,
        role=route,
        evidence_node_ids=evidence,
        cart_state=result.state,
        handoff_required=result.handoff,
    )
    return decision, response


def _ground_decision_in_context_cards(
    context: ConversationContext,
    graph: Any,
    decision: ConversationDecision,
    response: AgentResponse,
) -> tuple[ConversationDecision, AgentResponse]:
    """Keep declared evidence inside the exact package carried by the turn."""
    allowed = {card.id for card in context.context_cards}
    if not allowed:  # compatibility for older internal callers/tests
        return decision, response
    by_id = {node.id: node for node in graph.nodes}
    decisive_types = {"product", "offer", "faq", "copy", "rule"}
    unavailable_decisive = [
        node_id for node_id in decision.evidence_node_ids
        if node_id in by_id
        and by_id[node_id].node_type in decisive_types
        and node_id not in allowed
    ]
    filtered = [node_id for node_id in decision.evidence_node_ids if node_id in allowed]
    if unavailable_decisive:
        branch_scope = set(context.branch_node_ids or by_id)
        expandable = [
            node_id for node_id in unavailable_decisive if node_id in branch_scope
        ]
        cards = context_cards_service.cards_for_ids(
            graph=graph,
            graph_version=context.graph_version,
            graph_checksum=context.graph_checksum,
            ids=expandable,
        )
        if {card.id for card in cards} == set(unavailable_decisive):
            context.context_cards.extend(cards)
            for card in cards:
                node = by_id[card.id]
                context.rag_nodes.append({
                    "id": node.id,
                    "node_type": node.node_type,
                    "slug": node.slug,
                    "title": node.label,
                    "status": (node.data or {}).get("status"),
                    "source": (node.data or {}).get("source"),
                    "data": node.data or {},
                })
            return decision, response

        # A retrieval miss is technical, not a commercial handoff. Use the
        # exact graph-owned pending question when the contract exposes one.
        missing = list(response.cart_state.get("missing_fields") or [])
        question_id, question = graph_conversation_contract.fallback_question(
            context.graph_contract, missing
        )
        fallback_evidence = [
            item for item in [context.active_branch_node_id, question_id] if item
        ]
        return (
            decision.model_copy(update={
                "route": ConversationRoute.SDR,
                "confidence": 0.0,
                "handoff_reason": None,
                "evidence_node_ids": fallback_evidence,
            }),
            response.model_copy(update={
                "reply_text": question,
                "role": ConversationRoute.SDR,
                "handoff_required": False,
                "evidence_node_ids": fallback_evidence,
                "proof": {
                    "valid": False,
                    "errors": ["evidence_outside_branch_package"],
                    "repair_required": False,
                    "fallback_used": True,
                },
            }),
        )
    return (
        decision.model_copy(update={"evidence_node_ids": filtered}),
        response.model_copy(update={"evidence_node_ids": filtered}),
    )


def decide(
    context: ConversationContext,
    *,
    model_observation: dict[str, Any] | None = None,
    trace_id: str | None = None,
    lead_ref: int | None = None,
) -> tuple[ConversationDecision, AgentResponse]:
    _started_at = time.monotonic()
    decision, response = _decide_dispatch(context, model_observation=model_observation)
    observation = model_observation or {}
    is_repair = int(observation.get("repair_attempt") or 0) > 0
    token_usage = observation.get("token_usage") or (response.token_usage or {})
    emit_turn_event(
        agent_name="conversation.repair_call" if is_repair else "conversation.decide_llm_call",
        trace_id=trace_id,
        lead_ref=lead_ref,
        status="error" if response.reply_text is None and response.proof.get("errors") else "success",
        model_used=token_usage.get("model"),
        token_input=token_usage.get("prompt_tokens"),
        token_output=token_usage.get("completion_tokens"),
        latency_ms=token_usage.get("llm_latency_ms")
        or int((time.monotonic() - _started_at) * 1000),
        output_data={
            "reply_text": response.reply_text,
            "handoff_required": response.handoff_required,
            "route": decision.route.value,
            "intent": decision.intent,
        },
        metadata={
            "proof_valid": response.proof.get("valid"),
            "proof_errors": response.proof.get("errors"),
            "repair_required": response.proof.get("repair_required"),
        },
    )
    return decision, response


def _decide_dispatch(
    context: ConversationContext,
    *,
    model_observation: dict[str, Any] | None = None,
) -> tuple[ConversationDecision, AgentResponse]:
    if context.runtime_version == graph_agent_runtime_v3.RUNTIME_VERSION:
        return graph_agent_runtime_v3.decide(
            context, model_observation=model_observation
        )
    version, checksum, graph = _current_graph(context.persona_slug)
    if version != context.graph_version or checksum != context.graph_checksum:
        decision = ConversationDecision(
            intent="stale_graph",
            route=ConversationRoute.HUMAN,
            confidence=0,
            lead_stage=str(context.cart.get("_lead_stage") or "novo"),
            handoff_reason="graph_version_changed",
            evidence_node_ids=[],
        )
        response = AgentResponse(
            reply_text=None,
            role=ConversationRoute.HUMAN,
            evidence_node_ids=[],
            cart_state=context.cart,
            handoff_required=True,
        )
        return decision, response

    message = _message(context)
    normalized = _norm(message)
    business_model = _business_model(graph)
    proposal = _proposal_from_observation(model_observation)
    state = dict(context.cart)
    if proposal is None:
        state = _apply_model_fields(
            state,
            model_observation,
            business_model=business_model,
            graph=graph,
        )
    current_stage = str(state.pop("_lead_stage", "novo"))
    graph_changed = bool(state.pop("_graph_changed", False))
    if business_model == "appointment":
        if isinstance(model_observation, dict) and model_observation.get("contract_probe") is True:
            contract = context.graph_contract or graph_conversation_contract.compile_branch_contract(
                graph, context.active_branch_node_id
            )
            ledger = graph_conversation_contract.ledger_from_state(state, contract)
            pending = graph_conversation_contract.missing_fields(ledger, contract)
            probe_state = _state_from_ledger(state, ledger, contract, pending)
            return (
                ConversationDecision(
                    classifier="graph_contract_probe_v1",
                    intent="await_model_proposal",
                    route=ConversationRoute.SDR,
                    confidence=1.0,
                    lead_stage=current_stage,
                    evidence_node_ids=[],
                ),
                AgentResponse(
                    reply_text=None,
                    role=ConversationRoute.SDR,
                    evidence_node_ids=[],
                    cart_state=probe_state,
                    handoff_required=False,
                    proof={
                        "valid": True,
                        "mode": "contract_probe",
                        "missing_fields": pending,
                    },
                ),
            )
        if proposal is not None:
            if graph_changed:
                state.pop("_previous_graph_version", None)
                state.pop("_previous_graph_checksum", None)
            return _decide_appointment_proposal(
                context,
                graph,
                state=state,
                current_stage=current_stage,
                proposal=proposal,
                observation=model_observation or {},
            )
        appointment_decision, appointment_response = _decide_appointment(
            context,
            graph,
            graph_changed=graph_changed,
            current_stage=current_stage,
            state=state,
            message=message,
        )
        return _ground_decision_in_context_cards(
            context, graph, appointment_decision, appointment_response
        )
    catalog = catalog_from_graph(graph)
    engine = DeterministicSDR(catalog)
    product = catalog.find_product(message)
    if not product:
        last_slug = next(
            (
                item.get("product_slug")
                for item in reversed(state.get("items") or [])
                if item.get("product_slug")
            ),
            None,
        )
        product = next(
            (item for item in catalog.products if item.slug == last_slug),
            None,
        )

    explicit_human = any(
        value in normalized
        for value in (
            "atendimento humano",
            "falar com alguem",
            "falar com uma pessoa",
            "atendente",
            "humano",
        )
    )
    exceptional = any(
        value in normalized
        for value in (
            "reclamacao",
            "reclamar",
            "problema",
            "suporte",
            "negociar",
            "desconto especial",
        )
    )
    if graph_changed:
        intent = "stale_graph"
        route = ConversationRoute.HUMAN
        confidence = 0.0
        handoff_reason = "graph_version_changed"
        state["conversation_state"] = "handoff"
        result = {
            "reply": "Vou encaminhar para o atendimento revisar sua solicitaÃ§Ã£o com as informaÃ§Ãµes mais atuais.",
            "state": state,
            "handoff": True,
        }
    elif explicit_human or exceptional:
        intent = "request_human" if explicit_human else "exceptional_support"
        route = ConversationRoute.HUMAN
        confidence = 1.0
        state["conversation_state"] = "handoff"
        result = {
            "reply": "Vou encaminhar sua conversa para o atendimento humano.",
            "state": state,
            "handoff": True,
        }
        handoff_reason = intent
    else:
        result = engine.handle(
            message,
            state=state,
            history=context.messages,
        )
        state = result["state"]
        intent = str(result.get("intent") or "ununderstood")
        route = (
            ConversationRoute.HUMAN
            if result.get("handoff")
            else (
                ConversationRoute.CLOSER
                if intent in CLOSER_INTENTS
                else ConversationRoute.SDR
            )
        )
        confidence = 1.0 if intent != "ununderstood" else 0.7
        handoff_reason = None
        if intent == "ununderstood":
            attempts = int(state.get("clarification_attempts") or 0) + 1
            state["clarification_attempts"] = attempts
            if attempts > 1:
                route = ConversationRoute.HUMAN
                confidence = 0.0
                handoff_reason = "missing_approved_evidence"
                state["conversation_state"] = "handoff"
                result["reply"] = (
                    "NÃ£o encontrei evidÃªncia aprovada para responder. "
                    "Vou encaminhar ao atendimento humano."
                )
                result["handoff"] = True
        else:
            state["clarification_attempts"] = 0
        if intent == "confirm_order":
            route = ConversationRoute.HUMAN
            handoff_reason = "confirmed_pending_human"

    if not product:
        resolved_slug = next(
            (
                item.get("product_slug")
                for item in reversed(state.get("items") or [])
                if item.get("product_slug")
            ),
            None,
        )
        product = next(
            (item for item in catalog.products if item.slug == resolved_slug),
            None,
        )

    evidence_node_ids: list[str] = []
    if product:
        product_id = next(
            (
                node.id
                for node in graph.nodes
                if node.node_type == "product" and node.slug == product.slug
            ),
            None,
        )
        if product_id:
            evidence_node_ids.extend(_node_path(graph, product_id))
    for item in context.rag_nodes:
        if item.get("node_type") in {"persona", "brand", "tone", "rule", "briefing"}:
            evidence_node_ids.append(str(item["id"]))
    evidence_node_ids = list(dict.fromkeys(evidence_node_ids))
    if not evidence_node_ids:
        route = ConversationRoute.HUMAN
        confidence = 0.0
        handoff_reason = "missing_approved_evidence"
        result["handoff"] = True
        result["reply"] = None

    decision = ConversationDecision(
        intent=intent,
        route=route,
        confidence=confidence,
        cart_action=INTENT_ACTIONS.get(intent, CartAction.NONE),
        product_slug=product.slug if product else None,
        quantity=_quantity(message) if product else None,
        lead_stage=_stage(current_stage, intent, route),
        handoff_reason=handoff_reason,
        evidence_node_ids=evidence_node_ids,
    )
    response = AgentResponse(
        reply_text=result.get("reply"),
        role=route,
        evidence_node_ids=evidence_node_ids,
        cart_state=state,
        handoff_required=route == ConversationRoute.HUMAN,
    )
    return _ground_decision_in_context_cards(context, graph, decision, response)


_UNSAFE_CONFIRMATION_VERB = re.compile(
    r"confirmad[oa]|confirmo\b|fechad[oa]|reservad[oa]|"
    r"agendad[oa]\s+para|marcad[oa]\s+para",
    re.IGNORECASE,
)
_UNSAFE_MONEY_TOKEN = re.compile(r"r\$\s?\d", re.IGNORECASE)
_UNSAFE_SCHEDULE_TOKEN = re.compile(
    r"\b\d{1,2}[:h]\d{0,2}\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b"
)


def _reply_confirms_price_or_schedule(text: str | None) -> bool:
    """A reply must never both use confirmation language and state a price
    or a date/time in the same breath â€” only a human may finalize those.

    Informational replies (FAQ prices, durations) are unaffected because they
    never pair a confirmation verb with the figure; this only catches a
    reply that actually claims something is booked or settled.
    """
    if not text:
        return False
    if not _UNSAFE_CONFIRMATION_VERB.search(text):
        return False
    return bool(_UNSAFE_MONEY_TOKEN.search(text) or _UNSAFE_SCHEDULE_TOKEN.search(text))


def _reply_states_a_price(
    text: str | None,
    appointment_policy: dict[str, Any],
    *,
    cited_nodes: Any = (),
) -> bool:
    """A price statement by a persona that reserves prices for a human.

    Unlike _reply_confirms_price_or_schedule (which only catches a reply that
    *confirms* something), this fires on any monetary figure â€” value, range,
    approximation, installment â€” but only for a persona whose published graph
    says ``appointment_policy.price_disclosure == "human_only"``. Personas
    without that declaration are untouched.

    Carve-out: the graph may publish payment terms that legitimately contain
    money (a deposit threshold, installment plans). Those are allowed when a
    node cited as evidence for this reply publishes a ``payment_policy``
    claim. The regex and the carve-out live in graph_conversation_contract so
    the runtime guard and check_proposal() enforce one single rule.
    """
    return graph_conversation_contract.reply_discloses_blocked_price(
        text, appointment_policy, cited_nodes
    )


def _context_appointment_policy(context: ConversationContext) -> dict[str, Any]:
    """Published appointment_policy, read from the turn's own rag_nodes.

    commit() never holds the full GraphJson â€” only the flattened node dicts â€”
    so this mirrors _commercial_note_fields() instead of _appointment_policy().
    """
    persona_node = next(
        (item for item in context.rag_nodes if item.get("node_type") == "persona"),
        None,
    )
    data = (persona_node or {}).get("data") or {}
    policy = data.get("appointment_policy")
    return policy if isinstance(policy, dict) else {}


def _cited_context_nodes(
    context: ConversationContext, decision: ConversationDecision
) -> list[dict[str, Any]]:
    """The decisive graph nodes behind this reply, as carried by the turn.

    decision.evidence_node_ids is exactly what _knowledge_context_envelope
    records as ``decisive_node_ids``; resolving it against context.rag_nodes
    reuses the plumbing every other commit-time check already relies on.
    """
    cited = {str(node_id) for node_id in decision.evidence_node_ids or []}
    return [item for item in context.rag_nodes if str(item.get("id")) in cited]


def _knowledge_context_envelope(
    context: ConversationContext,
    decisive_node_ids: list[str],
) -> dict[str, Any]:
    """Serialize the exact immutable card package used for this response."""
    return {
        "contract_version": 1,
        "mode": "exact",
        "graph_version": context.graph_version,
        "graph_checksum": context.graph_checksum,
        "decisive_node_ids": list(dict.fromkeys(decisive_node_ids)),
        "active_branch_node_id": context.active_branch_node_id,
        "active_path_checksum": context.active_path_checksum,
        "graph_contract": context.graph_contract,
        "cards": [card.model_dump(mode="json") for card in context.context_cards],
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


def _structured_commercial_note_projection(
    *,
    document: dict[str, Any],
    cart_state: dict[str, Any],
) -> dict[str, Any]:
    active = list(dict.fromkeys(
        str(value) for value in cart_state.get("active_branch_node_ids") or [] if value
    ))
    focus = str(cart_state.get("active_branch_node_id") or "") or None
    grouped = cart_state.get("facts_by_key") or {}
    contracts = document.get("branch_contracts") or {}

    def fact_for(key: str, owner: str) -> dict[str, Any] | None:
        return next(
            (
                fact for fact in grouped.get(key, [])
                if str(fact.get("owner_node_id") or "") == owner
            ),
            None,
        )

    declarations: dict[tuple[str, str], set[str]] = {}
    for anchor in active:
        for field in (contracts.get(anchor) or {}).get("fields") or []:
            identity = (str(field.get("key") or ""), str(field.get("owner_node_id") or ""))
            declarations.setdefault(identity, set()).add(anchor)

    common: dict[str, Any] = {}
    services: dict[str, Any] = {}
    node_by_id = document.get("node_by_id") or {}
    for anchor in active:
        service_facts: dict[str, Any] = {}
        for field in (contracts.get(anchor) or {}).get("fields") or []:
            key = str(field.get("key") or "")
            owner = str(field.get("owner_node_id") or "")
            fact = fact_for(key, owner)
            if not fact or fact.get("status") not in {"known", "unknown", "declined"}:
                continue
            value = fact.get("value") if fact.get("status") == "known" else "desconhecido"
            if len(declarations.get((key, owner), set())) == len(active) and owner not in active:
                common[key] = value
            else:
                service_facts[key] = value
        node = node_by_id.get(anchor) or {}
        services[anchor] = {
            "slug": node.get("slug"),
            "title": node.get("title"),
            "facts": service_facts,
        }
    return {
        "version": 1,
        "focused_service_id": focus,
        "active_service_ids": active,
        "common_facts": common,
        "services": services,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _is_canonical_validation_lead(lead: dict[str, Any]) -> bool:
    """Require all canonical markers before bypassing real transport gates."""
    metadata = lead.get("metadata") or {}
    validation = metadata.get("validation") or {}
    return bool(
        isinstance(validation, dict)
        and validation.get("is_validation") is True
        and str(validation.get("session_id") or "").strip()
        and str(lead.get("lead_id") or "").startswith("validator_")
    )


def _handoff_branch_reset_facts(
    *,
    handoff_required: bool,
    handoff_level: str,
    active_branch: str | None,
    branch_contract: dict[str, Any],
    branch_facts: dict[str, Any],
    correlation_id: str,
) -> list[dict[str, Any]]:
    """Handoff is a journey transition, never a fact-destruction event.

    Branch-specific facts are invalidated only by an explicit incompatible
    branch switch during policy reconciliation. Keeping them here preserves
    the qualified request for human takeover and post-qualification support.
    """
    return []


def _ensure_reply_text_or_log(
    response: AgentResponse, *, lead_ref: Any, correlation_id: str,
) -> AgentResponse:
    """Last-resort floor against total silence.

    commit() had no guard at all on an empty reply_text -- it committed the
    graph turn/ledger/proof and returned {"ok": True, "message_id": None,
    ...} with no outbound message and no alert. The doubt/claim repair
    branch that caused this live (2026-08-18) is now fixed at the source
    (graph_agent_runtime_v3 resolves it immediately instead of returning
    reply_text=None), but this stays as a generic safety net for any other
    path (e.g. the genuinely-needs-fetching selected_branch_requires_-
    phase_b repair, which still depends on a real round trip) that might
    end up here with nothing to say. Never invents text that didn't come
    from the graph -- proof["text"] is graph-approved content (the doubt
    resolution's answer) when present, nothing else.
    """
    if response.reply_text:
        return response
    if response.proof.get("text"):
        return response.model_copy(update={"reply_text": response.proof["text"]})
    logger.error(
        "conversation turn committed with no reply_text lead_ref=%s "
        "correlation_id=%s proof_mode=%s",
        lead_ref, correlation_id, response.proof.get("mode"),
    )
    return response


AgentRole = Literal["sdr", "closer", "cs"]

# AGENT_ROADMAP.md item 7a: stage bands over the happy-path completion score.
# Not yet configurable per persona (personas.config.stage_thresholds) -- that
# read would add a DB round trip to every commit(); deferred until this
# module-level default has run in production without surprise.
DEFAULT_STAGE_THRESHOLDS: dict[str, float] = {
    "sdr": 0.50,
    "conversao": 0.75,
    "venda": 1.00,
    "pos_venda": 1.01,
}


def _qualification_score(required_field_count: int, resolved_required_count: int) -> float:
    if required_field_count <= 0:
        return 0.0
    return resolved_required_count / required_field_count


def resolve_agent_role(
    score: float, thresholds: dict[str, float] | None = None,
) -> AgentRole:
    """AGENT_ROADMAP.md item 7b -- label only, nothing reads this yet to pick
    a prompt or change the reply. SDR below the "sdr" band, Closer through
    "venda", CS from "pos_venda" up."""
    bands = thresholds or DEFAULT_STAGE_THRESHOLDS
    if score >= bands.get("pos_venda", 1.01):
        return "cs"
    if score >= bands.get("sdr", 0.50):
        return "closer"
    return "sdr"


def commit(
    *,
    lead_ref: int,
    context: ConversationContext,
    decision: ConversationDecision,
    response: AgentResponse,
    correlation_id: str,
    phone_number_id: str | None,
    channel_binding_id: str,
    inbound_buffer_id: str | None = None,
    expected_decision_owner: str | None = None,
    n8n_execution_id: str | None = None,
) -> dict[str, Any]:
    if _reply_confirms_price_or_schedule(response.reply_text):
        if response.proposal is not None:
            authorized = bool(context.graph_contract.get("confirmation_required"))
            safe_text = context.graph_contract.get("confirmation_text")
            decision = decision.model_copy(
                update={
                    "route": ConversationRoute.HUMAN if authorized else ConversationRoute.SDR,
                    "handoff_reason": (
                        decision.handoff_reason or "unsafe_reply_blocked_by_graph_proof"
                        if authorized else None
                    ),
                }
            )
            response = response.model_copy(
                update={
                    "reply_text": safe_text,
                    "role": ConversationRoute.HUMAN if authorized else ConversationRoute.SDR,
                    "handoff_required": authorized,
                    "proof": {
                        **response.proof,
                        "unsafe_confirmation_blocked": True,
                    },
                }
            )
        else:
            decision = decision.model_copy(
                update={
                    "route": ConversationRoute.HUMAN,
                    "handoff_reason": decision.handoff_reason
                    or "unsafe_reply_blocked_price_or_schedule_confirmation",
                }
            )
            response = response.model_copy(
                update={
                    "reply_text": (
                        "Vou encaminhar sua conversa para a equipe confirmar "
                        "os detalhes."
                    ),
                    "role": ConversationRoute.HUMAN,
                    "handoff_required": True,
                }
            )

    appointment_policy = _context_appointment_policy(context)
    if _reply_states_a_price(
        response.reply_text,
        appointment_policy,
        cited_nodes=_cited_context_nodes(context, decision),
    ):
        texts = appointment_policy.get("texts") or {}
        safe_text = str(texts.get("preco_humano") or "").strip() or (
            "O valor Ã© definido por uma pessoa da equipe. Vou encaminhar sua "
            "conversa para o atendimento."
        )
        decision = decision.model_copy(
            update={
                "route": ConversationRoute.HUMAN,
                "handoff_reason": "price_disclosure_requires_human",
            }
        )
        response = response.model_copy(
            update={
                "reply_text": safe_text,
                "role": ConversationRoute.HUMAN,
                "handoff_required": True,
                "proof": {
                    **response.proof,
                    "price_disclosure_blocked": True,
                },
            }
        )

    extracted_fields = _resolve_identified_service(
        dict(response.extracted_fields),
        response.identified_service_slug,
        context.available_services,
    )
    if extracted_fields:
        response = response.model_copy(
            update={
                "cart_state": _merge_extracted_fields(
                    response.cart_state, extracted_fields
                )
            }
        )

    if context.runtime_version != graph_agent_runtime_v3.RUNTIME_VERSION:
        response = response.model_copy(
            update={
                "reply_text": _ensure_trailing_question(
                    response.reply_text, response.cart_state, context
                )
            }
        )

    lead = supabase_client.get_lead_by_ref(lead_ref) or {}
    if not lead:
        raise LookupError("lead not found")
    # A commit cannot choose transport.  The persisted lead binding is the
    # authority and also protects n8n from committing to another persona.
    persisted_binding_id = lead.get("channel_binding_id")
    if not persisted_binding_id:
        raise RuntimeError("lead has no explicit channel binding")
    if channel_binding_id != persisted_binding_id:
        raise PermissionError("channel binding does not match lead")
    binding = supabase_client.get_workflow_binding_by_id(persisted_binding_id) or {}
    if not binding or binding.get("persona_id") != lead.get("persona_id"):
        raise PermissionError("channel binding does not match lead persona")
    if not binding.get("active"):
        raise RuntimeError("channel binding is inactive")
    binding_metadata = binding.get("metadata") or {}
    is_validation_lead = _is_canonical_validation_lead(lead)
    if (
        expected_decision_owner
        and binding_metadata.get("decision_owner") != expected_decision_owner
    ):
        raise RuntimeError(
            "binding decision owner does not authorize this commit path"
        )
    if (
        binding_metadata.get("safety_paused")
        or binding.get("connection_status") == "safety_paused"
    ) and not is_validation_lead:
        raise RuntimeError("channel binding is safety paused")
    channel_binding_id = persisted_binding_id
    phone_number_id = binding.get("whatsapp_phone_number_id")
    persona = supabase_client.get_persona(context.persona_slug) or {}
    if not persona or persona.get("id") != lead.get("persona_id"):
        raise PermissionError("conversation context does not match lead persona")

    if expected_decision_owner and not inbound_buffer_id:
        raise RuntimeError("inbound buffer id is required for a decision commit")
    knowledge_context = _knowledge_context_envelope(
        context,
        decision.evidence_node_ids,
    )
    commit_claim = None
    if inbound_buffer_id:
        commit_claim = supabase_client.claim_conversation_commit(
            inbound_buffer_id=inbound_buffer_id,
            binding_id=channel_binding_id,
            lead_ref=lead_ref,
            correlation_id=correlation_id,
        )
        claim_state = commit_claim.get("state")
        if claim_state == "completed":
            stored_result = commit_claim.get("result") or {}
            if not isinstance(stored_result, dict):
                raise RuntimeError("stored conversation commit result is invalid")
            return {
                **stored_result,
                "ok": True,
                "deduplicated": True,
            }
        if claim_state == "processing":
            supabase_client.record_whatsapp_safety_violation(
                binding_id=channel_binding_id,
                lead_ref=lead_ref,
                violation_key=f"conversation_commit_reentry:{inbound_buffer_id}",
                reason="compromised conversation commit re-execution",
            )
            raise RuntimeError("conversation commit is already processing")
        if claim_state != "claimed":
            raise RuntimeError("conversation commit claim returned an invalid state")

    message_id = f"ai:{correlation_id}"
    existing_outbound = (
        supabase_client.get_whatsapp_buffer_by_idempotency(message_id)
        if response.reply_text
        else None
    )
    if existing_outbound:
        if context.runtime_version == graph_agent_runtime_v3.RUNTIME_VERSION:
            # Atomic v3 commits create the proof and inert outbound in one DB
            # transaction. Reaching this branch means a second decision was
            # attempted against an envelope created outside that boundary.
            # Never manufacture a proof after seeing an outbox row.
            supabase_client.record_whatsapp_safety_violation(
                binding_id=channel_binding_id,
                lead_ref=lead_ref,
                violation_key=f"v3_preexisting_outbound:{inbound_buffer_id}",
                reason="v3 commit encountered an outbound outside its atomic transaction",
            )
            raise ConversationCommitFailed(
                failed_step="preexisting_outbound_guard",
                inbound_id=inbound_buffer_id,
                correlation_id=correlation_id,
                cause=RuntimeError("preexisting v3 outbound requires proof-based recovery"),
            )
        graph_turn = None
        result = {
            "ok": True,
            "message_id": message_id,
            "outbound_buffer_id": existing_outbound.get("id"),
            "deduplicated": True,
            "handoff": response.handoff_required,
            "ai_paused": response.handoff_required,
            "route": decision.route.value,
            "role": response.role.value,
            "intent": decision.intent,
            "reply_text": response.reply_text,
            "classifier": decision.classifier,
            "evidence_node_ids": decision.evidence_node_ids,
            "graph_version": context.graph_version,
            "graph_checksum": context.graph_checksum,
            "knowledge_context": knowledge_context,
            "proof": response.proof,
            "qualification": (lead.get("metadata") or {}).get("qualification") or {},
            "stage": lead.get("stage") or decision.lead_stage,
            "graph_turn": graph_turn,
        }
        if inbound_buffer_id:
            supabase_client.complete_conversation_commit(
                inbound_buffer_id=inbound_buffer_id,
                binding_id=channel_binding_id,
                lead_ref=lead_ref,
                correlation_id=correlation_id,
                result_payload=result,
            )
        return result
    previous_metadata = dict(lead.get("metadata") or {})
    v3_document: dict[str, Any] = {}
    if context.runtime_version == graph_agent_runtime_v3.RUNTIME_VERSION:
        facts = response.cart_state.get("facts") or {}
        facts_by_key = response.cart_state.get("facts_by_key") or {
            str(key): [fact]
            for key, fact in facts.items()
            if isinstance(fact, dict)
        }
        resolved = sorted(
            key for key, rows in facts_by_key.items()
            if any(
                isinstance(fact, dict)
                and fact.get("status") in {"known", "unknown", "declined"}
                for fact in rows or []
            )
        )
        missing = list(response.proof.get("missing_fields") or [])
        required_total = int(response.proof.get("required_field_count") or 0)
        resolved_required = max(0, required_total - len(missing))
        qualification_score = _qualification_score(required_total, resolved_required)
        qualified_stage = decision.lead_stage
        qualification = {
            "version": graph_agent_runtime_v3.RUNTIME_VERSION,
            "complete": bool(response.proof.get("qualification_complete", not missing)),
            "collection_complete": bool(response.proof.get("collection_complete", not missing)),
            "resolved_fields": resolved,
            "missing_fields": missing,
            "required_field_count": required_total,
            "resolved_required_count": resolved_required,
            "score": qualification_score,
            "suggested_role": resolve_agent_role(qualification_score),
            "source_evidence_node_ids": list(decision.evidence_node_ids),
            "stage": qualified_stage,
            "stage_source": "graph_contract",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    else:
        qualification, qualified_stage = lead_qualification.calculate(
            previous=previous_metadata.get("qualification"),
            business_model=str(
                response.cart_state.get("business_model") or "sales"
            ).lower(),
            intent=decision.intent,
            state={
                **response.cart_state,
                "_decision_product_slug": decision.product_slug,
            },
            current_stage=lead.get("stage") or decision.lead_stage,
            evidence_node_ids=decision.evidence_node_ids,
            update_stage=True,
        )
    metadata = {
        **previous_metadata,
        "conversation_state": response.cart_state,
        "qualification": qualification,
        "conversation_runtime": {
            "agent_slug": context.agent_slug,
            "graph_version": context.graph_version,
            "graph_checksum": context.graph_checksum,
            "last_intent": decision.intent,
            "last_route": decision.route.value,
            "evidence_node_ids": decision.evidence_node_ids,
            "knowledge_context": knowledge_context,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    # pending_reconfirmation (set by agents_service.resume_lead) only
    # governs the one turn right after a manual resume -- consumed by
    # build_context/the reply prompt this turn, then cleared here so it
    # doesn't linger forever.
    metadata.pop("pending_reconfirmation", None)
    if response.handoff_required and decision.handoff_reason:
        # Computed every handoff-authorizing turn but never persisted
        # before -- the dashboard had no way to show *why* a lead needed
        # attention (complaint vs. explicit human request vs. qualified
        # sale), only that it did.
        metadata["handoff_reason"] = decision.handoff_reason
    if (
        context.runtime_version != graph_agent_runtime_v3.RUNTIME_VERSION
        and response.cart_state.get("business_model") != "appointment"
    ):
        # Backward-compatible mirror for Baita conversations already consumed
        # by the legacy deterministic/n8n flow.
        metadata["vitoria_state"] = response.cart_state
    if context.runtime_version == graph_agent_runtime_v3.RUNTIME_VERSION:
        # v3's own fact ledger (branch-scoped, revisioned) is the only
        # source of truth here â€” no separate, hand-typed field list.
        # Confirmed live 2026-08-06: `commercial_note_fields` (a list
        # authored once on the persona node) had drifted from the real
        # per-branch field declarations â€” it named fields never actually
        # collected and omitted others that were. Mirroring every known
        # fact that belongs to the active branch, straight from `facts`,
        # means there is nothing left to keep in sync by hand. Facts are
        # further scoped to owners the active branch actually declares, so
        # a value from an abandoned branch/session never leaks into the
        # note (same class of bug as the missing_fields owner check in
        # graph_proof_checker_v3.pending_fields).
        #
        # Confirmed live 2026-08-08: comparing owner_node_id to
        # active_branch directly (instead of the branch's own declared
        # field owners) broke the moment shared fields (nome_cliente,
        # modelo_veiculo, vehicle_year, etc.) were fixed earlier that same
        # day to share one owner -- the persona node -- across every
        # branch, instead of each branch owning its own copy. Those
        # fields' owner_node_id is now the persona, never the branch
        # itself, so the old check silently excluded every one of them
        # from the note. valid_owners always includes active_branch
        # itself as a baseline (so this degrades to the old behavior if
        # the publication lookup below comes back empty for any reason),
        # plus whatever owner_node_id values the branch's own published
        # contract actually declares for its fields.
        facts = response.cart_state.get("facts") or {}
        active_branch = response.cart_state.get("active_branch_node_id")
        valid_owners = {active_branch} if active_branch else set()
        if active_branch:
            publication = supabase_client.get_active_graph_publication(
                str(persona.get("id") or "")
            ) or {}
            v3_document = publication.get("document_json") or {}
            contract = (v3_document.get("branch_contracts") or {}).get(active_branch) or {}
            valid_owners |= {
                str(field.get("owner_node_id"))
                for field in contract.get("fields") or []
                if field.get("owner_node_id")
            }
        commercial_note = {
            key: fact["value"]
            for key, fact in facts.items()
            if isinstance(fact, dict)
            and fact.get("status") == "known"
            and fact.get("value")
            and (not active_branch or fact.get("owner_node_id") in valid_owners or fact.get("owner_node_id") is None)
        }
        # Legacy debris that a pre-v3 lead may still carry in cart_state;
        # v3 never reads or writes it, so it should not persist forever.
        response.cart_state.pop("appointment_request", None)
    else:
        # No hardcoded field names or business_model gate: each persona
        # declares its own commercial_note_fields in its graph's persona
        # node data (any business model â€” appointment or otherwise).
        note_fields = _commercial_note_fields(context)
        appointment_request = dict(
            response.cart_state.get("appointment_request") or {}
        )
        commercial_note = dict(metadata.get("commercial_note") or {})
        # Values are resolved from whatever shape that persona's engine
        # produces (appointment_request first, since that's where
        # DeterministicAppointment nests its per-field state; cart_state
        # itself as a fallback for other engines like Baita's cart/order-
        # based DeterministicSDR).
        note_pool = {**response.cart_state, **appointment_request}
        for field in note_fields:
            if note_pool.get(field):
                commercial_note[field] = note_pool[field]
    if commercial_note:
        commercial_note["updated_at"] = datetime.now(timezone.utc).isoformat()
        metadata["commercial_note"] = commercial_note
    elif "commercial_note" in metadata:
        metadata.pop("commercial_note", None)
    if context.runtime_version == graph_agent_runtime_v3.RUNTIME_VERSION:
        projection = response.cart_state.get("commercial_note_projection")
        if isinstance(projection, dict) and projection.get("version"):
            metadata["commercial_note_projection"] = projection
        elif v3_document:
            metadata["commercial_note_projection"] = _structured_commercial_note_projection(
                document=v3_document,
                cart_state=response.cart_state,
            )
    lead_update = {"metadata": metadata, "stage": qualified_stage}
    if context.runtime_version == graph_agent_runtime_v3.RUNTIME_VERSION:
        facts = response.cart_state.get("facts") or {}
        customer_name = (facts.get("nome_cliente") or {}).get("value")
        # No hardcoded "servico": the selector field key is whatever the
        # compiled contract flags as branch_selection_field (product,
        # service, catalog item -- all the same mechanism). facts (above) is
        # scoped to a single currently-focused branch, so a lead with more
        # than one confirmed offering active at once (a multi-item pedido)
        # would collapse back to just one of them here -- aggregate over
        # every active branch's own fact instead, same source
        # commercial_note_projection already uses.
        selection_key = graph_agent_runtime_v3.branch_selection_field_key(v3_document)
        active_branch_ids = list(dict.fromkeys([
            *([active_branch] if active_branch else []),
            *(response.cart_state.get("active_branch_node_ids") or []),
        ]))
        offering_titles = (
            graph_agent_runtime_v3.active_offering_titles(
                v3_document, active_branch_ids, response.cart_state.get("facts_by_key") or {},
            )
            if v3_document and active_branch_ids else []
        )
        service_interest = (
            ", ".join(offering_titles) if offering_titles
            else (facts.get(selection_key) or {}).get("value")
        )

        def _is_known(fact: Any) -> bool:
            return (
                bool(fact)
                and str(fact.get("status")) == "known"
                and fact.get("value") not in (None, "")
            )

        # A handoff-authorizing turn only silences the AI outright (level=
        # "full") once the lead's minimum registration -- name + service --
        # is known, this turn or from an earlier session (facts persist for
        # the lead's whole lifetime, see conversation_ledgers). Otherwise
        # it's "partial": the lead is flagged for eventual human attention,
        # but lead_buffer keeps flowing so the SDR can keep collecting what's
        # missing instead of going silent on a customer who, say, wants to
        # complain but hasn't given their name yet.
        handoff_level = (
            "full"
            if (_is_known(facts.get("nome_cliente")) and _is_known(facts.get(selection_key)))
            else "partial"
        )
    else:
        customer_name = (
            appointment_request.get("nome_cliente")
            if response.cart_state.get("business_model") == "appointment"
            else response.cart_state.get("customer_name")
        )
        service_interest = appointment_request.get("servico") or decision.product_slug
        # The legacy engines' own handoff triggers (e.g. intent ==
        # "confirm_order") already only fire once qualification is fully
        # complete, so this always resolves to "full" -- unchanged behavior.
        handoff_level = "full"
    if customer_name:
        lead_update["nome"] = str(customer_name).strip()
    if service_interest:
        lead_update["interesse_produto"] = str(service_interest).strip()
    # v3's authoritative state is committed atomically below before this lead
    # projection is updated. Legacy engines keep their established ordering.
    if context.runtime_version != graph_agent_runtime_v3.RUNTIME_VERSION:
        if response.handoff_required:
            supabase_client.handoff_whatsapp_lead_state(
                lead_ref, metadata=metadata, stage=qualified_stage, level=handoff_level,
            )
            if customer_name or service_interest:
                supabase_client.update_lead(lead_ref, lead_update)
        else:
            supabase_client.update_lead(lead_ref, lead_update)

    supabase_client.insert_agent_log(
        {
            "lead_id": str(lead_ref),
            "persona_id": persona.get("id"),
            "agent_type": context.agent_slug,
            "action": "[INFO] conversation decision committed",
            "decision": decision.model_dump_json(),
            "metadata": {
                "level": "INFO",
                "component": "conversation_runtime",
                "correlation_id": correlation_id,
                "graph_version": context.graph_version,
                "graph_checksum": context.graph_checksum,
                "route": decision.route.value,
                "role": response.role.value,
                "intent": decision.intent,
                "evidence_node_ids": decision.evidence_node_ids,
                "knowledge_context": knowledge_context,
            },
            "input": {"messages": context.messages[-20:]},
            "output": {
                "decision": decision.model_dump(mode="json"),
                "response": response.model_dump(mode="json"),
                "qualification": qualification,
            },
        }
    )

    response = _ensure_reply_text_or_log(
        response, lead_ref=lead_ref, correlation_id=correlation_id,
    )

    buffer = None
    prepared_outbound: dict[str, Any] | None = None
    if response.reply_text and is_validation_lead:
        # Confirmed live 2026-08-08: whatsapp_outbox.enqueue_outbound's
        # _recipient_for_lead 409s for a validation lead (no real WhatsApp
        # recipient by design) -- correct, since giving it a fake-but-valid-
        # shaped phone to dodge that check would risk a real worker send.
        # But routing the reply through nothing at all meant the bot's turn
        # never reached the `messages` table either, so the validation-
        # leads conversation view showed only the customer's side.
        # enqueue_whatsapp_envelope is the same primitive enqueue_outbound
        # itself calls; call it directly with a buffer status outside
        # claim_whatsapp_buffer's claimable set ('buffered', 'retry',
        # 'pending_send' -- migration 065) so the row is inert to every
        # dispatch worker while the message row still gets written.
        prepared_outbound = {
            "buffer": {
                "persona_id": lead.get("persona_id"),
                "lead_ref": lead_ref,
                "channel_binding_id": channel_binding_id,
                "direction": "outbound",
                "status": "awaiting_proof",
                "batch_key": f"{lead.get('persona_id')}:{lead_ref}",
                "idempotency_key": message_id,
                "correlation_id": message_id,
                "payload": {
                    "text": response.reply_text,
                    "sender_type": "agent",
                    "validation": True,
                },
            },
            "message": {
                "lead_id": lead_ref,
                "role": "assistant",
                "content": response.reply_text,
                "direction": "outbound",
                "status": "sent",
                "channel": "whatsapp",
                "sender_id": message_id,
                "channel_binding_id": channel_binding_id,
                "correlation_id": message_id,
                "metadata": {
                    "agent_slug": context.agent_slug,
                    "role": response.role.value,
                    "intent": decision.intent,
                    "graph_version": context.graph_version,
                    "graph_checksum": context.graph_checksum,
                    "evidence_node_ids": decision.evidence_node_ids,
                    "knowledge_context": knowledge_context,
                    "token_usage": response.token_usage,
                    "trace_id": inbound_buffer_id,
                    "n8n_execution_id": n8n_execution_id,
                    "validation": True,
                },
            },
        }
        if context.runtime_version != graph_agent_runtime_v3.RUNTIME_VERSION:
            envelope = supabase_client.enqueue_whatsapp_envelope(
                buffer={**prepared_outbound["buffer"], "status": "sent"},
                message=prepared_outbound["message"],
            )
            buffer = {"id": envelope.get("buffer_id"), "status": envelope.get("status")}
            prepared_outbound = None
    elif response.reply_text:
        # message_id ("ai:<inbound correlation_id>") is unique to this
        # outbound leg â€” reusing the inbound correlation_id here made the
        # outbound message row share it with the inbound row that triggered
        # it, so complete_whatsapp_outbound_result's `WHERE correlation_id =
        # ...` (scoped only by binding, not direction) matched both rows and
        # tried to force the same external_message_id onto both, tripping
        # idx_messages_channel_external_unique. Confirmed live 2026-08-01.
        outbound_metadata = {
                "agent_slug": context.agent_slug,
                "role": response.role.value,
                "intent": decision.intent,
                "graph_version": context.graph_version,
                "graph_checksum": context.graph_checksum,
                "evidence_node_ids": decision.evidence_node_ids,
                "knowledge_context": knowledge_context,
                "token_usage": response.token_usage,
                "trace_id": inbound_buffer_id,
                "n8n_execution_id": n8n_execution_id,
            }
        if context.runtime_version == graph_agent_runtime_v3.RUNTIME_VERSION:
            prepared_outbound = whatsapp_outbox.prepare_outbound_envelope(
                lead=lead, text=response.reply_text, sender_type="agent",
                message_id=message_id, correlation_id=message_id,
                idempotency_key=message_id, initial_status="awaiting_proof",
                metadata=outbound_metadata,
            )
        else:
            envelope = whatsapp_outbox.enqueue_outbound(
                lead=lead, text=response.reply_text, sender_type="agent",
                message_id=message_id, correlation_id=message_id,
                idempotency_key=message_id, metadata=outbound_metadata,
            )
            buffer = {"id": envelope.get("buffer_id"), "status": envelope.get("status")}

    graph_turn = None
    if context.runtime_version == graph_agent_runtime_v3.RUNTIME_VERSION:
        active_branch = response.cart_state.get("active_branch_node_id")
        branch_contract: dict[str, Any] = {}
        if response.handoff_required and handoff_level == "full" and active_branch:
            publication = supabase_client.get_active_graph_publication(str(persona.get("id") or "")) or {}
            branch_contract = ((publication.get("document_json") or {}).get("branch_contracts") or {}).get(active_branch) or {}
        reset_facts = _handoff_branch_reset_facts(
            handoff_required=response.handoff_required,
            handoff_level=handoff_level,
            active_branch=active_branch,
            branch_contract=branch_contract,
            branch_facts=response.cart_state.get("facts") or {},
            correlation_id=correlation_id,
        )
        turn_envelope = {
            "canonical_inbound_id": str(inbound_buffer_id),
            "binding_id": channel_binding_id,
            "correlation_id": correlation_id,
            "persona_id": str(lead.get("persona_id")),
            "lead_ref": lead_ref,
            "publication_id": str(context.publication_id),
            "graph_checksum": context.graph_checksum,
            "active_branch_node_id": response.cart_state.get("active_branch_node_id"),
            "active_branch_node_ids": response.cart_state.get("active_branch_node_ids") or [],
            "confirmed_branch_node_ids": response.proof.get("confirmed_branch_node_ids") or [],
            "journey_action": response.proof.get("journey_action") or "continue",
            "agent_slug": context.agent_slug,
            "agent_role": response.role.value,
            "policy_version": response.proof.get("policy_version"),
            "asked_question_node_ids": response.cart_state.get("asked_question_node_ids") or [],
            "expected_revision": int(context.retrieval_trace.get("ledger_revision") or 0),
            "facts": [
                *(response.proof.get("accepted_facts") or []),
                *reset_facts,
            ],
            "retrieval_trace": {
                **context.retrieval_trace,
                "token_usage": response.token_usage or {},
            },
            "model_proposal": (
                response.proposal.model_dump(mode="json") if response.proposal
                else response.proof.get("model_proposal") or {}
            ),
            "proof_result": response.proof,
            "repair_result": {"fallback_used": bool(response.proof.get("fallback_used"))},
            "final_decision": decision.model_dump(mode="json"),
        }
        atomic_seed = {
            "ok": True, "message_id": message_id if response.reply_text else None,
            "deduplicated": False, "handoff": response.handoff_required,
            "ai_paused": response.handoff_required, "route": decision.route.value,
            "role": response.role.value, "intent": decision.intent,
            "reply_text": response.reply_text, "graph_version": context.graph_version,
            "graph_checksum": context.graph_checksum, "stage": qualified_stage,
            "classifier": decision.classifier,
            "evidence_node_ids": decision.evidence_node_ids,
            "knowledge_context": knowledge_context,
            "proof": response.proof,
            "qualification": qualification,
        }
        commit_started = time.monotonic()
        atomic_commit = _commit_graph_turn_and_outbox_or_raise(
            inbound_id=inbound_buffer_id, correlation_id=correlation_id,
            turn=turn_envelope,
            outbound_buffer=(prepared_outbound or {}).get("buffer"),
            outbound_message=(prepared_outbound or {}).get("message"),
            result_payload=atomic_seed,
        )
        commit_ms = round((time.monotonic() - commit_started) * 1000, 3)
        if atomic_commit.get("state") == "burst_superseded":
            return {
                "ok": True,
                "burst_superseded": True,
                "commit_state": "burst_superseded",
                "canonical_inbound_id": inbound_buffer_id,
                "correlation_id": correlation_id,
                "reply_text": None,
                "handoff": False,
                "technical_failure": False,
                "commit_ms": commit_ms,
            }
        graph_turn = atomic_commit.get("graph_turn") or {}
        if atomic_commit.get("outbound_buffer_id"):
            buffer = {
                "id": atomic_commit["outbound_buffer_id"],
                "status": atomic_commit.get("outbound_status"),
            }
        # Projection only: ledger/facts/proof/outbox above are already durable.
        if response.proof.get("journey_action") == "none":
            pass
        elif response.handoff_required:
            supabase_client.handoff_whatsapp_lead_state(
                lead_ref, metadata=metadata, stage=qualified_stage, level=handoff_level,
            )
            if customer_name or service_interest:
                supabase_client.update_lead(lead_ref, lead_update)
        else:
            supabase_client.update_lead(lead_ref, lead_update)

    supabase_client.insert_event(
        {
            "event_type": "conversation.decision_committed",
            "entity_type": "lead",
            "entity_id": str(lead_ref),
            "persona_id": persona.get("id"),
            "payload": {
                "correlation_id": correlation_id,
                "inbound_buffer_id": inbound_buffer_id,
                "trace_id": inbound_buffer_id,
                "decision": decision.model_dump(mode="json"),
                "response_role": response.role.value,
                "handoff": response.handoff_required,
                "graph_version": context.graph_version,
                "graph_checksum": context.graph_checksum,
                "outbound_buffer_id": (buffer or {}).get("id"),
                "qualification": qualification,
            },
        },
        source="conversation_runtime",
    )
    _turn_token_usage = response.token_usage or {}
    _turn_cost_usd = model_pricing.estimate_cost_usd(
        _turn_token_usage.get("model"),
        prompt_tokens=_turn_token_usage.get("prompt_tokens"),
        completion_tokens=_turn_token_usage.get("completion_tokens"),
        cache_hit_tokens=int(_turn_token_usage.get("cache_hit_tokens") or 0),
    )
    emit_turn_event(
        agent_name="conversation.commit",
        trace_id=inbound_buffer_id,
        lead_ref=lead_ref,
        persona_id=persona.get("id"),
        model_used=_turn_token_usage.get("model"),
        token_input=_turn_token_usage.get("prompt_tokens"),
        token_output=_turn_token_usage.get("completion_tokens"),
        output_data={
            "outbound_message_id": message_id if response.reply_text else None,
            "handoff_required": response.handoff_required,
            "handoff_reason": decision.handoff_reason,
            "evidence_node_ids": decision.evidence_node_ids,
        },
        metadata={
            "n8n_execution_id": n8n_execution_id,
            "conversation_id": lead_ref,
            "cost_usd": _turn_cost_usd,
            "repair_calls": _turn_token_usage.get("repair_calls"),
            "model_calls": _turn_token_usage.get("model_calls"),
            "llm_ms": _turn_token_usage.get("llm_latency_ms"),
            "cache_hit_tokens": _turn_token_usage.get("cache_hit_tokens"),
            "cache_miss_tokens": _turn_token_usage.get("cache_miss_tokens"),
            "prompt_estimated_tokens": _turn_token_usage.get("prompt_estimated_tokens"),
            "prompt_budget_limit": _turn_token_usage.get("prompt_budget_limit"),
            "prompt_budget_status": _turn_token_usage.get("prompt_budget_status"),
            "correlation_id": correlation_id,
            "commit_ms": locals().get("commit_ms"),
        },
    )
    result = {
        "ok": True,
        "message_id": message_id if response.reply_text else None,
        "outbound_buffer_id": (buffer or {}).get("id"),
        "deduplicated": False,
        "handoff": response.handoff_required,
        "ai_paused": response.handoff_required,
        "route": decision.route.value,
        "role": response.role.value,
        "intent": decision.intent,
        "reply_text": response.reply_text,
        "classifier": decision.classifier,
        "evidence_node_ids": decision.evidence_node_ids,
        "graph_version": context.graph_version,
        "graph_checksum": context.graph_checksum,
        "knowledge_context": knowledge_context,
        "proof": response.proof,
        "qualification": qualification,
        "stage": qualified_stage,
        "graph_turn": graph_turn,
    }
    if inbound_buffer_id and context.runtime_version != graph_agent_runtime_v3.RUNTIME_VERSION:
        supabase_client.complete_conversation_commit(
            inbound_buffer_id=inbound_buffer_id,
            binding_id=channel_binding_id,
            lead_ref=lead_ref,
            correlation_id=correlation_id,
            result_payload=result,
        )
    return result


# Fields from commit()'s full result that the n8n webhook boundary actually
# needs. commit()'s in-process return value stays untouched (execute_pipeline
# and the stored dedup payload both still get knowledge_context/proof/
# graph_turn/qualification for internal use) -- this only trims what crosses
# the wire back through n8n's "Return canonical result" node, which echoes
# whatever the commit HTTP call returned via `{{$json}}`. Confirmed live
# 2026-08-10: that echo regularly exceeded 64KB (one real turn measured
# 80438 bytes) because it carried the full graph_contract/RAG chunks/proof
# ledger, got silently truncated by the dispatch worker's response_limit,
# and the resulting invalid JSON was misread as a failed turn -- forcing a
# false safety-violation handoff on an otherwise-successful commit.
_DISPATCH_ENVELOPE_OPTIONAL_KEYS = (
    "message_id",
    "outbound_buffer_id",
    "reply_text",
    "route",
    "stage",
    "error",
    "deduplicated",
    "burst_superseded",
    "commit_state",
)


def dispatch_result_envelope(
    result: dict[str, Any], *, correlation_id: str
) -> dict[str, Any]:
    """Minimal, small, always-parseable contract for the n8n webhook caller.

    Only what `whatsapp_dispatch_worker._dispatch_inbound` actually reads
    (`ok`/`handoff`/`technical_failure`) plus enough identifiers and the
    outbound reply text for observability -- never the graph/RAG/proof
    payload that made the full result balloon past the response size limit.
    """
    envelope: dict[str, Any] = {
        "ok": bool(result.get("ok", True)),
        "handoff": bool(result.get("handoff")),
        "technical_failure": bool(result.get("technical_failure")),
        "correlation_id": correlation_id,
    }
    for key in _DISPATCH_ENVELOPE_OPTIONAL_KEYS:
        value = result.get(key)
        if value is not None:
            envelope[key] = value
    return envelope


def execute_pipeline(
    *,
    persona_slug: str,
    lead_ref: int,
    message: str,
    message_id: str | None,
    correlation_id: str,
    phone_number_id: str | None,
    channel_binding_id: str | None = None,
    inbound_buffer_id: str | None = None,
) -> dict[str, Any]:
    """Run the same context -> classify -> commit contract used by n8n."""
    context = build_context(
        persona_slug=persona_slug,
        lead_ref=lead_ref,
        message=message,
        message_id=message_id,
    )
    decision, response = decide(context)
    result = commit(
        lead_ref=lead_ref,
        context=context,
        decision=decision,
        response=response,
        correlation_id=correlation_id,
        phone_number_id=phone_number_id,
        channel_binding_id=channel_binding_id,
        inbound_buffer_id=inbound_buffer_id,
        expected_decision_owner="deterministic",
    )
    return {
        **result,
        "pipeline_contract": "conversation_v1",
        "classifier": decision.classifier,
        "context": {
            "graph_version": context.graph_version,
            "graph_checksum": context.graph_checksum,
        },
    }

