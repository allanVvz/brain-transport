"""One explainable Graph JSON context package shared by model and operator UI."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import deque
from time import monotonic
from typing import Any, Iterable

from schemas.conversation import ContextCard
from schemas.graph_json_v2 import DEFAULT_RELATION_WEIGHTS, GraphJson, Node
from services import (
    graph_compiler_v3,
    graph_conversation_contract,
    graph_json_v2_store,
    graph_markdown,
    supabase_client,
)


PERMANENT_TYPES = {"persona", "brand", "tone", "rule"}
CONTEXT_TYPES = {
    "product", "offer", "faq", "copy", "campaign", "product_group",
    "audience", "asset", "briefing", "brand", "tone", "rule", "persona",
}
TYPE_PRIORITY = {
    "faq": 1.0, "product": 1.0, "offer": 0.98, "rule": 0.96,
    "copy": 0.94, "tone": 0.9, "brand": 0.86, "campaign": 0.82,
    "product_group": 0.8, "audience": 0.72, "asset": 0.68,
    "persona": 0.65, "briefing": 0.5,
}
ALLOWED_RELATIONS = {
    "contains", "supports", "supports_copy", "answers", "answers_question",
    "about_product", "part_of_campaign", "applies_to", "targets",
    "represents", "uses_asset", "same_topic_as", "derived_from",
    "references", "visible_to_agent", "belongs_to_persona",
}
APPROVED = {"approved", "active", "validated", "ativo", "embedded"}
STOPWORDS = {
    "a", "as", "ao", "aos", "com", "da", "das", "de", "do", "dos",
    "e", "em", "na", "nas", "no", "nos", "o", "os", "ou", "para",
    "por", "que", "um", "uma", "qual", "quais",
}
_DIVERGENCE_EMITTED_AT: dict[str, float] = {}
_DIVERGENCE_TTL_SECONDS = 15 * 60
NON_PROMPT_FIELDS = {
    "active", "aliases", "approved_at", "approved_by", "branch_path",
    "canonical_key", "content_hash", "created_at", "empty", "entities",
    "file_path", "graph_checksum", "graph_id", "graph_json_id",
    "graph_json_import", "graph_json_node_id", "graph_json_parent_id",
    "graph_version", "knowledge_item_id", "knowledge_node_id", "language",
    "markdown", "markdown_checksum", "markdown_document", "markdown_renderer",
    "metadata", "parent_id", "protected", "question_count", "schema_version",
    "session_id", "source", "source_checksum", "source_node_id",
    "source_node_type", "status", "tags", "updated_at", "validation_status",
    "type", "persona", "slug", "title", "name", "relations", "rag_chunk",
    "source_file",
}


def _fold(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _tokens(value: Any) -> set[str]:
    return {
        item for item in _fold(value).split()
        if len(item) > 1 and item not in STOPWORDS
    }


def _sha256(value: str) -> str:
    return graph_compiler_v3.canonical_content_checksum(value)


def _content_checksum_matches(expected: str, rendered: str) -> tuple[bool, str]:
    actual = _sha256(rendered)
    normalized_expected = str(expected or "")
    if normalized_expected and not normalized_expected.startswith("sha256:"):
        normalized_expected = "sha256:" + normalized_expected
    if normalized_expected == actual:
        return True, actual
    # Historical cards created before canonical newline/NFC hashing remain
    # valid when their exact snapshot hash matches.
    legacy_raw = "sha256:" + hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    legacy_json_string = graph_compiler_v3.canonical_checksum(rendered)
    return normalized_expected in {legacy_raw, legacy_json_string}, actual


def _emit_checksum_divergence_once(
    *, persona_id: str, lead_ref: int, payload: dict[str, Any],
) -> None:
    now = monotonic()
    key = json.dumps(
        {"persona_id": persona_id, "lead_ref": lead_ref, **payload},
        sort_keys=True, ensure_ascii=False, default=str,
    )
    last = _DIVERGENCE_EMITTED_AT.get(key)
    if last is not None and now - last < _DIVERGENCE_TTL_SECONDS:
        return
    if len(_DIVERGENCE_EMITTED_AT) >= 2048:
        cutoff = now - _DIVERGENCE_TTL_SECONDS
        retained = {
            item: emitted_at for item, emitted_at in _DIVERGENCE_EMITTED_AT.items()
            if emitted_at >= cutoff
        }
        _DIVERGENCE_EMITTED_AT.clear()
        _DIVERGENCE_EMITTED_AT.update(retained)
    _DIVERGENCE_EMITTED_AT[key] = now
    emit_metric(
        "knowledge_context.checksum_divergence",
        persona_id=persona_id, lead_ref=lead_ref, payload=payload,
    )


def _rendered(node: Node) -> str:
    """Render only facts that are useful to the model, never trace metadata."""
    spec = dict(node.spec or {})
    data = dict(node.data or {})
    title = str(node.title or node.label or node.slug).strip()
    lines = [f"# {title}"]
    if node.node_type == "faq":
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        content = data.get("content")
        structured_content = content if isinstance(content, dict) else {}
        question = str(
            spec.get("question") or data.get("question")
            or structured_content.get("question") or title
        ).strip()
        answer = str(
            spec.get("answer") or data.get("answer")
            or structured_content.get("answer")
            or (content if isinstance(content, str) else "")
            or (data.get("metadata") or {}).get("answer") or ""
        ).strip()
        legacy_markdown = str(data.get("markdown") or "").strip() if node.markdown is None else ""
        if question:
            lines.extend(["", "Pergunta: " + question])
        if answer:
            lines.extend(["", "Resposta: " + answer])
        if not answer and (metadata.get("role") or data.get("role")) == "qualification_question":
            return "\n".join(lines).strip()
        if not answer and legacy_markdown:
            return "\n".join([f"# {title}", "", legacy_markdown]).strip()
        return "\n".join(lines).strip() if answer else ""

    summary = str(
        spec.get("summary") or spec.get("content") or spec.get("text")
        or data.get("summary") or data.get("content") or data.get("text")
        or data.get("statement")
        or (data.get("markdown") if node.markdown is None else "")
        or ""
    ).strip()
    if summary:
        lines.extend(["", summary])

    facts = {
        key: value for key, value in {**data, **spec}.items()
        if key not in NON_PROMPT_FIELDS
        and key not in {"summary", "content", "text", "statement", "question", "answer"}
        and value not in (None, "", [], {})
    }
    if facts:
        lines.extend(["", "Fatos estruturados:"])
        for key in sorted(facts):
            value = facts[key]
            rendered = (
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list)) else str(value)
            )
            lines.append(f"- {key.replace('_', ' ')}: {rendered}")
    return "\n".join(lines).strip() if len(lines) > 1 else ""


def _editable(node: Node) -> str:
    spec = node.spec or {}
    data = node.data or {}
    if node.node_type == "faq":
        content = data.get("content")
        structured_content = content if isinstance(content, dict) else {}
        value = str(
            spec.get("answer") or data.get("answer")
            or structured_content.get("answer")
            or (content if isinstance(content, str) else "")
            or (data.get("metadata") or {}).get("answer") or ""
        ).strip()
        if value and value != "{}":
            return value
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        if (metadata.get("role") or data.get("role")) == "qualification_question":
            return str(
                spec.get("question") or data.get("question")
                or structured_content.get("question") or node.title or ""
            ).strip()
        return value
    return str(
        spec.get("summary") or spec.get("content") or data.get("content")
        or data.get("summary") or (data.get("markdown") if node.markdown is None else "")
        or _rendered(node)
    ).strip()


def _search_text(node: Node) -> str:
    spec = node.spec or {}
    data = node.data or {}
    aliases = spec.get("aliases") or data.get("aliases") or []
    tags = spec.get("tags") or data.get("tags") or []
    entities = spec.get("entities") or data.get("entities") or []
    return " ".join(
        str(item) for item in (
            node.id, node.slug, node.title, aliases, tags, entities,
            data.get("summary"), data.get("content"), data.get("question"),
            data.get("answer"), _rendered(node),
        ) if item
    )


def _projection_maps(projection_nodes: Iterable[dict[str, Any]] | None) -> tuple[dict[str, str], dict[str, str]]:
    stable_to_uuid: dict[str, str] = {}
    uuid_to_stable: dict[str, str] = {}
    for row in projection_nodes or []:
        metadata = row.get("metadata") or {}
        stable = str(metadata.get("graph_json_node_id") or "")
        projected = str(row.get("id") or "")
        if stable and projected:
            stable_to_uuid[stable] = projected
            uuid_to_stable[projected] = stable
    return stable_to_uuid, uuid_to_stable


def canonicalize_ids(
    ids: Iterable[Any], projection_nodes: Iterable[dict[str, Any]] | None,
) -> list[str]:
    """Recognize both stable Graph ids and projected knowledge_nodes UUIDs."""
    _, uuid_to_stable = _projection_maps(projection_nodes)
    return list(dict.fromkeys(
        uuid_to_stable.get(str(value), str(value)) for value in ids if value
    ))


def _published_ids(graph: GraphJson) -> set[str]:
    by_id = {node.id: node for node in graph.nodes}
    action_ids = {
        node.id for node in graph.nodes
        if node.node_class == "action" and node.action and node.action.enabled
        and (
            node.node_type == "embedded"
            or str(node.action.consumer.kind).lower() in {"agent", "assistant", "conversation"}
        )
    }
    grants = {
        edge.source for edge in graph.edges
        if edge.lifecycle.status == "active" and edge.relation_type == "publishes_to"
        and edge.target in action_ids and edge.source in by_id
    }
    # v2.0 documents predate destination grants. Keep their approved content
    # usable during the migration, but v2.1 strictly honors publication.
    if graph.schema_version == "2.0" or not action_ids:
        return {node.id for node in graph.nodes if node.node_class == "knowledge"}
    return grants


def _relations(graph: GraphJson, node_id: str) -> list[dict[str, Any]]:
    return [
        {
            "id": edge.id,
            "source": edge.source,
            "target": edge.target,
            "relation_type": edge.relation_type,
            "weight": edge.weight,
        }
        for edge in graph.edges
        if edge.lifecycle.status == "active" and node_id in {edge.source, edge.target}
    ]


def _chunk_map(rag_chunks: Iterable[dict[str, Any]] | None) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for chunk in rag_chunks or []:
        metadata = {
            **((chunk.get("entry") or {}).get("metadata") or {}),
            **(chunk.get("metadata") or {}),
        }
        stable_id = str(
            metadata.get("graph_json_node_id") or metadata.get("canonical_node_id")
            or metadata.get("source_node_id") or ""
        )
        chunk_id = str(chunk.get("id") or "")
        if stable_id and chunk_id:
            result.setdefault(stable_id, []).append(chunk_id)
    return result


def _card(
    *, node: Node, graph: GraphJson, graph_version: int, graph_checksum: str,
    position: int, context_role: str, reason: dict[str, Any], path: list[str],
    projection_id: str | None, chunk_refs: list[str],
) -> ContextCard:
    rendered = _rendered(node)
    coordinate = graph_conversation_contract.coordinate_for_node(
        graph, node.id, graph_version=graph_version
    )
    return ContextCard(
        id=node.id,
        projection_node_id=projection_id,
        node_type=node.node_type,
        slug=node.slug,
        title=str(node.title or node.label or node.slug),
        rendered_content=rendered,
        editable_content=_editable(node),
        content_checksum=_sha256(rendered),
        revision=node.lifecycle.revision,
        graph_version=graph_version,
        graph_checksum=graph_checksum,
        context_role=context_role,
        position=position,
        selection_reason=reason,
        path=path,
        chunk_refs=chunk_refs,
        source=str(node.provenance.source or node.provenance.source_type or "pending_source"),
        status=node.lifecycle.status,
        relations=_relations(graph, node.id),
        technical_metadata={
            "aliases": (node.spec or {}).get("aliases") or (node.data or {}).get("aliases") or [],
            "tags": (node.spec or {}).get("tags") or (node.data or {}).get("tags") or [],
            "approved_at": node.lifecycle.approved_at,
            "approved_by": node.lifecycle.approved_by,
            "source_ref": node.provenance.source_ref,
            "source_checksum": node.provenance.source_checksum,
            "canonical_markdown_checksum": node.markdown.checksum if node.markdown else (node.data or {}).get("markdown_checksum"),
            **coordinate,
        },
    )


def resolve_cards(
    *, graph: GraphJson, graph_version: int, graph_checksum: str, query: str,
    rag_chunks: list[dict[str, Any]] | None = None,
    projection_nodes: list[dict[str, Any]] | None = None,
    decisive_node_ids: list[str] | None = None,
    branch_node_ids: list[str] | None = None,
    active_path_node_ids: list[str] | None = None,
    unresolved_fields: list[str] | None = None,
    max_cards: int = 12, max_tokens: int = 8000, max_distance: int = 3,
) -> list[ContextCard]:
    """Resolve exactly the package that both model and UI consume."""
    graph = graph_markdown.canonicalize_graph(graph, reject_markdown_drift=False)
    by_id = {node.id: node for node in graph.nodes}
    stable_to_uuid, _ = _projection_maps(projection_nodes)
    chunk_refs = _chunk_map(rag_chunks)
    allowed = _published_ids(graph) | {
        node.id for node in graph.nodes
        if node.node_class == "knowledge"
        and node.node_type in PERMANENT_TYPES
        and node.lifecycle.status in APPROVED
    }
    if branch_node_ids is not None:
        allowed &= set(branch_node_ids)
    approved = {
        node.id for node in graph.nodes
        if node.id in allowed and node.node_class == "knowledge"
        and node.node_type in CONTEXT_TYPES
        and node.lifecycle.status in APPROVED and bool(_rendered(node))
    }
    query_tokens = _tokens(query)
    catalog_request = bool(
        query_tokens
        & {"cardapio", "catalogo", "produtos", "servicos", "opcoes", "menu"}
    )
    decisive = set(canonicalize_ids(decisive_node_ids or [], projection_nodes))
    active_path = set(active_path_node_ids or [])
    pending_fields = set(unresolved_fields or [])

    seed_scores: dict[str, float] = {}
    for node_id in approved:
        node = by_id[node_id]
        haystack = _fold(_search_text(node))
        hay_tokens = _tokens(haystack)
        exact = _fold(query) in {_fold(node.id), _fold(node.slug), _fold(node.title)}
        lexical = len(query_tokens & hay_tokens) / max(1, len(query_tokens)) if query_tokens else 0.0
        matched_enough = exact or lexical >= 0.4 or node_id in chunk_refs or node_id in decisive
        score = (
            (1.0 if exact else 0.75 * lexical)
            + (0.2 if node_id in chunk_refs else 0.0)
            if matched_enough else 0.0
        )
        coordinate = graph_conversation_contract.coordinate_for_node(
            graph, node_id, graph_version=graph_version
        )
        path_overlap = len(set(coordinate["path_node_ids"]) & active_path) / max(1, len(active_path))
        metadata = (node.data or {}).get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        field_key = str(metadata.get("field_key") or (node.data or {}).get("field_key") or "")
        if path_overlap:
            score += 0.3 * path_overlap
        if field_key and field_key in pending_fields:
            score += 0.35
        if catalog_request and node.node_type in {"product", "offer", "product_group"}:
            score = max(score, 0.72)
        if node_id in decisive:
            score = max(score, 1.1)
        if score > 0:
            seed_scores[node_id] = score

    adjacency: dict[str, list[tuple[str, str, float]]] = {}
    for edge in graph.edges:
        relation = str(edge.relation_type or "references")
        if (
            edge.lifecycle.status != "active" or relation not in ALLOWED_RELATIONS
            or edge.source not in approved or edge.target not in approved
        ):
            continue
        weight = float(edge.weight or DEFAULT_RELATION_WEIGHTS.get(relation, 0.5))
        adjacency.setdefault(edge.source, []).append((edge.target, relation, weight))
        adjacency.setdefault(edge.target, []).append((edge.source, relation, weight))

    ranked: dict[str, dict[str, Any]] = {}
    for seed_id, seed_score in sorted(seed_scores.items(), key=lambda item: item[1], reverse=True)[:8]:
        queue = deque([(seed_id, [seed_id], [], 1.0)])
        visited = {seed_id: 0}
        while queue:
            node_id, path, relations, path_strength = queue.popleft()
            distance = len(path) - 1
            node = by_id[node_id]
            briefing_is_relevant = (
                node.node_type != "briefing"
                or node_id == seed_id
                or node_id in chunk_refs
                or node_id in decisive
            )
            direct_semantic_neighbor = (
                distance <= 1
                and by_id[seed_id].node_type in {"product", "offer", "faq", "copy", "product_group"}
                and node.node_type in {"product", "offer", "faq", "copy", "product_group"}
            )
            branch_is_relevant = not (
                node.node_type in {"product", "offer", "product_group"}
                and node_id != seed_id
                and node_id not in seed_scores
                and node_id not in chunk_refs
                and node_id not in decisive
                and not direct_semantic_neighbor
            )
            if node.node_type in {"faq", "copy"} and distance > 1 and node_id not in seed_scores:
                branch_is_relevant = False
            final = (
                0.42 * min(seed_score, 1.0)
                + 0.23 * path_strength
                + 0.15 * (1.0 / (distance + 1))
                + 0.12 * TYPE_PRIORITY.get(node.node_type, 0.5)
                + 0.08
            )
            candidate = {
                "score": final, "seed_node_id": seed_id, "path": path,
                "relations": relations, "distance": distance,
            }
            if briefing_is_relevant and branch_is_relevant and (node_id not in ranked or final > ranked[node_id]["score"]):
                ranked[node_id] = candidate
            if distance >= max_distance:
                continue
            for neighbor, relation, weight in adjacency.get(node_id, []):
                if visited.get(neighbor, 999) <= distance + 1:
                    continue
                visited[neighbor] = distance + 1
                queue.append((neighbor, [*path, neighbor], [*relations, relation], path_strength * weight))

    # Identity and policy cards are permanent only because the system prompt
    # consumes these same snapshots. Briefing intentionally stays contextual.
    ordered_ids = [
        node.id for node in graph.nodes
        if node.id in approved and node.node_type in PERMANENT_TYPES
    ]
    ordered_ids.extend(
        node_id for node_id, _ in sorted(ranked.items(), key=lambda item: (-item[1]["score"], item[0]))
        if node_id not in ordered_ids
    )
    selected: list[ContextCard] = []
    token_count = 0
    for node_id in ordered_ids:
        node = by_id[node_id]
        rendered = _rendered(node)
        estimated = max(1, len(rendered) // 4)
        if selected and (len(selected) >= max_cards or token_count + estimated > max_tokens):
            continue
        role = "permanent" if node.node_type in PERMANENT_TYPES else "contextual"
        detail = ranked.get(node_id) or {
            "score": 1.0, "seed_node_id": node_id, "path": [node_id],
            "relations": [], "distance": 0,
        }
        selected.append(_card(
            node=node, graph=graph, graph_version=graph_version,
            graph_checksum=graph_checksum, position=len(selected), context_role=role,
            reason={
                "reason": "system_prompt" if role == "permanent" else "query_and_graph",
                **detail,
            },
            path=list(detail["path"]), projection_id=stable_to_uuid.get(node_id),
            chunk_refs=chunk_refs.get(node_id, []),
        ))
        token_count += estimated
    return selected


def cards_for_ids(
    *, graph: GraphJson, graph_version: int, graph_checksum: str,
    ids: list[str], projection_nodes: list[dict[str, Any]] | None = None,
) -> list[ContextCard]:
    """Rebuild only explicitly recorded historical evidence, never inferred nodes."""
    graph = graph_markdown.canonicalize_graph(graph, reject_markdown_drift=False)
    stable_to_uuid, _ = _projection_maps(projection_nodes)
    canonical_ids = canonicalize_ids(ids, projection_nodes)
    by_id = {node.id: node for node in graph.nodes}
    cards: list[ContextCard] = []
    for node_id in canonical_ids:
        node = by_id.get(node_id)
        if not node or not _rendered(node):
            continue
        cards.append(_card(
            node=node, graph=graph, graph_version=graph_version,
            graph_checksum=graph_checksum, position=len(cards),
            context_role="reconstructed", reason={"reason": "recorded_evidence_id"},
            path=[node_id], projection_id=stable_to_uuid.get(node_id), chunk_refs=[],
        ))
    return cards


def current_graph(persona_slug: str) -> tuple[int, str, GraphJson]:
    current = graph_json_v2_store.load_current(persona_slug)
    if not current:
        raise LookupError("Published Graph JSON not found")
    version, graph = current
    event = graph_json_v2_store.latest_event(persona_slug) or {}
    checksum = str((event.get("payload") or {}).get("checksum") or graph_json_v2_store.checksum_graph(graph))
    return int(version), checksum, graph


def load_graph_version(persona_slug: str, version: int | None) -> tuple[int, str, GraphJson] | None:
    if not version:
        return None
    graph = graph_json_v2_store.load_activated_version(persona_slug, int(version))
    if not graph:
        return None
    return int(version), str(graph.content_checksum or graph_json_v2_store.checksum_graph(graph)), graph


def emit_metric(event_type: str, *, persona_id: str | None, lead_ref: int | None, payload: dict[str, Any]) -> None:
    try:
        supabase_client.insert_event(
            {
                "event_type": event_type,
                "entity_type": "message" if lead_ref else "knowledge_context",
                "entity_id": str(lead_ref or payload.get("response_message_id") or "context"),
                "persona_id": persona_id,
                "payload": payload,
            },
            source="context_cards",
        )
    except Exception:
        return


def _message_metadata(message: dict[str, Any]) -> dict[str, Any]:
    metadata = message.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            import json
            metadata = json.loads(metadata)
        except Exception:
            return {}
    return metadata if isinstance(metadata, dict) else {}


def _message_identity(message: dict[str, Any]) -> str:
    return str(
        message.get("message_id") or message.get("sender_id")
        or message.get("correlation_id") or message.get("id") or ""
    )


def _automatic_response(message: dict[str, Any]) -> bool:
    role = str(message.get("role") or "").lower()
    sender = str(message.get("sender_type") or "").lower()
    direction = str(message.get("direction") or "").lower()
    return (
        role == "assistant" or sender in {"ai", "assistant", "agent"}
        or (direction in {"outbound", "outbounding"} and role != "human")
    )


def response_context(
    *, persona_slug: str, persona_id: str, lead_ref: int,
    messages: list[dict[str, Any]], response_message_id: str | None,
    query: str, projection_nodes: list[dict[str, Any]] | None = None,
    limit: int = 12,
) -> dict[str, Any]:
    """Return exact per-response evidence or a conservative historical rebuild."""
    selected: dict[str, Any] | None = None
    if response_message_id:
        wanted = str(response_message_id)
        selected = next(
            (
                row for row in messages
                if wanted in {
                    str(row.get("id") or ""), str(row.get("message_id") or ""),
                    str(row.get("sender_id") or ""), str(row.get("correlation_id") or ""),
                }
            ),
            None,
        )
        if selected is None or not _automatic_response(selected):
            raise LookupError("Automatic response message not found")
    else:
        candidates = [row for row in messages if _automatic_response(row)]
        selected = next(
            (
                row for row in reversed(candidates)
                if (_message_metadata(row).get("knowledge_context") or {}).get("cards")
                or _message_metadata(row).get("evidence_node_ids")
                or (_message_metadata(row).get("knowledge_context") or {}).get("decisive_node_ids")
            ),
            candidates[-1] if candidates else None,
        )

    current_version, current_checksum, current = current_graph(persona_slug)
    current_cards = resolve_cards(
        graph=current, graph_version=current_version, graph_checksum=current_checksum,
        query=query, projection_nodes=projection_nodes, max_cards=limit,
    )
    if selected is None:
        return {
            "mode": "reconstructed",
            "persona_slug": persona_slug,
            "response": None,
            "used_cards": [],
            "related_cards": [card.model_dump(mode="json") for card in current_cards],
            "current_cards": {},
            "decisive_node_ids": [],
            "graph_version": current_version,
            "graph_checksum": current_checksum,
        }

    metadata = _message_metadata(selected)
    envelope = metadata.get("knowledge_context") or {}
    exact_cards: list[ContextCard] = []
    if isinstance(envelope, dict) and isinstance(envelope.get("cards"), list):
        for raw in envelope["cards"]:
            try:
                card = ContextCard.model_validate(raw)
                exact_cards.append(card)
                checksum_ok, actual_checksum = _content_checksum_matches(
                    card.content_checksum, card.rendered_content,
                )
                if not checksum_ok:
                    _emit_checksum_divergence_once(
                        persona_id=persona_id, lead_ref=lead_ref,
                        payload={
                            "response_message_id": _message_identity(selected),
                            "node_id": card.id,
                            "expected": card.content_checksum,
                            "actual": actual_checksum,
                            "reason": "snapshot_content_mismatch",
                        },
                    )
            except Exception:
                _emit_checksum_divergence_once(
                    persona_id=persona_id, lead_ref=lead_ref,
                    payload={"response_message_id": _message_identity(selected), "reason": "invalid_card_snapshot"},
                )
        mode = "exact"
        decisive_ids = canonicalize_ids(envelope.get("decisive_node_ids") or [], projection_nodes)
        graph_version = int(
            envelope.get("graph_version")
            or (exact_cards[0].graph_version if exact_cards else current_version)
        )
        graph_checksum = str(
            envelope.get("graph_checksum")
            or (exact_cards[0].graph_checksum if exact_cards else current_checksum)
        )
    else:
        mode = "reconstructed"
        decisive_ids = canonicalize_ids(
            metadata.get("evidence_node_ids")
            or envelope.get("decisive_node_ids")
            or [],
            projection_nodes,
        )
        try:
            graph_version = int(metadata.get("graph_version") or envelope.get("graph_version") or 0)
        except (TypeError, ValueError):
            graph_version = 0
        historical = load_graph_version(persona_slug, graph_version)
        if historical:
            graph_version, graph_checksum, historical_graph = historical
            exact_cards = cards_for_ids(
                graph=historical_graph, graph_version=graph_version,
                graph_checksum=graph_checksum, ids=decisive_ids,
                projection_nodes=projection_nodes,
            )
        else:
            graph_checksum = str(metadata.get("graph_checksum") or "unavailable")
        emit_metric(
            "knowledge_context.historical_reconstruction",
            persona_id=persona_id, lead_ref=lead_ref,
            payload={
                "response_message_id": _message_identity(selected),
                "graph_version": graph_version, "card_count": len(exact_cards),
            },
        )

    missing_ids = [node_id for node_id in decisive_ids if node_id not in {card.id for card in exact_cards}]
    if missing_ids:
        emit_metric(
            "knowledge_context.id_unmatched",
            persona_id=persona_id, lead_ref=lead_ref,
            payload={"response_message_id": _message_identity(selected), "node_ids": missing_ids},
        )
    if not exact_cards:
        emit_metric(
            "knowledge_context.empty",
            persona_id=persona_id, lead_ref=lead_ref,
            payload={"response_message_id": _message_identity(selected), "mode": mode},
        )

    used_ids = {card.id for card in exact_cards}
    current_by_id = {
        card.id: card.model_dump(mode="json")
        for card in cards_for_ids(
            graph=current, graph_version=current_version,
            graph_checksum=current_checksum, ids=list(used_ids),
            projection_nodes=projection_nodes,
        )
    }
    related = [card for card in current_cards if card.id not in used_ids]
    return {
        "mode": mode,
        "persona_slug": persona_slug,
        "response": {
            "id": selected.get("id"),
            "message_id": _message_identity(selected),
            "created_at": selected.get("created_at"),
            "text": selected.get("texto") or selected.get("content") or "",
        },
        "used_cards": [card.model_dump(mode="json") for card in exact_cards],
        "related_cards": [card.model_dump(mode="json") for card in related],
        "current_cards": current_by_id,
        "decisive_node_ids": decisive_ids,
        "graph_version": graph_version,
        "graph_checksum": graph_checksum,
        "current_graph_version": current_version,
        "current_graph_checksum": current_checksum,
    }

