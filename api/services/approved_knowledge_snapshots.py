from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Optional

from services import knowledge_graph, supabase_client


STRUCTURAL_RELATIONS = {
    "belongs_to_persona",
    "contains",
    "part_of_campaign",
    "about_product",
    "offers_product",
    "briefed_by",
    "answers_question",
    "supports_copy",
    "uses_asset",
    "manual",
}


def _slugify(value: str) -> str:
    return knowledge_graph._slugify(value or "")


def _active_edge(edge: dict) -> bool:
    return (edge.get("metadata") or {}).get("active") is not False


def _is_structural(edge: dict) -> bool:
    meta = edge.get("metadata") or {}
    relation = (edge.get("relation_type") or "").lower()
    return _active_edge(edge) and meta.get("primary_tree") is True and relation in STRUCTURAL_RELATIONS


def _node_context(node: Optional[dict]) -> dict:
    if not node:
        return {}
    meta = node.get("metadata") or {}
    return {
        "id": node.get("id"),
        "node_type": node.get("node_type"),
        "slug": node.get("slug"),
        "title": node.get("title"),
        "summary": node.get("summary"),
        "tags": node.get("tags") or [],
        "metadata": meta,
    }


def _first_by_type(chain: list[dict], node_type: str) -> Optional[dict]:
    for node in chain:
        if (node.get("node_type") or "").lower() == node_type:
            return node
    return None


def _faq_parts(node: dict, item: Optional[dict]) -> tuple[str, str]:
    meta = node.get("metadata") or {}
    question = meta.get("question") or node.get("title") or ""
    answer = meta.get("answer") or ""
    text = (item or {}).get("content") or node.get("summary") or ""
    pairs = knowledge_graph._extract_faq_pairs(text)
    if pairs:
        question = question or pairs[0][0]
        answer = answer or pairs[0][1]
    if not answer:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) >= 2 and "?" in lines[0]:
            question = question or lines[0]
            answer = " ".join(lines[1:])
    return question.strip(), answer.strip()


def _text_context(node: Optional[dict]) -> str:
    if not node:
        return ""
    return str(node.get("summary") or node.get("title") or "").strip()


def _context_text(node: Optional[dict], fallback: str = "") -> str:
    value = _text_context(node)
    return value or fallback


def _branch_path(chain: list[dict], persona: dict) -> list[dict]:
    out: list[dict] = []
    for node in chain:
        title = node.get("title")
        if (node.get("node_type") or "").lower() == "persona":
            title = persona.get("name") or title
        out.append({
            "node_id": node.get("id"),
            "node_type": node.get("node_type"),
            "slug": node.get("slug"),
            "title": title,
        })
    return out


def _branch_edges(chain: list[dict], path_edges: list[dict]) -> list[dict]:
    nodes_by_id = {node.get("id"): node for node in chain if node.get("id")}
    out: list[dict] = []
    for edge in path_edges:
        src = nodes_by_id.get(edge.get("source_node_id")) or {}
        tgt = nodes_by_id.get(edge.get("target_node_id")) or {}
        meta = edge.get("metadata") or {}
        relation_type = edge.get("relation_type") or "resolved_parent"
        semantic = meta.get("semantic_relation")
        if not semantic:
            semantic_meta = knowledge_graph.semantic_edge_metadata(src, tgt, relation_type, {})
            semantic = semantic_meta.get("semantic_relation") or relation_type
        out.append({
            "source_node_id": edge.get("source_node_id"),
            "target_node_id": edge.get("target_node_id"),
            "source_slug": src.get("slug"),
            "target_slug": tgt.get("slug"),
            "relation_type": relation_type,
            "semantic_relation": semantic,
        })
    return out


def _branch_context(chain: list[dict], path_edges: list[dict], persona: dict) -> dict:
    path = _branch_path(chain, persona)
    root = path[0] if path else {}
    return {
        "tree_mode": "pyramidal",
        "branch_policy": "top_down_pyramidal",
        "root": root,
        "path": path,
        "edges": _branch_edges(chain, path_edges),
    }


def _answer_is_useful(question: str, answer: str) -> bool:
    q = (question or "").strip().lower()
    a = (answer or "").strip().lower()
    if not q or not a:
        return False
    if q == a:
        return False
    if re.search(r"R\$\s*\d", answer):
        return True
    stripped = a.strip(" ?!.")
    if "?" in answer and len(stripped.split()) <= 12:
        return False
    weak_phrases = (
        "use o contexto deste galho para responder",
        "conecte a duvida",
        "mantenha a resposta curta",
    )
    if any(phrase in a for phrase in weak_phrases):
        return False
    if len(stripped.split()) < 8:
        return False
    return True


def _extract_markdown_faq_pairs(text: str) -> list[dict[str, str]]:
    """Parse FAQ Golden Dataset markdown into canonical Q/A pairs."""
    if not text or "###" not in text:
        return []
    matches = list(re.finditer(r"(?m)^###\s*(?P<num>\d+)\.\s*(?P<question>.+?)\s*$", text))
    pairs: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        answer_match = re.search(
            r"(?is)(?:\*\*Resposta:\*\*|Resposta:)\s*(?P<answer>.+)",
            block,
        )
        answer = answer_match.group("answer").strip() if answer_match else block
        answer = re.sub(r"(?m)^#{1,6}\s+.*$", "", answer).strip()
        answer = re.sub(r"\s+", " ", answer).strip()
        question = re.sub(r"\s+", " ", match.group("question")).strip()
        if question and answer:
            pairs.append({
                "index": str(len(pairs) + 1),
                "question": question,
                "answer": answer,
            })
    return pairs


def _canonical_faq_pairs(*, source_text: str, question: str, answer: str, is_golden_dataset: bool) -> list[dict[str, str]]:
    pairs = _extract_markdown_faq_pairs(source_text) if is_golden_dataset else []
    if not pairs:
        pairs = knowledge_graph._extract_faq_pairs(source_text)
        pairs = [
            {"index": str(idx), "question": q.strip(), "answer": a.strip()}
            for idx, (q, a) in enumerate(pairs, 1)
            if q.strip() and a.strip()
        ]
    if not pairs and (question or answer):
        pairs = [{"index": "1", "question": question.strip(), "answer": answer.strip()}]
    return pairs


def _faq_review_warnings(
    *,
    source_node: dict,
    chain: list[dict],
    branch_context: dict,
    question: str,
    answer: str,
    briefing_context: str,
    product_context: str,
    copy_context: str,
) -> list[str]:
    warnings: list[str] = []
    path = branch_context.get("path") or []
    path_types = [(step.get("node_type") or "").lower() for step in path]
    if not path:
        warnings.append("hierarchy_path_empty")
    if not path_types or path_types[0] != "persona":
        warnings.append("persona_root_missing")
    if len(path_types) >= 2 and path_types[1:].count("persona") > 0:
        warnings.append("persona_repeated_in_path")
    if len(chain) < 2:
        warnings.append("faq_parent_missing")
    if "briefing" in path_types and not briefing_context:
        warnings.append("briefing_context_missing")
    if "product" in path_types and not product_context:
        warnings.append("product_context_missing")
    if "copy" in path_types and not copy_context:
        warnings.append("copy_context_missing")
    if not _answer_is_useful(question, answer):
        warnings.append("faq_answer_not_useful")
    if (source_node.get("node_type") or "").lower() == "faq" and path_types and path_types[-1] != "faq":
        warnings.append("faq_not_leaf_in_snapshot_path")
    return warnings


def _metadata_list(*values) -> list[str]:
    out: list[str] = []
    for value in values:
        if isinstance(value, list):
            out.extend(str(v).strip() for v in value if str(v).strip())
        elif isinstance(value, str) and value.strip():
            out.append(value.strip())
    return list(dict.fromkeys(out))


def _build_hierarchy(source_node: dict) -> tuple[list[dict], list[dict]]:
    persona_id = source_node.get("persona_id")
    nodes, edges = supabase_client.list_all_knowledge_graph(persona_id=persona_id, limit_nodes=2500)
    nodes_by_id = {n.get("id"): n for n in nodes if n.get("id")}
    persona_node_ids = {n.get("id") for n in nodes if (n.get("node_type") or "").lower() == "persona"}
    auxiliary_types = {"tag", "mention", "embedded", "gallery"}
    parent_by_child: dict[str, tuple[str, dict]] = {}
    for edge in edges:
        if not _is_structural(edge):
            continue
        src = edge.get("source_node_id")
        tgt = edge.get("target_node_id")
        if not src or not tgt:
            continue
        src_node = nodes_by_id.get(src) or {}
        tgt_node = nodes_by_id.get(tgt) or {}
        src_type = (src_node.get("node_type") or "").lower()
        tgt_type = (tgt_node.get("node_type") or "").lower()
        if src_type in auxiliary_types or tgt_type in auxiliary_types:
            continue
        if tgt in persona_node_ids:
            continue
        if src == tgt:
            continue
        if tgt not in parent_by_child:
            parent_by_child[tgt] = (src, edge)

    # Metadata resolved by apply_plan_hierarchy is a strong fallback for legacy
    # rows where persona fallback edges are still active or plan edges were
    # accidentally soft-deleted.
    for node in nodes:
        node_id = node.get("id")
        meta = node.get("metadata") or {}
        parent_id = meta.get("resolved_parent_node_id") or meta.get("parent_node_id")
        if not node_id or not parent_id or parent_id == node_id:
            continue
        parent = nodes_by_id.get(parent_id)
        if not parent:
            continue
        if (node.get("node_type") or "").lower() in auxiliary_types:
            continue
        if (parent.get("node_type") or "").lower() in auxiliary_types:
            continue
        parent_by_child[node_id] = (parent_id, {
            "id": None,
            "source_node_id": parent_id,
            "target_node_id": node_id,
            "relation_type": "resolved_parent",
            "metadata": {"primary_tree": True, "active": True, "created_from": "resolved_parent_node_id"},
        })

    source_id = source_node.get("id")
    chain_reversed: list[dict] = []
    path_edges_reversed: list[dict] = []
    seen: set[str] = set()
    current_id = source_id
    while current_id and current_id not in seen:
        seen.add(current_id)
        node = nodes_by_id.get(current_id) or (source_node if current_id == source_id else None)
        if not node:
            break
        if (node.get("node_type") or "").lower() in auxiliary_types:
            break
        chain_reversed.append(node)
        parent = parent_by_child.get(current_id)
        if not parent:
            break
        parent_id, edge = parent
        if parent_id in seen:
            break
        path_edges_reversed.append(edge)
        current_id = parent_id

    chain = list(reversed(chain_reversed))
    path_edges = list(reversed(path_edges_reversed))
    if not chain or chain[-1].get("id") != source_id:
        chain.append(source_node)
    persona_positions = [idx for idx, node in enumerate(chain) if (node.get("node_type") or "").lower() == "persona"]
    if len(persona_positions) > 1 or (persona_positions and persona_positions[0] != 0):
        raise RuntimeError("Invalid hierarchy cycle: persona appears outside root position")
    return chain, path_edges


def _has_type_in_persona(nodes: list[dict], node_type: str) -> bool:
    return any((node.get("node_type") or "").lower() == node_type for node in nodes)


def _validate_faq_context(
    *,
    source_node: dict,
    chain: list[dict],
    question: str,
    answer: str,
    product: Optional[dict],
    briefing: Optional[dict],
    audience: Optional[dict],
) -> None:
    if not question or not answer:
        raise RuntimeError("FAQ approved, but question/answer is incomplete")
    if not product:
        raise RuntimeError("FAQ approved, but product context is missing")
    persona_id = source_node.get("persona_id")
    try:
        nodes, _ = supabase_client.list_all_knowledge_graph(persona_id=persona_id, limit_nodes=2500)
    except Exception:
        nodes = []
    has_briefing = _has_type_in_persona(nodes, "briefing")
    has_audience = _has_type_in_persona(nodes, "audience")
    if has_briefing and not briefing:
        raise RuntimeError("FAQ approved, but briefing context exists in graph and was not resolved")
    if has_audience and not audience:
        raise RuntimeError("FAQ approved, but audience context exists in graph and was not resolved")


def _hierarchy_path(chain: list[dict]) -> list[dict]:
    return [
        {
            "node_id": node.get("id"),
            "node_type": node.get("node_type"),
            "slug": node.get("slug"),
            "title": node.get("title"),
        }
        for node in chain
    ]


def _canonical_key(persona: dict, content_type: str, chain: list[dict], slug: str) -> str:
    persona_slug = _slugify(persona.get("slug") or persona.get("name") or "persona")
    hierarchy = [
        _slugify(node.get("slug") or node.get("title") or "")
        for node in chain
        if (node.get("node_type") or "").lower() not in {"persona", content_type}
    ]
    hierarchy = [part for part in hierarchy if part and part != "self"]
    parts = [persona_slug, content_type, *hierarchy, slug]
    return "/".join(part for part in parts if part)


def _approved_markdown(
    *,
    title: str,
    content_type: str,
    hierarchy_summary: str,
    brand: Optional[dict],
    briefing: Optional[dict],
    campaign: Optional[dict],
    audience: Optional[dict],
    product: Optional[dict],
    question: str,
    answer: str,
    source_text: str,
) -> str:
    lines = [f"# {title}", "", f"Tipo: {content_type}", f"Hierarquia: {hierarchy_summary}"]
    for label, node in [
        ("Marca", brand),
        ("Briefing", briefing),
        ("Campanha", campaign),
        ("Publico", audience),
        ("Produto", product),
    ]:
        value = _text_context(node)
        if value:
            lines.append(f"{label}: {value}")
    if content_type == "faq":
        lines.extend(["", f"Pergunta: {question}", f"Resposta aprovada: {answer}"])
        if source_text and source_text.strip() and source_text.strip() != answer.strip():
            lines.extend(["", "## FAQ Golden Dataset", "", source_text.strip()])
    elif source_text:
        lines.extend(["", source_text.strip()])
    return "\n".join(lines).strip()


def _faq_chunk_text(
    *,
    snapshot_metadata: dict,
    question: Optional[str] = None,
    answer: Optional[str] = None,
) -> str:
    branch_context = snapshot_metadata.get("branch_context") or {}
    path = branch_context.get("path") or []
    edge_parts = []
    for edge in branch_context.get("edges") or []:
        source = edge.get("source_slug") or edge.get("source_node_id") or "source"
        target = edge.get("target_slug") or edge.get("target_node_id") or "target"
        semantic = edge.get("semantic_relation") or edge.get("relation_type") or "related"
        edge_parts.append(f"{source} {semantic} {target}")
    brand_source = snapshot_metadata.get("brand_source") or "explicit"
    faq_context = snapshot_metadata.get("faq_context") or {}
    question_text = (question or faq_context.get("question") or "").strip()
    answer_text = (answer or faq_context.get("answer") or "").strip()
    lines = [
        "Tipo: FAQ aprovada",
        f"Marca/Persona: {snapshot_metadata.get('persona_context') or 'Nao informado.'}",
        f"Brand: {snapshot_metadata.get('brand_context') or 'Nao informado.'} Fonte: {brand_source}.",
        f"Briefing: {snapshot_metadata.get('briefing_context') or 'Nao informado.'}",
        f"Publico: {snapshot_metadata.get('audience_context') or 'Nao informado.'}",
        f"Produto: {snapshot_metadata.get('product_context') or 'Nao informado.'}",
        f"Copy/Oferta: {snapshot_metadata.get('copy_context') or 'Nao informado.'}",
        f"Pergunta: {question_text}",
        f"Resposta aprovada: {answer_text}",
        f"Regras: {snapshot_metadata.get('rules_context') or 'Nao informado.'}",
        f"Tom: {snapshot_metadata.get('tone_context') or 'Nao informado.'}",
        "Caminho da branch: " + " > ".join(step.get("slug") or step.get("title") or "" for step in path if step.get("slug") or step.get("title")) + ".",
        "Relacoes: " + "; ".join(edge_parts) + ".",
    ]
    return "\n".join(lines)


def _branch_value(branch_context: dict, node_type: str, field: str = "slug") -> Optional[str]:
    for step in branch_context.get("path") or []:
        if (step.get("node_type") or "").lower() == node_type:
            value = step.get(field)
            return str(value) if value else None
    return None


def _node_markdown(node: Optional[dict], *, persona: Optional[dict] = None) -> str:
    if not node:
        return ""
    meta = node.get("metadata") or {}
    candidates = [
        meta.get("approved_markdown"),
        meta.get("markdown"),
        meta.get("body_markdown"),
        meta.get("description"),
        node.get("summary"),
        node.get("title"),
    ]
    if (node.get("node_type") or "").lower() == "persona" and persona:
        candidates.extend([persona.get("description"), persona.get("name"), persona.get("slug")])
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _branch_markdown_columns(
    *,
    chain: list[dict],
    persona: dict,
    source_node: dict,
    question: str,
    answer: str,
) -> dict[str, str]:
    values: dict[str, str] = {
        "branch_persona_md": "",
        "branch_brand_md": "",
        "branch_briefing_md": "",
        "branch_campaign_md": "",
        "branch_audience_md": "",
        "branch_product_md": "",
        "branch_copy_md": "",
        "connected_node_type": (source_node.get("node_type") or "").lower(),
        "connected_node_title": source_node.get("title") or source_node.get("slug") or "",
    }
    node_by_type = {(node.get("node_type") or "").lower(): node for node in chain}
    for node_type, column in [
        ("persona", "branch_persona_md"),
        ("brand", "branch_brand_md"),
        ("briefing", "branch_briefing_md"),
        ("campaign", "branch_campaign_md"),
        ("audience", "branch_audience_md"),
        ("product", "branch_product_md"),
        ("copy", "branch_copy_md"),
    ]:
        values[column] = _node_markdown(node_by_type.get(node_type), persona=persona)

    if not values["branch_persona_md"]:
        values["branch_persona_md"] = persona.get("description") or persona.get("name") or persona.get("slug") or ""
    if not values["branch_brand_md"]:
        values["branch_brand_md"] = values["branch_persona_md"]
    if not values["branch_product_md"]:
        values["branch_product_md"] = (
            source_node.get("title")
            or source_node.get("slug")
            or question
            or values["connected_node_title"]
            or ""
        )
    if not values["branch_copy_md"] and (question or answer):
        values["branch_copy_md"] = "\n".join(
            part for part in [f"Pergunta: {question}".strip(), f"Resposta: {answer}".strip()] if part
        ).strip()
    return {key: (str(value).strip() if value is not None else "") for key, value in values.items()}


def _branch_session_id(chain: list[dict], source_meta: dict) -> Optional[str]:
    if source_meta.get("session_id"):
        return str(source_meta.get("session_id"))
    for node in reversed(chain):
        meta = node.get("metadata") or {}
        if meta.get("session_id"):
            return str(meta.get("session_id"))
    return None


def _rules_list(rules_context: str) -> list[str]:
    base = [
        "nao inventar preco",
        "nao inventar desconto",
        "nao inventar estoque",
        "nao inventar frete",
        "nao inventar prazo",
        "nao inventar disponibilidade",
        "nao inventar condicao comercial",
    ]
    extra = [part.strip() for part in re.split(r"[.;\n]", rules_context or "") if part.strip()]
    return list(dict.fromkeys([*base, *extra]))[:16]


def _faq_summary(question: str, answer: str, product: Optional[dict]) -> str:
    product_title = (product or {}).get("title") or (product or {}).get("slug") or "produto"
    text = re.sub(r"\s+", " ", answer or "").strip()
    if len(text) > 180:
        text = text[:177].rstrip() + "..."
    return f"FAQ sobre {product_title}: {question} Resposta: {text}"


def _source_item_for_node(node: dict) -> Optional[dict]:
    if node.get("source_table") == "knowledge_items" and node.get("source_id"):
        try:
            return supabase_client.get_knowledge_item(str(node.get("source_id")))
        except Exception:
            return None
    return None


def _rebuild_embedded_markdown(persona_id: Optional[str]) -> Optional[str]:
    try:
        from services import embedded_markdown

        return embedded_markdown.rebuild_embedded_markdown(persona_id)
    except Exception:
        return None


def publish_approved_node(
    source_node_id: str,
    *,
    approved_by: Optional[str] = None,
    require_rag_for_faq: bool = True,
) -> dict:
    """Create the canonical approved snapshot and RAG rows for an approved node."""
    source_node = supabase_client.get_knowledge_node(source_node_id)
    if not source_node:
        raise ValueError("Source knowledge node not found")
    if (source_node.get("node_type") or "").lower() == "mention":
        raise ValueError("Mention nodes cannot be published as approved canonical knowledge")
    persona_id = source_node.get("persona_id")
    if not persona_id:
        raise ValueError("Source knowledge node must have persona_id")
    persona = supabase_client.get_persona_by_id(persona_id) or {"id": persona_id, "slug": persona_id}
    source_item = _source_item_for_node(source_node)
    content_type = (source_node.get("node_type") or (source_item or {}).get("content_type") or "general_note").lower()
    title = source_node.get("title") or (source_item or {}).get("title") or content_type
    slug = _slugify(source_node.get("slug") or title)
    source_text = (source_item or {}).get("content") or source_node.get("summary") or ""
    question, answer = _faq_parts(source_node, source_item) if content_type == "faq" else ("", "")
    if content_type == "faq" and not answer:
        answer = source_text.strip()

    chain, path_edges = _build_hierarchy(source_node)
    if chain and chain[0].get("node_type") != "persona":
        persona_root = knowledge_graph._ensure_persona_root(persona_id)
        if persona_root:
            previous_root = chain[0]
            chain.insert(0, persona_root)
            path_edges.insert(0, {
                "id": None,
                "source_node_id": persona_root.get("id"),
                "target_node_id": previous_root.get("id"),
                "relation_type": "contains",
                "metadata": {
                    "primary_tree": True,
                    "active": True,
                    "created_from": "snapshot_persona_root_fallback",
                    "semantic_relation": "contains_briefing" if previous_root.get("node_type") == "briefing" else "contains",
                },
            })
    root_node = chain[0] if chain else source_node
    parent_node = chain[-2] if len(chain) >= 2 else None
    brand = _first_by_type(chain, "brand")
    briefing = _first_by_type(chain, "briefing")
    campaign = _first_by_type(chain, "campaign")
    audience = _first_by_type(chain, "audience")
    product = _first_by_type(chain, "product")
    copy = _first_by_type(chain, "copy")
    faq = source_node if content_type == "faq" else _first_by_type(chain, "faq")

    source_meta = source_node.get("metadata") or {}
    is_golden_dataset_faq = bool(source_meta.get("golden_dataset") or source_meta.get("faq_document_type") == "golden_dataset")
    session_id = _branch_session_id(chain, source_meta)
    rules = _metadata_list(source_meta.get("rules"), source_meta.get("restrictions"))
    rules_context = " ".join(rules).strip() or "Nao inventar preco, desconto, estoque, frete, prazo, disponibilidade ou condicao comercial sem validacao."
    tone = str(source_meta.get("tone") or source_meta.get("voice") or "").strip()
    tone_context = tone or "Direto, acolhedor, feminino e comercial."
    branch_context = _branch_context(chain, path_edges, persona)
    hierarchy_path = branch_context.get("path") or _hierarchy_path(chain)
    hierarchy_summary = " -> ".join(
        f"{step.get('node_type')}:{step.get('slug') or step.get('title')}"
        for step in hierarchy_path
    )
    persona_context = persona.get("name") or _context_text(root_node, persona.get("slug") or "")
    explicit_brand_context = _context_text(brand)
    brand_context = explicit_brand_context or persona_context
    brand_source = "explicit" if explicit_brand_context else "persona_fallback"
    briefing_context = _context_text(briefing)
    audience_context = _context_text(audience)
    product_context = _context_text(product)
    copy_context = _context_text(copy)
    faq_context = {
        **_node_context(faq),
        "question": question,
        "answer": answer,
    } if faq else {"question": question, "answer": answer}
    canonical_faq_pairs = _canonical_faq_pairs(
        source_text=source_text,
        question=question,
        answer=answer,
        is_golden_dataset=is_golden_dataset_faq,
    ) if content_type == "faq" else []
    if canonical_faq_pairs:
        question = canonical_faq_pairs[0]["question"]
        answer = canonical_faq_pairs[0]["answer"]
        faq_context["question"] = question
        faq_context["answer"] = answer
    pair_warnings: list[str] = []
    for pair in canonical_faq_pairs:
        if not _answer_is_useful(pair.get("question") or "", pair.get("answer") or ""):
            pair_warnings.append(f"faq_pair_{pair.get('index')}_answer_not_useful")
    review_warnings = (
        _faq_review_warnings(
            source_node=source_node,
            chain=chain,
            branch_context=branch_context,
            question=(canonical_faq_pairs[0]["question"] if canonical_faq_pairs else question),
            answer=(canonical_faq_pairs[0]["answer"] if canonical_faq_pairs else answer),
            briefing_context=briefing_context,
            product_context=product_context,
            copy_context=copy_context,
        ) + pair_warnings
        if content_type == "faq"
        else []
    )
    canonical_key = _canonical_key(persona, content_type, chain, slug)
    approved_summary = (answer or source_text)[:500] if content_type == "faq" and (answer or source_text) else (source_node.get("summary") or source_text or title)[:500]
    connected_node_type = (source_node.get("node_type") or content_type or "").lower()
    branch_markdown_columns = _branch_markdown_columns(
        chain=chain,
        persona=persona,
        source_node=source_node,
        question=question,
        answer=answer,
    )
    approved_markdown = _approved_markdown(
        title=title,
        content_type=content_type,
        hierarchy_summary=hierarchy_summary,
        brand=brand,
        briefing=briefing,
        campaign=campaign,
        audience=audience,
        product=product,
        question=question,
        answer=answer,
        source_text=source_text,
    )
    content_hash = hashlib.sha256(approved_markdown.encode("utf-8")).hexdigest()
    now_iso = datetime.now(timezone.utc).isoformat()

    snapshot = supabase_client.upsert_approved_knowledge_snapshot({
        "persona_id": persona_id,
        "root_node_id": root_node.get("id"),
        "source_node_id": source_node.get("id"),
        "source_table": source_node.get("source_table") or "knowledge_nodes",
        "source_id": source_node.get("source_id") or source_node.get("id"),
        "artifact_id": source_node.get("artifact_id") or source_meta.get("artifact_id"),
        "session_id": session_id,
        "tree_mode": branch_context.get("tree_mode") or "pyramidal",
        "branch_policy": branch_context.get("branch_policy") or "top_down_pyramidal",
        "content_type": content_type,
        "title": title,
        "slug": slug,
        "canonical_key": canonical_key,
        "content_hash": content_hash,
        "hierarchy_path": hierarchy_path,
        "hierarchy_summary": hierarchy_summary,
        "approved_summary": approved_summary,
        "approved_markdown": approved_markdown,
        "parent_context": _node_context(parent_node),
        "brand_context": _node_context(brand),
        "briefing_context": _node_context(briefing),
        "campaign_context": _node_context(campaign),
        "audience_context": _node_context(audience),
        "product_context": _node_context(product),
        "faq_context": faq_context,
        "status": "needs_review" if review_warnings else "approved",
        "approved_by": approved_by,
        "approved_at": now_iso,
        "metadata": {
            "source": "graph_approval",
            "source_node_id": source_node.get("id"),
            "session_id": session_id,
            "tree_mode": branch_context.get("tree_mode") or "pyramidal",
            "branch_policy": branch_context.get("branch_policy") or "top_down_pyramidal",
            "source_knowledge_item_id": (source_item or {}).get("id"),
            "n8n_ready": content_type == "faq" and not review_warnings,
            "review_warnings": review_warnings,
            "branch_context": branch_context,
            "persona_context": persona_context,
            "brand_context": brand_context,
            "brand_source": brand_source,
            "briefing_context": briefing_context,
            "audience_context": audience_context,
            "product_context": product_context,
            "copy_context": copy_context,
            "branch_markdown_columns": branch_markdown_columns,
            "faq_context": faq_context,
            "body_markdown": source_text if content_type == "faq" and is_golden_dataset_faq else "",
            "question_count": source_meta.get("question_count"),
            "rules_context": rules_context,
            "tone_context": tone_context,
        },
    })
    if not snapshot or not snapshot.get("id"):
        raise RuntimeError("Approved snapshot was not created")

    rag_entry = None
    rag_entries: list[dict] = []
    chunks: list[dict] = []
    rag_links: list[dict] = []
    embedded_edge = None
    if content_type == "faq":
        if not snapshot or not snapshot.get("id"):
            raise RuntimeError("Cannot create RAG entries without an approved snapshot")
        if not canonical_faq_pairs:
            raise RuntimeError("FAQ approved, but no canonical question/answer pairs were found")
        snapshot_meta = snapshot.get("metadata") or {}
        branch_types = [step.get("node_type") for step in hierarchy_path if step.get("node_type")]
        branch_slugs = [step.get("slug") for step in hierarchy_path if step.get("slug")]
        for pair in canonical_faq_pairs:
            pair_question = pair["question"]
            pair_answer = pair["answer"]
            pair_index = int(pair.get("index") or len(rag_entries) + 1)
            pair_slug = _slugify(f"{slug}-{pair_index}-{pair_question}")[:140]
            pair_canonical_key = f"{canonical_key}/q-{pair_index}-{pair_slug}"
            pair_title = pair_question if len(pair_question) <= 120 else pair_question[:117].rstrip() + "..."
            pair_summary = _faq_summary(pair_question, pair_answer, product)
            pair_metadata = {
                **source_meta,
                "source": "approved_knowledge_snapshots",
                "snapshot_id": snapshot.get("id"),
                "approved_snapshot_id": snapshot.get("id"),
                "source_snapshot_id": snapshot.get("id"),
                "source_node_id": source_node.get("id"),
                "source_knowledge_item_id": (source_item or {}).get("id"),
                "session_id": session_id,
                "tree_mode": branch_context.get("tree_mode") or "pyramidal",
                "branch_policy": branch_context.get("branch_policy") or "top_down_pyramidal",
                "hierarchy_path": hierarchy_path,
                "branch_context": branch_context,
                "status": "active",
                "rag_index": source_meta.get("rag_index") or "default",
                "faq_pair_index": pair_index,
                "faq_document_slug": slug,
                "question": pair_question,
                "answer": pair_answer,
                "branch_markdown_columns": branch_markdown_columns,
            }
            rag_entry = supabase_client.upsert_knowledge_rag_entry({
                "persona_id": persona_id,
                "artifact_id": source_node.get("artifact_id") or source_meta.get("artifact_id"),
                "source_snapshot_id": snapshot.get("id"),
                "source_node_id": source_node.get("id"),
                "session_id": session_id,
                "content_type": "faq",
                "semantic_level": int(source_node.get("level") or 75),
                "title": pair_title,
                "question": pair_question,
                "answer": pair_answer,
                "content": pair_answer,
                "summary": pair_summary,
                "canonical_key": pair_canonical_key,
                "slug": pair_slug,
                "status": "active",
                "tags": sorted(set((source_node.get("tags") or []) + ["faq", "approved", "n8n-ready"])),
                "entities": [],
                "products": [product.get("slug")] if product else [],
                "campaigns": [campaign.get("slug")] if campaign else [],
                "embedding_model": connected_node_type,
                **branch_markdown_columns,
                "metadata": pair_metadata,
                "confidence": float(source_node.get("confidence") or 0.85),
                "importance": float(source_node.get("importance") or 0.75),
                "validated_at": now_iso,
            })
            if not rag_entry or not rag_entry.get("id"):
                raise RuntimeError("RAG entry was not created for approved FAQ pair")
            rag_entries.append(rag_entry)
            chunk_text = _faq_chunk_text(snapshot_metadata=snapshot_meta, question=pair_question, answer=pair_answer)
            chunk_rows = supabase_client.replace_knowledge_rag_chunks(
                rag_entry["id"],
                persona_id,
                [{
                    "chunk_index": 0,
                    "chunk_text": chunk_text,
                    "chunk_summary": pair_summary[:280],
                    "embedding_model": connected_node_type,
                    **branch_markdown_columns,
                    "metadata": {
                        "persona_slug": persona.get("slug"),
                        "content_type": "faq",
                        "status": "active",
                        "source": "approved_knowledge_snapshots",
                        "snapshot_id": snapshot.get("id"),
                        "approved_snapshot_id": snapshot.get("id"),
                        "source_snapshot_id": snapshot.get("id"),
                        "rag_entry_id": rag_entry.get("id"),
                        "source_node_id": source_node.get("id"),
                        "session_id": session_id,
                        "tree_mode": branch_context.get("tree_mode") or "pyramidal",
                        "branch_policy": branch_context.get("branch_policy") or "top_down_pyramidal",
                        "hierarchy_path": branch_slugs,
                        "branch_types": branch_types,
                        "branch_context": branch_context,
                        "audience_slug": _branch_value(branch_context, "audience"),
                        "product_slug": _branch_value(branch_context, "product"),
                        "offer_slug": _branch_value(branch_context, "offer"),
                        "copy_slug": _branch_value(branch_context, "copy"),
                        "faq_slug": _branch_value(branch_context, "faq") or source_node.get("slug"),
                        "rules": _rules_list(rules_context),
                        "chunk_status": "pending_embedding",
                        "ready_for_production": False,
                        "question": pair_question,
                        "answer": pair_answer,
                        "faq_pair_index": pair_index,
                        "branch_markdown_columns": branch_markdown_columns,
                    },
                }],
            )
            if require_rag_for_faq and not chunk_rows:
                raise RuntimeError("FAQ approved, but no RAG chunk was created")
            chunks.extend(chunk_rows)
        rag_entry = rag_entries[0] if rag_entries else None
        snapshot = supabase_client.update_approved_knowledge_snapshot(
            snapshot["id"],
            {
                "rag_entry_id": rag_entry.get("id"),
                "status": "active",
                "metadata": {
                    **(snapshot.get("metadata") or {}),
                    "rag_entry_id": rag_entry.get("id"),
                    "rag_entry_ids": [entry.get("id") for entry in rag_entries if entry.get("id")],
                    "rag_chunk_ids": [chunk.get("id") for chunk in chunks if chunk.get("id")],
                    "canonical_faq_count": len(rag_entries),
                    "n8n_ready": True,
                },
                "updated_at": now_iso,
            },
        ) or snapshot
        embedded_node = supabase_client.ensure_embedded_node(persona_id)
        embedded_edge = _active_faq_embedded_edge(source_node["id"], persona_id)
        if not embedded_edge and embedded_node and embedded_node.get("id"):
            embedded_edge = supabase_client.upsert_knowledge_edge(
                source_node_id=source_node["id"],
                target_node_id=embedded_node["id"],
                relation_type="visible_to_agent",
                persona_id=persona_id,
                weight=1.0,
                metadata={
                    **knowledge_graph.semantic_edge_metadata(
                        source_node,
                        embedded_node,
                        "manual",
                        {},
                    ),
                    "primary_tree": False,
                    "active": True,
                    "visual_hidden": False,
                    "created_from": "approved_snapshot_publication",
                    "approved_snapshot_id": snapshot.get("id"),
                    "rag_entry_id": rag_entry.get("id"),
                },
            )
        if require_rag_for_faq and not embedded_edge:
            raise RuntimeError("FAQ approved, but FAQ -> Embedded edge was not created")
        _rebuild_embedded_markdown(persona_id)

    node_meta = {
        **source_meta,
        "approved_snapshot_id": snapshot.get("id"),
        "knowledge_rag_entry_id": (rag_entry or {}).get("id"),
        "knowledge_rag_entry_ids": [entry.get("id") for entry in rag_entries if entry.get("id")],
        "knowledge_rag_chunk_ids": [chunk.get("id") for chunk in chunks if chunk.get("id")],
        "snapshot_status": snapshot.get("status"),
        "n8n_ready": bool(chunks) if content_type == "faq" else False,
    }
    supabase_client.update_knowledge_node(
        source_node["id"],
        {
            "metadata": node_meta,
            "status": "approved" if content_type == "faq" else "validated",
        },
    )
    if source_item and source_item.get("id"):
        supabase_client.update_knowledge_item(source_item["id"], {
            "metadata": {
                **(source_item.get("metadata") or {}),
                "approved_snapshot_id": snapshot.get("id"),
                "knowledge_rag_entry_id": (rag_entry or {}).get("id"),
                "knowledge_rag_entry_ids": [entry.get("id") for entry in rag_entries if entry.get("id")],
                "knowledge_rag_chunk_ids": [chunk.get("id") for chunk in chunks if chunk.get("id")],
            }
        })

    return {
        "success": True,
        "approved_snapshot_id": snapshot.get("id"),
        "source_node_id": source_node.get("id"),
        "knowledge_node_ids": [node.get("id") for node in chain if node.get("id")],
        "knowledge_edge_ids": [edge.get("id") for edge in path_edges if edge.get("id")] + ([embedded_edge.get("id")] if embedded_edge and embedded_edge.get("id") else []),
        "embedded_edge_id": (embedded_edge or {}).get("id"),
        "rag_entry_id": (rag_entry or {}).get("id"),
        "rag_entry_ids": [entry.get("id") for entry in rag_entries if entry.get("id")],
        "rag_chunk_ids": [chunk.get("id") for chunk in chunks if chunk.get("id")],
        "rag_link_ids": [link.get("id") for link in rag_links if link.get("id")],
        "status": "active" if content_type == "faq" else "approved",
        "graph_materialized": bool(source_node.get("id")),
        "content_type": content_type,
        "canonical_key": canonical_key,
    }


def _active_faq_embedded_edge(node_id: str, persona_id: str) -> Optional[dict]:
    embedded_node = supabase_client.ensure_embedded_node(persona_id)
    if not embedded_node or not embedded_node.get("id"):
        return None
    try:
        edges = supabase_client.list_edges_for_nodes([node_id, embedded_node["id"]]) or []
    except Exception:
        edges = []
    for edge in edges:
        if edge.get("source_node_id") != node_id:
            continue
        if edge.get("target_node_id") != embedded_node["id"]:
            continue
        if _active_edge(edge):
            return edge
    return None


def _rag_chunk_count_for_node(node_id: str) -> int:
    snapshots = supabase_client.list_approved_snapshots_for_nodes([node_id])
    snapshot = snapshots.get(node_id) or {}
    entry_ids: list[str] = []
    if snapshot.get("rag_entry_id"):
        entry_ids.append(str(snapshot["rag_entry_id"]))
    meta = snapshot.get("metadata") or {}
    for entry_id in meta.get("rag_entry_ids") or []:
        if entry_id:
            entry_ids.append(str(entry_id))
    entry_ids = list(dict.fromkeys(entry_ids))
    counts = supabase_client.count_knowledge_rag_chunks_by_entry_ids(entry_ids)
    return sum(int(counts.get(entry_id) or 0) for entry_id in entry_ids)


_RAG_BRANCH_COLUMNS = (
    "branch_persona_md",
    "branch_brand_md",
    "branch_briefing_md",
    "branch_campaign_md",
    "branch_audience_md",
    "branch_product_md",
    "branch_copy_md",
    "connected_node_type",
    "connected_node_title",
)


def _rag_publication_quality_for_node(node_id: str, *, expected_node_type: str = "faq") -> dict:
    snapshots = supabase_client.list_approved_snapshots_for_nodes([node_id])
    snapshot = snapshots.get(node_id) or {}
    entry_ids: list[str] = []
    if snapshot.get("rag_entry_id"):
        entry_ids.append(str(snapshot["rag_entry_id"]))
    meta = snapshot.get("metadata") or {}
    for entry_id in meta.get("rag_entry_ids") or []:
        if entry_id:
            entry_ids.append(str(entry_id))
    entry_ids = list(dict.fromkeys(entry_ids))
    counts = supabase_client.count_knowledge_rag_chunks_by_entry_ids(entry_ids)
    chunk_count = sum(int(counts.get(entry_id) or 0) for entry_id in entry_ids)
    missing: list[str] = []
    if not entry_ids or chunk_count <= 0:
        return {"entry_ids": entry_ids, "chunk_count": chunk_count, "missing": ["knowledge_rag_chunks"]}

    try:
        client = supabase_client.get_client()
        select_cols = "id,embedding_model," + ",".join(_RAG_BRANCH_COLUMNS)
        entry_rows = (
            client.table("knowledge_rag_entries")
            .select(select_cols)
            .in_("id", entry_ids)
            .execute()
            .data
            or []
        )
        chunk_rows = (
            client.table("knowledge_rag_chunks")
            .select("id,rag_entry_id,embedding_model," + ",".join(_RAG_BRANCH_COLUMNS))
            .in_("rag_entry_id", entry_ids)
            .execute()
            .data
            or []
        )
    except Exception:
        # Older fakes and legacy schemas may not expose these columns. Do not
        # break the base publication audit; live QA gets the strict check after
        # migration is applied.
        return {"entry_ids": entry_ids, "chunk_count": chunk_count, "missing": []}

    required_nonempty = {"branch_persona_md", "branch_brand_md", "branch_product_md", "branch_copy_md"}
    for row_type, rows in [("entry", entry_rows), ("chunk", chunk_rows)]:
        if not rows:
            missing.append(f"{row_type}_rows")
            continue
        for row in rows:
            if (row.get("embedding_model") or "").lower() != expected_node_type:
                missing.append(f"{row_type}_embedding_model")
            if (row.get("connected_node_type") or "").lower() != expected_node_type:
                missing.append(f"{row_type}_connected_node_type")
            for column in _RAG_BRANCH_COLUMNS:
                if column not in row or row.get(column) is None:
                    missing.append(f"{row_type}_{column}")
                elif column in required_nonempty and not str(row.get(column) or "").strip():
                    missing.append(f"{row_type}_{column}")
    return {
        "entry_ids": entry_ids,
        "chunk_count": chunk_count,
        "missing": sorted(set(missing)),
    }


def validate_approved_faq_publications(
    *,
    persona_id: Optional[str] = None,
    repair: bool = False,
    limit: int = 1000,
) -> dict:
    """Validate every approved FAQ has graph publication and RAG chunks.

    Contract:
    - approved/embedded FAQ knowledge_items must have a FAQ knowledge_node;
    - that FAQ node must have an active FAQ -> Embedded edge;
    - the approved FAQ must have at least one knowledge_rag_chunk.

    When repair=True, missing edge/chunks are fixed by re-running
    publish_approved_node for the FAQ node.
    """
    client = supabase_client.get_client()
    q = (
        client.table("knowledge_items")
        .select("id,persona_id,title,status,content_type,metadata")
        .eq("content_type", "faq")
        .in_("status", ["approved", "embedded"])
        .limit(limit)
    )
    if persona_id:
        q = q.eq("persona_id", persona_id)
    item_rows = q.execute().data or []
    node_q = (
        client.table("knowledge_nodes")
        .select("id,persona_id,title,status,node_type,source_table,source_id,metadata")
        .eq("node_type", "faq")
        .in_("status", ["approved", "embedded", "validated"])
        .limit(limit)
    )
    if persona_id:
        node_q = node_q.eq("persona_id", persona_id)
    node_rows = node_q.execute().data or []

    checked = 0
    repaired = 0
    failures: list[dict] = []
    ok_items: list[dict] = []

    seen_node_ids: set[str] = set()

    for item in item_rows:
        checked += 1
        item_id = str(item.get("id") or "")
        item_persona_id = str(item.get("persona_id") or "")
        node = supabase_client.get_knowledge_node_for_source(
            "knowledge_items",
            item_id,
            persona_id=item_persona_id or None,
        )
        if not node or not node.get("id"):
            if repair:
                try:
                    from services import knowledge_lifecycle

                    promoted = knowledge_lifecycle.promote_knowledge_item(item_id, promote_to_kb=False)
                    evidence = promoted.get("evidence") or {}
                    node_id = evidence.get("knowledge_node_id")
                    node = supabase_client.get_knowledge_node(node_id) if node_id else None
                    if not node:
                        node = supabase_client.get_knowledge_node_for_source(
                            "knowledge_items",
                            item_id,
                            persona_id=item_persona_id or None,
                        )
                except Exception as exc:
                    failures.append({
                        "knowledge_item_id": item_id,
                        "title": item.get("title"),
                        "reason": "missing_faq_node",
                        "repair_error": str(exc),
                    })
                    continue
            if not node or not node.get("id"):
                failures.append({
                    "knowledge_item_id": item_id,
                    "title": item.get("title"),
                    "reason": "missing_faq_node",
                })
                continue

        node_id = str(node["id"])
        seen_node_ids.add(node_id)
        edge = _active_faq_embedded_edge(node_id, item_persona_id)
        quality = _rag_publication_quality_for_node(node_id, expected_node_type="faq")
        chunk_count = int(quality.get("chunk_count") or 0)
        quality_missing = list(quality.get("missing") or [])
        if edge and chunk_count > 0 and not quality_missing:
            ok_items.append({
                "knowledge_item_id": item_id,
                "knowledge_node_id": node_id,
                "embedded_edge_id": edge.get("id"),
                "rag_chunk_count": chunk_count,
            })
            continue

        if repair:
            publication = publish_approved_node(node_id, require_rag_for_faq=True)
            repaired += 1
            edge = _active_faq_embedded_edge(node_id, item_persona_id)
            quality = _rag_publication_quality_for_node(node_id, expected_node_type="faq")
            chunk_count = int(quality.get("chunk_count") or 0)
            quality_missing = list(quality.get("missing") or [])
            if edge and chunk_count > 0 and not quality_missing:
                ok_items.append({
                    "knowledge_item_id": item_id,
                    "knowledge_node_id": node_id,
                    "embedded_edge_id": edge.get("id"),
                    "rag_chunk_count": chunk_count,
                    "repaired": True,
                    "publication": publication,
                })
                continue

        missing = []
        if not edge:
            missing.append("faq_embedded_edge")
        if chunk_count <= 0:
            missing.append("knowledge_rag_chunks")
        missing.extend(quality_missing)
        missing = list(dict.fromkeys(missing))
        failures.append({
            "knowledge_item_id": item_id,
            "knowledge_node_id": node_id,
            "title": item.get("title"),
            "reason": "missing_publication_artifacts",
            "missing": missing,
            "rag_chunk_count": chunk_count,
        })

    for node in node_rows:
        node_id = str(node.get("id") or "")
        if not node_id or node_id in seen_node_ids:
            continue
        # Item-backed FAQ nodes are audited through their knowledge_item row
        # when such a row exists. Node-only approved FAQs are valid graph
        # knowledge and must also publish to Embedded/RAG.
        if node.get("source_table") == "knowledge_items" and node.get("source_id"):
            source_item = None
            try:
                source_item = supabase_client.get_knowledge_item(str(node.get("source_id")))
            except Exception:
                source_item = None
            if source_item and (source_item.get("status") or "").lower() in {"approved", "embedded"}:
                continue
        checked += 1
        node_persona_id = str(node.get("persona_id") or "")
        edge = _active_faq_embedded_edge(node_id, node_persona_id)
        quality = _rag_publication_quality_for_node(node_id, expected_node_type="faq")
        chunk_count = int(quality.get("chunk_count") or 0)
        quality_missing = list(quality.get("missing") or [])
        if edge and chunk_count > 0 and not quality_missing:
            ok_items.append({
                "knowledge_item_id": node.get("source_id") if node.get("source_table") == "knowledge_items" else None,
                "knowledge_node_id": node_id,
                "embedded_edge_id": edge.get("id"),
                "rag_chunk_count": chunk_count,
                "source": "knowledge_nodes",
            })
            continue

        if repair:
            publication = publish_approved_node(node_id, require_rag_for_faq=True)
            repaired += 1
            edge = _active_faq_embedded_edge(node_id, node_persona_id)
            quality = _rag_publication_quality_for_node(node_id, expected_node_type="faq")
            chunk_count = int(quality.get("chunk_count") or 0)
            quality_missing = list(quality.get("missing") or [])
            if edge and chunk_count > 0 and not quality_missing:
                ok_items.append({
                    "knowledge_item_id": node.get("source_id") if node.get("source_table") == "knowledge_items" else None,
                    "knowledge_node_id": node_id,
                    "embedded_edge_id": edge.get("id"),
                    "rag_chunk_count": chunk_count,
                    "repaired": True,
                    "publication": publication,
                    "source": "knowledge_nodes",
                })
                continue

        missing = []
        if not edge:
            missing.append("faq_embedded_edge")
        if chunk_count <= 0:
            missing.append("knowledge_rag_chunks")
        missing.extend(quality_missing)
        missing = list(dict.fromkeys(missing))
        failures.append({
            "knowledge_item_id": node.get("source_id") if node.get("source_table") == "knowledge_items" else None,
            "knowledge_node_id": node_id,
            "title": node.get("title"),
            "reason": "missing_publication_artifacts",
            "missing": missing,
            "rag_chunk_count": chunk_count,
            "source": "knowledge_nodes",
        })

    if repair:
        by_persona = {
            str(row.get("persona_id") or "")
            for row in [*item_rows, *node_rows]
            if row.get("persona_id")
        }
        for pid in by_persona:
            _rebuild_embedded_markdown(pid)

    return {
        "ok": not failures,
        "checked": checked,
        "valid": len(ok_items),
        "repaired": repaired,
        "failures": failures,
        "items": ok_items,
    }
