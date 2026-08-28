# -*- coding: utf-8 -*-
"""
Database-first KB intake pipeline for RAG-ready knowledge.

This is intentionally deterministic for the first version: it creates a raw
intake row, classifies common FAQ-shaped text, extracts basic structured facts,
creates a canonical RAG entry + chunk, and mirrors the entry into the existing
semantic graph.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from services import knowledge_graph, supabase_client


CONTENT_LEVELS = {
    "brand": 20,
    "campaign": 30,
    "product_group": 35,
    "product": 40,
    "offer": 45,
    "briefing": 50,
    "audience": 55,
    "tone": 60,
    "rule": 65,
    "copy": 70,
    "faq": 75,
    "asset": 80,
    "gallery": 82,
    "entity": 10,
    "general_note": 90,
}

ALLOWED_CONTENT_TYPES = set(CONTENT_LEVELS)

# Architectural rule (CLAUDE.md / README "KB vs RAG"):
#   Grafo            = todo conhecimento (aprovado ou não).
#   KB Validada      = todo conhecimento aprovado (kb_entries).
#   knowledge_rag    = SOMENTE FAQ aprovado (camada vetorial dos agentes).
# Today only "faq" qualifies; widening to other content types later is a
# one-line change here. Keep this gate in a single helper so callers cannot
# drift.
RAG_ELIGIBLE_CONTENT_TYPES: set[str] = {"faq"}


def is_rag_eligible(content_type: Optional[str]) -> bool:
    """Return True only when this content_type is allowed into knowledge_rag.

    The agents currently only retrieve FAQs from the vector layer; sending
    other shapes (product, copy, rule, brand, tone, entity, ...) would
    pollute retrieval and surface raw operational notes as if they were
    user-facing answers.
    """
    return (content_type or "").strip().lower() in RAG_ELIGIBLE_CONTENT_TYPES


def _slugify(value: str) -> str:
    return knowledge_graph._slugify(value)


def _fold(value: str) -> str:
    return knowledge_graph._fold(value)


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _fold(value or ""))


def _normalize_tags(tags) -> list[str]:
    return knowledge_graph._normalize_tags(tags)


def _resolve_persona(persona_id: Optional[str] = None, persona_slug: Optional[str] = None) -> dict:
    if persona_slug:
        persona = supabase_client.get_persona(persona_slug)
        if not persona:
            raise ValueError(f"Persona not found: {persona_slug}")
        return persona
    if persona_id:
        # get_persona resolves by slug only; use the private resolver to accept IDs.
        resolved = supabase_client._resolve_persona_id(persona_id)
        if not resolved:
            raise ValueError(f"Persona not found: {persona_id}")
        for persona in supabase_client.get_personas():
            if persona.get("id") == resolved:
                return persona
        return {"id": resolved, "slug": persona_id, "name": persona_id}
    raise ValueError("persona_id or persona_slug is required")


def _extract_faq(text: str) -> tuple[Optional[str], Optional[str]]:
    pairs = knowledge_graph._extract_faq_pairs(text)
    if pairs:
        return pairs[0]

    cleaned = (text or "").strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) >= 2 and "?" in lines[0]:
        return lines[0], " ".join(lines[1:])

    match = re.match(r"(?P<q>[^?\n]{8,}\?)\s*(?P<a>.+)", cleaned, flags=re.DOTALL)
    if match:
        return re.sub(r"\s+", " ", match.group("q")).strip(), re.sub(r"\s+", " ", match.group("a")).strip()

    return None, None


def _extract_price(text: str) -> Optional[dict]:
    price_re = re.compile(
        r"R\$\s*(?P<amount>\d{1,6}(?:[.,]\d{2})?)"
        r"(?:\s*(?:por|/)\s*(?P<unit>[A-Za-zÀ-ÿ0-9_-]+))?",
        flags=re.IGNORECASE,
    )
    match = price_re.search(text or "")
    if not match:
        percent = re.search(r"(?P<amount>\d{1,3})\s*%", text or "")
        if not percent:
            return None
        display = percent.group(0)
        return {
            "amount": int(percent.group("amount")),
            "currency": "PERCENT",
            "unit": "percentual",
            "display": display,
        }

    raw_amount = match.group("amount").replace(".", "").replace(",", ".")
    amount = float(raw_amount)
    if amount.is_integer():
        amount = int(amount)
    unit = match.group("unit")
    display = f"R$ {amount:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    if unit:
        display = f"{display} por {unit}"
    return {
        "amount": amount,
        "currency": "BRL",
        "unit": unit,
        "display": display,
    }


def _product_candidates(persona_id: str) -> list[dict]:
    try:
        return supabase_client.list_knowledge_nodes_by_type(["product"], persona_id=persona_id, limit=500)
    except Exception:
        return []


def _detect_products(text: str, persona_id: str, explicit_products: Optional[list[str]] = None) -> list[dict]:
    out: dict[str, dict] = {}
    for product in explicit_products or []:
        slug = _slugify(str(product))
        out[slug] = {"slug": slug, "title": str(product), "source": "explicit"}

    folded_text = _fold(text or "")
    compact_text = _compact(text or "")
    for node in _product_candidates(persona_id):
        slug = node.get("slug") or ""
        title = node.get("title") or slug
        meta = node.get("metadata") or {}
        values = [slug, title, *(node.get("tags") or [])]
        for key in ("aliases", "synonyms"):
            if isinstance(meta.get(key), list):
                values.extend(str(v) for v in meta[key])
        matched = False
        for value in values:
            folded = _fold(str(value))
            compact = _compact(str(value))
            if not folded:
                continue
            if folded in folded_text or (len(compact) >= 5 and compact in compact_text):
                matched = True
                break
        if matched:
            out[slug] = {"slug": slug, "title": title, "source": "graph"}

    return list(out.values())


def classify_intake(
    *,
    text: str,
    persona_id: str,
    title: Optional[str] = None,
    content_type: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
    detect_existing_products: bool = True,
) -> dict:
    meta = dict(metadata or {})
    question, answer = _extract_faq(text)
    requested_type = (content_type or "").strip().lower()
    inferred_type = requested_type or ("faq" if question and answer else "general_note")
    if inferred_type == "other" or inferred_type not in ALLOWED_CONTENT_TYPES:
        inferred_type = "general_note"
    normalized_tags = sorted(set(_normalize_tags(tags or []) + _normalize_tags(meta.get("tags"))))
    explicit_product_values = meta.get("products") or ([meta["product"]] if meta.get("product") else None)
    products = (
        _detect_products(
            " ".join([title or "", text or "", " ".join(normalized_tags)]),
            persona_id,
            explicit_products=explicit_product_values,
        )
        if detect_existing_products
        else [{"slug": _slugify(str(p)), "title": str(p), "source": "explicit"} for p in (explicit_product_values or [])]
    )
    product_slugs = [p["slug"] for p in products if p.get("slug")]

    price = _extract_price(text)
    if price:
        meta.setdefault("price", price)
        normalized_tags.append("preco")

    if inferred_type == "faq":
        entry_title = title or question or "FAQ"
        content = f"Pergunta: {question}\nResposta: {answer}"
        summary = answer[:300] if answer else text[:300]
    else:
        entry_title = title or (text.strip().splitlines()[0][:90] if text.strip() else "Conhecimento")
        content = text.strip()
        summary = text.strip()[:300]

    slug = _slugify(meta.get("slug") or entry_title)[:100]
    canonical_key = f"{inferred_type}:{slug}"

    return {
        "content_type": inferred_type,
        "semantic_level": CONTENT_LEVELS.get(inferred_type, 90),
        "title": entry_title,
        "question": question,
        "answer": answer,
        "content": content,
        "summary": summary,
        "canonical_key": canonical_key,
        "slug": slug,
        "tags": sorted(set(normalized_tags + product_slugs + [inferred_type])),
        "products": product_slugs,
        "campaigns": meta.get("campaigns") or ([meta["campaign"]] if meta.get("campaign") else []),
        "entities": meta.get("entities") or [],
        "metadata": meta,
        "confidence": 0.82 if inferred_type == "faq" else 0.55,
        "importance": 0.65 if inferred_type == "faq" else 0.5,
    }


def process_intake(
    *,
    raw_text: str,
    persona_id: Optional[str] = None,
    persona_slug: Optional[str] = None,
    source: str = "manual",
    source_ref: Optional[str] = None,
    title: Optional[str] = None,
    content_type: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
    submitted_by: Optional[str] = None,
    validate: bool = False,
    parent_node_id: Optional[str] = None,
    parent_relation_type: str = "manual",
    mirror_graph: bool = True,
) -> dict:
    persona = _resolve_persona(persona_id=persona_id, persona_slug=persona_slug)
    resolved_persona_id = persona["id"]

    intake = supabase_client.insert_knowledge_intake_message({
        "persona_id": resolved_persona_id,
        "source": source,
        "source_ref": source_ref,
        "raw_text": raw_text,
        "raw_payload": {
            "title": title,
            "content_type": content_type,
            "tags": tags or [],
            "metadata": metadata or {},
        },
        "submitted_by": submitted_by,
        "status": "received",
    })

    try:
        classified = classify_intake(
            text=raw_text,
            persona_id=resolved_persona_id,
            title=title,
            content_type=content_type,
            tags=tags,
            metadata=metadata,
        )
        status = "validated" if validate else "pending_validation"
        now_iso = datetime.now(timezone.utc).isoformat()
        entry_metadata = {**classified["metadata"], "rag_index": (classified["metadata"] or {}).get("rag_index", "default")}
        rag_allowed = is_rag_eligible(classified["content_type"])
        entry: Optional[dict] = None
        chunks: list[dict] = []
        if rag_allowed:
            entry = supabase_client.upsert_knowledge_rag_entry({
                "persona_id": resolved_persona_id,
                "intake_id": intake.get("id"),
                "content_type": classified["content_type"],
                "semantic_level": classified["semantic_level"],
                "title": classified["title"],
                "question": classified["question"],
                "answer": classified["answer"],
                "content": classified["content"],
                "summary": classified["summary"],
                "canonical_key": classified["canonical_key"],
                "slug": classified["slug"],
                "status": status,
                "tags": classified["tags"],
                "entities": classified["entities"],
                "products": classified["products"],
                "campaigns": classified["campaigns"],
                "metadata": entry_metadata,
                "confidence": classified["confidence"],
                "importance": classified["importance"],
                "validated_at": now_iso if validate else None,
            })

            chunks = supabase_client.replace_knowledge_rag_chunks(
                entry["id"],
                resolved_persona_id,
                [{
                    "chunk_index": 0,
                    "chunk_text": classified["content"],
                    "chunk_summary": classified["summary"],
                    "metadata": {
                        "content_type": classified["content_type"],
                        "canonical_key": classified["canonical_key"],
                        "embedding_status": "pending",
                        "rag_index": entry_metadata["rag_index"],
                    },
                }],
            )

        mirror = None
        main_tree_edge = None
        repair = None
        if mirror_graph:
            mirror = knowledge_graph.bootstrap_from_item(
                {
                    # Use the rag_entry id when we created one (so the graph
                    # node mirrors the canonical RAG row); fall back to the
                    # intake id when the content is not RAG-eligible so we
                    # still get a stable graph anchor.
                    "id": (entry or {}).get("id") or intake.get("id"),
                    "persona_id": resolved_persona_id,
                    "content_type": classified["content_type"],
                    "title": classified["title"],
                    "content": classified["content"],
                    "tags": classified["tags"],
                    "metadata": classified["metadata"],
                    "status": status,
                },
                frontmatter={
                    **classified["metadata"],
                    "slug": classified["slug"],
                    "product": classified["products"],
                    "campaigns": classified["campaigns"],
                    "tags": classified["tags"],
                },
                body=classified["content"],
                persona_id=resolved_persona_id,
                source_table="knowledge_rag_entries" if rag_allowed else "knowledge_intake_messages",
            )
            main_tree_edge = knowledge_graph.ensure_main_tree_connection(
                mirror,
                persona_id=resolved_persona_id,
                parent_node_id=parent_node_id or (classified["metadata"] or {}).get("parent_node_id"),
                relation_type=parent_relation_type or (classified["metadata"] or {}).get("parent_relation_type") or "manual",
            )
            repair = knowledge_graph.repair_primary_tree_connections(
                resolved_persona_id,
                [mirror.get("id")] if mirror and mirror.get("id") else None,
            )

        # knowledge_intake_messages.status does not accept "graph_only".
        # Non-RAG graph inserts should keep a standard lifecycle status.
        intake_status = "rag_created" if rag_allowed else status
        supabase_client.update_knowledge_intake_message(
            intake["id"],
            {"status": intake_status, "processed_at": now_iso},
        )

        return {
            "intake": {**intake, "status": intake_status, "processed_at": now_iso},
            "classification": classified,
            "rag_entry": entry,
            "chunks": chunks,
            "graph_node": mirror,
            "main_tree_edge": main_tree_edge,
            "primary_tree_repair": repair,
            "rag_eligible": rag_allowed,
        }
    except Exception as exc:
        if intake.get("id"):
            supabase_client.update_knowledge_intake_message(
                intake["id"],
                {
                    "status": "error",
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc)[:1000],
                },
            )
        raise


def _entry_slug(entry: dict, classified: Optional[dict] = None) -> str:
    return _slugify((entry.get("metadata") or {}).get("slug") or entry.get("slug") or (classified or {}).get("slug") or entry.get("title") or "")


def _default_campaign_entry(persona: dict, run_token: str) -> dict:
    persona_slug = persona.get("slug") or "persona"
    title = f"Campanha de Conhecimento [{run_token}]"
    slug = f"campanha-conhecimento-{persona_slug}-{run_token}"
    content = f"Campanha raiz para organizar o conhecimento criado para {persona_slug}."
    return {
        "content_type": "campaign",
        "slug": slug,
        "title": title,
        "content": content,
        "tags": [run_token, persona_slug, "campaign"],
        "metadata": {"slug": slug, "tags": [run_token, persona_slug, "campaign"], "auto_root": True},
    }


def _primary_parent_slug(entry: dict, campaign_slug: str) -> Optional[str]:
    ctype = (entry.get("content_type") or "").lower()
    meta = entry.get("metadata") or {}
    slug = meta.get("slug") or entry.get("slug") or ""
    if ctype == "campaign":
        return None
    if meta.get("parent_slug"):
        return meta.get("parent_slug")
    if entry.get("parent_slug"):
        return entry.get("parent_slug")
    if ctype == "audience":
        return campaign_slug
    if meta.get("product"):
        return meta.get("product")
    product = meta.get("products")
    if isinstance(product, list) and product:
        return product[0]
    if meta.get("audience"):
        return meta.get("audience")
    audiences = meta.get("audiences")
    if isinstance(audiences, list) and audiences:
        return audiences[0]
    if slug.startswith("beneficio-"):
        parts = slug.split("-")
        if len(parts) >= 3:
            audience = parts[-2] if parts[-1].startswith("e2e") else ""
            if audience:
                return f"publico-{audience}-{parts[-1]}"
    return campaign_slug


def _link_parent_by_target(links: Optional[list[dict]]) -> dict[str, tuple[str, str]]:
    by_target: dict[str, tuple[str, str]] = {}
    for link in links or []:
        target_slug = _slugify(str(link.get("target") or link.get("target_slug") or ""))
        source_slug = _slugify(str(link.get("source") or link.get("source_slug") or ""))
        if not target_slug or not source_slug:
            continue
        by_target.setdefault(target_slug, (source_slug, link.get("relation_type") or "manual"))
    return by_target


def process_intake_plan(
    *,
    entries: list[dict],
    persona_id: Optional[str] = None,
    persona_slug: Optional[str] = None,
    run_token: Optional[str] = None,
    links: Optional[list[dict]] = None,
    source: str = "plan",
    source_ref: Optional[str] = None,
    submitted_by: Optional[str] = None,
    validate: bool = True,
) -> dict:
    persona = _resolve_persona(persona_id=persona_id, persona_slug=persona_slug)
    resolved_persona_id = persona["id"]
    token = _slugify(run_token or source_ref or datetime.now(timezone.utc).strftime("plan-%Y%m%d%H%M%S"))
    now_iso = datetime.now(timezone.utc).isoformat()

    plan_entries = [dict(e) for e in entries or []]
    campaign_entry = next((e for e in plan_entries if (e.get("content_type") or "").lower() == "campaign"), None)
    if campaign_entry is None:
        campaign_entry = _default_campaign_entry(persona, token)
        plan_entries.insert(0, campaign_entry)

    classified_rows: list[tuple[dict, dict]] = []
    for entry in plan_entries:
        meta = dict(entry.get("metadata") or {})
        if entry.get("slug"):
            meta.setdefault("slug", entry.get("slug"))
        meta.setdefault("tags", entry.get("tags") or meta.get("tags") or [])
        classified = classify_intake(
            text=entry.get("content") or entry.get("raw_text") or "",
            persona_id=resolved_persona_id,
            title=entry.get("title"),
            content_type=entry.get("content_type"),
            tags=entry.get("tags") or meta.get("tags") or [],
            metadata=meta,
            detect_existing_products=False,
        )
        classified_rows.append((entry, classified))

    client = supabase_client.get_client()
    intake_payload = [
        {
            "persona_id": resolved_persona_id,
            "source": source,
            "source_ref": source_ref or token,
            "raw_text": classified["content"],
            "raw_payload": {
                "title": classified["title"],
                "content_type": classified["content_type"],
                "tags": classified["tags"],
                "metadata": classified["metadata"],
            },
            "submitted_by": submitted_by,
            "status": "rag_created",
            "processed_at": now_iso,
        }
        for _, classified in classified_rows
    ]
    intake_rows = client.table("knowledge_intake_messages").insert(intake_payload).execute().data or []

    # Only FAQ rows are eligible for the RAG layer (see is_rag_eligible).
    # Non-FAQ rows still get graph nodes and a campaign tree; they're just
    # invisible to the agent retrieval until promoted/converted.
    rag_payload = []
    for idx, (_, classified) in enumerate(classified_rows):
        if not is_rag_eligible(classified["content_type"]):
            continue
        rag_payload.append({
            "persona_id": resolved_persona_id,
            "intake_id": (intake_rows[idx] or {}).get("id") if idx < len(intake_rows) else None,
            "content_type": classified["content_type"],
            "semantic_level": classified["semantic_level"],
            "title": classified["title"],
            "question": classified["question"],
            "answer": classified["answer"],
            "content": classified["content"],
            "summary": classified["summary"],
            "canonical_key": classified["canonical_key"],
            "slug": classified["slug"],
            "status": "validated" if validate else "pending_validation",
            "tags": classified["tags"],
            "entities": classified["entities"],
            "products": classified["products"],
            "campaigns": classified["campaigns"],
            "metadata": {**classified["metadata"], "rag_index": (classified["metadata"] or {}).get("rag_index", "default")},
            "confidence": classified["confidence"],
            "importance": classified["importance"],
            "validated_at": now_iso if validate else None,
            "updated_at": now_iso,
        })
    rag_rows = (
        client.table("knowledge_rag_entries")
        .upsert(rag_payload, on_conflict="persona_id,canonical_key")
        .execute()
        .data
        or []
    ) if rag_payload else []
    rag_by_key = {row.get("canonical_key"): row for row in rag_rows}

    chunk_payload = []
    for _, classified in classified_rows:
        rag = rag_by_key.get(classified["canonical_key"])
        if not rag:
            continue
        chunk_payload.append({
            "rag_entry_id": rag["id"],
            "persona_id": resolved_persona_id,
            "chunk_index": 0,
            "chunk_text": classified["content"],
            "chunk_summary": classified["summary"],
                "metadata": {
                    "content_type": classified["content_type"],
                    "canonical_key": classified["canonical_key"],
                    "embedding_status": "pending",
                    "rag_index": (classified["metadata"] or {}).get("rag_index", "default"),
                },
        })
    if chunk_payload:
        client.table("knowledge_rag_chunks").upsert(
            chunk_payload,
            on_conflict="rag_entry_id,chunk_index",
        ).execute()

    node_payload = []
    for idx, (_, classified) in enumerate(classified_rows):
        rag = rag_by_key.get(classified["canonical_key"])
        intake_row = intake_rows[idx] if idx < len(intake_rows) else None
        # FAQ rows mirror the RAG entry; non-FAQ rows mirror the intake
        # message so the graph still has a stable source pointer.
        if rag:
            source_table = "knowledge_rag_entries"
            source_id = rag.get("id")
        else:
            source_table = "knowledge_intake_messages"
            source_id = (intake_row or {}).get("id")
        node_payload.append({
            "persona_id": resolved_persona_id,
            "source_table": source_table,
            "source_id": source_id,
            "node_type": classified["content_type"],
            "slug": classified["slug"],
            "title": classified["title"],
            "summary": classified["summary"],
            "tags": classified["tags"],
            "metadata": {
                **classified["metadata"],
                "content_type": classified["content_type"],
                "source_status": "validated" if validate else "pending_validation",
                "rag_eligible": is_rag_eligible(classified["content_type"]),
            },
            "status": "validated" if validate else "pending",
            **knowledge_graph._hierarchy_fields(classified["content_type"], classified["metadata"], confidence=classified["confidence"]),
        })
    try:
        graph_nodes = client.table("knowledge_nodes").insert(node_payload).execute().data or []
    except Exception:
        graph_nodes = []
        for row in node_payload:
            node = supabase_client.upsert_knowledge_node(row)
            if node:
                graph_nodes.append(node)
    node_by_slug: dict[str, dict] = {node.get("slug"): node for node in graph_nodes if node.get("slug")}

    campaign_slug = _slugify((campaign_entry.get("metadata") or {}).get("slug") or campaign_entry.get("slug") or campaign_entry.get("title") or "")
    campaign_node = node_by_slug.get(campaign_slug)
    persona_node = knowledge_graph._ensure_persona_root(resolved_persona_id)

    link_parent_by_target = _link_parent_by_target(links)
    main_edge_payload = []
    fallback_parent_nodes: list[dict] = []
    if persona_node and campaign_node:
        main_edge_payload.append({
            "source_node_id": persona_node["id"],
            "target_node_id": campaign_node["id"],
            "relation_type": "contains",
            "persona_id": resolved_persona_id,
            "weight": 1,
            "metadata": {"primary_tree": True, "created_from": "plan_bulk"},
        })
    for entry, classified in classified_rows:
        child = node_by_slug.get(classified["slug"])
        if not child or classified["slug"] == campaign_slug:
            continue
        meta = classified.get("metadata") or {}
        parent = None
        relation_type = "manual"
        if meta.get("parent_node_id"):
            parent = {"id": meta.get("parent_node_id")}
        if not parent:
            linked = link_parent_by_target.get(classified["slug"])
            if linked:
                parent = node_by_slug.get(linked[0])
                relation_type = linked[1] or "manual"
        if not parent:
            parent_slug = _primary_parent_slug(entry, campaign_slug)
            parent = node_by_slug.get(_slugify(str(parent_slug or "")))
        if not parent:
            parent = campaign_node
        if not parent:
            parent = persona_node
            relation_type = "belongs_to_persona"
            fallback_parent_nodes.append({
                "slug": child.get("slug"),
                "title": child.get("title"),
                "node_type": child.get("node_type"),
                "reason": "no_parent_resolved",
            })
        if parent and parent.get("id") != child.get("id"):
            main_edge_payload.append({
                "source_node_id": parent["id"],
                "target_node_id": child["id"],
                "relation_type": relation_type,
                "persona_id": resolved_persona_id,
                "weight": 1,
                "metadata": {"primary_tree": True, "created_from": "plan_bulk"},
            })

    for link in links or []:
        source_slug = _slugify(str(link.get("source") or link.get("source_slug") or ""))
        target_slug = _slugify(str(link.get("target") or link.get("target_slug") or ""))
        source_node = node_by_slug.get(source_slug)
        target_node = node_by_slug.get(target_slug)
        if source_node and target_node:
            main_edge_payload.append({
                "source_node_id": source_node["id"],
                "target_node_id": target_node["id"],
                "relation_type": link.get("relation_type") or "manual",
                "persona_id": resolved_persona_id,
                "weight": link.get("weight") or 1,
                "metadata": {**(link.get("metadata") or {}), "created_from": "plan_bulk", "primary_tree": True},
            })

    main_edges = []
    if main_edge_payload:
        try:
            main_edges = (
                client.table("knowledge_edges")
                .upsert(main_edge_payload, on_conflict="source_node_id,target_node_id,relation_type")
                .execute()
                .data
                or []
            )
        except Exception:
            for edge in main_edge_payload:
                created = supabase_client.upsert_knowledge_edge(
                    edge["source_node_id"],
                    edge["target_node_id"],
                    edge["relation_type"],
                    persona_id=edge.get("persona_id"),
                    weight=edge.get("weight") or 1,
                    metadata=edge.get("metadata") or {},
                )
                if created:
                    main_edges.append(created)

    aux_edges = []
    repair = knowledge_graph.repair_primary_tree_connections(
        resolved_persona_id,
        [node.get("id") for node in node_by_slug.values() if node.get("id")],
    )
    for node in repair.get("fallback_nodes") or []:
        fallback_parent_nodes.append({**node, "reason": "repair_persona_fallback"})

    return {
        "ok": True,
        "persona": persona,
        "run_token": token,
        "campaign": campaign_node,
        "entries_created": len(rag_rows),
        "nodes_created": len(node_by_slug),
        "main_edges": len([e for e in main_edges if e]),
        "auxiliary_edges": len(aux_edges),
        "fallback_parent_count": len(fallback_parent_nodes),
        "fallback_parent_nodes": fallback_parent_nodes[:25],
        "primary_tree_repair": repair,
        "rag_entries": rag_rows,
        "graph_nodes": list(node_by_slug.values()),
    }
