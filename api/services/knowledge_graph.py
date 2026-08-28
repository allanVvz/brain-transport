# -*- coding: utf-8 -*-
"""
knowledge_graph: lightweight semantic graph layer over knowledge_items / kb_entries.

Responsibilities:
- Keep a normalized graph of personas, products, campaigns, faqs, copies,
  assets, rules, etc. via knowledge_nodes + knowledge_edges (migration 008).
- Bootstrap nodes/edges from synced vault items so the existing flow
  (vault â†’ knowledge_items â†’ kb_entries) continues to work unchanged.
- Resolve a "chat context": given a lead's recent messages, find related
  products/campaigns/assets via term matching + 1-hop neighbourhood.

All public functions are defensive: if migration 008 is not yet applied or
any DB error occurs, they degrade to empty lists rather than raising. The
old text-based KB search keeps working as the fallback.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from uuid import UUID
from typing import Iterable, Optional

from services import supabase_client

logger = logging.getLogger("knowledge_graph")

# The graph layer must not ship client/product-specific canonical data.
# Product, campaign, brand and entity nodes are inferred from the current
# knowledge item metadata/frontmatter and from nodes that already exist.
_TOPIC_NODE_TYPES = {"product", "offer", "campaign", "brand", "entity", "persona", "audience"}

# Map content_type â†’ node_type for items mirrored into the graph.
_CONTENT_TYPE_TO_NODE: dict[str, str] = {
    "faq":      "faq",
    "copy":     "copy",
    "campaign": "campaign",
    "product":  "product",
    "asset":    "asset",
    "rule":     "rule",
    "tone":     "tone",
    "audience": "audience",
    "entity":   "entity",
    "offer":    "offer",
    "brand":    "brand",
    "briefing": "briefing",
    "prompt":   "rule",
    "competitor": "audience",
}

# Roles allowed in agent_visibility. Used for `visible_to_agent` edges.
_KNOWN_ROLES = {"sdr", "closer", "followup", "maker", "classifier"}

_VALIDATED_ITEM_STATUSES = {"approved", "embedded", "ativo", "active", "validated"}
_PENDING_ITEM_STATUSES = {"pending", "needs_persona", "needs_category", "draft"}

# Hierarchy levels for graph layout. Mirrors the operator-facing tree:
#   Persona (0)
#     â”œâ”€â”€ Briefing / Brand / Campaign (10 â€” top of any capture)
#     â”‚       â””â”€â”€ Audience (20)
#     â”‚             â””â”€â”€ Product / Entity (30)
#     â”‚                   â”œâ”€â”€ Copy / FAQ / Asset (40)
#     â”‚                   â”œâ”€â”€ Tone / Rule (40 â€” sibling concerns)
#     â”‚                   â””â”€â”€ Embedded RAG (50, only after approval)
# Importance scales independently of level so primary types (briefing,
# product) rank above utility nodes (tag, mention).
_NODE_HIERARCHY_DEFAULTS: dict[str, tuple[int, float]] = {
    "persona": (0, 1.00),
    "briefing": (10, 0.85),
    "brand": (10, 0.90),
    "campaign": (10, 0.80),
    "audience": (20, 0.75),
    "entity": (30, 0.70),
    "product": (30, 0.85),
    "offer": (35, 0.78),
    "tone": (40, 0.70),
    "rule": (40, 0.80),
    "copy": (40, 0.65),
    "faq": (40, 0.45),
    "asset": (40, 0.55),
    "embedded": (50, 0.95),
    "tag": (90, 0.30),
    "mention": (92, 0.25),
    "knowledge_item": (95, 0.40),
    "kb_entry": (95, 0.50),
}

_MAIN_TREE_RELATIONS = {
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

_TRACEABILITY_KEYS = ("session_id", "source_ref", "created_via", "tree_mode", "branch_policy")


_SEMANTIC_EDGE_BY_TYPES: dict[tuple[str, str], tuple[str, str]] = {
    ("persona", "briefing"): ("contains_briefing", "Persona contem briefing"),
    ("briefing", "audience"): ("defines_audience", "Briefing define publico"),
    ("audience", "product"): ("offers_product", "Publico recebe oferta de produto"),
    ("product", "offer"): ("defines_offer", "Produto define oferta"),
    ("product", "faq"): ("answers_question", "Produto responde pergunta"),
    ("offer", "copy"): ("supports_copy", "Oferta sustenta copy"),
    ("offer", "faq"): ("answers_question", "Oferta responde pergunta"),
    ("product", "copy"): ("supports_copy", "Produto sustenta copy"),
    ("copy", "faq"): ("answers_question", "Copy responde pergunta"),
    ("rule", "faq"): ("answers_question", "Regra orienta FAQ"),
    ("faq", "embedded"): ("published_to_rag", "FAQ publicado no RAG"),
}


def traceability_metadata(*sources: Optional[dict]) -> dict:
    """Copy flow lineage into graph JSON metadata without changing topology."""
    out: dict = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        nested = source.get("metadata")
        if isinstance(nested, dict):
            for key in _TRACEABILITY_KEYS:
                if nested.get(key) is not None and out.get(key) is None:
                    out[key] = nested.get(key)
        for key in _TRACEABILITY_KEYS:
            if source.get(key) is not None and out.get(key) is None:
                out[key] = source.get(key)
    if out:
        out.setdefault("tree_mode", "pyramidal")
        out.setdefault("branch_policy", "top_down_pyramidal")
    return out


def semantic_edge_metadata(
    source_node: Optional[dict],
    target_node: Optional[dict],
    relation_type: str,
    metadata: Optional[dict] = None,
) -> dict:
    """Add semantic labels as metadata while preserving relation_type."""
    merged = dict(metadata or {})
    source_type = ((source_node or {}).get("node_type") or "").lower()
    target_type = ((target_node or {}).get("node_type") or "").lower()
    semantic = _SEMANTIC_EDGE_BY_TYPES.get((source_type, target_type))
    if semantic:
        merged.setdefault("semantic_relation", semantic[0])
        merged.setdefault("semantic_label", semantic[1])
    elif relation_type:
        merged.setdefault("semantic_relation", relation_type)
    if merged.get("primary_tree") is True:
        merged.setdefault("tree_role", "primary_branch")
    merged.update(traceability_metadata((source_node or {}).get("metadata") or {}, (target_node or {}).get("metadata") or {}))
    if merged.get("session_id"):
        merged.setdefault("tree_mode", "pyramidal")
        merged.setdefault("branch_policy", "top_down_pyramidal")
    return merged


# Default relation_type used when a parent_slug is provided but no explicit
# relation. Pair = (parent_node_type, child_node_type). Fallback is "contains".
# Edge direction is always parent â†’ child (source = parent, target = child).
_DEFAULT_PARENT_RELATION: dict[tuple[str, str], str] = {
    ("brand", "briefing"): "contains",
    ("brand", "campaign"): "contains",
    ("brand", "product"): "contains",
    ("brand", "audience"): "contains",
    ("briefing", "campaign"): "contains",
    ("briefing", "audience"): "contains",
    ("briefing", "product"): "contains",
    ("briefing", "copy"): "contains",
    ("briefing", "faq"): "contains",
    ("briefing", "entity"): "contains",
    ("campaign", "audience"): "contains",
    ("campaign", "product"): "contains",
    ("campaign", "copy"): "contains",
    ("campaign", "faq"): "contains",
    ("campaign", "briefing"): "contains",
    ("campaign", "entity"): "contains",
    ("campaign", "asset"): "uses_asset",
    ("audience", "product"): "offers_product",
    ("audience", "briefing"): "contains",
    ("audience", "copy"): "supports_copy",
    ("audience", "faq"): "answers_question",
    ("audience", "entity"): "contains",
    ("entity", "product"): "contains",
    ("entity", "briefing"): "contains",
    ("product", "offer"): "contains",
    ("offer", "copy"): "supports_copy",
    ("offer", "faq"): "answers_question",
    ("offer", "asset"): "uses_asset",
    ("product", "copy"): "supports_copy",
    ("product", "faq"): "answers_question",
    ("copy", "faq"): "answers_question",
    ("rule", "faq"): "answers_question",
    ("product", "briefing"): "contains",
    ("product", "entity"): "contains",
    ("product", "asset"): "uses_asset",
}


# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _slugify(value: str) -> str:
    s = (value or "").strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "item"


def _fold(value: str) -> str:
    """Lowercase text and remove accents for cheap Portuguese matching."""
    text = unicodedata.normalize("NFKD", value or "")
    text = text.encode("ascii", "ignore").decode("ascii")
    return text.lower()


def _normalize_tags(tags) -> list[str]:
    """Accept tags as text[] OR jsonb (string list, comma-separated string,
    or single string). Always returns a list[str] in lowercase."""
    if not tags:
        return []
    if isinstance(tags, str):
        # comma-separated, possibly JSON-encoded
        try:
            import json as _json
            parsed = _json.loads(tags)
            if isinstance(parsed, list):
                return [str(t).strip().lower() for t in parsed if t]
        except Exception:
            pass
        return [t.strip().lower() for t in tags.split(",") if t.strip()]
    if isinstance(tags, list):
        return [str(t).strip().lower() for t in tags if t]
    return []


def _source_status(item: dict, source_table: str) -> str:
    if source_table == "kb_entries":
        return str(item.get("status") or "ATIVO")
    return str(item.get("status") or "pending")


def _is_validated_source(source_table: Optional[str], status: Optional[str], node_type: Optional[str] = None) -> bool:
    status_l = str(status or "").lower()
    if source_table == "knowledge_items":
        return status_l in {"approved", "embedded"}
    if source_table == "kb_entries":
        return status_l in {"ativo", "active", "validated"}
    if node_type in {"product", "campaign", "persona"} and status_l in {"active", "validated", ""}:
        return True
    return status_l in _VALIDATED_ITEM_STATUSES


def _validation_state(node: dict) -> str:
    meta = node.get("metadata") or {}
    source_table = node.get("source_table")
    status = meta.get("source_status") or node.get("status")
    return "validated" if _is_validated_source(source_table, status, node.get("node_type")) else "pending"


def _link_target(node: dict) -> str:
    meta = node.get("metadata") or {}
    source_table = node.get("source_table")
    source_id = node.get("source_id")
    node_type = node.get("node_type") or "node"
    slug = node.get("slug") or node.get("id")
    if node_type == "asset" and meta.get("file_path"):
        return f"/api-brain/knowledge/file?path={meta['file_path']}"
    if source_table == "knowledge_items" and source_id:
        return f"/knowledge/quality?item_id={source_id}"
    if source_table == "kb_entries" and source_id:
        return f"/knowledge/graph?focus={node_type}:{slug}&kb_entry_id={source_id}"
    return f"/knowledge/graph?focus={node_type}:{slug}"


def _decorate_node(node: dict) -> dict:
    state = _validation_state(node)
    out = dict(node)
    out["validation_status"] = state
    out["validated"] = state == "validated"
    out["link_target"] = _link_target(node)
    return out


def _tipo_to_node_type(tipo: str) -> str:
    t = (tipo or "").lower()
    return {
        "produto": "product",
        "product": "product",
        "oferta": "offer",
        "ofertas": "offer",
        "offer": "offer",
        "offers": "offer",
        "opcao": "offer",
        "opÃ§Ã£o": "offer",
        "variacao": "offer",
        "variaÃ§Ã£o": "offer",
        "product_variant": "offer",
        "purchase_option": "offer",
        "faq": "faq",
        "copy": "copy",
        "briefing": "briefing",
        "campanha": "campaign",
        "campaign": "campaign",
        "asset": "asset",
        "tom": "tone",
        "tone": "tone",
        "regra": "rule",
        "rule": "rule",
        "marca": "brand",
        "brand": "brand",
        "entidade": "entity",
        "entity": "entity",
    }.get(t, "knowledge_item")


def _common_prefix_len(a: str, b: str) -> int:
    """Return the length of the longest common prefix of a and b."""
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def _prefix_overlap(needle: str, haystack_tokens: set[str], min_prefix: int = 4) -> bool:
    """True when `needle` shares a prefix of at least min_prefix chars with
    any token in haystack_tokens. Useful for Portuguese plural/singular
    pairs like modal/modais, papel/papeis, animal/animais."""
    if len(needle) < min_prefix:
        return False
    for tok in haystack_tokens:
        if len(tok) < min_prefix:
            continue
        if _common_prefix_len(needle, tok) >= min_prefix:
            return True
    return False


def _term_matches(terms: list[str], *values: object) -> bool:
    """True when any detected term appears in the given title/content/tags."""
    if not terms:
        return False
    blob = _fold(" ".join(str(v or "") for v in values))
    if not blob:
        return False
    for term in terms:
        folded = _fold(str(term))
        if not folded:
            continue
        slug_term = folded.replace(" ", "-")
        if folded in blob or slug_term in blob:
            return True
    return False


def _title_from_slug(slug: str) -> str:
    return (slug or "").replace("-", " ").replace("_", " ").title()


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _structured_metadata(fm: dict) -> dict:
    """Metadata fields that should survive into graph nodes.

    Keep this generic: scenario/product-specific values live in the incoming
    metadata, not in code.
    """
    if not isinstance(fm, dict):
        return {}
    out: dict = {}
    nested = fm.get("metadata")
    if isinstance(nested, dict):
        out.update(nested)
    for key in (
        "aliases",
        "synonyms",
        "price",
        "colors_count",
        "size",
        "catalog_url",
        "brand",
        "campaign",
        "campaigns",
        "product",
        "products",
        "session_id",
        "source_ref",
        "created_via",
        "tree_mode",
        "branch_policy",
    ):
        if key in fm and fm.get(key) is not None:
            out[key] = fm.get(key)
    return out


def _hierarchy_fields(node_type: str, metadata: Optional[dict] = None, confidence: Optional[float] = None) -> dict:
    """Default hierarchy fields used by graph layout and curation views."""
    level, importance = _NODE_HIERARCHY_DEFAULTS.get(node_type, (50, 0.50))
    meta = metadata or {}
    try:
        level = int(meta.get("level", level))
    except Exception:
        pass
    try:
        importance = float(meta.get("importance", importance))
    except Exception:
        pass
    out = {"level": level, "importance": max(0.0, min(1.0, importance))}
    if confidence is not None:
        out["confidence"] = max(0.0, min(1.0, float(confidence)))
    elif meta.get("confidence") is not None:
        try:
            out["confidence"] = max(0.0, min(1.0, float(meta["confidence"])))
        except Exception:
            pass
    return out


def _ensure_persona_root(persona_id: Optional[str]) -> Optional[dict]:
    if not persona_id:
        return None
    return supabase_client.upsert_knowledge_node({
        "persona_id": persona_id,
        "node_type": "persona",
        "slug": "self",
        "title": "Persona",
        "metadata": {"role": "root"},
        **_hierarchy_fields("persona"),
    })


def _normalize_uuid(value: Optional[str]) -> Optional[str]:
    raw = str(value or "").strip()
    if raw.startswith("gn:"):
        raw = raw[3:]
    try:
        return str(UUID(raw))
    except Exception:
        return None


def ensure_main_tree_connection(
    node: Optional[dict],
    *,
    persona_id: Optional[str],
    parent_node_id: Optional[str] = None,
    relation_type: str = "manual",
) -> Optional[dict]:
    """Ensure a knowledge node participates in the primary tree.

    Auxiliary semantic relations such as has_tag, mentions and same_topic_as are
    still generated elsewhere. This helper only creates the required primary
    branch edge, falling back to persona -> node when no parent is supplied.
    """
    if not node or not node.get("id") or not persona_id:
        return None
    if (node.get("node_type") or "").lower() in {"persona", "embedded", "gallery", "tag", "mention"}:
        return None
    target_id = node["id"]
    meta = node.get("metadata") or {}
    source_id = _normalize_uuid(parent_node_id or meta.get("resolved_parent_node_id"))
    semantic_relation = (relation_type or "manual").strip() or "manual"
    rel = semantic_relation
    source_node: Optional[dict] = None
    if not source_id:
        persona_node = _ensure_persona_root(persona_id)
        if not persona_node:
            return None
        source_id = persona_node["id"]
        source_node = persona_node
        rel = "belongs_to_persona"
    elif source_id:
        try:
            source_node = supabase_client.get_knowledge_node(source_id)
        except Exception:
            source_node = None
    if source_id == target_id:
        return None
    source_type = ((source_node or {}).get("node_type") or "").lower()
    target_type = (node.get("node_type") or "").lower()
    canonical_relation = semantic_relation if semantic_relation in _MAIN_TREE_RELATIONS else _default_plan_relation(source_type, target_type)
    if canonical_relation not in _MAIN_TREE_RELATIONS:
        canonical_relation = "manual"
    return supabase_client.upsert_knowledge_edge(
        source_id,
        target_id,
        canonical_relation,
        persona_id=persona_id,
        weight=1,
        metadata=semantic_edge_metadata(
            source_node,
            node,
            canonical_relation,
            {
                "primary_tree": True,
                "created_from": "main_tree_guard",
                "semantic_relation": semantic_relation,
            },
        ),
    )


def repair_primary_tree_connections(
    persona_id: Optional[str],
    node_ids: Optional[list[str]] = None,
) -> dict:
    """Ensure every scoped non-persona node has at least one structural edge.

    This is the backend guard that mirrors the database trigger. Semantic edges
    such as has_tag, mentions and same_topic_as do not satisfy this requirement.
    """
    if not persona_id:
        return {"checked": 0, "repaired": 0, "fallback_nodes": []}
    client = supabase_client.get_client()
    query = client.table("knowledge_nodes").select("id,node_type,slug,title,persona_id").eq("persona_id", persona_id)
    if node_ids:
        ids = [str(nid) for nid in node_ids if nid]
        if not ids:
            return {"checked": 0, "repaired": 0, "fallback_nodes": []}
        query = query.in_("id", ids)
    nodes = query.limit(5000).execute().data or []
    nodes = [
        n for n in nodes
        if n.get("node_type") not in {"persona", "embedded", "gallery", "tag", "mention"}
    ]
    if not nodes:
        return {"checked": 0, "repaired": 0, "fallback_nodes": []}

    ids = [n["id"] for n in nodes if n.get("id")]
    source_edges = client.table("knowledge_edges").select("source_node_id,target_node_id,relation_type,metadata").in_("source_node_id", ids).limit(5000).execute().data or []
    target_edges = client.table("knowledge_edges").select("source_node_id,target_node_id,relation_type,metadata").in_("target_node_id", ids).limit(5000).execute().data or []
    connected: set[str] = set()
    for edge in [*source_edges, *target_edges]:
        meta = edge.get("metadata") or {}
        if meta.get("active") is False:
            continue
        if meta.get("primary_tree") is not True:
            continue
        if edge.get("target_node_id") in ids:
            connected.add(edge["target_node_id"])

    persona_node = _ensure_persona_root(persona_id)
    fallback_nodes: list[dict] = []
    if not persona_node:
        return {"checked": len(nodes), "repaired": 0, "fallback_nodes": []}
    for node in nodes:
        if node.get("id") in connected:
            continue
        edge = ensure_main_tree_connection(node, persona_id=persona_id)
        if edge:
            fallback_nodes.append({
                "id": node.get("id"),
                "slug": node.get("slug"),
                "title": node.get("title"),
                "node_type": node.get("node_type"),
            })
    return {
        "checked": len(nodes),
        "repaired": len(fallback_nodes),
        "fallback_nodes": fallback_nodes,
    }


def _default_plan_relation(parent_type: Optional[str], child_type: Optional[str]) -> str:
    child = (child_type or "").strip().lower()
    parent = (parent_type or "").strip().lower()
    if parent == "copy" and child == "faq":
        return "answers_question"
    if parent == "brand" and child == "campaign":
        return "contains"
    if parent == "campaign" and child == "briefing":
        return "contains"
    if parent == "briefing" and child == "audience":
        return "contains"
    if parent == "product_group" and child == "product":
        return "contains"
    if parent == "product_group" and child == "copy":
        return "supports_copy"
    if parent == "product_group" and child == "faq":
        return "answers_question"
    if parent == "audience" and child == "product_group":
        return "contains"
    if parent == "persona":
        return "contains"
    if child == "entity" or parent == "entity":
        return "contains"
    if parent == "audience" and child == "product":
        return "offers_product"
    if child == "faq":
        return "answers_question"
    if child == "copy":
        return "supports_copy"
    if child == "asset":
        return "uses_asset"
    if child == "briefing":
        return "briefed_by"
    if parent == "campaign":
        return "part_of_campaign"
    if parent == "product":
        return "about_product"
    return "manual"


def _preferred_parent_types(child_type: Optional[str]) -> tuple[str, ...]:
    child = (child_type or "").strip().lower()
    if child == "faq":
        return ("copy", "offer", "product")
    if child == "copy":
        return ("product", "campaign")
    if child == "campaign":
        return ("briefing", "brand", "persona")
    if child == "briefing":
        return ("brand", "campaign", "audience", "product")
    if child == "product":
        return ("audience", "entity", "campaign", "briefing", "brand")
    if child == "entity":
        return ("product", "audience", "campaign", "briefing", "brand")
    if child in {"rule", "tone", "asset"}:
        return ("product", "campaign", "brand", "audience", "briefing")
    if child == "audience":
        return ("campaign", "briefing", "brand", "entity")
    return ("product", "campaign", "brand", "audience", "briefing")


def _resolve_plan_node(
    *,
    slug: Optional[str],
    persona_id: Optional[str],
    nodes_by_slug: dict[str, dict],
) -> Optional[dict]:
    normalized = _slugify(str(slug or ""))
    if not normalized:
        return None
    if persona_id:
        try:
            persona = supabase_client.get_persona_by_id(persona_id) or {}
        except Exception:
            persona = {}
        if normalized and normalized == _slugify(str(persona.get("slug") or persona.get("name") or "")):
            normalized = "self"
    if normalized in {"self", "persona", "persona-root"}:
        return _ensure_persona_root(persona_id)
    node = nodes_by_slug.get(normalized)
    if node:
        return node
    node = supabase_client.get_knowledge_node_by_slug(normalized, persona_id=persona_id)
    if node:
        nodes_by_slug[normalized] = node
    return node


def _candidate_parent_slugs_from_entry(
    *,
    entry: dict,
    item: dict,
    node: dict,
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    tags = entry.get("tags") or item.get("tags") or []
    if node.get("node_type") == "faq":
        for raw_tag in tags:
            tag_slug = _slugify(str(raw_tag or ""))
            if tag_slug:
                candidates.append(("product", tag_slug))
    topic_relations = _topic_relations_for_item(
        {
            "title": entry.get("title") or item.get("title"),
            "content_type": entry.get("content_type") or item.get("content_type"),
            "tags": tags,
        },
        metadata or {},
        node.get("node_type") or "",
    )
    candidates.extend(topic_relations)
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str]] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def apply_plan_hierarchy(
    *,
    persona_id: Optional[str],
    persisted_items: list[dict],
    plan_entries: list[dict],
    plan_links: Optional[list[dict]] = None,
) -> dict:
    """Materialize the hierarchy declared by Sofia's knowledge plan.

    Every persisted item keeps at least one primary-tree edge. When the plan
    declares `metadata.parent_slug` or an explicit item in `links`, the child is
    reattached under that parent instead of staying flat under the persona root.
    """
    if not persona_id or not persisted_items:
        return {"items": [], "resolved_links": 0, "missing_links": []}

    nodes_by_slug: dict[str, dict] = {}
    items_report: list[dict] = []
    for index, item in enumerate(persisted_items):
        node = (
            supabase_client.get_knowledge_node_for_source("knowledge_items", str(item.get("id")), persona_id=persona_id)
            or supabase_client.get_knowledge_node((item.get("metadata") or {}).get("knowledge_node_id"))
        )
        if not node or not node.get("id"):
            continue
        if (node.get("node_type") or "").lower() in {"tag", "mention", "embedded", "gallery"}:
            continue
        entry = plan_entries[index] if index < len(plan_entries) else {}
        entry_slug = _slugify(
            str(
                entry.get("slug")
                or (item.get("metadata") or {}).get("slug")
                or item.get("file_path")
                or item.get("title")
                or item.get("id")
                or ""
            )
        )
        if entry_slug:
            nodes_by_slug[entry_slug] = node
        items_report.append({
            "item": item,
            "entry": entry,
            "node": node,
            "slug": entry_slug,
        })

    explicit_targets: dict[str, dict] = {}
    for link in plan_links or []:
        if not isinstance(link, dict):
            continue
        target_slug = _slugify(str(link.get("target_slug") or ""))
        source_slug = _slugify(str(link.get("source_slug") or ""))
        if not target_slug or not source_slug:
            continue
        explicit_targets[target_slug] = {
            "parent_slug": source_slug,
            "relation_type": (link.get("relation_type") or "").strip() or None,
        }

    resolved_links = 0
    missing_links: list[dict] = []
    for report in items_report:
        entry = report["entry"] or {}
        node = report["node"]
        child_slug = report["slug"]
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        explicit = explicit_targets.get(child_slug or "")
        parent_slug = (
            (metadata or {}).get("parent_slug")
            or (explicit or {}).get("parent_slug")
            or None
        )
        parent_node = _resolve_plan_node(slug=parent_slug, persona_id=persona_id, nodes_by_slug=nodes_by_slug)
        resolution_mode = "explicit_plan" if parent_slug else "deterministic_fallback"
        quarantine_reason = None
        if parent_slug and not parent_node:
            resolution_mode = "quarantined"
            quarantine_reason = "missing_explicit_parent"
        if not parent_node:
            candidate_pairs = _candidate_parent_slugs_from_entry(
                entry=entry,
                item=report["item"] or {},
                node=node,
            )
            for preferred_type in _preferred_parent_types(node.get("node_type")):
                candidate_slugs = [
                    slug for ntype, slug in candidate_pairs
                    if ntype == preferred_type and slug
                ]
                candidate_slugs = [slug for slug in dict.fromkeys(candidate_slugs) if slug != child_slug]
                if len(candidate_slugs) > 1:
                    resolution_mode = "quarantined"
                    quarantine_reason = f"ambiguous_{preferred_type}_parent"
                    break
                if not candidate_slugs:
                    continue
                candidate = candidate_slugs[0]
                parent_node = _resolve_plan_node(slug=candidate, persona_id=persona_id, nodes_by_slug=nodes_by_slug)
                if parent_node and parent_node.get("id") != node.get("id"):
                    parent_slug = candidate
                    resolution_mode = "deterministic_fallback"
                    break
            if resolution_mode == "quarantined":
                parent_node = None
        relation_type = (
            (explicit or {}).get("relation_type")
            or _default_plan_relation((parent_node or {}).get("node_type"), node.get("node_type"))
        )
        edge = None
        if resolution_mode != "quarantined":
            edge = ensure_main_tree_connection(
                node,
                persona_id=persona_id,
                parent_node_id=(parent_node or {}).get("id"),
                relation_type=relation_type,
            )
        node_meta = {
            **(node.get("metadata") or {}),
            "resolution_mode": resolution_mode,
            "quarantine_state": "structural" if resolution_mode == "quarantined" else None,
            "quarantine_reason": quarantine_reason,
            "resolved_parent_slug": parent_slug if resolution_mode != "quarantined" else None,
            "resolved_parent_node_id": (parent_node or {}).get("id") if resolution_mode != "quarantined" else None,
        }
        node_meta = {k: v for k, v in node_meta.items() if v is not None}
        supabase_client.update_knowledge_node(node.get("id"), {"metadata": node_meta})
        report["main_tree_edge"] = edge
        report["parent_node_id"] = (parent_node or {}).get("id")
        report["parent_slug"] = parent_slug or ("self" if not parent_node else None)
        report["resolution_mode"] = resolution_mode
        report["quarantine_reason"] = quarantine_reason
        if parent_slug and not parent_node:
            missing_links.append({
                "target_slug": child_slug,
                "missing_parent_slug": parent_slug,
                "resolution_mode": resolution_mode,
                "quarantine_reason": quarantine_reason,
            })
        elif edge and edge.get("id") and parent_node:
            resolved_links += 1

    return {
        "items": [
            {
                "knowledge_item_id": (report["item"] or {}).get("id"),
                "knowledge_node_id": (report["node"] or {}).get("id"),
                "slug": report.get("slug"),
                "parent_slug": report.get("parent_slug"),
                "parent_node_id": report.get("parent_node_id"),
                "main_tree_edge_id": ((report.get("main_tree_edge") or {}).get("id")),
                "resolution_mode": report.get("resolution_mode"),
                "quarantine_reason": report.get("quarantine_reason"),
            }
            for report in items_report
        ],
        "resolved_links": resolved_links,
        "missing_links": missing_links,
    }


def _fallback_nodes_from_tables(terms: list[str], persona_id: Optional[str], existing_source_ids: set[str]) -> list[dict]:
    """Surface matching legacy knowledge_items/kb_entries even before graph sync."""
    if not terms:
        return []

    out: list[dict] = []
    try:
        items = supabase_client.get_knowledge_items(persona_id=persona_id, limit=300, offset=0)
    except Exception:
        items = []
    for item in items or []:
        sid = str(item.get("id") or "")
        if not sid or sid in existing_source_ids:
            continue
        content = item.get("content") or ""
        item_tags = _normalize_tags(item.get("tags"))
        if not _term_matches(terms, item.get("title"), content, item.get("file_path"), " ".join(item_tags)):
            continue
        content_type = (item.get("content_type") or "knowledge_item").lower()
        out.append(_decorate_node({
            "id": f"ki:{sid}",
            "persona_id": item.get("persona_id"),
            "source_table": "knowledge_items",
            "source_id": item.get("id"),
            "node_type": _CONTENT_TYPE_TO_NODE.get(content_type, content_type),
            "slug": _slugify(item.get("file_path") or item.get("title") or sid)[:80],
            "title": item.get("title") or content_type,
            "summary": content[:400],
            "tags": item_tags or [_slugify(terms[0])],
            "metadata": {
                "file_path": item.get("file_path"),
                "source_status": item.get("status") or "pending",
                "fallback": True,
            },
            "status": "validated" if _is_validated_source("knowledge_items", item.get("status"), content_type) else "pending",
        }))

    try:
        entries = supabase_client.get_kb_entries(persona_id=persona_id, status="ATIVO")
    except Exception:
        entries = []
    for entry in entries or []:
        sid = str(entry.get("id") or "")
        if not sid or sid in existing_source_ids:
            continue
        content = entry.get("conteudo") or ""
        entry_tags = _normalize_tags(entry.get("tags"))
        if not _term_matches(terms, entry.get("titulo"), content, entry.get("produto"), " ".join(entry_tags)):
            continue
        node_type = _tipo_to_node_type(entry.get("tipo") or entry.get("categoria") or "")
        out.append(_decorate_node({
            "id": f"kb:{sid}",
            "persona_id": entry.get("persona_id"),
            "source_table": "kb_entries",
            "source_id": entry.get("id"),
            "node_type": node_type,
            "slug": _slugify(entry.get("titulo") or sid)[:80],
            "title": entry.get("titulo") or node_type,
            "summary": content[:400],
            "tags": entry_tags or [_slugify(terms[0])],
            "metadata": {
                "source_status": entry.get("status") or "ATIVO",
                "fallback": True,
            },
            "status": "validated",
        }))
    return out


def _persona_id_from_slug(persona_slug_or_id: Optional[str]) -> Optional[str]:
    """Resolve a persona slug or UUID into a UUID, or return None."""
    if not persona_slug_or_id:
        return None
    if len(persona_slug_or_id) == 36 and persona_slug_or_id.count("-") == 4:
        return persona_slug_or_id
    p = supabase_client.get_persona(persona_slug_or_id)
    return p.get("id") if p else None


# â”€â”€ Canonical nodes (idempotent) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def ensure_canonical_for_persona(persona_id: Optional[str]) -> dict:
    """Legacy no-op.

    Canonical graph nodes are now derived from real knowledge metadata instead
    of being shipped with client/product-specific defaults.
    """
    return {}


def rebuild_graph_for_persona(
    persona_id: Optional[str],
    limit_items: int = 1000,
    limit_kb: int = 1000,
) -> dict:
    """Re-run bootstrap_from_item for every existing knowledge_items + kb_entries
    row tied to a persona (or globally when persona_id is None).

    Idempotent â€” existing nodes/edges are preserved by the upsert keys, and
    bootstrap is fully driven by the source data (no client-specific seeds).

    Used after migration 008 is applied or whenever the graph drifts from
    the source-of-truth tables.

    Returns a counts dict with mirror/skip totals + first 20 errors.
    """
    counts: dict = {
        "items_seen": 0, "items_mirrored": 0, "items_skipped": 0,
        "kb_seen": 0, "kb_mirrored": 0, "kb_skipped": 0,
        "errors": [],
    }

    try:
        items = supabase_client.get_knowledge_items(
            persona_id=persona_id, limit=limit_items, offset=0,
        ) or []
    except Exception as exc:
        counts["errors"].append(f"get_knowledge_items: {exc}")
        items = []

    for item in items:
        counts["items_seen"] += 1
        try:
            mirror = bootstrap_from_item(
                item,
                frontmatter=item.get("metadata") or {},
                body=item.get("content") or "",
                persona_id=item.get("persona_id") or persona_id,
                source_table="knowledge_items",
            )
            if mirror:
                counts["items_mirrored"] += 1
            else:
                counts["items_skipped"] += 1
        except Exception as exc:
            counts["items_skipped"] += 1
            counts["errors"].append(f"item {item.get('id')}: {str(exc)[:120]}")

    try:
        entries = supabase_client.get_kb_entries(persona_id=persona_id, status="") or []
    except Exception as exc:
        counts["errors"].append(f"get_kb_entries: {exc}")
        entries = []

    for entry in entries[:limit_kb]:
        counts["kb_seen"] += 1
        try:
            mirror = bootstrap_from_item(
                {
                    "id": entry.get("id"),
                    "title": entry.get("titulo"),
                    "content_type": entry.get("categoria") or entry.get("tipo") or "faq",
                    "content": entry.get("conteudo") or "",
                    "tags": entry.get("tags") or [],
                    "status": entry.get("status") or "ATIVO",
                    "persona_id": entry.get("persona_id"),
                    "file_path": entry.get("link"),
                    "kb_id": entry.get("kb_id"),
                },
                frontmatter={},
                body=entry.get("conteudo") or "",
                persona_id=entry.get("persona_id") or persona_id,
                source_table="kb_entries",
            )
            if mirror:
                counts["kb_mirrored"] += 1
            else:
                counts["kb_skipped"] += 1
        except Exception as exc:
            counts["kb_skipped"] += 1
            counts["errors"].append(f"kb {entry.get('id')}: {str(exc)[:120]}")

    counts["errors"] = counts["errors"][:20]
    return counts


# â”€â”€ Bootstrap from a synced item â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _gather_blob(item: dict, frontmatter: dict, body: str) -> str:
    """Concatenate every field that may carry product/campaign hints."""
    parts: list[str] = []
    for field in ("title", "file_path", "content_type"):
        v = item.get(field) or ""
        parts.append(str(v))
    for v in _normalize_tags(item.get("tags")):
        parts.append(v)
    for k in ("product", "campaign", "campaigns", "title", "tags", "type", "aliases", "synonyms"):
        v = frontmatter.get(k)
        if isinstance(v, list):
            parts.extend([str(x) for x in v])
        elif v is not None:
            parts.append(str(v))
    if body:
        parts.append(body[:2000])
    return " ".join(parts)


def _explicit_relations_from_frontmatter(fm: dict) -> list[tuple[str, str]]:
    """Read graph.relates_to from frontmatter as [type:slug] strings.

    Returns list of (node_type, slug) tuples."""
    rel = []
    g = fm.get("graph") or {}
    if isinstance(g, dict):
        items = g.get("relates_to") or []
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, str) or ":" not in it:
                    continue
                ntype, slug = it.split(":", 1)
                if ntype.strip() and slug.strip():
                    rel.append((ntype.strip().lower(), _slugify(slug)))
    # Also accept common shorthand frontmatter.
    relation_fields = {
        "campaign": "campaign",
        "campaigns": "campaign",
        "product": "product",
        "products": "product",
        "offer": "offer",
        "offers": "offer",
        "copy": "copy",
        "copies": "copy",
        "brand": "brand",
        "brands": "brand",
        "entity": "entity",
        "entities": "entity",
        "persona": "persona",
        "personas": "persona",
        "audience": "audience",
        "audiences": "audience",
    }
    for key, ntype in relation_fields.items():
        for value in _as_list(fm.get(key)):
            if value:
                rel.append((ntype, _slugify(str(value))))

    # Tags like product:<slug> or campaign:<slug> are also explicit graph hints.
    for tag in _normalize_tags(fm.get("tags")):
        if ":" not in tag:
            continue
        ntype, slug = tag.split(":", 1)
        ntype = ntype.strip().lower()
        if ntype in (_TOPIC_NODE_TYPES | {"copy"}) and slug.strip():
            rel.append((ntype, _slugify(slug)))

    dedup: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for ntype, slug in rel:
        key = (ntype, slug)
        if slug and key not in seen:
            seen.add(key)
            dedup.append(key)
    return dedup


def _topic_relations_for_item(item: dict, fm: dict, node_type: str) -> list[tuple[str, str]]:
    """Infer topic nodes from explicit metadata and the item's own type."""
    rel = list(_explicit_relations_from_frontmatter(fm))

    title = item.get("title") or item.get("titulo") or ""
    if node_type in _TOPIC_NODE_TYPES and title:
        rel.append((node_type, _slugify(str(fm.get("slug") or fm.get(node_type) or title))))

    for tag in _normalize_tags(item.get("tags")):
        if ":" not in tag:
            continue
        ntype, slug = tag.split(":", 1)
        if ntype in (_TOPIC_NODE_TYPES | {"copy"}) and slug:
            rel.append((ntype, _slugify(slug)))

    dedup: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for ntype, slug in rel:
        key = (ntype, slug)
        if slug and key not in seen:
            seen.add(key)
            dedup.append(key)
    return dedup


def _relation_title(ntype: str, slug: str, item: dict, fm: dict) -> str:
    explicit_title = fm.get(f"{ntype}_title") or fm.get(f"{ntype}_name")
    if explicit_title:
        for key in (ntype, f"{ntype}s"):
            if any(value and _slugify(str(value)) == slug for value in _as_list(fm.get(key))):
                return str(explicit_title).replace("_", " ").strip()
    for key in (ntype, f"{ntype}s"):
        for value in _as_list(fm.get(key)):
            if value and _slugify(str(value)) == slug:
                return str(value).replace("_", " ").strip().title()
    if _CONTENT_TYPE_TO_NODE.get((item.get("content_type") or "").lower()) == ntype:
        title = item.get("title") or item.get("titulo")
        if title:
            return str(title)
    return _title_from_slug(slug)


def _extract_faq_pairs(text: str) -> list[tuple[str, str]]:
    """Extract Pergunta:/Resposta: blocks from vault, manual items, or KB text."""
    if not text:
        return []
    pattern = re.compile(
        r"Pergunta:\s*(?P<q>.*?)\s*Resposta:\s*(?P<a>.*?)(?=\n\s*Pergunta:|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    pairs: list[tuple[str, str]] = []
    for match in pattern.finditer(text):
        q = re.sub(r"\s+", " ", match.group("q")).strip()
        a = re.sub(r"\s+", " ", match.group("a")).strip()
        if q and a:
            pairs.append((q, a))
    return pairs


def _faq_slug(question: str, topic_slug: Optional[str] = None) -> str:
    base = _slugify(question)[:70].strip("-")
    if topic_slug and base and not base.startswith(f"{topic_slug}-"):
        return f"{topic_slug}-{base}"[:80].strip("-")
    return base[:80] or "faq"


def _extract_briefing_sections(title: str, text: str) -> list[tuple[str, str, str]]:
    """Return (slug, heading, summary) for markdown briefing sections."""
    out: list[tuple[str, str, str]] = []
    if title:
        out.append((_slugify(title), title, (text or "")[:500]))

    lines = (text or "").splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        heading = stripped.lstrip("#").strip()
        if not heading:
            continue
        slug = _slugify(heading)
        snippet = "\n".join(lines[i + 1:i + 6]).strip()[:500]
        out.append((slug, heading, snippet))

    dedup: dict[str, tuple[str, str, str]] = {}
    for row in out:
        dedup[row[0]] = row
    return list(dedup.values())


def _bootstrap_derived_subnodes(
    mirror: dict,
    item: dict,
    frontmatter: dict,
    body: str,
    persona_id: Optional[str],
    topic_nodes: list[dict],
    source_table: str,
) -> None:
    """Create FAQ, briefing and mention subnodes for any related topic."""
    if not mirror or not topic_nodes:
        return

    title = item.get("title") or item.get("titulo") or mirror.get("title") or ""
    tags = _normalize_tags(item.get("tags")) + _normalize_tags(frontmatter.get("tags"))
    content = body or item.get("content") or item.get("conteudo") or ""
    content_type = (item.get("content_type") or item.get("tipo") or "").lower()
    primary_topic = next((n for n in topic_nodes if n.get("node_type") == "product"), topic_nodes[0])
    topic_slug = primary_topic.get("slug") or _slugify(primary_topic.get("title") or "topic")
    topic_tags = {topic_slug, *(n.get("slug") for n in topic_nodes if n.get("slug")), *tags}

    status = _source_status(item, source_table)
    validated = _is_validated_source(source_table, status, mirror.get("node_type"))
    base_meta = {
        "source_status": status,
        "parent_node_id": mirror.get("id"),
        "parent_title": mirror.get("title"),
        "file_path": item.get("file_path"),
    }

    for topic in topic_nodes:
        rel = "about_product" if topic.get("node_type") == "product" else "same_topic_as"
        if topic.get("node_type") == "campaign":
            rel = "part_of_campaign"
        supabase_client.upsert_knowledge_edge(
            topic["id"], mirror["id"], rel, persona_id=persona_id,
        )

    for question, answer in _extract_faq_pairs(content):
        faq = supabase_client.upsert_knowledge_node({
            "persona_id": persona_id,
            "source_table": source_table,
            "source_id": item.get("id"),
            "node_type": "faq",
            "slug": _faq_slug(question, topic_slug),
            "title": question,
            "summary": answer[:500],
            "tags": sorted({"faq", *topic_tags}),
            "status": "validated" if validated else "pending",
            "metadata": {
                **base_meta,
                "derived_from": "faq_block",
                "question": question,
                "answer": answer,
            },
            **_hierarchy_fields("faq", confidence=0.86),
        })
        if faq:
            for topic in topic_nodes:
                topic_type = (topic.get("node_type") or "").lower()
                if topic_type in {"product", "entity", "audience"}:
                    supabase_client.upsert_knowledge_edge(
                        topic["id"], faq["id"], "answers_question", persona_id=persona_id,
                    )
            supabase_client.upsert_knowledge_edge(
                mirror["id"], faq["id"], "contains", persona_id=persona_id,
            )
            supabase_client.upsert_knowledge_edge(
                faq["id"], mirror["id"], "derived_from", persona_id=persona_id,
            )

    is_briefing = content_type == "briefing" or "briefing" in _fold(str(item.get("file_path") or ""))
    if is_briefing:
        for slug, heading, snippet in _extract_briefing_sections(title, content):
            briefing = supabase_client.upsert_knowledge_node({
                "persona_id": persona_id,
                "source_table": source_table,
                "source_id": item.get("id"),
                "node_type": "briefing",
                "slug": slug,
                "title": heading,
                "summary": snippet,
                "tags": sorted({"briefing", *topic_tags}),
                "status": "validated" if validated else "pending",
                "metadata": {**base_meta, "derived_from": "briefing_heading"},
                **_hierarchy_fields("briefing", confidence=0.78),
            })
            if briefing:
                for topic in topic_nodes:
                    supabase_client.upsert_knowledge_edge(
                        topic["id"], briefing["id"], "briefed_by", persona_id=persona_id,
                    )
                supabase_client.upsert_knowledge_edge(
                    mirror["id"], briefing["id"], "contains", persona_id=persona_id,
                )

    # General files related to a topic still get a small mention node. This
    # makes broad brand/tone/audience files visible without treating them as FAQ.
    canonical_types = {"faq", "product", "asset", "copy", "campaign", "briefing", "audience", "brand", "rule", "tone"}
    if content_type not in canonical_types and primary_topic.get("id") != mirror.get("id"):
        mention = supabase_client.upsert_knowledge_node({
            "persona_id": persona_id,
            "source_table": source_table,
            "source_id": item.get("id"),
            "node_type": "mention",
            "slug": f"{topic_slug}-{mirror.get('slug')}"[:80].strip("-"),
            "title": f"{primary_topic.get('title') or _title_from_slug(topic_slug)} - {title}",
            "summary": (content or mirror.get("summary") or "")[:300],
            "tags": sorted({"mention", *topic_tags}),
            "status": "validated" if validated else "pending",
            "metadata": {**base_meta, "derived_from": "topic_mention", "visual_hidden": True},
            **_hierarchy_fields("mention", confidence=0.55),
        })
        if mention:
            for topic in topic_nodes:
                supabase_client.upsert_knowledge_edge(
                    topic["id"], mention["id"], "mentions", persona_id=persona_id,
                )
            supabase_client.upsert_knowledge_edge(
                mention["id"], mirror["id"], "derived_from", persona_id=persona_id,
            )


def bootstrap_from_item(
    item: dict,
    frontmatter: Optional[dict] = None,
    body: str = "",
    persona_id: Optional[str] = None,
    source_table: str = "knowledge_items",
) -> Optional[dict]:
    """Create/refresh the graph nodes & edges that mirror this synced item.

    Always-on heuristics:
      - mirror the item itself as a node (`knowledge_item` or content_type-mapped node).
      - persona membership via belongs_to_persona.
      - explicit product/campaign/brand/entity relations from frontmatter/tags.
      - asset items also get `supports_campaign` / `uses_asset` edges.

    Returns the mirror node (or None when migration 008 isn't applied yet).
    """
    if not item:
        return None
    fm = frontmatter or {}
    persona_id = persona_id or item.get("persona_id")

    try:
        # 1) Mirror node for the item itself.
        content_type = (item.get("content_type") or "").lower()
        node_type = _CONTENT_TYPE_TO_NODE.get(content_type, source_table.rstrip("s"))
        # node_type fallback: "knowledge_items" â†’ "knowledge_item", "kb_entries" â†’ "kb_entrie"
        if node_type == "knowledge_item" or node_type == "kb_entrie":
            node_type = "knowledge_item" if source_table == "knowledge_items" else "kb_entry"

        slug_seed = fm.get("slug") or item.get("file_path") or item.get("title") or str(item.get("id") or "")
        slug = _slugify(slug_seed)[:80] or _slugify(str(item.get("id") or "x"))
        title = item.get("title") or item.get("titulo") or slug
        tags = _normalize_tags(item.get("tags"))
        source_status = _source_status(item, source_table)

        meta: dict = {
            "content_type": content_type or None,
            "file_path": item.get("file_path"),
            "file_type": item.get("file_type"),
            "asset_type": item.get("asset_type") or fm.get("asset_type"),
            "asset_function": item.get("asset_function") or fm.get("asset_function"),
            "kb_id": item.get("kb_id"),
            "source_status": source_status,
        }
        meta.update(_structured_metadata(fm))
        meta.update(traceability_metadata(item, fm))
        meta = {k: v for k, v in meta.items() if v is not None}

        mirror = supabase_client.upsert_knowledge_node({
            "persona_id": persona_id,
            "source_table": source_table,
            "source_id": item.get("id"),
            "node_type": node_type,
            "slug": slug,
            "title": title,
            "summary": (item.get("content") or item.get("conteudo") or "")[:400] or None,
            "tags": tags,
            "metadata": meta,
            "status": "validated" if _is_validated_source(source_table, source_status, node_type) else "pending",
            **_hierarchy_fields(node_type, meta, confidence=item.get("confidence")),
        })
        if not mirror:
            return None  # graph tables missing â€” silently skip, sync still works

        # 2a) Hierarchical parent (Sofia provides metadata.parent_slug).
        # When present and resolvable, this becomes the primary_tree edge so
        # the operator's hierarchical intent (brand â†’ campaign â†’ product â†’ faq)
        # is preserved instead of every node hanging off the persona root.
        parent_slug_raw = fm.get("parent_slug")
        parent_type_hint = (fm.get("parent_type") or "").strip().lower() or None
        explicit_parent_relation = (fm.get("parent_relation") or "").strip().lower() or None
        parent_node: Optional[dict] = None
        if parent_slug_raw and persona_id:
            parent_slug = _slugify(str(parent_slug_raw))
            parent_node = supabase_client.get_knowledge_node_by_slug(
                parent_slug,
                persona_id=persona_id,
                node_type=parent_type_hint,
            )
            if not parent_node and parent_type_hint:
                # Fallback: same persona, any node_type with that slug.
                parent_node = supabase_client.get_knowledge_node_by_slug(
                    parent_slug, persona_id=persona_id,
                )
            if parent_node and parent_node.get("id") == mirror["id"]:
                parent_node = None  # never self-parent

        used_explicit_parent = bool(parent_node and parent_node.get("id"))

        # 2b) Persona fallback. persona_id is the membership source of truth;
        # a visual edge is only needed for the first real node below persona.
        # primary_tree=true is set ONLY when there is no explicit hierarchical
        # parent above â€” otherwise the depth walker would prefer the persona
        # over the real parent and the tree would look flat again.
        if persona_id:
            persona_node = _ensure_persona_root(persona_id)
            if persona_node:
                supabase_client.upsert_knowledge_edge(
                    persona_node["id"],
                    mirror["id"],
                    "belongs_to_persona",
                    persona_id=persona_id,
                    metadata=semantic_edge_metadata(
                        persona_node,
                        mirror,
                        "belongs_to_persona",
                        {
                            "primary_tree": not used_explicit_parent,
                            "visual_hidden": used_explicit_parent,
                            "created_from": "bootstrap_from_item",
                        },
                    ),
                )

        # 2c) Hierarchical parent edge (parent â†’ mirror). Marked as the
        # canonical primary_tree edge so /knowledge/graph-data computes depth
        # correctly. If the parent isn't yet persisted (out-of-order save),
        # the edge is skipped â€” re-running bootstrap after the parent appears
        # will create it (idempotent upsert).
        if used_explicit_parent:
            parent_type = (parent_node.get("node_type") or "").lower()
            relation = (
                explicit_parent_relation
                or _DEFAULT_PARENT_RELATION.get((parent_type, node_type))
                or "contains"
            )
            supabase_client.upsert_knowledge_edge(
                parent_node["id"],
                mirror["id"],
                relation,
                persona_id=persona_id,
                weight=1.0,
                metadata=semantic_edge_metadata(
                    parent_node,
                    mirror,
                    relation,
                    {
                        "primary_tree": True,
                        "created_from": "bootstrap_parent_slug",
                        "parent_slug": parent_node.get("slug"),
                        "parent_type": parent_type,
                    },
                ),
            )

        # 3) Tag nodes + has_tag edges.
        for tag in tags:
            tnode = supabase_client.upsert_knowledge_node({
                "persona_id": persona_id,
                "node_type": "tag",
                "slug": _slugify(tag),
                "title": tag,
                **_hierarchy_fields("tag"),
            })
            if tnode:
                supabase_client.upsert_knowledge_edge(
                    mirror["id"],
                    tnode["id"],
                    "has_tag",
                    persona_id=persona_id,
                    metadata={
                        "graph_layer": "semantic_tags",
                        "primary_tree": False,
                        "visual_hidden": True,
                        "created_from": "bootstrap_tags",
                    },
                )

        # 4) Explicit topic relations from frontmatter/tags plus the item's own
        # type when it is itself a product/campaign/brand/entity.
        related: list[tuple[str, str]] = _topic_relations_for_item(item, fm, node_type)

        # 6) For each related (type, slug), upsert node + connect with the right edge.
        # FAQ is terminal in commercial branches: topic nodes point into FAQ.
        # The only valid outgoing FAQ edge is publication to Embedded.
        related_nodes: dict[tuple[str, str], dict] = {}
        for ntype, rslug in related:
            related_title = _relation_title(ntype, rslug, item, fm)
            target_meta = {"auto": True}
            if ntype == node_type and rslug == slug:
                target_meta.update(meta)
            target = supabase_client.upsert_knowledge_node({
                "persona_id": persona_id,
                "node_type": ntype,
                "slug": rslug,
                "title": related_title,
                "tags": [rslug],
                "metadata": target_meta,
                **_hierarchy_fields(ntype, target_meta),
            })
            if not target:
                continue
            related_nodes[(ntype, rslug)] = target
            ensure_main_tree_connection(
                target,
                persona_id=persona_id,
                relation_type="belongs_to_persona",
            )

            # Default relation for non-special content types.
            relation = "about_product" if ntype == "product" else (
                "part_of_campaign" if ntype == "campaign" else "same_topic_as"
            )

            if content_type == "asset":
                if ntype == "campaign":
                    relation = "supports_campaign"
                elif ntype == "product":
                    # campaign/product â†’ asset
                    supabase_client.upsert_knowledge_edge(
                        target["id"], mirror["id"], "uses_asset", persona_id=persona_id,
                    )
                    continue
            elif content_type == "faq" and ntype in {"product", "offer", "copy", "campaign", "audience", "brand", "entity"}:
                supabase_client.upsert_knowledge_edge(
                    target["id"],
                    mirror["id"],
                    "answers_question",
                    persona_id=persona_id,
                    metadata=semantic_edge_metadata(
                        target,
                        mirror,
                        "answers_question",
                        {
                            "primary_tree": False,
                            "created_from": "bootstrap_faq_reference",
                        },
                    ),
                )
                continue
            elif content_type == "copy" and ntype in ("product", "campaign"):
                relation = "supports_copy"

            supabase_client.upsert_knowledge_edge(
                mirror["id"], target["id"], relation, persona_id=persona_id,
            )
            if ntype == "product":
                supabase_client.upsert_knowledge_edge(
                    target["id"], mirror["id"], "about_product", persona_id=persona_id,
                )

        # 6) Product/campaign pairs from the same item are connected generically.
        products = [n for (ntype, _), n in related_nodes.items() if ntype == "product"]
        campaigns = [n for (ntype, _), n in related_nodes.items() if ntype == "campaign"]
        for product in products:
            for campaign in campaigns:
                supabase_client.upsert_knowledge_edge(
                    product["id"], campaign["id"], "part_of_campaign", persona_id=persona_id,
                )

        # 7) Derived subnodes for FAQ blocks, briefing headings and broad mentions.
        _bootstrap_derived_subnodes(
            mirror=mirror,
            item=item,
            frontmatter=fm,
            body=body,
            persona_id=persona_id,
            topic_nodes=[n for n in related_nodes.values() if n.get("node_type") in _TOPIC_NODE_TYPES],
            source_table=source_table,
        )

        # 8) Agent visibility from frontmatter.
        viz = fm.get("agent_visibility") or item.get("agent_visibility") or []
        if isinstance(viz, list):
            for role in viz:
                role_slug = _slugify(str(role))
                if not role_slug or role_slug not in _KNOWN_ROLES:
                    continue
                rnode = supabase_client.upsert_knowledge_node({
                    "persona_id": persona_id,
                    "node_type": "audience",   # role buckets live alongside audiences
                    "slug": f"role-{role_slug}",
                    "title": role_slug.upper(),
                    "metadata": {"role": role_slug},
                    **_hierarchy_fields("audience"),
                })
                if rnode:
                    supabase_client.upsert_knowledge_edge(
                        mirror["id"], rnode["id"], "visible_to_agent", persona_id=persona_id,
                    )

        return mirror
    except Exception as exc:
        # Never let graph maintenance abort vault sync or item promotion.
        logger.warning("bootstrap_from_item failed (item=%s): %s", item.get("id"), exc)
        return None


# â”€â”€ Chat context resolver â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _compute_graph_distances(
    seed_ids: set[str],
    nodes: list[dict],
    edges: list[dict],
    max_distance: int = 4,
) -> tuple[dict[str, int], dict[str, list[dict]]]:
    """BFS over knowledge_edges to label each node with graph_distance and the
    path used to reach it from the nearest seed.

    Returns:
      (distance, path) where:
        distance[node_id] -> int  (0 for seeds, +1 per hop; absent => unreached)
        path[node_id]     -> list[{"node_id","slug","title","relation_type"|None}]
                             (always starts at a seed and ends at node_id)
    """
    if not seed_ids or not nodes:
        return {}, {}

    node_index: dict[str, dict] = {n["id"]: n for n in nodes if n.get("id")}
    # Adjacency: for each node, list of (neighbor_id, relation_type, direction)
    # Edges are directional in the schema; for path reconstruction we walk
    # both directions, but tag the relation correctly so callers can render
    # "A --rel--> B" or "A <--rel-- B".
    adj: dict[str, list[tuple[str, str, str]]] = {}
    for e in edges:
        src, tgt = e.get("source_node_id"), e.get("target_node_id")
        rel = e.get("relation_type") or "related"
        if not src or not tgt:
            continue
        adj.setdefault(src, []).append((tgt, rel, "out"))
        adj.setdefault(tgt, []).append((src, rel, "in"))

    distance: dict[str, int] = {}
    predecessor: dict[str, tuple[str, str, str]] = {}  # nid -> (parent_id, rel, direction)
    queue: list[tuple[str, int]] = []

    for sid in seed_ids:
        if sid in node_index:
            distance[sid] = 0
            queue.append((sid, 0))

    head = 0
    while head < len(queue):
        nid, d = queue[head]
        head += 1
        if d >= max_distance:
            continue
        for (nb, rel, direction) in adj.get(nid, []):
            if nb in distance or nb not in node_index:
                continue
            distance[nb] = d + 1
            predecessor[nb] = (nid, rel, direction)
            queue.append((nb, d + 1))

    def _path_for(nid: str) -> list[dict]:
        chain: list[dict] = []
        cur = nid
        guard = 0
        while cur is not None and guard < max_distance + 2:
            n = node_index.get(cur, {})
            entry: dict = {
                "node_id": cur,
                "slug": n.get("slug"),
                "title": n.get("title"),
                "node_type": n.get("node_type"),
                "relation_type": None,
                "direction": None,
            }
            if cur in predecessor:
                _, rel, direction = predecessor[cur]
                entry["relation_type"] = rel
                entry["direction"] = direction
            chain.append(entry)
            cur = predecessor.get(cur, (None,))[0] if cur in predecessor else None
            guard += 1
        chain.reverse()
        return chain

    paths: dict[str, list[dict]] = {nid: _path_for(nid) for nid in distance}
    return distance, paths


def _candidate_terms_from_messages(messages: Iterable[dict], horizon_chars: int = 2000) -> list[str]:
    """Pull the most recent words/phrases from a conversation that look like
    proper nouns or quoted product names. We're intentionally simple here â€”
    actual matching is gated by the canonical product/campaign list anyway."""
    blob = " ".join(((m or {}).get("texto") or "") for m in messages)[-horizon_chars:]
    if not blob:
        return []
    # Words >= 3 chars, lowercase, deduped, preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for token in re.findall(r"[A-Za-zÃ€-Ã¿0-9][A-Za-zÃ€-Ã¿0-9'-]{2,}", blob):
        t = token.lower()
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _detect_terms(
    user_text: Optional[str],
    messages: list[dict],
    persona_id: Optional[str],
) -> list[str]:
    """Return topic terms to look up in the graph.

    Matching is driven by existing product/campaign/brand/entity nodes, using
    slug, title, tags and optional metadata aliases/synonyms. Recent message
    history is *always* included so the sidebar surfaces relevant knowledge
    even when the latest client message is short (e.g., a name or yes/no).
    """
    user_terms: list[str] = []
    if user_text:
        user_terms = re.findall(r"[A-Za-zÃ€-Ã¿0-9][A-Za-zÃ€-Ã¿0-9'-]{2,}", user_text.lower())
    msg_terms = _candidate_terms_from_messages(messages)

    seen_lower: set[str] = set()
    raw_terms: list[str] = []
    for t in [*user_terms, *msg_terms]:
        key = t.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        raw_terms.append(t)

    if not raw_terms:
        return []

    raw_blob = _fold(" ".join(raw_terms))
    raw_set = set(re.findall(r"[a-z0-9][a-z0-9'-]{1,}", raw_blob))

    detected: list[str] = []

    # Graph-driven: any topic node title, slug, tag, or alias that appears.
    try:
        canon_nodes = supabase_client.list_knowledge_nodes_by_type(
            ["product", "campaign", "brand", "entity", "audience"], persona_id=persona_id, limit=500,
        )
        for n in canon_nodes:
            slug = _fold(n.get("slug") or "")
            title = _fold(n.get("title") or "")
            tags = [_fold(t) for t in _normalize_tags(n.get("tags"))]
            meta = n.get("metadata") or {}
            aliases = [_fold(a) for a in _as_list(meta.get("aliases") or meta.get("synonyms")) if a]
            slug_parts = [p for p in slug.split("-") if len(p) >= 3]
            # Substring match (slug/title in blob) handles exact mentions.
            # Prefix match against tokens handles PT plural/singular pairs
            # (e.g., "modal" vs "modais", "papel" vs "papeis").
            matched = (
                (slug and slug in raw_blob)
                or (title and title in raw_blob)
                or any(part in raw_set for part in slug_parts)
                or any(_prefix_overlap(part, raw_set) for part in slug_parts)
                or (slug and _prefix_overlap(slug, raw_set))
                or any(tag and (tag in raw_set or tag in raw_blob or _prefix_overlap(tag, raw_set)) for tag in tags)
                or any(alias and (alias in raw_blob or _prefix_overlap(alias, raw_set)) for alias in aliases)
            )
            if matched:
                if n.get("title") and n["title"] not in detected:
                    detected.append(n["title"])
    except Exception as exc:
        logger.warning("_detect_terms graph lookup failed: %s", exc)

    out: list[str] = []
    seen: set[str] = set()
    for term in detected:
        key = _fold(term)
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
    return out


def _infer_persona_from_messages(
    lead_ref: int,
    user_text: Optional[str],
    interesse_produto: Optional[str],
    min_dominance: float = 0.70,
    min_hits: int = 1,
) -> Optional[str]:
    """Walk the knowledge graph to infer which persona owns the conversation.

    Strategy:
      1. Pull recent messages + lead.interesse_produto + user_text.
      2. Tokenize into candidate terms (same logic as _detect_terms).
      3. Match each term against ALL personas' knowledge_nodes (no scope).
      4. Score per-persona by counting matched nodes (weighted by node_type
         priority â€” product/brand/campaign/entity > faq/copy > tag/mention).
      5. Return the persona_id whose share of total hits >= min_dominance and
         whose absolute hit count >= min_hits. Otherwise None.

    This is intentionally conservative â€” when the conversation is ambiguous
    or short, we keep the safety block instead of guessing a persona and
    leaking another client's knowledge.
    """
    try:
        messages = supabase_client.get_messages(str(lead_ref), limit=30) or []
    except Exception:
        messages = []

    blob_parts: list[str] = []
    if user_text:
        blob_parts.append(user_text)
    if interesse_produto:
        blob_parts.append(interesse_produto)
    blob_parts.extend((m or {}).get("texto", "") for m in messages)
    blob = " ".join(p for p in blob_parts if p)
    if not blob.strip():
        return None

    # Tokenize same way _detect_terms does â€” but here we run global lookup.
    raw_tokens = re.findall(r"[A-Za-zÃ€-Ã¿0-9][A-Za-zÃ€-Ã¿0-9'-]{2,}", blob.lower())
    if not raw_tokens:
        return None
    raw_blob = _fold(" ".join(raw_tokens))
    raw_set = set(re.findall(r"[a-z0-9][a-z0-9'-]{1,}", raw_blob))

    # Pull canonical topic nodes globally and score by persona.
    try:
        canon_nodes = supabase_client.list_knowledge_nodes_by_type(
            ["product", "campaign", "brand", "entity"], persona_id=None, limit=1000,
        )
    except Exception:
        return None

    # Score map: persona_id -> weighted hits
    type_weight = {"product": 4, "brand": 3, "campaign": 3, "entity": 2}
    persona_score: dict[str, float] = {}
    matched_per_persona: dict[str, int] = {}

    for n in canon_nodes:
        pid = n.get("persona_id")
        if not pid:
            continue
        slug = _fold(n.get("slug") or "")
        title = _fold(n.get("title") or "")
        tags = [_fold(t) for t in _normalize_tags(n.get("tags"))]
        meta = n.get("metadata") or {}
        aliases = [_fold(a) for a in _as_list(meta.get("aliases") or meta.get("synonyms")) if a]
        slug_parts = [p for p in slug.split("-") if len(p) >= 3]
        matched = (
            (slug and slug in raw_blob)
            or (title and title in raw_blob)
            or any(part in raw_set for part in slug_parts)
            or any(_prefix_overlap(part, raw_set) for part in slug_parts)
            or (slug and _prefix_overlap(slug, raw_set))
            or any(tag and (tag in raw_set or tag in raw_blob or _prefix_overlap(tag, raw_set)) for tag in tags)
            or any(alias and (alias in raw_blob or _prefix_overlap(alias, raw_set)) for alias in aliases)
        )
        if matched:
            w = type_weight.get(n.get("node_type", ""), 1)
            persona_score[pid] = persona_score.get(pid, 0.0) + w
            matched_per_persona[pid] = matched_per_persona.get(pid, 0) + 1

    if not persona_score:
        return None
    total = sum(persona_score.values())
    best_pid, best_score = max(persona_score.items(), key=lambda kv: kv[1])
    dominance = best_score / total if total > 0 else 0
    if dominance >= min_dominance and matched_per_persona.get(best_pid, 0) >= min_hits:
        logger.info(
            "knowledge_graph: inferred persona for lead_ref=%s -> %s (dominance=%.2f, hits=%d)",
            lead_ref, best_pid, dominance, matched_per_persona.get(best_pid),
        )
        return best_pid
    return None


def get_chat_context(
    lead_ref: Optional[int],
    persona_id: Optional[str] = None,
    user_text: Optional[str] = None,
    limit: int = 12,
) -> dict:
    """Resolve a knowledge bundle relevant to a lead's conversation.

    Output shape (always present, even when graph is empty):
        {query_terms, nodes, edges, kb_entries, assets, summary}
    """
    lead_data: dict = {}
    if lead_ref:
        try:
            lead_data = supabase_client.get_lead_by_ref(int(lead_ref)) or {}
        except Exception:
            lead_data = {}
    if not persona_id:
        persona_id = lead_data.get("persona_id")

    persona_was_inferred = False
    if not persona_id and lead_ref:
        # Walk the graph: try to infer persona from message history. Only
        # accepted when one persona dominates the matches â€” otherwise we
        # keep the safety block to avoid leaking another client's knowledge.
        inferred = _infer_persona_from_messages(
            int(lead_ref),
            user_text=user_text,
            interesse_produto=(lead_data.get("interesse_produto") or "").strip() or None,
        )
        if inferred:
            persona_id = inferred
            persona_was_inferred = True
            # Backfill the lead so downstream calls (and future requests)
            # don't repeat this work. Best-effort: ignore failures.
            try:
                supabase_client.update_lead(int(lead_ref), {"persona_id": inferred})
            except Exception as exc:
                logger.warning("knowledge_graph: persona backfill failed for lead %s: %s", lead_ref, exc)

    if not persona_id:
        # Sem persona definida (lead sem vÃ­nculo OU chamada sem escopo) o
        # contexto nÃ£o pode ser global â€” devolverÃ­amos conhecimento de outro
        # cliente. Bloqueia explicitamente.
        reason = (
            "Lead sem persona vinculada e conversa nÃ£o bate com nenhum cliente conhecido"
            if lead_ref
            else "Persona nÃ£o especificada"
        )
        return {
            "query_terms": [],
            "entities": [],
            "intent": "fallback_text_search",
            "nodes": [],
            "edges": [],
            "kb_entries": [],
            "golden_dataset": [],
            "legacy_fallback_used": False,
            "assets": [],
            "similar": [],
            "validated": {"nodes": [], "kb_entries": [], "assets": []},
            "unvalidated": {"nodes": [], "kb_entries": [], "assets": []},
            "summary": f"{reason}; contexto de conhecimento bloqueado para evitar mistura entre clientes.",
            "persona_inferred": False,
        }
    lead_interest = (lead_data.get("interesse_produto") or "").strip()

    # 1. Recent messages provide context when no explicit q.
    messages: list[dict] = []
    if lead_ref:
        try:
            messages = supabase_client.get_messages(str(lead_ref), limit=20) or []
        except Exception:
            messages = []

    query_text = " ".join(x for x in [user_text, lead_interest] if x)
    terms = _detect_terms(query_text or None, messages, persona_id)
    try:
        golden_chunks = supabase_client.search_active_rag_chunks(
            persona_id=persona_id,
            query=query_text,
            limit=limit,
        )
    except Exception as exc:
        logger.warning("search_active_rag_chunks failed: %s", exc)
        golden_chunks = []

    # 2. Match nodes by each term.
    seed_nodes: dict[str, dict] = {}
    for term in terms:
        try:
            for n in supabase_client.find_knowledge_nodes(term, persona_id=persona_id, limit=8):
                seed_nodes[n["id"]] = n
        except Exception as exc:
            logger.warning("find_knowledge_nodes failed for %r: %s", term, exc)

    # 3. Neighbours. We expand two hops so topic nodes can surface FAQ,
    # briefing and source item subnodes without requiring every item to match
    # the text directly.
    nodes: list[dict] = []
    edges: list[dict] = []
    if seed_nodes:
        try:
            n_full, e_full = supabase_client.get_knowledge_neighbors(list(seed_nodes.keys()))
            node_map = {n["id"]: n for n in n_full if n.get("id")}
            edge_map = {e["id"]: e for e in e_full if e.get("id")}
            if node_map:
                n2, e2 = supabase_client.get_knowledge_neighbors(list(node_map.keys()))
                node_map.update({n["id"]: n for n in n2 if n.get("id")})
                edge_map.update({e["id"]: e for e in e2 if e.get("id")})
            nodes = list(node_map.values())[:limit * 6]
            edges = list(edge_map.values())[: limit * 10]
        except Exception as exc:
            logger.warning("get_knowledge_neighbors failed: %s", exc)
            nodes = list(seed_nodes.values())
            edges = []

    nodes = [_decorate_node(n) for n in nodes]
    existing_source_ids = {str(n.get("source_id")) for n in nodes if n.get("source_id")}
    nodes.extend(_fallback_nodes_from_tables(terms, persona_id, existing_source_ids))

    # 3.b BFS over edges so every node carries graph_distance + path back to
    # the nearest seed. Fallback nodes (no edges) keep distance=None.
    distance_map, path_map = _compute_graph_distances(
        seed_ids=set(seed_nodes.keys()),
        nodes=nodes,
        edges=edges,
    )
    for n in nodes:
        nid = n.get("id")
        if nid in distance_map:
            n["graph_distance"] = distance_map[nid]
            n["path"] = path_map.get(nid) or []
            n["path_slugs"] = [step.get("slug") for step in (path_map.get(nid) or [])]
            n["path_relations"] = [
                step.get("relation_type")
                for step in (path_map.get(nid) or [])
                if step.get("relation_type")
            ]
        else:
            # Fallback nodes / unreached nodes: keep field present but null
            n["graph_distance"] = None
            n["path"] = []
            n["path_slugs"] = []
            n["path_relations"] = []

    # 4. Split into convenient buckets for the UI.
    by_type: dict[str, list[dict]] = {}
    for n in nodes:
        by_type.setdefault(n.get("node_type", ""), []).append(n)

    kb_entry_nodes = [
        n for n in nodes
        if n.get("source_table") == "kb_entries"
        or n.get("node_type") in {"kb_entry", "faq", "copy"}
    ]
    # Batch fetch â€” avoids N+1 round-trip per kb_entry node.
    kb_source_ids = [
        str(n["source_id"])
        for n in kb_entry_nodes
        if n.get("source_id") and not golden_chunks
    ]
    try:
        kb_rows_by_id = supabase_client.get_kb_entries_by_ids(kb_source_ids) if kb_source_ids else {}
    except Exception as exc:
        logger.warning("get_kb_entries_by_ids failed: %s", exc)
        kb_rows_by_id = {}

    kb_entries: list[dict] = [
        {
            "id": chunk.get("id"),
            "rag_entry_id": chunk.get("rag_entry_id"),
            "titulo": (chunk.get("entry") or {}).get("title"),
            "conteudo": chunk.get("chunk_text") or chunk.get("chunk_summary") or "",
            "tipo": (chunk.get("entry") or {}).get("content_type") or "faq",
            "node_type": (chunk.get("entry") or {}).get("content_type") or "faq",
            "validation_status": "validated",
            "validated": True,
            "source": "golden_dataset",
            "metadata": {
                **((chunk.get("entry") or {}).get("metadata") or {}),
                **(chunk.get("metadata") or {}),
            },
        }
        for chunk in golden_chunks
    ]
    if not kb_entries:
        for n in kb_entry_nodes:
            sid = n.get("source_id")
            row = kb_rows_by_id.get(str(sid)) if sid else None
            if row:
                row_node_type = _tipo_to_node_type(row.get("tipo") or row.get("categoria") or "") or n.get("node_type")
                kb_entries.append({
                    **row,
                    "node_type": row_node_type,
                    "validation_status": "validated" if _is_validated_source("kb_entries", row.get("status"), row_node_type) else "pending",
                    "validated": _is_validated_source("kb_entries", row.get("status"), row_node_type),
                    "link_target": n.get("link_target") or _link_target(n),
                    "source": "kb_entries_fallback",
                })
                continue
            kb_entries.append({
                "id": n.get("id"),
                "titulo": n.get("title"),
                "conteudo": n.get("summary") or "",
                "tipo": n.get("node_type"),
                "tags": _normalize_tags(n.get("tags")),
                "node_type": n.get("node_type"),
                "validation_status": n.get("validation_status") or "pending",
                "validated": bool(n.get("validated")),
                "link_target": n.get("link_target"),
                "source": "graph_node_fallback",
            })

    assets: list[dict] = []
    for n in by_type.get("asset", []):
        meta = n.get("metadata") or {}
        path = meta.get("file_path")
        url = f"/api-brain/knowledge/file?path={path}" if path else None
        assets.append({
            "id": n.get("id"),
            "title": n.get("title"),
            "asset_type": meta.get("asset_type"),
            "asset_function": meta.get("asset_function"),
            "file_path": path,
            "url": url,
            "tags": _normalize_tags(n.get("tags")),
            "validation_status": n.get("validation_status") or "pending",
            "validated": bool(n.get("validated")),
            "link_target": n.get("link_target") or url,
        })

    # 5. Entities = the seed nodes that originally matched the query.
    entities = [
        {
            "id": n.get("id"),
            "type": n.get("node_type"),
            "slug": n.get("slug"),
            "title": n.get("title"),
            "link_target": _link_target(n),
        }
        for n in seed_nodes.values()
    ]

    # 6. Intent â€” simple keyword heuristic over the user text, then graph hits.
    intent = _detect_intent(user_text or " ".join(((m or {}).get("texto") or "") for m in messages),
                            by_type, assets)

    # 7. Human-readable summary.
    parts: list[str] = []
    if terms:
        parts.append("Termos detectados: " + ", ".join(terms))
    products = [n["title"] for n in by_type.get("product", [])]
    campaigns = [n["title"] for n in by_type.get("campaign", [])]
    if products:
        parts.append("Produtos: " + ", ".join(products))
    if campaigns:
        parts.append("Campanhas: " + ", ".join(campaigns))
    if assets:
        parts.append(f"{len(assets)} asset(s) relacionado(s)")
    summary = " Â· ".join(parts)

    validated_nodes = [n for n in nodes if n.get("validated")]
    unvalidated_nodes = [n for n in nodes if not n.get("validated")]

    # 8. Similarity ranking â€” strictly by graph_distance, with type as tiebreak
    # (product/campaign/faq/copy/asset before tag/mention). Seeds excluded.
    seed_id_set = set(seed_nodes.keys())
    type_priority = {"product": 0, "campaign": 1, "brand": 1, "faq": 2,
                     "copy": 3, "asset": 4, "briefing": 5, "rule": 6,
                     "tone": 6, "audience": 7, "tag": 9, "mention": 9}
    similar = sorted(
        [
            {
                "node_id": n.get("id"),
                "node_type": n.get("node_type"),
                "slug": n.get("slug"),
                "title": n.get("title"),
                "graph_distance": n.get("graph_distance"),
                "path": n.get("path") or [],
                "path_slugs": n.get("path_slugs") or [],
                "path_relations": n.get("path_relations") or [],
                "validated": bool(n.get("validated")),
                "link_target": n.get("link_target"),
            }
            for n in nodes
            if n.get("id") not in seed_id_set
            and n.get("graph_distance") is not None
        ],
        key=lambda r: (
            r["graph_distance"] if r["graph_distance"] is not None else 99,
            type_priority.get(r["node_type"], 8),
            (r["title"] or "").lower(),
        ),
    )[: limit * 2]

    return {
        "query_terms": terms,
        "entities": entities,
        "intent": intent,
        "nodes": nodes,
        "edges": edges,
        "kb_entries": kb_entries,
        "golden_dataset": kb_entries if golden_chunks else [],
        "legacy_fallback_used": not bool(golden_chunks),
        "assets": assets,
        "similar": similar,
        "validated": {
            "nodes": validated_nodes,
            "kb_entries": [e for e in kb_entries if e.get("validated")],
            "assets": [a for a in assets if a.get("validated")],
        },
        "unvalidated": {
            "nodes": unvalidated_nodes,
            "kb_entries": [e for e in kb_entries if not e.get("validated")],
            "assets": [a for a in assets if not a.get("validated")],
        },
        "summary": summary,
        "persona_id": persona_id,
        "persona_inferred": persona_was_inferred,
        "conversation_runtime": (
            (lead_data.get("metadata") or {}).get("conversation_runtime") or {}
        ),
        "evidence_node_ids": (
            ((lead_data.get("metadata") or {}).get("conversation_runtime") or {}).get(
                "evidence_node_ids"
            )
            or []
        ),
    }


def with_operator_context(context: dict, *, limit: int = 12) -> dict:
    """Add a stable operator projection while preserving all legacy arrays."""
    nodes = list(context.get("nodes") or [])
    evidence_ids = {
        str(value)
        for value in (
            context.get("evidence_node_ids")
            or ((context.get("conversation_runtime") or {}).get("evidence_node_ids"))
            or []
        )
    }
    type_priority = {
        "faq": 0, "rule": 1, "tone": 2, "product": 3,
        "product_group": 4, "copy": 5, "campaign": 6,
        "brand": 7, "persona": 8,
    }

    def rank(node: dict) -> tuple:
        distance = node.get("graph_distance")
        return (
            0 if str(node.get("id") or "") in evidence_ids else 1,
            type_priority.get(str(node.get("node_type") or ""), 20),
            distance if isinstance(distance, int) else 999,
            0 if node.get("validated") else 1,
            str(node.get("id") or ""),
        )

    ranked_nodes = sorted(nodes, key=rank)
    primary_nodes = (
        [node for node in ranked_nodes if str(node.get("id") or "") in evidence_ids]
        if evidence_ids
        else ranked_nodes[:limit]
    )
    primary = []
    for node in primary_nodes[:limit]:
        primary.append({
            "id": node.get("id"),
            "node_type": node.get("node_type"),
            "title": node.get("title"),
            "markdown": (
                (node.get("metadata") or {}).get("markdown")
                or node.get("summary")
                or ""
            ),
            "graph_distance": node.get("graph_distance"),
            "validated": bool(node.get("validated")),
            "path": node.get("path") or [],
            "used_in_last_decision": str(node.get("id") or "") in evidence_ids,
        })
    faq_rules = []
    for node in ranked_nodes:
        if node.get("node_type") not in {"faq", "rule", "tone"}:
            continue
        faq_rules.append({
            "id": node.get("id"),
            "node_type": node.get("node_type"),
            "title": node.get("title"),
            "markdown": (
                (node.get("metadata") or {}).get("markdown")
                or node.get("summary")
                or ""
            ),
            "graph_distance": node.get("graph_distance"),
            "validated": bool(node.get("validated")),
            "path": node.get("path") or [],
            "used_in_last_decision": str(node.get("id") or "") in evidence_ids,
        })
        if len(faq_rules) >= limit:
            break
    best_path = next(
        (item.get("path") or [] for item in primary if item.get("path")),
        [],
    )
    return {
        **context,
        "operator_context": {
            "primary": primary,
            "faq_rules": faq_rules,
            "graph_path": best_path,
            "ranking": "last_decision,exact_match,faq,distance,validation,id",
        },
    }


_INTENT_ASSET_KEYWORDS = re.compile(
    r"\b(imagem|imagens|foto|fotos|asset|assets|m[iÃ­]dia|midia|banner|hero|story|catalog|catÃ¡logo|video|v[iÃ­]deo)\b",
    re.IGNORECASE,
)


def _detect_intent(text: str, by_type: dict, assets: list[dict]) -> str:
    """Cheap rule-based intent classifier (no LLM).

    Order of precedence:
      1. asset_request â€” user mentions media/images/etc
      2. product_inquiry â€” there's at least one product node
      3. campaign_inquiry â€” there's at least one campaign node
      4. kb_lookup â€” only KB entries matched
      5. fallback_text_search â€” nothing matched.
    """
    blob = (text or "").lower()
    if assets and _INTENT_ASSET_KEYWORDS.search(blob):
        return "asset_request"
    if by_type.get("product"):
        return "product_inquiry"
    if by_type.get("campaign"):
        return "campaign_inquiry"
    if by_type.get("faq") or by_type.get("copy") or by_type.get("kb_entry"):
        return "kb_lookup"
    return "fallback_text_search"
    if parent == "persona":
        return "contains"
    if parent == "briefing" and child == "audience":
        return "contains"
    if parent == "audience" and child == "product":
        return "offers_product"

