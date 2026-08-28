"""Canonical fractal graph taxonomy.

Single source of truth for node types, relation types, and edge kinds. Reads
from the registry tables (`knowledge_node_type_registry`,
`knowledge_relation_type_registry`) after migration 039 introduced the
`canonical`, `alias_of`, `edge_kind` and `primary_one_to_one` columns.

The preferred create-path hierarchy is:

    persona -> brand -> briefing -> campaign -> audience
            -> product_group? -> product -> offer -> copy
            -> {faq, gallery}

Legacy `brand -> campaign -> briefing` / `campaign -> briefing` edges remain
accepted for old content, but the Create path prefers the serial
brand->briefing->campaign->audience spine so briefing is rendered in series.

Asset is a lateral layer that may connect to any node or to another asset via
`asset_pending` / `asset_approved` / `asset_related`.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from .supabase_client import get_client


# ── Static canonical fallbacks (in sync with migration 039) ──────────
# These act as a fallback in case the registries are unreachable.

CANONICAL_NODE_TYPES: tuple[str, ...] = (
    "persona",
    "brand",
    "briefing",
    "campaign",
    "audience",
    "product_group",
    "product",
    "offer",
    "copy",
    "faq",
    "gallery",
    "embedded",
    "marketing_workspace",
    "asset",
    "rule",
    "tone",
)

NODE_TYPE_ALIASES: dict[str, str] = {
    "product_collection": "product_group",
    "category": "product_group",
}

PRIMARY_CHAIN: tuple[tuple[str, str, str], ...] = (
    ("persona",       "brand",         "persona_has_brand"),
    ("brand",         "briefing",      "brand_has_briefing"),
    ("briefing",      "campaign",      "briefing_has_campaign"),
    ("campaign",      "audience",      "campaign_has_audience"),
    ("audience",      "product_group", "audience_has_product_group"),
    ("audience",      "product",       "offers_product"),
    ("product_group", "product",       "product_group_has_product"),
    ("product",       "offer",         "product_has_offer"),
    ("offer",         "copy",          "offer_has_copy"),
    ("copy",          "faq",           "copy_has_faq"),
    ("copy",          "gallery",       "copy_has_gallery"),
    # Legacy alternatives kept valid so older content can still be repaired.
    ("brand",         "campaign",      "contains"),
    ("briefing",      "audience",      "contains"),
)

# (source_type, target_type) tuples where the primary edge is 1:1 (vs N:N).
PRIMARY_ONE_TO_ONE: frozenset[tuple[str, str]] = frozenset({
    ("persona", "brand"),
    ("copy", "faq"),
    ("copy", "gallery"),
})

EDGE_KINDS: tuple[str, ...] = (
    "primary",
    "secondary",
    "asset_pending",
    "asset_approved",
)

V21_RELATIONS: tuple[dict[str, object], ...] = (
    {"relation_type": "contains", "label": "contém", "edge_kind": "primary", "default_weight": 0.90},
    {"relation_type": "targets", "label": "direciona para", "edge_kind": "secondary", "default_weight": 0.85},
    {"relation_type": "represents", "label": "representa", "edge_kind": "secondary", "default_weight": 0.90},
    {"relation_type": "uses_asset", "label": "usa asset", "edge_kind": "secondary", "default_weight": 0.85},
    {"relation_type": "supports", "label": "apoia", "edge_kind": "secondary", "default_weight": 0.80},
    {"relation_type": "answers", "label": "responde", "edge_kind": "secondary", "default_weight": 1.00},
    {"relation_type": "applies_to", "label": "aplica-se a", "edge_kind": "secondary", "default_weight": 0.90},
    {"relation_type": "derived_from", "label": "derivado de", "edge_kind": "secondary", "default_weight": 0.65},
    {"relation_type": "references", "label": "referencia", "edge_kind": "secondary", "default_weight": 0.55},
    {"relation_type": "publishes_to", "label": "publica em", "edge_kind": "secondary", "default_weight": 1.00},
)


# ── Cached registry snapshot ─────────────────────────────────────────

_CACHE_TTL_SECONDS = 60.0
_cache: dict[str, object] = {
    "fetched_at": 0.0,
    "node_types": [],
    "relations": [],
}


def _refresh_cache() -> None:
    client = get_client()
    node_rows = (
        client.table("knowledge_node_type_registry")
        .select(
            "node_type,label,description,default_level,default_importance,"
            "color,icon,sort_order,alias_of,deprecated_at,canonical,active"
        )
        .eq("active", True)
        .order("sort_order")
        .execute()
        .data
        or []
    )
    rel_rows = (
        client.table("knowledge_relation_type_registry")
        .select(
            "relation_type,label,inverse_label,source_node_types,"
            "target_node_types,default_weight,directional,sort_order,"
            "edge_kind,primary_one_to_one,canonical,active"
        )
        .eq("active", True)
        .order("sort_order")
        .execute()
        .data
        or []
    )
    _cache["fetched_at"] = time.monotonic()
    _cache["node_types"] = node_rows
    _cache["relations"] = rel_rows


def _ensure_cache() -> None:
    if (os.environ.get("KNOWLEDGE_TAXONOMY_OFFLINE") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        # Unit/CI graph validation is deterministic from the static canonical
        # registry and must not wait on an absent PostgREST endpoint.
        _cache["fetched_at"] = time.monotonic()
        return
    fetched = float(_cache.get("fetched_at") or 0.0)
    if (time.monotonic() - fetched) > _CACHE_TTL_SECONDS:
        try:
            _refresh_cache()
        except Exception:
            # If Supabase is briefly unavailable, fall back to whatever we
            # cached previously (or to empty lists, which still allows the
            # static constants above to drive validation).
            pass


def invalidate_cache() -> None:
    """Force the next call to re-read the registries."""
    _cache["fetched_at"] = 0.0


# ── Public API ───────────────────────────────────────────────────────


def canonical_node_type(raw: Optional[str]) -> Optional[str]:
    """Resolve aliases (e.g. product_collection -> product_group)."""
    if not raw:
        return None
    node = raw.strip().lower()
    if not node:
        return None
    _ensure_cache()
    for row in _cache.get("node_types", []) or []:
        if row.get("node_type") == node:
            alias = (row.get("alias_of") or "").strip().lower()
            return alias or node
    return NODE_TYPE_ALIASES.get(node, node)


def list_node_types(canonical_only: bool = True) -> list[dict]:
    _ensure_cache()
    rows = list(_cache.get("node_types", []) or [])
    if not rows:
        return [{"node_type": t, "canonical": True} for t in CANONICAL_NODE_TYPES]
    if canonical_only:
        rows = [r for r in rows if bool(r.get("canonical"))]
    return rows


def list_relations(canonical_only: bool = True) -> list[dict]:
    _ensure_cache()
    rows = list(_cache.get("relations", []) or [])
    if canonical_only:
        rows = [r for r in rows if bool(r.get("canonical"))]
    existing = {str(row.get("relation_type")) for row in rows}
    rows.extend({**row, "canonical": True, "active": True} for row in V21_RELATIONS if row["relation_type"] not in existing)
    return rows


def get_relation(relation_type: str) -> Optional[dict]:
    if not relation_type:
        return None
    rel = relation_type.strip().lower()
    _ensure_cache()
    for row in _cache.get("relations", []) or []:
        if (row.get("relation_type") or "").lower() == rel:
            return row
    return None


def edge_kind_for_relation(relation_type: str) -> str:
    row = get_relation(relation_type)
    if row and row.get("edge_kind"):
        return str(row["edge_kind"])
    if relation_type in {name for _, _, name in PRIMARY_CHAIN}:
        return "primary"
    if relation_type == "gallery_has_asset":
        return "asset_approved"
    if relation_type == "asset_pending":
        return "asset_pending"
    if relation_type == "asset_approved":
        return "asset_approved"
    return "secondary"


def is_primary_edge_allowed(
    source_type: Optional[str],
    target_type: Optional[str],
    relation_type: str,
) -> bool:
    """Whether a primary edge between these canonical types is canonical."""
    src = canonical_node_type(source_type)
    tgt = canonical_node_type(target_type)
    if not src or not tgt:
        return False
    rel = (relation_type or "").strip().lower()
    triple = (src, tgt, rel)
    if triple in PRIMARY_CHAIN:
        return True
    # Type-only check: when the relation label is unspecified, accept any
    # canonical (src -> tgt) pair from the primary chain regardless of label.
    # Callers like sofia_tools._violates_hierarchy validate by node type and
    # pass an empty relation, so without this audience->product_group and
    # product_group->product would be wrongly rejected.
    if not rel and any(a == src and b == tgt for a, b, _ in PRIMARY_CHAIN):
        return True
    row = get_relation(relation_type)
    if not row or row.get("edge_kind") != "primary":
        return False
    if not bool(row.get("canonical")):
        return False
    srcs = [str(x).lower() for x in (row.get("source_node_types") or [])]
    tgts = [str(x).lower() for x in (row.get("target_node_types") or [])]
    src_ok = not srcs or src in srcs
    tgt_ok = not tgts or tgt in tgts
    return src_ok and tgt_ok


def is_secondary_allowed(
    source_type: Optional[str],
    target_type: Optional[str],
) -> bool:
    """Secondary edges are allowed between any two known canonical nodes."""
    src = canonical_node_type(source_type)
    tgt = canonical_node_type(target_type)
    if not src or not tgt:
        return False
    known = {row.get("node_type") for row in list_node_types(canonical_only=False)}
    if not known:
        known = set(CANONICAL_NODE_TYPES) | set(NODE_TYPE_ALIASES.keys())
    return src in known and tgt in known


def is_asset_edge(relation_type: str) -> bool:
    return edge_kind_for_relation(relation_type) in {"asset_pending", "asset_approved"}


def canonical_chain() -> list[dict]:
    """Returns the canonical hierarchy in display order."""
    out: list[dict] = []
    for src, tgt, rel in PRIMARY_CHAIN:
        out.append(
            {
                "source": src,
                "target": tgt,
                "relation": rel,
                "one_to_one": (src, tgt) in PRIMARY_ONE_TO_ONE,
            }
        )
    return out


def taxonomy_snapshot() -> dict:
    """Compact payload for the frontend / agent prompt."""
    node_types = list_node_types(canonical_only=False)
    relations = list_relations(canonical_only=False)
    return {
        "node_types": [
            {
                "node_type": row.get("node_type"),
                "label": row.get("label"),
                "description": row.get("description"),
                "canonical": bool(row.get("canonical")),
                "alias_of": row.get("alias_of"),
                "default_level": row.get("default_level"),
                "color": row.get("color"),
                "icon": row.get("icon"),
                "sort_order": row.get("sort_order"),
            }
            for row in node_types
        ],
        "relations": [
            {
                "relation_type": row.get("relation_type"),
                "label": row.get("label"),
                "edge_kind": row.get("edge_kind") or edge_kind_for_relation(
                    str(row.get("relation_type") or "")
                ),
                "primary_one_to_one": bool(row.get("primary_one_to_one")),
                "source_node_types": row.get("source_node_types") or [],
                "target_node_types": row.get("target_node_types") or [],
                "canonical": bool(row.get("canonical")),
                "sort_order": row.get("sort_order"),
            }
            for row in relations
        ],
        "primary_chain": canonical_chain(),
        "edge_kinds": list(EDGE_KINDS),
    }
