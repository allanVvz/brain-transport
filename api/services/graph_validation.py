"""Shared graph hierarchy rules for BOTH Sofia paths (Create and Graph).

Single source of truth for "which parent a node type may hang from" and the
anti-hallucination product signals. Both the Create path
(`kb_intake_service.validate_sofia_knowledge_plan`) and the Graph path
(`sofia_orchestrator`) import from here so the two caminhos can never drift into
contradictory rules again.

The canonical *preferred* chain lives in `knowledge_taxonomy.PRIMARY_CHAIN`:

    persona -> brand -> briefing -> campaign -> audience
            -> product_group? -> product? -> copy? -> {faq, gallery}

`CANONICAL_PARENTS` below keeps optional cards optional. When an optional card
exists in the plan, children that use its context must connect top-down to it.
Example: if `product_group=Radar` exists, products in that group connect as
`product_group -> product`. Without a product_group, a product may connect
directly as `audience -> product`.
"""
from __future__ import annotations

from typing import Optional

from services import knowledge_taxonomy

# Allowed parent node_types per child node_type. Empty string / "self" means
# "directly under the persona root" and is accepted for top-level types.
CANONICAL_PARENTS: dict[str, set[str]] = {
    "persona":       set(),  # root
    "brand":         {"persona"},
    "briefing":      {"brand", "campaign"},
    "campaign":      {"briefing", "brand"},
    "audience":      {"campaign", "briefing", "brand"},
    "product_group": {"audience"},
    "product":       {"product_group", "audience"},
    "offer":         {"product"},
    "copy":          {"product", "product_group", "campaign", "audience"},
    # FAQ hangs only off the commercial leaf layer. The simplified preview
    # contract keeps it attached to copy/product/product_group.
    "faq":           {"copy", "product", "product_group"},
    "gallery":       {"copy"},
    "embedded":      {"faq"},
}

# Types that attach to the persona root as a protected/top-level branch.
TOP_LEVEL_TYPES: frozenset[str] = frozenset({"persona", "brand", "briefing"})

# Lateral types accept any parent (not part of the primary tree).
LATERAL_TYPES: frozenset[str] = frozenset({"asset", "gallery", "tag", "mention"})

_EMPTY_PARENTS = frozenset({"", "self", "root", "global", "persona", None})


def _canon(node_type: Optional[str]) -> Optional[str]:
    return knowledge_taxonomy.canonical_node_type(node_type)


def canonical_parents(child_type: Optional[str]) -> set[str]:
    """Allowed parent node types for `child_type` (canonical, alias-resolved)."""
    child = _canon(child_type)
    return set(CANONICAL_PARENTS.get(child or "", set()))


def parent_violation(child_type: Optional[str], parent_type: Optional[str]) -> Optional[str]:
    """Return a human-readable violation, or None when the parent is canonical.

    `parent_type` may be falsy/"self" meaning "directly under persona" — accepted
    for the top-level types. Asset is lateral and accepts any parent.
    """
    child = _canon(child_type)
    if not child or child == "asset":
        return None
    if child == "persona":
        return None
    parent = (parent_type or "").strip().lower()
    if parent in _EMPTY_PARENTS:
        # Attaching directly under persona is only valid for top-level types.
        if child in TOP_LEVEL_TYPES:
            return None
        allowed = sorted(canonical_parents(child))
        return f"{child} precisa de parent do tipo {allowed or ['persona']}"
    parent_canon = _canon(parent) or parent
    allowed = canonical_parents(child)
    if parent_canon in allowed:
        return None
    return (
        f"{child} nao pode ficar abaixo de {parent_canon}; "
        f"parent permitido: {sorted(allowed) or ['persona']}"
    )


def edge_violation(
    source_type: Optional[str],
    target_type: Optional[str],
    relation_type: str = "",
) -> Optional[str]:
    """Validate a primary-tree edge source->target by node type.

    A primary edge is valid when target's parent rules accept source. Falls back
    to `knowledge_taxonomy.is_primary_edge_allowed` for relation-specific checks.
    """
    src = _canon(source_type)
    tgt = _canon(target_type)
    if not src or not tgt:
        return f"tipos desconhecidos no edge {source_type}->{target_type}"
    if knowledge_taxonomy.is_primary_edge_allowed(src, tgt, relation_type):
        return None
    return parent_violation(tgt, src)


# ── Anti-hallucination (shared by both prompts/paths) ────────────────────────
# Broad campaign/positioning terms that must NOT be auto-materialized as products.
def contextual_parent_violation(
    child_type: Optional[str],
    parent_type: Optional[str],
    available_types: set[str],
) -> Optional[str]:
    """Validate optional-card context without making optional cards mandatory."""
    child = _canon(child_type)
    parent = _canon(parent_type) or (parent_type or "").strip().lower()
    available = {_canon(item) or item for item in (available_types or set()) if item}
    if child == "product" and "product_group" in available and parent != "product_group":
        return (
            "product_group existe no plano; confirme qual grupo recebe este "
            "produto e conecte product_group -> product"
        )
    if child == "copy" and parent not in {"product", "product_group", "campaign", "audience", "briefing", "brand"}:
        return "copy precisa ficar ligada ao card que ela contextualiza"
    return None


BROAD_CAMPAIGN_TERMS: frozenset[str] = frozenset({
    "esportivo", "esportivos", "esportiva", "esportivas",
    "moda inverno", "moda verao", "moda de inverno", "linha premium",
    "produto feminino", "produto masculino", "colecao nova", "nova colecao",
    "linha nova", "lancamento", "lifestyle", "casual", "premium",
})

# Phrases that ARE explicit signals to create products.
_PRODUCT_SIGNAL_MARKERS: tuple[str, ...] = (
    "use estes produtos", "use os produtos", "estes produtos", "esses produtos",
    "extraia do catalogo", "extrair do catalogo", "do catalogo",
    "produtos:", "modelos:",
)


def has_explicit_product_signal(text: Optional[str]) -> bool:
    """True when the operator explicitly asked to create concrete products.

    Signals: explicit product names lists, an explicit quantity ("N produtos"),
    "use estes produtos", "extraia do catalogo". Broad campaign terms alone
    (BROAD_CAMPAIGN_TERMS) are NOT a signal.
    """
    import re

    raw = (text or "").strip().lower()
    if not raw:
        return False
    if re.search(r"\b\d+\s+produtos?\b", raw):
        return True
    if re.search(r"\bprodutos?\s*(por|para)\s+(cada\s+)?(grupo|product_group)\b", raw):
        return True
    return any(marker in raw for marker in _PRODUCT_SIGNAL_MARKERS)
