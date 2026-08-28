# -*- coding: utf-8 -*-
"""Canonical landing-page asset slots.

A persona's catalog landing page (route /cardapio/<persona-slug> on the
ai-brain-catalogo frontend) has a fixed set of asset-bearing surfaces.
Each surface is a *slot* with:

  - a slot_key  — stable identifier consumed by the frontend
  - a parent node type — what kind of knowledge_node owns the slot
                          (campaign for hero/footer, category for product groups)
  - a default relation_type for the edge that binds the asset to the parent
  - a page_binding.section value carried in edge metadata so menu.py can
    re-emit it on the wire

Convention only — no new tables. All bindings live in
``knowledge_edges.metadata.page_binding`` plus the asset's
``metadata.asset_function``.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional


class LandingSlot(str, Enum):
    HERO = "hero"
    PRODUCT_GROUP_COVER = "product_group_cover"
    CAMPAIGN_FOOTER = "campaign_footer"
    PRODUCT_IMAGE = "product_image"
    BRAND_LOGO = "brand_logo"
    BRAND_SECONDARY = "brand_secondary"
    BRAND_COVER = "brand_cover"


_SLOT_CONFIG: dict[LandingSlot, dict] = {
    LandingSlot.HERO: {
        "label": "Hero da landing",
        "parent_node_type": "campaign",
        "relation_type": "uses_asset",
        "page_section": "hero",
        "asset_function": "campaign_hero",
        "needs_collection": True,
    },
    LandingSlot.PRODUCT_GROUP_COVER: {
        "label": "Capa de grupo de produto",
        "parent_node_type": "category",
        "relation_type": "category_has_asset",
        "page_section": "product_group",
        "asset_function": "category_cover",
        "needs_collection": False,
    },
    LandingSlot.CAMPAIGN_FOOTER: {
        "label": "Campanha de rodape",
        "parent_node_type": "campaign",
        "relation_type": "uses_asset",
        "page_section": "footer",
        "asset_function": "campaign_footer",
        "needs_collection": True,
    },
    LandingSlot.PRODUCT_IMAGE: {
        "label": "Imagem de produto",
        "parent_node_type": "product",
        "relation_type": "product_image",
        "page_section": "product",
        "asset_function": "product_image",
        "needs_collection": False,
    },
    LandingSlot.BRAND_LOGO: {
        "label": "Logo da marca",
        "parent_node_type": "brand",
        "relation_type": "brand_has_asset",
        "page_section": "brand",
        "asset_function": "brand_logo",
        "needs_collection": False,
    },
    LandingSlot.BRAND_SECONDARY: {
        "label": "Asset secundario da marca",
        "parent_node_type": "brand",
        "relation_type": "brand_has_asset",
        "page_section": "brand",
        "asset_function": "brand_secondary",
        "needs_collection": False,
    },
    LandingSlot.BRAND_COVER: {
        "label": "Capa da marca",
        "parent_node_type": "brand",
        "relation_type": "brand_has_asset",
        "page_section": "brand",
        "asset_function": "brand_cover",
        "needs_collection": False,
    },
}


# Slots offered to the user when an asset is already connected to a parent of
# the given node_type. The dashboard reads this map to render contextual
# dropdowns; menu.py reads metadata to round-trip the binding back out.
SLOTS_BY_PARENT_TYPE: dict[str, list[LandingSlot]] = {
    "brand": [LandingSlot.BRAND_LOGO, LandingSlot.BRAND_SECONDARY, LandingSlot.BRAND_COVER],
    "campaign": [LandingSlot.HERO, LandingSlot.CAMPAIGN_FOOTER],
    "category": [LandingSlot.PRODUCT_GROUP_COVER],
    "product": [LandingSlot.PRODUCT_IMAGE],
}


def slots_for_parent_type(parent_type: Optional[str]) -> list[dict]:
    slots = SLOTS_BY_PARENT_TYPE.get((parent_type or "").lower(), [])
    return [{"slot_key": slot.value, "label": _SLOT_CONFIG[slot]["label"]} for slot in slots]


def slot_config(slot: LandingSlot) -> dict:
    return _SLOT_CONFIG[slot]


def slot_for_key(slot_key: str) -> Optional[LandingSlot]:
    slot_key = str(slot_key or "").split(":", 1)[0]
    try:
        return LandingSlot(slot_key)
    except ValueError:
        return None


def slot_for_metadata(metadata: Optional[dict]) -> Optional[LandingSlot]:
    """Read back a slot from an edge.metadata payload.

    Looks at metadata.page_binding.slot_key first (preferred), then falls back
    to legacy page_binding.section/role values so we don't lose bindings created
    before this module existed.
    """
    if not metadata:
        return None
    binding = metadata.get("page_binding") or {}
    candidate = binding.get("slot_key") or binding.get("section")
    slot = slot_for_key(str(candidate or "").strip())
    if slot:
        return slot
    role = metadata.get("role") or binding.get("role")
    if role == "category_cover":
        return LandingSlot.PRODUCT_GROUP_COVER
    if role == "campaign_hero":
        return LandingSlot.HERO
    if role == "campaign_footer":
        return LandingSlot.CAMPAIGN_FOOTER
    if role == "product_image":
        return LandingSlot.PRODUCT_IMAGE
    if role == "brand_logo":
        return LandingSlot.BRAND_LOGO
    if role == "brand_secondary":
        return LandingSlot.BRAND_SECONDARY
    if role == "brand_cover":
        return LandingSlot.BRAND_COVER
    return None


def edge_metadata_for_slot(slot: LandingSlot, *, position: int = 0, label: Optional[str] = None) -> dict:
    cfg = _SLOT_CONFIG[slot]
    return {
        "role": cfg["asset_function"],
        "page_section": cfg["page_section"],
        "page_binding": {
            "slot_key": slot.value,
            "section": cfg["page_section"],
            "label": label or cfg["label"],
            "position": position,
        },
    }


def all_slots() -> list[dict]:
    return [
        {"slot_key": slot.value, **slot_config(slot)}
        for slot in LandingSlot
    ]
