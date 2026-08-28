from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional
from urllib.parse import quote


DEFAULT_FORMATS = [
    {
        "key": "cardapio",
        "label": "Cardapio",
        "description": "Menu comercial com categorias, produtos, FAQ, assets e CTA para WhatsApp.",
        "default_route_prefix": "cardapio",
        "default_collection_type": "menu",
        "structure": {
            "sections": ["hero", "categories", "products", "faq", "whatsapp_cta"],
            "primary_nodes": ["brand", "product_group", "product", "faq", "asset"],
        },
        "enabled": True,
        "sort_order": 10,
    },
    {
        "key": "landing_page",
        "label": "Landing page",
        "description": "Pagina de campanha/oferta com hero, prova, produto, FAQ e CTA para WhatsApp.",
        "default_route_prefix": "landing",
        "default_collection_type": "campaign",
        "structure": {
            "sections": ["hero", "campaign", "offer", "products", "faq", "whatsapp_cta"],
            "primary_nodes": ["brand", "campaign", "offer", "product", "faq", "asset"],
        },
        "enabled": True,
        "sort_order": 20,
    },
    {
        "key": "catalogo_roupas",
        "label": "Catalogo de roupas",
        "description": "Catalogo visual de moda com colecoes, grades de produto, assets e CTA para WhatsApp.",
        "default_route_prefix": "catalogo",
        "default_collection_type": "catalog",
        "structure": {
            "sections": ["hero", "collections", "product_grid", "size_notes", "faq", "whatsapp_cta"],
            "primary_nodes": ["brand", "product_group", "product", "faq", "asset"],
        },
        "enabled": True,
        "sort_order": 30,
    },
]


def slugify_site(value: str) -> str:
    value = unicodedata.normalize("NFD", str(value or "").strip().lower())
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return re.sub(r"(^-+|-+$)", "", value)


def normalize_whatsapp_phone(value: Optional[str]) -> str:
    return re.sub(r"\D+", "", str(value or ""))


def whatsapp_href(phone: Optional[str], message_template: Optional[str]) -> Optional[str]:
    digits = normalize_whatsapp_phone(phone)
    if not digits:
        return None
    text = str(message_template or "").strip()
    suffix = f"?text={quote(text)}" if text else ""
    return f"https://wa.me/{digits}{suffix}"


def format_by_key(formats: list[dict[str, Any]], key: str) -> Optional[dict[str, Any]]:
    return next((row for row in formats if row.get("key") == key and row.get("enabled") is not False), None)


def default_public_site_config(persona: dict[str, Any]) -> dict[str, Any]:
    slug = slugify_site(persona.get("slug") or persona.get("name") or "")
    name = persona.get("name") or persona.get("slug") or ""
    return {
        "site_slug": slug,
        "site_name": name,
        "format_key": "cardapio",
        "default_collection_slug": (
            ((persona.get("config") or {}).get("public_site") or {}).get("default_collection_slug")
            or (persona.get("config") or {}).get("default_collection_slug")
            or (persona.get("config") or {}).get("collection_slug")
            or f"cardapio-{slug}-v1"
        ),
        "whatsapp_phone": "",
        "whatsapp_message_template": f"Ola, vim pelo site {name} e quero mais informacoes.",
    }


def normalize_public_site_config(persona: dict[str, Any], formats: list[dict[str, Any]]) -> dict[str, Any]:
    config = persona.get("config") or {}
    raw = config.get("public_site") or {}
    default = default_public_site_config(persona)
    merged = {**default, **{k: v for k, v in raw.items() if v is not None}}
    merged["site_slug"] = slugify_site(merged.get("site_slug") or default["site_slug"])
    merged["site_name"] = str(merged.get("site_name") or default["site_name"]).strip()
    merged["format_key"] = str(merged.get("format_key") or default["format_key"]).strip()
    if not format_by_key(formats, merged["format_key"]):
        merged["format_key"] = default["format_key"]
    merged["default_collection_slug"] = slugify_site(
        merged.get("default_collection_slug") or default["default_collection_slug"]
    )
    merged["whatsapp_phone"] = normalize_whatsapp_phone(merged.get("whatsapp_phone"))
    merged["whatsapp_message_template"] = str(merged.get("whatsapp_message_template") or "").strip()
    return merged


def public_site_payload(
    persona: dict[str, Any],
    formats: list[dict[str, Any]],
    *,
    catalog_url: Optional[str] = None,
) -> dict[str, Any]:
    config = normalize_public_site_config(persona, formats)
    fmt = format_by_key(formats, config["format_key"]) or DEFAULT_FORMATS[0]
    route_prefix = (fmt.get("default_route_prefix") or "").strip("/")
    route_path = f"/{route_prefix}/{config['site_slug']}" if route_prefix else f"/{config['site_slug']}"
    href = whatsapp_href(config.get("whatsapp_phone"), config.get("whatsapp_message_template"))
    return {
        "slug": config["site_slug"],
        "name": config["site_name"],
        "format_key": config["format_key"],
        "format_label": fmt.get("label") or config["format_key"],
        "route_path": route_path,
        "catalog_url": catalog_url or persona.get("catalog_url"),
        "default_collection_slug": config["default_collection_slug"],
        "whatsapp": {
            "phone": config.get("whatsapp_phone") or None,
            "message_template": config.get("whatsapp_message_template") or None,
            "href": href,
        },
    }

