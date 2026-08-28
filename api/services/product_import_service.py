"""Multi-provider product import for the /marketing/produtos tab.

Normalizes products coming from Meta (WhatsApp Business catalog), CSV/Excel,
Shopify (via the existing crawler) and a mocked scraper into the internal
model: every product becomes a `knowledge_node` (node_type=product) with
status='pending' (staging via status â€” no new table, per CLAUDE.md Â§2).

Rules enforced here:
- Dedup by `source + external_id + catalog_id` (stored as
  `metadata.import_dedupe_key`). A re-import UPDATES the existing pending node
  instead of creating a duplicate.
- When a product_group is present, ensure a product_collection node + edge.
- When an image is present, create an asset node (marked as product image) and
  link it via a `product_image` edge.
- When commercial copy is present, create a copy node + `supports_copy` edge.
- NEVER connect to Embedded; FAQs are not generated here.
- Imported products enter as `pending`; approval is a separate step.

Network access (Meta Graph API / crawler) is injectable so tests run offline.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
import time
from typing import Any, Callable, Optional

from services import supabase_client

PROVIDERS = {"meta", "csv", "shopify", "scraper"}

META_GRAPH_VERSION = "v19.0"
META_GRAPH_BASE = "https://graph.facebook.com"


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #
def _slugify(value: str) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:60] or "produto"


def dedupe_key(source: str, external_id: Optional[str], catalog_id: Optional[str]) -> str:
    return f"{(source or '').strip()}:{(external_id or '').strip()}:{(catalog_id or '').strip() or '-'}"


def _first_price(value: Any) -> Optional[str]:
    """Accepts a string, number, list of prices, or Meta-shaped '12.30 BRL'."""
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        return _first_price(value[0]) if value else None
    text = str(value).strip()
    if not text:
        return None
    # Meta price often comes as "59.90 BRL" or "5990" (cents). Keep the numeric head.
    match = re.search(r"\d+[.,]?\d*", text)
    return match.group(0).replace(",", ".") if match else None


def _all_prices(value: Any) -> list[str]:
    """Distinct, order-preserving list of normalized prices from a value that
    may be a single price, a list of variant prices, or Meta-shaped strings."""
    out: list[str] = []

    def add(v: Any) -> None:
        p = _first_price(v)
        if p and p not in out:
            out.append(p)

    if isinstance(value, (list, tuple)):
        for v in value:
            add(v)
    elif value not in (None, ""):
        add(value)
    return out


def _price_cents(amount: Optional[str]) -> int:
    try:
        return int(round(float(str(amount)) * 100)) if amount else 0
    except (TypeError, ValueError):
        return 0


def _coalesce(raw: dict, *keys: str) -> Optional[str]:
    for key in keys:
        val = raw.get(key)
        if val not in (None, ""):
            return str(val).strip()
    return None


# --------------------------------------------------------------------------- #
# Normalization                                                                 #
# --------------------------------------------------------------------------- #
def normalize_imported_product(
    raw: dict,
    *,
    provider: str,
    catalog_id: Optional[str] = None,
) -> dict:
    """Map a provider-shaped product into the internal NormalizedProduct dict."""
    name = _coalesce(raw, "name", "title", "product_name") or "Produto importado"
    description = _coalesce(
        raw, "description", "commercial_description", "summary", "body_html", "rich_text_description"
    ) or ""
    external_id = _coalesce(raw, "external_id", "retailer_id", "id", "sku", "handle")
    image_url = _coalesce(raw, "image_url", "image", "image_link", "imageUrl")
    category = _coalesce(raw, "category", "product_type", "product_group", "categoria")
    product_group = _coalesce(raw, "product_group", "group", "collection", "category", "product_type")
    raw_prices = raw.get("prices") if raw.get("prices") not in (None, "", []) else raw.get("price")
    prices = _all_prices(raw_prices)
    price = prices[0] if prices else None
    currency = _coalesce(raw, "currency") or "BRL"

    return {
        "name": name,
        "description": description.strip(),
        "price": price,
        "prices": prices,
        "currency": currency,
        "category": category,
        "product_group": product_group,
        "source": provider,
        "external_id": external_id,
        "catalog_id": catalog_id,
        "image_url": image_url,
        "raw_payload": raw,
        "dedupe_key": dedupe_key(provider, external_id, catalog_id),
    }


# --------------------------------------------------------------------------- #
# Provider adapters (return list[raw dict])                                     #
# --------------------------------------------------------------------------- #
def _fetch_meta(config: dict, fetch: Optional[Callable[[dict], list[dict]]] = None) -> list[dict]:
    if fetch is not None:
        return list(fetch(config) or [])
    import httpx  # local import keeps module import cheap/offline-safe

    access_token = config.get("access_token")
    catalog_id = config.get("catalog_id")
    if not access_token or not catalog_id:
        raise ValueError("Meta import requires access_token and catalog_id.")
    url = f"{META_GRAPH_BASE}/{META_GRAPH_VERSION}/{catalog_id}/products"
    params = {
        "access_token": access_token,
        "fields": "id,retailer_id,name,description,price,currency,image_url,product_type,category",
        "limit": 200,
    }
    out: list[dict] = []
    with httpx.Client(timeout=20.0) as client:
        next_url: Optional[str] = url
        next_params: Optional[dict] = params
        for _ in range(10):  # cap pagination
            resp = client.get(next_url, params=next_params)
            resp.raise_for_status()
            payload = resp.json()
            out.extend(payload.get("data") or [])
            next_url = ((payload.get("paging") or {}).get("next")) or None
            next_params = None  # the `next` URL already carries query params
            if not next_url:
                break
    return out


def _parse_csv(file_bytes: bytes) -> list[dict]:
    text = file_bytes.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict] = []
    for row in reader:
        clean = {(k or "").strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
        if any(clean.values()):
            rows.append(clean)
    return rows


def _normalize_url(url: str) -> str:
    """Tolerate operator input without a scheme (e.g. 'vzlupas.com') â€” the
    crawler requires http/https + netloc."""
    from urllib.parse import urlparse

    candidate = (url or "").strip()
    if not candidate:
        return candidate
    if not urlparse(candidate).scheme:
        candidate = f"https://{candidate}"
    return candidate


def _fetch_shopify(config: dict) -> list[dict]:
    from services.catalog_crawler import crawl_catalog_url

    url = _normalize_url(config.get("url") or config.get("source_url") or "")
    if not url:
        raise ValueError("Shopify import requires a source url.")
    # High limit so the audit screen can list the whole catalog by collection.
    capture = crawl_catalog_url(url, limit=int(config.get("limit") or 500)) or {}
    # Surface brand logo/cover (home og:image/logo) for brand_has_asset linking.
    if capture.get("brand_assets") and not config.get("brand_assets"):
        config["brand_assets"] = capture.get("brand_assets")
    out: list[dict] = []
    for cand in capture.get("product_candidates") or []:
        out.append(
            {
                "title": cand.get("title"),
                "description": cand.get("description") or "",
                "prices": cand.get("prices"),
                "image_url": cand.get("image_url") or cand.get("image"),
                "handle": cand.get("handle"),
                "external_id": cand.get("handle") or cand.get("source"),
                # product_type is Shopify's closest signal to a collection.
                "product_group": cand.get("product_type"),
                "source": cand.get("source") or "shopify_json",
            }
        )
    return out


def _scraper_mock(config: dict) -> list[dict]:
    return [
        {"title": "Produto Scraper Mock 1", "description": "Item mockado de scraping.", "prices": ["49.90"], "external_id": "mock-1", "product_group": "Mock"},
        {"title": "Produto Scraper Mock 2", "description": "Item mockado de scraping.", "prices": ["59.90"], "external_id": "mock-2", "product_group": "Mock"},
    ]


def _raw_items_for_provider(
    provider: str,
    *,
    config: dict,
    items: Optional[list[dict]],
    file_bytes: Optional[bytes],
    fetch: Optional[Callable[[dict], list[dict]]],
) -> list[dict]:
    if items is not None:
        return list(items)
    if provider == "meta":
        return _fetch_meta(config, fetch=fetch)
    if provider == "csv":
        if not file_bytes:
            raise ValueError("CSV import requires file bytes.")
        return _parse_csv(file_bytes)
    if provider == "shopify":
        return (fetch(config) if fetch else _fetch_shopify(config)) or []
    if provider == "scraper":
        return _scraper_mock(config)
    raise ValueError(f"Unknown provider: {provider}")


# --------------------------------------------------------------------------- #
# Preview / audit (no writes)                                                   #
# --------------------------------------------------------------------------- #
def preview_products(
    *,
    provider: str,
    config: Optional[dict] = None,
    fetch: Optional[Callable[[dict], list[dict]]] = None,
) -> dict:
    """Crawl/list products WITHOUT importing, grouped by collection.

    Powers the audit screen: the operator toggles collections/products before
    confirming the import. No knowledge nodes are created here."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")
    config = dict(config or {})
    raw_items = _raw_items_for_provider(provider, config=config, items=None, file_bytes=None, fetch=fetch)

    groups: dict[str, dict] = {}
    order: list[str] = []
    for raw in raw_items:
        norm = normalize_imported_product(raw, provider=provider, catalog_id=config.get("catalog_id"))
        group = norm.get("product_group") or "Sem grupo"
        if group not in groups:
            groups[group] = {"key": _slugify(group), "label": group, "products": []}
            order.append(group)
        groups[group]["products"].append(
            {
                "external_id": norm.get("external_id"),
                "title": norm.get("name"),
                "thumbnail": norm.get("image_url"),
                "price": norm.get("price"),
                "currency": norm.get("currency"),
                "product_group": group,
                "has_image": bool(norm.get("image_url")),
                # raw item carried back so confirm can import without re-crawling
                "item": raw,
            }
        )
    collections = [{**groups[name], "count": len(groups[name]["products"])} for name in order]
    return {
        "provider": provider,
        "source_url": _normalize_url(config.get("url") or "") or None,
        "total": sum(c["count"] for c in collections),
        "collections": collections,
    }


# --------------------------------------------------------------------------- #
# Image download (real bytes -> storage), offline-safe / injectable             #
# --------------------------------------------------------------------------- #
_CT_EXT = {"image/jpeg": "jpg", "image/jpg": "jpg", "image/png": "png", "image/webp": "webp", "image/gif": "gif"}


def _download_image(
    persona_id: str,
    slug: str,
    image_url: str,
    downloader: Optional[Callable[[str], tuple[bytes, str]]] = None,
) -> dict:
    """Fetch image bytes and store them in the assets-raw bucket.

    Returns metadata to merge into the asset node. On any failure (offline,
    storage down, bad URL) returns {downloaded: False, download_error: ...} so
    the import never breaks â€” the asset keeps the source URL reference."""
    try:
        if downloader is not None:
            data, content_type = downloader(image_url)
        else:
            import httpx

            with httpx.Client(timeout=20.0, follow_redirects=True) as client:
                resp = client.get(image_url, headers={"User-Agent": "AI-Brain-Importer/0.1"})
                resp.raise_for_status()
                data = resp.content
                content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
        ext = _CT_EXT.get(content_type, "jpg")
        digest = hashlib.sha1((image_url or slug).encode("utf-8")).hexdigest()[:10]
        path = f"{persona_id}/products/{slug}-{digest}.{ext}"
        url = supabase_client.upload_to_storage("assets-raw", path, data, content_type)
        return {
            "downloaded": True,
            "url": url,
            "storage_bucket": "assets-raw",
            "storage_path": path,
            "file_path": f"assets-raw:{path}",
            "content_type": content_type,
            "bytes": len(data),
        }
    except Exception as exc:  # pragma: no cover - exercised via injected failures
        return {"downloaded": False, "download_error": str(exc)[:200]}


# --------------------------------------------------------------------------- #
# Node materialization                                                          #
# --------------------------------------------------------------------------- #
def _find_existing_by_dedupe(persona_id: str, key: str) -> Optional[dict]:
    for node in supabase_client.list_product_nodes(persona_id=persona_id):
        if (node.get("metadata") or {}).get("import_dedupe_key") == key:
            return node
    return None


def _ensure_group_node(persona_id: str, group_title: str) -> Optional[dict]:
    """Group is materialized as a canonical `product_group` node â€” this is what
    the cardÃ¡pio menu endpoint (/api/menu/{slug}) reads as a category, and the
    dashboard Kanban groups by metadata.product_group. Products bind to it via
    `metadata.category_slug == group.slug` and a `product_group_has_product`
    edge, so the imported catalog renders grouped on the persona's menu."""
    slug = _slugify(group_title)
    return supabase_client.upsert_knowledge_node(
        {
            "persona_id": persona_id,
            "node_type": "product_group",
            "slug": slug,
            "title": group_title,
            "metadata": {"source": "product_import", "product_group": group_title, "category_slug": slug},
            "status": "pending_validation",
        }
    )


def _materialize_product(
    persona_id: str,
    persona_slug: Optional[str],
    norm: dict,
    *,
    download_images: bool = False,
    image_downloader: Optional[Callable[[str], tuple[bytes, str]]] = None,
) -> dict:
    """Create or update the pending product node + group/asset/copy nodes."""
    existing = _find_existing_by_dedupe(persona_id, norm["dedupe_key"])
    group_node = _ensure_group_node(persona_id, norm["product_group"]) if norm.get("product_group") else None
    group_slug = (group_node or {}).get("slug")

    metadata = {
        "source": norm["source"],
        "external_id": norm.get("external_id"),
        "catalog_id": norm.get("catalog_id"),
        "import_dedupe_key": norm["dedupe_key"],
        "raw_payload": norm.get("raw_payload"),
        "product_group": norm.get("product_group"),
        # The cardÃ¡pio menu binds a product to its group by category_slug /
        # product_group_slug; keep collection_slug too for the dashboard filter.
        "collection_slug": group_slug,
        "category_slug": group_slug or (_slugify(norm["category"]) if norm.get("category") else None),
        "product_group_slug": group_slug,
        "persona_slug": persona_slug,
        "imported": True,
    }
    offer_prices = norm.get("prices") or ([norm["price"]] if norm.get("price") else [])
    if offer_prices:
        metadata["price"] = {"unit": {"amount": offer_prices[0], "currency": norm.get("currency") or "BRL"}}
        cents = [c for c in (_price_cents(p) for p in offer_prices) if c > 0]
        if cents:
            # Denormalized lowest price so the cardÃ¡pio keeps showing a price.
            metadata["price_cents"] = min(cents)
    metadata = {k: v for k, v in metadata.items() if v is not None}

    # New imports get a slug disambiguated by a short hash of the dedupe key so
    # the same SKU across different catalogs does not collide on (persona,type,slug).
    dedupe_suffix = hashlib.sha1(norm["dedupe_key"].encode("utf-8")).hexdigest()[:8]
    slug = (existing or {}).get("slug") or f"{_slugify(norm['name'])[:50]}-{dedupe_suffix}"
    product = supabase_client.upsert_knowledge_node(
        {
            "persona_id": persona_id,
            "node_type": "product",
            "slug": slug,
            "title": norm["name"],
            "summary": norm.get("description") or norm["name"],
            "tags": ["produto", "importado", norm["source"]],
            "metadata": metadata,
            "status": "pending_validation",
        }
    )
    if not product:
        raise RuntimeError("Could not upsert product node (knowledge tables unavailable).")

    created_nodes = {"product": product.get("id")}
    image_downloaded = False
    asset: Optional[dict] = None
    copy_node: Optional[dict] = None

    if group_node:
        # Canonical group -> product edge that the menu's _products_by_group reads.
        supabase_client.upsert_knowledge_edge(
            group_node["id"], product["id"], "product_group_has_product",
            persona_id=persona_id, weight=0.8,
            metadata={"primary_tree": True, "created_from": "product_import"},
        )
        created_nodes["product_group"] = group_node.get("id")

    if norm.get("image_url"):
        asset_metadata = {
            "is_product_image": True,
            # origin reference is always preserved (spec: manter referencia de origem).
            "image_source_url": norm["image_url"],
            "source": norm["source"],
            "parent_slug": slug,
            "parent_type": "product",
            # default thumbnail = external URL; overridden by stored URL on download.
            "url": norm["image_url"],
        }
        if download_images:
            stored = _download_image(persona_id, slug, norm["image_url"], image_downloader)
            asset_metadata.update(stored)
            image_downloaded = bool(stored.get("downloaded"))
        asset = supabase_client.upsert_knowledge_node(
            {
                "persona_id": persona_id,
                "node_type": "asset",
                "slug": _slugify(f"asset-{slug}"),
                "title": f"Imagem - {norm['name']}",
                "metadata": asset_metadata,
                "status": "pending_validation",
            }
        )
        if asset:
            supabase_client.upsert_knowledge_edge(
                product["id"], asset["id"], "product_image",
                persona_id=persona_id, weight=0.85,
                metadata={"primary_tree": True, "direction": "product_to_asset", "created_from": "product_import", "is_product_image": True},
            )
            # Gallery is the terminal curation sink.  The product remains the
            # asset's primary parent; this secondary edge is what authorizes
            # the image to appear in a public landing projection.
            gallery = supabase_client.ensure_gallery_node(persona_id)
            if gallery:
                supabase_client.upsert_knowledge_edge(
                    asset["id"], gallery["id"], "gallery_asset",
                    persona_id=persona_id, weight=0.9,
                    metadata={"primary_tree": False, "direction": "asset_to_gallery", "created_from": "product_import"},
                )
            created_nodes["asset"] = asset.get("id")

    # Every imported product gets one conservative commercial copy.  It only
    # repeats known product data and remains pending human validation.
    copy_summary = (norm.get("description") or f"{norm['name']}.").strip()
    copy_node = supabase_client.upsert_knowledge_node(
        {
            "persona_id": persona_id,
            "node_type": "copy",
            "slug": _slugify(f"copy-{slug}"),
            "title": f"Copy - {norm['name']}",
            "summary": copy_summary[:240],
            "metadata": {"source": norm["source"], "parent_slug": slug, "parent_type": "product", "default_for_product": True},
            "status": "pending_validation",
        }
    )
    if copy_node:
        supabase_client.upsert_knowledge_edge(
            product["id"], copy_node["id"], "contains",
            persona_id=persona_id, weight=0.7,
            metadata={"primary_tree": True, "created_from": "product_import"},
        )
        supabase_client.upsert_knowledge_edge(
            copy_node["id"], product["id"], "supports_copy",
            persona_id=persona_id, weight=0.7,
            metadata={"primary_tree": False, "created_from": "product_import"},
        )
        created_nodes["copy"] = copy_node.get("id")

        # The question must read like something a customer would actually
        # type â€” never a meta-question about the data's own review/approval
        # status. Approval is tracked via `status` (pending_validation ->
        # validated) for agents to read structurally; it must never leak
        # into customer-facing copy.
        faq = supabase_client.upsert_knowledge_node(
            {
                "persona_id": persona_id,
                "node_type": "faq",
                "slug": _slugify(f"faq-{slug}-informacoes"),
                "title": f"O que Ã© {norm['name']}?",
                "summary": copy_summary[:400],
                "metadata": {
                    "question": f"O que Ã© {norm['name']}?",
                    "answer": copy_summary[:400],
                    "source": norm["source"],
                    "parent_slug": copy_node.get("slug"),
                    "parent_type": "copy",
                    "source_node_id": copy_node.get("id"),
                    "source_node_type": "copy",
                    "branch_path": [],
                    "default_for_product": True,
                },
                "status": "pending_validation",
            }
        )
        if faq:
            supabase_client.upsert_knowledge_edge(
                copy_node["id"], faq["id"], "answers_question",
                persona_id=persona_id, weight=0.7,
                metadata={"primary_tree": True, "created_from": "product_import"},
            )
            created_nodes["faq"] = faq.get("id")

    # OFFERS: every distinct price becomes its own canonical `offer` node, each
    # with its own FAQ and a separate gallery_asset connection (per spec:
    # "o preÃ§o SEMPRE vem como offer; cada valor distinto = uma offer diferente;
    # gera um faq diferente; conectado separadamente em gallery").
    offer_count = 0
    if offer_prices:
        currency = norm.get("currency") or "BRL"
        for price in offer_prices:
            offer_slug = _slugify(f"offer-{slug}-{price}")
            offer = supabase_client.upsert_knowledge_node(
                {
                    "persona_id": persona_id,
                    "node_type": "offer",
                    "slug": offer_slug,
                    "title": f"{norm['name']} - {currency} {price}",
                    "summary": f"Oferta {currency} {price} de {norm['name']}.",
                    "metadata": {
                        "is_offer": True,
                        "amount": price,
                        "currency": currency,
                        "price_cents": _price_cents(price),
                        "price_label": f"{currency} {price}",
                        "source": norm["source"],
                        "parent_slug": slug,
                        "parent_type": "product",
                    },
                    "status": "pending_validation",
                }
            )
            if not offer:
                continue
            offer_count += 1
            # product -> offer (canonical)
            supabase_client.upsert_knowledge_edge(
                product["id"], offer["id"], "product_has_offer",
                persona_id=persona_id, weight=0.8,
                metadata={"primary_tree": True, "created_from": "product_import"},
            )
            # Offer remains optional context.  The default Copy stays on the
            # product branch so a catalogue with multiple prices does not give
            # one Copy several primary parents.
            if copy_node:
                supabase_client.upsert_knowledge_edge(
                    offer["id"], copy_node["id"], "offer_has_copy",
                    persona_id=persona_id, weight=0.6,
                    metadata={"primary_tree": False, "created_from": "product_import"},
                )
        created_nodes["offers"] = offer_count

    return {
        "product": product,
        "created_nodes": created_nodes,
        "was_update": bool(existing),
        "image_downloaded": image_downloaded,
        "offers": offer_count,
    }


# --------------------------------------------------------------------------- #
# Public entry point                                                            #
# --------------------------------------------------------------------------- #
def import_products(
    *,
    provider: str,
    persona_id: str,
    persona_slug: Optional[str] = None,
    config: Optional[dict] = None,
    items: Optional[list[dict]] = None,
    file_bytes: Optional[bytes] = None,
    fetch: Optional[Callable[[dict], list[dict]]] = None,
    download_images: bool = False,
    image_downloader: Optional[Callable[[str], tuple[bytes, str]]] = None,
) -> dict:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")
    if not persona_id:
        raise ValueError("persona_id is required for import.")
    config = dict(config or {})
    catalog_id = config.get("catalog_id")

    started = time.monotonic()
    raw_items = _raw_items_for_provider(
        provider, config=config, items=items, file_bytes=file_bytes, fetch=fetch
    )

    created = 0
    updated = 0
    skipped = 0
    images_downloaded = 0
    staging: list[dict] = []
    seen_keys: set[str] = set()

    for raw in raw_items:
        norm = normalize_imported_product(raw, provider=provider, catalog_id=catalog_id)
        if not norm.get("external_id") and not norm.get("name"):
            skipped += 1
            continue
        # Dedup within a single batch too (two rows with the same key).
        if norm["dedupe_key"] in seen_keys:
            skipped += 1
            continue
        seen_keys.add(norm["dedupe_key"])

        result = _materialize_product(
            persona_id, persona_slug, norm,
            download_images=download_images, image_downloader=image_downloader,
        )
        if result["was_update"]:
            updated += 1
        else:
            created += 1
        if result.get("image_downloaded"):
            images_downloaded += 1
        staging.append(
            {
                "slug": result["product"].get("slug"),
                "title": result["product"].get("title"),
                "status": result["product"].get("status"),
                "source": norm["source"],
                "dedupe_key": norm["dedupe_key"],
                "nodes": result["created_nodes"],
                "image_downloaded": result.get("image_downloaded", False),
            }
        )

    return {
        "provider": provider,
        "created": created,
        "updated": updated,
        "skipped": skipped,
        "images_downloaded": images_downloaded,
        "download_images": download_images,
        "total": len(raw_items),
        "elapsed_ms": int((time.monotonic() - started) * 1000),
        "staging": staging,
    }

