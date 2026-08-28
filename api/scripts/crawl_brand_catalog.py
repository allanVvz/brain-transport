"""Crawler reusavel para extrair colecoes + produtos de lojas Shopify (ou compativeis).

Uso:
    python crawl_brand_catalog.py --brand vzlupas --output catalog.json

Configuracao por brand fica em CATALOG_CONFIGS abaixo. Cada brand declara:
- base_url
- collections: lista de slugs do site (vira product_groups no grafo)
- products_per_collection: quantidade a buscar por colecao
- title: nome humano da colecao (override do slug)

Saida JSON pode ser consumida por scripts de seed do grafo.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from typing import Any


CATALOG_CONFIGS: dict[str, dict[str, Any]] = {
    "vzlupas": {
        "base_url": "https://www.vzlupas.com",
        "brand_node_slug": "vz-lupas",
        "collections": [
            {"slug": "plantaris", "title": "Plantaris", "products": 3},
            {"slug": "radar", "title": "Esportivas (Radar)", "products": 3},
            {"slug": "juliet", "title": "X-Metal (Juliet)", "products": 3},
        ],
    },
}


def _http_get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "ai-brain-catalog-crawler/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_collection(base_url: str, slug: str, limit: int) -> list[dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/collections/{slug}/products.json?limit={limit}"
    data = _http_get_json(url)
    raw_products = data.get("products", [])
    out = []
    for p in raw_products[:limit]:
        variants = p.get("variants") or []
        first_variant = variants[0] if variants else {}
        images = p.get("images") or []
        first_image = images[0].get("src") if images else None
        out.append({
            "handle": p.get("handle") or "",
            "title": (p.get("title") or "").strip(),
            "vendor": p.get("vendor") or "",
            "product_type": p.get("product_type") or "",
            "price": first_variant.get("price"),
            "image": first_image,
            "body_html": p.get("body_html") or "",
        })
    return out


def crawl(brand_key: str) -> dict[str, Any]:
    if brand_key not in CATALOG_CONFIGS:
        raise SystemExit(f"Unknown brand: {brand_key}. Available: {list(CATALOG_CONFIGS)}")
    cfg = CATALOG_CONFIGS[brand_key]
    result = {
        "brand_key": brand_key,
        "brand_node_slug": cfg["brand_node_slug"],
        "base_url": cfg["base_url"],
        "collections": [],
    }
    for col in cfg["collections"]:
        products = fetch_collection(cfg["base_url"], col["slug"], col["products"])
        result["collections"].append({
            "slug": col["slug"],
            "title": col["title"],
            "source_url": f"{cfg['base_url']}/collections/{col['slug']}",
            "products": products,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", required=True, choices=list(CATALOG_CONFIGS))
    parser.add_argument("--output", default="-", help="output file path or '-' for stdout")
    args = parser.parse_args()

    result = crawl(args.brand)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output == "-":
        sys.stdout.write(payload + "\n")
    else:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
        print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
