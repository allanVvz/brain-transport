"""Create the pending Copy -> FAQ branch required for existing products.

This is idempotent and intentionally does not create Offers or approve/embed
FAQs.  It is useful after importing a legacy catalog into the canonical graph.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services import supabase_client  # noqa: E402


def ensure_product_branches(persona_slug: str) -> dict:
    persona = supabase_client.get_persona(persona_slug)
    if not persona:
        raise RuntimeError(f"Persona not found: {persona_slug}")
    persona_id = persona["id"]
    products = [
        row for row in supabase_client.list_product_nodes(persona_id=persona_id, limit=5000)
        if str(row.get("status") or "").lower() != "archived"
    ]
    copies = faqs = 0
    for product in products:
        slug = str(product.get("slug") or product["id"])
        title = str(product.get("title") or slug)
        meta = product.get("metadata") if isinstance(product.get("metadata"), dict) else {}
        source = meta.get("source") or "pending_source"
        body = str(product.get("summary") or meta.get("description") or title).strip()
        copy = supabase_client.upsert_knowledge_node({
            "persona_id": persona_id,
            "node_type": "copy",
            "slug": f"copy-{slug}",
            "title": f"Copy - {title}",
            "summary": body[:400],
            "tags": ["copy", "default_product_branch"],
            "metadata": {
                "source": source,
                "parent_node_id": product["id"],
                "parent_slug": slug,
                "parent_type": "product",
                "default_for_product": True,
                "created_from": "ensure_product_default_branch",
            },
            "status": "pending_validation",
        })
        if not copy:
            raise RuntimeError(f"Could not create Copy for {slug}")
        supabase_client.upsert_knowledge_edge(
            product["id"], copy["id"], "product_has_copy", persona_id=persona_id,
            weight=0.7, metadata={"primary_tree": True, "active": True, "created_from": "ensure_product_default_branch"},
        )
        copies += 1
        faq = supabase_client.upsert_knowledge_node({
            "persona_id": persona_id,
            "node_type": "faq",
            "slug": f"faq-{slug}-informacoes",
            "title": f"O que e {title}?",
            "summary": body[:400],
            "tags": ["faq", "default_product_branch"],
            "metadata": {
                "question": f"O que e {title}?",
                "answer": body[:400],
                "source": source,
                "parent_node_id": copy["id"],
                "parent_slug": copy.get("slug"),
                "parent_type": "copy",
                "source_node_id": copy["id"],
                "source_node_type": "copy",
                "branch_path": [],
                "default_for_product": True,
                "created_from": "ensure_product_default_branch",
            },
            "status": "pending_validation",
        })
        if not faq:
            raise RuntimeError(f"Could not create FAQ for {slug}")
        supabase_client.upsert_knowledge_edge(
            copy["id"], faq["id"], "copy_has_faq", persona_id=persona_id,
            weight=0.7, metadata={"primary_tree": True, "active": True, "created_from": "ensure_product_default_branch"},
        )
        faqs += 1
    return {"ok": True, "persona_slug": persona_slug, "products": len(products), "copies": copies, "faqs": faqs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persona-slug", required=True)
    args = parser.parse_args()
    print(json.dumps(ensure_product_branches(args.persona_slug), ensure_ascii=False))


if __name__ == "__main__":
    main()

