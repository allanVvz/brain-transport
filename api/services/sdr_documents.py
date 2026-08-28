"""Compile persona Markdown into the canonical Graph JSON v2 contract.

The compiler is intentionally pure: it reads files, validates the complete
publication set and returns one document. Persistence only happens through the
canonical graph publisher/importer.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from schemas.graph_json_v2 import Edge, GraphJson, Node
from services import graph_json_v2_validator
from services.deterministic_sdr import _frontmatter


APPROVED_STATUSES = {"approved", "validated", "active", "ativo"}
MATERIALIZED_TYPES = {
    "persona",
    "brand",
    "campaign",
    "audience",
    "briefing",
    "tone",
    "rule",
    "product",
    "copy",
    "faq",
}
EVIDENCE_TYPES = {"source", "catalog"}
REQUIRED_FIELDS = {
    "persona",
    "type",
    "slug",
    "title",
    "source",
    "status",
    "tags",
    "metadata",
    "relations",
}


class GraphDocumentCompileError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Markdown publication blocked:\n- " + "\n- ".join(errors))


@dataclass(frozen=True)
class SourceDocument:
    path: Path
    relative_path: str
    meta: dict[str, Any]
    body: str
    checksum: str

    @property
    def slug(self) -> str:
        return str(self.meta.get("slug") or "").strip()

    @property
    def node_type(self) -> str:
        return str(self.meta.get("type") or "").strip().lower()


def _slug(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _read_documents(base: Path) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for path in sorted(base.rglob("*.md")):
        meta, body = _frontmatter(path)
        if not meta:
            continue
        raw = path.read_bytes()
        documents.append(
            SourceDocument(
                path=path,
                relative_path=path.relative_to(base).as_posix(),
                meta=meta,
                body=body.strip(),
                checksum=hashlib.sha256(raw).hexdigest(),
            )
        )
    return documents


def _status(meta: dict[str, Any]) -> str:
    return str(meta.get("status") or "").strip().lower()


def _active(meta: dict[str, Any]) -> bool:
    return meta.get("active", True) is not False


def _relations(meta: dict[str, Any]) -> list[dict[str, Any]]:
    rows = meta.get("relations")
    return [row for row in (rows or []) if isinstance(row, dict)]


def _first_relation_target(meta: dict[str, Any]) -> str | None:
    for relation in _relations(meta):
        target = str(relation.get("target") or "").strip()
        if target:
            return target
    return None


def _relation_for(parent_type: str, child_type: str) -> str:
    return {
        ("persona", "brand"): "belongs_to_persona",
        ("brand", "campaign"): "part_of_campaign",
        ("brand", "briefing"): "briefed_by",
        ("brand", "tone"): "contains",
        ("brand", "rule"): "contains",
        ("campaign", "audience"): "targets_audience",
        ("audience", "product_group"): "contains",
        ("product_group", "product"): "contains",
        ("product_group", "copy"): "supports_copy",
        ("product", "copy"): "supports_copy",
        ("copy", "faq"): "answers_question",
        ("product", "faq"): "answers_question",
        ("product_group", "faq"): "answers_question",
        ("campaign", "faq"): "answers_question",
        ("brand", "faq"): "answers_question",
    }.get((parent_type, child_type), "contains")


def _node_data(document: SourceDocument) -> dict[str, Any]:
    meta = document.meta
    metadata = dict(meta.get("metadata") or {})
    data = {
        **meta,
        "metadata": metadata,
        "markdown": document.body,
        "source_file": document.relative_path,
        "source_checksum": document.checksum,
        "rag_chunk": {
            "index": 0,
            "checksum": hashlib.sha256(document.body.encode("utf-8")).hexdigest(),
            "source_file": document.relative_path,
        },
    }
    data["status"] = _status(meta)
    data["active"] = _active(meta)
    return data


def _validate_documents(
    documents: list[SourceDocument],
    *,
    persona_slug: str,
) -> list[str]:
    errors: list[str] = []
    slug_documents: dict[str, list[SourceDocument]] = {}
    for document in documents:
        meta = document.meta
        missing = sorted(REQUIRED_FIELDS - set(meta))
        if missing:
            errors.append(f"{document.relative_path}: missing fields {', '.join(missing)}")
        if str(meta.get("persona") or "") != persona_slug:
            errors.append(f"{document.relative_path}: persona must be {persona_slug}")
        if not document.slug:
            errors.append(f"{document.relative_path}: slug is required")
        else:
            slug_documents.setdefault(document.slug, []).append(document)
        if not str(meta.get("source") or "").strip():
            errors.append(f"{document.relative_path}: source is required")
        if not document.body:
            errors.append(f"{document.relative_path}: Markdown body is required")
        if _active(meta) and _status(meta) not in APPROVED_STATUSES:
            errors.append(f"{document.relative_path}: active document status is not approved")
        if not isinstance(meta.get("tags"), list):
            errors.append(f"{document.relative_path}: tags must be a list")
        if not isinstance(meta.get("metadata"), dict):
            errors.append(f"{document.relative_path}: metadata must be an object")
        if not isinstance(meta.get("relations"), list):
            errors.append(f"{document.relative_path}: relations must be a list")
        if document.node_type == "product" and _active(meta):
            price = meta.get("price")
            if not isinstance(price, dict):
                errors.append(f"{document.relative_path}: product price must be an object")
            else:
                amount = price.get("amount")
                currency = str(price.get("currency") or "")
                if not isinstance(amount, (int, float)) or isinstance(amount, bool) or amount <= 0:
                    errors.append(f"{document.relative_path}: product price.amount must be positive")
                if len(currency) != 3 or not currency.isalpha():
                    errors.append(f"{document.relative_path}: product price.currency must be ISO-3")

    for slug, matches in slug_documents.items():
        if len(matches) > 1:
            paths = ", ".join(item.relative_path for item in matches)
            errors.append(f"duplicate slug {slug}: {paths}")

    known_slugs = set(slug_documents)
    catalog = next((document for document in documents if document.node_type == "catalog"), None)
    for category in (catalog.meta.get("categories") or []) if catalog else []:
        if isinstance(category, dict) and category.get("slug"):
            known_slugs.add(str(category["slug"]))
    for document in documents:
        for relation in _relations(document.meta):
            target = str(relation.get("target") or "").strip()
            if not target:
                errors.append(f"{document.relative_path}: relation without target")
            elif target not in known_slugs:
                errors.append(f"{document.relative_path}: relation target not found: {target}")
    return errors


def compile_persona_documents(
    root: str | Path,
    persona_slug: str,
    *,
    tenant: str = "qa",
) -> GraphJson:
    base = Path(root) / persona_slug
    if not base.is_dir():
        raise GraphDocumentCompileError([f"persona document directory not found: {base}"])

    documents = _read_documents(base)
    errors = _validate_documents(documents, persona_slug=persona_slug)
    if errors:
        raise GraphDocumentCompileError(errors)

    materialized = [
        document
        for document in documents
        if document.node_type in MATERIALIZED_TYPES
        and _active(document.meta)
        and _status(document.meta) in APPROVED_STATUSES
    ]
    by_slug = {document.slug: document for document in materialized}
    persona_document = by_slug.get(persona_slug)
    brand_document = next(
        (document for document in materialized if document.node_type == "brand"),
        None,
    )
    campaign_document = next(
        (document for document in materialized if document.node_type == "campaign"),
        None,
    )
    audience_document = next(
        (document for document in materialized if document.node_type == "audience"),
        None,
    )
    catalog_document = next(
        (document for document in documents if document.node_type == "catalog"),
        None,
    )
    for name, document in (
        ("persona", persona_document),
        ("brand", brand_document),
        ("campaign", campaign_document),
        ("audience", audience_document),
        ("catalog", catalog_document),
    ):
        if document is None:
            errors.append(f"required {name} document is missing")
    if errors:
        raise GraphDocumentCompileError(errors)

    nodes: list[Node] = []
    edges: list[Edge] = []
    ids: dict[str, str] = {}
    node_types: dict[str, str] = {}

    def add_node(
        *,
        node_type: str,
        slug: str,
        label: str,
        data: dict[str, Any],
        parent_slug: str | None,
    ) -> str:
        node_id = f"node:{node_type}:{slug}"
        ids[slug] = node_id
        node_types[slug] = node_type
        nodes.append(
            Node(
                id=node_id,
                node_type=node_type,
                slug=slug,
                label=label,
                parent_id=ids.get(parent_slug or ""),
                data=data,
            )
        )
        if parent_slug:
            parent_type = node_types[parent_slug]
            edges.append(
                Edge(
                    id=f"edge:primary:{parent_slug}:{slug}",
                    source=ids[parent_slug],
                    target=node_id,
                    relation=_relation_for(parent_type, node_type),
                    primary_tree=True,
                    metadata={"source": "sdr_markdown", "active": True},
                )
            )
        return node_id

    def add_document(document: SourceDocument, parent_slug: str | None) -> None:
        add_node(
            node_type=document.node_type,
            slug=document.slug,
            label=str(
                document.meta.get("title")
                or document.meta.get("name")
                or document.slug
            ),
            data=_node_data(document),
            parent_slug=parent_slug,
        )

    add_document(persona_document, None)
    add_document(brand_document, persona_slug)
    add_document(campaign_document, brand_document.slug)
    add_document(audience_document, campaign_document.slug)

    # Briefings, tone and rules keep their complete Markdown bodies and attach
    # to the approved commercial parent declared by relation (brand by default).
    for node_type in ("briefing", "tone", "rule"):
        for document in (
            item for item in materialized if item.node_type == node_type
        ):
            parent_slug = _first_relation_target(document.meta) or brand_document.slug
            if parent_slug not in ids:
                parent_slug = brand_document.slug
            add_document(document, parent_slug)

    category_slugs: set[str] = set()
    for category in catalog_document.meta.get("categories") or []:
        slug = str(category.get("slug") or "").strip()
        category_slugs.add(slug)
        category_body = (
            f"{category.get('title')}\n\n"
            f"Fonte: {category.get('source') or catalog_document.meta.get('source')}."
        )
        data = {
            **category,
            "persona": persona_slug,
            "type": "product_group",
            "status": _status(category),
            "active": _active(category),
            "tags": ["baita", "product_group", slug],
            "metadata": dict(category.get("metadata") or {}),
            "markdown": category_body,
            "source_file": catalog_document.relative_path,
            "source_checksum": catalog_document.checksum,
            "rag_chunk": {
                "index": 0,
                "checksum": hashlib.sha256(category_body.encode("utf-8")).hexdigest(),
                "source_file": catalog_document.relative_path,
            },
        }
        add_node(
            node_type="product_group",
            slug=slug,
            label=str(category.get("title") or slug),
            data=data,
            parent_slug=audience_document.slug,
        )
        # Preserve the declared campaign relation as a non-tree semantic link.
        edges.append(
            Edge(
                id=f"edge:campaign:{slug}",
                source=ids[slug],
                target=ids[campaign_document.slug],
                relation="part_of_campaign",
                primary_tree=False,
                metadata={"source": "sdr_markdown", "active": True},
            )
        )

    for document in (
        item for item in materialized if item.node_type == "product"
    ):
        category = str(document.meta.get("category") or "").strip()
        if category not in category_slugs:
            errors.append(
                f"{document.relative_path}: orphan product category {category or '(empty)'}"
            )
            continue
        add_document(document, category)
        edges.append(
            Edge(
                id=f"edge:campaign:{document.slug}",
                source=ids[document.slug],
                target=ids[campaign_document.slug],
                relation="part_of_campaign",
                primary_tree=False,
                metadata={"source": "sdr_markdown", "active": True},
            )
        )
    if errors:
        raise GraphDocumentCompileError(errors)

    for node_type in ("copy", "faq"):
        pending = [
            item for item in materialized if item.node_type == node_type
        ]
        while pending:
            progressed = False
            for document in list(pending):
                parent_slug = _first_relation_target(document.meta)
                if parent_slug not in ids:
                    continue
                add_document(document, parent_slug)
                pending.remove(document)
                progressed = True
            if not progressed:
                errors.extend(
                    f"{item.relative_path}: relation target is not materialized"
                    for item in pending
                )
                break
    if errors:
        raise GraphDocumentCompileError(errors)

    embedded_slug = "embedded-default"
    embedded_id = add_node(
        node_type="embedded",
        slug=embedded_slug,
        label="Embedded",
        data={
            "source": "sdr_markdown",
            "status": "validated",
            "active": True,
            "protected": True,
            "markdown": "# Embedded\n\nFAQs aprovadas da persona.",
        },
        parent_slug=None,
    )
    nodes[-1].parent_id = ids[persona_slug]

    nodes_by_id = {node.id: node for node in nodes}

    def branch_path(node_id: str) -> list[str]:
        result: list[str] = []
        current = nodes_by_id.get(node_id)
        seen: set[str] = set()
        while current and current.id not in seen:
            seen.add(current.id)
            result.insert(0, current.id)
            current = nodes_by_id.get(current.parent_id or "")
        return result

    for node in nodes:
        if node.node_type != "faq":
            continue
        parent = nodes_by_id.get(node.parent_id or "")
        node.data["source_node_id"] = parent.id if parent else None
        node.data["source_node_type"] = parent.node_type if parent else None
        node.data["branch_path"] = branch_path(parent.id) if parent else []
        edges.append(
            Edge(
                id=f"edge:embedded:{node.slug}",
                source=node.id,
                target=embedded_id,
                relation="visible_to_agent",
                primary_tree=False,
                metadata={"approved": True, "active": True},
            )
        )

    graph = GraphJson(
        schema_version="2.0",
        graph_id=f"{persona_slug}-markdown",
        tenant=tenant,
        persona_slug=persona_slug,
        brand_slug=brand_document.slug,
        status="published",
        nodes=nodes,
        edges=edges,
    )
    is_valid, graph_errors = graph_json_v2_validator.validate_graph_json(graph)
    if not is_valid:
        raise GraphDocumentCompileError(graph_errors)
    return graph

