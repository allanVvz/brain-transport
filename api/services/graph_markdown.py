"""Deterministic Markdown renderers for closed Graph JSON versions."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from schemas.graph_json_v2 import GraphJson, MarkdownDocument, Node


FACTUAL_NODE_TYPES = {
    "persona", "brand", "briefing", "campaign", "audience", "product_group",
    "product", "service", "offer", "copy", "faq", "rule", "tone", "asset",
}
VALIDATED_STATUSES = {"approved", "validated", "active", "ativo", "embedded"}
_NON_FACTUAL_KEYS = {
    "active", "branch_path", "content_hash", "empty", "file_path",
    "graph_checksum", "graph_id", "graph_json_id", "graph_json_import",
    "graph_json_node_id", "graph_json_parent_id", "graph_version",
    "knowledge_item_id", "knowledge_node_id", "language", "markdown",
    "markdown_checksum", "markdown_renderer", "markdown_document", "metadata",
    "parent_id", "protected", "question_count", "schema_version", "session_id",
    "source", "source_node_id", "source_node_type", "status", "summary", "tags",
    "validation_status", "action",
}


class GraphMarkdownError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("Canonical Markdown validation failed")
        self.errors = errors


def _sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _heading(value: str) -> str:
    return str(value or "").replace("_", " ").strip().capitalize()


def _render_value(key: str, value: Any, *, depth: int = 2) -> list[str]:
    if value is None or value == "" or value == [] or value == {}:
        return []
    title = _heading(key)
    prefix = "#" * min(6, depth)
    if isinstance(value, dict):
        lines = [f"{prefix} {title}", ""]
        for child_key in sorted(value):
            child_value = value[child_key]
            if isinstance(child_value, (dict, list)):
                rendered = _render_value(child_key, child_value, depth=depth + 1)
                if rendered:
                    lines.extend(rendered)
                    lines.append("")
            elif child_value is not None and child_value != "":
                lines.append(f"- **{_heading(child_key)}:** {child_value}")
        return lines
    if isinstance(value, list):
        lines = [f"{prefix} {title}", ""]
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"- `{json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}`")
            else:
                lines.append(f"- {item}")
        return lines
    return [f"{prefix} {title}", "", str(value)]


def _legacy_markdown_for_node(node: Node) -> str:
    data = dict(node.data or {})
    existing = str(data.get("markdown") or "").strip()
    if existing:
        return existing
    title = node.label or node.slug
    if node.node_type == "faq":
        question = str(data.get("question") or title).strip()
        answer = str(data.get("answer") or data.get("content") or "").strip()
        if not question or not answer:
            return ""
        return f"# {title}\n\n## Pergunta\n\n{question}\n\n## Resposta\n\n{answer}"
    body = str(data.get("content") or data.get("summary") or "").strip()
    lines = [f"# {title}"]
    if body:
        lines.extend(["", body])
    for key in sorted(data):
        if key in _NON_FACTUAL_KEYS or key == "content":
            continue
        rendered = _render_value(key, data[key])
        if rendered:
            lines.extend(["", *rendered])
    return "\n".join(lines).strip() if len(lines) > 1 else ""


def markdown_for_node(node: Node) -> str:
    """Compatibility renderer used by v2.0 importers."""
    return _legacy_markdown_for_node(node)


def _relation_lines(graph: GraphJson, node: Node) -> tuple[list[str], list[str], list[str], list[str]]:
    by_id = {item.id: item for item in graph.nodes}
    incoming: list[str] = []
    outgoing: list[str] = []
    lateral: list[str] = []
    publications: list[str] = []
    for edge in sorted(graph.edges, key=lambda item: item.id):
        if edge.lifecycle.status == "revoked" or node.id not in {edge.source, edge.target}:
            continue
        relation = edge.relation_type or edge.relation or "references"
        other_id = edge.target if edge.source == node.id else edge.source
        other = by_id.get(other_id)
        if not other:
            continue
        arrow = "â†’" if edge.source == node.id else "â†"
        line = f"- `{relation}` {arrow} {other.title or other.label} (`{other.node_type}`)"
        if relation == "publishes_to":
            publications.append(
                f"- {other.title or other.label} â€” {edge.lifecycle.status}"
            )
        elif relation == "contains" and edge.target == node.id:
            incoming.append(line)
        elif relation == "contains":
            outgoing.append(line)
        else:
            lateral.append(line)
    return incoming, outgoing, lateral, publications


def _render_v21_node(graph: GraphJson, node: Node) -> str:
    incoming, outgoing, lateral, publications = _relation_lines(graph, node)
    spec = dict(node.spec or {})
    if not spec:
        spec = {
            key: value for key, value in (node.data or {}).items()
            if key not in _NON_FACTUAL_KEYS and key not in {"content", "answer"}
        }
    summary = str(
        spec.pop("summary", None)
        or (node.data or {}).get("summary")
        or (node.data or {}).get("content")
        or ""
    ).strip()
    if node.node_type == "faq":
        spec.setdefault("question", (node.data or {}).get("question") or node.title)
        spec.setdefault("answer", (node.data or {}).get("answer") or (node.data or {}).get("content"))

    node_material = json.dumps(
        {
            "id": node.id,
            "type": node.node_type,
            "title": node.title,
            "lifecycle": node.lifecycle.model_dump(mode="json"),
            "provenance": node.provenance.model_dump(mode="json"),
            "spec": spec,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    frontmatter = {
        "graph_id": graph.graph_id,
        "graph_version": graph.graph_version,
        "node_id": node.id,
        "node_type": node.node_type,
        "slug": node.slug,
        "status": node.lifecycle.status,
        "source": node.provenance.source or node.provenance.source_type or "pending_source",
        "renderer": f"{node.node_type}@1",
        "content_checksum": _sha256(node_material),
    }
    lines = ["---", json.dumps(frontmatter, ensure_ascii=False, sort_keys=True), "---", "", f"# {node.title or node.label or node.slug}"]
    if summary:
        lines.extend(["", "## Resumo", "", summary])
    if spec:
        lines.extend(["", "## Dados estruturados", ""])
        for key in sorted(spec):
            value = spec[key]
            if value is not None and value != "" and value != [] and value != {}:
                rendered = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
                lines.append(f"- **{_heading(key)}:** {rendered}")
    for title, values in (
        ("RelaÃ§Ãµes ascendentes", incoming),
        ("RelaÃ§Ãµes descendentes", outgoing),
        ("RelaÃ§Ãµes laterais", lateral),
        ("PublicaÃ§Ã£o", publications),
    ):
        if values:
            lines.extend(["", f"## {title}", "", *values])
    lines.extend([
        "", "## Origem e rastreabilidade", "",
        f"- RevisÃ£o: {node.lifecycle.revision}",
        f"- Fonte: {frontmatter['source']}",
    ])
    return "\n".join(lines).strip()


# Explicit renderer names are part of the public domain contract.  They share
# the deterministic core today and can evolve independently by bumping @N.
def render_persona_md(graph: GraphJson, node: Node) -> str: return _render_v21_node(graph, node)
def render_brand_md(graph: GraphJson, node: Node) -> str: return _render_v21_node(graph, node)
def render_briefing_md(graph: GraphJson, node: Node) -> str: return _render_v21_node(graph, node)
def render_campaign_md(graph: GraphJson, node: Node) -> str: return _render_v21_node(graph, node)
def render_audience_md(graph: GraphJson, node: Node) -> str: return _render_v21_node(graph, node)
def render_product_group_md(graph: GraphJson, node: Node) -> str: return _render_v21_node(graph, node)
def render_product_md(graph: GraphJson, node: Node) -> str: return _render_v21_node(graph, node)
def render_service_md(graph: GraphJson, node: Node) -> str: return _render_v21_node(graph, node)
def render_offer_md(graph: GraphJson, node: Node) -> str: return _render_v21_node(graph, node)
def render_copy_md(graph: GraphJson, node: Node) -> str: return _render_v21_node(graph, node)
def render_faq_md(graph: GraphJson, node: Node) -> str: return _render_v21_node(graph, node)
def render_asset_md(graph: GraphJson, node: Node) -> str: return _render_v21_node(graph, node)
def render_rule_md(graph: GraphJson, node: Node) -> str: return _render_v21_node(graph, node)
def render_tone_md(graph: GraphJson, node: Node) -> str: return _render_v21_node(graph, node)
def render_action_node_md(graph: GraphJson, node: Node) -> str: return _render_v21_node(graph, node)


_RENDERERS: dict[str, Callable[[GraphJson, Node], str]] = {
    "persona": render_persona_md, "brand": render_brand_md,
    "briefing": render_briefing_md, "campaign": render_campaign_md,
    "audience": render_audience_md, "product_group": render_product_group_md,
    "product": render_product_md, "service": render_service_md,
    "offer": render_offer_md, "copy": render_copy_md,
    "faq": render_faq_md, "asset": render_asset_md, "rule": render_rule_md,
    "tone": render_tone_md, "gallery": render_action_node_md,
    "embedded": render_action_node_md, "marketing_workspace": render_action_node_md,
}


def canonicalize_graph(graph: GraphJson, *, reject_markdown_drift: bool = True) -> GraphJson:
    normalized = GraphJson.model_validate(graph.model_dump())
    errors: list[str] = []
    for node in normalized.nodes:
        if normalized.schema_version == "2.0":
            if node.node_type not in FACTUAL_NODE_TYPES:
                continue
            markdown = _legacy_markdown_for_node(node)
            status = str((node.data or {}).get("validation_status") or (node.data or {}).get("status") or "").lower()
            if not markdown:
                if status in VALIDATED_STATUSES:
                    errors.append(f"validated factual node {node.id} has insufficient content")
                continue
            node.data = {**(node.data or {}), "markdown": markdown}
            if node.node_type == "faq":
                node.data["markdown_document"] = True
                node.data["question_count"] = max(1, int(node.data.get("question_count") or 1))
            continue

        renderer = _RENDERERS.get(node.node_type)
        if renderer is None:
            errors.append(f"missing Markdown renderer for node type {node.node_type}")
            continue
        rendered = renderer(normalized, node)
        rendered_checksum = _sha256(rendered)
        supplied = node.markdown
        if supplied and reject_markdown_drift and supplied.content.strip() != rendered.strip():
            errors.append(f"node {node.id} Markdown differs from canonical renderer output")
            continue
        node.markdown = MarkdownDocument(
            renderer=f"{node.node_type}@1",
            checksum=rendered_checksum,
            content=rendered,
        )
        node.data = {
            **(node.data or {}),
            "markdown": rendered,
            "markdown_checksum": rendered_checksum,
            "markdown_renderer": f"{node.node_type}@1",
        }
    if errors:
        raise GraphMarkdownError(errors)
    return normalized

