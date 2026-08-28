"""One-time deterministic migration for the Aurora Graph JSON fixture."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schemas.graph_json_v2 import GraphJson
from services import graph_markdown, graph_json_v2_validator


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "aurora_graph_v2.json"


def migrate(path: Path = FIXTURE) -> dict:
    original = GraphJson.model_validate(json.loads(path.read_text(encoding="utf-8")))
    normalized = graph_markdown.canonicalize_graph(original)
    normalized.graph_id = "aurora-premium-automotive-markdown"
    valid, errors = graph_json_v2_validator.validate_graph_json(normalized)
    if not valid:
        raise RuntimeError(errors)
    if [node.id for node in original.nodes] != [node.id for node in normalized.nodes]:
        raise RuntimeError("node identity changed")
    if [edge.model_dump() for edge in original.edges] != [edge.model_dump() for edge in normalized.edges]:
        raise RuntimeError("edge contract changed")
    path.write_text(
        json.dumps(normalized.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    factual = [
        node for node in normalized.nodes
        if node.node_type in graph_markdown.FACTUAL_NODE_TYPES
    ]
    return {
        "nodes": len(normalized.nodes),
        "edges": len(normalized.edges),
        "markdown_documents": sum(
            1 for node in factual if (node.data or {}).get("markdown")
        ),
        "faqs": sum(1 for node in factual if node.node_type == "faq"),
    }


if __name__ == "__main__":
    print(migrate())
