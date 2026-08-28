import json
import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from schemas.graph_json_v2 import GraphJson, Node
from services import (
    graph_markdown,
    graph_json_v2_store,
    knowledge_catalog,
    knowledge_graph,
)


FIXTURE = Path(__file__).resolve().parents[1] / "scripts" / "fixtures" / "aurora_graph_v2.json"


def test_aurora_markdown_contract_and_catalog():
    graph = GraphJson.model_validate(json.loads(FIXTURE.read_text(encoding="utf-8")))
    normalized = graph_markdown.canonicalize_graph(graph)
    factual = [
        node for node in normalized.nodes
        if node.node_type in graph_markdown.FACTUAL_NODE_TYPES
    ]
    assert len(normalized.nodes) == 59
    assert len(normalized.edges) == 80
    assert len(factual) == 57
    assert all((node.data or {}).get("markdown") for node in factual)

    faq_nodes = [node for node in factual if node.node_type == "faq"]
    embedded = next(node for node in normalized.nodes if node.node_type == "embedded")
    faq_edges = [
        edge for edge in normalized.edges
        if edge.target == embedded.id and edge.source in {node.id for node in faq_nodes}
    ]
    assert len(faq_nodes) == 23
    assert len(faq_edges) == 23
    assert len({edge.source for edge in faq_edges}) == 23
    assert all(node.data["markdown_document"] is True for node in faq_nodes)
    assert all(node.data["question_count"] == 1 for node in faq_nodes)

    catalog = knowledge_catalog.project_graph(
        normalized,
        version=10,
        checksum=graph_json_v2_store.checksum_graph(normalized),
        persona_id="aurora-id",
        persona_name="Aurora",
    )
    assert catalog["graph"]["document_count"] == 57
    assert catalog["categories"][0]["key"] == "faqs"
    assert catalog["categories"][0]["count"] == 23
    assert catalog["embedded"]["faq_count"] == 23


def test_validated_empty_factual_node_is_rejected():
    graph = GraphJson(
        graph_id="empty",
        tenant="test",
        persona_slug="empty",
        status="published",
        nodes=[
            Node(
                id="persona",
                node_type="persona",
                slug="empty",
                label="Empty",
                data={"status": "validated", "source": "test"},
            )
        ],
    )
    try:
        graph_markdown.canonicalize_graph(graph)
    except graph_markdown.GraphMarkdownError as exc:
        assert "insufficient content" in exc.errors[0]
    else:
        raise AssertionError("validated empty node should fail")


def test_operator_context_prioritizes_last_decision_and_keeps_legacy_arrays():
    context = {
        "query_terms": ["lavagem"],
        "nodes": [
            {
                "id": "product",
                "node_type": "product",
                "title": "Lavagem",
                "summary": "R$ 180",
                "validated": True,
                "graph_distance": 0,
                "path": [{"node_id": "persona", "title": "Aurora"}],
            },
            {
                "id": "faq",
                "node_type": "faq",
                "title": "Quanto custa?",
                "summary": "A partir de R$ 180.",
                "validated": True,
                "graph_distance": 1,
                "path": [{"node_id": "persona", "title": "Aurora"}],
            },
        ],
        "edges": [{"id": "edge"}],
        "kb_entries": [],
        "assets": [],
        "evidence_node_ids": ["product"],
    }
    result = knowledge_graph.with_operator_context(context)
    assert result["edges"] == context["edges"]
    assert [item["id"] for item in result["operator_context"]["primary"]] == ["product"]
    assert [item["id"] for item in result["operator_context"]["faq_rules"]] == ["faq"]
    assert result["operator_context"]["graph_path"][0]["title"] == "Aurora"

