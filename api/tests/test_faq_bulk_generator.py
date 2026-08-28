from __future__ import annotations

import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from services import faq_bulk_generator


def _chain():
    return [
        {"node_type": "product", "title": "Vestido de teste", "content": "Vestido feminino, tamanho unico, R$ 79,90."},
        {"node_type": "campaign", "title": "Catalogo de teste", "content": "Campanha de qualificacao."},
        {"node_type": "brand", "title": "Marca de teste", "content": "Marca de moda feminina."},
    ]


def test_generate_faqs_for_chain_parses_llm_json_array(monkeypatch):
    canned = (
        'Aqui estao as FAQs:\n'
        '[{"question": "Tem tamanho P?", "answer": "O modelo e tamanho unico."}, '
        '{"question": "Qual o preco?", "answer": "R$ 79,90."}]'
    )
    monkeypatch.setattr(
        faq_bulk_generator.ModelRouter, "messages_create",
        lambda self, **kwargs: canned,
    )
    pairs = faq_bulk_generator.generate_faqs_for_chain(_chain(), max_questions=5)
    assert pairs == [
        {"question": "Tem tamanho P?", "answer": "O modelo e tamanho unico."},
        {"question": "Qual o preco?", "answer": "R$ 79,90."},
    ]


def test_generate_faqs_for_chain_returns_empty_on_unparseable_output(monkeypatch):
    monkeypatch.setattr(
        faq_bulk_generator.ModelRouter, "messages_create",
        lambda self, **kwargs: "desculpe, nao consigo ajudar com isso",
    )
    assert faq_bulk_generator.generate_faqs_for_chain(_chain()) == []


def test_generate_faqs_for_chain_empty_branch_short_circuits(monkeypatch):
    calls = []
    monkeypatch.setattr(
        faq_bulk_generator.ModelRouter, "messages_create",
        lambda self, **kwargs: calls.append(1) or "[]",
    )
    assert faq_bulk_generator.generate_faqs_for_chain([]) == []
    assert calls == []  # never calls the model for an empty branch


def test_generate_faqs_for_chain_includes_skill_content_in_prompt(monkeypatch):
    captured = {}

    def fake_create(self, **kwargs):
        captured["messages"] = kwargs["messages"]
        return "[]"

    monkeypatch.setattr(faq_bulk_generator.ModelRouter, "messages_create", fake_create)
    faq_bulk_generator.generate_faqs_for_chain(_chain(), skills=("aurora-premium-sdr",))
    prompt_text = captured["messages"][0]["content"]
    assert "Skill: aurora-premium-sdr" in prompt_text


def test_build_chain_from_live_graph_walks_contains_edges_to_root():
    nodes = [
        {"id": "n-product", "node_type": "product", "title": "Produto X", "summary": "sum"},
        {"id": "n-group", "node_type": "product_group", "title": "Grupo Y", "summary": "sum"},
        {"id": "n-campaign", "node_type": "campaign", "title": "Campanha Z", "summary": "sum"},
    ]
    edges = [
        {"source_node_id": "n-campaign", "target_node_id": "n-group", "relation_type": "contains", "metadata": {}},
        {"source_node_id": "n-group", "target_node_id": "n-product", "relation_type": "contains", "metadata": {}},
        # a non-"contains" edge involving the same nodes must be ignored
        {"source_node_id": "n-product", "target_node_id": "n-group", "relation_type": "same_topic_as", "metadata": {}},
    ]
    chain = faq_bulk_generator.build_chain_from_live_graph(nodes, edges, "n-product")
    assert [c["title"] for c in chain] == ["Produto X", "Grupo Y", "Campanha Z"]


def test_build_chain_from_live_graph_ignores_inactive_edges():
    nodes = [
        {"id": "n-product", "node_type": "product", "title": "Produto X", "summary": "sum"},
        {"id": "n-group", "node_type": "product_group", "title": "Grupo Y", "summary": "sum"},
    ]
    edges = [
        {"source_node_id": "n-group", "target_node_id": "n-product", "relation_type": "contains", "metadata": {"active": False}},
    ]
    chain = faq_bulk_generator.build_chain_from_live_graph(nodes, edges, "n-product")
    assert [c["title"] for c in chain] == ["Produto X"]

