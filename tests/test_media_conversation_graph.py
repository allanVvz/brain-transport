"""Received media hangs under campaign -> audience -> conversation -> asset.

A photo a customer sends should be traceable back to the campaign that
prompted the conversation. That requires a `conversation` node type in the
canonical chain, and it requires customer media to stay out of the RAG vector
layer — a lead's own words are not commercial truth.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "api"
for path in (API_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# ── canonical chain ──────────────────────────────────────────────────────

def test_conversation_hangs_under_audience():
    from services.graph_json_v2_validator import CANONICAL_PARENT

    assert CANONICAL_PARENT["conversation"] == ("audience",)
    # ...and audience already hangs under campaign, completing the chain the
    # CRM needs for attribution.
    assert "campaign" in CANONICAL_PARENT["audience"]


def test_asset_may_hang_under_a_conversation():
    from services.graph_json_v2_validator import CANONICAL_PARENT

    assert "conversation" in CANONICAL_PARENT["asset"]


def test_conversation_is_a_known_knowledge_type():
    from services.graph_json_v2_validator import V21_KNOWLEDGE_TYPES

    assert "conversation" in V21_KNOWLEDGE_TYPES


def test_graph_schema_accepts_a_conversation_node():
    from schemas.graph_json_v2 import Node

    node = Node(
        id="node:conversation:42",
        node_type="conversation",
        slug="conversa-42",
        label="Conversa — 5511999999999",
    )
    assert node.node_type == "conversation"


def test_conversation_node_carries_the_lead_and_is_not_rag_eligible(monkeypatch):
    from services import conversation_graph

    captured = {}
    monkeypatch.setattr(
        conversation_graph.supabase_client, "upsert_knowledge_node",
        lambda payload: captured.update(payload) or {"id": "node-1", **payload},
    )
    edges = []
    monkeypatch.setattr(
        conversation_graph.supabase_client, "upsert_knowledge_edge",
        lambda src, tgt, rel, **kw: edges.append((src, tgt, rel, kw)) or {"id": "edge-1"},
    )

    node = conversation_graph.ensure_conversation_node(
        persona_id="persona-1",
        lead={"id": 42, "nome": "Ana"},
        audience_node={"id": "audience-node-1"},
    )

    assert node["id"] == "node-1"
    assert captured["node_type"] == "conversation"
    assert captured["slug"] == "conversa-42"
    assert "source_id" not in captured
    assert captured["metadata"]["lead_id"] == 42
    # Customer conversation content must never feed the vector layer.
    assert captured["metadata"]["rag_eligible"] is False

    source, target, relation, kwargs = edges[0]
    assert (source, target, relation) == ("audience-node-1", "node-1", "contains")
    assert kwargs["metadata"]["primary_tree"] is True


def test_organic_contact_gets_a_conversation_node_without_an_audience_edge(monkeypatch):
    """No campaign is no reason to invent one."""
    from services import conversation_graph

    monkeypatch.setattr(
        conversation_graph.supabase_client, "upsert_knowledge_node",
        lambda payload: {"id": "node-2", **payload},
    )

    def _no_edge(*_a, **_k):
        raise AssertionError("an organic conversation must not get a primary-tree edge")

    monkeypatch.setattr(conversation_graph.supabase_client, "upsert_knowledge_edge", _no_edge)

    node = conversation_graph.ensure_conversation_node(
        persona_id="persona-1", lead={"id": 43}, audience_node=None,
    )
    assert node["id"] == "node-2"


def test_inbound_asset_reports_both_graph_edges_and_never_gets_a_landing_slot(monkeypatch):
    from routes import assets as assets_route
    from services import conversation_graph

    monkeypatch.setattr(conversation_graph.supabase_client, "get_asset", lambda _asset_id: {
        "id": "asset-1", "persona_id": "persona-1", "lead_id": 42,
        "name": "foto.png", "type": "image", "metadata": {"media": {"kind": "image"}},
        "storage_bucket": "whatsapp-media", "storage_path": "persona-1/42/foto.png",
    })
    monkeypatch.setattr(conversation_graph.supabase_client, "get_lead", lambda _lead_id: {"id": 42})
    monkeypatch.setattr(conversation_graph, "_audience_node_for_recipient", lambda *_args: None)
    monkeypatch.setattr(
        conversation_graph, "ensure_conversation_node",
        lambda **_kwargs: {"id": "conversation-1", "node_type": "conversation"},
    )
    captured = {}
    monkeypatch.setattr(assets_route, "_ensure_asset_graph_contract", lambda **kwargs: (
        captured.update(kwargs) or {
            "asset_node": {"id": "asset-node-1"},
            "parent_edge": {"id": "conversation-asset-edge-1"},
            "gallery_node": {"id": "gallery-1"},
            "gallery_edge": {"id": "asset-gallery-edge-1"},
        }
    ))

    attachment = conversation_graph.attach_inbound_asset("asset-1")

    assert attachment == {
        "attached": True,
        "status": "attached",
        "conversation_node_id": "conversation-1",
        "audience_node_id": None,
        "asset_node_id": "asset-node-1",
        "conversation_asset_edge_id": "conversation-asset-edge-1",
        "gallery_node_id": "gallery-1",
        "asset_gallery_edge_id": "asset-gallery-edge-1",
    }
    assert captured["parent_node"]["id"] == "conversation-1"
    assert captured["asset_function"] is None


# ── RAG gate ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("content_type", ["asset", "conversation", "gallery", "service"])
def test_customer_media_never_reaches_the_rag_layer(content_type):
    """The vector layer stays FAQ-only.

    An asset carries whatever the customer photographed or said out loud;
    letting that become retrievable "knowledge" would have the agent quoting
    one lead's words back to another.
    """
    from services.knowledge_rag_intake import is_rag_eligible

    assert is_rag_eligible(content_type) is False
    assert is_rag_eligible("faq") is True
