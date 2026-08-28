from __future__ import annotations

import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from routes import knowledge as knowledge_routes
from services import faq_bulk_generator, graph_compiler_v3


class _FakeRequest:
    headers: dict = {}


PERSONA_ID = "4acb2739-127e-4143-acf5-f5c3ea1aaa98"
PERSONA_SLUG = "tock-fatal"
NODE_PERSONA = "11111111-1111-1111-1111-111111111111"
NODE_AUDIENCE = "22222222-2222-2222-2222-222222222222"
NODE_PRODUCT = "33333333-3333-3333-3333-333333333333"


def _persona_node(node_id):
    return {
        "id": node_id, "persona_id": PERSONA_ID, "node_type": "persona", "slug": PERSONA_SLUG,
        "title": PERSONA_SLUG, "summary": f"Persona {PERSONA_SLUG}.", "tags": [], "status": "validated",
        "metadata": {"graph_json_node_id": f"persona:{PERSONA_SLUG}", "source": "test"},
    }


def _product_node():
    return {
        "id": NODE_PRODUCT, "persona_id": PERSONA_ID, "node_type": "product", "slug": "produto-x",
        "title": "Produto X", "summary": "Um produto de teste.", "tags": [], "status": "validated",
        "metadata": {"graph_json_node_id": "product:produto-x", "source": "test"},
    }


def _audience_node():
    return {
        "id": NODE_AUDIENCE, "persona_id": PERSONA_ID, "node_type": "audience", "slug": "retail",
        "title": "Retail", "summary": "retail branch", "tags": [], "status": "validated",
        "metadata": {"graph_json_node_id": "audience:retail", "capabilities": {"branch_anchor": True}, "source": "test"},
    }


def _base_nodes():
    return [_persona_node(NODE_PERSONA), _audience_node(), _product_node()]


def _base_edges():
    return [
        {
            "id": "edge-0", "source_node_id": NODE_PERSONA, "target_node_id": NODE_AUDIENCE,
            "relation_type": "contains", "metadata": {"active": True},
        },
        {
            "id": "edge-1", "source_node_id": NODE_PERSONA, "target_node_id": NODE_PRODUCT,
            "relation_type": "contains", "metadata": {"active": True},
        },
    ]


def test_generate_faqs_route_builds_plan_and_stores_session(monkeypatch):
    monkeypatch.setattr(knowledge_routes.auth_service, "assert_persona_access", lambda *a, **k: None)
    monkeypatch.setattr(
        knowledge_routes.auth_service, "current_user", lambda _req: {"id": "user-1", "email": "op@test"}
    )
    monkeypatch.setattr(knowledge_routes.supabase_client, "get_knowledge_node", lambda _id: _product_node())
    monkeypatch.setattr(
        knowledge_routes.supabase_client, "get_persona_by_id",
        lambda _id: {"id": PERSONA_ID, "slug": PERSONA_SLUG},
    )
    monkeypatch.setattr(
        knowledge_routes.supabase_client, "list_all_knowledge_graph",
        lambda **_kw: (_base_nodes(), _base_edges()),
    )
    monkeypatch.setattr(
        faq_bulk_generator.ModelRouter, "messages_create",
        lambda self, **kw: '[{"question": "Tem tamanho?", "answer": "Tamanho unico."}]',
    )
    saved_sessions = {}
    monkeypatch.setattr(
        knowledge_routes, "generate_faqs_for_graph_node",
        knowledge_routes.generate_faqs_for_graph_node,  # no-op, keep reference stable
    )

    # patch _save_session at the source module so the route's local import binds it
    import services.kb_intake_service as kb_intake_service
    monkeypatch.setattr(kb_intake_service, "_save_session", lambda s: saved_sessions.__setitem__(s["id"], s))

    body = knowledge_routes.GenerateFaqsBody(max_questions=5)
    result = knowledge_routes.generate_faqs_for_graph_node(NODE_PRODUCT, body, request=_FakeRequest())

    assert result["ok"] is True
    assert result["faqs"] == [{"question": "Tem tamanho?", "answer": "Tamanho unico."}]
    assert result["publication_plan"]["validation_errors"] == []
    assert result["publication_plan"]["disposition"] == "awaiting_approval"
    assert result["session_id"] in saved_sessions
    stored = saved_sessions[result["session_id"]]
    assert stored["stage"] == "awaiting_publication_approval"
    new_faq_ids = [n["id"] for n in stored["pending_graph_bundle"]["nodes"] if n["node_type"] == "faq"]
    assert len(new_faq_ids) == 1


def test_generate_faqs_route_returns_ok_false_when_generation_empty(monkeypatch):
    monkeypatch.setattr(knowledge_routes.auth_service, "assert_persona_access", lambda *a, **k: None)
    monkeypatch.setattr(
        knowledge_routes.auth_service, "current_user", lambda _req: {"id": "user-1", "email": "op@test"}
    )
    monkeypatch.setattr(knowledge_routes.supabase_client, "get_knowledge_node", lambda _id: _product_node())
    monkeypatch.setattr(
        knowledge_routes.supabase_client, "get_persona_by_id",
        lambda _id: {"id": PERSONA_ID, "slug": PERSONA_SLUG},
    )
    monkeypatch.setattr(
        knowledge_routes.supabase_client, "list_all_knowledge_graph",
        lambda **_kw: (_base_nodes(), _base_edges()),
    )
    monkeypatch.setattr(
        faq_bulk_generator.ModelRouter, "messages_create", lambda self, **kw: "nao consigo ajudar"
    )

    body = knowledge_routes.GenerateFaqsBody(max_questions=5)
    result = knowledge_routes.generate_faqs_for_graph_node(NODE_PRODUCT, body, request=_FakeRequest())
    assert result == {"ok": False, "faqs": [], "error": "FAQ generation produced no usable output"}

