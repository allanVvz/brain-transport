from __future__ import annotations

import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from services import graph_bundle, graph_bundle_adapter
from services.document_candidate_extractor import _structured_markdown_candidates

PERSONA_ID = "4acb2739-127e-4143-acf5-f5c3ea1aaa98"
PERSONA_SLUG = "tock-fatal"


def _plan(*entries: dict) -> dict:
    return {"persona_slug": PERSONA_SLUG, "source": "kb-intake-test", "entries": list(entries)}


def _entry(content_type: str, slug: str, *, status: str, parent_slug: str, title: str | None = None) -> dict:
    return {
        "content_type": content_type,
        "slug": slug,
        "title": title or slug,
        "status": status,
        "content": f"Conteudo de {slug}.",
        "metadata": {"parent_slug": parent_slug},
    }


def test_only_confirmed_entries_become_bundle_nodes():
    plan = _plan(
        _entry("brand", "minha-marca", status="confirmado", parent_slug="self"),
        _entry("campaign", "camp-1", status="confirmado", parent_slug="minha-marca"),
        _entry("product", "produto-1", status="confirmado", parent_slug="camp-1"),
        _entry("product", "produto-2", status="pendente_validacao", parent_slug="camp-1"),
    )
    session = {"id": "sess-1", "persona_id": PERSONA_ID, "persona_slug": PERSONA_SLUG}

    result = graph_bundle_adapter.normalized_plan_to_graph_bundle(plan, session)
    node_ids = {node["id"] for node in result["bundle"]["nodes"]}

    assert "product:produto-1" in node_ids
    assert "product:produto-2" not in node_ids
    assert {item["id"] for item in result["held_back"]} == {"product:produto-2"}


def test_child_of_unconfirmed_parent_is_held_back_too():
    plan = _plan(
        _entry("brand", "minha-marca", status="confirmado", parent_slug="self"),
        _entry("product", "produto-pendente", status="pendente_validacao", parent_slug="minha-marca"),
        _entry("copy", "copy-produto", status="confirmado", parent_slug="produto-pendente"),
    )
    session = {"id": "sess-1", "persona_id": PERSONA_ID, "persona_slug": PERSONA_SLUG}

    result = graph_bundle_adapter.normalized_plan_to_graph_bundle(plan, session)
    node_ids = {node["id"] for node in result["bundle"]["nodes"]}
    held_back_ids = {item["id"] for item in result["held_back"]}

    assert "copy:copy-produto" not in node_ids
    assert "copy:copy-produto" in held_back_ids
    assert "produto ainda nao" not in "".join(held_back_ids)  # sanity: no crash-string leaked


def test_bundle_shape_is_contains_tree_rooted_in_persona():
    plan = _plan(
        _entry("brand", "minha-marca", status="confirmado", parent_slug="self"),
        _entry("campaign", "camp-1", status="confirmado", parent_slug="minha-marca"),
    )
    session = {"id": "sess-1", "persona_id": PERSONA_ID, "persona_slug": PERSONA_SLUG}

    result = graph_bundle_adapter.normalized_plan_to_graph_bundle(plan, session)
    bundle = result["bundle"]

    # normalize_bundle enforces: single persona root, every node reachable
    # via exactly one "contains" edge. A structural error here means the
    # adapter produced something graph_bundle would reject.
    normalized = graph_bundle.normalize_bundle(bundle)
    assert normalized["persona"]["slug"] == PERSONA_SLUG
    assert {n["id"] for n in normalized["nodes"]} == {n["id"] for n in bundle["nodes"]}


def test_base_bundle_nodes_are_additive_not_duplicated():
    base_bundle = {
        "nodes": [{
            "id": "campaign:camp-1", "node_type": "campaign", "slug": "camp-1",
            "title": "Campanha 1", "summary": "Campanha ja publicada.", "tags": [],
            "status": "validated", "projection_node_id": "11111111-1111-1111-1111-111111111111",
            "data": {"source": "previous_publish"},
        }],
        "edges": [],
    }
    plan = _plan(
        _entry("product", "produto-novo", status="confirmado", parent_slug="camp-1"),
    )
    session = {"id": "sess-1", "persona_id": PERSONA_ID, "persona_slug": PERSONA_SLUG}

    result = graph_bundle_adapter.normalized_plan_to_graph_bundle(plan, session, base_bundle=base_bundle)
    bundle = result["bundle"]
    node_ids = [n["id"] for n in bundle["nodes"]]

    assert node_ids.count("campaign:camp-1") == 1
    assert "product:produto-novo" in node_ids
    assert {"source": "campaign:camp-1", "target": "product:produto-novo", "relation_type": "contains"} in [
        {k: e[k] for k in ("source", "target", "relation_type")} for e in bundle["edges"]
    ]


_SAMPLE_DOCUMENT = """
### Vestido de teste â€” R$ 79,90
- **Descricao:** Vestido feminino de teste.
- **Tamanho:** Unico
- **Preco:** R$ 79,90
- **Copy:** Uma peca versatil para o dia a dia.

### Calca de teste
- **Descricao:** Calca feminina de teste.
- **Tamanhos:** P ao GG
- **Preco:** R$ 99,90
- **Copy:** Confortavel e facil de combinar.
"""


def test_semantic_links_from_plan_survive_into_the_bundle():
    """sofia_tools.tool_connect_nodes appends to plan["links"] -- before this
    fix, normalized_plan_to_graph_bundle silently dropped every entry there,
    so no semantic edge (e.g. include_in_branch) Sofia ever created could
    reach the compiled GraphBundle."""
    plan = _plan(
        _entry("brand", "minha-marca", status="confirmado", parent_slug="self"),
        _entry("product", "produto-1", status="confirmado", parent_slug="minha-marca"),
    )
    plan["links"] = [{
        "source_slug": "tock-fatal", "target_slug": "produto-1",
        "relation_type": "visible_to_agent", "metadata": {"include_in_branch": True},
    }]
    session = {"id": "sess-1", "persona_id": PERSONA_ID, "persona_slug": PERSONA_SLUG}

    result = graph_bundle_adapter.normalized_plan_to_graph_bundle(plan, session)
    bundle = result["bundle"]

    semantic_edges = [e for e in bundle["edges"] if e["relation_type"] == "visible_to_agent"]
    assert len(semantic_edges) == 1
    edge = semantic_edges[0]
    assert edge["source"] == f"persona:{PERSONA_SLUG}"
    assert edge["target"] == "product:produto-1"
    assert edge["metadata"]["include_in_branch"] is True

    # Bundle must still be structurally valid (this doesn't replace the
    # "contains" primary tree, it's additive).
    normalized = graph_bundle.normalize_bundle(bundle)
    assert {n["id"] for n in normalized["nodes"]} == {n["id"] for n in bundle["nodes"]}


def test_semantic_link_to_held_back_node_is_skipped():
    plan = _plan(
        _entry("brand", "minha-marca", status="confirmado", parent_slug="self"),
        _entry("product", "produto-pendente", status="pendente_validacao", parent_slug="minha-marca"),
    )
    plan["links"] = [{
        "source_slug": "tock-fatal", "target_slug": "produto-pendente",
        "relation_type": "visible_to_agent", "metadata": {"include_in_branch": True},
    }]
    session = {"id": "sess-1", "persona_id": PERSONA_ID, "persona_slug": PERSONA_SLUG}

    result = graph_bundle_adapter.normalized_plan_to_graph_bundle(plan, session)
    bundle = result["bundle"]

    assert not [e for e in bundle["edges"] if e["relation_type"] == "visible_to_agent"]


def _minimal_bundle_with_branch_anchor():
    return {
        "bundle_version": "1.0",
        "persona": {"id": PERSONA_ID, "slug": PERSONA_SLUG},
        "metadata": {
            "embedding_profile": {
                "embedding_provider": "local",
                "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "embedding_dimension": 1536,
            },
        },
        "nodes": [
            {
                "id": f"persona:{PERSONA_SLUG}", "node_type": "persona", "slug": PERSONA_SLUG,
                "title": PERSONA_SLUG, "summary": "persona", "tags": [], "status": "validated",
                "data": {"source": "test"},
            },
            {
                "id": "audience:retail", "node_type": "audience", "slug": "retail",
                "title": "Retail", "summary": "retail branch", "tags": [], "status": "validated",
                "data": {"source": "test", "capabilities": {"branch_anchor": True}},
            },
            {
                "id": "faq:greeting", "node_type": "faq", "slug": "greeting",
                "title": "Oi", "summary": "Oi! Tudo bem?", "tags": [], "status": "validated",
                "data": {"source": "test", "question": "Oi", "answer": "Oi! Tudo bem?"},
            },
            {
                "id": "copy:orphan", "node_type": "copy", "slug": "orphan",
                "title": "Copy orfa", "summary": "Uma copy sem ramo.", "tags": [], "status": "validated",
                "data": {"source": "test"},
            },
        ],
        "edges": [
            {"id": "e1", "source": f"persona:{PERSONA_SLUG}", "target": "audience:retail", "relation_type": "contains", "weight": 1.0, "metadata": {}},
            {"id": "e2", "source": "audience:retail", "target": "faq:greeting", "relation_type": "contains", "weight": 1.0, "metadata": {}},
            {"id": "e3", "source": f"persona:{PERSONA_SLUG}", "target": "copy:orphan", "relation_type": "contains", "weight": 1.0, "metadata": {}},
        ],
    }


def test_ensure_branch_reachability_repairs_orphan_node():
    bundle = _minimal_bundle_with_branch_anchor()
    repaired = graph_bundle_adapter.ensure_branch_reachability(bundle)

    auto_edges = [e for e in repaired["edges"] if e.get("metadata", {}).get("source") == "auto_branch_reachability_repair"]
    assert len(auto_edges) == 1
    assert auto_edges[0]["source"] == f"persona:{PERSONA_SLUG}"
    assert auto_edges[0]["target"] == "copy:orphan"
    assert auto_edges[0]["metadata"]["include_in_branch"] is True

    document = graph_bundle.compile_bundle(repaired)
    assert "copy:orphan" in document["branch_memberships"]["audience:retail"]


def test_ensure_branch_reachability_is_noop_when_everything_already_reachable():
    bundle = _minimal_bundle_with_branch_anchor()
    # Move the orphan copy under the audience via "contains" instead --
    # already reachable, should need no repair.
    for edge in bundle["edges"]:
        if edge["id"] == "e3":
            edge["source"] = "audience:retail"

    repaired = graph_bundle_adapter.ensure_branch_reachability(bundle)
    auto_edges = [e for e in repaired["edges"] if e.get("metadata", {}).get("source") == "auto_branch_reachability_repair"]
    assert auto_edges == []
    assert repaired["edges"] == bundle["edges"]


def test_document_extractor_structured_path_matches_catalog_shape():
    candidates = _structured_markdown_candidates(_SAMPLE_DOCUMENT)
    assert len(candidates) == 2
    titles = {c["title"] for c in candidates}
    assert titles == {"Vestido de teste", "Calca de teste"}
    prices = {c["title"]: c["prices"] for c in candidates}
    assert prices["Vestido de teste"] == [79.90]
    assert prices["Calca de teste"] == [99.90]
    # Title's embedded price suffix must be stripped, matching this
    # session's manual cleanup of the real Tock Fatal catalog titles.
    assert "R$" not in [c["title"] for c in candidates][0]

