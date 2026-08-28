from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parent
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from services import graph_bundle, graph_bundle_publisher


def _bundle() -> dict:
    return json.loads(
        (REPO_ROOT / "data" / "graph_bundles" / "tock-fatal" / "sdr-qualification-v1.json")
        .read_text(encoding="utf-8")
    )


def test_stage_bundle_materializes_then_requires_exact_runtime_checksum(monkeypatch):
    bundle = _bundle()
    plan = graph_bundle.build_publication_plan(bundle)
    normalized = graph_bundle.normalize_bundle(bundle)
    persona, compiled_nodes, compiled_edges, _profile = graph_bundle._compiler_inputs(bundle)
    current_nodes = [
        {
            "id": node["projection_node_id"],
            "node_type": node["node_type"],
            "slug": node["slug"],
            "status": node["status"],
            "metadata": {},
        }
        for node in normalized["nodes"]
        if node["node_type"] in {"persona", "embed", "gallery"}
    ]
    current_nodes.append({
        "id": "582e44b8-505d-4e1b-adb0-2c8e3684f980",
        "node_type": "persona",
        "slug": "self",
        "status": "validated",
        "metadata": {"role": "root"},
    })
    graph_reads = iter([
        (current_nodes, []),
        (compiled_nodes + current_nodes[-1:], compiled_edges),
    ])
    monkeypatch.setattr(
        graph_bundle_publisher.supabase_client,
        "get_persona",
        lambda _slug: {**persona, "name": "Tock Fatal"},
    )
    monkeypatch.setattr(
        graph_bundle_publisher.supabase_client,
        "list_all_knowledge_graph",
        lambda **_kwargs: next(graph_reads),
    )
    materialized_nodes: list[dict] = []
    monkeypatch.setattr(
        graph_bundle_publisher.supabase_client,
        "upsert_knowledge_node",
        lambda row: materialized_nodes.append(row) or row,
    )
    replaced_nodes: list[dict] = []
    monkeypatch.setattr(
        graph_bundle_publisher.supabase_client,
        "update_knowledge_node",
        lambda node_id, row, **_kwargs: replaced_nodes.append(row) or {"id": node_id, **row},
    )
    monkeypatch.setattr(
        graph_bundle_publisher.supabase_client,
        "upsert_knowledge_edge",
        lambda *_args, **_kwargs: {"id": "edge-row"},
    )
    replaced_edges: list[dict] = []
    monkeypatch.setattr(
        graph_bundle_publisher.supabase_client,
        "update_knowledge_edge",
        lambda edge_id, row: replaced_edges.append(row) or {"id": edge_id, **row},
    )
    monkeypatch.setattr(
        graph_bundle_publisher.graph_compiler_v3,
        "compile_persona_publication",
        lambda *_args, **_kwargs: {
            "publication": {
                "id": "publication-1",
                "version": 1,
                "checksum": plan["runtime_checksum"],
                "status": "compiled",
            },
            "activation": None,
            "cached": False,
        },
    )
    monkeypatch.setattr(
        graph_bundle_publisher.supabase_client, "insert_event", lambda *_a, **_k: None
    )

    staged = graph_bundle_publisher.stage_bundle(
        bundle,
        approved_draft_checksum=plan["draft_checksum"],
        actor="test",
        embedder=lambda texts: [[0.0] * 1536 for _ in texts],
    )

    assert staged["publication"]["checksum"] == plan["runtime_checksum"]
    assert staged["activation"] is None
    assert materialized_nodes
    assert len(replaced_nodes) == len(normalized["nodes"])
    assert len(replaced_edges) == len(normalized["edges"])
    assert all("source_id" not in row for row in materialized_nodes)
    assert all(row["metadata"].get("graph_json_node_id") for row in materialized_nodes)
    assert all(set(row["metadata"]) >= {"active", "graph_json_edge_id"} for row in replaced_edges)


def test_stage_bundle_rejects_stale_human_approval_before_writes(monkeypatch):
    writes: list[dict] = []
    monkeypatch.setattr(
        graph_bundle_publisher.supabase_client,
        "upsert_knowledge_node",
        lambda row: writes.append(row),
    )

    try:
        graph_bundle_publisher.stage_bundle(
            _bundle(), approved_draft_checksum="sha256:stale", actor="test"
        )
        raise AssertionError("expected checksum rejection")
    except graph_bundle_publisher.GraphBundlePublishError as exc:
        assert "approved_draft_checksum_mismatch" in str(exc)

    assert writes == []


class _PublicationQuery:
    def __init__(self, publication: dict):
        self.publication = publication

    def select(self, *_args):
        return self

    def eq(self, *_args):
        return self

    def maybe_single(self):
        return self

    def execute(self):
        return SimpleNamespace(data=self.publication)


class _PublicationClient:
    def __init__(self, publication: dict):
        self.publication = publication
        self.rpc_calls: list[tuple[str, dict]] = []

    def table(self, _name):
        return _PublicationQuery(self.publication)

    def rpc(self, name, params):
        self.rpc_calls.append((name, params))
        return SimpleNamespace(execute=lambda: SimpleNamespace(data={"ok": True}))


def test_activate_staged_bundle_rejects_publication_of_another_persona(monkeypatch):
    bundle = _bundle()
    plan = graph_bundle.build_publication_plan(bundle)
    normalized = graph_bundle.normalize_bundle(bundle)
    foreign_publication = {
        "id": "publication-foreign",
        "persona_id": "persona-other",
        "status": "compiled",
        "checksum": plan["runtime_checksum"],
        "version": 7,
    }
    client = _PublicationClient(foreign_publication)
    monkeypatch.setattr(
        graph_bundle_publisher.supabase_client,
        "get_persona",
        lambda _slug: {"id": normalized["persona"]["id"]},
    )
    monkeypatch.setattr(graph_bundle_publisher.supabase_client, "get_client", lambda: client)

    with pytest.raises(graph_bundle_publisher.GraphBundlePublishError, match="persona_scope_mismatch"):
        graph_bundle_publisher.activate_staged_bundle(
            bundle,
            publication_id="publication-foreign",
            approved_draft_checksum=plan["draft_checksum"],
            approved_runtime_checksum=plan["runtime_checksum"],
            actor="test",
        )

    assert client.rpc_calls == []

