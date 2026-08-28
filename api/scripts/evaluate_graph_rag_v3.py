"""Evaluate the active GraphRAG publication through the production SQL RPC.

The dataset contains only published node identifiers and expected structural
outcomes. It is deliberately independent from customer copy and field names in
the backend runtime.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any


API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_ROOT))

from services import graph_agent_runtime_v3, graph_compiler_v3, supabase_client


DEFAULT_DATASET = API_ROOT / "evaluation" / "graph_rag_v3_retrieval.json"


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _question_for_missing(contract: dict[str, Any], missing: list[str]) -> str | None:
    by_key = {str(field.get("key")): field for field in contract.get("fields") or []}
    for key in missing:
        field = by_key.get(str(key))
        if field:
            return str(field.get("question_node_id") or "") or None
    return None


def _claim_is_grounded(
    contract: dict[str, Any], claim_type: str | None, returned_node_ids: set[str]
) -> bool:
    if not claim_type:
        return True
    for claim in contract.get("claims") or []:
        if str(claim.get("claim_type")) != claim_type:
            continue
        evidence = claim.get("evidence_node_ids") or []
        if claim.get("policy") and evidence and returned_node_ids.intersection(
            str(value) for value in evidence
        ):
            return True
    return False


def _claim_evidence_nodes(contract: dict[str, Any], claim_type: str | None) -> list[str]:
    if not claim_type:
        return []
    return list(dict.fromkeys(
        str(node_id)
        for claim in contract.get("claims") or []
        if str(claim.get("claim_type")) == claim_type and claim.get("policy")
        for node_id in claim.get("evidence_node_ids") or []
    ))


def evaluate(dataset: dict[str, Any]) -> dict[str, Any]:
    persona = supabase_client.get_persona(str(dataset["persona_slug"]))
    if not persona:
        raise RuntimeError(f"persona not found: {dataset['persona_slug']}")
    publication = supabase_client.get_active_graph_publication(str(persona["id"]))
    if not publication:
        raise RuntimeError("active GraphRAG v3 publication not found")
    document = publication.get("document_json") or {}
    cases = dataset.get("cases") or []
    vectors = graph_compiler_v3.query_embeddings([str(case["query"]) for case in cases])
    results: list[dict[str, Any]] = []
    latencies: list[float] = []
    recalled = 0
    expected_total = 0
    contamination_count = 0
    returned_total = 0
    question_correct = 0
    claim_correct = 0
    claim_total = 0
    max_context_tokens = 0

    for case, vector in zip(cases, vectors, strict=True):
        branch = str(case["branch_node_id"])
        contract = (document.get("branch_contracts") or {}).get(branch) or {}
        path = ((document.get("coordinates") or {}).get(branch) or {}).get("path_node_ids") or []
        started = time.perf_counter()
        rows = supabase_client.search_graph_rag_v3(
            persona_id=str(persona["id"]), publication_id=str(publication["id"]),
            branch_node_id=branch, query=str(case["query"]), query_embedding=vector,
            active_path_node_ids=path, missing_fields=case.get("missing_fields") or [],
            limit=max(40, int(dataset.get("top_k") or 10) * 4),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)
        top_rows = graph_agent_runtime_v3._mmr(
            rows, int(dataset.get("top_k") or 10)
        )
        returned = [str(row.get("source_node_id") or row.get("source_graph_node_id") or "") for row in top_rows]
        expected = {str(value) for value in case.get("expected_node_ids") or []}
        recalled_here = len(expected.intersection(returned))
        recalled += recalled_here
        expected_total += len(expected)
        allowed = set((document.get("branch_memberships") or {}).get(branch) or {})
        forbidden = {str(value) for value in case.get("forbidden_node_ids") or []}
        contaminated = [node for node in returned if node not in allowed or node in forbidden]
        contamination_count += len(contaminated)
        returned_total += len(returned)
        actual_question = _question_for_missing(contract, case.get("missing_fields") or [])
        expected_question = case.get("expected_next_question_node_id")
        question_ok = actual_question == expected_question
        question_correct += int(question_ok)
        claim_type = str(case.get("claim_type") or "") or None
        directed_claim_nodes = _claim_evidence_nodes(contract, claim_type)
        directed_claim_chunks = supabase_client.get_graph_rag_repair_chunks(
            publication_id=str(publication["id"]), branch_node_id=branch,
            requirements=[{"kind": "node", "id": node_id} for node_id in directed_claim_nodes],
        ) if directed_claim_nodes else []
        effective_claim_nodes = set(returned) | {
            str(row.get("source_graph_node_id") or row.get("source_node_id") or "")
            for row in directed_claim_chunks
        }
        claim_authorized_and_retrieved = _claim_is_grounded(
            contract, claim_type, effective_claim_nodes
        )
        expected_claim_authorized = bool(case.get("expected_claim_authorized", True))
        claim_ok = claim_authorized_and_retrieved == expected_claim_authorized
        if claim_type:
            claim_total += 1
            claim_correct += int(claim_ok)
        context_tokens = math.ceil(sum(len(str(row.get("chunk_text") or "")) for row in top_rows) / 4)
        max_context_tokens = max(max_context_tokens, context_tokens)
        results.append({
            "id": case["id"], "latency_ms": round(elapsed_ms, 2),
            "returned_node_ids": returned, "recalled": recalled_here,
            "expected": len(expected), "contaminated_node_ids": contaminated,
            "next_question_ok": question_ok, "claim_grounded": claim_ok,
            "claim_authorized_and_retrieved": claim_authorized_and_retrieved,
            "directed_claim_evidence_node_ids": sorted(
                node for node in effective_claim_nodes - set(returned) if node
            ),
            "context_tokens_estimate": context_tokens,
        })

    metrics = {
        "recall_at_10": recalled / max(1, expected_total),
        "branch_contamination": contamination_count / max(1, returned_total),
        "next_question_accuracy": question_correct / max(1, len(cases)),
        "grounded_commercial_claims": claim_correct / max(1, claim_total),
        "retrieval_p50_ms": round(statistics.median(latencies), 2) if latencies else 0,
        "retrieval_p95_ms": round(_percentile(latencies, 0.95), 2),
        "max_context_tokens_estimate": max_context_tokens,
        "configured_context_token_budget": int(dataset.get("max_context_tokens") or 0),
    }
    thresholds = dataset.get("thresholds") or {}
    failures = [
        f"{name}:{metrics[name]}<{minimum}"
        for name, minimum in thresholds.items() if float(metrics.get(name, 0)) < float(minimum)
    ]
    if max_context_tokens > int(dataset.get("max_context_tokens") or 0):
        failures.append("context_token_budget_exceeded")
    return {
        "persona_slug": persona["slug"], "publication_id": publication["id"],
        "publication_version": publication["version"], "graph_checksum": publication["checksum"],
        "compiler_version": publication["compiler_version"], "dataset_schema": dataset["schema_version"],
        "metrics": metrics, "passed": not failures, "failures": failures, "cases": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(json.loads(args.dataset.read_text(encoding="utf-8")))
    encoded = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    raise SystemExit(0 if report["passed"] else 1)
