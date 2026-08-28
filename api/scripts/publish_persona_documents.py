"""Compile and publish one persona's canonical Markdown documents."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


API_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = API_DIR.parent
for path in (API_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from services import graph_document_publisher, graph_json_v2_store
from services.sdr_documents import compile_persona_documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("persona_slug")
    parser.add_argument(
        "--documents-root",
        type=Path,
        default=ROOT_DIR / "docs" / "sdr",
    )
    parser.add_argument("--tenant", default="qa")
    parser.add_argument("--published-by", default="documents-cli")
    parser.add_argument("--expected-checksum")
    parser.add_argument("--expected-version", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    graph = compile_persona_documents(
        args.documents_root,
        args.persona_slug,
        tenant=args.tenant,
    )
    encoded = json.dumps(
        graph.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    checksum = "sha256:" + hashlib.sha256(encoded).hexdigest()
    if args.expected_checksum and checksum != args.expected_checksum:
        raise SystemExit(
            f"compiled checksum {checksum} does not match approved {args.expected_checksum}"
        )
    current = graph_json_v2_store.load_current(args.persona_slug)
    next_version = int(current[0]) + 1 if current else 1
    if args.expected_version is not None and next_version != args.expected_version:
        raise SystemExit(
            f"next version {next_version} does not match approved {args.expected_version}"
        )
    summary = {
        "persona_slug": args.persona_slug,
        "checksum": checksum,
        "documents_root": str(args.documents_root),
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "chunks": sum(
            1 for node in graph.nodes if (node.data or {}).get("rag_chunk")
        ),
    }
    if args.dry_run:
        print(json.dumps({**summary, "ok": True, "dry_run": True}, ensure_ascii=False))
        return

    result = graph_document_publisher.publish(
        graph=graph,
        persona_slug=args.persona_slug,
        brand_slug=graph.brand_slug,
        source="scripts.publish_persona_documents",
        note="Canonical Markdown publication",
        published_by=args.published_by,
        idempotency_key=f"markdown:{args.persona_slug}:{checksum}",
    )
    if args.expected_version is not None and int(result.get("version") or 0) != args.expected_version:
        raise RuntimeError("publisher returned an unexpected version")
    print(json.dumps({**summary, **result}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

