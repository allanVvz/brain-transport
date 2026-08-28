"""Publish missing canonical Graph JSON v2 documents for existing personas."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services import graph_json_v2_backfill, supabase_client  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persona", action="append", default=[], help="Persona slug; repeat for multiple.")
    parser.add_argument("--all", action="store_true", help="Process every active persona.")
    parser.add_argument("--force", action="store_true", help="Publish a new version even when one exists.")
    parser.add_argument("--dry-run", action="store_true", help="Build and validate without writing.")
    parser.add_argument("--no-materialize", action="store_true", help="Do not update derived graph rows.")
    args = parser.parse_args()

    slugs = [str(value).strip().lower() for value in args.persona if str(value).strip()]
    if args.all:
        slugs.extend(
            str(row.get("slug") or "").strip().lower()
            for row in (supabase_client.get_personas() or [])
            if row.get("slug") and row.get("active", True)
        )
    slugs = list(dict.fromkeys(slugs))
    if not slugs:
        raise SystemExit("Use --all or at least one --persona.")

    results: list[dict] = []
    for slug in slugs:
        if args.dry_run:
            _graph, report = graph_json_v2_backfill.build_from_derived_graph(slug)
            results.append({**report, "ok": report["valid"], "dry_run": True})
        else:
            results.append(
                graph_json_v2_backfill.publish_backfill(
                    slug,
                    force=args.force,
                    materialize=not args.no_materialize,
                )
            )

    print(json.dumps(results, ensure_ascii=False, indent=2))
    if any(result.get("ok") is not True for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

