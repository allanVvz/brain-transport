"""Compile a local GraphBundle and print a dry-run PublicationPlan."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


API_DIR = Path(__file__).resolve().parents[1]
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from services.graph_bundle import build_publication_plan  # noqa: E402


def _json_file(value: str) -> dict:
    with Path(value).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {value}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile a GraphBundle without publishing or writing production state."
    )
    parser.add_argument("bundle", help="Path to the GraphBundle JSON file")
    parser.add_argument("--against", help="Optional compiled v3 document to diff against")
    parser.add_argument("--next-version", type=int, default=1)
    parser.add_argument(
        "--include-document",
        action="store_true",
        help="Include the complete compiled candidate document in stdout",
    )
    args = parser.parse_args()

    try:
        plan = build_publication_plan(
            _json_file(args.bundle),
            current_document=_json_file(args.against) if args.against else None,
            next_version=args.next_version,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({
            "disposition": "blocked",
            "validation_errors": [f"input_error:{type(exc).__name__}:{exc}"],
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 1
    printable = dict(plan)
    if not args.include_document:
        printable.pop("candidate_document", None)
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not plan["validation_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

