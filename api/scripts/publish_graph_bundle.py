"""Stage and optionally activate one approved GraphBundle in production."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


API_DIR = Path(__file__).resolve().parents[1]
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from services import graph_bundle_publisher  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    parser.add_argument("--approved-draft-checksum", required=True)
    parser.add_argument("--approved-runtime-checksum", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    payload = json.loads(Path(args.bundle).read_text(encoding="utf-8"))
    if not args.apply:
        raise SystemExit("Refusing mutation without --apply")
    staged = graph_bundle_publisher.stage_bundle(
        payload,
        approved_draft_checksum=args.approved_draft_checksum,
        actor=args.actor,
    )
    publication = staged.get("publication") or {}
    result: dict = {
        "staged": True,
        "activated": False,
        "publication_id": publication.get("id"),
        "version": publication.get("version"),
        "draft_checksum": staged["plan"]["draft_checksum"],
        "runtime_checksum": publication.get("checksum"),
        "cached": staged.get("cached"),
    }
    if staged["plan"]["runtime_checksum"] != args.approved_runtime_checksum:
        raise graph_bundle_publisher.GraphBundlePublishError(
            "approved_runtime_checksum_mismatch"
        )
    if args.activate:
        activation = graph_bundle_publisher.activate_staged_bundle(
            payload,
            publication_id=str(publication["id"]),
            approved_draft_checksum=args.approved_draft_checksum,
            approved_runtime_checksum=args.approved_runtime_checksum,
            actor=args.actor,
        )
        result["activated"] = True
        result["activation"] = activation.get("activation")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
