from __future__ import annotations

import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from services.graph_bundle_error_translations import translate_error, translate_errors


def test_translates_primary_parent_missing_with_suggested_question():
    result = translate_error("bundle_primary_parent_missing:product:vestido-poa")
    assert "vestido-poa" in result["message"]
    assert result["suggested_question"] is not None
    assert "vestido-poa" in result["suggested_question"]


def test_translates_no_branch_anchor():
    result = translate_error("publication_has_no_branch_anchor_capability")
    assert "ramo de qualifica" in result["message"]
    assert result["suggested_question"]


def test_translates_node_not_publishable_with_status():
    result = translate_error("bundle_node_not_publishable:faq:foo:pending_validation")
    assert "faq:foo" in result["message"]
    assert "pending_validation" in result["message"]


def test_unrecognized_code_still_returns_usable_dict():
    result = translate_error("some_future_error_code:detail")
    assert "some_future_error_code" in result["message"]
    assert result["code"] == "some_future_error_code:detail"


def test_translate_errors_preserves_order():
    raw = [
        "bundle_primary_parent_missing:brand:x",
        "publication_has_no_branch_anchor_capability",
    ]
    results = translate_errors(raw)
    assert len(results) == 2
    assert results[0]["code"] == raw[0]
    assert results[1]["code"] == raw[1]

