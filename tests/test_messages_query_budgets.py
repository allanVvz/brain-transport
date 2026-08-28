from __future__ import annotations

import sys
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1] / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from routes import messages


def _rows(start: int, count: int) -> list[dict]:
    return [
        {"id": value, "created_at": f"2026-01-01T00:{value:02d}:00+00:00"}
        for value in range(start, start + count)
    ]


def test_message_page_supports_initial_before_and_after_cursors(monkeypatch):
    calls: list[dict] = []

    def page(_lead_ref, **kwargs):
        calls.append(kwargs)
        if kwargs.get("after_id"):
            return _rows(51, 51)
        if kwargs.get("before_id"):
            return _rows(0, 51)
        return _rows(0, 51)

    monkeypatch.setattr(messages.supabase_client, "get_messages_page", page)
    initial = messages._message_page(1, limit=50, after=None, before=None)
    before = messages._message_page(
        1, limit=50, after=None,
        before=messages._encode_message_cursor(initial["items"][0]),
    )
    after = messages._message_page(
        1, limit=50, before=None,
        after=messages._encode_message_cursor(initial["items"][-1]),
    )

    assert [row["id"] for row in initial["items"]] == list(range(1, 51))
    assert [row["id"] for row in before["items"]] == list(range(1, 51))
    assert [row["id"] for row in after["items"]] == list(range(51, 101))
    assert calls[1]["before_id"] == 1
    assert calls[2]["after_id"] == 50


def test_message_page_rejects_mixed_direction_cursors():
    cursor = messages._encode_message_cursor(
        {"id": 1, "created_at": "2026-01-01T00:00:00+00:00"}
    )
    import pytest
    with pytest.raises(Exception, match="apenas um cursor"):
        messages._message_page(1, limit=50, before=cursor, after=cursor)


def test_conversation_decoration_bulk_loads_leads(monkeypatch):
    calls: list[list[int]] = []

    def bulk(refs):
        calls.append(list(refs))
        return {
            value: {
                "id": value,
                "lead_id": f"lead-{value}",
                "stage": "novo",
                "metadata": {},
            }
            for value in refs
        }

    monkeypatch.setattr(messages.supabase_client, "get_leads_by_refs", bulk)
    monkeypatch.setattr(
        messages.runtime_client,
        "decorate_leads",
        lambda leads: [
            {**lead, "qualification": {}, "qualification_score": 0,
             "qualification_signals": [], "validation": {}}
            for lead in leads
        ],
    )
    monkeypatch.setattr(
        messages.supabase_client,
        "get_lead_by_ref",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("conversation decorator performed an N+1 lookup")
        ),
    )

    result = messages._decorate_conversations([
        {"lead_ref": 1, "last_message": "one"},
        {"lead_ref": 2, "last_message": "two"},
    ])

    assert calls == [[1, 2]]
    assert len(result) == 2
