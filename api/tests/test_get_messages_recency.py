"""Regression test for the 2026-08-02 stale-history bug in get_messages().

get_messages(limit=N) used to query messages.order(created_at, id, desc=False)
then .limit(N) â€” for any lead with more than N total messages, that returns
the OLDEST N messages, not the most recent N, even though every caller
(AI context building, the WhatsApp bot-echo-loop guard, chat history APIs)
expects "recent". Confirmed live: a 143-message test lead had its echo-loop
guard permanently matching a 6-day-old outbound reply because that old row
never left the first-20-messages window, silently re-pausing AI on every
resume attempt.
"""
import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from services import supabase_client


class _FakeResult:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows
        self.order_calls: list[tuple[str, bool]] = []

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def order(self, column, desc=False):
        self.order_calls.append((column, desc))
        return self

    def limit(self, _n):
        return self

    def execute(self):
        return _FakeResult(self._rows)


class _FakeTable:
    def __init__(self, rows):
        self._rows = rows
        self.last_query: _FakeQuery | None = None

    def table(self, _name):
        self.last_query = _FakeQuery(self._rows)
        return self.last_query


def test_get_messages_queries_descending_so_limit_keeps_the_newest_rows(monkeypatch):
    # The fake DB returns rows already in the (correct) newest-first order a
    # real descending query + limit would produce, mirroring only the two
    # most recent of a much longer real history.
    rows = [
        {"id": 300, "lead_id": 5, "created_at": "2026-08-02T07:03:22+00:00", "content": "oi", "direction": "inbound", "role": "user"},
        {"id": 299, "lead_id": 5, "created_at": "2026-08-02T07:02:15+00:00", "content": "ola", "direction": "inbound", "role": "user"},
    ]
    fake_client = _FakeTable(rows)
    monkeypatch.setattr(supabase_client, "get_client", lambda: fake_client)

    result = supabase_client.get_messages("5", limit=2)

    assert fake_client.last_query is not None
    assert ("created_at", True) in fake_client.last_query.order_calls
    assert ("id", True) in fake_client.last_query.order_calls
    # _sort_messages_for_chat must still return chronological (ascending) order.
    assert [row["id"] for row in result] == [299, 300]

