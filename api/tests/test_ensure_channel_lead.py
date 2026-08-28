"""Regression test for the 2026-08-01 Baita lead-duplication bug.

ensure_channel_lead() only looked up leads by external_contact_id, so any
lead created through a path that only set `telefone` (e.g. the legacy
/process route) was invisible to every webhook-driven inbound message â€”
guaranteeing a permanent duplicate lead for that contact. Confirmed live
on a real Baita customer ("Allan"): two leads, messages split across them.
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
    """Chainable stand-in for the Supabase fluent query builder."""

    def __init__(self, rows: list[dict], filters: dict | None = None):
        self._rows = rows
        self._filters = dict(filters or {})

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, field, value):
        return _FakeQuery(self._rows, {**self._filters, field: ("eq", value)})

    def is_(self, field, value):
        expected = None if value in {"null", None} else value
        return _FakeQuery(self._rows, {**self._filters, field: ("is", expected)})

    def maybe_single(self):
        return self

    def insert(self, payload):
        return _FakeInsert(self._rows, payload)

    def update(self, payload):
        return _FakeUpdate(self._rows, payload, self._filters)

    def _matches(self, row: dict) -> bool:
        for field, (kind, expected) in self._filters.items():
            if row.get(field) != expected:
                return False
        return True

    def execute(self):
        matches = [row for row in self._rows if self._matches(row)]
        return _FakeResult(matches[0] if matches else None)


class _FakeInsert:
    def __init__(self, rows, payload):
        self._rows = rows
        self._payload = payload

    def execute(self):
        row = {"id": 999, **self._payload}
        self._rows.append(row)
        return _FakeResult([row])


class _FakeUpdate:
    def __init__(self, rows, payload, filters):
        self._rows = rows
        self._payload = payload
        self._filters = filters

    def eq(self, field, value):
        self._filters = {**self._filters, field: ("eq", value)}
        return self

    def execute(self):
        for row in self._rows:
            if all(row.get(f) == v for f, (_kind, v) in self._filters.items()):
                row.update(self._payload)
        return _FakeResult(None)


class _FakeTable:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def select(self, *_args, **_kwargs):
        return _FakeQuery(self._rows)

    def insert(self, payload):
        return _FakeInsert(self._rows, payload)

    def update(self, payload):
        return _FakeUpdate(self._rows, payload, {})


class _FakeClient:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def table(self, name):
        assert name == "leads"
        return _FakeTable(self._rows)


def test_finds_lead_by_telefone_when_external_contact_id_never_matched(monkeypatch):
    rows = [
        {
            "id": 5, "persona_id": "p1", "nome": "Allan",
            "telefone": "555182608510", "external_contact_id": None,
            "channel_binding_id": None,
        }
    ]
    monkeypatch.setattr(supabase_client, "get_client", lambda: _FakeClient(rows))

    lead = supabase_client.ensure_channel_lead(
        persona_id="p1",
        channel_binding_id="binding-1",
        external_contact_id="555182608510",
    )

    assert lead["id"] == 5
    assert lead["external_contact_id"] == "555182608510"
    assert lead["channel_binding_id"] == "binding-1"
    # Backfilled in place â€” no duplicate row created.
    assert len(rows) == 1
    assert rows[0]["external_contact_id"] == "555182608510"


def test_creates_new_lead_when_no_existing_contact_matches(monkeypatch):
    rows: list[dict] = []
    monkeypatch.setattr(supabase_client, "get_client", lambda: _FakeClient(rows))

    lead = supabase_client.ensure_channel_lead(
        persona_id="p1",
        channel_binding_id="binding-1",
        external_contact_id="555199999999",
    )

    assert lead["external_contact_id"] == "555199999999"
    assert len(rows) == 1


def test_does_not_match_a_telefone_lead_belonging_to_a_different_persona(monkeypatch):
    rows = [
        {
            "id": 5, "persona_id": "other-persona", "nome": "Allan",
            "telefone": "555182608510", "external_contact_id": None,
            "channel_binding_id": None,
        }
    ]
    monkeypatch.setattr(supabase_client, "get_client", lambda: _FakeClient(rows))

    lead = supabase_client.ensure_channel_lead(
        persona_id="p1",
        channel_binding_id="binding-1",
        external_contact_id="555182608510",
    )

    assert lead["id"] != 5
    assert len(rows) == 2

