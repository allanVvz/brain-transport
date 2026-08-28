# -*- coding: utf-8 -*-
"""
Audit helpers â€” utilities for instrumenting mutation handlers with consistent
event payloads. Used by routes/assets.py, routes/graph.py and routes/knowledge.py
to emit `system_events` rows with a stable shape: actor + before/after + diff +
context.

Design:
- summarize_diff is shallow by design. Nested dicts are compared as serialized
  JSON; if you need a deeper trail, pass the projected leaf keys explicitly.
- current_actor never raises: a missing session yields {"user_id": None}, so
  background callers (workers, scripts) can also call it safely.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Optional


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except Exception:
            return str(value)
    if isinstance(value, (list, tuple, set)):
        try:
            return json.dumps(list(value), sort_keys=False, default=str)
        except Exception:
            return str(value)
    return value


def summarize_diff(
    before: Optional[dict],
    after: Optional[dict],
    keys: Optional[Iterable[str]] = None,
) -> dict:
    """Return a compact diff of (before, after) for the supplied keys.

    Output: {"changed": {<key>: {"before": x, "after": y}}, "unchanged_count": n}.
    Keys whose values are equal (after normalize) are omitted from `changed`.
    Nested dicts/lists are compared as serialized JSON â€” shallow compare.
    """
    before = before or {}
    after = after or {}
    if keys is None:
        key_set = set(before.keys()) | set(after.keys())
    else:
        key_set = set(keys)
    changed: dict[str, dict[str, Any]] = {}
    unchanged = 0
    for k in key_set:
        b = before.get(k)
        a = after.get(k)
        if _normalize(b) == _normalize(a):
            unchanged += 1
            continue
        changed[k] = {"before": b, "after": a}
    return {"changed": changed, "unchanged_count": unchanged}


def current_actor(request) -> dict:
    """Best-effort extraction of the calling user. Never raises.

    Returns {"user_id", "email", "role"}. Missing session yields a dict with
    all values None so callers can serialize it uniformly into payloads.
    """
    try:
        user = getattr(getattr(request, "state", None), "user", None)
        if not user:
            return {"user_id": None, "email": None, "role": None}
        return {
            "user_id": user.get("id"),
            "email": user.get("email"),
            "role": user.get("role"),
        }
    except Exception:
        return {"user_id": None, "email": None, "role": None}

