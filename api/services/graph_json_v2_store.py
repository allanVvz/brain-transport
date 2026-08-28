"""Persistent Graph JSON v2 version store backed by ``system_events``.

Graph JSON v2 is the canonical document consumed by the Graph UI.  Production
containers are disposable, so filesystem versions cannot be the source of
truth.  This store uses the existing audit/event table (no schema expansion)
and keeps the old ``storage_root`` helper only for locating legacy files during
explicit migrations.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from services import supabase_client

if TYPE_CHECKING:  # pragma: no cover
    from api.schemas.graph_json_v2 import GraphJson  # type: ignore
else:
    try:
        from schemas.graph_json_v2 import GraphJson
    except ModuleNotFoundError:  # pragma: no cover
        from api.schemas.graph_json_v2 import GraphJson


_LEGACY_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "graph_documents"
_EVENT_TYPE = "graph_document_published"
_COMMITTED_EVENT_TYPE = "graph_version_committed"
_ACTIVATED_EVENT_TYPE = "graph_version_activated"
_ENTITY_TYPE = "graph_document"


def _checksum(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def canonical_content_payload(graph: "GraphJson") -> dict[str, Any]:
    """Return stable semantic content; UI layout and revision envelope do not affect it."""
    payload = graph.model_dump(mode="json")
    for key in ("layout", "validation", "graph_version", "content_checksum", "status", "provenance"):
        payload.pop(key, None)
    for node in payload.get("nodes") or []:
        # Rendered Markdown is derived from the structured graph and must not
        # recursively participate in its own checksum.
        node.pop("markdown", None)
        data = node.get("data") or {}
        for key in ("markdown", "markdown_checksum", "markdown_renderer"):
            data.pop(key, None)
    return payload


def checksum_graph(graph: "GraphJson") -> str:
    return _checksum(canonical_content_payload(graph))


def _events(
    persona_slug: str,
    brand_slug: str | None = None,
    *,
    limit: int = 500,
    event_types: list[str] | None = None,
) -> list[dict]:
    try:
        rows = supabase_client.list_system_events(
            entity_type=_ENTITY_TYPE,
            event_types=event_types or [_ACTIVATED_EVENT_TYPE, _EVENT_TYPE],
            payload_equals={
                "persona_slug": persona_slug,
                **({"brand_slug": brand_slug} if brand_slug is not None else {}),
            },
            limit=limit,
        )
    except KeyError as exc:
        # Pure plan/preview tests and offline tooling deliberately run without
        # Supabase environment variables. In that context there simply is no
        # published document; production still propagates every database error.
        if str(exc).strip("'") not in {"SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"}:
            raise
        return []
    matching: list[dict] = []
    for row in rows:
        payload = row.get("payload") or {}
        if payload.get("persona_slug") != persona_slug:
            continue
        # The dashboard loads one canonical graph per persona.  When no brand
        # filter is requested, return the latest document regardless of its
        # optional brand scope.  An explicit brand remains exact-match.
        if brand_slug is not None and (payload.get("brand_slug") or None) != brand_slug:
            continue
        if not isinstance(payload.get("graph_json"), dict):
            continue
        matching.append(row)
    matching.sort(
        key=lambda row: (
            int(((row.get("payload") or {}).get("version") or 0)),
            str(row.get("created_at") or ""),
        ),
        reverse=True,
    )
    return matching


def latest_event(persona_slug: str, brand_slug: str | None = None) -> dict | None:
    # Each version emits only a small set of commit/activation events. Query a
    # narrow persona-scoped window instead of deserializing hundreds of full
    # Graph JSON payloads from unrelated personas.
    rows = _events(persona_slug, brand_slug, limit=20)
    return rows[0] if rows else None


def list_versions(persona_slug: str, brand_slug: str | None = None) -> list[int]:
    versions = {
        int(((row.get("payload") or {}).get("version") or 0))
        for row in _events(
            persona_slug,
            brand_slug,
            limit=500,
            event_types=[_COMMITTED_EVENT_TYPE, _ACTIVATED_EVENT_TYPE, _EVENT_TYPE],
        )
    }
    return sorted(version for version in versions if version > 0)


def load_version(
    persona_slug: str,
    version: int,
    brand_slug: str | None = None,
) -> "GraphJson | None":
    for row in _events(
        persona_slug,
        brand_slug,
        limit=500,
        event_types=[_COMMITTED_EVENT_TYPE, _ACTIVATED_EVENT_TYPE, _EVENT_TYPE],
    ):
        payload = row.get("payload") or {}
        if int(payload.get("version") or 0) != int(version):
            continue
        try:
            return GraphJson.model_validate(payload["graph_json"])
        except Exception:
            return None
    return None


def load_activated_version(
    persona_slug: str,
    version: int,
    brand_slug: str | None = None,
) -> "GraphJson | None":
    for row in _events(
        persona_slug,
        brand_slug,
        limit=500,
        event_types=[_ACTIVATED_EVENT_TYPE, _EVENT_TYPE],
    ):
        payload = row.get("payload") or {}
        if int(payload.get("version") or 0) != int(version):
            continue
        try:
            graph = GraphJson.model_validate(payload["graph_json"])
            # Early v2.1 activation events copied the committed envelope
            # verbatim. Activation is authoritative, so expose the correct
            # runtime state without mutating the immutable historical event.
            graph.status = "published"
            return graph
        except Exception:
            return None
    return None


def commit_version(
    *,
    persona_slug: str,
    brand_slug: str | None,
    expected_version: int,
    idempotency_key: str,
    reason: str,
    graph: "GraphJson",
    source: str,
    authored_by: str | None,
) -> dict[str, Any]:
    """Atomically allocate and append an immutable version through Postgres."""
    checksum = checksum_graph(graph)
    payload = graph.model_dump(mode="json")
    result = supabase_client.commit_graph_version_v2(
        persona_slug=persona_slug,
        brand_slug=brand_slug,
        expected_version=expected_version,
        idempotency_key=idempotency_key,
        reason=reason,
        graph_json=payload,
        content_checksum=checksum,
        source=source,
        authored_by=authored_by,
    )
    if not result or not result.get("graph_version"):
        raise RuntimeError("commit_graph_version_v2 returned an invalid result")
    return result


def activate_version(
    *,
    persona_slug: str,
    brand_slug: str | None,
    version: int,
    checksum: str,
    operation_id: str,
    projections: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    return supabase_client.activate_graph_projection_v2(
        persona_slug=persona_slug,
        brand_slug=brand_slug,
        graph_version=version,
        graph_checksum=checksum,
        operation_id=operation_id,
        projections=projections,
        source=source,
    )


def load_current(
    persona_slug: str,
    brand_slug: str | None = None,
) -> "tuple[int, GraphJson] | None":
    row = latest_event(persona_slug, brand_slug)
    if not row:
        return None
    payload = row.get("payload") or {}
    try:
        version = int(payload.get("version") or 0)
        graph = GraphJson.model_validate(payload["graph_json"])
        if row.get("event_type") == _ACTIVATED_EVENT_TYPE:
            graph.status = "published"
    except Exception:
        return None
    if version < 1:
        return None
    return version, graph


def save_version(
    persona_slug: str,
    version: int,
    graph: "GraphJson",
    *,
    brand_slug: str | None = None,
    source: str = "graph_json_v2_store",
    note: str | None = None,
    published_by: str | None = None,
    idempotency_key: str | None = None,
    projections: dict[str, Any] | None = None,
) -> str:
    graph_dict = graph.model_dump()
    checksum = _checksum(graph_dict)
    persona = supabase_client.get_persona(persona_slug) or {}
    document_brand = brand_slug if brand_slug is not None else graph.brand_slug
    doc_id = f"{persona_slug}:{document_brand or 'default'}:v{int(version)}"
    payload: dict[str, Any] = {
        "persona_slug": persona_slug,
        "brand_slug": document_brand,
        "version": int(version),
        "checksum": checksum,
        "graph_json": graph_dict,
        "source": source,
        "note": note,
        "published_by": published_by,
        "idempotency_key": idempotency_key,
        "projections": projections or {},
        "published_at": datetime.now(timezone.utc).isoformat(),
    }
    event = supabase_client.insert_event(
        {
            "event_type": _EVENT_TYPE,
            "entity_type": _ENTITY_TYPE,
            "entity_id": doc_id,
            "persona_id": persona.get("id"),
            "payload": payload,
            "level": "info",
            "source": source,
        },
        source=source,
    )
    if not event:
        raise RuntimeError("Failed to persist Graph JSON v2 in system_events")
    return checksum


def storage_root() -> Path:
    """Return the legacy file location for explicit migration tooling only."""
    return _LEGACY_DATA_ROOT
