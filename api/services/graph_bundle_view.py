"""Read-only catalog and view projection for GraphBundle v3.

This module intentionally has no publication, activation, or persistence path.
It combines the versioned bundles shipped with the repository, Sofia's pending
bundle drafts, and the already materialized ``graph_publications`` rows.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from services import graph_bundle, supabase_client


REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_ROOT = REPO_ROOT / "data" / "graph_bundles"
_SAFE_PERSONA = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$")


class GraphBundleViewNotFound(LookupError):
    pass


def _assert_safe_persona_slug(persona_slug: str) -> None:
    if not _SAFE_PERSONA.fullmatch(persona_slug or ""):
        raise GraphBundleViewNotFound("graph_bundle_persona_not_found")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _iso_mtime(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        return None


def _bundle_persona_slug(bundle: dict[str, Any]) -> str:
    return str((bundle.get("persona") or {}).get("slug") or "")


def _plan_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    plan = graph_bundle.build_publication_plan(bundle)
    return {
        "state": "blocked" if plan.get("validation_errors") else "draft",
        "draft_checksum": plan.get("draft_checksum"),
        "runtime_checksum": plan.get("runtime_checksum"),
        "disposition": plan.get("disposition"),
        "validation_errors": list(plan.get("validation_errors") or []),
        "candidate_document": plan.get("candidate_document"),
    }


def _local_drafts(persona_slug: str) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    persona_dir = BUNDLE_ROOT / persona_slug
    if not persona_dir.is_dir():
        return
    for path in sorted(persona_dir.glob("*.json")):
        if path.name.endswith(".PLAN.json"):
            continue
        bundle = _read_json(path)
        if not bundle or _bundle_persona_slug(bundle) != persona_slug:
            continue
        yield bundle, {
            "ref": f"bundle:{path.name}",
            "origin": "versioned_bundle",
            "label": path.stem,
            "updated_at": _iso_mtime(path),
        }


def _sofia_drafts(persona_slug: str) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    for session in supabase_client.list_kb_intake_sessions(limit=500):
        bundle = (session or {}).get("pending_graph_bundle")
        if not isinstance(bundle, dict) or _bundle_persona_slug(bundle) != persona_slug:
            continue
        session_id = str((session or {}).get("id") or "unknown")
        yield bundle, {
            "ref": f"sofia:{session_id}",
            "origin": "sofia_draft",
            "label": f"Sofia · {session_id[:8]}",
            "updated_at": session.get("_updated_at"),
        }


def _persona_or_not_found(persona_slug: str) -> dict[str, Any]:
    persona = supabase_client.get_persona(persona_slug)
    if not persona:
        raise GraphBundleViewNotFound("graph_bundle_persona_not_found")
    return persona


def _publication_rows(persona_id: str, *, include_document: bool) -> list[dict[str, Any]]:
    fields = "*" if include_document else (
        "id,persona_id,version,checksum,status,compiler_version,created_at,updated_at"
    )
    response = (
        supabase_client.get_client()
        .table("graph_publications")
        .select(fields)
        .eq("persona_id", persona_id)
        .in_("status", ["compiled", "active"])
        .order("version", desc=True)
        .execute()
    )
    return list(response.data or [])


def list_versions(persona_slug: str) -> dict[str, Any]:
    _assert_safe_persona_slug(persona_slug)
    persona = _persona_or_not_found(persona_slug)
    versions: list[dict[str, Any]] = []

    for bundle, identity in [*_local_drafts(persona_slug), *_sofia_drafts(persona_slug)]:
        summary = _plan_summary(bundle)
        versions.append({
            **identity,
            "source": "draft",
            "state": summary["state"],
            "checksum": summary["draft_checksum"],
            "runtime_checksum": summary["runtime_checksum"],
            "disposition": summary["disposition"],
            "validation_error_count": len(summary["validation_errors"]),
            "version": (bundle.get("metadata") or {}).get("version") or bundle.get("version"),
        })

    for row in _publication_rows(str(persona["id"]), include_document=False):
        state = "active" if row.get("status") == "active" else "staged"
        versions.append({
            "ref": f"publication:{row['id']}",
            "source": "publication",
            "origin": "graph_publications",
            "state": state,
            "label": f"Publicação v{row.get('version')} · {state}",
            "version": row.get("version"),
            "checksum": row.get("checksum"),
            "runtime_checksum": row.get("checksum"),
            "compiler_version": row.get("compiler_version"),
            "validation_error_count": 0,
            "updated_at": row.get("updated_at") or row.get("created_at"),
        })

    active = next((item for item in versions if item["state"] == "active"), None)
    staged = next((item for item in versions if item["state"] == "staged"), None)
    draft = next((item for item in versions if item["source"] == "draft"), None)
    selected = active or staged or draft
    return {
        "backend": "v3",
        "persona": {"id": persona.get("id"), "slug": persona_slug, "name": persona.get("name")},
        "versions": versions,
        "default_ref": selected.get("ref") if selected else None,
        "read_only": True,
    }


def _find_draft(persona_slug: str, ref: str) -> tuple[dict[str, Any], dict[str, Any]]:
    sources = _local_drafts(persona_slug) if ref.startswith("bundle:") else _sofia_drafts(persona_slug)
    for bundle, identity in sources:
        if identity["ref"] == ref:
            return bundle, identity
    raise GraphBundleViewNotFound("graph_bundle_draft_not_found")


def _derived_branch_memberships(document: dict[str, Any]) -> dict[str, Any]:
    memberships = document.get("branch_memberships")
    if isinstance(memberships, dict):
        return memberships
    nodes = [node for node in document.get("nodes") or [] if isinstance(node, dict)]
    edges = [edge for edge in document.get("edges") or [] if isinstance(edge, dict)]
    children: dict[str, list[str]] = {}
    for edge in edges:
        if edge.get("relation_type") == "contains":
            children.setdefault(str(edge.get("source") or ""), []).append(str(edge.get("target") or ""))
    result: dict[str, dict[str, Any]] = {}
    for node in nodes:
        data = node.get("data") or {}
        if not isinstance(data, dict) or not (data.get("capabilities") or {}).get("branch_anchor"):
            continue
        anchor = str(node.get("id") or "")
        pending = [anchor]
        seen: set[str] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(children.get(current, []))
        result[anchor] = {node_id: {"node_id": node_id} for node_id in sorted(seen)}
    return result


def get_view(persona_slug: str, *, source: str, ref: str) -> dict[str, Any]:
    _assert_safe_persona_slug(persona_slug)
    persona = _persona_or_not_found(persona_slug)
    if source == "draft":
        if not (ref.startswith("bundle:") or ref.startswith("sofia:")):
            raise GraphBundleViewNotFound("graph_bundle_draft_not_found")
        bundle, identity = _find_draft(persona_slug, ref)
        summary = _plan_summary(bundle)
        document = summary["candidate_document"] or bundle
        return {
            "backend": "v3",
            "persona": {"id": persona.get("id"), "slug": persona_slug, "name": persona.get("name")},
            "source": "draft",
            "ref": ref,
            "origin": identity["origin"],
            "state": summary["state"],
            "version": (bundle.get("metadata") or {}).get("version") or bundle.get("version"),
            "checksum": summary["draft_checksum"],
            "runtime_checksum": summary["runtime_checksum"],
            "disposition": summary["disposition"],
            "validation_errors": summary["validation_errors"],
            "document": document,
            "branch_memberships": _derived_branch_memberships(document),
            "read_only": True,
        }

    if source != "publication" or not ref.startswith("publication:"):
        raise GraphBundleViewNotFound("graph_bundle_publication_not_found")
    publication_id = ref.removeprefix("publication:")
    row = next(
        (row for row in _publication_rows(str(persona["id"]), include_document=True) if str(row.get("id")) == publication_id),
        None,
    )
    if not row:
        raise GraphBundleViewNotFound("graph_bundle_publication_not_found")
    document = row.get("document_json") or {}
    if isinstance(document, str):
        try:
            document = json.loads(document)
        except ValueError:
            document = {}
    state = "active" if row.get("status") == "active" else "staged"
    return {
        "backend": "v3",
        "persona": {"id": persona.get("id"), "slug": persona_slug, "name": persona.get("name")},
        "source": "publication",
        "ref": ref,
        "origin": "graph_publications",
        "state": state,
        "version": row.get("version"),
        "checksum": row.get("checksum"),
        "runtime_checksum": row.get("checksum"),
        "compiler_version": row.get("compiler_version"),
        "validation_errors": [],
        "document": document,
        "branch_memberships": _derived_branch_memberships(document),
        "read_only": True,
    }
