from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_OFFLINE", "true")
os.environ.setdefault("KNOWLEDGE_TAXONOMY_OFFLINE", "true")
os.environ.setdefault("CURRENT_SCHEMA_VERSION", "130")

import main
from workers.runner import WORKERS


FORBIDDEN_PREFIXES = (
    "/agents",
    "/agent-harness",
    "/wa-validator",
    "/personas",
    "/knowledge",
    "/auth",
    "/leads",
)


def test_service_identity_and_readiness_surface():
    assert main.app.title == "Brain Transport"
    paths = {route.path for route in main.app.routes}
    assert "/health" in paths
    assert "/health/ready" in paths


def test_worker_group_is_domain_scoped():
    assert set(WORKERS) == {
        "health_check",
        "media_ingest",
        "whatsapp_dispatch",
    }


def test_public_surface_excludes_other_domains():
    paths = {route.path for route in main.app.routes}
    offenders = sorted(
        path
        for path in paths
        for prefix in FORBIDDEN_PREFIXES
        if path == prefix or path.startswith(prefix + "/")
    )
    assert offenders == []
