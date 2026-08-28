from __future__ import annotations

import os
from functools import lru_cache

import certifi


@lru_cache(maxsize=1)
def get_ca_bundle_path() -> str:
    custom = (os.environ.get("AI_BRAIN_CA_BUNDLE") or "").strip()
    return custom or certifi.where()


@lru_cache(maxsize=1)
def _inject_system_trust_store() -> bool:
    disabled = (os.environ.get("AI_BRAIN_DISABLE_SYSTEM_TRUSTSTORE") or "").strip().lower()
    if disabled in {"1", "true", "yes", "on"}:
        return False

    try:
        import truststore  # type: ignore

        truststore.inject_into_ssl()
        return True
    except Exception:
        return False


def configure_trust_store() -> str:
    if _inject_system_trust_store():
        return "system"

    bundle = get_ca_bundle_path()
    os.environ.setdefault("SSL_CERT_FILE", bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
    os.environ.setdefault("CURL_CA_BUNDLE", bundle)
    return bundle

