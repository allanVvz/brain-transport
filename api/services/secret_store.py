from __future__ import annotations

import base64
import hashlib
import os

from cryptography.fernet import Fernet, InvalidToken


def _secret_material() -> str:
    value = (
        (os.environ.get("AI_BRAIN_SECRETS_KEY") or "").strip()
        or (os.environ.get("AI_BRAIN_AUTH_SECRET") or "").strip()
        or (os.environ.get("NEXTAUTH_SECRET") or "").strip()
    )
    if value:
        return value
    if (os.environ.get("ENVIRONMENT") or "").lower() == "production":
        raise RuntimeError("AI_BRAIN_SECRETS_KEY is required in production")
    return "dev-only-ai-brain-secrets-key-change-me"


def _fernet() -> Fernet:
    digest = hashlib.sha256(_secret_material().encode("utf-8")).digest()
    key = base64.urlsafe_b64encode(digest)
    return Fernet(key)


def encrypt_secret(value: str) -> str:
    if value is None:
        raise ValueError("Secret value is required.")
    return _fernet().encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt_secret(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return _fernet().decrypt(value.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise ValueError("Stored credential could not be decrypted.") from exc


