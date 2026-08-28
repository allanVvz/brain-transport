import hmac
import os

from fastapi import HTTPException


def authorize_webhook_token(supplied: str | None) -> None:
    expected = (os.environ.get("AI_BRAIN_WEBHOOK_TOKEN") or "").strip()
    production = (os.environ.get("ENVIRONMENT") or "").strip().lower() == "production"
    if not expected:
        if production:
            raise HTTPException(503, "webhook token is not configured")
        return
    if not supplied or not hmac.compare_digest(
        supplied.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(401, "invalid webhook token")
