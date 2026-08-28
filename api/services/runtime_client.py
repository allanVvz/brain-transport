"""Authenticated client for conversation decisions owned by runtime."""

from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import HTTPException

from utils.tls import get_ca_bundle_path


def _configuration() -> tuple[str, str]:
    base_url = (os.environ.get("BRAIN_RUNTIME_URL") or "").strip().rstrip("/")
    token = (os.environ.get("AI_BRAIN_WEBHOOK_TOKEN") or "").strip()
    if not base_url or not token:
        raise RuntimeError("conversation runtime is not configured")
    return base_url, token


def execute_inbound(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one canonical inbound; runtime owns commit idempotency."""
    base_url, token = _configuration()
    try:
        with httpx.Client(timeout=45, verify=get_ca_bundle_path()) as client:
            response = client.post(
                base_url + "/internal/v1/conversations/execute",
                json=payload,
                headers={"X-Webhook-Token": token},
            )
    except httpx.HTTPError as exc:
        raise RuntimeError("conversation runtime is unavailable") from exc
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        raise RuntimeError(
            str(detail or f"conversation runtime returned HTTP {response.status_code}")
        )
    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError("conversation runtime returned an invalid response") from exc
    if not isinstance(result, dict):
        raise RuntimeError("conversation runtime returned an invalid response")
    return result


def decorate_leads(
    leads: list[dict[str, Any]],
    *,
    persona_id: str | None = None,
    validation_scope: str = "all",
) -> list[dict[str, Any]]:
    base_url, token = _configuration()
    try:
        with httpx.Client(timeout=15, verify=get_ca_bundle_path()) as client:
            response = client.post(
                base_url + "/internal/v1/runtime/leads/decorate",
                json={"leads": leads, "persona_id": persona_id, "validation_scope": validation_scope},
                headers={"X-Webhook-Token": token},
            )
    except httpx.HTTPError as exc:
        raise HTTPException(502, "Conversation runtime is unavailable.") from exc
    if response.status_code >= 400:
        raise HTTPException(response.status_code, "Conversation runtime rejected lead decoration.")
    try:
        items = response.json().get("items")
    except (ValueError, AttributeError) as exc:
        raise HTTPException(502, "Conversation runtime returned invalid lead decorations.") from exc
    if not isinstance(items, list):
        raise HTTPException(502, "Conversation runtime returned invalid lead decorations.")
    return [item for item in items if isinstance(item, dict)]
