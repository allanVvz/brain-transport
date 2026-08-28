from __future__ import annotations

import os

import httpx

from utils.tls import get_ca_bundle_path


def attach_inbound_asset(asset_id: str) -> dict:
    base_url = (os.environ.get("BRAIN_CONTROL_PLANE_URL") or "").strip().rstrip("/")
    token = (os.environ.get("AI_BRAIN_WEBHOOK_TOKEN") or "").strip()
    if not base_url or not token:
        raise RuntimeError("control plane is not configured")
    try:
        with httpx.Client(timeout=20, verify=get_ca_bundle_path()) as client:
            response = client.post(
                base_url + f"/internal/v1/control-plane/assets/{asset_id}/attach-inbound-graph",
                headers={"X-Webhook-Token": token},
            )
    except httpx.HTTPError as exc:
        raise RuntimeError("control plane is unavailable") from exc
    if response.status_code >= 400:
        raise RuntimeError(f"control plane returned HTTP {response.status_code}")
    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError("control plane returned an invalid response") from exc
    if not isinstance(result, dict):
        raise RuntimeError("control plane returned an invalid response")
    return result
