import hashlib
import hmac
import json
import os
import time
import httpx
from typing import Any, Optional
from utils.tls import get_ca_bundle_path


def _headers() -> dict:
    return {"X-N8N-API-KEY": os.environ["N8N_API_KEY"]}


def send_to_webhook(
    url: str,
    payload: dict,
    secret: Optional[str] = None,
    timeout: float = 10.0,
    response_limit: int = 300,
) -> tuple[int, str]:
    """POST a payload to an arbitrary n8n webhook URL.

    If secret is given, signs the body with HMAC-SHA256 and sends the
    signature in X-Hub-Signature-256 (GitHub-compatible format).

    Returns (status_code, body_preview). Raises httpx.HTTPError on connection
    failures so the caller can mark the message as failed.
    """
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if secret:
        sig = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
        headers["X-Hub-Signature-256"] = f"sha256={sig}"
        headers["X-Timestamp"] = str(int(time.time()))
        headers["X-Webhook-Token"] = secret
    with httpx.Client(timeout=timeout, verify=get_ca_bundle_path()) as client:
        resp = client.post(url, content=body, headers=headers)
        return resp.status_code, (resp.text or "")[:max(300, response_limit)]


def _base() -> str:
    return os.environ["N8N_BASE_URL"].rstrip("/")


def get_executions(
    limit: int = 100,
    status: Optional[str] = None,
    workflow_id: Optional[str] = None,
    *,
    include_data: bool = False,
) -> list:
    params: dict = {"limit": limit, "includeData": include_data}
    if status:
        params["status"] = status
    if workflow_id:
        params["workflowId"] = workflow_id

    with httpx.Client(timeout=15, verify=get_ca_bundle_path()) as client:
        response = client.get(f"{_base()}/api/v1/executions", headers=_headers(), params=params)
        response.raise_for_status()
        return response.json().get("data", [])


def get_execution(execution_id: str) -> dict:
    with httpx.Client(timeout=15, verify=get_ca_bundle_path()) as client:
        response = client.get(
            f"{_base()}/api/v1/executions/{execution_id}",
            headers=_headers(),
            params={"includeData": True},
        )
        response.raise_for_status()
        return response.json()


def get_workflows() -> list:
    with httpx.Client(timeout=15, verify=get_ca_bundle_path()) as client:
        response = client.get(f"{_base()}/api/v1/workflows", headers=_headers())
        response.raise_for_status()
        return response.json().get("data", [])


def get_workflow(workflow_id: str) -> Optional[dict]:
    """Fetch one workflow's definition. Returns None if it no longer exists.

    n8n's public API has no read endpoint for credentials themselves (GET by
    id or list both 405) â€” a node's stored `credentials.<type>.id` can go
    stale (object deleted elsewhere in n8n) with nothing here to detect it.
    This only confirms the *workflow* still exists and which credential id
    its node currently references, not that the credential object is real.
    """
    with httpx.Client(timeout=15, verify=get_ca_bundle_path()) as client:
        response = client.get(f"{_base()}/api/v1/workflows/{workflow_id}", headers=_headers())
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()


def create_credential(
    *,
    name: str,
    credential_type: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    with httpx.Client(timeout=15, verify=get_ca_bundle_path()) as client:
        response = client.post(
            f"{_base()}/api/v1/credentials",
            headers=_headers(),
            json={"name": name, "type": credential_type, "data": data},
        )
        response.raise_for_status()
        return response.json()


def delete_credential(credential_id: str) -> None:
    if not credential_id:
        return
    with httpx.Client(timeout=15, verify=get_ca_bundle_path()) as client:
        response = client.delete(
            f"{_base()}/api/v1/credentials/{credential_id}",
            headers=_headers(),
        )
        if response.status_code != 404:
            response.raise_for_status()


def create_workflow(workflow: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: workflow[key]
        for key in ("name", "nodes", "connections", "settings")
        if key in workflow
    }
    with httpx.Client(timeout=20, verify=get_ca_bundle_path()) as client:
        response = client.post(
            f"{_base()}/api/v1/workflows",
            headers=_headers(),
            json=payload,
        )
        response.raise_for_status()
        return response.json()


def update_workflow(workflow_id: str, workflow: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: workflow[key]
        for key in ("name", "nodes", "connections", "settings")
        if key in workflow
    }
    with httpx.Client(timeout=20, verify=get_ca_bundle_path()) as client:
        response = client.put(
            f"{_base()}/api/v1/workflows/{workflow_id}",
            headers=_headers(),
            json=payload,
        )
        response.raise_for_status()
        return response.json()


def activate_workflow(workflow_id: str) -> dict[str, Any]:
    with httpx.Client(timeout=15, verify=get_ca_bundle_path()) as client:
        response = client.post(
            f"{_base()}/api/v1/workflows/{workflow_id}/activate",
            headers=_headers(),
        )
        response.raise_for_status()
        return response.json()


def deactivate_workflow(workflow_id: str) -> dict[str, Any]:
    with httpx.Client(timeout=15, verify=get_ca_bundle_path()) as client:
        response = client.post(
            f"{_base()}/api/v1/workflows/{workflow_id}/deactivate",
            headers=_headers(),
        )
        response.raise_for_status()
        return response.json()


def ping() -> tuple[bool, int]:
    try:
        with httpx.Client(timeout=5, verify=get_ca_bundle_path()) as client:
            import time
            t0 = time.monotonic()
            response = client.get(f"{_base()}/api/v1/workflows", headers=_headers())
            ms = int((time.monotonic() - t0) * 1000)
            return response.status_code == 200, ms
    except Exception:
        return False, -1

