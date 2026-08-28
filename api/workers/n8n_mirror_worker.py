import os
from datetime import datetime, timezone
from workers.base_worker import BaseWorker
from services import integration_service, n8n_client, supabase_client, sre_logger


def _walk_values(value):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_values(nested)


def _first_nested_value(payload: dict, keys: tuple[str, ...]):
    for node in _walk_values(payload):
        if not isinstance(node, dict):
            continue
        for key in keys:
            value = node.get(key)
            if value not in (None, "", [], {}):
                return value
    return None


class N8nMirrorWorker(BaseWorker):
    name = "N8nMirrorWorker"
    interval = int(os.environ.get("N8N_MIRROR_INTERVAL", 300))

    def _run_cycle(self):
        if not integration_service.system_service_has_runtime_credentials("n8n"):
            sre_logger.info(self.name, "skipped: n8n integration credentials are not configured")
            return
        # includeData is required for runData/node errors. Without it, n8n
        # can report an execution as "success" while an HTTP node followed
        # its continueErrorOutput branch, leaving the Logs tab empty exactly
        # when an operator needs the failure payload.
        executions = n8n_client.get_executions(limit=50, include_data=True)
        synced = 0
        for ex in executions:
            try:
                started = ex.get("startedAt") or ex.get("started_at")
                finished = ex.get("stoppedAt") or ex.get("finished_at")
                duration = None
                if started and finished:
                    try:
                        s = datetime.fromisoformat(started.replace("Z", "+00:00"))
                        f = datetime.fromisoformat(finished.replace("Z", "+00:00"))
                        duration = int((f - s).total_seconds() * 1000)
                    except Exception:
                        pass

                node_errors = []
                data = ex.get("data") or {}
                if isinstance(data, dict):
                    for node_name, node_data in data.get("resultData", {}).get("runData", {}).items():
                        for run in (node_data or []):
                            if run.get("error"):
                                node_errors.append({"node": node_name, "error": run["error"]})

                workflow_name = (
                    (ex.get("workflowData") or {}).get("name")
                    or (ex.get("workflow") or {}).get("name")
                    or ex.get("workflowName")
                    or _first_nested_value(ex, ("workflow_name", "workflowName"))
                    or ""
                )
                lead_id = _first_nested_value(ex, ("lead_id", "leadId", "lead_ref", "leadRef"))
                persona_id = _first_nested_value(ex, ("persona_id", "personaId"))

                supabase_client.upsert_n8n_execution({
                    "n8n_id": str(ex["id"]),
                    "workflow_name": workflow_name,
                    "status": ex.get("status", "unknown"),
                    "started_at": started,
                    "finished_at": finished,
                    "duration_ms": duration,
                    "node_errors": node_errors,
                    "lead_id": str(lead_id) if lead_id not in (None, "") else None,
                    "persona_id": str(persona_id) if persona_id not in (None, "") else None,
                })
                synced += 1
            except Exception as exc:
                sre_logger.error(self.name, f"failed to mirror execution id={ex.get('id')}", exc)

        sre_logger.info(self.name, f"synced {synced}/{len(executions)} executions")

