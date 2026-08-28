from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from schemas.agent_harness import MessageCreate, RunStatus, SessionCreate, ToolEffect, ToolGrant
from services import graph_json_v2_store, supabase_client
from services.agent_harness_repository import AgentHarnessRepository, database_uuid, now_iso, redact
from services.agent_harness_tools import HARNESS_TOOL_REGISTRY
from services.agent_tool_registry import ToolManifest


MAX_DELEGATION_DEPTH = 2
MAX_STEPS_PER_RUN = 12
COORDINATOR_KEY = "sofia"
GRAPH_AGENT = "graph_card_specialist"
CAMPAIGN_AGENT = "campaign_operator"
QA_AGENT = "qa_validator"


def validate_delegation(path: list[str], next_agent: str) -> list[str]:
    normalized = [str(item).strip() for item in path if str(item).strip()]
    if next_agent in normalized:
        raise HTTPException(422, "Ciclo de delegacao bloqueado.")
    # Depth counts transitions: Sofia -> specialist -> QA is depth 2.
    if len(normalized) >= MAX_DELEGATION_DEPTH + 1:
        raise HTTPException(422, "Profundidade maxima de delegacao atingida.")
    return [*normalized, next_agent]


def classify_intent(message: str) -> tuple[str, str]:
    text = (message or "").lower()
    phones = re.findall(r"\+?\d[\d\s().-]{6,}\d", message or "")
    campaign_terms = ("campanha", "contato", "telefone", "whatsapp", "import", "disparo", "consent")
    if phones or any(term in text for term in campaign_terms):
        if "cancel" in text:
            return "campaign.cancel", CAMPAIGN_AGENT
        if "paus" in text:
            return "campaign.pause", CAMPAIGN_AGENT
        return "campaign.contacts", CAMPAIGN_AGENT
    return "graph.cards", GRAPH_AGENT


def _session_id(user_id: str, persona_id: str, idempotency_key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"brain-agent-session:{user_id}:{persona_id}:{idempotency_key}"))


def _graph_snapshot(persona_id: str) -> tuple[int | None, str | None, str | None]:
    persona = supabase_client.get_persona_by_id(persona_id)
    slug = str((persona or {}).get("slug") or "") or None
    if not slug:
        return None, None, None
    loaded = graph_json_v2_store.load_current(slug)
    if not loaded:
        return None, None, slug
    version, graph = loaded
    return int(version), graph_json_v2_store.checksum_graph(graph), slug


def _has_grant(
    repository: AgentHarnessRepository,
    *,
    user_id: str,
    persona_id: str,
    manifest: ToolManifest,
) -> bool:
    now = datetime.now(timezone.utc)
    return any(
        grant.is_valid(
            now=now, tool_name=manifest.name, version=manifest.version,
            checksum=manifest.schema_checksum,
        )
        for grant in repository.grants(user_id, persona_id)
    )


def _step_payload(
    *,
    run_id: str,
    order: int,
    manifest: ToolManifest,
    arguments: dict[str, Any],
    idempotency_key: str,
    status: str,
    parent_step_id: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "parent_step_id": parent_step_id,
        "step_order": order,
        "agent_key": manifest.owner_agent,
        "tool_name": manifest.name,
        "tool_version": manifest.version,
        "effect": manifest.effect.value,
        "risk": manifest.risk.value,
        "schema_checksum": manifest.schema_checksum,
        "input_payload": arguments,
        "output_payload": {},
        "status": status,
        "idempotency_key": idempotency_key,
        "contains_sensitive_data": bool(manifest.sensitive_fields),
    }


class SofiaAgentHarness:
    def __init__(self, repository: AgentHarnessRepository | None = None) -> None:
        self.repository = repository or AgentHarnessRepository()
        HARNESS_TOOL_REGISTRY.validate()

    def create_session(self, body: SessionCreate, *, user_id: str) -> dict[str, Any]:
        graph_version, graph_hash, persona_slug = _graph_snapshot(body.persona_id)
        session_id = _session_id(user_id, body.persona_id, body.idempotency_key)
        existing = self.repository.get_session(session_id)
        if existing:
            return self.session_view(existing)
        row = self.repository.create_session({
            "id": session_id,
            "coordinator_key": body.coordinator_key,
            "user_id": database_uuid(user_id),
            "persona_id": body.persona_id,
            "selected_context": {
                **body.selected_context,
                "persona_slug": body.selected_context.get("persona_slug") or persona_slug,
                "created_reason": body.reason,
            },
            "graph_version": body.graph_version if body.graph_version is not None else graph_version,
            "graph_hash": body.graph_hash or graph_hash,
            "selected_node_ids": body.selected_node_ids,
            "memory_summary": "",
            "status": "active",
            "revision": 1,
        })
        self.repository.audit(
            "agent_session_created", entity_type="agent_session", entity_id=session_id,
            persona_id=body.persona_id,
            payload={"coordinator_key": body.coordinator_key, "reason": body.reason, "idempotency_key": body.idempotency_key},
        )
        return self.session_view(row)

    def adopt_legacy_session(
        self,
        *,
        session_id: str,
        persona_id: str,
        user_id: str,
        source: str,
        selected_context: dict[str, Any] | None = None,
        memory_summary: str = "",
    ) -> dict[str, Any]:
        canonical_session_id = database_uuid(session_id) or _session_id(
            user_id,
            persona_id,
            f"legacy:{source}:{session_id}",
        )
        existing = self.repository.get_session(canonical_session_id)
        if existing:
            return existing
        graph_version, graph_hash, persona_slug = _graph_snapshot(persona_id)
        return self.repository.create_session({
            "id": canonical_session_id,
            "coordinator_key": COORDINATOR_KEY,
            "user_id": database_uuid(user_id),
            "persona_id": persona_id,
            "selected_context": {
                **(selected_context or {}),
                "persona_slug": (selected_context or {}).get("persona_slug") or persona_slug,
                "adapter_source": source,
            },
            "graph_version": graph_version,
            "graph_hash": graph_hash,
            "selected_node_ids": (selected_context or {}).get("selected_node_ids") or [],
            "memory_summary": memory_summary,
            "status": "active",
            "revision": 1,
        })

    def record_adapter_run(
        self,
        *,
        session: dict[str, Any],
        user_message: str,
        response: dict[str, Any],
        source: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        intent, agent = classify_intent(user_message)
        run = self.repository.create_run({
            "session_id": session["id"],
            "parent_run_id": None,
            "intent": intent,
            "assigned_agent_key": agent,
            "input_payload": {"message": user_message, "adapter_source": source},
            "response_payload": response,
            "plan": {"delegation_path": [COORDINATOR_KEY, agent], "adapter_source": source},
            "artifacts": [],
            "status": RunStatus.COMPLETED.value if response.get("ok", True) else RunStatus.FAILED.value,
            "revision": 1,
            "idempotency_key": idempotency_key,
            "started_at": now_iso(),
            "completed_at": now_iso(),
        })
        return run

    def session_view(self, session: dict[str, Any]) -> dict[str, Any]:
        runs = self.repository.list_runs(str(session["id"]))
        return {
            **session,
            "runs": [self.run_view(run, include_private=False) for run in runs],
        }

    def run_view(self, run: dict[str, Any], *, include_private: bool = False) -> dict[str, Any]:
        steps = self.repository.list_steps(str(run["id"]))
        payload = {**run, "steps": steps}
        return payload if include_private else redact(payload)

    def add_message(self, session: dict[str, Any], body: MessageCreate, *, user_id: str) -> dict[str, Any]:
        if int(session.get("revision") or 0) != body.expected_revision:
            raise HTTPException(409, "A sessao mudou; recarregue antes de continuar.")
        intent, agent = classify_intent(body.message)
        delegation_path = validate_delegation([COORDINATOR_KEY], agent)
        run = self.repository.create_run({
            "session_id": session["id"],
            "parent_run_id": None,
            "intent": intent,
            "assigned_agent_key": agent,
            "input_payload": {"message": body.message, "context": body.context, "reason": body.reason},
            "response_payload": {},
            "plan": {"delegation_path": delegation_path, "steps": []},
            "artifacts": [],
            "status": RunStatus.RUNNING.value,
            "revision": 1,
            "idempotency_key": body.idempotency_key,
            "started_at": now_iso(),
        })
        # Idempotent replay returns the already materialized run.
        if run.get("status") != RunStatus.RUNNING.value or self.repository.list_steps(str(run["id"])):
            return self.run_view(run)

        try:
            if agent == CAMPAIGN_AGENT:
                run = self._plan_campaign_message(run, session, body, user_id=user_id)
            else:
                run = self.repository.update_run(
                    str(run["id"]), expected_revision=int(run["revision"]), changes={
                        "status": RunStatus.COMPLETED.value,
                        "response_payload": {
                            "message": "O graph_card_specialist esta pronto para localizar cards e preparar patches Graph JSON 2.1.",
                            "next_capabilities": ["graph.find_cards", "graph.create_card_draft", "graph.validate_patch"],
                        },
                        "completed_at": now_iso(),
                    },
                )
            summary = f"{agent}: {intent}; status={run['status']}"
            self.repository.update_session(
                str(session["id"]), expected_revision=body.expected_revision,
                changes={
                    "memory_summary": summary,
                    "selected_context": {**(session.get("selected_context") or {}), **body.context, "last_run_id": run["id"]},
                },
            )
            self.repository.audit(
                "agent_run_updated", entity_type="agent_run", entity_id=str(run["id"]),
                persona_id=str(session.get("persona_id") or "") or None,
                payload={"intent": intent, "agent": agent, "status": run["status"], "reason": body.reason},
            )
            return self.run_view(run)
        except Exception as exc:
            latest = self.repository.get_run(str(run["id"])) or run
            if latest.get("status") not in {RunStatus.COMPLETED.value, RunStatus.FAILED.value}:
                self.repository.update_run(
                    str(run["id"]), expected_revision=int(latest.get("revision") or 1), changes={
                        "status": RunStatus.FAILED.value,
                        "error_code": type(exc).__name__,
                        "error_message": str(exc)[:1000],
                        "completed_at": now_iso(),
                    },
                )
            raise

    def _plan_campaign_message(self, run: dict[str, Any], session: dict[str, Any], body: MessageCreate, *, user_id: str) -> dict[str, Any]:
        action = str(body.context.get("action") or "").strip().lower()
        if run.get("intent") in {"campaign.pause", "campaign.cancel"} or action in {"pause", "cancel"}:
            operation = "cancel" if run.get("intent") == "campaign.cancel" or action == "cancel" else "pause"
            campaign_id = str(body.context.get("campaign_id") or "").strip()
            expected_campaign_revision = int(body.context.get("campaign_revision") or 0)
            if not campaign_id or expected_campaign_revision < 1:
                raise HTTPException(422, "campaign_id e campaign_revision sao obrigatorios.")
            pending_tool = {
                "name": f"campaigns.{operation}",
                "arguments": {
                    "campaign_id": campaign_id,
                    "expected_revision": expected_campaign_revision,
                    "idempotency_key": f"{body.idempotency_key}:{operation}",
                    "reason": body.reason,
                },
                "idempotency_key": f"{body.idempotency_key}:{operation}-step",
            }
            return self.repository.update_run(
                str(run["id"]), expected_revision=int(run["revision"]), changes={
                    "status": RunStatus.AWAITING_APPROVAL.value,
                    "plan": {
                        "delegation_path": [COORDINATOR_KEY, CAMPAIGN_AGENT],
                        "pending_tool": pending_tool, "qa_required_before_write": True,
                    },
                    "response_payload": {
                        "message": f"Confirme para {operation} a campanha. Nenhuma alteracao foi feita.",
                        "specialist": CAMPAIGN_AGENT,
                    },
                },
            )

        if action in {"campaign_preview", "campaign_create_draft"} or body.context.get("import_batch_ids"):
            preview_args = {
                "persona_id": session["persona_id"],
                "audience_id": body.context.get("audience_id"),
                "import_batch_ids": body.context.get("import_batch_ids") or [],
                "purpose": body.context.get("purpose") or "ofertas_e_novidades",
                "campaign_kind": body.context.get("campaign_kind") or "consent_request",
                "name": body.context.get("name"),
                "objective": body.context.get("objective"),
                "provider": "meta_cloud",
                "template_name": body.context.get("template_name"),
                "template_language": body.context.get("template_language") or "pt_BR",
                "message": body.context.get("campaign_message"),
                "variables": body.context.get("variables") or {},
                "assets": body.context.get("assets") or [],
                "policy_overrides": body.context.get("policy_overrides") or {},
            }
            preview_result = self.execute_tool(
                run=run, session=session, user_id=user_id, tool_name="campaigns.preview",
                arguments=preview_args, idempotency_key=f"{body.idempotency_key}:campaign-preview",
                one_time_approval=False,
            )
            run = self.repository.get_run(str(run["id"])) or run
            preview = (preview_result.get("output") or {}).get("preview") or {}
            pending_tool = None
            status = RunStatus.COMPLETED.value
            message = "Preview de campanha pronto; nenhuma outbox foi criada."
            if action == "campaign_create_draft" or body.context.get("create_draft") is True:
                pending_tool = {
                    "name": "campaigns.create_draft",
                    "arguments": {
                        **preview_args,
                        "name": preview_args.get("name") or "Campanha Sofia",
                        "expected_revision": 0,
                        "expected_preview_checksum": preview.get("preview_checksum"),
                        "idempotency_key": f"{body.idempotency_key}:campaign-create",
                        "reason": body.reason,
                    },
                    "idempotency_key": f"{body.idempotency_key}:campaign-create-step",
                }
                status = RunStatus.AWAITING_APPROVAL.value
                message = "Preview pronto. Confirme para criar somente o draft; nenhum envio sera iniciado."
            return self.repository.update_run(
                str(run["id"]), expected_revision=int(run["revision"]), changes={
                    "status": status,
                    "plan": {
                        "delegation_path": [COORDINATOR_KEY, CAMPAIGN_AGENT],
                        "pending_tool": pending_tool, "qa_required_before_write": True,
                        "provider_ready": preview.get("provider_ready"),
                    },
                    "response_payload": {"message": message, "specialist": CAMPAIGN_AGENT, "preview": preview},
                    "completed_at": now_iso() if status == RunStatus.COMPLETED.value else None,
                },
            )

        parse_result = self.execute_tool(
            run=run, session=session, user_id=user_id, tool_name="contacts.parse_list",
            arguments={"raw_text": body.message},
            idempotency_key=f"{body.idempotency_key}:parse", one_time_approval=False,
        )
        run = self.repository.get_run(str(run["id"])) or run
        contacts = (parse_result.get("output") or {}).get("contacts") or []
        audience_id = body.context.get("audience_id")
        audience_name = body.context.get("audience_name") or "Contatos Sofia"
        preview_args = {
            "persona_id": session["persona_id"], "audience_id": audience_id,
            "audience_name": audience_name, "contacts": contacts,
            "consent_purpose": body.context.get("consent_purpose") or "ofertas_e_novidades",
        }
        preview_result = self.execute_tool(
            run=run, session=session, user_id=user_id, tool_name="imports.preview",
            arguments=preview_args, idempotency_key=f"{body.idempotency_key}:preview",
            one_time_approval=False,
        )
        run = self.repository.get_run(str(run["id"])) or run
        pending_tool = None
        status = RunStatus.COMPLETED.value
        message = "Preview pronto. Selecione um grupo semantico para confirmar o import."
        if audience_id and contacts:
            pending_tool = {
                "name": "imports.create",
                "arguments": {
                    **preview_args,
                    "audience_id": audience_id,
                    "contact_basis": body.context.get("contact_basis") or "imported_without_evidence",
                    "contact_basis_statement": body.context.get("contact_basis_statement"),
                    "idempotency_key": f"{body.idempotency_key}:create",
                    "reason": body.reason,
                },
                "idempotency_key": f"{body.idempotency_key}:create-step",
            }
            # A valid persistent grant allows autonomous write; otherwise the
            # exact tool call is stored for point approval.
            manifest = HARNESS_TOOL_REGISTRY.get("imports.create")
            if manifest and _has_grant(
                self.repository, user_id=user_id, persona_id=str(session["persona_id"]), manifest=manifest,
            ):
                self.execute_tool(
                    run=run, session=session, user_id=user_id, tool_name="imports.create",
                    arguments=pending_tool["arguments"], idempotency_key=pending_tool["idempotency_key"],
                    one_time_approval=False,
                )
                run = self.repository.get_run(str(run["id"])) or run
                pending_tool = None
                message = "Import concluido. Consentimentos sem evidencia ficaram pending."
            else:
                status = RunStatus.AWAITING_APPROVAL.value
                message = "Preview pronto. Confirme para criar o import; nenhuma escrita foi feita."
        plan = {
            "delegation_path": [COORDINATOR_KEY, CAMPAIGN_AGENT],
            "preview_counts": (preview_result.get("output") or {}).get("preview", {}).get("counts", {}),
            "pending_tool": pending_tool,
            "qa_required_before_write": True,
        }
        return self.repository.update_run(
            str(run["id"]), expected_revision=int(run["revision"]), changes={
                "status": status,
                "plan": plan,
                "response_payload": {
                    "message": message,
                    "specialist": CAMPAIGN_AGENT,
                    "preview": (preview_result.get("output") or {}).get("preview", {}),
                },
                "completed_at": now_iso() if status == RunStatus.COMPLETED.value else None,
            },
        )

    def execute_tool(
        self,
        *,
        run: dict[str, Any],
        session: dict[str, Any],
        user_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        idempotency_key: str,
        one_time_approval: bool,
    ) -> dict[str, Any]:
        manifest = HARNESS_TOOL_REGISTRY.get(tool_name)
        if manifest is None or manifest.deprecated:
            raise HTTPException(422, f"Ferramenta indisponivel: {tool_name}")
        steps = self.repository.list_steps(str(run["id"]))
        if len(steps) >= MAX_STEPS_PER_RUN:
            raise HTTPException(422, "Limite de 12 steps por run atingido.")
        if manifest.persona_scoped and str(arguments.get("persona_id") or session.get("persona_id")) != str(session.get("persona_id")):
            raise HTTPException(403, "Ferramenta tentou sair do escopo da persona.")
        requires_approval = manifest.effect in {ToolEffect.WRITE, ToolEffect.DESTRUCTIVE, ToolEffect.EXTERNAL}
        persistent_grant = _has_grant(
            self.repository, user_id=user_id, persona_id=str(session["persona_id"]), manifest=manifest,
        ) if requires_approval else False
        approved = one_time_approval or persistent_grant
        # Destructive operations ignore persistent grants.
        if manifest.effect == ToolEffect.DESTRUCTIVE:
            approved = one_time_approval
        if requires_approval and not approved:
            step = self.repository.create_step(_step_payload(
                run_id=str(run["id"]), order=len(steps) + 1, manifest=manifest,
                arguments=arguments, idempotency_key=idempotency_key,
                status=RunStatus.AWAITING_APPROVAL.value,
            ))
            return {"status": RunStatus.AWAITING_APPROVAL.value, "step": step, "output": {}}

        parent_step_id = None
        if manifest.effect in {ToolEffect.WRITE, ToolEffect.DESTRUCTIVE, ToolEffect.EXTERNAL}:
            qa_path = validate_delegation([COORDINATOR_KEY, manifest.owner_agent], QA_AGENT)
            qa_manifest = HARNESS_TOOL_REGISTRY.get("qa.validate_operation")
            assert qa_manifest is not None
            qa_arguments = {
                "operation": manifest.name,
                "persona_id": session["persona_id"],
                "payload": arguments,
                "expected_graph_version": session.get("graph_version"),
                "graph_hash": session.get("graph_hash"),
            }
            qa_step = self.repository.create_step(_step_payload(
                run_id=str(run["id"]), order=len(steps) + 1, manifest=qa_manifest,
                arguments=qa_arguments, idempotency_key=f"{idempotency_key}:qa",
                status=RunStatus.RUNNING.value,
            ))
            started = time.monotonic()
            qa_output = qa_manifest.invoke({"delegation_path": qa_path}, qa_arguments)
            qa_step = self.repository.update_step(str(qa_step["id"]), {
                "status": RunStatus.COMPLETED.value if qa_output.get("ok") else RunStatus.FAILED.value,
                "output_payload": qa_output,
                "duration_ms": int((time.monotonic() - started) * 1000),
            })
            if not qa_output.get("ok"):
                raise HTTPException(422, {"message": "QA bloqueou a operacao.", "violations": qa_output.get("violations") or []})
            parent_step_id = str(qa_step["id"])
            steps = self.repository.list_steps(str(run["id"]))

        step = self.repository.create_step(_step_payload(
            run_id=str(run["id"]), order=len(steps) + 1, manifest=manifest,
            arguments=arguments, idempotency_key=idempotency_key,
            status=RunStatus.RUNNING.value, parent_step_id=parent_step_id,
        ))
        if step.get("status") == RunStatus.COMPLETED.value:
            return {"status": step["status"], "step": step, "output": step.get("output_payload") or {}}
        started = time.monotonic()
        context = {
            "session_id": session["id"],
            "run_id": run["id"],
            "user_id": user_id,
            "database_user_id": database_uuid(user_id),
            "persona_id": session["persona_id"],
        }
        executor: ThreadPoolExecutor | None = None
        try:
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(manifest.invoke, context, arguments)
            output = future.result(timeout=manifest.timeout_seconds)
            executor.shutdown(wait=True)
            executor = None
            status = RunStatus.COMPLETED.value if output.get("ok", True) else RunStatus.FAILED.value
            step = self.repository.update_step(str(step["id"]), {
                "status": status,
                "output_payload": output,
                "duration_ms": int((time.monotonic() - started) * 1000),
            })
            self.repository.audit(
                "agent_tool_executed", entity_type="agent_run_step", entity_id=str(step["id"]),
                persona_id=str(session["persona_id"]),
                payload={
                    "tool": manifest.name, "version": manifest.version, "effect": manifest.effect.value,
                    "risk": manifest.risk.value, "schema_checksum": manifest.schema_checksum,
                    "input": arguments, "output": output,
                },
            )
            return {"status": status, "step": step, "output": output}
        except FutureTimeout as exc:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            self.repository.update_step(str(step["id"]), {
                "status": RunStatus.FAILED.value, "duration_ms": manifest.timeout_seconds * 1000,
                "error_code": "tool_timeout", "error_message": "Tempo limite da ferramenta excedido.",
            })
            raise HTTPException(504, "Tempo limite da ferramenta excedido.") from exc
        except Exception as exc:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            self.repository.update_step(str(step["id"]), {
                "status": RunStatus.FAILED.value,
                "duration_ms": int((time.monotonic() - started) * 1000),
                "error_code": type(exc).__name__, "error_message": str(exc)[:1000],
            })
            raise

    def approve_run(self, run: dict[str, Any], session: dict[str, Any], *, expected_revision: int, user_id: str, reason: str, idempotency_key: str) -> dict[str, Any]:
        if int(run.get("revision") or 0) != expected_revision:
            raise HTTPException(409, "O run mudou; recarregue antes de aprovar.")
        if run.get("status") != RunStatus.AWAITING_APPROVAL.value:
            raise HTTPException(409, "Este run nao aguarda aprovacao.")
        pending = (run.get("plan") or {}).get("pending_tool") or {}
        if not pending.get("name"):
            raise HTTPException(409, "O run nao possui ferramenta pendente.")
        result = self.execute_tool(
            run=run, session=session, user_id=user_id, tool_name=pending["name"],
            arguments=pending.get("arguments") or {}, idempotency_key=pending.get("idempotency_key") or idempotency_key,
            one_time_approval=True,
        )
        latest = self.repository.get_run(str(run["id"])) or run
        updated = self.repository.update_run(
            str(run["id"]), expected_revision=int(latest["revision"]), changes={
                "status": RunStatus.COMPLETED.value,
                "plan": {**(run.get("plan") or {}), "pending_tool": None, "approved_reason": reason},
                "response_payload": {"message": "Operacao confirmada e validada pelo QA.", "result": result.get("output")},
                "completed_at": now_iso(),
            },
        )
        return self.run_view(updated)

    def cancel_run(self, run: dict[str, Any], *, expected_revision: int, reason: str) -> dict[str, Any]:
        if int(run.get("revision") or 0) != expected_revision:
            raise HTTPException(409, "O run mudou; recarregue antes de cancelar.")
        if run.get("status") in {RunStatus.COMPLETED.value, RunStatus.CANCELLED.value}:
            raise HTTPException(409, "Run encerrado nao pode ser cancelado.")
        updated = self.repository.update_run(
            str(run["id"]), expected_revision=expected_revision, changes={
                "status": RunStatus.CANCELLED.value,
                "response_payload": {"message": "Run cancelado pelo operador.", "reason": reason},
                "completed_at": now_iso(),
            },
        )
        return self.run_view(updated)


def validate_grants(grants: list[ToolGrant]) -> None:
    now = datetime.now(timezone.utc)
    for grant in grants:
        manifest = HARNESS_TOOL_REGISTRY.get(grant.tool_name)
        if manifest is None or manifest.deprecated:
            raise HTTPException(422, f"Ferramenta desconhecida ou depreciada: {grant.tool_name}")
        if grant.tool_version != manifest.version or grant.schema_checksum != manifest.schema_checksum:
            raise HTTPException(409, f"Schema/version mudou para {grant.tool_name}; gere um novo grant.")
        expiry = grant.expires_at
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if expiry <= now:
            raise HTTPException(422, f"Grant expirado para {grant.tool_name}.")


def capabilities_payload() -> dict[str, Any]:
    return {
        "coordinator": COORDINATOR_KEY,
        "specialists": [GRAPH_AGENT, CAMPAIGN_AGENT, QA_AGENT],
        "limits": {"max_delegation_depth": MAX_DELEGATION_DEPTH, "max_steps_per_run": MAX_STEPS_PER_RUN},
        "tools": HARNESS_TOOL_REGISTRY.capabilities(),
        "external_adapters": {"mcp": False, "stdio": False, "streaming": "polling"},
    }

