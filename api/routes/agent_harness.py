from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from schemas.agent_harness import GrantsPut, MessageCreate, RunMutation, SessionCreate
from services import auth_service
from services.agent_harness import SofiaAgentHarness, capabilities_payload, validate_grants
from services.agent_harness_repository import AgentHarnessRepository


router = APIRouter(prefix="/agent-harness", tags=["agent-harness"])


def _services() -> tuple[AgentHarnessRepository, SofiaAgentHarness]:
    repository = AgentHarnessRepository()
    return repository, SofiaAgentHarness(repository)


def _assert_session(request: Request, repository: AgentHarnessRepository, session_id: str, capability: str = "view") -> dict:
    session = repository.get_session(session_id)
    if not session:
        raise HTTPException(404, "Sessao do harness nao encontrada.")
    auth_service.assert_persona_capability(request, capability, persona_id=session.get("persona_id"))
    user = auth_service.current_user(request)
    owner = str(session.get("user_id") or "")
    if owner and owner != str(user.get("id") or "") and not auth_service.is_admin(user):
        raise HTTPException(403, "Acesso negado para esta sessao.")
    return session


def _assert_run(request: Request, repository: AgentHarnessRepository, run_id: str, capability: str = "view") -> tuple[dict, dict]:
    run = repository.get_run(run_id)
    if not run:
        raise HTTPException(404, "Run nao encontrado.")
    session = _assert_session(request, repository, str(run["session_id"]), capability)
    return run, session


@router.post("/sessions")
def create_session(body: SessionCreate, request: Request):
    auth_service.assert_persona_capability(request, "view", persona_id=body.persona_id)
    _, harness = _services()
    return harness.create_session(body, user_id=str(auth_service.current_user(request)["id"]))


@router.get("/sessions/{session_id}")
def get_session(session_id: str, request: Request):
    repository, harness = _services()
    session = _assert_session(request, repository, session_id)
    return harness.session_view(session)


@router.post("/sessions/{session_id}/messages")
def create_message(session_id: str, body: MessageCreate, request: Request):
    repository, harness = _services()
    session = _assert_session(request, repository, session_id, "edit")
    return harness.add_message(
        session, body, user_id=str(auth_service.current_user(request)["id"]),
    )


@router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request):
    repository, harness = _services()
    run, _ = _assert_run(request, repository, run_id)
    return harness.run_view(run)


@router.post("/runs/{run_id}/approve")
def approve_run(run_id: str, body: RunMutation, request: Request):
    repository, harness = _services()
    run, session = _assert_run(request, repository, run_id, "edit")
    return harness.approve_run(
        run, session, expected_revision=body.expected_revision,
        user_id=str(auth_service.current_user(request)["id"]), reason=body.reason,
        idempotency_key=body.idempotency_key,
    )


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str, body: RunMutation, request: Request):
    repository, harness = _services()
    run, _ = _assert_run(request, repository, run_id, "edit")
    return harness.cancel_run(run, expected_revision=body.expected_revision, reason=body.reason)


@router.get("/capabilities")
def capabilities(request: Request):
    auth_service.current_user(request)
    return capabilities_payload()


@router.get("/grants")
def get_grants(
    request: Request,
    persona_id: str = Query(...),
    user_id: str | None = Query(None),
):
    repository, _ = _services()
    current = auth_service.current_user(request)
    target_user = user_id or str(current["id"])
    capability = "manage" if target_user != str(current["id"]) else "view"
    auth_service.assert_persona_capability(request, capability, persona_id=persona_id)
    state = repository.grant_state(target_user, persona_id)
    if not state:
        raise HTTPException(404, "Acesso do usuario a persona nao encontrado.")
    return state


@router.put("/grants")
def put_grants(body: GrantsPut, request: Request):
    repository, _ = _services()
    current = auth_service.current_user(request)
    capability = "manage" if body.user_id != str(current["id"]) else "edit"
    auth_service.assert_persona_capability(request, capability, persona_id=body.persona_id)
    validate_grants(body.grants)
    updated = repository.put_grants(
        user_id=body.user_id, persona_id=body.persona_id,
        expected_revision=body.expected_revision, grants=body.grants,
    )
    repository.audit(
        "agent_tool_grants_updated", entity_type="user_persona_access",
        entity_id=str(updated.get("id") or body.user_id), persona_id=body.persona_id,
        payload={
            "actor_user_id": current.get("id"), "target_user_id": body.user_id,
            "grant_count": len(body.grants), "reason": body.reason,
            "idempotency_key": body.idempotency_key,
        },
    )
    return updated

