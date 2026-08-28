import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi import FastAPI
from fastapi.testclient import TestClient

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from services import auth_service
from services.whatsapp_providers.evolution import EvolutionWhatsAppProvider
from middleware.auth import auth_middleware
from routes import evolution_webhook, portal
from workers.whatsapp_dispatch_worker import WhatsAppDispatchWorker
from workers import whatsapp_dispatch_worker


def request_for(user: dict, access: list[dict]):
    return SimpleNamespace(state=SimpleNamespace(user=user, persona_access=access))


def test_portal_chat_context_reuses_operator_projection_and_persona_scope(monkeypatch):
    request = request_for(
        {"id": "u1", "role": "user", "account_type": "client"},
        [{
            "persona_id": "p1", "persona_slug": "aurora",
            "can_view": True, "can_edit": True, "can_manage": False,
        }],
    )
    monkeypatch.setattr(
        portal.supabase_client,
        "get_persona",
        lambda _slug: {"id": "p1", "slug": "aurora"},
    )
    monkeypatch.setattr(
        portal.supabase_client,
        "get_lead_by_ref",
        lambda _lead_id: {"id": 42, "persona_id": "p1"},
    )
    captured = {}
    monkeypatch.setattr(
        portal.knowledge_graph,
        "get_chat_context",
        lambda **kwargs: captured.update(kwargs) or {"nodes": [], "edges": []},
    )
    monkeypatch.setattr(
        portal.knowledge_graph,
        "with_operator_context",
        lambda context, *, limit: {
            **context,
            "operator_context": {
                "primary": [], "faq_rules": [], "graph_path": [],
            },
            "limit": limit,
        },
    )
    messages = [{"id": "m1", "role": "assistant"}]
    monkeypatch.setattr(
        portal.supabase_client,
        "get_messages",
        lambda _lead_id, *, limit: messages,
    )
    monkeypatch.setattr(
        portal.supabase_client,
        "list_all_knowledge_graph",
        lambda **_kwargs: ([], []),
    )
    captured_turn = {}
    monkeypatch.setattr(
        portal.context_cards_service,
        "response_context",
        lambda **kwargs: captured_turn.update(kwargs) or {
            "mode": "exact", "used_cards": [], "related_cards": [],
        },
    )

    result = portal.knowledge_chat_context(
        request,
        persona_slug="aurora",
        lead_ref=42,
        q="lavagem",
        response_message_id="m1",
        limit=7,
    )

    assert captured == {
        "lead_ref": 42,
        "persona_id": "p1",
        "user_text": "lavagem",
        "limit": 7,
    }
    assert result["operator_context"] == {
        "primary": [], "faq_rules": [], "graph_path": [],
    }
    assert captured_turn == {
        "persona_slug": "aurora",
        "persona_id": "p1",
        "lead_ref": 42,
        "messages": messages,
        "response_message_id": "m1",
        "query": "lavagem",
        "projection_nodes": [],
        "limit": 7,
    }
    assert result["mode"] == "exact"
    assert result["limit"] == 7


def test_password_hash_roundtrip_and_rejects_wrong_password():
    encoded = auth_service.hash_password("a-strong-temporary-password")
    assert "a-strong-temporary-password" not in encoded
    assert auth_service.verify_password("a-strong-temporary-password", encoded)
    assert not auth_service.verify_password("wrong", encoded)


def test_auth_persona_catalog_excludes_routing_secrets(monkeypatch):
    monkeypatch.setattr(auth_service.supabase_client, "get_personas", lambda: [{
        "id": "p1",
        "slug": "baita-conveniencia",
        "name": "Baita Conveniencia",
        "active": True,
        "config": {"private": "value"},
        "outbound_webhook_secret": "do-not-expose",
        "inbound_webhook_token": "do-not-expose",
    }])

    assert auth_service.get_auth_personas() == [{
        "id": "p1",
        "slug": "baita-conveniencia",
        "name": "Baita Conveniencia",
        "active": True,
    }]


def test_viewer_role_caps_inconsistent_manage_grant():
    request = request_for(
        {"id": "u1", "role": "viewer", "account_type": "client"},
        [{
            "persona_id": "p1", "persona_slug": "cliente",
            "can_view": True, "can_edit": True, "can_manage": True,
        }],
    )
    resolved = auth_service.assert_persona_capability(
        request, "view", persona_id="p1"
    )
    assert resolved["capabilities"] == {
        "view": True, "edit": False, "manage": False, "manage_members": False,
    }
    with pytest.raises(HTTPException) as error:
        auth_service.assert_persona_capability(request, "edit", persona_id="p1")
    assert error.value.status_code == 403


def test_agency_manager_can_manage_members():
    request = request_for(
        {"id": "u1", "role": "user", "account_type": "agency"},
        [{
            "persona_id": "p1", "persona_slug": "cliente",
            "can_view": True, "can_edit": True, "can_manage": True,
        }],
    )
    resolved = auth_service.assert_persona_capability(
        request, "manage", persona_slug="cliente"
    )
    assert resolved["capabilities"]["manage_members"] is True


def test_evolution_normalizes_lid_alt_and_text_without_guessing_phone():
    provider = EvolutionWhatsAppProvider()
    events = provider.normalize_webhook({
        "event": "messages.upsert",
        "instance": "brain-client",
        "data": {
            "key": {
                "id": "message-1",
                "remoteJid": "12345@lid",
                "remoteJidAlt": "5511999999999@s.whatsapp.net",
                "fromMe": False,
            },
            "message": {"conversation": "Oi"},
        },
    })
    assert events == [{
        "event_type": "MESSAGES_UPSERT",
        "instance": "brain-client",
        "external_message_id": "message-1",
        "external_contact_id": "5511999999999@s.whatsapp.net",
        "remote_jid": "12345@lid",
        "remote_jid_alt": "5511999999999@s.whatsapp.net",
        "from_me": False,
        "status": None,
        "text": "Oi",
        "raw": {
            "key": {
                "id": "message-1",
                "remoteJid": "12345@lid",
                "remoteJidAlt": "5511999999999@s.whatsapp.net",
                "fromMe": False,
            },
            "message": {"conversation": "Oi"},
        },
    }]


def test_evolution_qr_is_exposed_only_as_a_valid_png_data_url():
    encoded = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
    assert EvolutionWhatsAppProvider._normalize_qr(encoded) == (
        f"data:image/png;base64,{encoded}"
    )
    assert EvolutionWhatsAppProvider._normalize_qr(
        f"data:image/jpeg;base64,{encoded}"
    ) is None
    assert EvolutionWhatsAppProvider._normalize_qr("not-a-qr") is None


def test_evolution_waits_for_async_qr_without_persisting_it(monkeypatch):
    provider = EvolutionWhatsAppProvider()
    encoded = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
    responses = iter([
        {"count": 0},
        {"qrcode": {"count": 1}},
        {"base64": f"data:image/png;base64,{encoded}"},
    ])
    calls = []
    monkeypatch.setattr(
        provider,
        "_request",
        lambda method, path: calls.append((method, path)) or next(responses),
    )
    monkeypatch.setattr("services.whatsapp_providers.evolution.time.sleep", lambda _seconds: None)

    result = provider.get_qr_code({"provider_instance_key": "brain-baita"})

    assert result == {
        "status": "qr_ready",
        "qr": {"base64": f"data:image/png;base64,{encoded}"},
    }
    assert calls == [
        ("GET", "/instance/connectionState/brain-baita"),
        ("GET", "/instance/connect/brain-baita"),
        ("GET", "/instance/connect/brain-baita"),
    ]


def test_evolution_provisions_without_starting_qr_handshake(monkeypatch):
    provider = EvolutionWhatsAppProvider()
    captured = {}
    monkeypatch.setattr(
        provider,
        "_request",
        lambda method, path, **kwargs: captured.update({
            "method": method,
            "path": path,
            **kwargs,
        }) or {"ok": True},
    )

    provider.provision_instance(
        "brain-baita",
        "instance-token",
        "http://api:8080/webhook",
        webhook_token="signed-callback",
    )

    assert captured["method"] == "POST"
    assert captured["path"] == "/instance/create"
    assert captured["json"]["qrcode"] is False
    assert captured["json"]["webhook"]["headers"] == {
        "X-Brain-Webhook-Token": "signed-callback"
    }
    assert "signed-callback" not in captured["json"]["webhook"]["url"]


def test_evolution_webhook_target_keeps_signature_out_of_url(monkeypatch):
    monkeypatch.setenv("EVOLUTION_WEBHOOK_HMAC_SECRET", "test-only-webhook-secret")

    url, callback_token = portal._evolution_webhook_target(
        "binding-1",
        "https://api.example.com",
    )

    assert url == "https://api.example.com/webhooks/evolution/binding-1"
    assert callback_token
    assert callback_token not in url
    assert callback_token == evolution_webhook._callback_token("binding-1")


def test_unconfigured_whatsapp_channel_hides_provider_controls_from_client(monkeypatch):
    request = request_for(
        {"id": "u1", "role": "user", "account_type": "client"},
        [{
            "persona_id": "p1", "persona_slug": "baita-conveniencia",
            "can_view": True, "can_edit": True, "can_manage": True,
        }],
    )
    monkeypatch.setattr(
        portal.supabase_client,
        "get_persona",
        lambda _slug: {"id": "p1", "slug": "baita-conveniencia"},
    )
    monkeypatch.setattr(portal, "_binding_for_persona", lambda _persona_id: None)

    assert portal.whatsapp_channel("baita-conveniencia", request) == {
        "configured": False,
        "status": "disabled",
        "can_manage_provider": False,
        "available_providers": [],
    }


def test_provider_switch_requires_explicit_confirmation(monkeypatch):
    request = request_for(
        {"id": "admin-1", "role": "admin", "account_type": "internal"},
        [],
    )
    monkeypatch.setattr(
        portal.supabase_client,
        "get_persona",
        lambda _slug: {"id": "p1", "slug": "baita-conveniencia"},
    )
    with pytest.raises(HTTPException) as error:
        portal.select_whatsapp_provider(
            "baita-conveniencia",
            portal.WhatsAppProviderBody(provider="evolution_baileys", confirmed=False),
            request,
        )
    assert error.value.status_code == 400


def test_client_manager_cannot_switch_provider(monkeypatch):
    request = request_for(
        {"id": "u1", "role": "user", "account_type": "client"},
        [{
            "persona_id": "p1", "persona_slug": "baita-conveniencia",
            "can_view": True, "can_edit": True, "can_manage": True,
        }],
    )
    with pytest.raises(HTTPException) as error:
        portal.select_whatsapp_provider(
            "baita-conveniencia",
            portal.WhatsAppProviderBody(provider="evolution_baileys", confirmed=True),
            request,
        )
    assert error.value.status_code == 403


def test_provider_switch_rolls_back_to_meta_when_evolution_provision_fails(monkeypatch):
    request = request_for(
        {"id": "admin-1", "role": "admin", "account_type": "internal"},
        [],
    )
    meta = {
        "id": "meta-1", "persona_id": "p1", "channel": "whatsapp",
        "provider": "meta_cloud", "active": True,
        "whatsapp_phone_number_id": "configured",
    }
    updates = []
    monkeypatch.setenv("EVOLUTION_ENABLED", "true")
    monkeypatch.setenv("AI_BRAIN_PUBLIC_API_URL", "http://api:8080")
    monkeypatch.setenv("AI_BRAIN_AUTH_SECRET", "test-only-secret")
    monkeypatch.setattr(
        portal.supabase_client,
        "get_persona",
        lambda _slug: {"id": "p1", "slug": "baita-conveniencia"},
    )
    monkeypatch.setattr(portal, "_whatsapp_bindings", lambda _id: [meta])
    monkeypatch.setattr(
        portal.supabase_client,
        "upsert_workflow_binding",
        lambda _payload: {"id": "evo-1", "provider_instance_key": "brain-baita"},
    )
    monkeypatch.setattr(
        portal.supabase_client,
        "update_workflow_binding",
        lambda binding_id, payload: updates.append((binding_id, payload)) or payload,
    )

    class FailingProvider:
        def provision_instance(self, *_args, **_kwargs):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(portal, "get_provider", lambda _name: FailingProvider())
    with pytest.raises(HTTPException) as error:
        portal.select_whatsapp_provider(
            "baita-conveniencia",
            portal.WhatsAppProviderBody(provider="evolution_baileys", confirmed=True),
            request,
        )
    assert error.value.status_code == 502
    assert not any(binding_id == "meta-1" and payload.get("active") is False for binding_id, payload in updates)
    assert ("evo-1", {"active": False, "connection_status": "failed"}) in updates


def test_deterministic_inbound_propagates_evolution_binding(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        whatsapp_dispatch_worker.supabase_client,
        "get_persona_by_id",
        lambda _id: {"id": "p1", "slug": "baita-conveniencia", "process_mode": "internal"},
    )
    monkeypatch.setattr(
        whatsapp_dispatch_worker.supabase_client,
        "get_lead_by_ref",
        lambda _id: {"id": 7, "metadata": {}},
    )
    monkeypatch.setattr(
        whatsapp_dispatch_worker.supabase_client,
        "get_messages",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        whatsapp_dispatch_worker.supabase_client,
        "get_workflow_binding_by_id",
        lambda _id: {
            "id": "evo-1", "persona_id": "p1", "provider": "evolution_baileys",
            "active": True,
            "metadata": {"decision_owner": "deterministic"},
        },
    )
    monkeypatch.setattr(
        whatsapp_dispatch_worker.supabase_client,
        "mark_whatsapp_attempt",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        whatsapp_dispatch_worker.conversation_runtime,
        "execute_pipeline",
        lambda **kwargs: captured.update(kwargs) or {
            "handoff": False, "classifier": {}, "route": "knowledge",
        },
    )
    monkeypatch.setattr(
        whatsapp_dispatch_worker.supabase_client,
        "complete_whatsapp_buffer",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        whatsapp_dispatch_worker.event_emitter,
        "emit",
        lambda *_args, **_kwargs: None,
    )
    WhatsAppDispatchWorker()._dispatch_inbound({
        "id": "buffer-1", "persona_id": "p1", "lead_ref": 7,
        "channel_binding_id": "evo-1", "whatsapp_phone_number_id": None,
        "external_message_id": "wamid-1", "correlation_id": "corr-1",
        "payload": {"text": "Qual o horario?"},
    })
    assert captured["channel_binding_id"] == "evo-1"
    assert captured["phone_number_id"] is None


def test_client_session_is_denied_internal_api_but_allowed_portal(monkeypatch):
    monkeypatch.setattr(auth_service, "get_session_payload", lambda _token: {"sub": "u1"})
    monkeypatch.setattr(auth_service, "get_user_by_id", lambda _id: {
        "id": "u1", "email": "client@example.com", "role": "user",
        "account_type": "client", "must_change_password": False, "is_active": True,
    })
    monkeypatch.setattr(auth_service, "get_user_access", lambda _id: [])
    app = FastAPI()
    app.middleware("http")(auth_middleware)

    @app.get("/knowledge/graph")
    def internal():
        return {"leaked": True}

    @app.get("/portal/ping")
    def portal():
        return {"ok": True}

    client = TestClient(app)
    client.cookies.set(auth_service.SESSION_COOKIE, "signed")
    assert client.get("/knowledge/graph").status_code == 403
    assert client.get("/portal/ping").json() == {"ok": True}


def test_client_session_cannot_call_dashboard_endpoints(monkeypatch):
    monkeypatch.setattr(auth_service, "get_session_payload", lambda _token: {"sub": "u1"})
    monkeypatch.setattr(auth_service, "get_user_by_id", lambda _id: {
        "id": "u1", "email": "client@example.com", "role": "user",
        "account_type": "client", "must_change_password": False, "is_active": True,
    })
    monkeypatch.setattr(auth_service, "get_user_access", lambda _id: [])
    app = FastAPI()
    app.middleware("http")(auth_middleware)

    for route in ("/health/score", "/leads", "/insights"):
        app.add_api_route(route, lambda: {"leaked": True}, methods=["GET"])

    @app.get("/health")
    def health():
        return {"status": "ok"}

    client = TestClient(app)
    client.cookies.set(auth_service.SESSION_COOKIE, "signed")
    assert client.get("/health").json() == {"status": "ok"}
    for route in ("/health/score", "/leads?limit=1000&offset=0", "/insights?status=open"):
        response = client.get(route)
        assert response.status_code == 403
        assert response.json() == {"detail": "Acesso negado."}


def test_client_session_response_declares_safe_portal_home(monkeypatch):
    user = {
        "id": "u1", "email": "client@example.com", "role": "user",
        "account_type": "client", "must_change_password": False, "is_active": True,
    }
    monkeypatch.setattr(auth_service, "get_auth_personas", lambda: [{
        "id": "aurora-id", "slug": "aurora", "name": "Aurora", "active": True,
    }])
    monkeypatch.setattr(auth_service, "get_user_access", lambda _id: [{
        "persona_id": "aurora-id", "persona_slug": "aurora",
        "can_view": True, "can_edit": True, "can_manage": True,
    }])

    session = auth_service.build_session_response(user)

    assert session["navigation"] == {
        "surface": "client_portal",
        "home_url": "/clientes/aurora/mensagens",
    }
    assert session["access_profile"] == "client_manager"


def test_must_change_password_is_warning_and_does_not_block_portal(monkeypatch):
    monkeypatch.setattr(auth_service, "get_session_payload", lambda _token: {"sub": "u1"})
    monkeypatch.setattr(auth_service, "get_user_by_id", lambda _id: {
        "id": "u1", "email": "client@example.com", "role": "user",
        "account_type": "client", "must_change_password": True, "is_active": True,
    })
    monkeypatch.setattr(auth_service, "get_user_access", lambda _id: [])
    app = FastAPI()
    app.middleware("http")(auth_middleware)

    @app.get("/portal/ping")
    def portal():
        return {"ok": True}

    client = TestClient(app)
    client.cookies.set(auth_service.SESSION_COOKIE, "signed")
    assert client.get("/portal/ping").status_code == 200


def test_client_pages_include_authorized_persona_without_channel(monkeypatch):
    request = request_for(
        {"id": "u1", "role": "user", "account_type": "client"},
        [{
            "persona_id": "aurora-id", "persona_slug": "aurora",
            "can_view": True, "can_edit": True, "can_manage": True,
        }],
    )
    monkeypatch.setattr(
        portal.auth_service,
        "get_auth_personas",
        lambda: [{"id": "aurora-id", "slug": "aurora", "name": "Aurora", "active": True}],
    )
    monkeypatch.setattr(portal, "_binding_for_persona", lambda _id: None)

    assert portal.client_pages(request) == [{
        "slug": "aurora",
        "name": "Aurora",
        "url": "/clientes/aurora/mensagens",
        "capabilities": {
            "view": True,
            "edit": True,
            "manage": True,
            "manage_members": False,
        },
        "channel": {
            "configured": False,
            "provider": None,
            "status": "disabled",
        },
    }]


def test_portal_pipeline_uses_persona_labels_and_summary(monkeypatch):
    request = request_for(
        {"id": "u1", "role": "user", "account_type": "client"},
        [{
            "persona_id": "aurora-id", "persona_slug": "aurora",
            "can_view": True, "can_edit": True, "can_manage": True,
        }],
    )
    monkeypatch.setattr(
        portal.supabase_client,
        "get_persona",
        lambda _slug: {
            "id": "aurora-id",
            "slug": "aurora",
            "config": {"portal": {
                "business_model": "appointment",
                "conversion_stage": "fechado",
                "stage_labels": {"fechado": "Agendado"},
            }},
        },
    )
    monkeypatch.setattr(
        portal.supabase_client,
        "get_leads_for_persona_ids",
        lambda *_args, **_kwargs: [
            {"id": 1, "stage": "novo"},
            {"id": 2, "stage": "fechado"},
        ],
    )

    result = portal.pipeline(request, "aurora")
    closed = next(stage for stage in result["stages"] if stage["id"] == "fechado")
    assert closed["label"] == "Agendado"
    assert result["summary"] == {
        "total": 2,
        "conversion_stage": "fechado",
        "converted": 1,
        "open": 1,
    }
    assert result["business_model"] == "appointment"


def test_evolution_pending_ack_is_audit_only(monkeypatch):
    events = []
    monkeypatch.setattr(
        portal.supabase_client,
        "insert_event",
        lambda data, **kwargs: events.append((data, kwargs)),
    )
    portal.supabase_client.update_whatsapp_delivery_by_binding(
        "binding-1", "message-1", "PENDING"
    )

    assert events[0][0]["event_type"] == "whatsapp.delivery_ack_ignored"
    assert events[0][0]["payload"] == {
        "external_message_id": "message-1",
        "status": "PENDING",
    }


class _FakeUpdateResult:
    def __init__(self, data):
        self.data = data


class _FakeUpdateQuery:
    def __init__(self, table_rows, payload):
        self._rows = table_rows
        self._payload = payload
        self._filters: dict = {}

    def eq(self, key, value):
        self._filters[key] = value
        return self

    def execute(self):
        row = dict(self._rows[0])
        if all(row.get(k) == v for k, v in self._filters.items()):
            row.update(self._payload)
            self._rows[0] = row
            return _FakeUpdateResult([row])
        return _FakeUpdateResult([])


class _FakeLeadsTable:
    def __init__(self, rows):
        self._rows = rows

    def update(self, payload):
        return _FakeUpdateQuery(self._rows, payload)


class _FakeLeadsClient:
    def __init__(self, rows):
        self._rows = rows

    def table(self, _name):
        return _FakeLeadsTable(self._rows)


def test_portal_update_lead_merges_commercial_note_into_appointment_request(monkeypatch):
    """A client-portal edit must land the same way an admin edit does: the
    commercial_note display mirror AND conversation_state.appointment_
    request (the AI's actual working memory) both get the new value, via
    the shared supabase_client.merge_commercial_note helper.
    """
    request = request_for(
        {"id": "u1", "role": "user", "account_type": "client"},
        [{
            "persona_id": "p1", "persona_slug": "aurora",
            "can_view": True, "can_edit": True, "can_manage": False,
        }],
    )
    lead_row = {
        "id": 29,
        "persona_id": "p1",
        "metadata": {
            "conversation_state": {
                "missing_fields": ["servico", "modelo_veiculo"],
                "appointment_request": {},
            },
        },
    }
    monkeypatch.setattr(
        portal.supabase_client, "get_persona", lambda _slug: {"id": "p1", "slug": "aurora"},
    )
    monkeypatch.setattr(
        portal.supabase_client, "get_lead_by_ref", lambda _ref: lead_row,
    )
    rows = [dict(lead_row)]
    monkeypatch.setattr(portal.supabase_client, "get_client", lambda: _FakeLeadsClient(rows))

    body = portal.LeadPatchBody(commercial_note={"servico": "chapeacao"})
    result = portal.update_lead(29, body, request, persona_slug="aurora")

    state = result["metadata"]["conversation_state"]
    assert state["appointment_request"]["servico"] == "chapeacao"
    assert state["missing_fields"] == ["modelo_veiculo"]
    assert result["metadata"]["commercial_note"]["servico"] == "chapeacao"


def test_portal_has_no_persona_wide_automation_control():
    paths = {route.path for route in portal.router.routes}
    assert not any(path.endswith("/automation") for path in paths)


def test_portal_update_lead_still_supports_interesse_produto(monkeypatch):
    request = request_for(
        {"id": "u1", "role": "user", "account_type": "client"},
        [{
            "persona_id": "p1", "persona_slug": "aurora",
            "can_view": True, "can_edit": True, "can_manage": False,
        }],
    )
    lead_row = {"id": 29, "persona_id": "p1", "metadata": {}}
    monkeypatch.setattr(
        portal.supabase_client, "get_persona", lambda _slug: {"id": "p1", "slug": "aurora"},
    )
    monkeypatch.setattr(portal.supabase_client, "get_lead_by_ref", lambda _ref: lead_row)
    rows = [dict(lead_row)]
    monkeypatch.setattr(portal.supabase_client, "get_client", lambda: _FakeLeadsClient(rows))

    body = portal.LeadPatchBody(interesse_produto="chapeacao")
    result = portal.update_lead(29, body, request, persona_slug="aurora")

    assert result["interesse_produto"] == "chapeacao"

