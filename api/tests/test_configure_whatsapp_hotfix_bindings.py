"""Regression test for the 2026-08-01 (and 2026-08-02) deploy-time binding
revert.

configure_whatsapp_hotfix_bindings.py runs on every deploy
(ops/vps/deploy.sh's configure_whatsapp_bindings step) and used to
unconditionally force the Evolution binding back to
decision_owner=deterministic â€” silently undoing an intentional activation
of Aurora's n8n_agents flow on the very next deploy (fixed 2026-08-01).
The Meta binding kept the same unconditional reset with no exception at
all, on the assumption "Meta never uses the agentic engine" â€” that stopped
being true the moment the settings UI let an operator switch any persona
to n8n_agents, and confirmed live 2026-08-02: switching baita-conveniencia
(meta_cloud) to n8n_agents got silently reverted by the very next deploy.
Both bindings are now treated symmetrically: reset only when NOT already a
complete, valid n8n_agents configuration.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from scripts import configure_whatsapp_hotfix_bindings as hotfix
from scripts.configure_whatsapp_hotfix_bindings import _is_complete_n8n_agents_binding


def test_complete_n8n_agents_binding_is_recognized():
    binding = {
        "n8n_workflow_id": "wf-123",
        "metadata": {
            "decision_owner": "n8n_agents",
            "conversation_webhook_url": "http://n8n:5678/webhook/aurora/conversation",
        },
    }
    assert _is_complete_n8n_agents_binding(binding) is True


def test_deterministic_binding_is_not_preserved():
    binding = {"n8n_workflow_id": None, "metadata": {"decision_owner": "deterministic"}}
    assert _is_complete_n8n_agents_binding(binding) is False


def test_half_configured_n8n_agents_binding_is_not_preserved():
    """decision_owner flipped to n8n_agents but missing a workflow id or
    webhook is exactly as dangerous as before â€” must still be reset."""
    missing_workflow = {
        "n8n_workflow_id": None,
        "metadata": {
            "decision_owner": "n8n_agents",
            "conversation_webhook_url": "http://n8n:5678/webhook/aurora/conversation",
        },
    }
    assert _is_complete_n8n_agents_binding(missing_workflow) is False

    missing_webhook = {
        "n8n_workflow_id": "wf-123",
        "metadata": {"decision_owner": "n8n_agents", "conversation_webhook_url": ""},
    }
    assert _is_complete_n8n_agents_binding(missing_webhook) is False


def test_empty_binding_is_not_preserved():
    assert _is_complete_n8n_agents_binding({}) is False


def test_main_preserves_a_complete_meta_n8n_agents_binding(monkeypatch):
    """Regression test for the exact bug found live 2026-08-02: the Meta
    binding used to have zero preservation logic â€” the very next deploy
    after an operator switched baita-conveniencia to n8n_agents silently
    reverted it back to deterministic. Confirms the Meta binding now gets
    the same treatment as Evolution: left alone when complete and valid.
    """
    meta_persona = {"id": "persona-meta", "slug": "baita-conveniencia"}
    evolution_persona = {"id": "persona-evo", "slug": "aurora"}
    meta_binding = {
        "id": "binding-meta",
        "provider": "meta_cloud",
        "active": True,
        "n8n_workflow_id": "wf-meta",
        "provider_secret_ciphertext": "encrypted-token",
        "metadata": {
            "decision_owner": "n8n_agents",
            "conversation_webhook_url": "http://n8n:5678/webhook/baita-conveniencia/conversation",
        },
    }
    evolution_binding = {
        "id": "binding-evo",
        "provider": "evolution_baileys",
        "active": True,
        "n8n_workflow_id": None,
        "provider_secret_ciphertext": "encrypted-evo",
        "metadata": {"decision_owner": "deterministic"},
    }
    updates: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        hotfix.supabase_client,
        "get_persona",
        lambda slug: meta_persona if slug == "baita-conveniencia" else evolution_persona,
    )
    monkeypatch.setattr(
        hotfix.supabase_client,
        "get_workflow_bindings",
        lambda persona_id: (
            [meta_binding] if persona_id == "persona-meta" else [evolution_binding]
        ),
    )
    monkeypatch.setattr(
        hotfix.supabase_client,
        "update_workflow_binding",
        lambda binding_id, update: updates.append((binding_id, update)),
    )
    monkeypatch.setattr(hotfix.supabase_client, "insert_event", lambda *a, **k: None)
    monkeypatch.setattr(hotfix.secret_store, "decrypt_secret", lambda _ct: "same-token")
    monkeypatch.setattr(hotfix.secret_store, "encrypt_secret", lambda _value: "encrypted-token")
    monkeypatch.setenv("META_WHATSAPP_ACCESS_TOKEN", "same-token")
    monkeypatch.setattr(
        sys, "argv",
        [
            "configure_whatsapp_hotfix_bindings.py",
            "--meta-persona", "baita-conveniencia",
            "--evolution-persona", "aurora",
            "--apply",
        ],
    )

    hotfix.main()

    meta_updates = [update for binding_id, update in updates if binding_id == "binding-meta"]
    assert len(meta_updates) == 1
    # Preserved: no reset of channel/provider/n8n_workflow_id/metadata â€”
    # only the (already-matching) credential ciphertext gets touched.
    assert "metadata" not in meta_updates[0]
    assert "n8n_workflow_id" not in meta_updates[0]

    evo_updates = [update for binding_id, update in updates if binding_id == "binding-evo"]
    assert len(evo_updates) == 1
    assert evo_updates[0]["metadata"]["decision_owner"] == "deterministic"

