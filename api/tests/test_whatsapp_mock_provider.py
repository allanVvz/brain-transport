"""MockWhatsAppProvider is exercised for real by Phase 6's E2E stack, but
its own correctness (Protocol compliance, no accidental network calls,
in-memory bookkeeping) is cheap to pin down here without Docker."""
import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from services.whatsapp_providers import get_provider
from services.whatsapp_providers.base import WhatsAppProvider
from services.whatsapp_providers.mock import MockWhatsAppProvider


def setup_function(_fn):
    MockWhatsAppProvider.reset()


def test_registry_returns_mock_provider():
    provider = get_provider("mock")
    assert isinstance(provider, MockWhatsAppProvider)


def test_implements_every_protocol_method():
    protocol_methods = (
        "provision_instance", "get_connection_status", "get_qr_code", "send_text",
        "send_media", "restart_instance", "logout_instance", "normalize_webhook",
    )
    for name in protocol_methods:
        assert hasattr(WhatsAppProvider, name), f"WhatsAppProvider Protocol drifted: no {name}"
        assert callable(getattr(MockWhatsAppProvider, name, None)), f"missing {name}"


def test_send_text_records_and_returns_external_id():
    provider = MockWhatsAppProvider()
    binding = {"id": "binding-1", "provider_instance_key": "instance-1"}
    result = provider.send_text(binding, "5511999999999", "oi")

    assert result["id"].startswith("mock-")
    assert result["messageId"] == result["id"]
    assert len(MockWhatsAppProvider.sent) == 1
    assert MockWhatsAppProvider.sent[0]["recipient"] == "5511999999999"
    assert MockWhatsAppProvider.sent[0]["content"] == {"kind": "text", "text": "oi"}


def test_send_is_visible_across_separate_instances():
    """registry.get_provider() constructs a new instance per call site
    (worker vs. test assertion) â€” the log must be shared, not per-instance,
    or nothing would ever be observable from the test side."""
    get_provider("mock").send_text({"id": "b1"}, "5511888888888", "hello")
    observed = MockWhatsAppProvider().sent
    assert len(observed) == 1
    assert observed[0]["recipient"] == "5511888888888"


def test_send_media_records_media_payload():
    provider = MockWhatsAppProvider()
    binding = {"id": "binding-1", "provider_instance_key": "instance-1"}
    media = {"mediatype": "image", "media": "https://example.com/x.png"}
    result = provider.send_media(binding, "5511999999999", media)

    assert result["id"]
    assert MockWhatsAppProvider.sent[0]["content"] == {"kind": "media", "media": media}


def test_reset_clears_sent_log_and_instances():
    provider = MockWhatsAppProvider()
    provider.provision_instance("inst-1", "token", "https://hook", webhook_token="tok")
    provider.send_text({"id": "b1", "provider_instance_key": "inst-1"}, "5511999999999", "oi")
    assert MockWhatsAppProvider.sent
    assert MockWhatsAppProvider.instances

    MockWhatsAppProvider.reset()
    assert MockWhatsAppProvider.sent == []
    assert MockWhatsAppProvider.instances == {}


def test_qr_code_never_returns_a_fake_image_and_marks_connected():
    provider = MockWhatsAppProvider()
    binding = {"id": "binding-1", "provider_instance_key": "instance-1"}
    result = provider.get_qr_code(binding)
    assert result == {"status": "connected", "qr": None}
    assert provider.get_connection_status(binding)["instance"]["state"] == "connected"


def test_logout_removes_instance_state():
    provider = MockWhatsAppProvider()
    binding = {"id": "binding-1", "provider_instance_key": "instance-1"}
    provider.get_qr_code(binding)
    assert "instance-1" in MockWhatsAppProvider.instances

    provider.logout_instance(binding)
    assert "instance-1" not in MockWhatsAppProvider.instances


def test_normalize_webhook_mirrors_evolution_shape():
    provider = MockWhatsAppProvider()
    payload = {
        "event": "messages.upsert",
        "instance": "inst-1",
        "data": {
            "key": {"id": "wamid.abc", "remoteJid": "5511999999999@s.whatsapp.net", "fromMe": False},
            "message": {"conversation": "oi"},
        },
    }
    events = provider.normalize_webhook(payload)
    assert len(events) == 1
    event = events[0]
    assert event["event_type"] == "MESSAGES.UPSERT"
    assert event["external_message_id"] == "wamid.abc"
    assert event["external_contact_id"] == "5511999999999@s.whatsapp.net"
    assert event["text"] == "oi"
    assert event["from_me"] is False

