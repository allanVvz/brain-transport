"""The media ingest worker must always release the conversation.

The worker is the slow half of media ingest: it downloads the bytes, reads
them, writes the extracted text back into the buffer and lifts the dispatch
hold. The property that matters most is that *every* path releases the hold —
a failed download or a failed transcription must degrade to the placeholder,
never leave the customer waiting on a stuck queue or produce an empty user
turn for the agent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "api"
for path in (API_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# ── descriptor text ──────────────────────────────────────────────────────

def test_transcribed_audio_is_labelled_for_the_agent():
    """The agent must never mistake a transcription for typed text."""
    from services.media_ingest import describe

    text, status = describe(
        {"kind": "audio", "voice_note": True},
        {"transcript": "quanto custa a juliet preta?"},
    )
    assert text == "[audio do cliente]: quanto custa a juliet preta?"
    assert status == "completed"


def test_failed_transcription_falls_back_to_the_placeholder():
    from services.media_ingest import describe

    text, status = describe({"kind": "audio", "voice_note": True}, {"transcript": ""})
    assert text == "[o cliente enviou um audio]"
    assert status == "failed"
    # The one thing that must never happen.
    assert text.strip() != ""


def test_captioned_image_keeps_the_caption_and_adds_the_reading():
    from services.media_ingest import describe

    text, status = describe(
        {"kind": "image", "caption": "essa aqui"},
        {"visual_summary": "oculos juliet preto"},
    )
    assert text == "essa aqui\n[imagem enviada pelo cliente: oculos juliet preto]"
    assert status == "completed"


def test_image_without_reading_still_keeps_the_caption():
    from services.media_ingest import describe

    text, status = describe({"kind": "image", "caption": "essa aqui"}, {})
    assert text == "essa aqui"
    assert status == "failed"


def test_document_excerpt_is_capped():
    from services.media_ingest import describe

    body = "\n".join(f"linha {i}" for i in range(50))
    text, status = describe({"kind": "document", "filename": "pedido.pdf"}, {"extracted_text": body})
    assert text.startswith("[documento: pedido.pdf]")
    assert status == "completed"
    # 1 header line + 20 content lines.
    assert len(text.splitlines()) == 21


@pytest.mark.parametrize("kind", ["audio", "image", "video", "document"])
def test_every_kind_has_a_non_empty_placeholder(kind):
    from services.media_ingest import placeholder_text

    assert placeholder_text({"kind": kind}).strip()


# ── worker failure handling ──────────────────────────────────────────────

@pytest.fixture
def worker(monkeypatch):
    from workers import media_ingest_worker as mod

    return mod


def test_failure_marks_the_asset_and_releases_the_hold(worker, monkeypatch):
    """A file that cannot be fetched must not block the reply."""
    updates = []
    resolved = []
    monkeypatch.setattr(
        worker.supabase_client, "update_asset",
        lambda asset_id, patch: updates.append((asset_id, patch)),
    )
    monkeypatch.setattr(
        worker.supabase_client, "resolve_media_buffer",
        lambda buffer_id, text, **kw: resolved.append((buffer_id, text, kw)) or {"resolved": True},
    )

    instance = worker.MediaIngestWorker()
    instance._fail(
        {
            "id": "asset-1",
            "metadata": {"media": {"kind": "audio", "voice_note": True, "buffer_id": "buf-9"}},
        },
        "provider returned no bytes",
    )

    assert updates[0][1]["status"] == "failed"
    assert updates[0][1]["metadata"]["reading_error"] == "provider returned no bytes"

    buffer_id, text, kwargs = resolved[0]
    assert buffer_id == "buf-9"
    assert text == "[o cliente enviou um audio]"
    assert kwargs["reading_status"] == "failed"


def test_one_bad_asset_does_not_stall_the_queue(worker, monkeypatch):
    """A single unreadable file must not stop the others from being read."""
    assets = [
        {"id": "bad", "metadata": {"media": {"kind": "audio", "buffer_id": "buf-bad"}}},
        {"id": "good", "metadata": {"media": {"kind": "audio", "buffer_id": "buf-good"}}},
    ]
    monkeypatch.setattr(worker.supabase_client, "claim_pending_media_assets", lambda limit=5: assets)

    seen = []
    failed = []

    def _process(asset):
        seen.append(asset["id"])
        if asset["id"] == "bad":
            raise RuntimeError("boom")

    monkeypatch.setattr(worker.MediaIngestWorker, "_process", staticmethod(_process))
    monkeypatch.setattr(
        worker.MediaIngestWorker, "_fail",
        lambda self, asset, error: failed.append(asset["id"]),
    )
    monkeypatch.setattr(worker.sre_logger, "error", lambda *a, **k: None)

    worker.MediaIngestWorker()._run_cycle()

    assert seen == ["bad", "good"]
    assert failed == ["bad"]


def test_message_projection_failure_does_not_fail_media_ingest(worker, monkeypatch):
    instance = worker.MediaIngestWorker()
    monkeypatch.setattr(
        worker.supabase_client, "link_inbound_media_asset_to_message",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("projection unavailable")),
    )
    monkeypatch.setattr(instance, "_is_stale", lambda _asset: True)
    failed = []
    monkeypatch.setattr(instance, "_fail", lambda asset, error: failed.append((asset["id"], error)))

    instance._process({
        "id": "asset-1", "message_id": 2378,
        "metadata": {"media": {"kind": "image"}},
    })

    assert failed == [("asset-1", "reading timed out")]


def test_unknown_fetch_strategy_is_rejected(worker):
    instance = worker.MediaIngestWorker()
    with pytest.raises(RuntimeError, match="unsupported media fetch strategy"):
        instance._download({"provider": "mock"}, {"fetch_ref": {"strategy": "carrier_pigeon"}})
