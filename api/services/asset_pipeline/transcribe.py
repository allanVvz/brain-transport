# -*- coding: utf-8 -*-
"""Speech-to-text for WhatsApp voice notes, via faster-whisper.

Runs locally inside the API container on purpose: a customer's voice note is
personal data, and a local model keeps it from leaving the VPS while also
removing any per-minute cost.

The model is loaded once per process and cached — construction is the
expensive part (weights load + warm-up), transcription itself is cheap by
comparison. `small` with int8 quantization runs near real time on CPU, which
is what `media_ingest.MEDIA_HOLD_SECONDS` is sized against.

ffmpeg must be present in the image: WhatsApp sends OGG/Opus, which
faster-whisper decodes through it.
"""
from __future__ import annotations

import logging
import os
import tempfile
import threading
from typing import Optional

from .schemas import TranscriptionResult

logger = logging.getLogger("asset_pipeline.transcribe")

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
# Set by the Dockerfile to the weights baked in at build time. Left unset in
# local dev, where faster-whisper falls back to the HuggingFace cache.
WHISPER_CACHE_PATH = os.environ.get("WHISPER_CACHE_PATH") or None
# WhatsApp caps voice notes well below this; the guard is against a malicious
# or corrupted upload pinning a worker thread for minutes.
MAX_AUDIO_SECONDS = int(os.environ.get("WHISPER_MAX_AUDIO_SECONDS", "600"))

_model = None
_model_lock = threading.Lock()


def _load_model():
    """Load (and memoize) the whisper model.

    Double-checked locking: several worker threads may hit this at once on the
    first audio message after a deploy, and loading the weights twice would
    double the memory footprint.
    """
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        from faster_whisper import WhisperModel

        logger.info(
            "loading faster-whisper model=%s device=%s compute=%s",
            WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
        )
        _model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            download_root=WHISPER_CACHE_PATH,
        )
        return _model


def available() -> bool:
    """Whether transcription can run in this process.

    Checked before claiming work so a deployment without the dependency
    degrades to the placeholder text instead of failing every audio message.
    """
    try:
        import faster_whisper  # noqa: F401
    except Exception:
        return False
    return True


def run(file_bytes: bytes, *, suffix: str = ".ogg", language: Optional[str] = "pt") -> TranscriptionResult:
    """Transcribe audio bytes.

    ``language`` defaults to Portuguese: forcing it is both faster and more
    accurate than autodetection on the short, noisy clips typical of voice
    notes. Pass ``None`` to let the model detect.
    """
    if not file_bytes:
        return TranscriptionResult(
            transcript="", language=None, duration_seconds=None,
            model_used=WHISPER_MODEL, error="empty audio payload",
        )
    if not available():
        return TranscriptionResult(
            transcript="", language=None, duration_seconds=None,
            model_used=WHISPER_MODEL, error="faster-whisper is not installed",
        )

    # faster-whisper reads from a path (it shells out to ffmpeg for decoding),
    # so the bytes have to land on disk first.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        model = _load_model()
        segments, info = model.transcribe(
            tmp_path,
            language=language,
            vad_filter=True,          # drop silence — voice notes start/end with it
            beam_size=1,              # greedy: the quality gain from beams is not
                                      # worth the latency inside a dispatch hold
        )
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        if duration > MAX_AUDIO_SECONDS:
            return TranscriptionResult(
                transcript="", language=getattr(info, "language", None),
                duration_seconds=duration, model_used=WHISPER_MODEL,
                error=f"audio longer than {MAX_AUDIO_SECONDS}s",
            )
        text = " ".join((segment.text or "").strip() for segment in segments).strip()
        return TranscriptionResult(
            transcript=text,
            language=getattr(info, "language", None),
            duration_seconds=duration,
            model_used=WHISPER_MODEL,
        )
    except Exception as exc:
        logger.warning("transcription failed: %s", exc)
        return TranscriptionResult(
            transcript="", language=None, duration_seconds=None,
            model_used=WHISPER_MODEL, error=str(exc)[:300],
        )
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
