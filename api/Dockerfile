# Multi-stage production image for the AI Brain backend.
# The same image runs the API (gunicorn) and the workers (workers.runner);
# the compose file overrides the command for the workers service.
# Build context is the repository root. Only api/ and the runtime SDR catalog
# are copied, so image contents stay tied to the release SHA.

# ── Builder ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ARG PIP_TRUSTED_HOST=""
ARG GRAPH_RAG_LOCAL_EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ARG WHISPER_MODEL="small"

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY api/requirements.txt ./requirements.txt
RUN PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST}" pip install -r requirements.txt
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='${GRAPH_RAG_LOCAL_EMBEDDING_MODEL}', cache_dir='/opt/venv/fastembed-cache')"
# Bake the whisper weights into the image. Downloading them on first use would
# make the first voice note after a deploy time out against the dispatch hold,
# and would fail outright on a VPS with restricted egress.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8', download_root='/opt/venv/whisper-cache')"

# ── Runtime ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    FASTEMBED_CACHE_PATH=/opt/venv/fastembed-cache \
    WHISPER_CACHE_PATH=/opt/venv/whisper-cache \
    PATH="/opt/venv/bin:$PATH"

# ffmpeg decodes WhatsApp voice notes (OGG/Opus) for faster-whisper; without
# it every audio message falls back to the "[o cliente enviou um audio]"
# placeholder instead of a transcription.
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user.
RUN useradd --create-home --uid 10001 appuser

# Writable Obsidian vault dir for the kb-intake/Sofia save pipeline (VAULT_PATH).
# A fresh named volume mounted here inherits this ownership on first init.
RUN mkdir -p /data/vault && chown -R appuser:appuser /data

COPY --from=builder /opt/venv /opt/venv
RUN chown -R appuser:appuser /opt/venv/fastembed-cache

WORKDIR /app
COPY --chown=appuser:appuser api/ /app/
COPY --chown=appuser:appuser docs/sdr/ /docs/sdr/

USER appuser

EXPOSE 8080

# Lightweight liveness probe against the API's /health (no curl on slim).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/health',timeout=4).status==200 else 1)"

CMD ["sh", "-c", "exec gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:${PORT} --workers ${GUNICORN_WORKERS:-1} --timeout ${GUNICORN_TIMEOUT:-120} --max-requests ${GUNICORN_MAX_REQUESTS:-1000} --max-requests-jitter ${GUNICORN_MAX_REQUESTS_JITTER:-100}"]
