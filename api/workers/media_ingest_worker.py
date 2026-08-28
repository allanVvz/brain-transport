# -*- coding: utf-8 -*-
"""Download, read and publish inbound WhatsApp media.

The webhook registers the attachment and holds dispatch
(`media_ingest.MEDIA_HOLD_SECONDS`); this worker does the slow half â€” fetch the
bytes from the provider, store them in the private bucket, run the asset
pipeline, then write the extracted text back into the buffer and release the
hold.

Failure is always survivable: an asset that cannot be fetched or read is marked
`failed` and the conversation proceeds with the placeholder text the webhook
already wrote. The message is never lost and the customer is never left waiting
on a stuck queue.
"""
from __future__ import annotations

import logging
import os
import socket
from datetime import datetime, timedelta, timezone

from services import knowledge_graph, media_ingest, sre_logger, supabase_client
from services.asset_pipeline import AssetPipelineContext, run_pipeline
from services.whatsapp_providers import get_provider
from workers.base_worker import BaseWorker

logger = logging.getLogger("workers.media_ingest")

# An asset still in `reading` past this is treated as abandoned (worker killed
# mid-cycle). It gets marked failed so it stops being claimed forever.
STALE_READING_SECONDS = int(os.environ.get("WHATSAPP_MEDIA_STALE_SECONDS", "600"))


class MediaIngestWorker(BaseWorker):
    """Turn a registered WhatsApp attachment into a readable asset."""

    name = "MediaIngestWorker"
    # Tighter than most workers: the dispatch hold is counting down, so a file
    # sitting unclaimed directly delays the reply.
    interval = 3

    def __init__(self) -> None:
        super().__init__()
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}:media-ingest"

    def _run_cycle(self) -> None:
        for asset in supabase_client.claim_pending_media_assets(limit=5):
            try:
                self._process(asset)
            except Exception as exc:
                # One bad file must not stall the rest of the queue.
                sre_logger.error(
                    self.name,
                    f"media ingest failed asset={asset.get('id')}",
                    exc,
                )
                self._fail(asset, str(exc)[:300])

    # â”€â”€ one asset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _process(self, asset: dict) -> None:
        asset_id = str(asset.get("id") or "")
        metadata = asset.get("metadata") or {}
        descriptor = metadata.get("media") or {}
        if asset.get("message_id") and asset_id:
            try:
                supabase_client.link_inbound_media_asset_to_message(
                    int(asset["message_id"]), asset_id,
                )
            except Exception as exc:
                logger.warning(
                    "message media projection failed asset=%s message=%s: %s",
                    asset_id, asset.get("message_id"), exc,
                )
        if self._is_stale(asset):
            self._fail(asset, "reading timed out")
            return

        binding = self._binding(descriptor)
        if not binding:
            self._fail(asset, "channel binding not found")
            return

        content = self._download(binding, descriptor)
        if not content:
            self._fail(asset, "provider returned no bytes")
            return

        # Store first, read second: even if the pipeline fails, the operator
        # can still open the file in the CRM.
        storage_path = self._store(asset, descriptor, content)

        bundle = run_pipeline(
            content,
            AssetPipelineContext(
                persona_id=asset.get("persona_id"),
                persona_slug=None,
                upload_context="whatsapp_inbound",
                original_filename=asset.get("original_filename") or "midia",
                mime=asset.get("mime_type") or descriptor.get("mime"),
            ),
        )
        for row in bundle.rows_to_persist:
            try:
                supabase_client.insert_asset_reading({**row, "asset_id": asset_id})
            except Exception as exc:
                logger.warning("asset_reading insert failed asset=%s: %s", asset_id, exc)

        text, reading_status = media_ingest.describe(
            descriptor,
            {
                "transcript": bundle.transcription.transcript if bundle.transcription else "",
                "extracted_text": bundle.extracted_text,
                "visual_summary": bundle.visual_summary,
            },
        )

        supabase_client.update_asset(asset_id, {
            "status": "ready",
            "storage_path": storage_path,
            "file_size": len(content),
            "metadata": {
                **metadata,
                "reading_status": reading_status,
                "extracted_text": (bundle.extracted_text or "")[:8000],
                "visual_summary": bundle.visual_summary or "",
                "descriptor_text": text,
            },
        })

        # Release the dispatch hold. Doing this last means a crash anywhere
        # above leaves the buffer held until it expires naturally, which
        # degrades to the placeholder rather than to silence.
        buffer_id = descriptor.get("buffer_id")
        if buffer_id:
            result = supabase_client.resolve_media_buffer(
                str(buffer_id), text, reading_status=reading_status,
            )
            if not result.get("resolved"):
                logger.info(
                    "media buffer not released asset=%s reason=%s",
                    asset_id, result.get("reason"),
                )

        self._attach_to_graph(asset_id)

    # â”€â”€ steps â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _is_stale(self, asset: dict) -> bool:
        created = asset.get("created_at")
        if not created:
            return False
        try:
            started = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
        except ValueError:
            return False
        return datetime.now(timezone.utc) - started > timedelta(seconds=STALE_READING_SECONDS)

    def _binding(self, descriptor: dict) -> dict | None:
        binding_id = descriptor.get("binding_id")
        if not binding_id:
            return None
        return supabase_client.get_workflow_binding_by_id(str(binding_id))

    def _download(self, binding: dict, descriptor: dict) -> bytes:
        """Fetch the bytes using whichever strategy the provider recorded."""
        provider = get_provider(binding.get("provider"))
        fetch_ref = descriptor.get("fetch_ref") or {}
        strategy = fetch_ref.get("strategy")
        if strategy == "meta_graph":
            return provider.get_media_bytes(binding, str(fetch_ref.get("media_id") or ""))
        if strategy == "evolution_base64":
            return provider.get_media_base64(binding, fetch_ref.get("message_key") or {})
        raise RuntimeError(f"unsupported media fetch strategy: {strategy!r}")

    def _store(self, asset: dict, descriptor: dict, content: bytes) -> str:
        """Upload to the private bucket under a persona/lead-scoped key."""
        persona_id = asset.get("persona_id")
        lead_id = asset.get("lead_id")
        name = knowledge_graph._slugify(
            (asset.get("original_filename") or descriptor.get("kind") or "midia").rsplit(".", 1)[0]
        )[:60] or "midia"
        suffix = _extension(descriptor, asset.get("original_filename"))
        path = f"{persona_id}/{lead_id}/{asset.get('id')}-{name}{suffix}"
        supabase_client.upload_to_storage(
            supabase_client.WHATSAPP_MEDIA_BUCKET,
            path,
            content,
            asset.get("mime_type") or descriptor.get("mime") or "application/octet-stream",
        )
        return path

    def _attach_to_graph(self, asset_id: str) -> None:
        """Hang the asset under conversation â†’ audience â†’ campaign.

        Best-effort on purpose: the file is already stored and the reply is
        already unblocked, so a graph publish failure must not fail ingest.
        """
        try:
            from services import conversation_graph

            conversation_graph.attach_inbound_asset(asset_id)
        except Exception as exc:
            logger.warning("graph attach skipped asset=%s: %s", asset_id, exc)

    def _fail(self, asset: dict, error: str) -> None:
        """Mark the asset failed and release the hold with the placeholder."""
        metadata = asset.get("metadata") or {}
        descriptor = metadata.get("media") or {}
        try:
            supabase_client.update_asset(str(asset.get("id")), {
                "status": "failed",
                "metadata": {**metadata, "reading_status": "failed", "reading_error": error},
            })
        except Exception as exc:
            logger.warning("could not mark asset failed=%s: %s", asset.get("id"), exc)
        buffer_id = descriptor.get("buffer_id")
        if not buffer_id:
            return
        try:
            supabase_client.resolve_media_buffer(
                str(buffer_id),
                media_ingest.placeholder_text(descriptor),
                reading_status="failed",
            )
        except Exception as exc:
            logger.warning("could not release held buffer=%s: %s", buffer_id, exc)


def _extension(descriptor: dict, filename: str | None) -> str:
    if filename and "." in filename:
        return "." + filename.rsplit(".", 1)[-1].lower()
    mime = (descriptor.get("mime") or "").split(";")[0].strip().lower()
    return {
        "audio/ogg": ".ogg", "audio/opus": ".opus", "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a", "audio/amr": ".amr",
        "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
        "video/mp4": ".mp4", "application/pdf": ".pdf",
    }.get(mime, "")

