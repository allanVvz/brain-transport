from __future__ import annotations

import sys
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1] / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from services import supabase_client


def test_persisted_asset_hydrates_message_preview_without_denormalized_metadata():
    rows = [{
        "id": 2378,
        "metadata": {"media": {"kind": "image", "mime": "image/jpeg"}},
    }]
    assets = [{"id": "asset-1", "message_id": 2378, "status": "ready"}]

    projected = supabase_client._project_message_media_asset_refs(rows, assets)

    assert projected[0]["metadata"]["asset_id"] == "asset-1"
    assert projected[0]["metadata"]["media_asset_status"] == "ready"
    assert "asset_id" not in rows[0]["metadata"]
