# -*- coding: utf-8 -*-
"""Video mock — no real video processing. Returns canned reading."""
from __future__ import annotations


MOCK_PAYLOAD = {
    "reading_status": "mocked",
    "video_reading_mocked": True,
    "visual_summary": "Video recebido. Leitura automatica ainda em modo simulado.",
    "frames_extracted": [],
}


def run() -> dict:
    return dict(MOCK_PAYLOAD)
