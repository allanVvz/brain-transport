from __future__ import annotations

from .evolution import EvolutionWhatsAppProvider
from .meta import MetaWhatsAppProvider
from .mock import MockWhatsAppProvider


def get_provider(name: str):
    if name == "evolution_baileys":
        return EvolutionWhatsAppProvider()
    if name in {"meta_cloud", "", None}:
        return MetaWhatsAppProvider()
    if name == "mock":
        return MockWhatsAppProvider()
    raise ValueError(f"Unsupported WhatsApp provider: {name}")
