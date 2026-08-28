"""Per-model token pricing used to compute cost_usd on observability events.

No pricing figures existed anywhere in this repo before this file -- the
cost investigation that led to this observability work found the codebase
had zero real usage/cost tracking, only pre-call budget *estimates*. The
table below is intentionally left unconfirmed (confirmed=False, prices
None) until someone checks it against the actual DeepSeek billing
dashboard/contract for the model names really in use. estimate_cost_usd()
returns None rather than a fabricated number for any unconfirmed or
unknown model, so a missing price is visibly absent in a dashboard (a gap
to fill in) instead of silently wrong.

Tiers are effective-dated so a past event's cost stays what it was
actually billed at even if pricing changes later -- estimate_cost_usd()
picks the tier in effect at `at` (default: now), not the newest one.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class PriceTier:
    effective_from: datetime
    confirmed: bool
    # USD per 1,000,000 tokens. None when not yet confirmed.
    input_per_million: float | None
    output_per_million: float | None
    cached_input_per_million: float | None = None


# Keyed by the exact `model` string as it appears in token_usage (the
# DeepSeek response's own model id, forwarded verbatim by n8n -- see
# aurora-conversation.json / persona-conversation-template.json).
PRICING: dict[str, list[PriceTier]] = {
    "deepseek-v4-flash": [
        PriceTier(
            effective_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
            confirmed=False,
            input_per_million=None,
            output_per_million=None,
            cached_input_per_million=None,
        ),
    ],
}


def _tier_for(model: str, at: datetime) -> PriceTier | None:
    tiers = PRICING.get(model)
    if not tiers:
        return None
    applicable = [tier for tier in tiers if tier.effective_from <= at]
    if not applicable:
        return None
    return max(applicable, key=lambda tier: tier.effective_from)


def estimate_cost_usd(
    model: str | None,
    *,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    cache_hit_tokens: int = 0,
    at: datetime | None = None,
) -> float | None:
    """Returns USD cost for one LLM call, or None if pricing isn't confirmed
    for this model yet -- never guesses a number for a dashboard to trust."""
    if not model:
        return None
    tier = _tier_for(model, at or datetime.now(timezone.utc))
    if not tier or not tier.confirmed:
        return None
    if tier.input_per_million is None or tier.output_per_million is None:
        return None
    prompt = max(0, (prompt_tokens or 0) - cache_hit_tokens)
    cached = min(cache_hit_tokens, prompt_tokens or 0)
    cost = (prompt / 1_000_000) * tier.input_per_million
    cost += (completion_tokens or 0) / 1_000_000 * tier.output_per_million
    if cached and tier.cached_input_per_million is not None:
        cost += (cached / 1_000_000) * tier.cached_input_per_million
    return round(cost, 6)
