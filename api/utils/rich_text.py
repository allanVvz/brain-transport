"""Universal rich-text cleanup for imported copy.

Imported product copy (Shopify, scrapers, CMS pastes) arrives as messy HTML
(`<p data-pm-slice>`, `<span>`, `<div class>`, `<h2 data-start>`, `<br>`...).
We never want raw tags to reach a consumer â€” not the cardapio, not agents, not
RAG. `to_clean_markdown` converts the known structural tags into a tiny,
predictable markdown subset and drops everything else (tags + attributes),
yielding clean text that renders richly on the front and reads cleanly for
agents.

Kept (mapped to markdown): p, br, strong/b, em/i, ul/ol/li, h1-6.
Dropped (kept inner text): span, div, a, img wrappers, data-*/class/style, and
any other tag. Idempotent on plain text / already-clean markdown.
"""
from __future__ import annotations

import html as _html
import re

_BLOCK_BREAK = "\n\n"

# Inline emphasis -------------------------------------------------------------
_BOLD_RE = re.compile(r"<\s*(strong|b)\b[^>]*>(.*?)<\s*/\s*\1\s*>", re.IGNORECASE | re.DOTALL)
_ITALIC_RE = re.compile(r"<\s*(em|i)\b[^>]*>(.*?)<\s*/\s*\1\s*>", re.IGNORECASE | re.DOTALL)
# Headings --------------------------------------------------------------------
_HEADING_RE = re.compile(r"<\s*h[1-6]\b[^>]*>(.*?)<\s*/\s*h[1-6]\s*>", re.IGNORECASE | re.DOTALL)
# List items ------------------------------------------------------------------
_LI_RE = re.compile(r"<\s*li\b[^>]*>(.*?)<\s*/\s*li\s*>", re.IGNORECASE | re.DOTALL)
# Line / block breaks ---------------------------------------------------------
_BR_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
_BLOCK_CLOSE_RE = re.compile(r"<\s*/\s*(p|div|ul|ol|section|article|h[1-6])\s*>", re.IGNORECASE)
_BLOCK_OPEN_RE = re.compile(r"<\s*(p|div|ul|ol|section|article)\b[^>]*>", re.IGNORECASE)
# Any remaining tag -----------------------------------------------------------
_ANY_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+\n")


def _looks_like_html(text: str) -> bool:
    return bool(re.search(r"<[a-zA-Z/][^>]*>", text))


def to_clean_markdown(raw: object) -> str:
    """Convert messy HTML copy into a tiny markdown subset. Safe on plain text."""
    if raw is None:
        return ""
    text = raw if isinstance(raw, str) else str(raw)
    if not text.strip():
        return ""

    if not _looks_like_html(text):
        # Plain text (or already markdown): just normalize whitespace + entities.
        return _finalize(_html.unescape(text))

    # Inline emphasis first (so nested tags inside headings/li survive).
    text = _BOLD_RE.sub(lambda m: f"**{m.group(2).strip()}**", text)
    text = _ITALIC_RE.sub(lambda m: f"*{m.group(2).strip()}*", text)

    # Headings -> "## ..." on their own block.
    text = _HEADING_RE.sub(lambda m: f"{_BLOCK_BREAK}## {m.group(1).strip()}{_BLOCK_BREAK}", text)

    # List items -> "- ..." each on its own line.
    text = _LI_RE.sub(lambda m: f"\n- {m.group(1).strip()}", text)

    # Breaks and block boundaries.
    text = _BR_RE.sub("\n", text)
    text = _BLOCK_CLOSE_RE.sub(_BLOCK_BREAK, text)
    text = _BLOCK_OPEN_RE.sub(_BLOCK_BREAK, text)

    # Drop every remaining tag, keep inner text.
    text = _ANY_TAG_RE.sub("", text)

    return _finalize(_html.unescape(text))


def _finalize(text: str) -> str:
    # Normalize non-breaking / unicode spaces (&nbsp; -> \xa0) to plain spaces.
    text = text.replace("\xa0", " ").replace("â€‹", "").replace("â€‰", " ")
    text = _TRAILING_WS_RE.sub("\n", text)
    # Collapse runs of spaces/tabs (entities like &nbsp; become spaces).
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = _MULTI_BLANK_RE.sub(_BLOCK_BREAK, text)
    # Trim each line, then the whole blob.
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def strip_to_text(raw: object) -> str:
    """Plain-text variant (no markdown markers) â€” for places that want bare text."""
    md = to_clean_markdown(raw)
    md = re.sub(r"\*\*(.*?)\*\*", r"\1", md)
    md = re.sub(r"\*(.*?)\*", r"\1", md)
    md = re.sub(r"^#{1,6}\s*", "", md, flags=re.MULTILINE)
    md = re.sub(r"^-\s+", "", md, flags=re.MULTILINE)
    return md.strip()

