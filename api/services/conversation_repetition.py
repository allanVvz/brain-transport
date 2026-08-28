"""Deterministic anti-repetition policy shared by conversation quality gates."""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable


REPETITION_THRESHOLD = 0.92
QUESTION_PARAPHRASE_THRESHOLD = 0.60
QUESTION_SUFFIX_MIN_RATIO = 0.60
_EMPTY_BRIDGE_TOKENS = {
    "ah", "beleza", "bom", "certo", "claro", "entao", "entendi", "legal",
    "ok", "otimo", "perfeito", "sim", "ta", "tudo", "vamos",
}


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def sequential_similarity(left: Any, right: Any) -> float:
    """Normalized Levenshtein similarity, mirrored by the Node driver."""
    a, b = normalize_text(left), normalize_text(right)
    if not a or not b:
        return 1.0 if a == b else 0.0
    if a == b:
        return 1.0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for row, char_a in enumerate(a, start=1):
        current = [row]
        for column, char_b in enumerate(b, start=1):
            current.append(min(
                current[column - 1] + 1,
                previous[column] + 1,
                previous[column - 1] + (char_a != char_b),
            ))
        previous = current
    return 1.0 - (previous[-1] / max(len(a), len(b)))


def token_overlap(left: Any, right: Any) -> float:
    a = set(normalize_text(left).split())
    b = set(normalize_text(right).split())
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a), len(b))


def semantic_similarity(left: Any, right: Any) -> float:
    return max(sequential_similarity(left, right), token_overlap(left, right))


def is_semantic_repetition(left: Any, right: Any) -> bool:
    a, b = normalize_text(left), normalize_text(right)
    if not a or not b:
        return False
    return a == b or semantic_similarity(a, b) >= REPETITION_THRESHOLD


def _contextual_bridge(reply: Any, question_text: Any) -> str:
    folded_reply = normalize_text(reply)
    folded_question = normalize_text(question_text)
    if not folded_reply or not folded_question:
        return ""
    offset = folded_reply.rfind(folded_question)
    if offset >= 0:
        return folded_reply[:offset].strip()

    # A contextual resumption remains valid when the model lightly
    # paraphrases the graph-owned question. Compare token suffixes so the
    # prefix can still be audited as the bridge; comparing the whole reply
    # would let a substantive prefix hide a bare repeated question.
    reply_tokens = folded_reply.split()
    question_tokens = folded_question.split()
    minimum_suffix = max(3, int(len(question_tokens) * QUESTION_SUFFIX_MIN_RATIO))
    best_offset = -1
    best_score = 0.0
    for token_offset in range(len(reply_tokens)):
        suffix_tokens = reply_tokens[token_offset:]
        if len(suffix_tokens) < minimum_suffix:
            break
        score = semantic_similarity(" ".join(suffix_tokens), folded_question)
        if score > best_score:
            best_score = score
            best_offset = token_offset
    if best_offset < 0 or best_score < QUESTION_PARAPHRASE_THRESHOLD:
        return ""
    return " ".join(reply_tokens[:best_offset]).strip()


def has_substantive_contextual_bridge(reply: Any, question_text: Any) -> bool:
    bridge = _contextual_bridge(reply, question_text)
    substantive = [
        token for token in bridge.split()
        if token not in _EMPTY_BRIDGE_TOKENS and len(token) > 1
    ]
    return len(substantive) >= 2


def assess_repetition(
    *,
    current_reply: Any,
    recent_replies: Iterable[Any] = (),
    question_node_id: str | None = None,
    question_text: str | None = None,
    asked_question_node_ids: Iterable[Any] = (),
    max_attempts: int = 0,
    field_pending: bool = False,
    terminal_intent: str | None = None,
    previous_terminal_intent: str | None = None,
) -> dict[str, Any]:
    """Return deterministic repetition failures for one prospective reply.

    ``max_attempts`` is the number of contextual resumptions after the initial
    question. Therefore a value of one permits at most two emissions of the
    same ``question_node_id``.
    """
    normalized_max = max(0, min(int(max_attempts), 1))
    previous = [str(value or "") for value in recent_replies if str(value or "").strip()]
    semantic_matches = [
        index for index, value in enumerate(previous)
        if is_semantic_repetition(value, current_reply)
    ]
    failures: list[str] = []
    if semantic_matches:
        failures.append("semantic_repetition")

    asked = [str(value or "") for value in asked_question_node_ids]
    previous_emissions = asked.count(str(question_node_id or "")) if question_node_id else 0
    repeated_question = bool(question_node_id and previous_emissions > 0)
    if repeated_question:
        if previous_emissions >= 1 + normalized_max:
            failures.append("question_attempt_budget_exceeded")
        if not field_pending:
            failures.append("question_field_not_pending")
        if not has_substantive_contextual_bridge(current_reply, question_text):
            failures.append("contextual_bridge_required")

    if terminal_intent and terminal_intent == previous_terminal_intent:
        failures.append("terminal_repetition")

    failures = list(dict.fromkeys(failures))
    return {
        "passed": not failures,
        "failures": failures,
        "normalized_reply": normalize_text(current_reply),
        "semantic_match_indexes": semantic_matches,
        "similarities": [semantic_similarity(value, current_reply) for value in previous],
        "question_node_id": question_node_id,
        "previous_question_emissions": previous_emissions,
        "allowed_question_emissions": 1 + normalized_max,
        "contextual_bridge": _contextual_bridge(current_reply, question_text),
        "terminal_intent": terminal_intent,
    }

