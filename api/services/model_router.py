# -*- coding: utf-8 -*-
"""
ModelRouter - provider-agnostic LLM caller with strict provider isolation.

Priority order:
  1. OpenAI cascade   (gpt-4o-mini -> gpt-4o -> gpt-3.5-turbo)
  2. Anthropic fallback (ANTHROPIC_MODEL env, default claude-haiku-4-5-20251001)

Rules:
  - An OpenAI model name (gpt-*, o1*, o3*) is NEVER passed to Anthropic.
  - An Anthropic model name (claude-*) is NEVER passed to OpenAI.
  - On OpenAI 401 (invalid_api_key) the cascade stops immediately and we move
    to Anthropic (no point retrying every gpt-* with the same bad key).
  - When both providers are exhausted we raise ModelRouterError with a stable
    operator-facing message; per-attempt details are appended for debugging.
"""
from __future__ import annotations

import logging
import json
import os
from typing import Any, Optional

logger = logging.getLogger("model_router")


def _llm_insecure_ssl() -> bool:
    """Should LLM HTTP clients skip SSL verification?

    Workaround for Windows boxes where SChannel can't verify CRL revocation
    on api.openai.com / api.anthropic.com. Turn on via env
    `INSECURE_LLM_SSL=1` only on local/QA machines that already need
    similar workarounds for Supabase. Production never sets this.
    """
    raw = (os.environ.get("INSECURE_LLM_SSL") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


_LLM_HTTP_CLIENT = None


def _llm_http_client():
    """Returns a shared httpx.Client honoring INSECURE_LLM_SSL, or None."""
    global _LLM_HTTP_CLIENT
    if _LLM_HTTP_CLIENT is not None:
        return _LLM_HTTP_CLIENT
    if not _llm_insecure_ssl():
        return None
    try:
        import httpx
    except Exception:
        return None
    _LLM_HTTP_CLIENT = httpx.Client(verify=False, timeout=120)
    logger.warning("ModelRouter using INSECURE LLM HTTP client (SSL verify off). Do NOT set in production.")
    return _LLM_HTTP_CLIENT

# OpenAI cascade - tried in this exact order on model-not-found / permission errors
_OPENAI_CHAIN: list[str] = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
_OPENAI_KNOWN: frozenset[str] = frozenset(_OPENAI_CHAIN)

# Anthropic default fallback - overridable via env so we can flip Sonnet/Haiku
# without redeploying. Must be a real claude-* model id; never an OpenAI id.
_ANTHROPIC_DEFAULT = "claude-haiku-4-5-20251001"


def _is_openai_model(model: str) -> bool:
    if not isinstance(model, str):
        return False
    return model in _OPENAI_KNOWN or model.startswith(("gpt-", "o1", "o3"))


def _is_anthropic_model(model: str) -> bool:
    if not isinstance(model, str):
        return False
    return model.startswith("claude-")


def _anthropic_fallback_model() -> str:
    """Resolve the Anthropic target model. Always returns a claude-* id."""
    env = (os.environ.get("ANTHROPIC_MODEL") or "").strip()
    if env and _is_anthropic_model(env):
        return env
    return _ANTHROPIC_DEFAULT


def _mask_key(key: Optional[str]) -> str:
    """Safe representation of an API key for logs. Shows shape + a small fingerprint;
    never the full secret. Returns '<unset>' / '<short>' for missing or invalid keys."""
    if not key:
        return "<unset>"
    k = str(key)
    if len(k) < 12:
        return "<short>"
    return f"{k[:6]}...{k[-4:]} (len={len(k)})"


def _looks_like_openai_key(key: Optional[str]) -> bool:
    if not key:
        return False
    k = str(key).strip()
    # Accept the two formats currently issued by OpenAI: sk-... and sk-proj-...
    return k.startswith("sk-") and len(k) >= 20


def _looks_like_anthropic_key(key: Optional[str]) -> bool:
    if not key:
        return False
    k = str(key).strip()
    return k.startswith("sk-ant-") and len(k) >= 20


def _to_openai_tools(anthropic_tools: list[dict]) -> list[dict]:
    """Converte a schema canonical (formato Anthropic) para o formato que a
    OpenAI Chat Completions API espera: {"type":"function","function":{name,description,parameters}}.

    Mantemos `SOFIA_TOOLS_SCHEMA` no formato Anthropic porque ele e mais
    sucinto (`input_schema` direto), e fazemos a traducao aqui na hora do
    request OpenAI.
    """
    result: list[dict] = []
    for tool in anthropic_tools or []:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name") or "").strip()
        if not name:
            continue
        result.append({
            "type": "function",
            "function": {
                "name": name,
                "description": str(tool.get("description") or ""),
                "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
            },
        })
    return result


def _tool_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _messages_for_openai(messages: list[dict]) -> list[dict]:
    """Translate the provider-neutral tool transcript to OpenAI's native roles."""
    result: list[dict] = []
    for message in messages or []:
        role = message.get("role")
        if role == "tool":
            call_id = str(message.get("tool_call_id") or "").strip()
            if not call_id:
                raise ValueError("OpenAI tool result requires tool_call_id")
            result.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": _tool_content(message.get("content")),
            })
            continue
        if role == "assistant" and message.get("tool_calls"):
            calls = []
            for call in message.get("tool_calls") or []:
                call_id = str(call.get("id") or "").strip()
                if not call_id:
                    raise ValueError("OpenAI assistant tool call requires id")
                calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": str(call.get("name") or ""),
                        "arguments": _tool_content(call.get("arguments") or {}),
                    },
                })
            result.append({"role": "assistant", "content": message.get("content") or None, "tool_calls": calls})
            continue
        result.append({"role": role, "content": message.get("content", "")})
    return result


def _messages_for_anthropic(messages: list[dict]) -> list[dict]:
    """Translate canonical calls/results to Anthropic tool_use/tool_result blocks."""
    result: list[dict] = []
    index = 0
    while index < len(messages or []):
        message = messages[index]
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            blocks: list[dict] = []
            if message.get("content"):
                blocks.append({"type": "text", "text": str(message["content"])})
            for call in message.get("tool_calls") or []:
                call_id = str(call.get("id") or "").strip()
                if not call_id:
                    raise ValueError("Anthropic tool_use requires id")
                blocks.append({
                    "type": "tool_use", "id": call_id,
                    "name": str(call.get("name") or ""),
                    "input": call.get("arguments") or {},
                })
            result.append({"role": "assistant", "content": blocks})
            index += 1
            continue
        if role == "tool":
            blocks: list[dict] = []
            while index < len(messages) and messages[index].get("role") == "tool":
                tool_message = messages[index]
                call_id = str(tool_message.get("tool_call_id") or "").strip()
                if not call_id:
                    raise ValueError("Anthropic tool_result requires tool_call_id")
                blocks.append({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": _tool_content(tool_message.get("content")),
                    "is_error": bool(tool_message.get("is_error", False)),
                })
                index += 1
            result.append({"role": "user", "content": blocks})
            continue
        result.append({"role": role, "content": message.get("content", "")})
        index += 1
    return result


def _error_status_code(exc: Exception) -> Optional[int]:
    """Best-effort HTTP status code extraction from provider SDK exceptions."""
    for attr in ("status_code", "http_status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(exc, "response", None)
    if resp is not None:
        v = getattr(resp, "status_code", None)
        if isinstance(v, int):
            return v
    return None


# All selectable model IDs exposed to callers (OpenAI side - what the UI offers)
AVAILABLE_MODELS: dict[str, str] = {
    "gpt-4o-mini":  "GPT-4o Mini - Rapido",
    "gpt-4o":       "GPT-4o - Balanceado",
    "gpt-3.5-turbo": "GPT-3.5 Turbo - Legado",
}


# Final operator-facing message when no provider answers
_NO_PROVIDER_MSG = (
    "Nenhum provedor de IA disponivel. "
    "Verifique OPENAI_API_KEY ou creditos Anthropic."
)


class ModelRouterError(RuntimeError):
    """Raised when every provider in the cascade has been exhausted."""


class ModelRouter:
    """
    Single entry-point for all LLM calls in the platform.

    Usage:
        router = ModelRouter()
        text = router.chat("gpt-4o-mini", prompt)
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
    ) -> None:
        self._openai_key = (openai_api_key or os.environ.get("OPENAI_API_KEY", "") or "").strip()
        self._anthropic_key = (anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "") or "").strip()

        if self._openai_key and not _looks_like_openai_key(self._openai_key):
            logger.warning(
                "ModelRouter OPENAI_API_KEY does not look like an OpenAI key (%s); "
                "calls may return 401.",
                _mask_key(self._openai_key),
            )
        if self._anthropic_key and not _looks_like_anthropic_key(self._anthropic_key):
            logger.warning(
                "ModelRouter ANTHROPIC_API_KEY does not look like an Anthropic key (%s); "
                "calls may return 401.",
                _mask_key(self._anthropic_key),
            )

    # -- Public API ------------------------------------------------------------

    def chat(self, model: str, prompt: str, max_tokens: int = 1024) -> str:
        """Call LLM with cascade fallback. Returns response text.
        Raises ModelRouterError only if ALL providers fail."""
        return self.messages_create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )

    def messages_create(
        self,
        model: str,
        messages: list,
        system: str = "",
        max_tokens: int = 1024,
        tools: Optional[list[dict]] = None,
    ) -> Any:
        """Multi-turn conversation with cascade fallback.
        messages must follow the Anthropic format:
            [{"role": "user"|"assistant", "content": str}]

        When `tools` is provided, returns dict:
            {"text": str, "tool_calls": [{"id": str, "name": str, "arguments": dict}], "stop_reason": str, "provider": "openai|anthropic"}
        Quando `tools` e None, mantem retorno legado (str) para nao quebrar
        callers existentes.

        Raises ModelRouterError only if ALL providers fail."""
        errors: list[str] = []

        # If caller asked for an Anthropic model explicitly, do NOT route through
        # OpenAI first (it would be a wasted 404 / wrong-provider call).
        if _is_anthropic_model(model):
            result = self._try_anthropic_messages(model, messages, system, max_tokens, errors, tools=tools)
            if result is not None:
                return result
            # Anthropic explicit failure -> still try OpenAI as a last resort,
            # but only with a known OpenAI model (never a claude-* id).
            result = self._try_openai_messages(_OPENAI_CHAIN[0], messages, system, max_tokens, errors, tools=tools)
            if result is not None:
                return result
            raise ModelRouterError(self._final_error_message(errors))

        # Default: OpenAI cascade first, Anthropic fallback second.
        result = self._try_openai_messages(model, messages, system, max_tokens, errors, tools=tools)
        if result is not None:
            return result

        # Anthropic fallback - ignore the OpenAI model name entirely.
        result = self._try_anthropic_messages(
            _anthropic_fallback_model(), messages, system, max_tokens, errors, tools=tools
        )
        if result is not None:
            return result

        raise ModelRouterError(self._final_error_message(errors))

    # -- Provider implementations ----------------------------------------------

    def _try_openai_messages(
        self,
        model: str,
        messages: list,
        system: str,
        max_tokens: int,
        errors: list[str],
        tools: Optional[list[dict]] = None,
    ) -> Optional[Any]:
        if not self._openai_key:
            errors.append("OpenAI: no OPENAI_API_KEY configured")
            logger.warning(
                "ModelRouter skip provider=openai reason=missing_key key=%s",
                _mask_key(self._openai_key),
            )
            return None

        try:
            from openai import (
                OpenAI,
                NotFoundError,
                PermissionDeniedError,
                AuthenticationError,
                BadRequestError,
                RateLimitError,
            )
        except Exception as exc:
            errors.append(f"OpenAI/import: {exc}")
            logger.warning("ModelRouter provider=openai error_type=ImportError msg=%s", exc)
            return None

        client = OpenAI(api_key=self._openai_key, http_client=_llm_http_client())
        oai_msgs: list = []
        if system:
            oai_msgs.append({"role": "system", "content": system})
        oai_msgs.extend(_messages_for_openai(messages))

        # Build try-order: requested model first (only if it's an OpenAI id),
        # then the rest of the cascade. Never pass a claude-* id to OpenAI.
        resolved = model if _is_openai_model(model) else _OPENAI_CHAIN[0]
        order = [resolved] + [m for m in _OPENAI_CHAIN if m != resolved]

        oai_tools = _to_openai_tools(tools) if tools else None

        for attempt in order:
            try:
                logger.info(
                    "ModelRouter call provider=openai model=%s key=%s tools=%s",
                    attempt, _mask_key(self._openai_key), bool(oai_tools),
                )
                kwargs: dict = {
                    "model": attempt,
                    "max_tokens": max_tokens,
                    "messages": oai_msgs,
                }
                if oai_tools:
                    kwargs["tools"] = oai_tools
                    kwargs["tool_choice"] = "auto"
                resp = client.chat.completions.create(**kwargs)
                logger.info("ModelRouter ok provider=openai model=%s", attempt)
                if tools is None:
                    return (resp.choices[0].message.content or "").strip()
                msg = resp.choices[0].message
                tool_calls_raw = getattr(msg, "tool_calls", None) or []
                tool_calls: list[dict] = []
                for call in tool_calls_raw:
                    fn = getattr(call, "function", None)
                    name = getattr(fn, "name", None) if fn else None
                    args_raw = getattr(fn, "arguments", None) if fn else None
                    parsed_args: dict = {}
                    if isinstance(args_raw, str) and args_raw.strip():
                        try:
                            import json as _json
                            parsed_args = _json.loads(args_raw)
                        except Exception:
                            parsed_args = {"_raw": args_raw}
                    elif isinstance(args_raw, dict):
                        parsed_args = args_raw
                    tool_calls.append({
                        "id": getattr(call, "id", "") or "",
                        "name": name or "",
                        "arguments": parsed_args,
                    })
                return {
                    "text": (msg.content or "").strip(),
                    "tool_calls": tool_calls,
                    "stop_reason": resp.choices[0].finish_reason or "stop",
                    "provider": "openai",
                    "model": attempt,
                }

            except AuthenticationError as exc:
                status = _error_status_code(exc) or 401
                logger.warning(
                    "ModelRouter fail provider=openai model=%s error_type=AuthenticationError status=%s key=%s",
                    attempt, status, _mask_key(self._openai_key),
                )
                errors.append(
                    f"OpenAI/{attempt}: 401 invalid_api_key (check OPENAI_API_KEY)"
                )
                # 401 will repeat for every attempt - stop the cascade here.
                return None

            except PermissionDeniedError as exc:
                status = _error_status_code(exc) or 403
                logger.warning(
                    "ModelRouter fail provider=openai model=%s error_type=PermissionDeniedError status=%s",
                    attempt, status,
                )
                errors.append(f"OpenAI/{attempt}: {status} permission_denied")
                continue

            except NotFoundError as exc:
                status = _error_status_code(exc) or 404
                logger.warning(
                    "ModelRouter fail provider=openai model=%s error_type=NotFoundError status=%s",
                    attempt, status,
                )
                errors.append(f"OpenAI/{attempt}: {status} model_not_found")
                continue

            except BadRequestError as exc:
                status = _error_status_code(exc) or 400
                logger.warning(
                    "ModelRouter fail provider=openai model=%s error_type=BadRequestError status=%s",
                    attempt, status,
                )
                errors.append(f"OpenAI/{attempt}: {status} bad_request")
                continue

            except RateLimitError as exc:
                status = _error_status_code(exc) or 429
                logger.warning(
                    "ModelRouter fail provider=openai model=%s error_type=RateLimitError status=%s",
                    attempt, status,
                )
                errors.append(f"OpenAI/{attempt}: {status} rate_limited")
                # Rate limit is provider-wide; try next provider, not next model.
                return None

            except Exception as exc:
                status = _error_status_code(exc)
                logger.warning(
                    "ModelRouter fail provider=openai model=%s error_type=%s status=%s",
                    attempt, type(exc).__name__, status,
                )
                errors.append(f"OpenAI/{attempt}: {type(exc).__name__}")
                continue

        return None  # all OpenAI models exhausted

    def _try_anthropic_messages(
        self,
        model: str,
        messages: list,
        system: str,
        max_tokens: int,
        errors: list[str],
        tools: Optional[list[dict]] = None,
    ) -> Optional[Any]:
        if not self._anthropic_key:
            errors.append("Anthropic: no ANTHROPIC_API_KEY configured")
            logger.warning(
                "ModelRouter skip provider=anthropic reason=missing_key key=%s",
                _mask_key(self._anthropic_key),
            )
            return None

        # Defense in depth: even if a caller hands us 'gpt-4o-mini', we MUST
        # NOT send that to Anthropic. Force to the configured claude-* model.
        target = model if _is_anthropic_model(model) else _anthropic_fallback_model()

        try:
            import anthropic
        except Exception as exc:
            errors.append(f"Anthropic/import: {exc}")
            logger.warning("ModelRouter provider=anthropic error_type=ImportError msg=%s", exc)
            return None

        try:
            logger.info(
                "ModelRouter call provider=anthropic model=%s key=%s tools=%s",
                target, _mask_key(self._anthropic_key), bool(tools),
            )
            client = anthropic.Anthropic(api_key=self._anthropic_key, http_client=_llm_http_client())
            kwargs: dict = {
                "model": target,
                "max_tokens": max_tokens,
                "messages": _messages_for_anthropic(messages),
            }
            if system:
                kwargs["system"] = system
            if tools:
                # Anthropic ja aceita o formato {name, description, input_schema}
                # nativamente -- mesmo formato que SOFIA_TOOLS_SCHEMA exporta.
                kwargs["tools"] = list(tools)
            resp = client.messages.create(**kwargs)
            logger.info("ModelRouter ok provider=anthropic model=%s", target)
            if tools is None:
                # Compatibilidade legado: devolve apenas a primeira mensagem de texto.
                for block in resp.content or []:
                    if getattr(block, "type", "") == "text":
                        return (getattr(block, "text", "") or "").strip()
                return ""
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for block in resp.content or []:
                btype = getattr(block, "type", "")
                if btype == "text":
                    text_parts.append(getattr(block, "text", "") or "")
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": getattr(block, "id", "") or "",
                        "name": getattr(block, "name", "") or "",
                        "arguments": getattr(block, "input", {}) or {},
                    })
            return {
                "text": "".join(text_parts).strip(),
                "tool_calls": tool_calls,
                "stop_reason": getattr(resp, "stop_reason", "") or "end_turn",
                "provider": "anthropic",
                "model": target,
            }

        except Exception as exc:
            status = _error_status_code(exc)
            msg = str(exc)
            error_type = type(exc).__name__
            # Friendly classifications for the common cases the user hit.
            if "credit balance" in msg.lower() or "insufficient" in msg.lower():
                friendly = "credit_balance_too_low"
            elif status == 401 or "invalid x-api-key" in msg.lower() or "authentication" in msg.lower():
                friendly = "invalid_api_key"
            elif status == 429:
                friendly = "rate_limited"
            else:
                friendly = error_type
            logger.warning(
                "ModelRouter fail provider=anthropic model=%s error_type=%s status=%s friendly=%s",
                target, error_type, status, friendly,
            )
            errors.append(f"Anthropic/{target}: {friendly}")
            return None

    # -- Final error formatting -------------------------------------------------

    @staticmethod
    def _final_error_message(errors: list[str]) -> str:
        if not errors:
            return _NO_PROVIDER_MSG
        detail = "; ".join(errors)
        return f"{_NO_PROVIDER_MSG} Detalhes: {detail}"

    # -- Asset pipeline shims --------------------------------------------------

    def cheap_text(self, prompt: str, max_tokens: int = 256) -> str:
        """Cheap text completion for tasks like renaming / classification when
        text is already extracted. Configurable via ASSET_RENAME_MODEL env."""
        model = os.environ.get("ASSET_RENAME_MODEL", "gpt-4o-mini")
        return self.chat(model, prompt, max_tokens=max_tokens)

    def vision_extract(
        self,
        file_bytes: bytes,
        mime: str,
        prompt: str = "Extraia o texto desta imagem e descreva o conteudo visual em 1 linha.",
        max_tokens: int = 512,
    ) -> Optional[dict]:
        """Image-to-text fallback. Returns dict with extracted_text/visual_summary/model_used.
        Returns None when no vision-capable provider is configured.
        Model configurable via ASSET_VISION_MODEL env (default: gpt-4o-mini)."""
        model = os.environ.get("ASSET_VISION_MODEL", "gpt-4o-mini")

        if not self._openai_key:
            logger.warning(
                "ModelRouter skip provider=openai feature=vision_extract reason=missing_key key=%s",
                _mask_key(self._openai_key),
            )
            return None

        try:
            import base64 as _b64
            from openai import OpenAI

            b64 = _b64.b64encode(file_bytes).decode("ascii")
            data_url = f"data:{mime or 'image/png'};base64,{b64}"
            client = OpenAI(api_key=self._openai_key, http_client=_llm_http_client())
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            parts = [p.strip() for p in text.split("\n\n") if p.strip()]
            extracted = parts[0] if parts else text
            summary = parts[-1] if len(parts) > 1 else ""
            return {
                "extracted_text": extracted,
                "visual_summary": summary or (extracted[:160] if extracted else ""),
                "model_used": model,
                "raw": text,
            }
        except Exception as exc:
            status = _error_status_code(exc)
            logger.warning(
                "ModelRouter fail provider=openai feature=vision_extract model=%s error_type=%s status=%s",
                model, type(exc).__name__, status,
            )
            return None


# Module-level singleton - re-used across the process lifetime
_router: Optional[ModelRouter] = None


def get_router() -> ModelRouter:
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router
