"""Generic graph-backed deterministic appointment conversation engine."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from services.deterministic_sdr import Catalog, Product, _brl, _norm


def _duration(minutes: int | None) -> str:
    if not minutes:
        return ""
    if minutes < 60:
        return f"{minutes} minutos"
    hours, remainder = divmod(minutes, 60)
    if not remainder:
        return f"{hours} hora" if hours == 1 else f"{hours} horas"
    return f"{hours} hora e {remainder} minutos"


def _service_fact(product: Product, *, allow_price: bool = True) -> str:
    """One factual sentence about a service.

    ``allow_price=False`` is used when the persona's published policy reserves
    every price statement for a human: the money figure is then never emitted,
    even if a price somehow survives in the catalog data.
    """
    facts: list[str] = []
    duration = _duration(product.duration_minutes)
    if duration:
        facts.append(f"leva cerca de {duration}")
    if allow_price and product.price is not None:
        prefix = "parte de " if product.price_qualifier == "starting_at" else "custa "
        facts.append(f"{prefix}{_brl(product.price)}")
    if not facts:
        return product.public_name
    return f"{product.public_name} {' e '.join(facts)}"


def _missing(request: dict[str, Any], required: Iterable[str]) -> list[str]:
    return [field for field in required if not request.get(field)]


def _extract_name(text: str) -> str | None:
    match = re.search(
        r"\b(?:meu nome (?:e|Ã©)|me chamo|sou o|sou a)\s+"
        r"([A-Za-zÃ€-Ã¿]{2,}(?:\s+[A-Za-zÃ€-Ã¿]{2,}){0,3})",
        text,
        re.IGNORECASE,
    )
    return match.group(1).strip(" .,") if match else None


def _extract_date(text: str) -> str | None:
    match = re.search(r"\b(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b", text)
    if match:
        return match.group(1)
    match = re.search(
        r"\b(?:dia|data)\s+([A-Za-zÃ€-Ã¿0-9][A-Za-zÃ€-Ã¿0-9 /-]{1,20})",
        text,
        re.IGNORECASE,
    )
    return match.group(1).strip(" .,") if match else None


def _extract_window(text: str) -> str | None:
    normalized = _norm(text)
    for term, value in (
        ("de manha", "manhÃ£"),
        ("pela manha", "manhÃ£"),
        ("manha", "manhÃ£"),
        ("de tarde", "tarde"),
        ("pela tarde", "tarde"),
        ("tarde", "tarde"),
        ("de noite", "noite"),
        ("pela noite", "noite"),
        ("noite", "noite"),
    ):
        if term in normalized:
            return value
    match = re.search(r"\b(?:as|Ã s)\s*(\d{1,2}(?::\d{2})?)\b", text, re.IGNORECASE)
    return match.group(1) if match else None


def _extract_size(text: str) -> str | None:
    normalized = _norm(text)
    aliases = (
        ("hatch", "hatch"),
        ("sedan", "sedan"),
        ("suv", "SUV"),
        ("picape", "picape"),
        ("pickup", "picape"),
        ("compacto", "compacto"),
        ("medio", "mÃ©dio"),
        ("grande", "grande"),
    )
    return next((value for alias, value in aliases if re.search(rf"\b{alias}\b", normalized)), None)


def _plain_value(text: str) -> str | None:
    value = text.strip(" .,")
    return value if 1 < len(value) <= 160 else None


@dataclass
class AppointmentResult:
    reply: str | None
    intent: str
    state: dict[str, Any]
    handoff: bool = False
    handoff_reason: str | None = None
    product: Product | None = None


class DeterministicAppointment:
    """Collect a booking request without ever confirming price or availability."""

    def __init__(self, catalog: Catalog, *, policy: dict[str, Any] | None = None):
        self.catalog = catalog
        self.policy = dict(policy or {})
        self.policy_required_fields = tuple(
            str(field).strip()
            for field in self.policy.get("required_fields") or []
            if str(field).strip()
        )
        self.field_questions = dict(self.policy.get("field_questions") or {})
        self.texts = dict(self.policy.get("texts") or {})
        # Graph-declared, never persona-specific: when the published policy
        # reserves price disclosure for a human, this engine must not state a
        # value, a range or an approximation for any service.
        self.price_human_only = (
            str(self.policy.get("price_disclosure") or "").strip().lower()
            == "human_only"
        )

        if not self.policy_required_fields:
            raise ValueError(
                "appointment_policy.required_fields must be defined by the graph"
            )

    def _text(self, key: str, fallback: str) -> str:
        return str(self.texts.get(key) or fallback)

    def _fact(self, product: Product) -> str:
        return _service_fact(product, allow_price=not self.price_human_only)

    def _new_state(self, state: dict[str, Any] | None) -> dict[str, Any]:
        current = dict(state or {})
        request = dict(current.get("appointment_request") or {})
        current.update(
            {
                "business_model": "appointment",
                "appointment_request": request,
                "missing_fields": list(current.get("missing_fields") or []),
                "conversation_state": current.get("conversation_state") or "discovering",
            }
        )
        return current

    def _product(
        self,
        request: dict[str, Any],
        message: str,
        *,
        identified_service_slug: str | None = None,
    ) -> Product | None:
        if identified_service_slug:
            identified = next(
                (
                    item
                    for item in self.catalog.products
                    if item.slug == identified_service_slug
                ),
                None,
            )
            if identified:
                request["servico"] = identified.slug
                return identified
        product = self.catalog.find_product(message)
        if product:
            request["servico"] = product.slug
            return product
        slug = request.get("servico")
        return next((item for item in self.catalog.products if item.slug == slug), None)

    def _required(self, product: Product | None) -> tuple[str, ...]:
        required = (
            product.required_fields
            if product and product.required_fields
            else self.policy_required_fields
        )
        missing_questions = [
            field
            for field in required
            if not str(self.field_questions.get(field) or "").strip()
        ]
        if missing_questions:
            raise ValueError(
                "appointment_policy.field_questions missing required fields: "
                + ", ".join(missing_questions)
            )
        return tuple(required)

    def _collect(self, message: str, request: dict[str, Any], expected: str | None) -> None:
        name = _extract_name(message)
        date = _extract_date(message)
        window = _extract_window(message)
        size = _extract_size(message)
        if name:
            request["nome_cliente"] = name
        if date:
            request["data_desejada"] = date
        if window:
            request["janela_horario"] = window
        if size:
            request["vehicle_size"] = size

        plain = _plain_value(message)
        if not plain or not expected or request.get(expected):
            return
        if expected == "nome_cliente":
            if re.fullmatch(r"[A-Za-zÃ€-Ã¿]{2,}(?:\s+[A-Za-zÃ€-Ã¿]{2,}){0,3}", plain):
                request[expected] = plain
        elif expected == "vehicle_size":
            if size:
                request[expected] = size
        elif expected == "data_desejada":
            if date:
                request[expected] = date
        elif expected == "janela_horario":
            if window:
                request[expected] = window
        else:
            # Any other field (modelo_veiculo, condicao, vehicle_year,
            # vehicle_color, objective, can_visit_in_person, and future
            # per-product fields) accepts free text as-is.
            request[expected] = plain

    def _list_services(self) -> str:
        lines = []
        for product in self.catalog.products:
            fact = self._fact(product)
            qualifier = "." if fact.endswith(".") else ""
            lines.append(f"- {fact}{qualifier}")
        return self._text("cabecalho_servicos", "ServiÃ§os disponÃ­veis:") + "\n" + "\n".join(lines)

    def handle(
        self,
        message: str,
        *,
        state: dict[str, Any] | None = None,
        identified_service_slug: str | None = None,
    ) -> AppointmentResult:
        text = (message or "").strip()
        normalized = _norm(text)
        state = self._new_state(state)
        request = state["appointment_request"]

        if state.get("conversation_state") == "handoff":
            # Every other transition into "handoff" in this function passes
            # handoff=True (propagates to response.handoff_required, which
            # is what keeps ai_paused set and the lead visibly escalated).
            # This branch â€” messages arriving *after* the lead is already
            # in the sticky handoff state â€” used AppointmentResult's
            # default handoff=False, so the worker treated a silently
            # dropped message as a normal "sent" turn instead of
            # re-confirming the lead needs a human. Confirmed live
            # 2026-08-01: repeated messages to an already-handed-off lead
            # got zero reply and no re-pause.
            return AppointmentResult(None, "handoff", state, True, "already_handoff")

        exceptional_terms = (
            "reclamacao", "reclamar", "garantia", "desconto", "problema",
            "duvida tecnica", "servico nao cadastrado", "cancelar", "reagendar",
        )
        if any(term in normalized for term in exceptional_terms):
            state["conversation_state"] = "handoff"
            return AppointmentResult(
                self._text(
                    "encaminhamento_excepcional",
                    "Vou encaminhar sua solicitaÃ§Ã£o para a equipe responsÃ¡vel.",
                ),
                "exceptional_support",
                state,
                True,
                "exceptional_support",
            )
        if any(
            term in normalized
            for term in ("atendimento humano", "falar com alguem", "falar com uma pessoa", "atendente", "humano")
        ):
            state["conversation_state"] = "handoff"
            return AppointmentResult(
                self._text(
                    "atendimento_humano",
                    "Vou encaminhar sua conversa para o atendimento humano.",
                ),
                "request_human",
                state,
                True,
                "request_human",
            )

        message_product = self.catalog.find_product(text)
        product = self._product(
            request,
            text,
            identified_service_slug=identified_service_slug,
        )
        explicit_booking = any(
            term in normalized
            for term in ("agendar", "agendamento", "marcar", "reservar", "quero horario")
        )
        explicit_quote = any(
            term in normalized for term in ("orcamento", "cotacao", "quanto fica")
        )
        explicit_service_interest = bool(product) and any(
            term in normalized
            for term in (
                "quero fazer",
                "quero contratar",
                "quero realizar",
                "gostaria de fazer",
                "preciso fazer",
                "tenho interesse",
            )
        )
        is_price = any(term in normalized for term in ("preco", "custa", "valor", "quanto"))
        is_service_query = any(term in normalized for term in ("como funciona", "quanto tempo", "duracao", "leva"))
        # "quanto tempo leva" also contains "quanto", so it reads as a price
        # question below. Both branches used to return the same _service_fact
        # sentence, so it never mattered; once the price branch hands off to a
        # human it does, and a duration question must not be answered with the
        # price-is-human-only text.
        is_duration_query = any(
            term in normalized
            for term in ("quanto tempo", "duracao", "demora", "leva")
        )
        is_list = (
            normalized in {"servicos", "listar servicos", "lista de servicos"}
            or any(term in normalized for term in ("quais servicos", "que servicos", "opcoes de servico"))
        )
        greeting = normalized in {"oi", "ola", "bom dia", "boa tarde", "boa noite"}
        if (
            greeting
            or is_list
            or message_product
            or (product and (is_price or is_service_query))
            or explicit_booking
            or explicit_quote
            or explicit_service_interest
            or state.get("conversation_state") == "collecting"
        ):
            state["clarification_attempts"] = 0

        collecting = state.get("conversation_state") == "collecting"
        previous_missing = list(state.get("missing_fields") or [])
        expected = previous_missing[0] if collecting and previous_missing else None
        self._collect(text, request, expected)
        product = self._product(
            request,
            text,
            identified_service_slug=identified_service_slug,
        )
        required = self._required(product)
        missing = _missing(request, required)
        state["missing_fields"] = missing

        if collecting or explicit_booking or explicit_quote or explicit_service_interest:
            state["conversation_state"] = "collecting"
            if not missing:
                state["conversation_state"] = "handoff"
                fact = self._fact(product) if product else "O serviÃ§o solicitado"
                if product:
                    fact = f"A {fact[:1].lower()}{fact[1:]}"
                reply = f"{fact}. " + self._text(
                    "complemento_confirmacao",
                    "Registrei sua preferÃªncia; a equipe confirmarÃ¡ "
                    "o valor final e o horÃ¡rio.",
                )
                return AppointmentResult(
                    reply,
                    "complete_booking_request",
                    state,
                    True,
                    "appointment_confirmation_required",
                    product,
                )
            intent = (
                "request_booking"
                if explicit_booking
                else "request_quote"
                if explicit_quote
                else "request_service"
                if explicit_service_interest
                else {
                    "modelo_veiculo": "provide_vehicle",
                    "vehicle_size": "provide_vehicle",
                    "condicao": "provide_condition",
                    "data_desejada": "provide_date",
                    "janela_horario": "provide_time_window",
                }.get(expected or "", "request_booking")
            )
            prefix = ""
            if product and (explicit_booking or explicit_quote or explicit_service_interest):
                prefix = f"{self._fact(product)}. "
            return AppointmentResult(
                prefix + self.field_questions[missing[0]],
                intent,
                state,
                product=product,
            )

        if greeting:
            return AppointmentResult(
                self.catalog.greeting or "OlÃ¡. Como posso ajudar?",
                "greeting",
                state,
            )
        if is_list:
            return AppointmentResult(self._list_services(), "list_services", state)
        if product and is_price and not (self.price_human_only and is_duration_query):
            if self.price_human_only:
                # The graph reserves every price statement for a person on the
                # team: answer with the published text and escalate, exactly
                # like the other handoff paths in this method.
                state["conversation_state"] = "handoff"
                return AppointmentResult(
                    self._text(
                        "preco_humano",
                        "O valor Ã© definido por uma pessoa da equipe. Vou "
                        "encaminhar sua conversa para o atendimento.",
                    ),
                    "consult_price",
                    state,
                    True,
                    "price_requires_human",
                    product,
                )
            return AppointmentResult(f"{self._fact(product)}.", "consult_price", state, product=product)
        if product and (is_service_query or message_product):
            if (
                not product.process_documented
                and str(self.texts.get("lacuna_conhecimento") or "").strip()
            ):
                # The graph sells this service but publishes no process for it.
                # Describing it anyway is exactly the invention this guard
                # exists to prevent.
                state["conversation_state"] = "handoff"
                return AppointmentResult(
                    self.texts["lacuna_conhecimento"],
                    "consult_service",
                    state,
                    True,
                    "knowledge_gap_requires_human",
                    product,
                )
            if (
                is_duration_query
                and product.duration_minutes is None
                and str(self.texts.get("lacuna_conhecimento") or "").strip()
            ):
                # The duration is genuinely absent from the published graph and
                # the persona published a knowledge-gap text for exactly this:
                # hand off instead of inventing a number. Personas that publish
                # no such text keep the previous behaviour.
                state["conversation_state"] = "handoff"
                return AppointmentResult(
                    self.texts["lacuna_conhecimento"],
                    "consult_service",
                    state,
                    True,
                    "knowledge_gap_requires_human",
                    product,
                )
            return AppointmentResult(f"{self._fact(product)}.", "consult_service", state, product=product)

        attempts = int(state.get("clarification_attempts") or 0) + 1
        state["clarification_attempts"] = attempts
        if attempts > 1:
            state["conversation_state"] = "handoff"
            return AppointmentResult(
                self._text(
                    "encaminhamento_duvida_persistente",
                    "JÃ¡ chamei a equipe para continuar com vocÃª. Se puder, me "
                    "diga seu nome, o serviÃ§o que procura e o modelo do carro "
                    "enquanto aguarda.",
                ),
                "ununderstood",
                state,
                True,
                "missing_approved_evidence",
            )
        return AppointmentResult(
            self._text(
                "esclarecimento_duvida",
                "Vou chamar a equipe para te ajudar direitinho nisso. "
                "Enquanto isso, me conta seu nome, o serviÃ§o que vocÃª procura "
                "e o modelo do carro?",
            ),
            "ununderstood",
            state,
        )

