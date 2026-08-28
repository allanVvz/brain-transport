"""Deterministic, document-backed conversational sales engine.

This module is intentionally independent from a model provider and from the
database.  Markdown is parsed into a small catalog, while the conversation
state is a JSON-compatible dictionary that can live in ``leads.metadata``.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml


def _norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _brl(value: float) -> str:
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


@dataclass(frozen=True)
class Product:
    slug: str
    name: str
    display_name: str | None = None
    aliases: tuple[str, ...] = ()
    category: str = ""
    unit: str = ""
    price: float | None = None
    currency: str = "BRL"
    status: str = "active"
    duration_minutes: int | None = None
    capacity: int = 1
    price_qualifier: str = "fixed"
    required_fields: tuple[str, ...] = ()
    confirmation_required: bool = False
    booking_provider: str | None = None
    # A graph may publish a service it sells without publishing how the service
    # works. Answering "how does it work" from an undocumented process is how a
    # confident invention gets sent to a customer, so the graph says so
    # explicitly and the runtime hands the question to a person.
    process_documented: bool = True

    @property
    def terms(self) -> tuple[str, ...]:
        raw = (
            _norm(self.name),
            _norm(self.display_name or ""),
            _norm(self.slug),
            *(_norm(a) for a in self.aliases),
        )
        compact = tuple(term.replace(" ", "") for term in raw if len(term) >= 4)
        return tuple(dict.fromkeys((*raw, *compact)))

    @property
    def public_name(self) -> str:
        return self.display_name or self.name


@dataclass
class Catalog:
    persona_slug: str
    name: str | None = None
    greeting: str = "Oi! Como posso ajudar?"
    products: list[Product] = field(default_factory=list)
    categories: dict[str, list[str]] = field(default_factory=dict)

    def find_product(self, text: str) -> Product | None:
        query = _norm(text)
        query_tokens = set(query.split()) - {
            "a", "ao", "com", "custa", "de", "do", "e", "em", "o",
            "os", "por", "preco", "qual", "quanto", "valor",
        }
        if not query_tokens:
            return None
        ranked: list[tuple[int, int, Product]] = []
        for product in self.products:
            best_overlap = 0
            best_term_length = 0
            for term in product.terms:
                term_tokens = set(term.split())
                overlap = len(query_tokens & term_tokens)
                if term and term in query:
                    # A multi-word term appearing verbatim is a strong,
                    # specific signal regardless of message length. A
                    # single generic word (e.g. "pintura", the paint
                    # material â€” also this persona's full-repaint service
                    # name) is only trustworthy at that strength inside a
                    # short, direct message ("quero pintura"); inside a
                    # long descriptive sentence ("minha pintura ta toda
                    # fosca...") it's usually the customer describing
                    # their car, not naming a service â€” confirmed live
                    # 2026-08-01, where this false-positive silently
                    # locked in the wrong service before the agentic
                    # engine's own symptom-based inference could run.
                    if len(term_tokens) > 1 or len(query_tokens) <= 5:
                        overlap = max(overlap, len(term_tokens) + 10)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_term_length = len(term_tokens)
            if best_overlap:
                ranked.append((best_overlap, best_term_length, product))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_score, _, best_product = ranked[0]
        # A one-token match is safe only when that token identifies one product.
        if best_score == 1 and sum(score == 1 for score, _, _ in ranked) > 1:
            return None
        return best_product

    def find_category(self, text: str) -> str | None:
        query_tokens = {_singular_token(token) for token in _norm(text).split()}
        for category in self.categories:
            category_tokens = {
                _singular_token(token)
                for token in _norm(category).split()
                if token not in {"a", "as", "de", "do", "e", "em", "o", "os"}
            }
            if category_tokens and category_tokens.issubset(query_tokens):
                return category
        return None

    def in_category(self, category: str) -> list[Product]:
        return [p for p in self.products if _norm(p.category) == _norm(category) and p.status == "active"]


def _is_short_product_followup(message: str) -> bool:
    """Whether it is safe to inherit the product from an earlier turn.

    A full product question must never be silently replaced by an item mentioned
    in old chat history.  History is useful only for concise references after a
    product was just discussed (for example, ``quero 2``).
    """
    normalized = _norm(message)
    tokens = normalized.split()
    if not tokens or len(tokens) > 4:
        return False
    if _quantity(normalized) is not None:
        ignored = {
            "a", "as", "adiciona", "adicionar", "altera", "alterar",
            "dois", "duas", "muda", "mude", "o", "os", "para", "por",
            "quero", "tres", "uma", "um",
        }
        remaining = [token for token in tokens if not token.isdigit() and token not in ignored]
        return not remaining
    return normalized in {
        "esse", "essa", "este", "esta", "o mesmo", "a mesma",
        "tambem", "mais um", "adiciona", "adicionar",
    }


def _singular_token(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def _is_generic_category_query(message: str, category: str) -> bool:
    generic = {
        "a", "as", "de", "do", "e", "em", "lista", "listar", "me",
        "mostra", "mostrar", "o", "opcao", "opcoes", "opcoe", "os", "quais", "qual", "que",
        "quero", "tambem", "tem", "uma", "um",
    }
    category_tokens = {_singular_token(token) for token in _norm(category).split()}
    remaining = {
        _singular_token(token)
        for token in _norm(message).split()
        if _singular_token(token) not in category_tokens and token not in generic
    }
    return not remaining


def _price_from_graph_data(data: dict[str, Any], metadata: dict[str, Any]) -> tuple[float | None, str]:
    """Extract a validated product price without inventing commercial facts.

    Graph JSON v2 writes ``price.amount``.  Earlier published graph documents
    retained the same approved value as integer ``price_cents``; supporting it
    makes the reader backward compatible and avoids falsely claiming a known
    price is absent.
    """
    price = data.get("price") or metadata.get("price") or {}
    if isinstance(price, dict):
        amount = price.get("amount")
        if amount is None and isinstance(price.get("unit"), dict):
            amount = price["unit"].get("amount")
        if amount is not None:
            try:
                return float(amount), str(price.get("currency") or "BRL")
            except (TypeError, ValueError):
                pass
    cents = data.get("price_cents")
    if cents is None:
        cents = metadata.get("price_cents")
    if cents is not None:
        try:
            return float(cents) / 100, "BRL"
        except (TypeError, ValueError):
            pass
    return None, "BRL"


def _frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", raw, re.S)
    if not match:
        return {}, raw
    return (yaml.safe_load(match.group(1)) or {}), match.group(2)


def load_catalog(root: str | Path, persona_slug: str) -> Catalog:
    """Load only one persona's documents; never fall back to another persona."""
    base = Path(root) / persona_slug
    persona_meta: dict[str, Any] = {}
    persona_file = base / "persona.md"
    if persona_file.exists():
        persona_meta, _ = _frontmatter(persona_file)
    persona_config = persona_meta.get("metadata") or {}
    catalog = Catalog(
        persona_slug=persona_slug,
        name=persona_meta.get("name"),
        greeting=persona_meta.get("greeting")
        or persona_config.get("greeting")
        or "Oi! Como posso ajudar?",
    )
    product_dir = base / "products"
    for path in sorted(product_dir.glob("*.md")) if product_dir.exists() else []:
        meta, _ = _frontmatter(path)
        if meta.get("type") != "product" or str(meta.get("persona") or persona_slug) != persona_slug:
            continue
        approved = str(meta.get("status") or "").lower() in {
            "approved",
            "validated",
            "active",
            "ativo",
        }
        if not approved or meta.get("active", True) is False:
            continue
        price = meta.get("price") or {}
        metadata = meta.get("metadata") or {}
        catalog.products.append(Product(
            slug=str(meta.get("slug") or path.stem),
            name=str(meta.get("name") or meta.get("title") or path.stem),
            display_name=metadata.get("display_name"),
            aliases=tuple(str(x) for x in (meta.get("aliases") or [])), category=str(meta.get("category") or ""),
            unit=str(meta.get("unit") or ""), price=float(price["amount"]) if price.get("amount") is not None else None,
            currency=str(price.get("currency") or "BRL"), status="active",
        ))
    for path in [base / "catalog.md"]:
        if path.exists():
            meta, _ = _frontmatter(path)
            for category in meta.get("categories") or []:
                if isinstance(category, dict):
                    key = category.get("slug") or category.get("name") or category.get("title")
                    if key:
                        catalog.categories[str(key)] = [
                            str(x) for x in (category.get("products") or [])
                        ]
                else:
                    catalog.categories[str(category)] = []
    for product in catalog.products:
        catalog.categories.setdefault(product.category, []).append(product.slug)
    return catalog


def catalog_from_graph(graph: Any) -> Catalog:
    """Build the deterministic catalog only from one published Graph JSON."""
    persona = next(
        (node for node in graph.nodes if node.node_type == "persona"),
        None,
    )
    persona_data = (persona.data or {}) if persona else {}
    persona_metadata = persona_data.get("metadata") or {}
    catalog = Catalog(
        persona_slug=graph.persona_slug,
        name=(persona.label if persona else graph.persona_slug),
        greeting=persona_data.get("greeting")
        or persona_metadata.get("greeting")
        or "Oi! Como posso ajudar?",
    )
    category_by_id: dict[str, str] = {}
    for node in graph.nodes:
        if node.node_type != "product_group":
            continue
        category_by_id[node.id] = node.slug
        catalog.categories[node.slug] = []
    for node in graph.nodes:
        if node.node_type != "product":
            continue
        data = node.data or {}
        metadata = data.get("metadata") or {}
        if data.get("active", True) is False:
            continue
        if str(data.get("status") or "").lower() not in {
            "approved",
            "validated",
            "active",
            "ativo",
        }:
            continue
        unit_price, currency = _price_from_graph_data(data, metadata)
        category = category_by_id.get(node.parent_id or "") or str(
            data.get("category") or metadata.get("category_slug") or ""
        )
        product = Product(
            slug=node.slug,
            name=str(data.get("name") or data.get("title") or node.label),
            display_name=metadata.get("display_name"),
            aliases=tuple(str(value) for value in (data.get("aliases") or [])),
            category=category,
            unit=str(data.get("unit") or ""),
            price=unit_price,
            currency=currency,
            status="active",
            duration_minutes=(
                int((data.get("booking") or {}).get("duration_minutes"))
                if (data.get("booking") or {}).get("duration_minutes") is not None
                else None
            ),
            capacity=int((data.get("booking") or {}).get("capacity") or 1),
            price_qualifier=str(
                data.get("price_qualifier")
                or metadata.get("price_qualifier")
                or "fixed"
            ),
            required_fields=tuple(
                str(value)
                for value in (
                    (data.get("booking") or {}).get("required_fields")
                    or metadata.get("required_fields")
                    or []
                )
            ),
            confirmation_required=bool(
                (data.get("booking") or {}).get("confirmation_required", False)
            ),
            process_documented=bool(data.get("process_documented", True)),
            booking_provider=(
                str((data.get("booking") or {}).get("provider"))
                if (data.get("booking") or {}).get("provider")
                else None
            ),
        )
        catalog.products.append(product)
        catalog.categories.setdefault(category, []).append(product.slug)
    return catalog


def new_state() -> dict[str, Any]:
    return {"conversation_state": "discovering", "customer_name": None, "items": [], "address": {}, "confirmation_status": "pending"}


def _quantity(message: str) -> int | None:
    match = re.search(r"\b(\d{1,3})\b", _norm(message))
    if match:
        return int(match.group(1))
    words = {"uma": 1, "um": 1, "duas": 2, "dois": 2, "tres": 3, "quatro": 4, "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10}
    for word, value in words.items():
        if re.search(rf"\b{word}\b", _norm(message)):
            return value
    return None


def _is_yes(message: str) -> bool:
    return _norm(message) in {"sim", "s", "confirmo", "pode", "pode sim", "ok", "fechado", "confirmado"}


def _is_no(message: str) -> bool:
    return _norm(message) in {"nao", "n", "cancela", "cancelar", "deixa", "corrigir"}


def _address(message: str) -> dict[str, str]:
    result: dict[str, str] = {}
    only_number = re.fullmatch(r"\s*(\d{1,6})\s*", message)
    if only_number:
        return {"number": only_number.group(1)}
    number = re.search(r"(?:,|\s)(\d{1,6})(?:,|\s|$)", message)
    if number:
        result["number"] = number.group(1)
    if number:
        street = message[:number.start()].strip(" ,")
        if street:
            result["street"] = street
        tail = message[number.end():].strip(" ,")
        if tail:
            result["city"] = tail.split(",")[0].strip()
    elif len(message.split()) >= 2:
        result["street"] = message.strip()
    return result


class DeterministicSDR:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog

    def handle(self, message: str, *, state: dict[str, Any] | None = None, history: Iterable[dict[str, Any]] = ()) -> dict[str, Any]:
        text = (message or "").strip()
        state = {**new_state(), **(state or {})}
        state["items"] = [dict(item) for item in state.get("items") or []]
        intent = "ununderstood"
        normalized = _norm(text)
        if state.get("conversation_state") == "handoff":
            return {"reply": None, "intent": "handoff", "state": state, "handoff": False}
        if normalized in {"oi", "ola", "bom dia", "boa tarde", "boa noite"}:
            return {"reply": self.catalog.greeting, "intent": "greeting", "state": state, "handoff": False}
        if any(term in normalized for term in ("atendimento humano", "falar com alguem", "falar com uma pessoa", "humano")):
            state["conversation_state"] = "handoff"
            return {"reply": "Claro. Vou encaminhar vocÃª para o atendimento humano.", "intent": "request_human", "state": state, "handoff": True}
        product = self.catalog.find_product(text)
        category = self.catalog.find_category(text)
        if category and _is_generic_category_query(text, category):
            product = None
        # Addresses frequently contain catalog tokens or numbers (for example
        # "Rua QA, 100"). While collecting an address, a syntactically clear
        # address wins over a fuzzy product match. Explicit category/product
        # interruptions without address syntax still work.
        address_candidate = _address(text)
        address_prefix = re.match(
            r"^(rua|avenida|av|travessa|rodovia|estrada)\b",
            normalized,
        )
        if (
            state.get("conversation_state") == "awaiting_address"
            and address_candidate
            and (address_candidate.get("number") or address_prefix)
        ):
            product = None
            category = None
        # Only concise references may inherit a product from cart/history.
        address_number = (
            state.get("conversation_state") == "awaiting_address"
            and re.fullmatch(r"\s*\d{1,6}\s*", text) is not None
        )
        if not product and not category and not address_number and _is_short_product_followup(text):
            last_slug = next((i.get("product_slug") for i in reversed(state["items"]) if i.get("product_slug")), None)
            if not last_slug:
                for turn in reversed(list(history)):
                    if str(turn.get("sender_type") or "").lower() in {"lead", "client", "user"}:
                        candidate = self.catalog.find_product(str(turn.get("texto") or turn.get("content") or ""))
                        if candidate:
                            last_slug = candidate.slug
                            break
            product = next((p for p in self.catalog.products if p.slug == last_slug), None)
        if state["items"] and normalized in {"quanto fica", "qual o total", "total"}:
            total = sum((i.get("unit_price") or 0) * int(i.get("quantity") or 0) for i in state["items"])
            return {"reply": f"O total parcial Ã© {_brl(total)}.", "intent": "consult_price", "state": state, "handoff": False}
        if state.get("conversation_state") == "awaiting_confirmation" and not product and not category:
            if _is_yes(text):
                state["conversation_state"] = "handoff"
                state["confirmation_status"] = "confirmed_pending_human"
                total = sum(
                    (item.get("unit_price") or 0)
                    * int(item.get("quantity") or 0)
                    for item in state["items"]
                )
                summary = ", ".join(
                    f"{item['quantity']} "
                    f"{next((product.public_name for product in self.catalog.products if product.slug == item['product_slug']), item['product_slug'])}"
                    for item in state["items"]
                )
                return {
                    "reply": (
                        f"Pedido confirmado: {summary}, total de {_brl(total)}. "
                        "Vou encaminhar para o atendimento finalizar."
                    ),
                    "intent": "confirm_order",
                    "state": state,
                    "handoff": True,
                }
            if _is_no(text):
                state["conversation_state"] = "building_order"
                state["confirmation_status"] = "cancelled"
                return {"reply": "Tudo bem. O que vocÃª deseja corrigir no pedido?", "intent": "deny_order", "state": state, "handoff": False}
        if state.get("conversation_state") == "awaiting_address" and not product and not category:
            addr = _address(text)
            state["address"] = {**(state.get("address") or {}), **addr}
            if not state["address"].get("street"):
                return {"reply": "Qual Ã© a rua ou avenida do endereÃ§o?", "intent": "provide_address", "state": state, "handoff": False}
            if not state["address"].get("number"):
                return {"reply": "Qual Ã© o nÃºmero do endereÃ§o?", "intent": "provide_address", "state": state, "handoff": False}
            state["conversation_state"] = "awaiting_confirmation"
            total = sum((i.get("unit_price") or 0) * int(i.get("quantity") or 0) for i in state["items"])
            summary = ", ".join(f"{i['quantity']} {next((p.public_name for p in self.catalog.products if p.slug == i['product_slug']), i['product_slug'])}" for i in state["items"])
            return {"reply": f"Pedido: {summary}, total de {_brl(total)}, para {text}. Posso confirmar este pedido?", "intent": "provide_address", "state": state, "handoff": False}
        if state.get("conversation_state") == "awaiting_name" and re.fullmatch(r"[A-Za-zÃ€-Ã¿]{2,}(?:\s+[A-Za-zÃ€-Ã¿]{2,}){0,2}", text):
            state["customer_name"] = text
            if state["items"]:
                state["conversation_state"] = "awaiting_address"
                return {"reply": f"Obrigada, {text}. Qual endereÃ§o devo usar?", "intent": "provide_name", "state": state, "handoff": False}
        if product:
            qty = _quantity(text) or 1
            is_price_question = any(word in normalized for word in ("preco", "custa", "valor", "quanto fica"))
            if is_price_question and product.price is not None:
                return {"reply": f"{product.public_name}: {_brl(product.price)}.", "intent": "consult_price", "state": state, "handoff": False}
            if product.price is None:
                return {"reply": f"NÃ£o encontrei um preÃ§o aprovado para {product.public_name}. VocÃª quer consultar outro item?", "intent": "consult_price", "state": state, "handoff": False}
            if any(word in normalized for word in ("remove", "tirar", "excluir")):
                state["items"] = [i for i in state["items"] if i.get("product_slug") != product.slug]
                return {"reply": f"Certo, removi {product.public_name} do pedido.", "intent": "remove_item", "state": state, "handoff": False}
            existing = next((i for i in state["items"] if i.get("product_slug") == product.slug), None)
            if existing and _quantity(text):
                existing["quantity"] = qty
                intent = "change_quantity"
            elif not existing:
                state["items"].append({"product_slug": product.slug, "quantity": qty, "unit_price": product.price})
                intent = "add_item"
            else:
                intent = "consult_product"
            if len(state["items"]) == 1 and not _quantity(text) and not state.get("customer_name"):
                return {"reply": f"Sim. {product.public_name}{(' ' + product.unit) if product.unit else ''} â€” {_brl(product.price)}. Quantas unidades vocÃª deseja?", "intent": "consult_product", "state": state, "handoff": False}
            total = sum((i.get("unit_price") or 0) * int(i.get("quantity") or 0) for i in state["items"])
            summary = ", ".join(f"{i['quantity']} {next((p.public_name for p in self.catalog.products if p.slug == i['product_slug']), i['product_slug'])}" for i in state["items"])
            address = dict(state.get("address") or {})
            if (
                state.get("customer_name")
                and address.get("street")
                and address.get("number")
            ):
                state["conversation_state"] = "awaiting_confirmation"
                destination = f"{address['street']}, {address['number']}"
                return {
                    "reply": (
                        f"Pedido: {summary}, total de {_brl(total)}, para "
                        f"{destination}. Posso confirmar este pedido?"
                    ),
                    "intent": intent,
                    "state": state,
                    "handoff": False,
                }
            state["conversation_state"] = "awaiting_address" if state.get("customer_name") else "awaiting_name"
            next_question = f"Obrigada, {state['customer_name']}. Qual endereÃ§o devo usar?" if state.get("customer_name") else "Qual Ã© o seu nome?"
            return {"reply": f"Certo: {summary}, total de {_brl(total)}. {next_question}", "intent": intent, "state": state, "handoff": False}
        if category:
            products = self.catalog.in_category(category)
            listed = "\n".join(
                f"- {p.public_name} â€” {_brl(p.price)}" if p.price is not None else f"- {p.public_name}"
                for p in products
            ) or "- nenhum item aprovado no momento"
            return {"reply": f"Em {category}, encontrei:\n{listed}\nQual item vocÃª deseja?", "intent": "consult_category", "state": state, "handoff": False}
        suggestions = ", ".join(sorted(self.catalog.categories)[:3])
        return {"reply": f"NÃ£o localizei esse item. Posso ajudar com: {suggestions}. VocÃª quer outro produto?", "intent": intent, "state": state, "handoff": False}

