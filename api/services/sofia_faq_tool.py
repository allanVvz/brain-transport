"""Sofia FAQ tool â€” `adaptar_faqs_universais_ao_grafo`.

Turns universal commercial questions into FAQs adapted to the *active branch* of
a persona's graph. It never invents a loose/generic FAQ: every suggestion is
anchored to the most specific node available (Product > Product Group > whatever
lower node exists) and worded with the real product/brand context.

Everything here is pure over already-loaded graph data (nodes + edges), so it is
unit-testable without a live Supabase. Persistence and the HTTP surface live in
the route layer; this module only resolves context and produces suggestions.

Authoritative doc: `agents/Sofia Tool â€” AdaptaÃ§Ã£o de FAQs Comerciais Universais
ao Grafo da Persona.md` (empty at time of writing â€” this is the minimal
implementation derived from the task rules).
"""
from __future__ import annotations

import re
from typing import Any, Optional

SOURCE_TOOL = "adaptar_faqs_universais_ao_grafo"
DEFAULT_FAQ_COUNT = 5
MAX_FAQ_COUNT = 20

# Node types that can anchor a FAQ, from most to least specific.
_PARENT_PREFERENCE = ["product", "offer", "product_group", "copy", "campaign", "audience", "briefing", "brand"]

# Universal commercial questions. Each adapts to {product} (the anchor's label)
# and {brand}. Kept deterministic so "gerar novamente" is reproducible per slot.
_COMMERCIAL_FAQ_TEMPLATES: list[dict[str, str]] = [
    {
        "question": "Como faÃ§o para comprar o {product} na {brand}?",
        "answer": "VocÃª pode solicitar atendimento pela {brand}, confirmar a disponibilidade do {product} e finalizar a compra com envio para a sua regiÃ£o.",
    },
    {
        "question": "O {product} tem proteÃ§Ã£o UV?",
        "answer": "Sim, o {product} oferece proteÃ§Ã£o contra raios UV. Confirme os detalhes tÃ©cnicos com o atendimento da {brand} antes de finalizar.",
    },
    {
        "question": "O {product} acompanha caixa e flanela de microfibra?",
        "answer": "O {product} normalmente acompanha case e flanela de microfibra. A {brand} confirma os itens inclusos no momento da compra.",
    },
    {
        "question": "Qual Ã© o prazo de envio do {product}?",
        "answer": "O prazo de envio do {product} depende da sua regiÃ£o. A {brand} informa o prazo estimado assim que o pedido Ã© confirmado.",
    },
    {
        "question": "O {product} combina mais com uso casual ou para praia/esporte?",
        "answer": "O {product} Ã© versÃ¡til e funciona bem tanto para o dia a dia quanto para praia e esportes. O atendimento da {brand} ajuda a escolher conforme o seu uso.",
    },
    {
        "question": "Posso pedir atendimento antes de comprar o {product}?",
        "answer": "Sim. Fale com a {brand} antes da compra para tirar dÃºvidas sobre o {product}, disponibilidade e formas de pagamento.",
    },
    {
        "question": "O {product} tem garantia?",
        "answer": "O {product} conta com garantia. As condiÃ§Ãµes e o prazo sÃ£o confirmados pela {brand} no momento da compra.",
    },
    {
        "question": "Quais formas de pagamento a {brand} aceita para o {product}?",
        "answer": "A {brand} oferece diferentes formas de pagamento para o {product}. Consulte o atendimento para ver as opÃ§Ãµes disponÃ­veis e parcelamento.",
    },
    {
        "question": "O {product} Ã© original?",
        "answer": "Sim, o {product} vendido pela {brand} Ã© original. VocÃª pode confirmar a procedÃªncia diretamente com o atendimento.",
    },
    {
        "question": "Como faÃ§o a troca ou devoluÃ§Ã£o do {product}?",
        "answer": "Caso precise trocar ou devolver o {product}, a {brand} orienta o passo a passo dentro do prazo previsto apÃ³s o recebimento.",
    },
    {
        "question": "O {product} estÃ¡ disponÃ­vel em outras cores ou variaÃ§Ãµes?",
        "answer": "O {product} pode ter outras variaÃ§Ãµes disponÃ­veis. Pergunte ao atendimento da {brand} quais opÃ§Ãµes existem em estoque.",
    },
    {
        "question": "VocÃªs entregam o {product} para todo o Brasil?",
        "answer": "A {brand} envia o {product} para diversas regiÃµes. Confirme a cobertura de entrega para o seu endereÃ§o com o atendimento.",
    },
]

_COMMERCIAL_FAQ_TEMPLATES = [
    {
        "question": "Como faco para comprar o {product} na {brand}?",
        "answer": "Fale com a {brand} e informe que tem interesse no {product}. A equipe confirma disponibilidade, valores, variacoes e proximo passo do pedido. {context}",
    },
    {
        "question": "O {product} e indicado para quem quer revender?",
        "answer": "Pode ser indicado quando as condicoes comerciais, quantidade e margem fizerem sentido para o perfil da compra. A {brand} confirma os kits e valores atualizados. {context}",
    },
    {
        "question": "Quais opcoes de quantidade existem para o {product}?",
        "answer": "As opcoes podem variar por estoque e condicao comercial. O atendimento da {brand} confirma se ha unidade, kit ou lote disponivel para o {product}. {context}",
    },
    {
        "question": "Quais cores, tamanhos ou variacoes do {product} estao disponiveis?",
        "answer": "A disponibilidade muda conforme estoque. Peca para a {brand} confirmar as variacoes atuais do {product} antes de fechar o pedido. {context}",
    },
    {
        "question": "Por que o {product} pode ser uma boa escolha?",
        "answer": "A melhor resposta deve conectar o {product} ao objetivo da pessoa, como uso proprio, presente, revenda ou reposicao de estoque. {context}",
    },
    {
        "question": "Posso pedir atendimento antes de comprar o {product}?",
        "answer": "Sim. Fale com a {brand} antes da compra para tirar duvidas sobre o {product}, disponibilidade e formas de pagamento. {context}",
    },
    {
        "question": "O que preciso confirmar antes de fechar o pedido do {product}?",
        "answer": "Confirme disponibilidade, valores, variacoes, quantidade, forma de pagamento e prazo diretamente com a {brand}. {context}",
    },
    {
        "question": "Quais formas de pagamento a {brand} aceita para o {product}?",
        "answer": "A {brand} informa as formas de pagamento atuais para o {product}, incluindo eventuais condicoes por quantidade ou kit.",
    },
    {
        "question": "O {product} esta disponivel para pronta entrega?",
        "answer": "A disponibilidade precisa ser confirmada no atendimento da {brand}, porque estoque e variacoes podem mudar.",
    },
    {
        "question": "Como funciona troca, garantia ou devolucao do {product}?",
        "answer": "A {brand} orienta as condicoes aplicaveis antes da compra, junto com disponibilidade, valores e prazo.",
    },
]


def clamp_count(count: Optional[int]) -> int:
    try:
        value = int(count) if count is not None else DEFAULT_FAQ_COUNT
    except (TypeError, ValueError):
        value = DEFAULT_FAQ_COUNT
    if value < 1:
        value = 1
    return min(value, MAX_FAQ_COUNT)


def _node_type(node: Optional[dict]) -> str:
    return str((node or {}).get("node_type") or (node or {}).get("data", {}).get("node_type") or "").strip().lower()


def _node_label(node: Optional[dict]) -> str:
    node = node or {}
    data = node.get("data") if isinstance(node.get("data"), dict) else {}
    return str(
        node.get("title")
        or node.get("label")
        or data.get("label")
        or data.get("title")
        or node.get("slug")
        or data.get("slug")
        or ""
    ).strip()


def node_markdown(node: Optional[dict]) -> str:
    """The contextual Markdown body of a node (metadata.markdown is canonical,
    summary is the fallback)."""
    node = node or {}
    meta = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    data = node.get("data") if isinstance(node.get("data"), dict) else {}
    data_meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    return str(
        (meta or {}).get("markdown")
        or (meta or {}).get("body")
        or data_meta.get("markdown")
        or data_meta.get("body")
        or data.get("markdown")
        or node.get("summary")
        or data.get("content_preview")
        or ""
    ).strip()


def _context_snippet(markdown: str) -> str:
    text = (markdown or "").strip()
    if not text:
        return ""
    first = next((line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")), "")
    if not first:
        return ""
    if len(first) > 180:
        first = first[:177].rstrip() + "â€¦"
    return f"Contexto confirmado: {first}"


def _clean_branch_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return ""
    text = re.sub(
        r"^(copy|briefing|campanha|faq)\s+(de\s+)?(divulgacao|divulga[cÃ§][aÃ£]o|atendimento)?\s*(para|sobre)?\s*",
        "",
        text,
        flags=re.I,
    ).strip(" .:-")
    return text[:220].rstrip()


def _clean_branch_label(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().strip("'\"`")


def _branch_customer_context(context: dict[str, Any], sellable: Optional[dict]) -> str:
    """Summarize the whole branch in customer-facing terms."""
    path = [entry for entry in (context.get("path") or []) if isinstance(entry, dict)]
    by_type: dict[str, dict] = {}
    for entry in path:
        ctype = str(entry.get("node_type") or "").lower()
        if ctype and ctype not in by_type:
            by_type[ctype] = entry

    parts: list[str] = []
    brand = _clean_branch_label((by_type.get("brand") or {}).get("label") or context.get("brand"))
    audience = _clean_branch_label((by_type.get("audience") or {}).get("label"))
    product_group = _clean_branch_label((by_type.get("product_group") or {}).get("label") or context.get("product_group"))
    product = _clean_branch_label((sellable or {}).get("name") or context.get("product"))
    if brand:
        parts.append(f"marca {brand}")
    if audience:
        parts.append(f"publico {audience}")
    if product_group:
        parts.append(f"grupo {product_group}")
    if product:
        parts.append(f"produto {product}")

    return "; ".join(parts).strip()


def detect_faq_generation_intent(command: str) -> Optional[dict[str, Any]]:
    """Detect FAQ generate/regenerate/update intents and an optional quantity.

    Returns {"intent": <name>, "count": <int|None>} or None when the command is
    not about FAQ generation. Intents map to the task's vocabulary:
      gerar_faqs_para_node / gerar_n_faqs_para_node / regenerar_faq /
      gerar_novamente_faqs_do_node / atualizar_faq.
    """
    text = (command or "").strip().lower()
    if not text:
        return None

    count = _extract_count(text)
    mentions_faq = bool(re.search(r"\bfaqs?\b", text) or "pergunta" in text)
    gen_verb = any(v in text for v in ("gere", "gerar", "gera ", "crie", "criar", "cria "))
    again = "novamente" in text or "de novo" in text or "regener" in text or "regere" in text
    update_verb = any(v in text for v in ("atualiz", "melhore", "melhorar", "refaz", "refazer", "reescrev"))

    if again and (mentions_faq or "perguntas" in text):
        return {"intent": "gerar_novamente_faqs_do_node", "count": count}
    if gen_verb and mentions_faq:
        return {"intent": "gerar_n_faqs_para_node" if count else "gerar_faqs_para_node", "count": count}
    if update_verb and mentions_faq:
        return {"intent": "atualizar_faq", "count": count}
    return None


def _extract_count(text: str) -> Optional[int]:
    match = re.search(r"(\d+)\s*(perguntas|pergunta|faqs|faq)", text)
    if match:
        return clamp_count(int(match.group(1)))
    match = re.search(r"\b(\d+)\b", text)
    if match and ("pergunta" in text or "faq" in text):
        return clamp_count(int(match.group(1)))
    return None


def _build_parent_index(nodes: list[dict], edges: list[dict]) -> dict[str, str]:
    """child_node_id -> parent_node_id using primary/structural edges."""
    parent_by_child: dict[str, str] = {}
    for edge in edges or []:
        meta = edge.get("metadata") or {}
        if meta.get("active") is False:
            continue
        src = edge.get("source_node_id") or edge.get("source")
        tgt = edge.get("target_node_id") or edge.get("target")
        if src and tgt and tgt not in parent_by_child:
            parent_by_child[tgt] = src
    return parent_by_child


def select_faq_parent(target_node: dict, nodes: list[dict], edges: list[dict]) -> dict:
    """Pick the most specific node to anchor FAQs onto.

    - A product/offer/group target anchors on itself.
    - A FAQ target climbs to its parent (the product/group it belongs to).
    - Anything else anchors on itself (lower node available rule).
    Never returns None: the active branch always has at least the target node.
    """
    ttype = _node_type(target_node)
    if ttype == "faq":
        parent_by_child = _build_parent_index(nodes, edges)
        nodes_by_id = {n.get("id"): n for n in nodes}
        current = target_node.get("id")
        # Climb until we hit a non-faq anchor (product preferred).
        seen = set()
        while current and current not in seen:
            seen.add(current)
            parent_id = parent_by_child.get(current)
            parent = nodes_by_id.get(parent_id)
            if parent and _node_type(parent) != "faq":
                return parent
            current = parent_id
        return target_node
    return target_node


def build_branch_context(parent_node: dict, nodes: list[dict], edges: list[dict]) -> dict[str, Any]:
    """Walk up the branch from the anchor to collect brand/product context."""
    parent_by_child = _build_parent_index(nodes, edges)
    nodes_by_id = {n.get("id"): n for n in nodes}
    context: dict[str, Any] = {
        "anchor_label": _node_label(parent_node),
        "anchor_type": _node_type(parent_node),
        "brand": None,
        "product": None,
        "product_group": None,
        "path": [],
        "markdown_by_type": {},
        "nearest_markdown": "",
        "branch_markdown": "",
    }
    branch_md_blocks: list[str] = []
    current = parent_node
    seen = set()
    while current and current.get("id") not in seen:
        seen.add(current.get("id"))
        ctype = _node_type(current)
        label = _node_label(current)
        md = node_markdown(current)
        entry = {"node_type": ctype, "label": label, "id": current.get("id"), "markdown": md}
        context["path"].insert(0, entry)
        if md:
            context["markdown_by_type"].setdefault(ctype, md)
            branch_md_blocks.insert(0, f"## {ctype}: {label}\n{md}")
            if not context["nearest_markdown"]:
                context["nearest_markdown"] = md
        if ctype == "brand" and not context["brand"]:
            context["brand"] = label
        if ctype in ("product", "offer") and not context["product"]:
            context["product"] = label
        if ctype == "product_group" and not context["product_group"]:
            context["product_group"] = label
        current = nodes_by_id.get(parent_by_child.get(current.get("id")))
    context["branch_markdown"] = "\n\n".join(branch_md_blocks).strip()
    return context


# Node types that represent a real sellable/commercial object. Only these (or a
# product_group that actually has products) unlock purchase/shipping/warranty
# templates. Audience/Brand/Briefing are NOT sellable.
_SELLABLE_TYPES = {"product", "offer", "service", "course", "event"}

# Substrings that must never appear in a FAQ for a non-sellable node (audience,
# brand, briefing, discovery). Guardrail against "Como comprar o TÃ©cnicos?".
_COMMERCIAL_GUARDRAIL_TOKENS = (
    "frete",
    "acompanha caixa",
    "acompanha case",
    "flanela",
    "prazo de envio",
    "prazo de entrega",
    "parcelar",
    "Ã  vista",
    "boleto",
)

# â”€â”€ Template sets per node category â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_PRODUCT_GROUP_TEMPLATES = [
    {"question": "Qual a diferenÃ§a entre os modelos de {subject} disponÃ­veis?", "answer": "Os modelos de {subject} variam em estilo e especificaÃ§Ã£o. A {brand} ajuda a comparar as opÃ§Ãµes e indicar a melhor para vocÃª."},
    {"question": "Quais opÃ§Ãµes de {subject} a {brand} oferece?", "answer": "A {brand} trabalha com diferentes versÃµes de {subject}. Consulte o atendimento para ver o que estÃ¡ disponÃ­vel em estoque."},
    {"question": "Qual {subject} Ã© mais indicado para uso casual?", "answer": "Depende do seu perfil de uso. A {brand} recomenda o modelo de {subject} mais adequado conforme a sua necessidade."},
    {"question": "Qual Ã© a faixa de preÃ§o dos modelos de {subject}?", "answer": "Os preÃ§os de {subject} variam conforme o modelo. O atendimento da {brand} informa os valores atualizados."},
    {"question": "Como escolher o {subject} ideal para o meu perfil?", "answer": "Conte para a {brand} como pretende usar o {subject} e o atendimento orienta a melhor escolha."},
]

_OFFER_TEMPLATES = [
    {"question": "AtÃ© quando a oferta {subject} fica disponÃ­vel?", "answer": "A oferta {subject} tem prazo limitado. Confirme a data final com o atendimento da {brand}."},
    {"question": "Qual Ã© o desconto da oferta {subject}?", "answer": "A oferta {subject} tem condiÃ§Ã£o especial. O atendimento da {brand} detalha o desconto aplicado."},
    {"question": "Quais sÃ£o as condiÃ§Ãµes da oferta {subject}?", "answer": "As regras da oferta {subject} sÃ£o confirmadas pela {brand} antes da compra."},
    {"question": "A oferta {subject} tem limite de quantidade?", "answer": "A oferta {subject} pode ter quantidade limitada. Consulte a {brand} sobre a disponibilidade."},
    {"question": "A oferta {subject} inclui algum bÃ´nus?", "answer": "A {brand} informa se a oferta {subject} acompanha algum bÃ´nus ou vantagem adicional."},
]

_CAMPAIGN_TEMPLATES = [
    {"question": "Como funciona a campanha {subject}?", "answer": "A campanha {subject} tem regras prÃ³prias. A {brand} explica como participar e quais sÃ£o os benefÃ­cios."},
    {"question": "Qual Ã© o objetivo da campanha {subject}?", "answer": "A campanha {subject} foi criada com um objetivo especÃ­fico. O atendimento da {brand} detalha a proposta."},
    {"question": "Qual o benefÃ­cio de participar da campanha {subject}?", "answer": "Participar da campanha {subject} traz vantagens divulgadas pela {brand}."},
    {"question": "AtÃ© quando vai a campanha {subject}?", "answer": "A campanha {subject} tem perÃ­odo definido. Confirme as datas com a {brand}."},
    {"question": "Para qual pÃºblico Ã© a campanha {subject}?", "answer": "A campanha {subject} Ã© direcionada a um pÃºblico especÃ­fico. A {brand} esclarece para quem ela faz sentido."},
    {"question": "Como faÃ§o para participar da campanha {subject}?", "answer": "Fale com a {brand} para entender o passo a passo de participaÃ§Ã£o na campanha {subject}."},
]

_BRAND_TEMPLATES = [
    {"question": "Por que escolher a {brand}?", "answer": "A {brand} se diferencia pela proposta e pelo atendimento. Fale com a equipe para conhecer os diferenciais."},
    {"question": "Como funciona o atendimento da {brand}?", "answer": "A {brand} oferece atendimento para tirar dÃºvidas e orientar sua decisÃ£o."},
    {"question": "Quais sÃ£o os diferenciais da {brand}?", "answer": "A {brand} tem diferenciais prÃ³prios de posicionamento e serviÃ§o, apresentados pelo atendimento."},
    {"question": "Por quais canais consigo falar com a {brand}?", "answer": "A {brand} disponibiliza canais de contato para atendimento â€” consulte qual Ã© o melhor para vocÃª."},
    {"question": "A {brand} Ã© confiÃ¡vel?", "answer": "A {brand} preza por transparÃªncia no atendimento e na relaÃ§Ã£o com o pÃºblico."},
]

_BRIEFING_TEMPLATES = [
    {"question": "Qual Ã© a proposta descrita neste briefing?", "answer": "Este briefing define a proposta e o direcionamento do trabalho. {context}"},
    {"question": "Para qual pÃºblico este briefing foi pensado?", "answer": "O briefing descreve o pÃºblico-alvo pretendido. {context}"},
    {"question": "Qual problema este briefing quer resolver?", "answer": "O objetivo do briefing Ã© endereÃ§ar um problema especÃ­fico. {context}"},
    {"question": "Qual Ã© o posicionamento sugerido neste briefing?", "answer": "O briefing indica um posicionamento a ser seguido. {context}"},
    {"question": "Que tipo de oferta este briefing prepara?", "answer": "O briefing prepara o terreno para uma oferta futura. {context}"},
]

_COPY_TEMPLATES = [
    {"question": "Qual Ã© a principal promessa desta copy?", "answer": "A copy apresenta uma promessa central ao leitor. {context}"},
    {"question": "Qual argumento esta copy usa para convencer?", "answer": "A copy se apoia em um argumento principal. {context}"},
    {"question": "Que objeÃ§Ã£o esta copy procura responder?", "answer": "A copy antecipa e responde a uma objeÃ§Ã£o comum. {context}"},
    {"question": "Para quem esta copy foi escrita?", "answer": "A copy fala diretamente com um pÃºblico especÃ­fico. {context}"},
    {"question": "Qual aÃ§Ã£o esta copy pede ao leitor?", "answer": "A copy conduz o leitor a uma aÃ§Ã£o clara. {context}"},
]

_AUDIENCE_TEMPLATES = [
    {"question": "Quais temas o pÃºblico {audience} costuma acompanhar mais de perto?", "answer": "Mapear os interesses de {descriptor} ajuda a Sofia a oferecer conteÃºdo e soluÃ§Ãµes relevantes."},
    {"question": "Esse pÃºblico estÃ¡ mais interessado em adquirir produtos, aprender, resolver problemas ou se atualizar?", "answer": "Entender a intenÃ§Ã£o de {descriptor} define se a abordagem deve ser de venda, educaÃ§Ã£o ou suporte."},
    {"question": "Que nÃ­vel de explicaÃ§Ã£o a Sofia deve usar com {descriptor}?", "answer": "Calibrar a linguagem para {descriptor} evita ser raso demais ou tÃ©cnico demais."},
    {"question": "Quais dÃºvidas o pÃºblico {audience} costuma ter antes de escolher um produto, curso ou serviÃ§o?", "answer": "Conhecer as dÃºvidas de {descriptor} permite preparar respostas que destravam a decisÃ£o."},
    {"question": "O pÃºblico {audience} jÃ¡ tem uma soluÃ§Ã£o em mente ou precisa de recomendaÃ§Ã£o?", "answer": "Saber o estÃ¡gio de {descriptor} orienta entre recomendar ou apenas confirmar a escolha."},
    {"question": "O pÃºblico {audience} prefere explicaÃ§Ã£o tÃ©cnica ou linguagem simples?", "answer": "Ajustar o tom para {descriptor} melhora a compreensÃ£o e a confianÃ§a."},
]

_AUDIENCE_OBJECT_TEMPLATES = [
    {"question": "Quais {object} fazem mais sentido para o pÃºblico {audience}?", "answer": "Relacionar {object} ao perfil de {descriptor} ajuda a recomendar a opÃ§Ã£o certa."},
    {"question": "O pÃºblico {audience} busca {object} para uso pessoal ou profissional?", "answer": "O contexto de uso de {object} por {descriptor} muda a recomendaÃ§Ã£o."},
    {"question": "Quais dÃºvidas o pÃºblico {audience} costuma ter sobre {object}?", "answer": "Antecipar dÃºvidas de {descriptor} sobre {object} acelera a decisÃ£o."},
    {"question": "O pÃºblico {audience} prioriza preÃ§o, qualidade ou suporte ao escolher {object}?", "answer": "Entender a prioridade de {descriptor} orienta como apresentar {object}."},
    {"question": "Como apresentar {object} para o pÃºblico {audience}?", "answer": "A forma de apresentar {object} deve falar a lÃ­ngua de {descriptor}."},
]

_DISCOVERY_TEMPLATES = [
    {"question": "Qual Ã© o principal interesse relacionado a {subject}?", "answer": "Entender o interesse central em {subject} guia as prÃ³ximas perguntas e respostas. {context}"},
    {"question": "Que tipo de dÃºvida costuma surgir sobre {subject}?", "answer": "Mapear as dÃºvidas sobre {subject} ajuda a preparar respostas Ãºteis. {context}"},
    {"question": "Qual contexto Ã© mais importante entender sobre {subject}?", "answer": "Clarear o contexto de {subject} evita respostas genÃ©ricas. {context}"},
    {"question": "Que objetivo a pessoa costuma ter ao buscar {subject}?", "answer": "Saber o objetivo por trÃ¡s de {subject} direciona a conversa. {context}"},
]


def _build_children_index(nodes: list[dict], edges: list[dict]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {}
    for edge in edges or []:
        meta = edge.get("metadata") or {}
        if meta.get("active") is False:
            continue
        src = edge.get("source_node_id") or edge.get("source")
        tgt = edge.get("target_node_id") or edge.get("target")
        if src and tgt:
            children.setdefault(src, []).append(tgt)
    return children


def find_sellable_in_branch(parent_node: dict, nodes: list[dict], edges: list[dict], context: dict[str, Any]) -> Optional[dict]:
    """Look for a real sellable object in the branch â€” ancestors first, then
    descendants. A product_group only counts when it actually has a product."""
    nodes_by_id = {n.get("id"): n for n in nodes}
    for entry in context.get("path", []):
        if str(entry.get("node_type") or "").lower() in _SELLABLE_TYPES:
            return {"type": entry["node_type"], "name": entry["label"]}
    children = _build_children_index(nodes, edges)
    stack = list(children.get(parent_node.get("id"), []))
    seen: set[str] = set()
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        node = nodes_by_id.get(nid)
        ntype = _node_type(node)
        if ntype in _SELLABLE_TYPES:
            return {"type": ntype, "name": _node_label(node)}
        stack.extend(children.get(nid, []))
    return None


def classify_faq_target(parent_node: dict, sellable: Optional[dict]) -> str:
    """Map a node to a FAQ category so templates match the real object."""
    ntype = _node_type(parent_node)
    if ntype in {"product", "service", "course", "event"}:
        return "product"
    if ntype == "offer":
        return "offer"
    if ntype == "product_group":
        return "product_group"
    if ntype == "campaign":
        return "campaign"
    if ntype == "brand":
        return "brand"
    if ntype == "briefing":
        return "briefing"
    if ntype == "copy":
        return "copy"
    if ntype == "audience":
        return "audience_object" if sellable else "audience"
    return "discovery"


def _audience_descriptor(name: str, markdown: str) -> str:
    text = (markdown or "").strip()
    if not text:
        return f"o pÃºblico {name}" if name else "esse pÃºblico"
    first = next((line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")), "")
    sentence = first.split(".")[0].strip() if first else ""
    if sentence:
        return sentence[0].lower() + sentence[1:] if len(sentence) > 1 else sentence
    return f"o pÃºblico {name}" if name else "esse pÃºblico"


def _violates_commercial_guardrail(question: str, node_name: str) -> bool:
    q = (question or "").lower()
    name = (node_name or "").strip().lower()
    if any(token in q for token in _COMMERCIAL_GUARDRAIL_TOKENS):
        return True
    if name:
        for verb in ("comprar", "garantia d", "preÃ§o d", "preco d"):
            # forbid only when the verb targets the node itself, e.g. "comprar o TÃ©cnicos"
            for connector in (f"{verb} o {name}", f"{verb} a {name}", f"{verb} {name}"):
                if connector in q:
                    return True
    return False


def adaptar_faqs_universais_ao_grafo(
    *,
    target_node: dict,
    nodes: list[dict],
    edges: list[dict],
    count: Optional[int] = None,
    persona_slug: Optional[str] = None,
) -> dict[str, Any]:
    """Produce N FAQ suggestions adapted to the target node's *type* and the real
    commercial object of its branch.

    The category (product / product_group / offer / campaign / brand / briefing /
    copy / audience / audience_object / discovery) decides which template set is
    used, so an Audience is never treated as a buyable Product. Purchase /
    shipping / warranty wording is unlocked only when a real sellable object
    exists in the branch.
    """
    n = clamp_count(count)
    parent = select_faq_parent(target_node, nodes, edges)
    context = build_branch_context(parent, nodes, edges)
    sellable = find_sellable_in_branch(parent, nodes, edges, context)
    category = classify_faq_target(parent, sellable)
    if _node_type(target_node) == "faq" and _node_type(parent) == "copy" and sellable:
        category = "product"

    label = _node_label(parent)
    brand = context.get("brand") or "loja"
    factual_markdown = (
        (context.get("markdown_by_type") or {}).get((sellable or {}).get("type"))
        or context.get("nearest_markdown")
        or ""
    )
    context_parts = [
        _branch_customer_context(context, sellable),
        _context_snippet(factual_markdown),
    ]
    context_snippet = ". ".join(part for part in context_parts if part)
    nearest_md = context.get("nearest_markdown") or ""

    if category in {"product", "offer"}:
        subject = context.get("product") or context.get("product_group") or label or "produto"
        templates = _COMMERCIAL_FAQ_TEMPLATES if category == "product" else _OFFER_TEMPLATES
    elif category == "product_group":
        subject = context.get("product_group") or label
        templates = _PRODUCT_GROUP_TEMPLATES
    elif category == "campaign":
        subject = label
        templates = _CAMPAIGN_TEMPLATES
    elif category == "brand":
        subject = brand = context.get("brand") or label
        templates = _BRAND_TEMPLATES
    elif category == "briefing":
        subject = label
        templates = _BRIEFING_TEMPLATES
    elif category == "copy":
        subject = label
        templates = _copy_templates_for(nearest_md)
    elif category == "audience":
        subject = label
        templates = _AUDIENCE_TEMPLATES
    elif category == "audience_object":
        subject = label
        templates = _AUDIENCE_OBJECT_TEMPLATES
    else:
        subject = label or "este node"
        templates = _DISCOVERY_TEMPLATES

    params = {
        "subject": subject,
        "product": subject,
        "brand": brand,
        "audience": label,
        "descriptor": _audience_descriptor(label, nearest_md),
        "object": (sellable or {}).get("name") or "esses produtos",
        "context": context_snippet,
    }

    suggestions: list[dict[str, str]] = []
    for i in range(n):
        template = templates[i % len(templates)]
        question = template["question"].format(**params)
        answer = template["answer"].format(**params)
        if i == 0 and context_snippet and "{context}" not in template["answer"]:
            answer = f"{answer} {context_snippet}"
        answer = answer.replace("  ", " ").strip()
        suggestions.append({"question": question, "answer": answer})

    # Guardrail: a non-sellable node can never carry purchase/shipping wording.
    if category in {"audience", "brand", "briefing", "discovery"}:
        suggestions = [s for s in suggestions if not _violates_commercial_guardrail(s["question"], label)]

    return {
        "parent_node_id": parent.get("id"),
        "parent_node_type": _node_type(parent),
        "parent_node_label": label,
        "count": len(suggestions),
        "category": category,
        "commercial_object_type": (sellable or {}).get("type"),
        "commercial_object_name": (sellable or {}).get("name"),
        "source_tool": SOURCE_TOOL,
        "source_context": {
            "brand": context.get("brand"),
            "product": context.get("product"),
            "product_group": context.get("product_group"),
            "category": category,
            "commercial_object_type": (sellable or {}).get("type"),
            "commercial_object_name": (sellable or {}).get("name"),
            "path": context.get("path"),
            "branch_markdown": context.get("branch_markdown"),
            "persona_slug": persona_slug,
            "generated_from_node_id": target_node.get("id"),
            "generated_from_node_slug": target_node.get("slug"),
        },
        "suggestions": suggestions,
    }


def _copy_templates_for(markdown: str) -> list[dict[str, str]]:
    """Copy FAQs include spec-driven questions extracted from the copy body
    (e.g. 'i5', '2TB') so the questions are about what the copy actually says."""
    specs = _extract_spec_tokens(markdown)
    templates: list[dict[str, str]] = []
    for spec in specs[:3]:
        templates.append({
            "question": f"O que significa {spec} citado nesta copy?",
            "answer": f"A copy menciona {spec}. Vale explicar de forma simples o que {spec} representa e qual o benefÃ­cio prÃ¡tico.",
        })
    templates.extend(_COPY_TEMPLATES)
    return templates


def _extract_spec_tokens(markdown: str) -> list[str]:
    text = (markdown or "")
    tokens: list[str] = []
    # e.g. 2TB, 512GB, 16GB, 1080p
    for match in re.findall(r"\b\d+\s?(?:TB|GB|MB|GHz|MHz|MP|W|p|fps)\b", text, flags=re.IGNORECASE):
        tokens.append(match.strip())
    # e.g. i5, i7, Ryzen 7, RTX 4060
    for match in re.findall(r"\b(?:i[3579]|ryzen\s?\d|rtx\s?\d{3,4}|gtx\s?\d{3,4})\b", text, flags=re.IGNORECASE):
        tokens.append(match.strip())
    seen: set[str] = set()
    unique: list[str] = []
    for tok in tokens:
        key = tok.lower()
        if key not in seen:
            seen.add(key)
            unique.append(tok)
    return unique

