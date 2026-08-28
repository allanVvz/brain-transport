from schemas.context import Context

_INTENT_SCORES = {
    "quer_comprar": 40,
    "quer_condicao_pagamento": 30,
    "quer_preco_frete": 30,
    "quer_preco_produto": 25,
    "interesse_produto": 15,
    "comparando_opcoes": 10,
    "especificacao_produto": 10,
    "duvida_geral": 5,
    "follow_up": 5,
    "suporte": 0,
    "sem_intencao_clara": 0,
}

_OBJECTION_PENALTIES = {
    "objecao_preco": -10,
    "objecao_tempo": -5,
    "objecao_confianca": -15,
    "comparando_concorrente": -5,
    "sem_resposta": -10,
}


def compute_score(ctx: Context) -> tuple[int, list[str], str]:
    c = ctx.classification or {}
    score = 0
    tags: list[str] = []

    def tag(t: str):
        if t not in tags:
            tags.append(t)

    intent = c.get("intent", "")
    score += _INTENT_SCORES.get(intent, 0)
    if intent:
        tag(intent)

    interest = c.get("interest_level", "")
    if interest == "alto":
        score += 20
        tag("lead_quente")
    elif interest == "medio":
        score += 10
        tag("lead_morno")
    else:
        tag("lead_frio")

    urgency = c.get("urgency", "")
    if urgency == "alta":
        score += 20
        tag("urgencia_alta")
    elif urgency == "media":
        score += 10
        tag("urgencia_media")
    else:
        tag("urgencia_baixa")

    fit = c.get("fit", "")
    if fit == "bom":
        score += 15
        tag("fit_bom")
    elif fit == "neutro":
        score += 5
    else:
        tag("fit_ruim")

    if ctx.lead.interesse_produto:
        tag("produto_identificado")
    if ctx.lead.cep:
        score += 10
        tag("cep_detectado")
    if ctx.lead.cidade:
        tag("cidade_detectada")

    for obj in c.get("objections", []):
        score += _OBJECTION_PENALTIES.get(obj, 0)
        tag(obj)

    score = max(0, min(score, 100))

    funnel = "novo"
    if score > 20:
        funnel = "contatado"
    if score > 40:
        funnel = "engajado"
    if score > 60:
        funnel = "qualificado"
    if score > 80:
        funnel = "oportunidade"

    return score, tags, funnel


def decide(ctx: Context) -> str:
    c = ctx.classification or {}
    route = c.get("route_hint", "SDR")
    score = ctx.score
    intent = c.get("intent", "")

    closing_intents = {"quer_comprar", "quer_condicao_pagamento"}

    if (
        route == "SDR"
        and score > 80
        and ctx.lead.interesse_produto
        and intent in closing_intents
    ):
        return "CLOSER"

    return route
