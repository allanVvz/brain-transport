# -*- coding: utf-8 -*-
"""
Marketing â€” text generation for marketing copy, ads, emails, social, etc.

Distinct from /generate (which is for Figma campaign cards):
- This route is text-only and persona-aware.
- Backed by ModelRouter (OpenAI cascade + Anthropic fallback).
- System prompts are distilled from the curated marketing skills:
  copywriting, marketing-psychology, customer-research, content-strategy,
  cold-email, email-sequence, ad-creative, lead-magnets, social-content.

Output is plain text/markdown so the dashboard can render and (optionally)
persist as a knowledge_items row of content_type=copy.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from services import auth_service, integration_service, supabase_client
from services.model_router import ModelRouter, ModelRouterError, AVAILABLE_MODELS

logger = logging.getLogger("marketing")

router = APIRouter(prefix="/marketing", tags=["marketing"])


# â”€â”€ Mode catalog â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Each mode maps to a system-prompt template. The template is distilled from
# the corresponding marketing skill (see ~/.claude/skills-staging) and adapted
# for the Brain AI context (persona-aware, structured output).
#
# Adding a new mode is a single dict entry â€” no other code changes required.

class ModeSpec(BaseModel):
    key: str
    label: str
    description: str
    inputs: list[dict]  # [{name, label, placeholder, type:"text"|"textarea"|"select", required?}]
    system_prompt: str
    user_prompt_template: str


def _persona_block(persona_id: Optional[str]) -> str:
    """Build a persona context block to prepend to the system prompt.

    Pulls brand/tone/product/briefing nodes from the knowledge graph so the
    output matches the persona's voice and catalog. Best-effort â€” empty
    block when graph is unavailable or persona is None.
    """
    if not persona_id:
        return ""
    try:
        # Brand + tone + briefing summaries set voice/positioning context.
        canonical = supabase_client.list_knowledge_nodes_by_type(
            ["brand", "tone", "briefing", "rule"], persona_id=persona_id, limit=20,
        ) or []
        # Top products give the catalog that copy can reference.
        products = supabase_client.list_knowledge_nodes_by_type(
            ["product"], persona_id=persona_id, limit=12,
        ) or []
    except Exception as exc:
        logger.warning("persona_block fetch failed: %s", exc)
        return ""

    if not canonical and not products:
        return ""

    parts: list[str] = []
    if canonical:
        parts.append("## Contexto da marca / tom / regras (nÃ£o inventar â€” sÃ³ usar):")
        for n in canonical:
            title = (n.get("title") or n.get("slug") or "").strip()
            summary = (n.get("summary") or "").strip()[:300]
            ntype = n.get("node_type")
            if title:
                parts.append(f"- **[{ntype}] {title}**" + (f": {summary}" if summary else ""))
    if products:
        parts.append("\n## Produtos/ofertas catalogadas:")
        for n in products:
            title = (n.get("title") or n.get("slug") or "").strip()
            meta = n.get("metadata") or {}
            facts: list[str] = []
            price = meta.get("price") or {}
            if price.get("display"):
                facts.append(price["display"])
            if meta.get("colors_count") is not None:
                facts.append(f"{meta['colors_count']} cores")
            url = meta.get("catalog_url") or meta.get("url")
            extra = f" â€” {', '.join(facts)}" if facts else ""
            extra += f" â€” {url}" if url else ""
            parts.append(f"- {title}{extra}")
    return "\n".join(parts) + "\n"


# Skill-distilled system prompts. Each is concise (~150 words) and assumes
# the persona block is appended above it at call-time.
_BASE_VOICE = (
    "VocÃª Ã© um copywriter senior que escreve em portuguÃªs brasileiro coloquial "
    "e direto. Nunca invente fatos sobre produto, preÃ§o ou polÃ­tica â€” use sÃ³ o "
    "que estiver no contexto da persona. Quando faltar dado, pergunte ao "
    "operador em vez de chutar."
)

_MODES: dict[str, ModeSpec] = {
    "copywriting": ModeSpec(
        key="copywriting",
        label="Copy de Produto",
        description="Texto de venda/oferta para um produto especÃ­fico, baseado em Ã¢ngulos psicolÃ³gicos e preÃ§o estruturado.",
        inputs=[
            {"name": "product",   "label": "Produto",   "type": "text",     "placeholder": "Ex.: HigienizaÃ§Ã£o de Cadeiras Prime",  "required": True},
            {"name": "audience",  "label": "PÃºblico",   "type": "text",     "placeholder": "Ex.: donas de casa em Novo Hamburgo"},
            {"name": "angle",     "label": "Ã‚ngulo",    "type": "select",   "options": ["benefÃ­cio", "dor", "prova social", "urgÃªncia", "preÃ§o/valor"]},
            {"name": "format",    "label": "Formato",   "type": "select",   "options": ["headline + parÃ¡grafo", "post Instagram", "anÃºncio Meta Ads", "WhatsApp"]},
            {"name": "extra",     "label": "Notas",     "type": "textarea", "placeholder": "Ex.: enfatizar regional, tom seguro"},
        ],
        system_prompt=(
            f"{_BASE_VOICE}\n\n"
            "Sua tarefa Ã© escrever copy de venda baseado em Ã¢ngulo psicolÃ³gico claro. "
            "Estruture: (1) headline curta com tensÃ£o; (2) corpo com 1 benefÃ­cio + 1 prova "
            "concreta; (3) CTA especÃ­fico (nÃ£o 'saiba mais'). Use o preÃ§o estruturado quando "
            "houver. SaÃ­da em markdown."
        ),
        user_prompt_template=(
            "Produto: {product}\n"
            "PÃºblico: {audience}\n"
            "Ã‚ngulo: {angle}\n"
            "Formato: {format}\n"
            "Notas adicionais: {extra}\n\n"
            "Gere a copy."
        ),
    ),

    "cold_email": ModeSpec(
        key="cold_email",
        label="E-mail Frio (cold email)",
        description="SequÃªncia inicial de outreach personalizada. Curto, com gancho relevante e CTA Ãºnico.",
        inputs=[
            {"name": "target",     "label": "Lead/empresa", "type": "text",     "placeholder": "Cargo ou nome do contato + empresa", "required": True},
            {"name": "hook",       "label": "Gancho",        "type": "textarea", "placeholder": "Algo especÃ­fico observado (post, vaga, evento)"},
            {"name": "offer",      "label": "Oferta",        "type": "text",     "placeholder": "Ex.: Demo de 15min mostrando X"},
            {"name": "tone",       "label": "Tom",           "type": "select",   "options": ["formal", "informal", "consultivo"]},
        ],
        system_prompt=(
            f"{_BASE_VOICE}\n\n"
            "Escreva e-mail frio que segue: (1) linha de assunto com curiosidade ou benefÃ­cio "
            "especÃ­fico (â‰¤8 palavras); (2) primeira frase referenciando o gancho; (3) 2-3 frases "
            "conectando o gancho Ã  oferta; (4) CTA Ãºnico e fÃ¡cil (nunca mÃºltiplas perguntas); "
            "(5) PS opcional com prova social. Total â‰¤ 120 palavras. SaÃ­da como bloco "
            "markdown com Subject: e Body:."
        ),
        user_prompt_template=(
            "Lead: {target}\n"
            "Gancho observado: {hook}\n"
            "Oferta: {offer}\n"
            "Tom: {tone}\n\n"
            "Escreva o e-mail frio."
        ),
    ),

    "email_sequence": ModeSpec(
        key="email_sequence",
        label="SequÃªncia de E-mail",
        description="SÃ©rie de 3-5 e-mails para nurture/onboarding/recovery, com progressÃ£o lÃ³gica.",
        inputs=[
            {"name": "goal",     "label": "Objetivo",  "type": "select",   "options": ["nurture", "onboarding", "carrinho abandonado", "winback", "lead magnet follow-up"], "required": True},
            {"name": "audience", "label": "PÃºblico",   "type": "text",     "placeholder": "Ex.: leads que baixaram o lead magnet"},
            {"name": "count",    "label": "Quantos e-mails", "type": "select", "options": ["3", "5", "7"]},
            {"name": "extra",    "label": "Notas",     "type": "textarea"},
        ],
        system_prompt=(
            f"{_BASE_VOICE}\n\n"
            "Construa uma sequÃªncia numerada onde cada e-mail tem um Ãºnico objetivo "
            "psicolÃ³gico (educar â†’ engajar â†’ desejo â†’ urgÃªncia â†’ CTA). Para cada e-mail: "
            "subject, preview text, body curto (â‰¤80 palavras), CTA. Mostre o intervalo "
            "sugerido entre cada (ex.: D+0, D+2, D+4). Markdown."
        ),
        user_prompt_template=(
            "Objetivo: {goal}\n"
            "PÃºblico: {audience}\n"
            "Quantidade: {count} e-mails\n"
            "Notas: {extra}\n\n"
            "Construa a sequÃªncia."
        ),
    ),

    "ad_creative": ModeSpec(
        key="ad_creative",
        label="AnÃºncio (Meta/Google Ads)",
        description="Variantes de criativo para teste A/B com mÃºltiplos Ã¢ngulos e formatos.",
        inputs=[
            {"name": "product",  "label": "Produto",   "type": "text", "required": True},
            {"name": "platform", "label": "Plataforma","type": "select", "options": ["Meta Feed", "Meta Stories", "Google Search", "Google Display", "TikTok"]},
            {"name": "variants", "label": "Variantes", "type": "select", "options": ["3", "5", "8"]},
            {"name": "extra",    "label": "Notas",     "type": "textarea"},
        ],
        system_prompt=(
            f"{_BASE_VOICE}\n\n"
            "Gere variantes de criativo, cada uma com Ã¢ngulo distinto (benefÃ­cio, dor, "
            "comparaÃ§Ã£o, prova social, urgÃªncia). Para cada variante: headline, descriÃ§Ã£o "
            "(respeitando limite da plataforma), CTA. Anote qual Ã¢ngulo psicolÃ³gico cada "
            "variante explora. SaÃ­da como tabela markdown."
        ),
        user_prompt_template=(
            "Produto: {product}\n"
            "Plataforma: {platform}\n"
            "Variantes: {variants}\n"
            "Notas: {extra}\n\n"
            "Gere as variantes."
        ),
    ),

    "lead_magnet": ModeSpec(
        key="lead_magnet",
        label="Lead Magnet",
        description="IdÃ©ia + outline para um lead magnet (e-book, checklist, calculadora) que captura e qualifica.",
        inputs=[
            {"name": "audience", "label": "PÃºblico-alvo", "type": "text", "required": True},
            {"name": "pain",     "label": "Dor principal", "type": "textarea", "required": True},
            {"name": "format",   "label": "Formato",      "type": "select", "options": ["checklist", "e-book", "template", "calculadora", "mini-curso"]},
        ],
        system_prompt=(
            f"{_BASE_VOICE}\n\n"
            "Proponha um lead magnet que: (1) tenha tÃ­tulo com ganho especÃ­fico no nome; "
            "(2) outline em 5-8 seÃ§Ãµes; (3) exemplo de hook na intro; (4) call to action de "
            "upgrade no final que conecta ao produto. Markdown."
        ),
        user_prompt_template=(
            "PÃºblico: {audience}\n"
            "Dor: {pain}\n"
            "Formato: {format}\n\n"
            "Proponha o lead magnet completo."
        ),
    ),

    "social_content": ModeSpec(
        key="social_content",
        label="Posts de Social",
        description="Bateria de posts para feed/Stories alinhados ao tom da marca.",
        inputs=[
            {"name": "platform", "label": "Plataforma", "type": "select", "options": ["Instagram Feed", "Instagram Stories", "LinkedIn", "TikTok caption"]},
            {"name": "theme",    "label": "Tema",       "type": "text", "required": True},
            {"name": "count",    "label": "Quantidade", "type": "select", "options": ["3", "5", "10"]},
            {"name": "extra",    "label": "Notas",      "type": "textarea"},
        ],
        system_prompt=(
            f"{_BASE_VOICE}\n\n"
            "Gere posts numerados, cada um com objetivo claro (ensinar, vender, engajar, "
            "provar). Use frases curtas, evite jargÃ£o, respeite o tom da marca. Para cada post: "
            "tipo (educacional/promo/prova social), texto completo, hashtags se aplicÃ¡vel, CTA. "
            "Markdown."
        ),
        user_prompt_template=(
            "Plataforma: {platform}\n"
            "Tema: {theme}\n"
            "Quantidade: {count}\n"
            "Notas: {extra}\n\n"
            "Gere os posts."
        ),
    ),

    "content_strategy": ModeSpec(
        key="content_strategy",
        label="EstratÃ©gia de ConteÃºdo",
        description="Plano editorial ou pilar de conteÃºdo orientado a um objetivo de marketing.",
        inputs=[
            {"name": "goal",     "label": "Objetivo",   "type": "select", "options": ["aumentar trÃ¡fego", "gerar leads", "nutrir base", "posicionar autoridade", "lanÃ§ar produto"], "required": True},
            {"name": "audience", "label": "PÃºblico",    "type": "text"},
            {"name": "horizon",  "label": "Horizonte",  "type": "select", "options": ["1 mÃªs", "1 trimestre", "6 meses"]},
        ],
        system_prompt=(
            f"{_BASE_VOICE}\n\n"
            "Construa o plano com: (1) tese/posicionamento; (2) 3 pilares de conteÃºdo com "
            "exemplos de tÃ³picos; (3) calendÃ¡rio sugerido (ritmo + formatos); (4) mÃ©tricas "
            "de sucesso por pilar; (5) prÃ³ximos passos acionÃ¡veis. Markdown."
        ),
        user_prompt_template=(
            "Objetivo: {goal}\n"
            "PÃºblico: {audience}\n"
            "Horizonte: {horizon}\n\n"
            "Construa o plano."
        ),
    ),

    "marketing_psychology": ModeSpec(
        key="marketing_psychology",
        label="AnÃ¡lise PsicolÃ³gica",
        description="Aplica gatilhos cognitivos a uma situaÃ§Ã£o de venda, oferta ou objeÃ§Ã£o.",
        inputs=[
            {"name": "situation", "label": "SituaÃ§Ã£o", "type": "textarea", "required": True, "placeholder": "CenÃ¡rio, oferta ou objeÃ§Ã£o a tratar"},
            {"name": "goal",      "label": "Objetivo", "type": "text",     "placeholder": "O que vocÃª quer que aconteÃ§a depois"},
        ],
        system_prompt=(
            f"{_BASE_VOICE}\n\n"
            "Identifique 3-5 gatilhos psicolÃ³gicos relevantes (ex.: prova social, "
            "escassez, ancoragem, reciprocidade, autoridade) e mostre como aplicar cada um "
            "para a situaÃ§Ã£o. Para cada: explicaÃ§Ã£o curta + exemplo de frase pronta. Markdown."
        ),
        user_prompt_template=(
            "SituaÃ§Ã£o: {situation}\n"
            "Objetivo: {goal}\n\n"
            "Mostre os gatilhos aplicÃ¡veis."
        ),
    ),
}


# â”€â”€ Schemas â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class GenerateRequest(BaseModel):
    mode: str = Field(..., description="Key from /marketing/modes")
    inputs: dict = Field(default_factory=dict)
    persona_id: Optional[str] = None
    model: Optional[str] = Field(None, description="OpenAI/Anthropic model id; default gpt-4o-mini")
    max_tokens: int = Field(1500, ge=100, le=4000)


class GenerateResponse(BaseModel):
    content: str
    model_used: Optional[str] = None
    mode: str
    persona_id: Optional[str] = None


class ModeListResponse(BaseModel):
    modes: list[dict]
    available_models: dict[str, str]


# â”€â”€ Routes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@router.get("/modes", response_model=ModeListResponse)
def list_modes():
    """List all creation modes with their input schema. Used by the dashboard
    to render the form dynamically."""
    modes_payload = [
        {
            "key": m.key,
            "label": m.label,
            "description": m.description,
            "inputs": m.inputs,
        }
        for m in _MODES.values()
    ]
    return ModeListResponse(modes=modes_payload, available_models=AVAILABLE_MODELS)


@router.post("/generate", response_model=GenerateResponse)
def generate(body: GenerateRequest, request: Request):
    if body.persona_id:
        auth_service.assert_persona_access(request, persona_id=body.persona_id)
    spec = _MODES.get(body.mode)
    if not spec:
        raise HTTPException(status_code=404, detail=f"Unknown mode '{body.mode}'. See /marketing/modes")

    # Build user prompt by templating; missing keys become "(nÃ£o informado)" so
    # the model still has structure even with partial input.
    safe_inputs = {k: (v if v not in (None, "") else "(nÃ£o informado)") for k, v in body.inputs.items()}
    try:
        user_prompt = spec.user_prompt_template.format_map(_DefaultDict(safe_inputs))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid inputs: {exc}")

    # Compose system prompt: persona context (if any) + skill prompt.
    persona_block = _persona_block(body.persona_id)
    system_prompt = (persona_block + "\n" if persona_block else "") + spec.system_prompt

    user = auth_service.current_user(request)
    user_id = user.get("id") or ""
    router_ = ModelRouter(
        openai_api_key=integration_service.get_enabled_user_secret(user_id, "openai"),
        anthropic_api_key=integration_service.get_enabled_user_secret(user_id, "anthropic"),
    )
    requested_model = body.model or "gpt-4o-mini"
    try:
        content = router_.messages_create(
            model=requested_model,
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt,
            max_tokens=body.max_tokens,
        )
    except ModelRouterError as exc:
        logger.error("marketing.generate exhausted providers: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))

    return GenerateResponse(
        content=content,
        model_used=requested_model,  # exact router-selected model is logged but not exposed
        mode=body.mode,
        persona_id=body.persona_id,
    )


class _DefaultDict(dict):
    """Format-map helper: missing keys become '(nÃ£o informado)' instead of KeyError."""
    def __missing__(self, key: str) -> str:
        return "(nÃ£o informado)"

