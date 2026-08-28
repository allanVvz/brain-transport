"""
Generation route â€” gera campaign.json a partir da base de conhecimento.
Consumido pelo Figma plugin campaign_builder.
"""
import json
import os
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import anthropic

from services import supabase_client

router = APIRouter(prefix="/generate", tags=["generation"])

_FORMATS = {
    "post_feed":      {"width": 1080, "height": 1080,  "label": "Post Feed (quadrado)"},
    "post_retrato":   {"width": 1080, "height": 1350,  "label": "Post Feed (retrato 4:5)"},
    "story":          {"width": 1080, "height": 1920,  "label": "Story / Reels"},
    "carrossel":      {"width": 1080, "height": 1080,  "label": "Carrossel Feed"},
    "banner_email":   {"width": 600,  "height": 300,   "label": "Banner E-mail"},
    "banner_wide":    {"width": 1920, "height": 600,   "label": "Banner Wide"},
}

_INTENTIONS = [
    "compra_impulso", "engajamento", "educacional",
    "prova_social", "lancamento", "oferta_relampago",
    "branding", "cta_direto",
]

_CAMPAIGN_SCHEMA_EXAMPLE = {
    "campaign": {
        "name": "Nome da campanha",
        "client": "slug-do-cliente",
        "brief": "descriÃ§Ã£o do brief",
        "generated_at": "ISO timestamp",
    },
    "pages": [
        {
            "name": "Nome da pÃ¡gina (ex: Feed Posts)",
            "cards": [
                {
                    "id": "card_01",
                    "name": "Card 01 â€” Headline Principal",
                    "format": "post_feed",
                    "width": 1080,
                    "height": 1080,
                    "intention": "compra_impulso",
                    "color_scheme": "dark",
                    "background": {"type": "solid", "color": "#111111"},
                    "bg_options": [
                        {"id": "bg1", "label": "Fundo escuro", "type": "solid", "color": "#111111", "selected": True},
                        {"id": "bg2", "label": "Imagem principal", "type": "image", "file": "bg_principal.jpg"},
                    ],
                    "foreground": {"overlay": "gradient_bottom", "opacity": 0.65},
                    "copy": {
                        "headline": "TEXTO DO HEADLINE EM MAIÃšSCULAS",
                        "body": "Texto complementar direto ao ponto.",
                        "cta": "Chamada para aÃ§Ã£o!",
                    },
                    "assets_layer": [
                        {"id": "logo", "label": "Logo do cliente", "position": "top_left", "nota": "versÃ£o branca"},
                    ],
                }
            ],
        }
    ],
}


def _build_kb_context(persona_id: str) -> str:
    """Build a KB context string for the persona."""
    entries = supabase_client.get_kb_entries(persona_id=persona_id, status="ATIVO")
    if not entries:
        return "(base de conhecimento vazia para este cliente)"

    sections: dict[str, list[str]] = {}
    for e in entries:
        tipo = e.get("tipo", "geral")
        text = f"[{e.get('titulo', '')}] {e.get('conteudo', '')}"
        sections.setdefault(tipo, []).append(text)

    lines = []
    priority = ["brand", "tom", "produto", "regra", "briefing", "faq"]
    order = priority + [k for k in sections if k not in priority]
    for tipo in order:
        if tipo not in sections:
            continue
        lines.append(f"\n### {tipo.upper()}")
        for item in sections[tipo][:5]:
            lines.append(f"- {item[:300]}")

    return "\n".join(lines)


def _build_prompt(persona_name: str, kb_ctx: str, brief: str, formats: list[str], n_cards: int) -> str:
    schema_str = json.dumps(_CAMPAIGN_SCHEMA_EXAMPLE, ensure_ascii=False, indent=2)
    formats_desc = "\n".join(f"- {k}: {v['label']} ({v['width']}x{v['height']}px)" for k, v in _FORMATS.items() if k in formats)
    intentions_str = ", ".join(_INTENTIONS)

    return f"""VocÃª Ã© um especialista em design de campanhas digitais para redes sociais.
Seu trabalho Ã© gerar um arquivo campaign.json vÃ¡lido para o plugin Figma Campaign Builder.

=== CLIENTE ===
{persona_name}

=== BASE DE CONHECIMENTO DO CLIENTE ===
{kb_ctx}

=== BRIEF DA CAMPANHA ===
{brief}

=== FORMATOS SOLICITADOS ===
{formats_desc}

=== INSTRUÃ‡Ã•ES ===
Gere exatamente {n_cards} card(s) distribuÃ­dos nos formatos solicitados.
Cada card deve ter copy 100% alinhado com o tom de voz e identidade do cliente.
Use as informaÃ§Ãµes da base de conhecimento para personalizar headline, body e cta.
Seja criativo mas fiel ao brand â€” copy em portuguÃªs brasileiro, tom correto para o cliente.

IntenÃ§Ãµes disponÃ­veis: {intentions_str}

Para cada card:
- headline: mÃ¡ximo 5 palavras, impacto mÃ¡ximo, MAIÃšSCULAS se o brand usar
- body: 1-2 frases, benefÃ­cio claro
- cta: 2-4 palavras, aÃ§Ã£o direta
- bg_options: sempre inclua ao menos 1 opÃ§Ã£o solid e 1 opÃ§Ã£o image
- foreground.overlay: "gradient_bottom", "dark_top" ou "none"
- color_scheme: "dark" ou "light" de acordo com o fundo
- assets_layer: inclua logo na position "top_left" sempre

=== SCHEMA ESPERADO (SIGA EXATAMENTE) ===
{schema_str}

RESPONDA APENAS com o JSON vÃ¡lido, sem markdown, sem explicaÃ§Ãµes, sem cÃ³digo fences.
O JSON deve comeÃ§ar com {{ e terminar com }}.
"""


class GenerateRequest(BaseModel):
    persona_slug: str
    formats: list[str] = ["post_feed", "story"]
    brief: str
    n_cards: int = 3
    model: str = "claude-sonnet-4-6"
    campaign_name: Optional[str] = None


@router.get("/formats")
def list_formats():
    return [{"id": k, **v} for k, v in _FORMATS.items()]


@router.post("/publication")
def generate_publication(body: GenerateRequest):
    # Validate formats
    invalid = [f for f in body.formats if f not in _FORMATS]
    if invalid:
        raise HTTPException(400, f"Formatos invÃ¡lidos: {invalid}. DisponÃ­veis: {list(_FORMATS.keys())}")

    if body.n_cards < 1 or body.n_cards > 12:
        raise HTTPException(400, "n_cards deve ser entre 1 e 12")

    # Load persona
    persona = supabase_client.get_persona(body.persona_slug)
    if not persona:
        raise HTTPException(404, f"Persona nÃ£o encontrada: {body.persona_slug}")

    persona_id = persona["id"]
    persona_name = persona.get("name", body.persona_slug)

    # Build KB context
    kb_ctx = _build_kb_context(persona_id)

    # Build prompt and call Claude
    prompt = _build_prompt(
        persona_name=persona_name,
        kb_ctx=kb_ctx,
        brief=body.brief,
        formats=body.formats,
        n_cards=body.n_cards,
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    client_ai = anthropic.Anthropic(api_key=api_key)

    try:
        response = client_ai.messages.create(
            model=body.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
    except Exception as e:
        raise HTTPException(500, f"Claude API error: {e}")

    # Parse the JSON response
    try:
        # Strip possible markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        campaign_json = json.loads(raw)
    except Exception as e:
        raise HTTPException(500, f"JSON parse error: {e}. Raw response: {raw[:500]}")

    # Inject metadata
    campaign_json.setdefault("campaign", {})
    campaign_json["campaign"]["client"] = body.persona_slug
    campaign_json["campaign"]["brief"] = body.brief
    campaign_json["campaign"]["generated_at"] = datetime.now(timezone.utc).isoformat()
    campaign_json["campaign"]["model"] = body.model
    if body.campaign_name:
        campaign_json["campaign"]["name"] = body.campaign_name

    # Log the generation
    supabase_client.insert_event({
        "event_type": "publication_generated",
        "payload": {
            "persona_slug": body.persona_slug,
            "formats": body.formats,
            "n_cards": body.n_cards,
            "model": body.model,
            "brief": body.brief[:200],
        },
    })

    return campaign_json

