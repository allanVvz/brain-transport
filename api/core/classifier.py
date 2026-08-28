import json
import re
from schemas.context import Context
from pathlib import Path
from services.model_router import get_router

_PROMPT = (Path(__file__).parent.parent / "prompts" / "classifier.md").read_text(encoding="utf-8")

_DEFAULTS = {
    "intent": "duvida_geral",
    "interest_level": "medio",
    "urgency": "baixa",
    "fit": "neutro",
    "objections": [],
    "summary": "Classificação falhou — usando defaults",
    "route_hint": "SDR",
}


def classify(ctx: Context) -> dict:
    history_text = "\n".join(
        f"[{m.get('sender_type','?')}] {m.get('texto','')}"
        for m in ctx.historico[-10:]
    )

    classifier_input = (
        f"INFORMAÇÕES DO LEAD:\n"
        f"Nome: {ctx.lead.nome or 'não informado'}\n"
        f"Produto de interesse: {ctx.lead.interesse_produto or 'não informado'}\n"
        f"Stage atual: {ctx.lead.stage}\n"
        f"Canal: {ctx.lead.canal}\n"
        f"Cidade: {ctx.lead.cidade or 'não informada'}\n"
        f"CEP: {ctx.lead.cep or 'não informado'}\n\n"
        f"MENSAGEM DO LEAD:\n{ctx.mensagem}\n\n"
        f"HISTÓRICO:\n{history_text or 'sem histórico'}"
    )

    prompt = _PROMPT.replace("{{classifier_input}}", classifier_input)

    raw = get_router().chat("claude-haiku-4-5-20251001", prompt, max_tokens=512)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return dict(_DEFAULTS)
