import asyncio
import httpx
import logging
from schemas.context import Context
from services.model_router import get_router

logger = logging.getLogger("base_agent")


class BaseAgent:
    name: str = "base"
    model: str = "gpt-4o-mini"

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    async def run(self, ctx: Context) -> dict:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(self.endpoint, json={
                    "lead_id": ctx.lead.id,
                    "nome": ctx.lead.nome,
                    "stage": ctx.lead.stage,
                    "canal": ctx.lead.canal,
                    "mensagem": ctx.mensagem,
                    "interesse_produto": ctx.lead.interesse_produto,
                    "cidade": ctx.lead.cidade,
                    "cep": ctx.lead.cep,
                    "classification": ctx.classification,
                    "kb_chunks": ctx.kb_chunks,
                    "historico": ctx.historico[-5:],
                })
                response.raise_for_status()
                result = response.json()
                result["agent"] = self.name
                result["model"] = self.model
                return result
        except Exception as exc:
            logger.warning(
                "Agent %s: external service %s unavailable (%s) — using inline model router",
                self.name, self.endpoint, exc,
            )
            return await asyncio.to_thread(self._run_inline, ctx)

    def _run_inline(self, ctx: Context) -> dict:
        kb_text = (
            "\n".join(f"- {c}" for c in ctx.kb_chunks)
            if ctx.kb_chunks
            else "(base de conhecimento vazia)"
        )
        history_parts: list[str] = []
        for msg in (ctx.historico or [])[-5:]:
            role = "Sofia" if msg.get("sender_type") == "agent" else "Cliente"
            text = (msg.get("texto") or "").strip()
            if text:
                history_parts.append(f"{role}: {text}")
        history_text = "\n".join(history_parts) if history_parts else "(sem histórico)"

        prompt = self._build_inline_prompt(ctx, kb_text, history_text)
        try:
            reply = get_router().chat(self.model, prompt, max_tokens=512)
        except Exception as exc:
            logger.error("Agent %s inline fallback failed: %s", self.name, exc)
            reply = "Olá! Como posso ajudar você hoje?"
        return {
            "reply": reply,
            "agent": self.name,
            "model": f"{self.model}(inline)",
            "detected_fields": {},
        }

    def _build_inline_prompt(self, ctx: Context, kb_text: str, history_text: str) -> str:
        return f"""Você é Sofia, assistente de vendas via WhatsApp. Responda à mensagem do cliente usando a base de conhecimento abaixo.

=== BASE DE CONHECIMENTO ===
{kb_text}

=== HISTÓRICO DA CONVERSA ===
{history_text}

=== MENSAGEM DO CLIENTE ===
{ctx.mensagem}

Responda de forma direta, amigável e focada em conversão. Máximo 3 parágrafos curtos."""
