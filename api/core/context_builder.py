from schemas.context import Context, Lead
from schemas.events import LeadEvent
from services import supabase_client, knowledge_graph
from services.knowledge_service import search_kb_text


def build(event: LeadEvent) -> Context:
    persona_id: str | None = None
    if event.persona_slug:
        persona_data = supabase_client.get_persona(event.persona_slug)
        if persona_data:
            persona_id = persona_data.get("id")

    lead_data = supabase_client.ensure_lead_for_persona(
        lead_id=event.lead_id,
        lead_ref=event.lead_ref,
        persona_slug_or_id=persona_id or event.persona_slug,
        nome=event.nome,
        stage=event.stage,
        canal=event.canal,
        mensagem=event.mensagem,
        interesse_produto=event.interesse_produto,
        cidade=event.cidade,
        cep=event.cep,
        whatsapp_phone_number_id=event.whatsapp_phone_number_id,
    ) or supabase_client.get_lead(event.lead_id) or {}

    lead = Lead(
        id=event.lead_id,
        ref=event.lead_ref or lead_data.get("id"),
        nome=event.nome or lead_data.get("nome"),
        stage=event.stage or lead_data.get("stage", "novo"),
        canal=event.canal,
        interesse_produto=event.interesse_produto or lead_data.get("interesse_produto"),
        cidade=event.cidade or lead_data.get("cidade"),
        cep=event.cep or lead_data.get("cep"),
        ai_enabled=lead_data.get("ai_enabled", True),
        metadata=lead_data.get("metadata") or {},
    )

    historico = supabase_client.get_messages(str(lead.ref or event.lead_id), limit=20)

    query = " ".join(filter(None, [event.mensagem, lead.interesse_produto, lead.stage]))
    kb_chunks = search_kb_text(query, persona_id=persona_id, top_k=5)

    # Enrich with semantic graph context (products/campaigns/assets surfaced
    # via knowledge_nodes+edges). Output is appended to kb_chunks as plain
    # strings so existing agent prompts keep working unchanged.
    try:
        graph_ctx = knowledge_graph.get_chat_context(
            lead_ref=lead.ref,
            persona_id=persona_id,
            user_text=event.mensagem,
            limit=8,
        )
        graph_lines: list[str] = []
        for n in graph_ctx.get("nodes", []):
            ntype = n.get("node_type")
            if ntype in ("product", "campaign", "rule", "tone"):
                line = f"[{ntype}] {n.get('title')}"
                if n.get("summary"):
                    line += f" â€” {n['summary'][:240]}"
                graph_lines.append(line)
        for a in graph_ctx.get("assets", []):
            graph_lines.append(
                f"[asset] {a.get('title')} ({a.get('asset_function') or a.get('asset_type') or 'media'})"
            )
        if graph_lines:
            kb_chunks = list(kb_chunks) + graph_lines
    except Exception:
        # Graph is optional context; never block the agent on graph errors.
        pass

    return Context(
        lead=lead,
        mensagem=event.mensagem,
        historico=historico,
        kb_chunks=kb_chunks,
        persona_slug=event.persona_slug,
        metadata=lead_data.get("metadata") or {},
    )

