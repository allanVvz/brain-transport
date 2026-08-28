from services import supabase_client
from datetime import datetime, timedelta, timezone
from typing import Optional


def analyze(persona_id: Optional[str] = None) -> list[dict]:
    insights = []
    leads = supabase_client.get_leads(limit=500)

    if not leads:
        return insights

    # leads estagnados por stage
    stale_threshold = datetime.now(timezone.utc) - timedelta(hours=48)
    stale_by_stage: dict[str, int] = {}

    for lead in leads:
        updated_raw = lead.get("updated_at") or lead.get("last_update") or lead.get("created_at")
        if not updated_raw:
            continue
        try:
            updated = datetime.fromisoformat(updated_raw.replace("Z", "+00:00"))
        except Exception:
            continue
        if updated < stale_threshold:
            stage = lead.get("stage", "desconhecido")
            stale_by_stage[stage] = stale_by_stage.get(stage, 0) + 1

    total_stale = sum(stale_by_stage.values())
    if total_stale > 0:
        stage_summary = ", ".join(f"{s}: {n}" for s, n in stale_by_stage.items())
        severity = "critical" if total_stale > 20 else "warning"
        insights.append({
            "persona_id": persona_id,
            "severity": severity,
            "category": "business",
            "title": f"{total_stale} leads sem movimentaÃ§Ã£o hÃ¡ mais de 48h",
            "description": f"DistribuiÃ§Ã£o por stage: {stage_summary}",
            "recommendation": "Criar automaÃ§Ã£o de follow-up. Leads quentes esfriam apÃ³s 24h sem contato.",
            "affected_component": "lead funnel / follow-up",
            "score_impact": -8 if severity == "warning" else -15,
        })

    # ai_enabled=false bloqueando leads
    ai_disabled = sum(1 for l in leads if not l.get("ai_enabled", True))
    if ai_disabled > 0:
        insights.append({
            "persona_id": persona_id,
            "severity": "info",
            "category": "business",
            "title": f"{ai_disabled} lead(s) com ai_enabled=false",
            "description": "Esses leads nÃ£o recebem respostas da IA.",
            "recommendation": "Verificar se foram desativados intencionalmente ou Ã© bug do fluxo de criaÃ§Ã£o.",
            "affected_component": "lead.ai_enabled flag",
            "score_impact": -3,
        })

    # conversÃ£o de funil (ratio novo â†’ contatado)
    by_stage: dict[str, int] = {}
    for lead in leads:
        s = lead.get("stage", "novo")
        by_stage[s] = by_stage.get(s, 0) + 1

    total = len(leads)
    novos = by_stage.get("novo", 0) + by_stage.get("nao_qualificado", 0)
    if total > 0 and novos / total > 0.70:
        insights.append({
            "persona_id": persona_id,
            "severity": "warning",
            "category": "business",
            "title": f"{novos/total:.0%} dos leads ainda em stage 'novo' â€” funil bloqueado no topo",
            "description": f"{novos} de {total} leads nunca passaram da etapa inicial.",
            "recommendation": "Revisar prompt do SDR e critÃ©rios de avanÃ§o de stage.",
            "affected_component": "SDR Agent / stage transition",
            "score_impact": -10,
        })

    return insights

