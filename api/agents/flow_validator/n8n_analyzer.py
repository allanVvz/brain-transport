from services import n8n_client, supabase_client
from datetime import datetime, timedelta
from typing import Optional


def analyze(persona_id: Optional[str] = None) -> list[dict]:
    insights = []
    error_rate = supabase_client.get_n8n_error_rate(hours=24)

    if error_rate > 0.10:
        insights.append({
            "persona_id": persona_id,
            "severity": "critical",
            "category": "reliability",
            "title": f"Taxa de erro n8n alta: {error_rate:.1%} nas últimas 24h",
            "description": f"{error_rate:.1%} das execuções falharam. Threshold aceitável: 10%.",
            "recommendation": "Verificar logs do n8n. Possível problema em node de Airtable ou Supabase.",
            "affected_component": "n8n executions",
            "score_impact": -20,
        })
    elif error_rate > 0.05:
        insights.append({
            "persona_id": persona_id,
            "severity": "warning",
            "category": "reliability",
            "title": f"Taxa de erro n8n elevada: {error_rate:.1%}",
            "description": "Acima do threshold de 5%.",
            "recommendation": "Monitorar. Verificar se é intermitente ou sistemático.",
            "affected_component": "n8n executions",
            "score_impact": -8,
        })

    executions = supabase_client.get_n8n_executions(limit=200)
    if executions:
        durations = [e["duration_ms"] for e in executions if e.get("duration_ms")]
        if durations:
            avg_ms = sum(durations) / len(durations)
            if avg_ms > 10000:
                insights.append({
                    "persona_id": persona_id,
                    "severity": "warning",
                    "category": "performance",
                    "title": f"Latência média alta: {avg_ms/1000:.1f}s por execução",
                    "description": "Execuções lentas impactam tempo de resposta ao lead.",
                    "recommendation": "Identificar nodes lentos. Classifier Agent e SDR/Closer são os candidatos principais.",
                    "affected_component": "n8n workflow latency",
                    "score_impact": -10,
                })

        latest = max((e.get("finished_at") or e.get("started_at") for e in executions if e.get("started_at")), default=None)
        if latest:
            try:
                last_dt = datetime.fromisoformat(latest.replace("Z", "+00:00"))
                gap = datetime.now(last_dt.tzinfo) - last_dt
                if gap > timedelta(hours=2):
                    insights.append({
                        "persona_id": persona_id,
                        "severity": "critical",
                        "category": "reliability",
                        "title": f"Nenhuma execução n8n nas últimas {gap.seconds//3600}h",
                        "description": "O fluxo principal pode estar parado.",
                        "recommendation": "Verificar se o trigger WhatsApp está ativo e se o n8n está online.",
                        "affected_component": "WhatsApp Trigger",
                        "score_impact": -25,
                    })
            except Exception:
                pass

    return insights
