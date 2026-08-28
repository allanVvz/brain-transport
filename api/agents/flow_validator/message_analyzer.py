from services import supabase_client
from datetime import datetime, timezone
from typing import Optional


def analyze(persona_id: Optional[str] = None) -> list[dict]:
    insights = []
    messages = supabase_client.get_recent_messages(hours=24, limit=1000)

    if not messages:
        return insights

    inbound = [m for m in messages if m.get("direction") == "inbound" or m.get("sender_type") == "lead"]
    outbound = [m for m in messages if m.get("direction") == "outbound" or m.get("sender_type") in ("agent", "ai")]

    # response rate
    inbound_lead_ids = {m.get("lead_id") or m.get("lead_ref") for m in inbound}
    outbound_lead_ids = {m.get("lead_id") or m.get("lead_ref") for m in outbound}
    unanswered_leads = inbound_lead_ids - outbound_lead_ids

    if len(unanswered_leads) > 0:
        rate = len(unanswered_leads) / max(len(inbound_lead_ids), 1)
        if rate > 0.05:
            insights.append({
                "persona_id": persona_id,
                "severity": "critical" if rate > 0.15 else "warning",
                "category": "business",
                "title": f"{len(unanswered_leads)} conversa(s) sem resposta da IA nas últimas 24h",
                "description": f"{rate:.1%} das conversas ativas não receberam resposta.",
                "recommendation": "Verificar se o fluxo n8n está processando todas as mensagens. Verificar logs de erro.",
                "affected_component": "SDR/Closer Agents",
                "score_impact": -12,
            })

    # avg response latency via agent_logs
    logs = supabase_client.get_agent_logs(limit=200)
    if logs:
        latencies = [l["latency_ms"] for l in logs if l.get("latency_ms") and l["latency_ms"] > 0]
        if latencies:
            avg_ms = sum(latencies) / len(latencies)
            if avg_ms > 10000:
                insights.append({
                    "persona_id": persona_id,
                    "severity": "warning",
                    "category": "performance",
                    "title": f"Latência média dos agentes: {avg_ms/1000:.1f}s",
                    "description": "Lead espera mais de 10s por resposta. Impacta experiência.",
                    "recommendation": "Usar modelos mais rápidos (Haiku) para SDR. Reservar Sonnet para Closer.",
                    "affected_component": "SDR/Closer Agent latency",
                    "score_impact": -8,
                })

    # agent distribution
    sdr_count = sum(1 for l in logs if l.get("agent_name") == "SDR")
    closer_count = sum(1 for l in logs if l.get("agent_name") == "Closer")
    total_agent = sdr_count + closer_count

    if total_agent > 10 and closer_count / total_agent < 0.05:
        insights.append({
            "persona_id": persona_id,
            "severity": "info",
            "category": "business",
            "title": "Menos de 5% das conversas chegam ao Closer",
            "description": f"SDR: {sdr_count}, Closer: {closer_count}. Baixa taxa de avanço para fechamento.",
            "recommendation": "Revisar critérios de roteamento SDR → Closer no decision_engine.",
            "affected_component": "decision_engine / routing",
            "score_impact": -5,
        })

    return insights
