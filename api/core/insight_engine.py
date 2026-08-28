from schemas.context import Context
from services import supabase_client
from datetime import datetime


def record(ctx: Context, agent_result: dict, latency_ms: int) -> None:
    supabase_client.insert_agent_log({
        "lead_id": ctx.lead.id,
        "agent_name": agent_result.get("agent", "unknown"),
        "input": {
            "mensagem": ctx.mensagem,
            "stage": ctx.lead.stage,
            "route_hint": ctx.route_hint,
        },
        "output": agent_result,
        "latency_ms": latency_ms,
        "model_used": agent_result.get("model"),
        "status": "success" if agent_result.get("reply") else "error",
        "created_at": datetime.utcnow().isoformat(),
    })
