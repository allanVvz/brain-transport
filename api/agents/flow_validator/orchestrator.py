import os
import json
import anthropic
from datetime import datetime, timezone
from agents.flow_validator import n8n_analyzer, funnel_analyzer, message_analyzer, architecture_analyzer
from services import supabase_client

_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


def _compute_health_score(insights: list[dict]) -> dict:
    penalties = sum(i.get("score_impact", 0) for i in insights)

    base = {
        "performance": 25,
        "reliability": 25,
        "architecture": 25,
        "business": 25,
    }

    for insight in insights:
        cat = insight.get("category", "")
        impact = abs(insight.get("score_impact", 0))
        if cat in base:
            base[cat] = max(0, base[cat] - impact)

    total = sum(base.values())
    return {
        "score_total": total,
        "score_performance": base["performance"],
        "score_reliability": base["reliability"],
        "score_architecture": base["architecture"],
        "score_business": base["business"],
        "open_critical": sum(1 for i in insights if i.get("severity") == "critical"),
        "open_warnings": sum(1 for i in insights if i.get("severity") == "warning"),
    }


def _deduplicate(new_insights: list[dict], existing_titles: list[str]) -> list[dict]:
    return [i for i in new_insights if i["title"] not in existing_titles]


def run(persona_id: str | None = None) -> dict:
    print(f"[flow-validator] starting analysis cycle — {datetime.now(timezone.utc).isoformat()}")

    all_insights: list[dict] = []

    for name, analyzer in [
        ("n8n", n8n_analyzer),
        ("funnel", funnel_analyzer),
        ("message", message_analyzer),
        ("architecture", architecture_analyzer),
    ]:
        try:
            found = analyzer.analyze(persona_id=persona_id)
            all_insights.extend(found)
            print(f"[flow-validator] {name}: {len(found)} insights")
        except Exception as e:
            print(f"[flow-validator] {name} error: {e}")

    # deduplicate vs open insights already in DB
    existing = supabase_client.get_open_insights_titles()
    new_insights = _deduplicate(all_insights, existing)

    # persist new insights
    for insight in new_insights:
        try:
            supabase_client.insert_insight({
                **insight,
                "status": "open",
            })
        except Exception as e:
            print(f"[flow-validator] insert_insight error: {e}")

    # compute and persist health score
    score_data = _compute_health_score(all_insights)
    score_data["persona_id"] = persona_id
    score_data["snapshot_at"] = datetime.now(timezone.utc).isoformat()

    try:
        supabase_client.insert_health_snapshot(score_data)
    except Exception as e:
        print(f"[flow-validator] insert_health error: {e}")

    print(f"[flow-validator] done. score={score_data['score_total']} new_insights={len(new_insights)}")
    return score_data
