import json
from pathlib import Path
from typing import Optional

_WORKFLOWS_DIR = Path(__file__).parent.parent.parent / "n8n-workflows"

_ANTI_PATTERNS = [
    {
        "key": "vectorStoreInMemory",
        "severity": "critical",
        "category": "reliability",
        "title": "Vector store in-memory detectado — KB perdida em restarts",
        "description": (
            "O node vectorStoreInMemory do n8n não persiste dados. "
            "Em qualquer restart do n8n, toda a KB é apagada. "
            "Durante o rebuild (30-60s/hora), queries retornam vazio."
        ),
        "recommendation": "Migrar KB para Supabase pgvector (já suportado no Brain AI).",
        "affected_component": "legacy knowledge workflow",
        "score_impact": -15,
    },
    {
        "key": "memoryPostgresChat",
        "severity": "warning",
        "category": "architecture",
        "title": "Memória conversacional em PostgreSQL separado do Supabase",
        "description": (
            "Os agentes SDR e Closer usam tabelas chat_history em conexões Postgres distintas "
            "com janelas diferentes (10 msgs vs 30 msgs), criando inconsistência de contexto."
        ),
        "recommendation": "Centralizar histórico na tabela messages do Supabase. Brain AI lê diretamente.",
        "affected_component": "Memory Conversacional / Classifier Agent",
        "score_impact": -8,
    },
    {
        "key": "lmChatOpenAi",
        "severity": "info",
        "category": "architecture",
        "title": "Múltiplos modelos OpenAI com custos e capacidades distintas",
        "description": "gpt-4.1-nano (classifier), gpt-4o-mini (SDR), gpt-5.4-pro (Closer). Dependência de 3 modelos externos.",
        "recommendation": "Migrar classifier e SDR para Claude Haiku. Closer para Claude Sonnet. Reduz fornecedores.",
        "affected_component": "Model SDR / Model Closer / OpenAI Chat Model",
        "score_impact": -5,
    },
]

_DUPLICATE_ROUTING = {
    "severity": "warning",
    "category": "architecture",
    "title": "Lógica de routing definida em 3 nodes distintos",
    "description": (
        "O route_hint é computado pelo Classifier Agent, sobrescrito pelo Compute Score & Tags "
        "e novamente ajustado pelo Finalize Funnel Decision. Três pontos de decisão para a mesma variável."
    ),
    "recommendation": "Centralizar routing no Brain AI (decision_engine.py). n8n só recebe o resultado.",
    "affected_component": "Classifier Agent / Compute Score & Tags / Finalize Funnel Decision",
    "score_impact": -8,
}


def analyze(persona_id: Optional[str] = None) -> list[dict]:
    insights = []

    workflow_texts = []
    for path in _WORKFLOWS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            workflow_texts.append((path.name, json.dumps(data)))
        except Exception:
            continue

    if not workflow_texts:
        return insights

    found_types: set[str] = set()
    for _, text in workflow_texts:
        for pattern in _ANTI_PATTERNS:
            if pattern["key"] in text and pattern["key"] not in found_types:
                found_types.add(pattern["key"])
                base = {k: v for k, v in pattern.items() if k != "key"}
                base["persona_id"] = persona_id
                insights.append(base)

    # detectar routing triplicado: presença de todos os 3 nodes
    all_text = " ".join(t for _, t in workflow_texts)
    routing_nodes = ["Compute Score", "Finalize Funnel Decision", "Classifier Agent"]
    if all(node in all_text for node in routing_nodes):
        dup = dict(_DUPLICATE_ROUTING)
        dup["persona_id"] = persona_id
        insights.append(dup)

    # hardcoded stage mapping
    if '"Contato Inicial"' in all_text and '"Novo lead"' in all_text:
        insights.append({
            "persona_id": persona_id,
            "severity": "warning",
            "category": "architecture",
            "title": "Stage names hardcoded em expressões ternárias no n8n",
            "description": (
                "O mapeamento de stages (ex: 'contatado' → 'Contato Inicial') está espalhado "
                "em múltiplos nodes como expressões inline. Qualquer mudança exige editar N lugares."
            ),
            "recommendation": "Centralizar enum de stages em tabela Supabase ou constante no Brain AI.",
            "affected_component": "Midware Crm / Edit Fields nodes",
            "score_impact": -5,
        })

    return insights
