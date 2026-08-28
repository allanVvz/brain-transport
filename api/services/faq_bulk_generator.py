"""Generate real FAQ (question+answer) content for one branch of the
knowledge tree -- the piece `sofia_tools.tool_generate_faq_from_branch` has
always been a placeholder for (see its docstring: "O conteudo real ...
sera preenchido pelo worker da Janela 4" -- that worker was never built).

Two callers share this one engine:
1. `sofia_tools.tool_generate_faq_from_branch`, inline during a Sofia chat
   session -- the branch chain comes from the in-progress plan.
2. `POST /knowledge/graph/{node_id}/generate-faqs` (api/routes/knowledge.py)
   for an already-published node, outside any chat session -- the branch
   chain comes from the live graph (`knowledge_nodes`/`knowledge_edges`).

Both produce plain `{question, answer}` pairs grounded ONLY in the branch's
own title/content/tags -- the prompt explicitly forbids inventing facts not
present in the chain, mirroring the project's "não inventar produtos/preço"
rule already enforced elsewhere (docs/tock-fatal-modal-marketing-graph.md's
`regra-nao-inventar-produtos-tock`).

Writing quality: extends the same "load a markdown file as prompt content"
pattern already used by `kb_intake_service._load_agent_prompt` (which loads
`agents/sofia_criar.md`), but for `.claude/skills/*/SKILL.md` files -- no
production code read a skill file before this.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from services.model_router import ModelRouter, ModelRouterError

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Skills whose guidance is relevant to writing FAQ copy for a conversational
# sales/SDR agent. Extend this list as more persona-specific skills land
# under .claude/skills/ -- missing files are skipped silently (fallback to
# just the branch content, no skill has ever been a hard requirement).
_DEFAULT_SKILLS = ("aurora-premium-sdr", "aurora-conversation-evaluator")


def _load_skill_content(skill_name: str) -> str:
    path = _REPO_ROOT / ".claude" / "skills" / skill_name / "SKILL.md"
    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


def _skills_context(skill_names: tuple[str, ...]) -> str:
    blocks = []
    for name in skill_names:
        content = _load_skill_content(name)
        if content:
            blocks.append(f"### Skill: {name}\n{content[:4000]}")
    if not blocks:
        return ""
    return (
        "Guias de tom/qualidade de escrita já validados neste projeto "
        "(aplique o espírito, não copie literalmente -- essas skills foram "
        "escritas para outra persona):\n\n" + "\n\n".join(blocks)
    )


def _chain_to_text(chain: list[dict[str, Any]]) -> str:
    lines = []
    for node in chain:
        title = str(node.get("title") or node.get("slug") or "").strip()
        content = str(node.get("content") or node.get("summary") or "").strip()
        node_type = str(node.get("node_type") or node.get("content_type") or "").strip()
        line = f"- [{node_type}] {title}"
        if content:
            line += f": {content[:500]}"
        lines.append(line)
    return "\n".join(lines)


def generate_faqs_for_chain(
    chain: list[dict[str, Any]],
    *,
    max_questions: int = 8,
    model: str = "gpt-4o-mini",
    skills: tuple[str, ...] = _DEFAULT_SKILLS,
) -> list[dict[str, str]]:
    """Return up to `max_questions` `{question, answer}` pairs grounded in
    `chain` (branch ancestors, closest node first -- see the two adapters
    below for how each caller builds this list). Returns [] on any LLM
    failure or unparseable output -- callers keep the placeholder entry
    rather than silently publishing fabricated content."""
    if not chain:
        return []
    max_questions = max(1, min(20, int(max_questions or 8)))
    branch_text = _chain_to_text(chain)
    skills_text = _skills_context(skills)
    prompt = (
        f"Gere até {max_questions} pares de pergunta e resposta (FAQ) que um "
        "cliente real perguntaria sobre este ramo do catálogo, no WhatsApp. "
        "Use SOMENTE os fatos abaixo -- nunca invente preço, prazo, estoque, "
        "composição ou qualquer dado que não esteja explicitamente escrito. "
        "Perguntas curtas e diretas, como uma pessoa real escreveria. "
        "Respostas curtas, no mesmo tom do ramo.\n\n"
        f"Ramo (do mais específico ao mais geral):\n{branch_text}\n\n"
        + (f"{skills_text}\n\n" if skills_text else "")
        + 'Responda SOMENTE um array JSON: [{"question": str, "answer": str}, ...]. '
        "Sem texto fora do array."
    )
    try:
        router = ModelRouter()
        raw = router.messages_create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            system=(
                "Você escreve FAQ para atendimento comercial no WhatsApp. "
                "Nunca inventa fato não fornecido. Responde só com JSON válido."
            ),
            max_tokens=4000,
        )
    except ModelRouterError:
        return []
    match = re.search(r"\[.*\]", raw if isinstance(raw, str) else "", re.S)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return []
    pairs: list[dict[str, str]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if question and answer:
            pairs.append({"question": question, "answer": answer})
    return pairs[:max_questions]


def build_chain_from_live_graph(node_rows: list[dict], edge_rows: list[dict], node_id: str) -> list[dict[str, Any]]:
    """Adapter for the graph-sidebar entry point: walk `contains` edges
    upward from `node_id` (an existing, published `knowledge_nodes.id`) to
    the persona root, using the already-fetched full node/edge set for a
    persona (e.g. `supabase_client.list_all_knowledge_graph(persona_id=...)`)."""
    nodes_by_id = {str(row.get("id")): row for row in node_rows}
    parent_by_child: dict[str, str] = {}
    for edge in edge_rows:
        metadata = edge.get("metadata") or {}
        if str(edge.get("relation_type") or "") != "contains":
            continue
        if metadata.get("active", True) is False:
            continue
        parent_by_child[str(edge.get("target_node_id"))] = str(edge.get("source_node_id"))

    chain: list[dict[str, Any]] = []
    cursor = str(node_id)
    seen: set[str] = set()
    while cursor and cursor not in seen and len(chain) < 16:
        seen.add(cursor)
        row = nodes_by_id.get(cursor)
        if not row:
            break
        chain.append({
            "node_type": row.get("node_type"),
            "title": row.get("title"),
            "content": row.get("summary"),
            "tags": row.get("tags") or [],
        })
        cursor = parent_by_child.get(cursor) or ""
    return chain
