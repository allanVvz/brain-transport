# -*- coding: utf-8 -*-
"""
Sofia tools â€” surface deterministica para mutacao incremental do
knowledge_plan que vive em `session.normalized_plan`.

Antes desta camada, cada mensagem operacional disparava regeneracao
completa do plano via regex em `<knowledge_plan>{json}</knowledge_plan>`
(`kb_intake_service.py:_extract_plan`). Tools dao a Sofia (modelo LLM) um
caminho estruturado para criar nodes, mudar parents, anexar assets e
ajustar expansion policies sem reescrever o JSON inteiro a cada turno.

Cada handler:
  * recebe a session dict do kb_intake e os argumentos validados pelo
    provider (OpenAI/Anthropic tool-use);
  * muta o normalized_plan IN PLACE;
  * roda normalize_validate_summarize_plan em seguida e armazena o
    plan_state via _store_plan_state â€” o front recebe o estado novo na
    proxima resposta de /kb-intake/message;
  * devolve um dict serializavel com {ok, summary, errors, ...} que o
    modelo le como tool_result para decidir o proximo passo.

A schema JSON publicada em SOFIA_TOOLS_SCHEMA segue o formato compativel
com Anthropic tools (tambem aceito pela OpenAI Responses/Chat Completions
APIs com transformacao simples â€” vide model_router.messages_create).
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from schemas.agent_harness import ToolEffect, ToolResult, ToolRisk
from services.agent_tool_registry import ToolManifest, ToolRegistry

# Importacoes do servico Sofia: ficam dentro das funcoes para evitar
# import circular (kb_intake_service ja importa este modulo).


def _kb():
    """Atalho preguicoso para o modulo do servico â€” evita import circular."""
    from services import kb_intake_service as _service
    return _service


# --------------------------------------------------------------------------- #
# Helpers internos                                                            #
# --------------------------------------------------------------------------- #

def _plan_from_session(session: dict) -> dict:
    plan = session.get("normalized_plan") or session.get("knowledge_plan") or {"entries": [], "links": []}
    if not isinstance(plan, dict):
        plan = {"entries": [], "links": []}
    plan.setdefault("entries", [])
    plan.setdefault("links", [])
    return plan


def _persona_slug(session: dict, plan: dict) -> str:
    classification = session.get("classification") or {}
    return str(
        plan.get("persona_slug")
        or session.get("persona_slug")
        or classification.get("persona_slug")
        or "global"
    ).strip() or "global"


def _commit(session: dict, plan: dict, *, last_change: str) -> dict:
    """Re-normaliza e re-valida o plano apos a mutacao, persiste no session.

    Retorna o `plan_state` resultante (mesma shape que o /kb-intake/message
    devolve no campo `plan_state`).
    """
    kb = _kb()
    plan_state = kb.normalize_validate_summarize_plan(plan, session, live_edit=True)
    kb._store_plan_state(session, plan_state, last_change=last_change)
    return plan_state


def _ok(summary: str, plan_state: dict, **extra: Any) -> dict:
    state_summary = plan_state.get("summary") or {}
    validation = plan_state.get("validation") or {}
    return {
        "ok": True,
        "summary": summary,
        "entry_count": state_summary.get("entry_count"),
        "valid": validation.get("valid", True),
        "blocking_violations": validation.get("blocking_violations") or [],
        **extra,
    }


def _err(summary: str, **extra: Any) -> dict:
    return {"ok": False, "summary": summary, **extra}


def _find_entry(plan: dict, slug: str) -> Optional[dict]:
    target = (slug or "").strip().lower()
    if not target:
        return None
    for entry in plan.get("entries") or []:
        if str(entry.get("slug") or "").strip().lower() == target:
            return entry
    return None


# --------------------------------------------------------------------------- #
# Handlers                                                                    #
# --------------------------------------------------------------------------- #

def _canonical_parent_for(content_type: str) -> Optional[str]:
    """Return the canonical parent type for a child type per the fractal hierarchy.
    persona -> brand -> briefing -> campaign -> audience -> product_group ->
    product -> offer -> copy -> {faq, gallery}. Asset is lateral (no parent rule)."""
    from services import knowledge_taxonomy
    child = knowledge_taxonomy.canonical_node_type(content_type)
    for src, tgt, _rel in knowledge_taxonomy.PRIMARY_CHAIN:
        if tgt == child:
            return src
    return None


def _violates_hierarchy(plan: dict, content_type: str, parent_slug: Optional[str]) -> Optional[str]:
    """Return a violation message, or None when the parent type is canonical for the child.

    Asset and gallery are lateral and accept any parent. Persona is the root
    (parent must be empty). Every other type must have a parent_slug, and that
    parent's content_type must match PRIMARY_CHAIN. Aliases (product_collection,
    category -> product_group) are normalized."""
    from services import knowledge_taxonomy
    child = knowledge_taxonomy.canonical_node_type(content_type) or content_type
    if child == "asset":
        return None  # lateral
    if child == "persona":
        if parent_slug:
            return "persona e raiz da arvore â€” nao deve ter parent_slug"
        return None
    if not parent_slug:
        expected = _canonical_parent_for(child)
        return f"{child} exige parent_slug do tipo {expected}" if expected else None
    entries_by_slug = {
        str(e.get("slug") or "").lower(): e
        for e in (plan.get("entries") or [])
        if isinstance(e, dict)
    }
    parent_entry = entries_by_slug.get(str(parent_slug).strip().lower())
    if not parent_entry:
        return f"parent_slug '{parent_slug}' nao existe no plano â€” crie o pai antes"
    parent_type = knowledge_taxonomy.canonical_node_type(parent_entry.get("content_type")) or parent_entry.get("content_type")
    if not knowledge_taxonomy.is_primary_edge_allowed(parent_type, child, ""):
        expected = _canonical_parent_for(child) or "?"
        return (
            f"hierarquia invalida: {parent_type} -> {child} nao e canonico "
            f"(esperado pai {expected})"
        )
    return None


def tool_create_node(session: dict, **args: Any) -> dict:
    """Adiciona uma entry ao knowledge_plan.

    Enforce canonical fractal hierarchy: persona->brand->briefing->campaign->
    audience->product_group->product->offer->copy->{faq,gallery}. Asset is
    lateral. Violations short-circuit with an explicit error so Sofia can
    fix the chain before continuing â€” keeps the tree top-down and clean.
    """
    kb = _kb()
    content_type = str(args.get("content_type") or "").strip().lower()
    if not content_type:
        return _err("content_type obrigatorio")
    title = str(args.get("title") or "").strip()
    if not title:
        return _err("title obrigatorio")
    content = str(args.get("content") or title)
    slug = str(args.get("slug") or "").strip() or kb._slug_for_plan_entry(title)
    plan = _plan_from_session(session)
    if _find_entry(plan, slug):
        return _err(f"ja existe entry com slug={slug}", slug=slug)
    parent_slug = args.get("parent_slug")
    if parent_slug is not None and not str(parent_slug).strip():
        parent_slug = None
    violation = _violates_hierarchy(plan, content_type, parent_slug)
    if violation:
        return _err(
            violation,
            content_type=content_type,
            parent_slug=parent_slug,
            canonical_parent=_canonical_parent_for(content_type),
        )
    tags = args.get("tags") or []
    metadata = args.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    if parent_slug:
        metadata = {**metadata, "parent_slug": str(parent_slug)}
    status = str(args.get("status") or "pendente_validacao")
    entry = kb._normalize_plan_entry({
        "content_type": content_type,
        "title": title,
        "slug": slug,
        "status": status,
        "content": content,
        "tags": list(tags),
        "metadata": metadata,
    })
    plan["entries"].append(entry)
    plan_state = _commit(session, plan, last_change=f"tool:create_node {content_type}:{slug}")
    return _ok(f"criado {content_type}:{slug}", plan_state, slug=slug)


def tool_set_parent(session: dict, **args: Any) -> dict:
    """Atualiza metadata.parent_slug de uma entry existente, validando hierarquia."""
    slug = str(args.get("slug") or "").strip()
    parent_slug = args.get("parent_slug")
    if not slug:
        return _err("slug obrigatorio")
    plan = _plan_from_session(session)
    entry = _find_entry(plan, slug)
    if not entry:
        return _err(f"entry nao encontrada: {slug}")
    if parent_slug is not None and not str(parent_slug).strip():
        parent_slug = None
    violation = _violates_hierarchy(plan, str(entry.get("content_type") or ""), parent_slug)
    if violation:
        return _err(violation, slug=slug, parent_slug=parent_slug)
    kb = _kb()
    kb._set_entry_parent_slug(entry, str(parent_slug) if parent_slug else None)
    plan_state = _commit(session, plan, last_change=f"tool:set_parent {slug}->{parent_slug}")
    return _ok(f"parent de {slug} agora aponta para {parent_slug or 'nada'}", plan_state)


def tool_connect_nodes(session: dict, **args: Any) -> dict:
    """Adiciona um link em plan.links (relation_type semantico)."""
    source_slug = str(args.get("source_slug") or "").strip()
    target_slug = str(args.get("target_slug") or "").strip()
    relation_type = str(args.get("relation_type") or "contains").strip() or "contains"
    if not source_slug or not target_slug:
        return _err("source_slug e target_slug sao obrigatorios")
    plan = _plan_from_session(session)
    links = plan.setdefault("links", [])
    # dedupe pelo trio
    for link in links:
        if (
            str(link.get("source_slug") or "").strip() == source_slug
            and str(link.get("target_slug") or "").strip() == target_slug
            and str(link.get("relation_type") or "").strip() == relation_type
        ):
            return _ok("link ja existia", _commit(session, plan, last_change="tool:connect_nodes noop"))
    links.append({
        "source_slug": source_slug,
        "target_slug": target_slug,
        "relation_type": relation_type,
    })
    plan_state = _commit(session, plan, last_change=f"tool:connect_nodes {source_slug}->{target_slug}")
    return _ok(f"link {source_slug} -[{relation_type}]-> {target_slug}", plan_state)


def tool_delete_node(session: dict, **args: Any) -> dict:
    """Remove uma entry e os links que a referenciam."""
    slug = str(args.get("slug") or "").strip()
    if not slug:
        return _err("slug obrigatorio")
    plan = _plan_from_session(session)
    entries = plan.get("entries") or []
    new_entries = [e for e in entries if str(e.get("slug") or "").strip().lower() != slug.lower()]
    if len(new_entries) == len(entries):
        return _err(f"entry nao encontrada: {slug}")
    plan["entries"] = new_entries
    plan["links"] = [
        link for link in (plan.get("links") or [])
        if str(link.get("source_slug") or "").strip().lower() != slug.lower()
        and str(link.get("target_slug") or "").strip().lower() != slug.lower()
    ]
    plan_state = _commit(session, plan, last_change=f"tool:delete_node {slug}")
    return _ok(f"removido {slug}", plan_state)


def tool_set_expansion_policy(session: dict, **_args: Any) -> dict:
    """DEPRECATED. A polÃ­tica de expansÃ£o piramidal foi removida.

    Mantida como stub para nÃ£o quebrar o tool-loop caso o modelo ainda
    invoque (cache do prompt anterior, plano legado etc.). NÃ£o muta nada
    e retorna ok=False com mensagem clara.
    """
    return _err(
        "set_expansion_policy foi removida. Use generate_faq_from_branch "
        "para FAQ e crie demais nodes apenas sob pedido do operador."
    )


def tool_generate_faq_from_branch(session: dict, **args: Any) -> dict:
    """Gera entries de FAQ reais lendo o galho ancestral do `parent_slug` no plano.

    Em modo CRIAR o galho fica no `session.normalized_plan`. Esta tool:
      * Caminha do `parent_slug` atÃ© a persona via `metadata.parent_slug`,
        coletando tÃ­tulo/content/tags.
      * Calcula `branch_hash` (sha256 do dump canÃ´nico) para curadoria.
      * Chama `services.faq_bulk_generator.generate_faqs_for_chain` para
        gerar as perguntas/respostas de verdade (LLM, grounded sÃ³ no que
        estÃ¡ no galho, com o contexto das skills de tom jÃ¡ validadas no
        repositÃ³rio) e cria uma entry `faq` por par, cada uma com
        `status='pendente_validacao'` (curadoria continua exigida antes de
        publicar). Se a geraÃ§Ã£o falhar (LLM indisponÃ­vel, saÃ­da
        inaparseÃ¡vel), cai de volta para uma Ãºnica entry placeholder, como
        antes.
    """
    import hashlib
    import json as _json

    parent_slug = str(args.get("parent_slug") or "").strip()
    if not parent_slug:
        return _err("parent_slug obrigatorio")
    try:
        max_questions = max(1, min(20, int(args.get("max_questions") or 8)))
    except (TypeError, ValueError):
        max_questions = 8

    plan = _plan_from_session(session)
    entries_by_slug = {
        str(e.get("slug") or "").lower(): e
        for e in (plan.get("entries") or [])
        if isinstance(e, dict)
    }
    chain: list[dict] = []
    seen: set[str] = set()
    cursor = parent_slug.lower()
    while cursor and cursor not in seen and len(chain) < 12:
        seen.add(cursor)
        entry = entries_by_slug.get(cursor)
        if not entry:
            break
        chain.append({
            "slug": entry.get("slug"),
            "content_type": entry.get("content_type"),
            "title": entry.get("title"),
            "content": entry.get("content"),
            "tags": entry.get("tags") or [],
        })
        meta = entry.get("metadata") or {}
        next_slug = str(meta.get("parent_slug") or "").strip().lower()
        cursor = next_slug or ""

    if not chain:
        return _err(
            f"parent_slug '{parent_slug}' nao encontrado no plano; crie a "
            "copy/produto primeiro antes de gerar FAQ"
        )

    branch_payload = _json.dumps(chain, ensure_ascii=False, sort_keys=True)
    branch_hash = hashlib.sha256(branch_payload.encode("utf-8")).hexdigest()[:16]

    from services import faq_bulk_generator

    kb = _kb()
    parent_entry = chain[0]
    base_title = parent_entry.get("title") or parent_entry.get("slug") or parent_slug
    base_slug = f"faq-{kb._slug_for_plan_entry(base_title)}"

    def _next_slug(base: str) -> str:
        slug = base
        suffix = 1
        while _find_entry(plan, slug):
            suffix += 1
            slug = f"{base}-{suffix}"
        return slug

    pairs = faq_bulk_generator.generate_faqs_for_chain(chain, max_questions=max_questions)
    created_slugs: list[str] = []

    if pairs:
        for pair in pairs:
            slug = _next_slug(base_slug)
            entry = kb._normalize_plan_entry({
                "content_type": "faq",
                "title": pair["question"],
                "slug": slug,
                "status": "pendente_validacao",
                "content": pair["answer"],
                "tags": ["faq", "auto-from-branch"],
                "metadata": {
                    "parent_slug": parent_slug,
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "generate_via": "branch",
                    "source_branch_hash": branch_hash,
                    "branch_chain": [c.get("slug") for c in chain if c.get("slug")],
                },
            })
            plan["entries"].append(entry)
            created_slugs.append(slug)
        message = f"{len(created_slugs)} FAQs geradas para o galho de {parent_slug}"
    else:
        # LLM indisponÃ­vel ou saÃ­da inaparseÃ¡vel -- volta pro placeholder
        # antigo em vez de deixar o operador sem nada.
        slug = _next_slug(base_slug)
        faq_title = f"FAQ â€” {base_title}"
        placeholder_md = (
            "<!-- GeraÃ§Ã£o automÃ¡tica indisponÃ­vel no momento; revise manualmente. -->\n"
            f"## {faq_title}\n\nPerguntas pendentes para o galho:\n"
            + "\n".join(f"- {c.get('content_type')}: {c.get('title')}" for c in chain)
        )
        entry = kb._normalize_plan_entry({
            "content_type": "faq",
            "title": faq_title,
            "slug": slug,
            "status": "pendente_validacao",
            "content": placeholder_md,
            "tags": ["faq", "auto-from-branch", "generation-failed"],
            "metadata": {
                "parent_slug": parent_slug,
                "generate_via": "branch",
                "source_branch_hash": branch_hash,
                "branch_chain": [c.get("slug") for c in chain if c.get("slug")],
                "max_questions": max_questions,
            },
        })
        plan["entries"].append(entry)
        created_slugs.append(slug)
        message = f"GeraÃ§Ã£o automÃ¡tica falhou; criei um placeholder para o galho de {parent_slug}"

    plan_state = _commit(
        session, plan, last_change=f"tool:generate_faq_from_branch {parent_slug}",
    )
    return _ok(
        message,
        plan_state,
        slugs=created_slugs,
        branch_hash=branch_hash,
        branch_depth=len(chain),
    )


def tool_validate_hierarchy(session: dict, **args: Any) -> dict:
    """Valida que o `slug` estÃ¡ em um caminho canÃ´nico do grafo fractal."""
    from services import knowledge_taxonomy

    slug = str(args.get("slug") or "").strip().lower()
    if not slug:
        return _err("slug obrigatorio")
    plan = _plan_from_session(session)
    entries_by_slug = {
        str(e.get("slug") or "").lower(): e
        for e in (plan.get("entries") or [])
        if isinstance(e, dict)
    }
    entry = entries_by_slug.get(slug)
    if not entry:
        return _err(f"slug {slug} nao esta no plano")
    chain: list[tuple[str, str]] = []
    seen: set[str] = set()
    cursor = slug
    while cursor and cursor not in seen and len(chain) < 12:
        seen.add(cursor)
        cur_entry = entries_by_slug.get(cursor)
        if not cur_entry:
            break
        ctype = knowledge_taxonomy.canonical_node_type(cur_entry.get("content_type")) or cur_entry.get("content_type")
        chain.append((cursor, ctype or "?"))
        meta = cur_entry.get("metadata") or {}
        cursor = str(meta.get("parent_slug") or "").strip().lower() or ""
    violations: list[str] = []
    for i in range(len(chain) - 1):
        child_slug, child_type = chain[i]
        parent_slug, parent_type = chain[i + 1]
        if not knowledge_taxonomy.is_primary_edge_allowed(parent_type, child_type, ""):
            allowed = any(
                p == parent_type and c == child_type
                for p, c, _ in knowledge_taxonomy.PRIMARY_CHAIN
            )
            if not allowed:
                violations.append(
                    f"{parent_type} -> {child_type} nao e primary canonico (slug {child_slug})"
                )
    return {
        "ok": not violations,
        "summary": "canonical" if not violations else f"{len(violations)} violacoes",
        "chain": [{"slug": s, "type": t} for s, t in chain],
        "violations": violations,
    }


def tool_attach_session_asset(session: dict, **args: Any) -> dict:
    """Cria uma entry asset a partir de um arquivo ja em session.asset_readings.

    Args:
      reading_index (int) â€” indice em session.asset_readings (default 0,
        ou seja o ultimo upload se a session estiver mantendo so o mais
        recente).
      parent_slug (str) â€” slug do produto/oferta pai. Obrigatorio.
      asset_function (str?) â€” campo metadata.asset_function.
      title (str?) â€” sobrescreve o titulo derivado do filename.
    """
    parent_slug = str(args.get("parent_slug") or "").strip()
    if not parent_slug:
        return _err("parent_slug obrigatorio para conectar o asset")
    readings = session.get("asset_readings") or []
    if not readings:
        return _err("session nao tem asset_readings pendentes; o operador precisa subir um asset primeiro")
    reading_index = int(args.get("reading_index") or len(readings) - 1)
    if not (0 <= reading_index < len(readings)):
        return _err(f"reading_index fora do intervalo (0..{len(readings)-1})")
    reading = readings[reading_index]
    file_meta = reading.get("file") if isinstance(reading, dict) else {}
    file_meta = file_meta if isinstance(file_meta, dict) else {}
    filename = file_meta.get("filename") or file_meta.get("name") or f"asset-{reading_index}"
    asset_url = file_meta.get("url") or file_meta.get("public_url")
    storage_path = file_meta.get("storage_path") or file_meta.get("asset_id")
    title = str(args.get("title") or filename or "Asset visual")
    asset_function = str(args.get("asset_function") or "product_showcase")
    kb = _kb()
    slug_seed = f"asset-{kb._slug_for_plan_entry(title)}"
    plan = _plan_from_session(session)
    base_slug = slug_seed
    suffix = 1
    while _find_entry(plan, slug_seed):
        suffix += 1
        slug_seed = f"{base_slug}-{suffix}"
    visual_summary = ""
    if isinstance(reading.get("reading"), dict):
        visual_summary = str(reading["reading"].get("visual_summary") or reading["reading"].get("extracted_text") or "")
    metadata = {
        "parent_slug": parent_slug,
        "asset_function": asset_function,
        "source_filename": filename,
        "source_reading_index": reading_index,
    }
    if asset_url:
        metadata["asset_url"] = asset_url
    if storage_path:
        metadata["storage_path"] = storage_path
    entry = kb._normalize_plan_entry({
        "content_type": "asset",
        "title": title,
        "slug": slug_seed,
        "status": "pendente_validacao",
        "content": visual_summary or f"Asset visual {title}",
        "tags": ["asset", asset_function],
        "metadata": metadata,
    })
    plan["entries"].append(entry)
    plan_state = _commit(session, plan, last_change=f"tool:attach_session_asset {slug_seed}->{parent_slug}")
    return _ok(
        f"asset {title} conectado a {parent_slug} (uses_asset)",
        plan_state,
        slug=slug_seed,
        asset_url=asset_url,
    )


def tool_validate_plan(session: dict, **_args: Any) -> dict:
    """Re-roda a validacao do plano e devolve o diagnostico atual."""
    plan = _plan_from_session(session)
    plan_state = _commit(session, plan, last_change="tool:validate_plan")
    validation = plan_state.get("validation") or {}
    diagnostic = plan_state.get("diagnostic") or {}
    return {
        "ok": True,
        "summary": "valid" if validation.get("valid") else "blocked",
        "valid": validation.get("valid", True),
        "blocking_violations": validation.get("blocking_violations") or [],
        "root_causes": [rc.get("kind") for rc in (diagnostic.get("root_causes") or [])],
        "sofia_questions": diagnostic.get("sofia_questions") or [],
    }


def tool_find_existing_persona_nodes(session: dict, **args: Any) -> dict:
    """Consulta o grafo da persona em busca de nodes que a Sofia pode reusar.

    Hoje aproveita o `pre_init_review` ja calculado pelo kb_intake quando
    a sessao foi iniciada. Para tipos/buscas fora desse cache, consulta o
    supabase via supabase_client.
    """
    types_arg = args.get("types") or []
    if isinstance(types_arg, str):
        types_arg = [types_arg]
    types = [str(t).strip().lower() for t in types_arg if str(t).strip()]
    query = str(args.get("query") or "").strip().lower()
    pre_init = session.get("pre_init_review") if isinstance(session.get("pre_init_review"), dict) else None
    found: dict[str, list[dict]] = {}
    if pre_init and isinstance(pre_init.get("existing_nodes_found"), dict):
        existing = pre_init["existing_nodes_found"]
        for ntype, items in existing.items():
            if types and ntype not in types:
                continue
            bucket: list[dict] = []
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                if query and query not in str(item.get("slug") or "").lower() and query not in str(item.get("title") or "").lower():
                    continue
                bucket.append({
                    "slug": item.get("slug"),
                    "title": item.get("title"),
                    "node_type": ntype,
                })
            if bucket:
                found[ntype] = bucket[:10]
    return {
        "ok": True,
        "summary": f"{sum(len(v) for v in found.values())} nodes encontrados",
        "nodes_by_type": found,
        "from_cache": bool(pre_init),
    }


# --------------------------------------------------------------------------- #
# Registry + JSON Schema                                                      #
# --------------------------------------------------------------------------- #

_LEGACY_TOOLS_REGISTRY_UNUSED: dict[str, Any] = {
    "create_node": tool_create_node,
    "set_parent": tool_set_parent,
    "connect_nodes": tool_connect_nodes,
    "delete_node": tool_delete_node,
    "set_expansion_policy": tool_set_expansion_policy,  # deprecated stub
    "attach_session_asset": tool_attach_session_asset,
    "validate_plan": tool_validate_plan,
    "find_existing_persona_nodes": tool_find_existing_persona_nodes,
    "generate_faq_from_branch": tool_generate_faq_from_branch,
    "validate_hierarchy": tool_validate_hierarchy,
}


_LEGACY_TOOLS_SCHEMA_UNUSED: list[dict] = [
    {
        "name": "create_node",
        "description": (
            "Adiciona uma entry ao knowledge_plan. Use para criar brand, briefing, "
            "campaign, audience, product, offer, copy, rule, faq, asset ou tone. "
            "Sempre informe parent_slug exceto para top-level (brand, briefing)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content_type": {"type": "string", "description": "Tipo do node. Ex: product, faq, copy, rule."},
                "title": {"type": "string"},
                "content": {"type": "string"},
                "slug": {"type": "string", "description": "Opcional; derivado do title quando ausente."},
                "parent_slug": {"type": "string", "description": "Slug do node pai. 'self' significa direto na persona."},
                "tags": {"type": "array", "items": {"type": "string"}},
                "metadata": {"type": "object", "description": "Campos arbitrarios; nunca repita parent_slug."},
                "status": {"type": "string", "description": "confirmado | inferido | pendente_validacao"},
            },
            "required": ["content_type", "title"],
        },
    },
    {
        "name": "set_parent",
        "description": "Move um node existente para outro parent. Use para corrigir hierarquia errada.",
        "input_schema": {
            "type": "object",
            "properties": {
                "slug": {"type": "string"},
                "parent_slug": {"type": "string", "description": "Novo parent. Use 'self' para mover para a raiz."},
            },
            "required": ["slug", "parent_slug"],
        },
    },
    {
        "name": "connect_nodes",
        "description": "Adiciona uma edge semantica em plan.links. Use para relacoes alem da arvore principal.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_slug": {"type": "string"},
                "target_slug": {"type": "string"},
                "relation_type": {
                    "type": "string",
                    "description": "Ex: uses_asset, answers_question, supports_copy, belongs_to_persona, contains.",
                },
            },
            "required": ["source_slug", "target_slug", "relation_type"],
        },
    },
    {
        "name": "delete_node",
        "description": "Remove um node do plano e todas as edges que o referenciam.",
        "input_schema": {
            "type": "object",
            "properties": {"slug": {"type": "string"}},
            "required": ["slug"],
        },
    },
    {
        "name": "set_expansion_policy",
        "description": "DEPRECATED. A politica de expansao piramidal foi removida. Nao chame esta tool â€” use generate_faq_from_branch para FAQ e crie os demais nodes apenas sob pedido do operador.",
        "input_schema": {
            "type": "object",
            "properties": {
                "block": {"type": "string", "enum": ["faq", "asset"]},
            },
            "required": ["block"],
        },
    },
    {
        "name": "generate_faq_from_branch",
        "description": (
            "Gera uma entry FAQ placeholder a partir do galho ancestral de `parent_slug` "
            "(geralmente uma copy). O conteudo real e preenchido depois pela curadoria, "
            "lendo persona -> ... -> copy em tempo real. Nao escreva perguntas a mao; "
            "chame esta tool sempre que o operador pedir FAQ."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "parent_slug": {"type": "string", "description": "Slug do node pai (geralmente a copy)."},
                "max_questions": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8},
            },
            "required": ["parent_slug"],
        },
    },
    {
        "name": "validate_hierarchy",
        "description": (
            "Valida que o caminho ancestral do `slug` no plano segue a hierarquia "
            "canonica do grafo fractal (persona -> brand -> campaign -> briefing -> "
            "audience -> product_group -> product -> offer -> copy -> {faq, gallery})."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "slug": {"type": "string"},
            },
            "required": ["slug"],
        },
    },
    {
        "name": "attach_session_asset",
        "description": (
            "Quando o operador acaba de subir um arquivo via paperclip, cria a entry asset no plano "
            "conectada ao produto/oferta pai. Le session.asset_readings â€” nao inventa URL."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "parent_slug": {"type": "string"},
                "reading_index": {"type": "integer", "minimum": 0},
                "asset_function": {"type": "string", "description": "product_showcase, brand_reference, campaign_hero, copy_support, maker_material."},
                "title": {"type": "string"},
            },
            "required": ["parent_slug"],
        },
    },
    {
        "name": "validate_plan",
        "description": "Re-roda a validacao e devolve o diagnostico atual. Use antes de encerrar a resposta para garantir que nao restam violacoes.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "find_existing_persona_nodes",
        "description": (
            "Lista nodes que ja existem para a persona da sessao, antes de criar duplicados. "
            "Filtra por types e query (substring de slug/title)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "types": {"type": "array", "items": {"type": "string"}},
                "query": {"type": "string"},
            },
        },
    },
]


class _ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateNodeArgs(_ToolArgs):
    content_type: str
    title: str
    content: str | None = None
    slug: str | None = None
    parent_slug: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: str = "pendente_validacao"


class SetParentArgs(_ToolArgs):
    slug: str
    parent_slug: str


class ConnectNodesArgs(_ToolArgs):
    source_slug: str
    target_slug: str
    relation_type: str


class SlugArgs(_ToolArgs):
    slug: str


class ExpansionArgs(_ToolArgs):
    block: str


class FaqArgs(_ToolArgs):
    parent_slug: str
    max_questions: int = Field(8, ge=1, le=20)


class AssetArgs(_ToolArgs):
    parent_slug: str
    reading_index: int | None = Field(None, ge=0)
    asset_function: str | None = None
    title: str | None = None


class EmptyArgs(_ToolArgs):
    pass


class FindNodesArgs(_ToolArgs):
    types: list[str] = Field(default_factory=list)
    query: str | None = None


def _legacy_manifest(
    name: str,
    model: type[BaseModel],
    handler: Any,
    description: str,
    *,
    effect: ToolEffect = ToolEffect.DRAFT,
    risk: ToolRisk = ToolRisk.LOW,
    deprecated: bool = False,
) -> ToolManifest:
    return ToolManifest(
        name=name,
        version="1.0.0",
        owner_agent="qa_validator" if name.startswith("validate") else "graph_card_specialist",
        description=description,
        input_model=model,
        output_model=ToolResult,
        effect=effect,
        risk=risk,
        permission="view" if effect == ToolEffect.READ else "edit",
        persona_scoped=True,
        timeout_seconds=30,
        idempotent=True,
        handler=handler,
        deprecated=deprecated,
    )


SOFIA_TOOL_MANIFESTS = ToolRegistry([
    _legacy_manifest("create_node", CreateNodeArgs, tool_create_node, "Cria um draft de node no plano canonico."),
    _legacy_manifest("set_parent", SetParentArgs, tool_set_parent, "Move um draft para um parent canonico."),
    _legacy_manifest("connect_nodes", ConnectNodesArgs, tool_connect_nodes, "Propoe uma edge semantica no plano."),
    _legacy_manifest("delete_node", SlugArgs, tool_delete_node, "Compatibilidade antiga de remocao de draft.", effect=ToolEffect.DESTRUCTIVE, risk=ToolRisk.HIGH, deprecated=True),
    _legacy_manifest("set_expansion_policy", ExpansionArgs, tool_set_expansion_policy, "Ferramenta removida.", deprecated=True),
    _legacy_manifest("generate_faq_from_branch", FaqArgs, tool_generate_faq_from_branch, "Gera FAQ draft a partir do galho ancestral."),
    _legacy_manifest("validate_hierarchy", SlugArgs, tool_validate_hierarchy, "Valida a hierarquia do draft.", effect=ToolEffect.READ),
    _legacy_manifest("attach_session_asset", AssetArgs, tool_attach_session_asset, "Anexa asset persistido da sessao ao draft."),
    _legacy_manifest("validate_plan", EmptyArgs, tool_validate_plan, "Valida o plano atual sem escrever.", effect=ToolEffect.READ),
    _legacy_manifest("find_existing_persona_nodes", FindNodesArgs, tool_find_existing_persona_nodes, "Localiza nodes existentes da persona.", effect=ToolEffect.READ),
])
SOFIA_TOOLS_REGISTRY = {
    manifest.name: manifest.handler for manifest in SOFIA_TOOL_MANIFESTS.all()
}
SOFIA_TOOLS_SCHEMA = SOFIA_TOOL_MANIFESTS.model_schemas()


def dispatch_tool_call(session: dict, name: str, arguments: dict | None) -> dict:
    """Despacha uma tool call vinda do LLM para o handler apropriado."""
    manifest = SOFIA_TOOL_MANIFESTS.get(name)
    if not manifest or manifest.deprecated:
        return _err(f"tool nao registrada: {name}")
    try:
        return manifest.invoke(session, arguments if isinstance(arguments, dict) else {})
    except Exception as exc:
        return _err(f"erro ao executar {name}: {exc}", exception=type(exc).__name__)

