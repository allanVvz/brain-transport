from __future__ import annotations

import os
import re
import time
from typing import Any, Optional

from services import supabase_client
from services import knowledge_taxonomy
from services import graph_validation

DEFAULT_CONFIDENCE_THRESHOLD = 0.65
PERSONA_CONFIDENT_SCORE = 0.85
DEFAULT_MEMORY_TTL_SECONDS = 1800
DEFAULT_MEMORY_MAX_TURNS = 5
_SESSION_MEMORY: dict[str, dict[str, Any]] = {}


# D2: the Orquestrar operating prompt lives in agents/sofia_orquestrar.md and is
# loaded at runtime, with a concise inline fallback when the file is absent.
_INLINE_OPERATING_PROMPT = (
    "Sofia em modo Graph: edição cirúrgica do graph_json canônico. "
    "Resolva persona/node/operação, valide a cadeia canônica "
    "(persona->brand->briefing->campaign->audience->product_group->product->copy->faq->embedded), "
    "e emita um Patch (PatchOperation). FAQ no menor nó; FAQ->Embedded só após aprovação; "
    "persona/embedded/gallery são protegidos."
)


def _load_agent_prompt(name: str, fallback: str) -> str:
    try:
        from pathlib import Path as _Path

        path = _Path(__file__).resolve().parents[2] / "agents" / name
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
    except Exception:
        pass
    return fallback


OPERATING_PROMPT = _load_agent_prompt("sofia_orquestrar.md", _INLINE_OPERATING_PROMPT)


def _threshold() -> float:
    raw = os.getenv("SOFIA_GRAPH_COMMAND_MIN_SCORE", str(DEFAULT_CONFIDENCE_THRESHOLD)).strip()
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_CONFIDENCE_THRESHOLD
    return max(0.0, min(1.0, value))


def _memory_ttl_seconds() -> int:
    raw = os.getenv("SOFIA_SHORT_TERM_MEMORY_TTL_SECONDS", str(DEFAULT_MEMORY_TTL_SECONDS)).strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MEMORY_TTL_SECONDS
    return max(60, min(86400, value))


def _memory_max_turns() -> int:
    raw = os.getenv("SOFIA_SHORT_TERM_MEMORY_MAX_TURNS", str(DEFAULT_MEMORY_MAX_TURNS)).strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MEMORY_MAX_TURNS
    return max(1, min(20, value))


def _clean_expired_sessions(now: Optional[float] = None) -> None:
    now_ts = now or time.time()
    ttl = _memory_ttl_seconds()
    expired = [
        session_id
        for session_id, state in _SESSION_MEMORY.items()
        if (now_ts - float(state.get("updated_at") or 0.0)) > ttl
    ]
    for session_id in expired:
        _SESSION_MEMORY.pop(session_id, None)


def _normalise_session_id(value: Optional[str]) -> str:
    return str(value or "").strip().lower()


def _supabase_plan_store_enabled() -> bool:
    return bool((os.getenv("SUPABASE_URL") or "").strip() and (os.getenv("SUPABASE_SERVICE_KEY") or "").strip())


def _load_supabase_state(session_id: str) -> Optional[dict[str, Any]]:
    if not _supabase_plan_store_enabled():
        return None
    try:
        row = supabase_client.get_sofia_plan_session(session_id)
    except Exception:
        return None
    if not isinstance(row, dict):
        return None
    return {
        "updated_at": time.time(),
        "active_persona_slug": str(row.get("active_persona_slug") or row.get("persona_slug") or "").strip().lower(),
        "last_referenced_node": dict(row.get("last_referenced_node") or {}),
        "recent_turns": list(row.get("recent_turns") or []),
        "plan_json": dict(row.get("plan_json") or {}),
    }


def _persist_state(session_id: str, state: dict[str, Any], fallback_persona_slug: str) -> None:
    _SESSION_MEMORY[session_id] = dict(state)
    if not _supabase_plan_store_enabled():
        return
    plan_json = state.get("plan_json")
    if not isinstance(plan_json, dict):
        return
    persona_slug = str(
        plan_json.get("persona_slug")
        or state.get("active_persona_slug")
        or fallback_persona_slug
        or ""
    ).strip().lower()
    try:
        supabase_client.upsert_sofia_plan_session(
            session_id=session_id,
            persona_slug=persona_slug,
            plan_json=plan_json,
            active_persona_slug=str(state.get("active_persona_slug") or persona_slug).strip().lower(),
            last_referenced_node=state.get("last_referenced_node") if isinstance(state.get("last_referenced_node"), dict) else {},
            recent_turns=list(state.get("recent_turns") or []),
        )
    except Exception:
        return


def get_session_state(session_id: Optional[str]) -> dict[str, Any]:
    key = _normalise_session_id(session_id)
    if not key:
        return {}
    _clean_expired_sessions()
    state = _SESSION_MEMORY.get(key)
    if not state:
        state = _load_supabase_state(key)
        if state:
            _SESSION_MEMORY[key] = dict(state)
    return dict(state) if state else {}


def _default_plan_json(session_id: str, persona_slug: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "persona_slug": persona_slug,
        "active_context": {
            "brand_slug": None,
            "selected_node_id": None,
            "last_referenced_node": None,
        },
        "plan": {
            "briefing": [],
            "campaign": [],
            "audience": [],
            "product_group": [],
            "product": [],
            "offer": [],
            "copy": [],
            "faq": [],
            "rule": [],
            "asset": [],
            "gallery": [],
        },
        "graph_patch_queue": [],
        "blocking_issues": [],
        "suggestions": [],
        "pending_issues": [],
        "validation": {
            "is_valid": True,
            "suggestions": [],
            "pending": [],
            "blocking": [],
        },
    }


def _orchestrator_plan_to_entries(plan_json: dict[str, Any]) -> dict[str, Any]:
    """Flatten the orchestrator section shape (`plan.<section>[]`) into the
    normalized-plan `{persona_slug, entries, links}` shape so it can be converted
    to the single canonical GraphJson."""
    plan = plan_json.get("plan") if isinstance(plan_json.get("plan"), dict) else {}
    entries: list[dict[str, Any]] = []
    for section, items in plan.items():
        for item in items or []:
            if not isinstance(item, dict):
                continue
            slug = str(item.get("slug") or item.get("title") or section).strip()
            entries.append(
                {
                    "content_type": section,
                    "title": item.get("title") or slug,
                    "slug": slug,
                    "status": item.get("status") or "pending_validation",
                    "content": item.get("content") or item.get("summary") or "",
                    "tags": item.get("tags") or [section],
                    "metadata": {
                        "parent_slug": item.get("parent_slug") or "self",
                        **(item.get("metadata") if isinstance(item.get("metadata"), dict) else {}),
                    },
                }
            )
    return {
        "persona_slug": str(plan_json.get("persona_slug") or "").strip().lower(),
        "entries": entries,
        "links": [],
    }


def plan_json_to_graph_json(plan_json: dict[str, Any]) -> dict[str, Any]:
    """Canonical view of a session plan_json as a GraphJson dict (D1).

    The section shape remains the deterministic working scratch; this is the ONE
    canonical document the front-end, importer and validator consume. Best-effort:
    returns an empty graph dict if conversion fails (never raises into the chat)."""
    try:
        from services import graph_json_importer

        graph = graph_json_importer.normalized_plan_to_graph_json(
            _orchestrator_plan_to_entries(plan_json),
            {"persona_slug": plan_json.get("persona_slug")},
        )
        return graph.model_dump()
    except Exception:
        return {}


def graph_json_to_plan_json(graph_json: dict[str, Any], *, session_id: Optional[str] = None) -> dict[str, Any]:
    """Inverse bridge: rebuild the section-shaped plan from a canonical GraphJson.

    Used when the front-end edits the graph_json (Orquestrar write-through) and the
    session needs its sections refreshed to match the published document."""
    graph = graph_json if isinstance(graph_json, dict) else {}
    nodes = graph.get("nodes") or []
    persona_slug = str(graph.get("persona_slug") or "").strip().lower()
    by_id: dict[str, dict] = {n.get("id"): n for n in nodes if isinstance(n, dict)}
    plan_json = _default_plan_json(session_id or "ephemeral", persona_slug)
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = str(node.get("node_type") or "").strip().lower()
        if node_type in ("persona", "embedded"):
            continue
        section = plan_json["plan"].get(node_type)
        if section is None:
            continue
        parent = by_id.get(node.get("parent_id"))
        parent_slug = str((parent or {}).get("slug") or "self")
        section.append(
            {
                "slug": node.get("slug"),
                "title": node.get("label") or node.get("slug"),
                "parent_slug": parent_slug,
                "status": str((node.get("data") or {}).get("status") or "pending_validation"),
            }
        )
    plan_json["graph_json"] = graph
    return plan_json


def _deep_merge(base: Any, patch: Any) -> Any:
    if isinstance(base, dict) and isinstance(patch, dict):
        merged = dict(base)
        for key, value in patch.items():
            merged[key] = _deep_merge(merged.get(key), value)
        return merged
    if isinstance(base, list) and isinstance(patch, list):
        return patch
    return patch


def _validate_plan_json(plan_json: dict[str, Any]) -> dict[str, Any]:
    plan = plan_json.get("plan") if isinstance(plan_json.get("plan"), dict) else {}
    suggestions: list[dict[str, str]] = []
    pending: list[dict[str, str]] = []
    blocking: list[dict[str, str]] = []

    if not plan.get("faq"):
        suggestions.append({"code": "FAQ_RECOMMENDED", "message": "Adicionar FAQ melhora cobertura de objeções, mas não bloqueia criação."})
    if not plan.get("rule"):
        suggestions.append({"code": "RULE_RECOMMENDED", "message": "Adicionar regra comercial é recomendado, mas não bloqueia criação."})

    # Map every node slug -> its node type (the section name) so we can validate
    # parent hierarchy by TYPE using the SHARED graph_validation rules (same as
    # the Create path). product_group -> product is valid; product_group under
    # product is blocked here, not via a divergent marker list.
    slug_to_type: dict[str, str] = {}
    for section in plan.keys():
        for item in plan.get(section) or []:
            if isinstance(item, dict) and item.get("slug"):
                slug_to_type[str(item["slug"]).strip().lower()] = section

    for section in ("campaign", "audience", "product_group", "product", "offer", "copy", "faq"):
        for idx, item in enumerate(plan.get(section) or []):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                pending.append({"code": "MISSING_TITLE", "message": f"{section}[{idx}] sem title."})
            if section in {"campaign", "audience", "product_group", "product", "offer", "copy", "faq"}:
                parent = str(item.get("parent_slug") or "").strip()
                if not parent:
                    pending.append({"code": "MISSING_PARENT", "message": f"{section}[{idx}] sem parent_slug."})
                else:
                    parent_type = slug_to_type.get(parent.lower())
                    if parent_type:
                        viol = graph_validation.parent_violation(section, parent_type)
                        if viol:
                            blocking.append({"code": "INVALID_PARENT", "message": f"{section}[{idx}] {viol}"})

    # Item #1 — every non-top-level node must have a complete path to the persona.
    # Walk parent_slug up: rooted when it reaches "self"/persona, a top-level type
    # under persona, or an existing graph node (parent not declared in this plan).
    parent_by_slug: dict[str, str] = {}
    for section in plan.keys():
        for item in plan.get(section) or []:
            if isinstance(item, dict) and item.get("slug"):
                parent_by_slug[str(item["slug"]).strip().lower()] = str(item.get("parent_slug") or "").strip().lower()
    _persona_parent = {"self", "persona"}
    for section in ("campaign", "audience", "product_group", "product", "offer", "copy", "faq"):
        for idx, item in enumerate(plan.get(section) or []):
            if not isinstance(item, dict) or not item.get("slug"):
                continue
            cur = str(item["slug"]).strip().lower()
            seen: set[str] = set()
            rooted = False
            for _ in range(64):
                parent = parent_by_slug.get(cur)
                if parent in _persona_parent:
                    rooted = True
                    break
                if not parent:
                    rooted = slug_to_type.get(cur) in graph_validation.TOP_LEVEL_TYPES
                    break
                if parent in seen:
                    break  # cycle — reported via markers
                if parent not in parent_by_slug:
                    rooted = True  # parent is an existing graph node
                    break
                seen.add(parent)
                cur = parent
            if not rooted:
                blocking.append({"code": "NO_PATH_TO_PERSONA", "message": f"{section}[{idx}] sem caminho completo ate a persona."})

    blocking_markers = {
        "cycle",
        "orphan",
        "edge_inverted",
        "product_above_product_group",
        "embed_without_approved_faq",
        "persistence_failure",
        "critical_duplication",
    }
    for edge in plan_json.get("graph_patch_queue") or []:
        if not isinstance(edge, dict):
            continue
        marker = str(edge.get("marker") or "").strip().lower()
        if marker in blocking_markers:
            blocking.append({"code": marker.upper(), "message": str(edge.get("message") or marker)})

    return {
        "is_valid": len(blocking) == 0,
        "suggestions": suggestions,
        "pending": pending,
        "blocking": blocking,
    }


def _validate_patch_canonical(graph_patch: Optional[dict[str, list[dict]]]) -> list[str]:
    """Validate a graph_patch's primary edges against the SHARED canonical rules.

    Conservative: only edges whose BOTH endpoints are new nodes in THIS patch
    (node types known without a DB lookup) are checked, so existing-node ops are
    never wrongly flagged. Empty list = canonical. Same rules as the Create path.
    """
    if not isinstance(graph_patch, dict):
        return []
    type_by_ref: dict[str, str] = {}
    for node in graph_patch.get("nodes_upsert") or []:
        slug = str(node.get("slug") or "").strip().lower()
        if slug:
            type_by_ref[f"slug:{slug}"] = str(node.get("node_type") or "")
    violations: list[str] = []
    for edge in graph_patch.get("edges_upsert") or []:
        if (edge.get("metadata") or {}).get("primary_tree") is not True:
            continue
        src_t = type_by_ref.get(str(edge.get("source_ref") or "").strip())
        tgt_t = type_by_ref.get(str(edge.get("target_ref") or "").strip())
        if src_t and tgt_t:
            viol = graph_validation.edge_violation(src_t, tgt_t, str(edge.get("relation_type") or ""))
            if viol:
                violations.append(viol)
    return violations


_EMPTY_GRAPH_PLAN_SECTIONS = (
    "briefing", "campaign", "audience", "product_group", "product",
    "offer", "copy", "faq", "rule", "asset", "gallery",
)


def _empty_graph_plan() -> dict[str, list[dict]]:
    return {section: [] for section in _EMPTY_GRAPH_PLAN_SECTIONS}


def _ga_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower()).strip("-") or "node"


def _parse_group_names(command: str) -> list[str]:
    """Extract explicit product-group names, e.g. "grupos: Radar, Juliet e HSTN"
    or "(Radar, Juliet, HSTN)". Never invents names (anti-hallucination)."""
    raw = None
    m = re.search(r"grupos?\s*[:\-]?\s*([A-Za-z0-9 ,;]+?)(?:\s+com\b|\s+por\b|[\.\n]|$)", command, re.I)
    if m:
        raw = m.group(1)
    if not raw:
        m2 = re.search(r"\(([^)]+)\)", command)
        if m2:
            raw = m2.group(1)
    if not raw:
        return []
    parts = re.split(r",|;|\be\b", raw, flags=re.I)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 1 and not p.strip().isdigit()]


def _repair_graph_plan(plan: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], bool]:
    """Real repair: reparent nodes whose parent type violates the SHARED rules to
    the first available canonical parent. Returns (plan, changed)."""
    slug_to_type: dict[str, str] = {}
    by_type: dict[str, list[str]] = {}
    for section, items in plan.items():
        for item in items:
            slug = str(item.get("slug") or "").strip().lower()
            if slug:
                slug_to_type[slug] = section
                by_type.setdefault(section, []).append(slug)
    changed = False
    for section in ("product_group", "product", "offer", "copy", "faq"):
        for item in plan.get(section, []):
            parent = str(item.get("parent_slug") or "").strip().lower()
            if parent in ("", "self", "persona"):
                continue
            ptype = slug_to_type.get(parent)
            if ptype and graph_validation.parent_violation(section, ptype):
                for allowed in sorted(graph_validation.canonical_parents(section)):
                    if by_type.get(allowed):
                        item["parent_slug"] = by_type[allowed][0]
                        changed = True
                        break
    return plan, changed


def is_graph_agent_intent(command: str) -> bool:
    """True when a Graph command is a tree-building/decision/repair intent that
    the shared graph agent should handle (groups, offer, rule, skip decisions,
    repair). Single-node commands (audience/campaign/product/copy/connect) stay
    on the legacy deterministic flow."""
    t = (command or "").strip().lower()
    return bool(re.search(
        r"\bgrupos?\b|product[_ ]group|\boferta\b|\bregra\b|"
        r"sem\s+assets?|sem\s+oferta|sem\s+regra|resolver\s+pend|consertar|\brepair\b",
        t,
    ))


def graph_agent_summary(result: dict[str, Any]) -> str:
    """Short operator-facing summary of a run_graph_agent_command result."""
    plan = result.get("plan") or {}
    state = result.get("state") or {}
    parts: list[str] = []
    for section in ("product_group", "product", "offer", "copy", "faq", "rule"):
        n = len(plan.get(section) or [])
        if n:
            parts.append(f"{n} {section}")
    skips = [k.replace("skip_", "sem ") for k in ("skip_assets", "skip_offer", "skip_rule") if state.get(k)]
    head = "Criei: " + ", ".join(parts) if parts else "Plano atualizado"
    if result.get("repaired"):
        head = "Resolvi as pendencias. " + head
    if skips:
        head += f" (decisao: {', '.join(skips)})"
    blocking = (result.get("validation") or {}).get("blocking") or []
    if blocking:
        head += f". Pendencias bloqueantes: {len(blocking)}"
    return head


def run_graph_agent_command(
    *,
    command: str,
    persona_slug: str,
    session_id: Optional[str] = None,
    session_state: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Graph agent: builds/edits the graph from a command, sharing the Create
    path's tools/taxonomy/validation. Deterministic-first (structured groups,
    decision persistence, real repair); the LLM-tools loop is the free-form
    fallback. Returns {plan(sections), state, validation, repaired, ok}."""
    text = (command or "").strip().lower()
    state = dict(session_state or (get_session_state(session_id) if session_id else {}) or {})
    plan = _empty_graph_plan()
    existing = state.get("graph_plan")
    if isinstance(existing, dict):
        for section in plan:
            plan[section] = list(existing.get(section) or [])

    def _ensure_audience() -> str:
        if not plan["audience"]:
            plan["audience"].append({"slug": "publico-padrao", "title": "Publico Padrao", "parent_slug": "self"})
        return str(plan["audience"][0]["slug"])

    def _ensure_product() -> str:
        if not plan["product"]:
            aud = _ensure_audience()
            plan["product"].append({"slug": "produto", "title": "Produto", "parent_slug": aud})
        return str(plan["product"][0]["slug"])

    # #6/#7 — persist skip decisions in the session state.
    if re.search(r"\bsem\s+assets?\b|pular\s+assets?|skip\s+assets?", text):
        state["skip_assets"] = True
    if re.search(r"\bsem\s+oferta", text):
        state["skip_offer"] = True
    if re.search(r"\bsem\s+regra", text):
        state["skip_rule"] = True

    # #2 — materialize product_group nodes from EXPLICIT names (never invented).
    if re.search(r"\bgrupos?\b|product[_ ]group", text):
        aud = _ensure_audience()
        for name in _parse_group_names(command):
            gslug = "grupo-" + _ga_slug(name)
            if not any(g.get("slug") == gslug for g in plan["product_group"]):
                plan["product_group"].append({"slug": gslug, "title": name, "parent_slug": aud})

    # #4 — offer must have an exit (copy under it).
    if re.search(r"\boferta\b", text) and not state.get("skip_offer"):
        prod = _ensure_product()
        if not any(o.get("slug") == "oferta" for o in plan["offer"]):
            plan["offer"].append({"slug": "oferta", "title": "Oferta", "parent_slug": prod})
        if not any(c.get("parent_slug") == "oferta" for c in plan["copy"]):
            plan["copy"].append({"slug": "copy-oferta", "title": "Copy da Oferta", "parent_slug": "oferta"})

    # #5 — rule connected to scope, and FAQ anchored to a commercial leaf.
    if re.search(r"\bregra\b", text) and not state.get("skip_rule"):
        if not plan["rule"]:
            plan["rule"].append({"slug": "regra-comercial", "title": "Regra Comercial", "parent_slug": "self"})
    if re.search(r"\bfaq\b", text):
        anchor = (
            (plan["copy"][0]["slug"] if plan["copy"] else None)
            or (plan["product_group"][0]["slug"] if plan["product_group"] else None)
            or _ensure_product()
        )
        if anchor and not any(f.get("slug") == "faq-1" for f in plan["faq"]):
            plan["faq"].append({"slug": "faq-1", "title": "FAQ", "parent_slug": anchor})

    # #8 — "resolver pendencias" runs a real repair pass.
    repaired = False
    if re.search(r"resolver\s+pend|corrigir|consertar|\brepair\b", text):
        plan, repaired = _repair_graph_plan(plan)
        repaired = True

    validation = _validate_plan_json({"plan": plan})
    state["graph_plan"] = plan
    if session_id:
        persisted = get_session_state(session_id) or {}
        persisted.update(state)
        _persist_state(session_id, persisted, persona_slug)

    return {"ok": True, "plan": plan, "state": state, "validation": validation, "repaired": repaired}


def get_or_create_plan_json(
    *,
    session_id: Optional[str],
    persona_slug: str,
    selected_node_id: Optional[str] = None,
) -> dict[str, Any]:
    key = _normalise_session_id(session_id)
    if not key:
        plan_json = _default_plan_json("ephemeral", persona_slug)
        plan_json["active_context"]["selected_node_id"] = selected_node_id
        plan_json["validation"] = _validate_plan_json(plan_json)
        plan_json["suggestions"] = list(plan_json["validation"]["suggestions"])
        plan_json["pending_issues"] = list(plan_json["validation"]["pending"])
        plan_json["blocking_issues"] = list(plan_json["validation"]["blocking"])
        return plan_json
    now_ts = time.time()
    _clean_expired_sessions(now_ts)
    state = _SESSION_MEMORY.get(key)
    if not state:
        state = _load_supabase_state(key)
    state = state or {"recent_turns": []}
    plan_json = state.get("plan_json")
    if not isinstance(plan_json, dict) or not isinstance(plan_json.get("plan"), dict):
        plan_json = _default_plan_json(key, persona_slug)
    plan_json["session_id"] = key
    plan_json["persona_slug"] = str(persona_slug or plan_json.get("persona_slug") or "").strip().lower()
    active_context = plan_json.get("active_context") if isinstance(plan_json.get("active_context"), dict) else {}
    active_context["selected_node_id"] = selected_node_id or active_context.get("selected_node_id")
    active_context["last_referenced_node"] = state.get("last_referenced_node") or active_context.get("last_referenced_node")
    plan_json["active_context"] = active_context
    plan_json["validation"] = _validate_plan_json(plan_json)
    plan_json["suggestions"] = list(plan_json["validation"]["suggestions"])
    plan_json["pending_issues"] = list(plan_json["validation"]["pending"])
    plan_json["blocking_issues"] = list(plan_json["validation"]["blocking"])
    plan_json["graph_json"] = plan_json_to_graph_json(plan_json)
    state["plan_json"] = plan_json
    state["updated_at"] = now_ts
    _persist_state(key, state, persona_slug)
    return dict(plan_json)


def apply_plan_json_patch(
    *,
    session_id: Optional[str],
    persona_slug: str,
    patch: Optional[dict[str, Any]] = None,
    command: Optional[str] = None,
) -> dict[str, Any]:
    key = _normalise_session_id(session_id)
    if not key:
        raise ValueError("session_id is required")
    plan_json = get_or_create_plan_json(session_id=key, persona_slug=persona_slug)
    next_plan = _deep_merge(plan_json, patch or {})
    if not isinstance(next_plan.get("plan"), dict):
        next_plan["plan"] = dict(_default_plan_json(key, persona_slug)["plan"])

    text = (command or "").strip().lower()
    if "crie" in text or "criar" in text:
        if "produto" in text:
            title = "Produto"
            parts = re_split_words(command or "")
            if len(parts) > 2:
                title = " ".join(parts[-2:]).strip().title()
            next_plan["plan"]["product"].append({"title": title, "parent_slug": None, "status": "pending_parent"})
        elif "audience" in text or "audiencia" in text or "audiência" in text or "publico" in text or "público" in text:
            if not next_plan["plan"]["audience"]:
                title = "Audience Padrão" if ("padrao" in text or "padr" in text or "default" in text) else "Audience"
                slug = "audiencia-padrao" if ("padrao" in text or "padr" in text or "default" in text) else "audience"
                next_plan["plan"]["audience"].append({"slug": slug, "title": title, "parent_slug": None, "status": "pending_parent"})
        elif "campanha" in text or "campaign" in text:
            title = _title_from_command(command or "", markers=("chamada", "chamado", "campanha", "campaign"), default="Campaign")
            next_plan["plan"]["campaign"].append({"title": title, "parent_slug": None, "status": "pending_parent"})

    state = _SESSION_MEMORY.get(key)
    if not state:
        state = _load_supabase_state(key)
    state = state or {"recent_turns": []}
    next_plan["validation"] = _validate_plan_json(next_plan)
    next_plan["suggestions"] = list(next_plan["validation"]["suggestions"])
    next_plan["pending_issues"] = list(next_plan["validation"]["pending"])
    next_plan["blocking_issues"] = list(next_plan["validation"]["blocking"])
    next_plan["graph_json"] = plan_json_to_graph_json(next_plan)
    state["plan_json"] = next_plan
    state["updated_at"] = time.time()
    _persist_state(key, state, persona_slug)
    return dict(next_plan)


def re_split_words(text: str) -> list[str]:
    return [tok for tok in (text or "").replace(",", " ").split() if tok.strip()]


def _title_from_command(command: str, *, markers: tuple[str, ...], default: str) -> str:
    words = re_split_words(command)
    lowered = [word.lower() for word in words]
    start = -1
    for marker in markers:
        if marker in lowered:
            start = lowered.index(marker)
    if start < 0 or start >= len(words) - 1:
        return default
    title_words = words[start + 1 :]
    if title_words and title_words[0].lower() in {"de", "do", "da", "um", "uma"}:
        title_words = title_words[1:]
    title = " ".join(title_words).strip()
    return title.title() if title else default


def plan_json_partial_success_message(plan_json: dict[str, Any], command: str) -> Optional[str]:
    plan = plan_json.get("plan") if isinstance(plan_json.get("plan"), dict) else {}
    text = (command or "").strip().lower()
    if not ("crie" in text or "criar" in text):
        return None

    if "produto" in text and plan.get("product"):
        item = plan["product"][-1]
        title = str(item.get("title") or "Produto").strip()
        if not str(item.get("parent_slug") or "").strip():
            return f"Criei o produto {title} no plano. Ele ainda esta sem parent definido. Quer conectar esse produto a qual Product Group?"
        return f"Criei o produto {title} no plano."

    if ("campanha" in text or "campaign" in text) and plan.get("campaign"):
        item = plan["campaign"][-1]
        title = str(item.get("title") or "Campaign").strip()
        if not str(item.get("parent_slug") or "").strip():
            return f"Criei a campanha {title} no plano. Falta definir abaixo de qual Briefing/Brand ela deve ficar."
        return f"Criei a campanha {title} no plano."

    if ("audience" in text or "publico" in text or "pÃºblico" in text) and plan.get("audience"):
        item = plan["audience"][-1]
        title = str(item.get("title") or "Audience").strip()
        if not str(item.get("parent_slug") or "").strip():
            return f"Criei uma {title} no plano. Falta definir onde ela deve entrar. Quer conectar abaixo de qual Campaign?"
        return f"Criei uma {title} no plano."

    return None


def remember_turn(
    *,
    session_id: Optional[str],
    persona_slug: str,
    command: str,
    operation_result: dict[str, Any],
    last_referenced_node: Optional[dict[str, Any]] = None,
) -> None:
    key = _normalise_session_id(session_id)
    if not key:
        return
    _clean_expired_sessions()
    now_ts = time.time()
    state = _SESSION_MEMORY.get(key)
    if not state:
        state = _load_supabase_state(key)
    state = state or {"recent_turns": []}
    recent_turns = list(state.get("recent_turns") or [])
    recent_turns.append(
        {
            "command": str(command or "").strip(),
            "operation": str(operation_result.get("operation") or ""),
            "score": float(operation_result.get("score") or 0.0),
            "at": now_ts,
        }
    )
    max_turns = _memory_max_turns()
    if len(recent_turns) > max_turns:
        recent_turns = recent_turns[-max_turns:]
    state = {
        "updated_at": now_ts,
        "active_persona_slug": str(persona_slug or "").strip().lower(),
        "last_operation_result": dict(operation_result or {}),
        "last_referenced_node": dict(last_referenced_node or state.get("last_referenced_node") or {}),
        "recent_turns": recent_turns,
        # Keep plan_json across turns; otherwise POST /sofia/graph-command clears
        # in-session planning state.
        "plan_json": dict(state.get("plan_json") or {}),
    }
    _persist_state(key, state, persona_slug)


def resolve_persona(command_text: str, fallback_persona_slug: str) -> dict[str, Any]:
    text = (command_text or "").strip().lower()
    if "allanvvz" in text:
        return {"slug": "allanvvz", "score": 0.99, "source": "heuristic"}
    if "vz lupas" in text or "vzlupas" in text:
        return {"slug": "allanvvz", "score": 0.9, "source": "heuristic"}
    return {"slug": fallback_persona_slug, "score": 0.51, "source": "fallback"}


def resolve_operation(command_text: str) -> dict[str, Any]:
    text = (command_text or "").strip().lower()
    if "reencaixe" in text and ("vz lupas" in text or "vzlupas" in text):
        return {"operation": "reparent_brand", "score": 0.98, "source": "heuristic"}
    if ("conectar" in text or "colocar" in text) and ("brand" in text or "vz lupas" in text or "vzlupas" in text) and ("persona" in text or "allanvvz" in text):
        return {"operation": "reparent_brand", "score": 0.96, "source": "heuristic"}
    if ("conectar" in text or "colocar" in text) and ("audience" in text or "audi" in text) and ("juliet" in text or "radar" in text or "plantaris" in text or "product group" in text):
        return {"operation": "connect_product_group_to_audience", "score": 0.94, "source": "heuristic"}
    if ("criar" in text or "crie" in text or "adicionar" in text) and ("copy" in text):
        return {"operation": "create_copy_node", "score": 0.92, "source": "heuristic"}
    if ("criar" in text or "crie" in text or "adicionar" in text) and ("faq" in text):
        return {"operation": "create_faq_node", "score": 0.92, "source": "heuristic"}
    if ("criar" in text or "crie" in text or "adicionar" in text) and ("produto" in text or "product" in text):
        return {"operation": "create_product_node", "score": 0.92, "source": "heuristic"}
    return {"operation": "validate_canonical_chain", "score": 0.42, "source": "fallback"}


def resolve_node(command_text: str, *, selected_node_id: Optional[str] = None, session_state: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    selected = str(selected_node_id or "").strip().lower()
    if selected:
        selected_type = "product_group" if "grupo-" in selected or "product_group" in selected else "unknown"
        return {
            "node_ref": selected,
            "node_type": selected_type,
            "score": 0.95,
            "source": "selected_node_id",
            "candidates": [{"node_ref": selected, "score": 0.95}],
        }
    last_ref = (session_state or {}).get("last_referenced_node") if isinstance(session_state, dict) else {}
    if isinstance(last_ref, dict):
        slug = str(last_ref.get("slug") or "").strip().lower()
        if slug:
            return {
                "node_ref": f"slug:{slug}",
                "node_type": str(last_ref.get("node_type") or "unknown"),
                "score": 0.88,
                "source": "session_memory",
                "candidates": [{"node_ref": f"slug:{slug}", "score": 0.88}],
            }
    text = (command_text or "").lower()
    if "audience" in text or "audi" in text or "padrao" in text or "padr" in text:
        return {
            "node_ref": "slug:padrao-vz-lupas",
            "node_type": "audience",
            "score": 0.88,
            "source": "heuristic",
            "candidates": [{"node_ref": "slug:padrao-vz-lupas", "score": 0.88}],
        }
    for group in ("juliet", "radar", "plantaris"):
        if group in text:
            return {
                "node_ref": f"slug:grupo-{group}",
                "node_type": "product_group",
                "score": 0.9,
                "source": "heuristic",
                "candidates": [{"node_ref": f"slug:grupo-{group}", "score": 0.9}],
            }
    if "vz lupas" in text or "vzlupas" in text:
        return {
            "node_ref": "slug:vz-lupas",
            "node_type": "brand",
            "score": 0.9,
            "source": "heuristic",
            "candidates": [{"node_ref": "slug:vz-lupas", "score": 0.9}],
        }
    return {
        "node_ref": None,
        "node_type": None,
        "score": 0.4,
        "source": "fallback",
        "candidates": [],
    }


def _simple_graph_patch(command: str, operation: str, node_result: dict[str, Any]) -> Optional[dict[str, list[dict]]]:
    text = (command or "").strip().lower()
    if operation == "reparent_brand":
        return _reencaixe_patch()
    if operation == "connect_product_group_to_audience":
        group = next((name for name in ("juliet", "radar", "plantaris") if name in text), "juliet")
        return {
            "nodes_upsert": [],
            "nodes_delete": [],
            "edges_delete": [],
            "edges_upsert": [{
                "source_ref": "slug:padrao-vz-lupas",
                "target_ref": f"slug:grupo-{group}",
                "relation_type": "audience_has_product_group",
                "metadata": {"primary_tree": True, "active": True, "created_from": "sofia_graph_command"},
            }],
        }
    if operation == "create_product_node":
        parent_ref = str(node_result.get("node_ref") or "").strip() if node_result.get("node_type") == "product_group" else ""
        return {
            "nodes_upsert": [{
                "node_type": "product",
                "slug": "produto",
                "title": "Produto",
                "summary": "Produto criado pela Sofia.",
                "status": "pending_validation",
                "metadata": {"source_url": "pending_source"},
            }],
            "nodes_delete": [],
            "edges_delete": [],
            "edges_upsert": ([{
                "source_ref": parent_ref,
                "target_ref": "slug:produto",
                "relation_type": "product_group_has_product",
                "metadata": {"primary_tree": True, "active": True, "created_from": "sofia_graph_command"},
            }] if parent_ref else []),
        }
    if operation == "create_copy_node":
        group = next((name for name in ("juliet", "radar", "plantaris") if name in text), "")
        parent_ref = f"slug:grupo-{group}" if group else str(node_result.get("node_ref") or "")
        return {
            "nodes_upsert": [{
                "node_type": "copy",
                "slug": f"copy-{group or 'node'}",
                "title": f"Copy {group.title() if group else 'Node'}",
                "summary": "Copy criada pela Sofia.",
                "status": "pending_validation",
            }],
            "nodes_delete": [],
            "edges_delete": [],
            "edges_upsert": ([{
                "source_ref": parent_ref,
                "target_ref": f"slug:copy-{group or 'node'}",
                "relation_type": "contains",
                "metadata": {"primary_tree": True, "active": True, "created_from": "sofia_graph_command"},
            }] if parent_ref else []),
        }
    if operation == "create_faq_node":
        parent_ref = str(node_result.get("node_ref") or "")
        return {
            "nodes_upsert": [{
                "node_type": "faq",
                "slug": "faq-node",
                "title": "FAQ",
                "summary": "FAQ criada pela Sofia.",
                "status": "pending_validation",
            }],
            "nodes_delete": [],
            "edges_delete": [],
            "edges_upsert": ([{
                "source_ref": parent_ref,
                "target_ref": "slug:faq-node",
                "relation_type": "product_has_faq",
                "metadata": {"primary_tree": True, "active": True, "created_from": "sofia_graph_command"},
            }] if parent_ref else []),
        }
    return None


def _reencaixe_patch() -> dict[str, list[dict]]:
    return {
        "nodes_upsert": [
            {
                "node_type": "brand",
                "slug": "vz-lupas",
                "title": "VZ Lupas",
                "summary": "Brand reencaixada pela Sofia na cadeia canonica.",
            }
        ],
        "nodes_delete": [],
        "edges_upsert": [
            {
                "source_ref": "persona:self",
                "target_ref": "slug:vz-lupas",
                "relation_type": "persona_has_brand",
                "metadata": {"primary_tree": True, "active": True},
            }
        ],
        "edges_delete": [],
    }


def plan_graph_command(
    *,
    command: str,
    context: Any,
    persona_slug: str,
    persona_tool_result: Optional[dict[str, Any]] = None,
    operation_tool_result: Optional[dict[str, Any]] = None,
    node_tool_result: Optional[dict[str, Any]] = None,
    session_state: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    threshold = _threshold()
    persona_result = persona_tool_result or resolve_persona(command, persona_slug)
    operation_result = operation_tool_result or resolve_operation(command)
    selected_node_ids = list(getattr(context, "selected_node_ids", []) or [])
    node_result = node_tool_result or resolve_node(
        command,
        selected_node_id=(selected_node_ids[0] if selected_node_ids else None),
        session_state=session_state,
    )
    persona_score = float(persona_result.get("score") or 0.0)
    operation_score = float(operation_result.get("score") or 0.0)
    node_score = float(node_result.get("score") or 0.0)
    tool_calls = [
        {
            "name": "resolve-persona",
            "arguments": {"text": command},
            "result": persona_result,
            "score": persona_score,
        },
        {
            "name": "resolve-node",
            "arguments": {"text": command},
            "result": node_result,
            "score": node_score,
        },
        {
            "name": "resolve-operation",
            "arguments": {"text": command},
            "result": operation_result,
            "score": operation_score,
        },
        {
            "name": "validate-canonical-chain",
            "arguments": {"operation": operation_result.get("operation")},
            "result": {"canonical_chain_respected": True, "violations": []},
            "score": 1.0,
        },
    ]

    graph_patch: Optional[dict[str, list[dict]]] = None
    has_persona = False
    has_node = False
    is_structured_intent = (context.client_action or "").strip().lower() == "structured_intent"
    if is_structured_intent:
        patch = context.graph_patch or {}
        graph_patch = {
            "nodes_upsert": list(patch.get("nodes_upsert") or []),
            "nodes_delete": list(patch.get("nodes_delete") or []),
            "edges_upsert": list(patch.get("edges_upsert") or []),
            "edges_delete": list(patch.get("edges_delete") or []),
        }
    else:
        candidates = list(operation_result.get("candidates") or [])
        top = float(candidates[0].get("score")) if candidates else operation_score
        second = float(candidates[1].get("score")) if len(candidates) > 1 else -1.0
        active_persona = str(persona_result.get("slug") or persona_slug or "").strip().lower()
        has_persona = bool(active_persona and persona_score >= PERSONA_CONFIDENT_SCORE)
        has_node = bool(node_result.get("node_ref")) and node_score >= threshold
        op_ambiguous = second >= 0.0 and abs(top - second) <= 0.05
        op_all_low = top < threshold
        if (not has_persona and not has_node) or op_ambiguous or op_all_low:
            tool_calls.append(
                {
                    "name": "generate-graph-patch",
                    "arguments": {"operation": operation_result.get("operation")},
                    "result": None,
                    "score": 0.0,
                }
            )
            if not has_persona and not has_node:
                message = "Nao encontrei persona ativa nem node alvo. Informe a persona ou selecione um node no grafo."
            elif op_ambiguous:
                message = "A operacao ficou ambigua entre duas opcoes proximas. Confirme a acao exata que voce quer aplicar."
            else:
                message = "Nao consegui identificar a operacao com confianca suficiente. Descreva a acao de forma mais especifica."
            return {
                "ok": True,
                "persisted": False,
                "sofia_message": message,
                "graph_patch": None,
                "tool_calls": tool_calls,
                "threshold": threshold,
                "needs_clarification": True,
            }

    if not is_structured_intent:
        graph_patch = _simple_graph_patch(command, str(operation_result.get("operation") or ""), node_result)

    if not graph_patch:
        if operation_score < threshold:
            message = "Nao consegui identificar a operacao com confianca suficiente. Descreva a acao de forma mais especifica."
        elif not has_persona and not has_node:
            message = "Nao encontrei persona ativa nem node alvo. Informe a persona ou selecione um node no grafo."
        else:
            message = "Nao consegui montar patch para a operacao resolvida. Informe o node alvo para continuar."
        return {
            "ok": True,
            "persisted": False,
            "sofia_message": message,
            "graph_patch": None,
            "tool_calls": tool_calls,
            "threshold": threshold,
            "needs_clarification": True,
        }
    tool_calls.append(
        {
            "name": "generate-graph-patch",
            "arguments": {"operation": operation_result.get("operation")},
            "result": graph_patch,
            "score": 1.0,
        }
    )

    # Real canonical-chain validation (was a stub) — shared rules with Create.
    canonical_violations = _validate_patch_canonical(graph_patch)
    for call in tool_calls:
        if call.get("name") == "validate-canonical-chain":
            call["result"] = {
                "canonical_chain_respected": not canonical_violations,
                "violations": canonical_violations,
            }
            call["score"] = 1.0 if not canonical_violations else 0.0
    if canonical_violations:
        return {
            "ok": True,
            "persisted": False,
            "sofia_message": "Patch viola a hierarquia canonica: " + "; ".join(canonical_violations[:3]),
            "graph_patch": None,
            "tool_calls": tool_calls,
            "threshold": threshold,
            "needs_clarification": True,
        }

    return {
        "ok": True,
        "persisted": True,
        "sofia_message": "Comando aplicado com cadeia canonica validada.",
        "graph_patch": graph_patch,
        "tool_calls": tool_calls,
        "threshold": threshold,
        "needs_clarification": False,
    }
