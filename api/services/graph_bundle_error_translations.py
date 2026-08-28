"""Portuguese translations + a suggested next question for the raw error
codes services.graph_bundle/graph_compiler_v3 raise. Consumed by the save()
response (kb_intake_service._save_via_graph_bundle) and by the graph-sidebar
route so an operator/Sofia session never has to interpret a bare code like
`bundle_primary_parent_missing:product:foo`.

Each entry is a (prefix, translator) pair; `prefix` matches the start of the
raw error string, `translator` receives the remainder (after the prefix) and
returns {"message": str, "suggested_question": str | None}. Unrecognized
codes fall back to a generic message that still tells the operator this is
a real, blocking problem rather than silently disappearing.
"""
from __future__ import annotations

from typing import Any, Callable

_Translator = Callable[[str], dict[str, Any]]


def _primary_parent_missing(rest: str) -> dict[str, Any]:
    return {
        "message": f"O item '{rest}' nÃ£o tem um pai claro na Ã¡rvore de conhecimento.",
        "suggested_question": (
            f"Onde o item '{rest}' deveria entrar? Responda com o nome da marca, "
            "campanha, audiÃªncia ou produto que deveria ser o pai dele."
        ),
    }


def _primary_parent_ambiguous(rest: str) -> dict[str, Any]:
    return {
        "message": f"O item '{rest}' estÃ¡ sendo apontado como filho de mais de um lugar ao mesmo tempo.",
        "suggested_question": f"Qual desses Ã© o pai correto de '{rest}'? SÃ³ pode ser um.",
    }


def _no_branch_anchor(_rest: str) -> dict[str, Any]:
    return {
        "message": (
            "Esta persona ainda nÃ£o tem nenhum ramo de qualificaÃ§Ã£o "
            "(ex.: varejo/atacado, uso prÃ³prio/revenda). Toda persona de "
            "vendas precisa de pelo menos um."
        ),
        "suggested_question": (
            "Que jeitos diferentes de comprar/ser atendido existem aqui? "
            "Vou marcar o primeiro como ramo de qualificaÃ§Ã£o."
        ),
    }


def _node_not_publishable(rest: str) -> dict[str, Any]:
    # `rest` is "<node_type>:<slug>:<status>" -- the node id itself contains
    # a colon, so split from the right on the LAST colon to isolate status.
    node_id, _, status = rest.rpartition(":")
    return {
        "message": f"O item '{node_id}' ainda estÃ¡ com status '{status}' e nÃ£o pode ser publicado.",
        "suggested_question": f"Posso confirmar '{node_id}' agora, ou vocÃª ainda quer revisar antes?",
    }


def _source_pending(rest: str) -> dict[str, Any]:
    return {
        "message": f"O item '{rest}' nÃ£o tem uma fonte de informaÃ§Ã£o registrada.",
        "suggested_question": f"De onde veio a informaÃ§Ã£o de '{rest}'? (conversa, site, documento enviado...)",
    }


def _content_required(rest: str) -> dict[str, Any]:
    return {
        "message": f"O item '{rest}' estÃ¡ sem nenhum conteÃºdo (resumo, pergunta/resposta ou texto).",
        "suggested_question": f"O que '{rest}' deveria dizer?",
    }


def _persona_slug_mismatch(rest: str) -> dict[str, Any]:
    return {
        "message": "O node de persona estÃ¡ com um identificador que nÃ£o bate com a persona da sessÃ£o.",
        "suggested_question": None,
    }


def _cycle(rest: str) -> dict[str, Any]:
    return {
        "message": f"Existe um ciclo na Ã¡rvore envolvendo '{rest}' -- um item acabou virando pai de si mesmo, indiretamente.",
        "suggested_question": f"Qual desses nÃ³s em '{rest}' deveria realmente ser o pai, e qual o filho?",
    }


def _duplicate_edge(rest: str) -> dict[str, Any]:
    return {
        "message": f"A conexÃ£o '{rest}' foi proposta mais de uma vez.",
        "suggested_question": None,
    }


_TRANSLATORS: list[tuple[str, _Translator]] = [
    ("bundle_primary_parent_missing:", _primary_parent_missing),
    ("bundle_primary_parent_ambiguous:", _primary_parent_ambiguous),
    ("bundle_primary_cycle:", _cycle),
    ("publication_has_no_branch_anchor_capability", _no_branch_anchor),
    ("bundle_node_not_publishable:", _node_not_publishable),
    ("bundle_node_source_pending:", _source_pending),
    ("bundle_node_content_required:", _content_required),
    ("bundle_persona_slug_mismatch:", _persona_slug_mismatch),
    ("bundle_duplicate_node_id:", lambda rest: {
        "message": f"O identificador '{rest}' estÃ¡ sendo usado por mais de um item.",
        "suggested_question": None,
    }),
    ("bundle_duplicate_edge:", _duplicate_edge),
]


def translate_error(raw_error: str) -> dict[str, Any]:
    """Translate one raw graph_bundle/graph_compiler_v3 error code. Always
    returns a usable dict, even for an unrecognized code."""
    raw_error = str(raw_error or "").strip()
    for prefix, translator in _TRANSLATORS:
        if raw_error.startswith(prefix):
            rest = raw_error[len(prefix):].lstrip(":")
            result = translator(rest)
            result.setdefault("code", raw_error)
            return result
    return {
        "code": raw_error,
        "message": f"Erro tÃ©cnico nÃ£o catalogado: {raw_error}",
        "suggested_question": "Isso precisa de revisÃ£o manual antes de continuar.",
    }


def translate_errors(raw_errors: list[str]) -> list[dict[str, Any]]:
    return [translate_error(e) for e in (raw_errors or [])]

