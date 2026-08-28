"""Desfecho comercial da jornada, derivado em um lugar so.

`delivered`, `service_completed` e `cancelled` nao sao estados: os tres colapsam
em `state='closed'` e so se distinguem por `metadata.closing_event`. A UI precisa
de um valor unico e estavel por lead, entao a leitura vive aqui e nao espalhada
por rota, componente e query.

Dois eixos diferentes moram aqui, e confundi-los foi a origem de um bug:

- `journey_outcome` e do **pedido**. Vale para o pedido corrente e, quando ele
  fecha, continua descrevendo o ultimo pedido -- senao o pedido concluido some
  da tela no instante em que e concluido.
- `lead_converted` e do **lead** e e permanente. Depois do primeiro agendamento
  ou compra o lead esta convertido para sempre: cancelar o pedido nao desfaz, e
  o pedido seguinte nao volta o lead para "qualificado".
"""
from __future__ import annotations

from typing import Any, Optional

from services import supabase_client

QUALIFICADO = "qualificado"
CONVERTIDO = "convertido"
VENDIDO = "vendido"
ENTREGUE = "entregue"
CANCELADO = "cancelado"

OUTCOMES = (QUALIFICADO, CONVERTIDO, VENDIDO, ENTREGUE, CANCELADO)

SALES = "sales"
APPOINTMENT = "appointment"

# Estados em que a qualificacao ja foi concluida mas nada comercial aconteceu.
_QUALIFIED_STATES = {"qualified_confirmed", "handed_off"}
_COMPLETION_EVENTS = {"delivered", "service_completed"}


def derive(journey: dict[str, Any] | None) -> Optional[str]:
    """Desfecho de uma jornada, ou None quando ainda nao ha nenhum.

    O terminal vence: uma jornada vendida e depois cancelada le `cancelado`,
    e uma vendida e entregue le `entregue`.
    """
    if not journey:
        return None
    state = str(journey.get("state") or "").strip().lower()
    metadata = journey.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}

    if state == "closed":
        closing = str(metadata.get("closing_event") or "").strip().lower()
        if closing == "cancelled":
            return CANCELADO
        if closing in _COMPLETION_EVENTS:
            return ENTREGUE
        return None
    if state == "converted":
        return VENDIDO if metadata.get("sold") else CONVERTIDO
    if state in _QUALIFIED_STATES:
        return QUALIFICADO
    return None


def summarize(journeys: list[dict[str, Any]]) -> dict[str, Any]:
    """Resume todas as jornadas de um lead num estado unico para a tela.

    Le a jornada de maior `sequence` -- corrente ou nao. Ler so a corrente
    fazia o pedido concluido desaparecer da tela assim que fechava, e o botao
    de venda voltava a parecer disponivel enquanto o backend respondia 409
    porque nao existe jornada corrente para receber o evento.
    """
    if not journeys:
        return {
            "journey_outcome": None, "lead_converted": False,
            "journey_is_open": False, "journey_sequence": 0,
        }
    ordered = sorted(journeys, key=lambda row: int(row.get("sequence") or 0))
    latest = ordered[-1]
    return {
        "journey_outcome": derive(latest),
        # Permanente: basta uma jornada ter convertido alguma vez.
        "lead_converted": any(row.get("converted_at") for row in ordered),
        "journey_is_open": bool(latest.get("is_current")),
        "journey_sequence": int(latest.get("sequence") or 0),
    }


def summaries_for_leads(persona_id: str, lead_refs: list[int]) -> dict[int, dict[str, Any]]:
    """Estado de jornada de varios leads numa leitura so.

    A lista de conversas pinta todos os itens de uma vez -- uma consulta por
    lead transformaria a tela em N+1.
    """
    refs = sorted({int(ref) for ref in lead_refs if ref})
    if not persona_id or not refs:
        return {}
    por_lead: dict[int, list[dict[str, Any]]] = {}
    for journey in supabase_client.get_journeys_by_lead_refs(persona_id, refs):
        ref = journey.get("lead_ref")
        if ref is None:
            continue
        por_lead.setdefault(int(ref), []).append(journey)
    return {ref: summarize(rows) for ref, rows in por_lead.items()}


def outcomes_for_leads(persona_id: str, lead_refs: list[int]) -> dict[int, Optional[str]]:
    """Compatibilidade: so o desfecho, sem os demais eixos."""
    return {
        ref: resumo.get("journey_outcome")
        for ref, resumo in summaries_for_leads(persona_id, lead_refs).items()
    }


def business_models_for_personas(persona_ids: list[str]) -> dict[str, str]:
    """`business_model` de cada persona, numa leitura so.

    Ele decide o par de eventos do pedido: produto e comprado e entregue,
    servico e agendado e concluido. Sem isso a tela registraria compra e
    entrega para uma persona de agendamento.
    """
    configs = supabase_client.get_persona_configs_by_ids(list(persona_ids))
    models: dict[str, str] = {}
    for pid, config in configs.items():
        portal = config.get("portal") if isinstance(config.get("portal"), dict) else {}
        model = str(portal.get("business_model") or config.get("business_model") or SALES)
        models[pid] = APPOINTMENT if model.strip().lower() == APPOINTMENT else SALES
    return models


_VAZIO = {
    "journey_outcome": None, "lead_converted": False,
    "journey_is_open": False, "journey_sequence": 0,
}


def decorate_leads(rows: list[dict[str, Any]], persona_id: str | None = None) -> list[dict[str, Any]]:
    """Anexa o estado da jornada e o `business_model` a cada lead decorado.

    `stage` continua sendo o funil manual: o desfecho e um eixo novo e
    independente, nunca uma reescrita do estagio.
    """
    if not rows:
        return rows
    grouped: dict[str, list[int]] = {}
    for row in rows:
        pid = str(persona_id or row.get("persona_id") or "")
        ref = row.get("id")
        if not pid or ref is None:
            continue
        grouped.setdefault(pid, []).append(int(ref))

    resumos: dict[tuple[str, int], dict[str, Any]] = {}
    for pid, refs in grouped.items():
        for ref, resumo in summaries_for_leads(pid, refs).items():
            resumos[(pid, ref)] = resumo
    models = business_models_for_personas(list(grouped))

    for row in rows:
        pid = str(persona_id or row.get("persona_id") or "")
        ref = row.get("id")
        resumo = (
            resumos.get((pid, int(ref)), _VAZIO) if pid and ref is not None else _VAZIO
        )
        row.update(resumo)
        # O carimbo vence a derivacao: `converted_at` pode ser limpo pelo
        # seletor de estado, e a conversao da lead nao pode regredir com ele.
        row["lead_converted"] = lead_converted(row) or bool(resumo.get("lead_converted"))
        row["business_model"] = models.get(pid, SALES)
    return rows


def lead_converted(lead: dict[str, Any] | None) -> bool:
    """A lead ja converteu alguma vez?

    Permanente por construcao: `leads.metadata.first_converted_at` e escrito
    uma unica vez na primeira venda ou agendamento e ninguem o apaga -- nem o
    cancelamento (que estorna a compra, nao a conversao), nem o seletor de
    estado, nem o ciclo seguinte.
    """
    metadata = (lead or {}).get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return bool(metadata.get("first_converted_at"))
