"""Publish only Aurora's canonical graph fixture; never creates accounts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from schemas.graph_json_v2 import Edge, EdgeLifecycle, GraphJson, PublicationGrant
from services import (
    graph_compiler_v3,
    graph_conversation_contract,
    graph_document_publisher,
    graph_json_v21_adapter,
    graph_json_v2_store,
)


FIXTURE = ROOT / "scripts" / "fixtures" / "aurora_graph_v2.json"


def build_graph() -> GraphJson:
    """Build Aurora's v2.1 graph and preserve the 44-node agent dataset.

    The historical fixture published every approved factual node into the
    persona-wide RAG.  During the v2.1 cutover we make that authorization
    explicit against Aurora's isolated Embedded action so the dialogue loses
    neither rules, tone, products nor FAQs.
    """
    legacy = GraphJson.model_validate(json.loads(FIXTURE.read_text(encoding="utf-8")))
    graph = graph_json_v21_adapter.upgrade_to_v21(legacy)
    graph = graph_conversation_contract.materialize_qualification_questions(graph)
    persona_node = next(node for node in graph.nodes if node.node_type == "persona")
    conversation_policy = dict((persona_node.data or {}).get("conversation_policy") or {})
    conversation_policy["question_repetition"] = {
        **dict(conversation_policy.get("question_repetition") or {}),
        "max_attempts": 1,
    }
    persona_node.data = {**dict(persona_node.data or {}), "conversation_policy": conversation_policy}
    appointment_policy = (persona_node.data or {}).get("appointment_policy") or {}
    question_ids = appointment_policy.get("field_question_node_ids") or {}
    conditional_fields = appointment_policy.get("conditional_fields") or {}
    node_by_id = {node.id: node for node in graph.nodes}
    primary_parent_id = {
        edge.target: edge.source
        for edge in graph.edges
        if edge.relation_type == "contains" and edge.lifecycle.status == "active"
    }

    def value_schema(field_key: str) -> dict:
        # Schema belongs to the published Aurora graph; the runtime never knows
        # these field names or their vertical in advance.
        if field_key == "vehicle_year":
            return {"anyOf": [
                {"type": "string", "pattern": "^[0-9]{4}$"},
                {"type": "integer", "minimum": 1886, "maximum": 2200},
            ]}
        if field_key == "can_visit_in_person":
            return {"type": ["string", "boolean"]}
        return {"type": "string", "minLength": 1}

    service_values = [
        {
            "value": node.slug,
            "aliases": list(dict.fromkeys([
                str(node.title or node.label or node.slug),
                *[str(value) for value in ((node.data or {}).get("aliases") or [])],
            ])),
        }
        for node in graph.nodes if node.node_type in {"product", "service"}
    ]

    def field_validation(field_key: str) -> dict:
        invalid_response = "Não consegui entender essa informação com segurança."
        if field_key == "servico":
            return {
                "mode": "enum", "values": service_values,
                "invalid_response": "Não entendi exatamente qual serviço você quis dizer.",
            }
        if field_key == "objective":
            return {
                "mode": "enum",
                "values": [
                    {
                        "value": "vender_em_breve",
                        "aliases": ["vender o carro", "pretendo vender", "vender em breve"],
                    },
                    {
                        "value": "continuar_cuidar_proteger",
                        "aliases": [
                            "continuar com o veículo e cuidar bem dele",
                            "continuar com o carro",
                            "cuidado e proteção",
                        ],
                    },
                ],
                "invalid_response": "Não consegui identificar se o objetivo é vender ou continuar cuidando do veículo.",
            }
        if field_key == "can_visit_in_person":
            return {
                "mode": "enum",
                "values": [
                    {"value": True, "aliases": ["sim", "consigo levar", "posso levar"]},
                    {"value": False, "aliases": ["não", "prefiro seguir por aqui", "não consigo levar"]},
                ],
                "invalid_response": invalid_response,
            }
        if field_key == "vehicle_year":
            return {"mode": "schema", "invalid_response": invalid_response}
        semantic = {
            "nome_cliente": {
                "semantic_type": "human_full_name",
                "description": "Nome e sobrenome completos informados pelo cliente.",
                "examples": ["Beatriz Souza", "José da Silva", "Ana Paula Lima"],
                # The model reads a name far better than any string
                # comparison can. Above this confidence its reading stands on
                # its own (evidence and shape are still proved by the
                # backend), and confirming becomes the last resort instead of
                # the default -- which is what deadlocked the live flow on
                # 2026-08-19, when "allan rodrigues" could not match the
                # model's own "Allan Rodrigues".
                "model_confidence_min": 0.90,
                "min_tokens": 2,
                "max_tokens": 6,
                "confirmation_policy": "last_resort",
            },
            "modelo_veiculo": {
                "description": "Modelo ou identificação comercial do veículo.",
                "examples": ["Onix", "Civic", "Corolla Cross"],
            },
            "condicao": {
                "description": "Relato literal do estado atual ou incômodo percebido no veículo.",
                "examples": ["riscos na porta", "bancos manchados"],
            },
            "vehicle_color": {
                "description": "Cor informada para o veículo.",
                "examples": ["prata", "preto", "azul"],
            },
            "reclamacao_relato": {
                "description": "Relato literal do cliente sobre a ocorrência reclamada.",
                "examples": ["o problema voltou depois do atendimento"],
            },
        }.get(field_key) or {
            "description": "Informação comercial livre declarada por este node.",
            "examples": ["informação fornecida pelo cliente"],
        }
        return {"mode": "semantic", **semantic, "invalid_response": invalid_response}

    for node in graph.nodes:
        data = dict(node.data or {})
        capabilities = dict(data.get("capabilities") or {})
        parent_node = node_by_id.get(primary_parent_id.get(node.id))
        if node.node_type == "product":
            capabilities["branch_anchor"] = True
            booking = data.get("booking") if isinstance(data.get("booking"), dict) else {}
            field_guidance = (
                booking.get("field_guidance")
                if isinstance(booking.get("field_guidance"), dict) else {}
            )
            required = [str(field) for field in booking.get("required_fields") or [] if field]
            for field_key, branch_slugs in conditional_fields.items():
                if node.slug in (branch_slugs or []) and field_key not in required:
                    required.append(str(field_key))
            # Fields authored in the fixture (optional ones such as the remote-track
            # questions) survive; a generated field always wins on key conflict so the
            # required set stays derived from the published booking contract.
            authored_fields = {
                str(field.get("key")): field
                for field in ((data.get("qualification") or {}).get("fields") or [])
                if isinstance(field, dict) and field.get("key")
            }
            data["qualification"] = {"fields": [{
                "key": field_key,
                # Confirmed live 2026-08-08: every product/service node lists
                # the same qualification fields (nome_cliente, objective,
                # can_visit_in_person, modelo_veiculo, vehicle_year,
                # condicao, vehicle_color) with a *different* owner_node_id
                # per branch, even though they mean the same thing and share
                # the same question node regardless of which service the
                # customer is asking about. graph_proof_checker_v3 requires
                # fact.owner_node_id == field.owner_node_id before counting a
                # field resolved (commit 6538461), so any branch switch --
                # including one caused only by the classifier's own
                # imprecision, not a real change of mind -- reopened every
                # one of these as unanswered. "servico" is the one field
                # that legitimately differs per branch (it's who the branch
                # even is) and is auto-derived from active_branch_node_id
                # server-side regardless of what's declared here, so it
                # keeps its own branch as owner; every other field shares
                # the persona node as owner across all branches.
                "owner_node_id": node.id if field_key == "servico" else persona_node.id,
                "scope": "branch" if field_key == "servico" else "persona",
                "question_node_id": question_ids.get(field_key),
                "required": True,
                "accepted_statuses": (
                    ["known", "unknown"] if field_key == "vehicle_color" else ["known"]
                ),
                "value_schema": value_schema(field_key),
                "validation": field_validation(field_key),
                "normalization": (
                    "Retorne quatro dígitos." if field_key == "vehicle_year" else None
                ),
                "depends_on": [],
                "condition": None,
                "priority": 1.0 if field_key in {"servico", "modelo_veiculo"} else 0.7,
                "overwrite_policy": "explicit_correction",
                "context_guidance": str(field_guidance.get(field_key) or ""),
                # Confirmed live 2026-08-18: only nome_cliente (the literal
                # appointment_policy.identity_field) survived into a new
                # journey/appointment cycle -- every other persona-scoped
                # fact (vehicle model/color/year/condition) was silently
                # dropped even though scope="persona" already marks them as
                # customer-owned, not tied to one specific pedido. A
                # returning customer had to restate the whole vehicle from
                # scratch, only the service should ever need reconfirming.
                # docs/architecture/SDR_JOURNEY_STATE_MACHINE.md's own
                # documented default is "persona.data.qualification.fields
                # -> carry_over: true"; this now matches it, generalizing
                # to any future persona-scoped field automatically instead
                # of requiring a one-off code change. objective and
                # can_visit_in_person are the deliberate exceptions -- they
                # describe intent for THIS visit (why the customer is here,
                # whether they can come in person), not stable customer/
                # vehicle identity, so they still get reconfirmed each cycle.
                "carry_over": (
                    field_key != "servico"
                    and field_key not in {"objective", "can_visit_in_person"}
                ),
            } for field_key in required]}
            data["qualification"]["fields"].extend(
                field for key, field in authored_fields.items() if key not in required
            )
            claims = list(data.get("claims") or [])
            if data.get("price"):
                claims.append({
                    "claim_type": "price", "policy": {
                        "mode": "informational",
                        "qualifier": data.get("price_qualifier") or "published",
                    }, "evidence_node_ids": [node.id],
                })
            if booking.get("duration_minutes"):
                claims.append({
                    "claim_type": "duration", "policy": {"mode": "informational"},
                    "evidence_node_ids": [node.id],
                })
            data["claims"] = claims
            data["completion"] = {"required_fields": required}
            # The fixture authors which rule answers a price or scheduling question;
            # only the completion target is imposed here.
            data["handoff"] = {
                "on_completion": "aurora-rule-operation",
                **(data.get("handoff") or {}),
            }
        elif node.node_type == "service":
            # A "service" branch anchor (BRANCH_TYPES in
            # graph_conversation_contract.py already reserves this type)
            # covers non-sales intents -- talking to a human, filing a
            # complaint -- that don't need the product loop's vehicle
            # qualification. The fixture authors qualification.fields
            # directly (owner_node_id, value_schema and all); Python only
            # backfills question_node_id, since that id is only known once
            # materialize_qualification_questions() has run.
            capabilities["branch_anchor"] = True
            data["qualification"] = {"fields": [
                {
                    **field,
                    "question_node_id": question_ids.get(str(field.get("key"))),
                    "validation": field.get("validation")
                    or field_validation(str(field.get("key"))),
                }
                for field in ((data.get("qualification") or {}).get("fields") or [])
                if isinstance(field, dict) and field.get("key")
            ]}
        elif node.id == "aurora-rule-operation":
            capabilities.update({"global_context": True, "handoff_rule": True})
            data["handoff_rule"] = {
                "id": "aurora-human-confirmation",
                "condition": "qualification_complete",
                "text": appointment_policy.get("texts", {}).get("complemento_confirmacao"),
            }
            # Claims authored in the fixture (payment policy) are preserved.
            claims = list(data.get("claims") or [])
            claims.extend([
                {"claim_type": "availability", "policy": {"mode": "informational"},
                 "evidence_node_ids": [node.id],
                 "intent_aliases": ["disponibilidade", "vaga", "tem horário"]},
                {"claim_type": "schedule", "policy": {"mode": "human_confirmation_required"},
                 "evidence_node_ids": [node.id],
                 "intent_aliases": ["agenda", "agendamento", "confirmar horário"]},
            ])
            data["claims"] = claims
        elif node.node_type == "rule" and (
            data.get("handoff_rule") or capabilities.get("handoff_rule")
        ):
            handoff_rule = dict(data.get("handoff_rule") or {})
            # A rule authored with scope "branch" only needs to reach its own
            # branch's closure -- normal parent/child reachability already
            # gets it there, since these rules are authored as children of
            # their own service/product node. Forcing global_context on them
            # would leak an always-authorized handoff (condition: null) into
            # every unrelated branch. Every other rule the fixture publishes
            # as a handoff rule must still reach every branch closure,
            # otherwise graph_compiler_v3 rejects the references to it.
            branch_scoped = handoff_rule.pop("scope", None) == "branch"
            capabilities["handoff_rule"] = True
            if not branch_scoped:
                capabilities["global_context"] = True
            # The fixture names which published text answers this rule; the copy itself
            # stays in appointment_policy.texts so it is authored in exactly one place.
            text_key = handoff_rule.pop("text_key", None) or "atendimento_humano"
            handoff_rule.setdefault(
                "text", appointment_policy.get("texts", {}).get(text_key)
            )
            data["handoff_rule"] = handoff_rule
        elif (
            node.node_type == "faq"
            and parent_node is not None
            and parent_node.node_type == "product_group"
            and data.get("role") != "qualification_question"
        ):
            # FAQs authored directly under the catalog/service group describe
            # the portfolio as a whole, not one product branch.
            capabilities["global_context"] = True
        elif node.node_type in {"tone"}:
            capabilities["global_context"] = True
        if capabilities:
            data["capabilities"] = capabilities
        node.data = data
    embedded = next(node for node in graph.nodes if node.node_type == "embedded")
    if embedded.action is None:
        raise RuntimeError("Aurora Embedded action is missing")
    embedded.slug = "sdr-aurora"
    embedded.title = "Golden Dataset SDR Aurora"
    embedded.label = embedded.title
    embedded.action.destination_id = "dataset:sdr-aurora"
    embedded.action.consumer.kind = "agent"
    embedded.action.consumer.ref = "sdr:aurora"

    active_sources = {
        edge.source
        for edge in graph.edges
        if edge.target == embedded.id
        and edge.relation_type == "publishes_to"
        and edge.lifecycle.status == "active"
    }
    for node in graph.nodes:
        if (
            node.node_class != "knowledge"
            or node.node_type == "persona"
            or node.lifecycle.status != "approved"
            or node.id in active_sources
        ):
            continue
        graph.edges.append(
            Edge(
                id=f"edge:publish:{node.id}:sdr-aurora",
                source=node.id,
                target=embedded.id,
                relation_type="publishes_to",
                relation_class="publication",
                primary_tree=False,
                lifecycle=EdgeLifecycle(status="active"),
                grant=PublicationGrant(
                    mode="manual",
                    actor="production-release",
                    reason="Preserve Aurora's approved agent dataset during Graph v2.1 cutover",
                ),
                metadata={"migration": "aurora-v20-to-v21"},
            )
        )
    return graph


def publish(*, expected_version: int | None = None) -> dict:
    graph = build_graph()
    current = graph_json_v2_store.load_current("aurora", graph.brand_slug)
    base_version = int(expected_version) if expected_version is not None else (int(current[0]) if current else 0)
    checksum = graph_json_v2_store.checksum_graph(graph)
    return graph_document_publisher.commit(
        graph=graph,
        persona_slug="aurora",
        brand_slug=graph.brand_slug,
        source="aurora_markdown_release",
        reason="Aurora Graph JSON v2.1 canonical rollout",
        published_by="production-release",
        expected_version=base_version,
        idempotency_key=f"aurora-graph-v21:{checksum}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-version", type=int)
    parser.add_argument("--skip-v3", action="store_true")
    args = parser.parse_args()
    result = publish(expected_version=args.expected_version)
    v3_result = None
    if result.get("ok") and not args.skip_v3:
        v3_result = graph_compiler_v3.compile_persona_publication("aurora", activate=True)
    print(json.dumps({
        "ok": result.get("ok"),
        "version": result.get("version"),
        "checksum": result.get("checksum"),
        "idempotent_replay": result.get("idempotent_replay"),
        "graph_agent_runtime_v3": ({
            "publication_id": (v3_result or {}).get("publication", {}).get("id"),
            "version": (v3_result or {}).get("publication", {}).get("version"),
            "checksum": (v3_result or {}).get("publication", {}).get("checksum"),
        } if v3_result else None),
    }))
