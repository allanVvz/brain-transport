from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from schemas.agent_harness import CardDraft, CardPatch, ToolEffect, ToolResult, ToolRisk
from schemas.graph_json_v2 import GraphJson
from services import campaigns_service, graph_document_publisher, graph_json_v2_store, supabase_client
from services.agent_tool_registry import ToolManifest, ToolRegistry


class EmptyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FindCardsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    persona_id: str
    node_types: list[str] = Field(default_factory=list)
    query: str | None = None
    limit: int = Field(50, ge=1, le=200)


class CardDraftInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    persona_id: str
    card: CardDraft
    expected_graph_version: int = Field(ge=0)
    graph_hash: str = Field(min_length=8)
    idempotency_key: str = Field(min_length=8, max_length=200)
    reason: str = Field(min_length=3, max_length=500)


class PatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    persona_id: str
    persona_slug: str
    patch: CardPatch
    graph_json: dict[str, Any] | None = None


class EditCardDraftInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    persona_id: str
    node_id: str
    changes: dict[str, Any]
    expected_graph_version: int = Field(ge=0)
    graph_hash: str = Field(min_length=8)
    idempotency_key: str = Field(min_length=8, max_length=200)
    reason: str = Field(min_length=3, max_length=500)


class ConnectCardsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    persona_id: str
    source_node_id: str
    target_node_id: str
    relation_type: str
    expected_graph_version: int = Field(ge=0)
    graph_hash: str = Field(min_length=8)
    idempotency_key: str = Field(min_length=8, max_length=200)
    reason: str = Field(min_length=3, max_length=500)


class RevokeEdgeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    persona_id: str
    edge_id: str
    expected_graph_version: int = Field(ge=0)
    graph_hash: str = Field(min_length=8)
    idempotency_key: str = Field(min_length=8, max_length=200)
    reason: str = Field(min_length=3, max_length=500)


class AttachAssetDraftInput(EditCardDraftInput):
    asset_id: str
    asset_url: str | None = None


class GenerateFaqDraftInput(EditCardDraftInput):
    max_questions: int = Field(8, ge=1, le=20)


class PhoneContact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    phone: str
    name: str | None = None


class ParseContactsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raw_text: str = Field(min_length=1, max_length=100000)
    default_country_code: str = Field("55", pattern=r"^\d{1,3}$")


class ImportPreviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    persona_id: str
    audience_id: str | None = None
    audience_name: str | None = None
    contacts: list[PhoneContact] = Field(min_length=1, max_length=10000)
    consent_purpose: str = "ofertas_e_novidades"


class ImportCreateInput(ImportPreviewInput):
    audience_id: str
    contact_basis: Literal[
        "explicit_consent", "customer_initiated", "existing_customer",
        "imported_without_evidence", "legacy_unknown", "unknown",
    ] = "imported_without_evidence"
    contact_basis_statement: str | None = None
    idempotency_key: str = Field(min_length=8, max_length=200)
    reason: str = Field(min_length=3, max_length=500)


class ResolveAudienceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    persona_id: str
    query: str = Field(min_length=1, max_length=200)
    create_if_missing: bool = False


class CreateAudienceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    persona_id: str
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None
    idempotency_key: str = Field(min_length=8, max_length=200)
    reason: str = Field(min_length=3, max_length=500)


class ResolveConsentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    persona_id: str
    lead_ids: list[int] = Field(min_length=1, max_length=10000)
    purpose: str
    channel: str = "whatsapp"


class CampaignPayloadInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    persona_id: str
    audience_id: str
    import_batch_ids: list[str] = Field(min_length=1)
    purpose: str
    campaign_kind: Literal["consent_request", "promotional"]
    name: str | None = None
    objective: str | None = None
    provider: Literal["meta_cloud"] = "meta_cloud"
    template_name: str | None = None
    template_language: str = "pt_BR"
    message: str | None = None
    variables: dict[str, Any] = Field(default_factory=dict)
    assets: list[dict[str, Any]] = Field(default_factory=list)
    policy_overrides: dict[str, Any] = Field(default_factory=dict)


class CampaignCreateDraftInput(CampaignPayloadInput):
    name: str = Field(min_length=1, max_length=200)
    expected_revision: int = 0
    expected_preview_checksum: str = Field(min_length=16, max_length=100)
    idempotency_key: str = Field(min_length=8, max_length=200)
    reason: str = Field(min_length=3, max_length=500)


class CampaignStatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    campaign_id: str
    expected_revision: int = Field(gt=0)
    idempotency_key: str = Field(min_length=8, max_length=200)
    reason: str = Field(min_length=3, max_length=500)


class QAInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: str
    persona_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    expected_graph_version: int | None = Field(None, ge=0)
    graph_hash: str | None = None


def _result(summary: str, **extra: Any) -> dict[str, Any]:
    return {"ok": True, "summary": summary, **extra}


def _slugify(value: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return clean or "grupo"


def _normalize_phone(value: str, country: str = "55") -> str | None:
    digits = re.sub(r"\D", "", value or "")
    if not digits:
        return None
    if len(digits) in {10, 11} and country == "55":
        digits = country + digits
    if digits.startswith("00"):
        digits = digits[2:]
    return digits if 10 <= len(digits) <= 15 else None


def parse_contacts(_context: dict[str, Any], **args: Any) -> dict[str, Any]:
    raw = str(args["raw_text"])
    country = str(args.get("default_country_code") or "55")
    candidates = re.findall(r"\+?\d[\d\s().-]{6,}\d", raw)
    seen: set[str] = set()
    contacts: list[dict[str, Any]] = []
    invalid = 0
    for candidate in candidates:
        phone = _normalize_phone(candidate, country)
        if not phone:
            invalid += 1
            continue
        if phone in seen:
            continue
        seen.add(phone)
        contacts.append({"phone": phone, "name": None})
    return _result(
        f"{len(contacts)} contatos validos e deduplicados.",
        contacts=contacts,
        counts={
            "input_candidates": len(candidates),
            "valid_unique": len(contacts),
            "duplicates_removed": max(0, len(candidates) - invalid - len(contacts)),
            "invalid": invalid,
        },
    )


def find_cards(_context: dict[str, Any], **args: Any) -> dict[str, Any]:
    node_types = args.get("node_types") or [
        "persona", "brand", "campaign", "product", "audience", "faq", "copy", "asset",
    ]
    rows = supabase_client.list_knowledge_nodes_by_type(
        node_types, persona_id=args["persona_id"], limit=args.get("limit") or 50
    )
    query = str(args.get("query") or "").lower().strip()
    if query:
        rows = [
            row for row in rows
            if query in str(row.get("slug") or "").lower()
            or query in str(row.get("title") or "").lower()
        ]
    return _result(f"{len(rows)} cards encontrados.", cards=rows[: args.get("limit", 50)])


def create_card_draft(_context: dict[str, Any], **args: Any) -> dict[str, Any]:
    card = CardDraft.model_validate(args["card"])
    operation = {
        "op": "add_node",
        "value": card.model_dump(mode="json"),
    }
    patch = CardPatch(
        expected_graph_version=args["expected_graph_version"],
        graph_hash=args["graph_hash"],
        idempotency_key=args["idempotency_key"],
        reason=args["reason"],
        operations=[operation],
    )
    return _result("Draft de card criado; nenhuma escrita foi feita.", card=card, patch=patch)


def _draft_patch(args: dict[str, Any], operation: dict[str, Any], summary: str) -> dict[str, Any]:
    patch = CardPatch(
        expected_graph_version=args["expected_graph_version"],
        graph_hash=args["graph_hash"], idempotency_key=args["idempotency_key"],
        reason=args["reason"], operations=[operation],
    )
    return _result(summary, patch=patch)


def edit_card_draft(_context: dict[str, Any], **args: Any) -> dict[str, Any]:
    return _draft_patch(args, {"op": "update_node", "target_id": args["node_id"], "value": args["changes"]}, "Draft de edicao criado.")


def connect_cards_draft(_context: dict[str, Any], **args: Any) -> dict[str, Any]:
    edge_id = f"edge:{args['source_node_id']}:{args['relation_type']}:{args['target_node_id']}"
    return _draft_patch(args, {
        "op": "add_edge", "target_id": edge_id,
        "value": {
            "id": edge_id, "source": args["source_node_id"], "target": args["target_node_id"],
            "relation_type": args["relation_type"], "primary_tree": False,
            "lifecycle": {"status": "proposed"},
        },
    }, "Draft de conexao criado.")


def revoke_edge_draft(_context: dict[str, Any], **args: Any) -> dict[str, Any]:
    return _draft_patch(args, {"op": "revoke_edge", "target_id": args["edge_id"], "value": {}}, "Revogacao de edge preparada sem deletar nodes.")


def attach_asset_draft(_context: dict[str, Any], **args: Any) -> dict[str, Any]:
    return _draft_patch(args, {
        "op": "attach_asset", "target_id": args["node_id"],
        "value": {"asset_id": args["asset_id"], "asset_url": args.get("asset_url")},
    }, "Asset da sessao anexado ao draft.")


def generate_faq_draft(_context: dict[str, Any], **args: Any) -> dict[str, Any]:
    return _draft_patch(args, {
        "op": "generate_faq", "target_id": args["node_id"],
        "value": {"max_questions": args.get("max_questions") or 8, "source": "branch"},
    }, "FAQ de galho preparada para validacao.")


def validate_patch(_context: dict[str, Any], **args: Any) -> dict[str, Any]:
    patch = CardPatch.model_validate(args["patch"])
    protected = {"persona", "embedded", "gallery"}
    violations: list[str] = []
    for operation in patch.operations:
        node_type = str((operation.value or {}).get("node_type") or "").lower()
        if operation.op == "archive_node" and node_type in protected:
            violations.append(f"node protegido nao pode ser arquivado: {node_type}")
    current = graph_json_v2_store.load_current(args["persona_slug"])
    if current:
        version, graph = current
        checksum = graph_json_v2_store.checksum_graph(graph)
        if int(version) != patch.expected_graph_version or checksum != patch.graph_hash:
            raise HTTPException(409, "Graph version/hash mudou; gere um novo preview.")
    return {
        "ok": not violations,
        "summary": "Patch valido." if not violations else "Patch bloqueado.",
        "violations": violations,
        "qa": {"validated": not violations, "validator": "qa_validator"},
    }


def publish_patch(context: dict[str, Any], **args: Any) -> dict[str, Any]:
    validation = validate_patch(context, **args)
    if not validation["ok"]:
        raise HTTPException(422, validation["violations"])
    patch = CardPatch.model_validate(args["patch"])
    loaded = graph_json_v2_store.load_current(args["persona_slug"])
    if not loaded:
        raise HTTPException(404, "Graph publicado nao encontrado para a persona.")
    _, current_graph = loaded
    graph = GraphJson.model_validate(args.get("graph_json") or current_graph.model_dump(mode="json"))
    operations: list[dict[str, Any]] = []
    for item in patch.operations:
        value = dict(item.value or {})
        if item.op == "add_node":
            card = CardDraft.model_validate(value)
            operations.append({
                "op": "add_node",
                "node": {
                    "id": f"node:{card.node_type}:{card.slug}",
                    "node_type": card.node_type,
                    "slug": card.slug,
                    "title": card.title,
                    "lifecycle": {"status": "pending_validation", "revision": 1},
                    "provenance": {**card.provenance, "source": card.source},
                    "spec": {**card.spec, "summary": card.summary, "content": card.content, "tags": card.tags},
                },
            })
        elif item.op in {"archive_node", "update_node"}:
            operations.append({
                "op": item.op,
                "node_id": item.target_id,
                "patch": value,
            })
        elif item.op in {"add_edge", "revoke_edge"}:
            operations.append({
                "op": item.op,
                "edge_id": item.target_id,
                "edge": value if item.op == "add_edge" else None,
            })
    next_graph = graph_document_publisher.apply_operations(graph, operations)
    actor = str(context.get("user_id") or "") or None
    return graph_document_publisher.commit(
        graph=next_graph,
        persona_slug=args["persona_slug"],
        brand_slug=next_graph.brand_slug,
        source="agent_harness.graph_card_specialist",
        reason=patch.reason,
        published_by=actor,
        expected_version=patch.expected_graph_version,
        idempotency_key=patch.idempotency_key,
        session_id=str(context.get("session_id") or "") or None,
    )


def resolve_semantic_group(_context: dict[str, Any], **args: Any) -> dict[str, Any]:
    persona_id = args["persona_id"]
    query = str(args["query"]).strip()
    audiences = supabase_client.get_audiences(persona_id)
    match = next(
        (
            row for row in audiences
            if str((row.get("metadata") or {}).get("kind") or "semantic_group") == "semantic_group"
            and query.lower() in {str(row.get("slug") or "").lower(), str(row.get("name") or "").lower()}
        ),
        None,
    )
    if match:
        return _result("Grupo semantico existente localizado.", audience=match, proposed=False)
    proposal = {
        "persona_id": persona_id,
        "name": query,
        "slug": _slugify(query),
        "source_type": "manual",
        "metadata": {"kind": "semantic_group", "proposed_by": "campaign_operator"},
    }
    return _result("Grupo nao existe; draft de criacao preparado.", audience=proposal, proposed=True)


def create_semantic_group(context: dict[str, Any], **args: Any) -> dict[str, Any]:
    slug = _slugify(args["name"])
    existing = supabase_client.get_audience_by_slug(args["persona_id"], slug)
    if existing:
        audience = existing
    else:
        audience = supabase_client.create_audience({
            "persona_id": args["persona_id"], "name": args["name"], "slug": slug,
            "description": args.get("description"), "source_type": "manual",
            "metadata": {
                "kind": "semantic_group", "idempotency_key": args["idempotency_key"],
                "created_by": "campaign_operator", "reason": args["reason"],
            },
            "created_by_user_id": context.get("database_user_id"),
        })
    node = supabase_client.sync_audience_node(audience)
    if node:
        from services import knowledge_graph
        persona_node = knowledge_graph._ensure_persona_root(args["persona_id"])
        if persona_node:
            supabase_client.upsert_knowledge_edge(
                source_node_id=node["id"], target_node_id=persona_node["id"],
                relation_type="belongs_to_persona", persona_id=args["persona_id"],
                weight=1, metadata={"primary_tree": False, "active": True, "created_from": "agent_harness"},
            )
    return _result("Grupo semantico criado e ligado a Persona.", audience=audience, graph_node=node)


def preview_import(_context: dict[str, Any], **args: Any) -> dict[str, Any]:
    contacts = [PhoneContact.model_validate(item).model_dump() for item in args["contacts"]]
    unique = {item["phone"]: item for item in contacts}
    return _result(
        "Preview de import pronto; nenhum contato foi gravado.",
        preview={
            "persona_id": args["persona_id"],
            "audience_id": args.get("audience_id"),
            "audience_name": args.get("audience_name"),
            "consent_purpose": args.get("consent_purpose"),
            "contacts": list(unique.values()),
            "counts": {
                "received": len(contacts),
                "valid_unique": len(unique),
                "duplicates_removed": len(contacts) - len(unique),
            },
        },
    )


def create_import(context: dict[str, Any], **args: Any) -> dict[str, Any]:
    preview = preview_import(context, **args)["preview"]
    audience = supabase_client.get_audience(args["audience_id"])
    if not audience or str(audience.get("persona_id")) != str(args["persona_id"]):
        raise HTTPException(404, "Grupo semantico nao encontrado nesta persona.")
    if str((audience.get("metadata") or {}).get("kind") or "semantic_group") != "semantic_group":
        raise HTTPException(422, "O import exige um grupo semantico.")
    if args["contact_basis"] == "explicit_consent" and not str(args.get("contact_basis_statement") or "").strip():
        raise HTTPException(422, "Evidencia e obrigatoria para consentimento explicito.")

    client = supabase_client.get_client()
    existing = (
        client.table("lead_import_batches").select("*")
        .eq("provenance->>idempotency_key", args["idempotency_key"]).limit(1).execute().data or []
    )
    if existing:
        if existing[0].get("status") == "completed":
            return _result("Replay idempotente do import.", batch=existing[0], counts=preview["counts"])
        raise HTTPException(409, "Import anterior com a mesma idempotencia nao foi concluido; reconcilie o batch antes de repetir.")

    batch_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    contacts = preview["contacts"]
    checksum = hashlib.sha256(json.dumps(contacts, sort_keys=True).encode("utf-8")).hexdigest()
    batch = supabase_client.create_lead_import_batch({
        "id": batch_id,
        "persona_id": args["persona_id"],
        "audience_id": args["audience_id"],
        "filename": "sofia-contact-list.txt",
        "file_checksum": f"sha256:{checksum}",
        "source": "agent_harness",
        "contact_basis": args["contact_basis"],
        "status": "processing",
        "provenance": {
            "reason": args["reason"],
            "consent_purpose": args["consent_purpose"],
            "idempotency_key": args["idempotency_key"],
        },
        "created_by_user_id": context.get("database_user_id"),
        "created_at": now,
        "updated_at": now,
    })
    created = 0
    updated = 0
    lead_ids: list[int] = []
    for index, contact in enumerate(contacts, start=1):
        rows = (
            client.table("leads").select("id").eq("persona_id", args["persona_id"])
            .eq("lead_id", contact["phone"]).limit(1).execute().data or []
        )
        lead = supabase_client.ensure_lead_for_persona(
            lead_id=contact["phone"], persona_slug_or_id=args["persona_id"],
            nome=contact.get("name"), stage="novo", canal="bulk_import",
        )
        if not lead:
            raise RuntimeError(f"contato {index} nao foi persistido")
        lead_id = int(lead["id"])
        lead_ids.append(lead_id)
        updated += int(bool(rows))
        created += int(not rows)
        supabase_client.replace_lead_semantic_group(
            lead_id=lead_id, persona_id=args["persona_id"], audience_id=args["audience_id"],
            created_by_user_id=context.get("database_user_id"),
            idempotency_key=f"{args['idempotency_key']}:group:{lead_id}", reason=args["reason"],
        )
        supabase_client.insert_lead_import_row({
            "batch_id": batch_id, "persona_id": args["persona_id"], "row_index": index,
            "lead_id": lead_id, "status": "updated" if rows else "created",
            "parsed_data": {"lead_id": contact["phone"], "nome": contact.get("name")},
        })
        status = "granted" if args["contact_basis"] == "explicit_consent" else "pending"
        supabase_client.insert_contact_consent({
            "lead_id": lead_id, "persona_id": args["persona_id"], "channel": "whatsapp",
            "purpose": args["consent_purpose"], "status": status,
            "source": "agent_harness", "import_batch_id": batch_id, "effective_at": now,
            "granted_at": now if status == "granted" else None,
            "evidence": {"contact_basis": args["contact_basis"], "statement": args.get("contact_basis_statement")},
            "actor_user_id": context.get("database_user_id"),
            "idempotency_key": f"{args['idempotency_key']}:consent:{lead_id}",
        }, audit_payload={"reason": args["reason"], "contact_basis": args["contact_basis"]})
    supabase_client.update_lead_import_batch(batch_id, {
        "status": "completed", "total_rows": len(contacts), "valid_rows": len(contacts),
        "phone_rows": len(contacts), "created_leads": created, "updated_leads": updated,
        "invalid_rows": 0, "completed_at": now, "updated_at": now,
    })
    return _result(
        "Import concluido com consentimentos classificados como pending quando nao comprovados.",
        batch={**batch, "id": batch_id, "status": "completed"},
        counts={**preview["counts"], "created": created, "updated": updated},
        lead_ids=lead_ids,
        consent_status="granted" if args["contact_basis"] == "explicit_consent" else "pending",
    )


def resolve_consents(_context: dict[str, Any], **args: Any) -> dict[str, Any]:
    resolved = []
    for lead_id in args["lead_ids"]:
        row = supabase_client.get_latest_contact_consent(
            lead_id=lead_id, persona_id=args["persona_id"],
            channel=args.get("channel") or "whatsapp", purpose=args["purpose"],
        )
        resolved.append({"lead_id": lead_id, "status": (row or {}).get("status") or "pending"})
    return _result("Consentimentos resolvidos.", consents=resolved)


def campaign_preview(_context: dict[str, Any], **args: Any) -> dict[str, Any]:
    return _result("Preview de campanha calculado.", preview=campaigns_service.preview_campaign(args))


def campaign_create(context: dict[str, Any], **args: Any) -> dict[str, Any]:
    detail = campaigns_service.create_campaign_draft(args, actor_user_id=context.get("user_id"))
    return _result("Draft de campanha criado; nenhum envio foi iniciado.", campaign=detail)


def _campaign_status(context: dict[str, Any], status: str, **args: Any) -> dict[str, Any]:
    detail = campaigns_service.update_campaign_status(
        args["campaign_id"], expected_revision=args["expected_revision"], status=status,
        idempotency_key=args["idempotency_key"], reason=args["reason"],
        actor_user_id=context.get("user_id"),
    )
    return _result(f"Campanha {status}.", campaign=detail)


def campaign_pause(context: dict[str, Any], **args: Any) -> dict[str, Any]:
    return _campaign_status(context, "paused", **args)


def campaign_cancel(context: dict[str, Any], **args: Any) -> dict[str, Any]:
    return _campaign_status(context, "cancelled", **args)


def qa_validate(_context: dict[str, Any], **args: Any) -> dict[str, Any]:
    payload_text = json.dumps(args.get("payload") or {}, ensure_ascii=False)
    violations: list[str] = []
    if re.search(r"(?<!\d)\d{10,15}(?!\d)", payload_text) and args["operation"].startswith("graph."):
        violations.append("PII detectada em payload destinado ao Graph")
    return {
        "ok": not violations,
        "summary": "QA aprovado." if not violations else "QA bloqueou a operacao.",
        "violations": violations,
        "validator": "qa_validator",
        "read_only": True,
    }


def _manifest(
    name: str,
    owner: str,
    input_model: type[BaseModel],
    handler: Any,
    *,
    effect: ToolEffect,
    risk: ToolRisk,
    permission: str,
    description: str,
    sensitive: tuple[str, ...] = (),
) -> ToolManifest:
    return ToolManifest(
        name=name, version="1.0.0", owner_agent=owner, description=description,
        input_model=input_model, output_model=ToolResult, effect=effect, risk=risk,
        permission=permission, persona_scoped=True, timeout_seconds=30,
        idempotent=True, sensitive_fields=sensitive, handler=handler,
    )


HARNESS_TOOL_REGISTRY = ToolRegistry([
    _manifest("graph.find_cards", "graph_card_specialist", FindCardsInput, find_cards, effect=ToolEffect.READ, risk=ToolRisk.LOW, permission="view", description="Localiza cards existentes na persona."),
    _manifest("graph.create_card_draft", "graph_card_specialist", CardDraftInput, create_card_draft, effect=ToolEffect.DRAFT, risk=ToolRisk.LOW, permission="edit", description="Cria um CardDraft e Patch versionado sem publicar."),
    _manifest("graph.edit_card_draft", "graph_card_specialist", EditCardDraftInput, edit_card_draft, effect=ToolEffect.DRAFT, risk=ToolRisk.MEDIUM, permission="edit", description="Prepara edicao cirurgica versionada."),
    _manifest("graph.connect_cards", "graph_card_specialist", ConnectCardsInput, connect_cards_draft, effect=ToolEffect.DRAFT, risk=ToolRisk.MEDIUM, permission="edit", description="Prepara edge semantica sem publicar."),
    _manifest("graph.revoke_edge", "graph_card_specialist", RevokeEdgeInput, revoke_edge_draft, effect=ToolEffect.DRAFT, risk=ToolRisk.MEDIUM, permission="edit", description="Revoga edge sem deletar node."),
    _manifest("graph.attach_session_asset", "graph_card_specialist", AttachAssetDraftInput, attach_asset_draft, effect=ToolEffect.DRAFT, risk=ToolRisk.MEDIUM, permission="edit", description="Anexa asset persistido ao draft."),
    _manifest("graph.generate_faq_from_branch", "graph_card_specialist", GenerateFaqDraftInput, generate_faq_draft, effect=ToolEffect.DRAFT, risk=ToolRisk.MEDIUM, permission="edit", description="Gera FAQ draft a partir do galho."),
    _manifest("graph.validate_patch", "qa_validator", PatchInput, validate_patch, effect=ToolEffect.READ, risk=ToolRisk.LOW, permission="view", description="Valida patch, nodes protegidos e versao do Graph."),
    _manifest("graph.publish_patch", "graph_card_specialist", PatchInput, publish_patch, effect=ToolEffect.WRITE, risk=ToolRisk.HIGH, permission="edit", description="Publica patch pelo publisher canonico, apos QA."),
    _manifest("contacts.parse_list", "campaign_operator", ParseContactsInput, parse_contacts, effect=ToolEffect.READ, risk=ToolRisk.MEDIUM, permission="edit", description="Normaliza e deduplica telefones sem gravar.", sensitive=("raw_text", "contacts", "phone")),
    _manifest("imports.preview", "campaign_operator", ImportPreviewInput, preview_import, effect=ToolEffect.DRAFT, risk=ToolRisk.MEDIUM, permission="edit", description="Gera preview de import sem gravar.", sensitive=("contacts", "phone")),
    _manifest("imports.create", "campaign_operator", ImportCreateInput, create_import, effect=ToolEffect.WRITE, risk=ToolRisk.HIGH, permission="edit", description="Cria import idempotente e consentimentos, sem projetar telefones no Graph.", sensitive=("contacts", "phone")),
    _manifest("audiences.resolve_semantic_group", "campaign_operator", ResolveAudienceInput, resolve_semantic_group, effect=ToolEffect.DRAFT, risk=ToolRisk.LOW, permission="view", description="Seleciona grupo semantico existente ou propoe um novo."),
    _manifest("audiences.create_semantic_group", "campaign_operator", CreateAudienceInput, create_semantic_group, effect=ToolEffect.WRITE, risk=ToolRisk.HIGH, permission="edit", description="Cria audience semantica e edges canÃ´nicas da Persona."),
    _manifest("consents.resolve", "campaign_operator", ResolveConsentInput, resolve_consents, effect=ToolEffect.READ, risk=ToolRisk.MEDIUM, permission="view", description="Consulta consentimento aplicavel.", sensitive=("lead_ids",)),
    _manifest("campaigns.preview", "campaign_operator", CampaignPayloadInput, campaign_preview, effect=ToolEffect.DRAFT, risk=ToolRisk.MEDIUM, permission="edit", description="Calcula elegibilidade e preview sem criar outbox."),
    _manifest("campaigns.create_draft", "campaign_operator", CampaignCreateDraftInput, campaign_create, effect=ToolEffect.WRITE, risk=ToolRisk.HIGH, permission="edit", description="Cria somente draft de campanha; nunca envia."),
    _manifest("campaigns.pause", "campaign_operator", CampaignStatusInput, campaign_pause, effect=ToolEffect.WRITE, risk=ToolRisk.HIGH, permission="edit", description="Pausa campanha com revisao e idempotencia."),
    _manifest("campaigns.cancel", "campaign_operator", CampaignStatusInput, campaign_cancel, effect=ToolEffect.DESTRUCTIVE, risk=ToolRisk.CRITICAL, permission="edit", description="Cancela campanha com confirmacao pontual obrigatoria."),
    _manifest("qa.validate_operation", "qa_validator", QAInput, qa_validate, effect=ToolEffect.READ, risk=ToolRisk.LOW, permission="view", description="Valida operacao e nao escreve."),
])

HARNESS_TOOL_REGISTRY.validate()

