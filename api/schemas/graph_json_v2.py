"""Typed canonical Graph JSON contract.

``2.1`` is the write contract.  ``2.0`` remains readable during the cutover and
is adapted by the same models so legacy publishers and the dashboard keep
working while projections move to destination-scoped grants.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


KnowledgeNodeType = Literal[
    "persona", "brand", "briefing", "campaign", "audience",
    "product_group", "product", "offer", "copy", "faq", "rule", "tone",
    "asset",
    # One per lead with an open conversation; parent of any media the customer
    # sends over WhatsApp. Lives under the audience, which lives under the
    # campaign that originated the outreach.
    "conversation",
]
ActionNodeType = Literal["gallery", "embedded", "marketing_workspace"]


class Lifecycle(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: Literal[
        "draft", "pending_validation", "approved", "rejected", "archived",
        "proposed", "active", "revoked",
    ] = "draft"
    revision: int = Field(1, ge=1)
    approved_revision_checksum: str | None = None
    approved_at: datetime | None = None
    approved_by: str | None = None
    activated_at: datetime | None = None
    revoked_at: datetime | None = None


class Provenance(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str | None = None
    source_type: str | None = None
    source_ref: str | None = None
    source_checksum: str | None = None
    authored_by: str | None = None
    created_at: datetime | None = None
    base_version: int | None = Field(None, ge=0)
    reason: str | None = None


class MarkdownDocument(BaseModel):
    renderer: str
    checksum: str
    content: str


class ActionConsumer(BaseModel):
    kind: str
    ref: str


class ActionProjection(BaseModel):
    model_config = ConfigDict(extra="allow")

    kind: str
    format_ref: str | None = None
    embedding_profile_ref: str | None = None
    chunking_profile_ref: str | None = None


class ActionPolicy(BaseModel):
    model_config = ConfigDict(extra="allow")

    requires_approved_content: bool = True
    asset_requires_textual_description: bool = True
    auto_connect: list[dict[str, Any]] = Field(default_factory=list)


class ActionSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    destination_id: str
    destination_type: str
    enabled: bool = True
    consumer: ActionConsumer
    accepted_node_types: list[str] = Field(default_factory=list)
    projection: ActionProjection
    policy: ActionPolicy = Field(default_factory=ActionPolicy)


class Node(BaseModel):
    """A stable graph node.

    ``data``/``label``/``parent_id`` are the 2.0 compatibility surface.  New
    documents use ``title``, typed lifecycle/provenance/spec/action fields and
    express hierarchy only through ``contains`` edges.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    node_class: Literal["knowledge", "action"] = "knowledge"
    node_type: str
    slug: str
    title: str | None = None
    label: str | None = None
    parent_id: str | None = None
    lifecycle: Lifecycle = Field(default_factory=Lifecycle)
    provenance: Provenance = Field(default_factory=Provenance)
    spec: dict[str, Any] = Field(default_factory=dict)
    action: ActionSpec | None = None
    markdown: MarkdownDocument | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def adapt_v20(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        raw = dict(value)
        raw["title"] = raw.get("title") or raw.get("label") or raw.get("slug")
        raw["label"] = raw.get("label") or raw["title"]
        node_type = str(raw.get("node_type") or "")
        if node_type in {"gallery", "embedded", "marketing_workspace"}:
            raw["node_class"] = "action"
        data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
        if "lifecycle" not in raw:
            status = str(data.get("validation_status") or data.get("status") or "draft").lower()
            status_aliases = {
                "validated": "approved", "ativo": "approved", "embedded": "approved",
                "aprovado": "approved", "approved": "approved",
                "pendente": "pending_validation", "pendente_validacao": "pending_validation",
                "pending": "pending_validation", "pending_validation": "pending_validation",
                "rejeitado": "rejected", "rejected": "rejected",
                "arquivado": "archived", "inactive": "archived", "archived": "archived",
                "active": "active", "revoked": "revoked", "proposed": "proposed",
                "draft": "draft",
            }
            # The 2.0 status field was intentionally free-form. Preserve it in
            # ``data`` while mapping only the lifecycle state into the closed
            # v2.1 vocabulary, so legacy previews never fail model parsing.
            raw["lifecycle"] = {"status": status_aliases.get(status, "draft")}
        if "provenance" not in raw:
            raw["provenance"] = {"source": data.get("source")}
        return raw

    @model_validator(mode="after")
    def expose_legacy_projection(self) -> "Node":
        self.title = self.title or self.label or self.slug
        self.label = self.label or self.title
        status = self.lifecycle.status
        legacy = dict(self.data or {})
        legacy.setdefault("status", status)
        source = self.provenance.source or self.provenance.source_type
        if source:
            legacy.setdefault("source", source)
        if self.spec:
            legacy.update({key: value for key, value in self.spec.items() if key not in legacy})
        if self.markdown:
            legacy["markdown"] = self.markdown.content
            legacy["markdown_checksum"] = self.markdown.checksum
            legacy["markdown_renderer"] = self.markdown.renderer
        if self.action:
            legacy["action"] = self.action.model_dump(mode="json")
        self.data = legacy
        return self


class EdgeLifecycle(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: Literal["proposed", "active", "revoked"] = "active"
    activated_at: datetime | None = None
    revoked_at: datetime | None = None


class PublicationGrant(BaseModel):
    model_config = ConfigDict(extra="allow")

    mode: Literal["manual", "policy"] = "manual"
    actor: str | None = None
    reason: str | None = None
    policy_id: str | None = None
    source_approval_checksum: str | None = None


class Edge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    target: str
    relation_type: str | None = None
    relation: str | None = None
    relation_class: Literal["hierarchy", "semantic", "provenance", "publication"] | None = None
    weight: float | None = Field(None, ge=0)
    primary_tree: bool = True
    lifecycle: EdgeLifecycle = Field(default_factory=EdgeLifecycle)
    grant: PublicationGrant | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def adapt_relation(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        raw = dict(value)
        relation = raw.get("relation_type") or raw.get("relation") or "contains"
        raw["relation_type"] = relation
        raw["relation"] = relation
        if "relation_class" not in raw:
            raw["relation_class"] = {
                "contains": "hierarchy",
                "derived_from": "provenance",
                "publishes_to": "publication",
            }.get(str(relation), "semantic")
        return raw

    @model_validator(mode="after")
    def keep_relation_aliases(self) -> "Edge":
        relation = self.relation_type or self.relation or "contains"
        self.relation_type = relation
        self.relation = relation
        if self.weight is None:
            self.weight = DEFAULT_RELATION_WEIGHTS.get(relation, 1.0)
        if relation == "publishes_to":
            self.primary_tree = False
            self.relation_class = "publication"
        return self


DEFAULT_RELATION_WEIGHTS: dict[str, float] = {
    "answers": 1.0,
    "contains": 0.9,
    "represents": 0.9,
    "applies_to": 0.9,
    "targets": 0.85,
    "uses_asset": 0.85,
    "supports": 0.8,
    "derived_from": 0.65,
    "references": 0.55,
    "publishes_to": 1.0,
}


class Layout(BaseModel):
    engine: str = "top-down-canonical"
    positions: dict[str, list[float]] = Field(default_factory=dict)


class Validation(BaseModel):
    is_valid: bool = True
    errors: list[str] = Field(default_factory=list)


class GraphJson(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Missing version means the historical 2.0 shape. New writes explicitly
    # declare 2.1, preventing old clients from accidentally changing semantics.
    schema_version: Literal["2.0", "2.1"] = "2.0"
    graph_id: str
    tenant: str
    persona_slug: str
    brand_slug: str | None = None
    environment: str = "production"
    graph_version: int | None = Field(None, ge=1)
    status: Literal[
        "draft", "validated", "committed", "projecting", "published",
        "projection_failed", "withdrawn",
    ] = "draft"
    content_checksum: str | None = None
    provenance: Provenance = Field(default_factory=Provenance)
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    layout: Layout = Field(default_factory=Layout)
    validation: Validation = Field(default_factory=Validation)


class PatchOperation(BaseModel):
    op: Literal[
        "add_node", "remove_node", "update_node", "approve_node", "archive_node",
        "add_edge", "remove_edge", "revoke_edge", "set_status",
    ]
    node: Node | None = None
    edge: Edge | None = None
    node_id: str | None = None
    edge_id: str | None = None
    id: str | None = None
    patch: dict[str, Any] = Field(default_factory=dict)
    value: Any | None = None


class Patch(BaseModel):
    description: str
    expected_graph_version: int | None = Field(None, ge=0)
    graph_hash: str | None = None
    idempotency_key: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    operations: list[PatchOperation] = Field(default_factory=list)

