from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolEffect(str, Enum):
    READ = "read"
    DRAFT = "draft"
    WRITE = "write"
    DESTRUCTIVE = "destructive"
    EXTERNAL = "external"


class ToolRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RunStatus(str, Enum):
    PLANNED = "planned"
    AWAITING_APPROVAL = "awaiting_approval"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolGrant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    tool_version: str
    schema_checksum: str
    granted_by: str
    granted_at: datetime
    expires_at: datetime
    revoked_at: datetime | None = None
    reason: str = Field(min_length=3, max_length=500)

    def is_valid(self, *, now: datetime, tool_name: str, version: str, checksum: str) -> bool:
        expiry = self.expires_at
        if expiry.tzinfo is None and now.tzinfo is not None:
            expiry = expiry.replace(tzinfo=now.tzinfo)
        return bool(
            self.revoked_at is None
            and self.tool_name == tool_name
            and self.tool_version == version
            and self.schema_checksum == checksum
            and expiry > now
        )


class CardRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relation_type: str
    target: str
    confidence: float = Field(1.0, ge=0, le=1)
    weight: float = Field(1.0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CardDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_type: str
    slug: str
    title: str
    summary: str | None = None
    content: str | dict[str, Any] | None = None
    lifecycle: dict[str, Any] = Field(default_factory=lambda: {"status": "draft", "revision": 1})
    status: Literal[
        "draft", "pending_source", "pending_validation", "validated",
        "approved", "rejected", "archived",
    ] = "pending_validation"
    provenance: dict[str, Any] = Field(default_factory=dict)
    source: str = "pending_source"
    tags: list[str] = Field(default_factory=list)
    spec: dict[str, Any] = Field(default_factory=dict)
    relations: list[CardRelation] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_content(self) -> "CardDraft":
        if not (self.summary or self.content):
            raise ValueError("summary ou content e obrigatorio")
        return self


class PatchOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal[
        "add_node", "update_node", "archive_node", "add_edge", "revoke_edge",
        "attach_asset", "generate_faq",
    ]
    target_id: str | None = None
    value: dict[str, Any] = Field(default_factory=dict)


class CardPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_graph_version: int = Field(ge=0)
    graph_hash: str = Field(min_length=8)
    idempotency_key: str = Field(min_length=8, max_length=200)
    reason: str = Field(min_length=3, max_length=500)
    operations: list[PatchOperation] = Field(min_length=1, max_length=50)


class SessionCreate(BaseModel):
    persona_id: str
    coordinator_key: str = "sofia"
    selected_context: dict[str, Any] = Field(default_factory=dict)
    selected_node_ids: list[str] = Field(default_factory=list)
    graph_version: int | None = Field(None, ge=0)
    graph_hash: str | None = None
    idempotency_key: str = Field(min_length=8, max_length=200)
    reason: str = Field(min_length=3, max_length=500)


class MessageCreate(BaseModel):
    message: str = Field(min_length=1, max_length=30000)
    expected_revision: int = Field(ge=1)
    idempotency_key: str = Field(min_length=8, max_length=200)
    reason: str = Field(min_length=3, max_length=500)
    context: dict[str, Any] = Field(default_factory=dict)


class RunMutation(BaseModel):
    expected_revision: int = Field(ge=1)
    idempotency_key: str = Field(min_length=8, max_length=200)
    reason: str = Field(min_length=3, max_length=500)


class GrantsPut(BaseModel):
    user_id: str
    persona_id: str
    expected_revision: int = Field(ge=1)
    idempotency_key: str = Field(min_length=8, max_length=200)
    reason: str = Field(min_length=3, max_length=500)
    grants: list[ToolGrant]


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    ok: bool
    summary: str | None = None
