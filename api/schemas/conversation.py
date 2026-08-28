"""Strict contracts shared by Brain and persona n8n workflows."""
from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConversationRoute(StrEnum):
    SDR = "SDR"
    CLOSER = "CLOSER"
    HUMAN = "HUMAN"


class ConversationJourneyState(StrEnum):
    COLLECTING = "collecting"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    QUALIFIED_CONFIRMED = "qualified_confirmed"
    HANDED_OFF = "handed_off"
    CONVERTED = "converted"
    CLOSED = "closed"


class ConversationOperationalMode(StrEnum):
    COLLECTION = "collection"
    CONFIRMATION = "confirmation"
    POST_QUALIFICATION_SUPPORT = "post_qualification_support"


class CartAction(StrEnum):
    NONE = "none"
    ADD_ITEM = "add_item"
    CHANGE_QUANTITY = "change_quantity"
    REMOVE_ITEM = "remove_item"
    SHOW_TOTAL = "show_total"
    CONFIRM_ORDER = "confirm_order"
    CANCEL_ORDER = "cancel_order"


class ContextCard(StrictModel):
    """Immutable knowledge snapshot injected into one model turn.

    ``id`` is always the stable Graph JSON node id. Database projection UUIDs
    are deliberately kept as trace metadata and never become card identity.
    """

    id: str = Field(min_length=1)
    projection_node_id: str | None = None
    node_type: str = Field(min_length=1)
    slug: str = Field(min_length=1)
    title: str = Field(min_length=1)
    rendered_content: str = Field(min_length=1)
    editable_content: str = ""
    content_checksum: str = Field(min_length=1)
    revision: int = Field(ge=1)
    graph_version: int = Field(ge=1)
    graph_checksum: str = Field(min_length=1)
    context_role: str
    position: int = Field(ge=0)
    selection_reason: dict[str, Any] = Field(default_factory=dict)
    path: list[str] = Field(default_factory=list)
    chunk_refs: list[str] = Field(default_factory=list)
    source: str = "pending_source"
    status: str = "approved"
    relations: list[dict[str, Any]] = Field(default_factory=list)
    technical_metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationFactStatus(StrEnum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    DECLINED = "declined"
    NEEDS_CONFIRMATION = "needs_confirmation"
    INVALID = "invalid"


class BranchAction(StrEnum):
    NONE = "none"
    KEEP = "keep"
    SELECT = "select"
    SWITCH = "switch"
    ADD = "add"


class ServiceOperationAction(StrEnum):
    ADD = "add"
    KEEP = "keep"
    DROP = "drop"


class ServiceOperation(StrictModel):
    """Graph-proved mutation of the active service set for one inbound."""

    action: ServiceOperationAction
    branch_anchor_node_id: str = Field(min_length=1)
    branch_path_checksum: str = Field(min_length=1)
    evidence_span: str = Field(min_length=1)
    evidence_type: str = Field(
        default="exact_catalog",
        pattern="^(exact_catalog|confirmed_candidate|explicit_change)$",
    )
    resolution_method: str = "exact_catalog"
    score: float | None = Field(default=None, ge=0, le=1)
    margin: float | None = Field(default=None, ge=0, le=1)


class ServiceObservation(StrictModel):
    """Non-authoritative service signal observed by the model."""

    branch_anchor_node_id: str = Field(min_length=1)
    evidence_span: str = Field(min_length=1)
    observed_intent: str = Field(
        default="mention",
        pattern="^(mention|select|add|switch|remove|question)$",
    )
    confidence: float = Field(default=0.0, ge=0, le=1)


class InteractionKind(StrEnum):
    CONTINUE_CURRENT = "continue_current"
    NEW_DEMAND = "new_demand"
    POST_COMPLETION_QUESTION = "post_completion_question"
    COURTESY_CLOSE = "courtesy_close"
    POST_SALE_OPERATION = "post_sale_operation"
    UNCLEAR = "unclear"


class JourneyAction(StrEnum):
    CONTINUE = "continue"
    OPEN = "open"
    NONE = "none"


class InteractionObservation(StrictModel):
    """Non-authoritative semantic observation reconciled by the backend."""

    kind: InteractionKind = InteractionKind.UNCLEAR
    evidence_span: str = ""
    confidence: float = Field(default=0.0, ge=0, le=1)


class SharedMemoryFact(StrictModel):
    key: str = Field(min_length=1)
    value: Any | None = None
    owner_node_id: str = Field(min_length=1)
    status: ConversationFactStatus
    confidence: float | None = Field(default=None, ge=0, le=1)
    source: str = "conversation_fact"
    journey_id: str | None = None
    recorded_at: str | None = None
    source_message_id: str | None = None
    reuse_policy: str = "historical_only"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SharedLeadMemory(StrictModel):
    """Agent-neutral, complete memory envelope for one persona-scoped lead."""

    profile_facts: list[SharedMemoryFact] = Field(default_factory=list)
    current_journey: dict[str, Any] | None = None
    historical_facts: list[SharedMemoryFact] = Field(default_factory=list)
    journey_outcomes: list[dict[str, Any]] = Field(default_factory=list)
    pending_items: list[dict[str, Any]] = Field(default_factory=list)
    recent_messages: list[dict[str, Any]] = Field(default_factory=list)
    agent_activity: list[dict[str, Any]] = Field(default_factory=list)
    policy_version: str = "shared_lead_memory_v1"


class ExtractedFact(StrictModel):
    field_key: str = Field(min_length=1)
    value: Any | None = None
    status: ConversationFactStatus = ConversationFactStatus.KNOWN
    source_message_id: str | None = None
    owner_node_id: str = Field(min_length=1)
    evidence_span: str = ""
    confidence: float = Field(default=0.0, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CommercialClaim(StrictModel):
    claim_type: str = Field(pattern="^(price|availability|schedule|stock|duration|service_detail|other)$")
    value: dict[str, Any] = Field(default_factory=dict)
    evidence_node_ids: list[str] = Field(default_factory=list)
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class ConversationProposal(StrictModel):
    """One model suggestion that must be proved against the graph contract."""

    interaction_observation: InteractionObservation = Field(
        default_factory=InteractionObservation
    )
    branch_action: BranchAction = BranchAction.NONE
    branch_anchor_node_id: str | None = None
    branch_path_checksum: str | None = None
    branch_evidence_span: str = ""
    service_observations: list[ServiceObservation] = Field(default_factory=list)
    service_operations: list[ServiceOperation] = Field(default_factory=list)
    extracted_facts: list[ExtractedFact] = Field(default_factory=list)
    claims: list[CommercialClaim] = Field(default_factory=list)
    next_question_node_id: str | None = None
    cited_node_ids: list[str] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    reply: str = ""
    qualification_complete: bool = False
    handoff_requested: bool = False


class ConversationContext(StrictModel):
    persona_slug: str
    agent_slug: str
    graph_version: int = Field(ge=1)
    graph_checksum: str = Field(min_length=1)
    messages: list[dict[str, Any]]
    cart: dict[str, Any]
    rag_nodes: list[dict[str, Any]]
    rag_paths: list[list[str]]
    rag_chunks: list[dict[str, Any]] = Field(default_factory=list)
    context_cards: list[ContextCard] = Field(default_factory=list)
    system_prompt: str = ""
    available_services: list[dict[str, str]] = Field(default_factory=list)
    active_branch_node_id: str | None = None
    active_branch_node_ids: list[str] = Field(default_factory=list)
    # Branches (any offering -- service or product) already confirmed in this
    # ledger, per conversation_ledger_branches.state='completed'. Lets the
    # backend tell "this active branch already had its own confirmation
    # cycle" apart from "this one is new and still needs one" once a journey
    # has more than one active branch. See _decide's post_qualification_support
    # gate in graph_agent_runtime_v3.py.
    completed_branch_node_ids: list[str] = Field(default_factory=list)
    ledger_id: str | None = None
    active_path_checksum: str | None = None
    branch_node_ids: list[str] = Field(default_factory=list)
    graph_contract: dict[str, Any] = Field(default_factory=dict)
    publication_id: str | None = None
    runtime_version: str = "conversation_v2"
    retrieval_trace: dict[str, Any] = Field(default_factory=dict)
    known_facts: list[dict[str, Any]] = Field(default_factory=list)
    time_since_last_client_message: str | None = None
    pending_reconfirmation: bool = False
    journey_id: str | None = None
    journey_sequence: int = Field(default=1, ge=1)
    journey_state: ConversationJourneyState = ConversationJourneyState.COLLECTING
    pending_field_key: str | None = None
    pending_question_node_id: str | None = None
    last_handoff: dict[str, Any] = Field(default_factory=dict)
    operational_mode: ConversationOperationalMode = ConversationOperationalMode.COLLECTION
    shared_memory: SharedLeadMemory = Field(default_factory=SharedLeadMemory)
    post_completion_state: dict[str, Any] = Field(default_factory=dict)


class ConversationDecision(StrictModel):
    classifier: str = "deterministic_v1"
    intent: str
    route: ConversationRoute
    confidence: float = Field(ge=0, le=1)
    cart_action: CartAction = CartAction.NONE
    product_slug: str | None = None
    quantity: int | None = Field(default=None, ge=1)
    lead_stage: str
    handoff_reason: str | None = None
    evidence_node_ids: list[str] = Field(default_factory=list)


class AgentResponse(StrictModel):
    reply_text: str | None
    role: ConversationRoute
    evidence_node_ids: list[str] = Field(default_factory=list)
    cart_state: dict[str, Any]
    handoff_required: bool = False
    extracted_fields: dict[str, str] = Field(default_factory=dict)
    identified_service_slug: str | None = None
    proposal: ConversationProposal | None = None
    proof: dict[str, Any] = Field(default_factory=dict)
    repair_context_cards: list[dict[str, Any]] = Field(default_factory=list)
    # Billed usage reported by the reply-drafting model call (e.g. DeepSeek's
    # OpenAI-compatible {prompt_tokens, completion_tokens, total_tokens}).
    # Optional because not every route (deterministic fallbacks, HUMAN
    # handoff) makes a paid model call.
    token_usage: dict[str, Any] | None = None

