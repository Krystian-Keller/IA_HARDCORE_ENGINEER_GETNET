from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints, model_validator

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Route = Literal["knowledge_getnet", "knowledge_web", "customer_support"]
GuardrailCategory = Literal["supported", "irrelevant", "sensitive", "unsafe"]
EscalationReason = Literal[
    "unsafe_request", "sensitive_request", "irrelevant_request", "unsupported_request",
    "insufficient_information", "routing_failure", "internal_error",
    "human_requested",
]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: NonEmptyText = Field(max_length=5000)
    user_id: NonEmptyText = Field(max_length=100)
    conversation_id: UUID | None = None


class ChatResponse(BaseModel):
    answer: NonEmptyText
    agents_used: list[str]
    tools_used: list[str]
    sources: list[str] = Field(default_factory=list)
    request_id: str | None = None
    route: Route | None = None
    guardrail_category: GuardrailCategory | None = None
    status: Literal["answered", "escalated", "awaiting_human"] = "answered"
    escalation_reason: EscalationReason | None = None
    conversation_id: UUID | None = None
    requires_human: bool = False
    handoff_id: UUID | None = None
    data_mode: Literal["synthetic"] = "synthetic"
    # Internal evidence is available to local evaluation, never serialized to clients.
    evidence: list[dict] = Field(default_factory=list, exclude=True)


class GuardrailResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: GuardrailCategory
    reason: NonEmptyText = Field(max_length=500)


class KnowledgeAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sufficient_context: bool
    answer: NonEmptyText = Field(description=(
        "Directly answer the user's question in the same language as the question, "
        "translating source facts when needed. Use only supplied evidence."
    ))


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    route: Route


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agents: list[Route] = Field(min_length=1, max_length=2)
    needs_synthesis: bool = False

    @model_validator(mode="after")
    def validate_steps(self) -> "ExecutionPlan":
        if len(set(self.agents)) != len(self.agents):
            raise ValueError("Specialists must not repeat")
        if len(self.agents) == 2:
            if self.agents[0] != "customer_support" or self.agents[1] != "knowledge_getnet":
                raise ValueError("Mixed plans must run support followed by Getnet knowledge")
            if not self.needs_synthesis:
                raise ValueError("Mixed plans require synthesis")
        elif self.needs_synthesis:
            raise ValueError("A single specialist does not require synthesis")
        return self


class Message(BaseModel):
    role: Literal["user", "assistant", "human"]
    content: str


class ConversationContext(BaseModel):
    messages: list[Message] = Field(default_factory=list)
    previous_results: list[dict] = Field(default_factory=list)

    def prompt_data(self) -> str:
        return self.model_dump_json()

    def retrieval_query(self, message: str) -> str:
        # User history resolves short follow-ups without searching private tool records.
        return "\n".join([*(m.content for m in self.messages if m.role == "user"), message])


class HandoffReply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    response: NonEmptyText = Field(max_length=5000)
    user_id: NonEmptyText | None = Field(default=None, max_length=100)


class ToolDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tools: list[Literal["get_customer_profile", "get_customer_activity"]] = Field(
        min_length=1, max_length=2
    )


class Chunk(BaseModel):
    text: NonEmptyText
    source: HttpUrl


class SearchResult(BaseModel):
    title: NonEmptyText
    snippet: NonEmptyText
    url: HttpUrl


class AppError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
