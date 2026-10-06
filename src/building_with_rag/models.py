"""Shared typed contracts. Later stories extend these additively; never replace them."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from building_with_rag.registry import Pattern


class SemanticFilters(BaseModel):
    act: list[str] = Field(default_factory=list)
    status: list[str] = Field(default_factory=list)
    access_level: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    pattern: Pattern
    caller_id: str | None = None
    filters: SemanticFilters | None = None
    limit: int = Field(default=5, ge=1, le=20)
    generate_answer: bool = False
    required_acts: list[str] = Field(default_factory=list)
    chapter: str | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str  # act-qualified, e.g. "BNS:103"
    section_id: str
    act: str
    text: str
    heading: str | None = None
    score: float | None = None
    source_file: str | None = None
    source_page: int | None = None
    chapter: str | None = None


class OmittedCandidate(BaseModel):
    chunk_id: str
    omitted_reason: str


class SubquestionEvidence(BaseModel):
    subquestion: str
    status: Literal["evidenced", "no_evidence"]
    results: list[RetrievedChunk] = Field(default_factory=list)
    reason: str | None = None


class StructuredSignals(BaseModel):
    intent: Literal["exact_lookup", "filter", "aggregation"]
    act: str | None = None
    section_number: str | None = None
    chapter: str | None = None


class GenerationResult(BaseModel):
    outcome: Literal["answered", "insufficient_evidence", "unavailable", "malformed"]
    answer: str | None = None
    claims: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    supporting_passages: list[RetrievedChunk] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    trace: dict[str, Any] = Field(default_factory=dict)
    context_outcome: str | None = None
    confidence: float | None = None
    draft_answer: str | None = None
    issues: list[str] = Field(default_factory=list)
    attempts: int | None = None
    low_confidence_reason: str | None = None


class QueryResult(BaseModel):
    pattern: Pattern
    status: str
    message: str
    trace: dict[str, Any] = Field(default_factory=dict)
    results: list[RetrievedChunk] = Field(default_factory=list)
    generation: GenerationResult | None = None
    omitted_candidates: list[OmittedCandidate] = Field(default_factory=list)
    subquestions: list[SubquestionEvidence] = Field(default_factory=list)
    hyde_direct_candidates: list[RetrievedChunk] = Field(default_factory=list)
    hyde_query_candidates: list[RetrievedChunk] = Field(default_factory=list)
    hyde_hypothetical_text_debug: str | None = None


# --- OpenAI-compatible contracts (text only) ---


class RagOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: Pattern | None = None
    filters: SemanticFilters | None = None
    limit: int | None = Field(default=None, ge=1, le=20)
    required_acts: list[str] | None = None
    chapter: str | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    # Unknown browser-supplied fields are ignored, never trusted.
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    n: int = Field(default=1, ge=1)
    rag_options: RagOptions | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]


class ChatCompletionDelta(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "building-with-rag"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class OpenAIErrorBody(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None


class OpenAIErrorEnvelope(BaseModel):
    error: OpenAIErrorBody


# --- MongoDB schema contract (no database work in the seed) ---


class SectionDocument(BaseModel):
    """Shape of one document in the sections/chunks collection."""

    chunk_id: str
    section_id: str
    act: str
    text: str
    heading: str | None = None
    chapter: str | None = None
    status: str | None = None
    access_level: str | None = None
    source_file: str | None = None
    source_page: int | None = None
    embedding: list[float] | None = None  # voyage-3.5, 1,024 dimensions
    embedding_model: str = "voyage-3.5"
    embedding_version: str = "voyage-3.5"
    embedding_dimensions: int = 1024
