"""Shared API contracts. Later stories extend additively; never rename or add provider variants."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.registry import Pattern


class SemanticFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    act: list[str] = Field(default_factory=list)
    status: list[str] = Field(default_factory=list)
    access_level: list[str] = Field(default_factory=list)

    @field_validator("act")
    @classmethod
    def _check_act(cls, values: list[str]) -> list[str]:
        return [schema.validate_act(v) for v in values]

    @field_validator("status")
    @classmethod
    def _check_status(cls, values: list[str]) -> list[str]:
        return [schema.validate_status(v) for v in values]

    @field_validator("access_level")
    @classmethod
    def _check_access_level(cls, values: list[str]) -> list[str]:
        return [schema.validate_access_level(v) for v in values]


class QueryRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
    pattern: Pattern
    caller_id: str | None = None
    filters: SemanticFilters | None = None
    limit: int = Field(default=5, ge=1, le=20)
    generate_answer: bool = False
    required_acts: list[str] | None = None
    chapter: str | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    section_id: str
    act: str
    text: str
    heading: str
    score: float
    # Optional source fields; missing in the source means None, never a guess.
    chunk_index: int | None = None
    act_label: str | None = None
    status: str | None = None
    chapter: str | None = None
    chapter_title: str | None = None
    section_number: int | None = None
    source_pdf: str | None = None
    source_sha256: str | None = None
    needs_review: bool | None = None
    # Hybrid only: per-route score/rank (None when that route did not return the chunk).
    semantic_score: float | None = None
    semantic_rank: int | None = None
    keyword_score: float | None = None
    keyword_rank: int | None = None
    fused_score: float | None = None
    fused_rank: int | None = None
    # Hybrid-reranked only (None elsewhere).
    rerank_score: float | None = None
    rerank_rank: int | None = None
    omitted_reason: str | None = None


class Claim(BaseModel):
    text: str
    evidence_labels: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    label: str
    chunk_id: str
    section_id: str
    act: str
    heading: str = ""
    chapter: str | None = None
    section_number: int | None = None
    source_pdf: str | None = None


class Issue(BaseModel):
    attempt: int
    check: str  # citation_labels | claim_cited | support
    detail: str


class AttemptRecord(BaseModel):
    attempt: int
    status: str  # passed | failed | unjudged
    chars: int = 0
    latency_ms: int = 0


class GenerationResult(BaseModel):
    text: str = ""
    model: str | None = None
    # answered | insufficient_evidence | unavailable | malformed
    outcome: str | None = None
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    supporting_passages: list[RetrievedChunk] = Field(default_factory=list)
    provider: str = "openai-compatible"
    trace: dict = Field(default_factory=dict)
    context_outcome: str | None = None  # assembled | empty
    confidence: str | None = None  # "high" | "low" | None (nothing could be judged)
    issues: list[Issue] = Field(default_factory=list)
    attempts: list[AttemptRecord] = Field(default_factory=list)
    draft_answer: str = ""
    low_confidence_reason: str = ""


class QueryResult(BaseModel):
    pattern: str
    status: str
    message: str
    trace: dict
    results: list[RetrievedChunk] = Field(default_factory=list)
    generation: GenerationResult | None = None
    # Additive, empty-by-default fields later modes use:
    omitted_candidates: list[RetrievedChunk] = Field(default_factory=list)
    subquestions: list[str] = Field(default_factory=list)
    hyde_direct_candidates: list[RetrievedChunk] = Field(default_factory=list)
    hyde_query_candidates: list[RetrievedChunk] = Field(default_factory=list)
    hyde_hypothetical_text_debug: str | None = None


class ChatFilters(BaseModel):
    act: list[str] = Field(default_factory=list)
    status: list[str] = Field(default_factory=list)


class ChatRagOptions(BaseModel):
    pattern: Pattern = Pattern.SEMANTIC
    act: list[str] = Field(default_factory=list)
    status: list[str] = Field(default_factory=list)
    filters: ChatFilters | None = None  # nested form sent by the Open WebUI Pipe
    access_level: list[str] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=20)
    required_acts: list[str] | None = None
    chapter: str | None = None


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(system|developer|user|assistant)$")
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    n: int = 1
    rag_options: ChatRagOptions | None = None
