"""One shared path for /v1/query and /v1/chat/completions.

retrieve -> answer_events (one generation/validation operation) -> final GenerationResult.
"""

from collections.abc import Iterator

from building_with_rag.contracts import GenerationResult, QueryRequest, QueryResult
from building_with_rag.generation.answer import stream_answer
from building_with_rag.registry import Pattern, run_pattern
from building_with_rag.retrieval.agentic import run_agentic
from building_with_rag.retrieval.decomposition import run_decomposition
from building_with_rag.retrieval.hybrid import run_hybrid
from building_with_rag.retrieval.hyde import run_hyde
from building_with_rag.retrieval.mcp_client import run_mcp
from building_with_rag.retrieval.rerank import run_hybrid_reranked
from building_with_rag.retrieval.semantic import run_semantic
from building_with_rag.retrieval.structured import run_structured

_REASON_MAX = 200

# Modes with real retrieval; they share one retrieve -> answer path.
REAL_PATTERNS = frozenset({
    Pattern.SEMANTIC, Pattern.HYBRID, Pattern.HYBRID_RERANKED, Pattern.STRUCTURED,
    Pattern.DECOMPOSITION, Pattern.HYDE, Pattern.AGENTIC, Pattern.MCP,
})

# Modes that answer only from some statuses; other real modes always reach the answer step.
_ANSWERABLE = {
    Pattern.STRUCTURED.value: {"ok"},
    Pattern.DECOMPOSITION.value: {"ok", "partial_answer"},
    Pattern.HYDE.value: {"ok"},
    Pattern.AGENTIC.value: {"ready_for_synthesis", "partial_result"},
    Pattern.MCP.value: {"ready_for_synthesis", "partial_result"},
}


def answerable(result: QueryResult) -> bool:
    """True when the result may reach the grounded-answer step (no model call otherwise)."""
    if result.pattern not in {p.value for p in REAL_PATTERNS}:
        return False
    allowed = _ANSWERABLE.get(result.pattern)
    if allowed is not None and result.status not in allowed:
        return False
    return bool(result.results) or result.pattern not in (Pattern.AGENTIC.value, Pattern.MCP.value)


def retrieve(request: QueryRequest) -> QueryResult:
    """Semantic/hybrid/hybrid-reranked -> real retrieval; every other mode -> its placeholder. HTTP errors propagate."""
    if request.pattern == Pattern.SEMANTIC:
        return run_semantic(request)
    if request.pattern == Pattern.HYBRID:
        return run_hybrid(request)
    if request.pattern == Pattern.HYBRID_RERANKED:
        return run_hybrid_reranked(request)
    if request.pattern == Pattern.STRUCTURED:
        return run_structured(request)
    if request.pattern == Pattern.DECOMPOSITION:
        return run_decomposition(request)
    if request.pattern == Pattern.HYDE:
        return run_hyde(request)
    if request.pattern == Pattern.AGENTIC:
        return run_agentic(request)
    if request.pattern == Pattern.MCP:
        return run_mcp(request)
    payload = run_pattern(request.pattern, request.question, request.caller_id)
    return QueryResult(**payload)


def answer_events(question: str, retrieval: QueryResult) -> Iterator[tuple[str, object]]:
    """Yield ("notice"|"text", str) pieces, then ("final", GenerationResult | None).

    Placeholder modes and non-ok structured results yield their message and no generation.
    """
    if not answerable(retrieval):
        yield ("text", retrieval.message)
        yield ("final", None)
        return
    if retrieval.status in ("partial_answer", "partial_result"):
        yield ("text", retrieval.message + "\n\n")
    limits = {"max_passages": 25, "max_chars": 60_000} if retrieval.pattern == Pattern.MCP.value else {}
    yield from stream_answer(question, retrieval.results, **limits)


def _source_line(c) -> str:
    prefix, _, number = c.section_id.partition(":")
    return f"{c.label} · {prefix.upper()} §{number} · {c.heading or '(no heading)'} · {c.section_id}"


def render_footer(generation: GenerationResult | None) -> str:
    """Text appended after the streamed answer (starts with a blank line); may be empty."""
    if generation is None:
        return ""
    if generation.outcome == "answered" and generation.confidence == "high":
        lines = ["", "", "Evidence check passed — confidence: high", "Sources:"]
        lines += [_source_line(c) for c in generation.citations]
        return "\n".join(lines)
    if generation.outcome == "insufficient_evidence":
        reason = str(generation.trace.get("reason") or "").strip()
        if len(reason) > _REASON_MAX:
            reason = reason[: _REASON_MAX - 1].rstrip() + "…"
        sentence = "The retrieved BNS/IPC evidence is not sufficient to answer this question."
        return "\n\n" + sentence + (f" Reason: {reason}" if reason else "")
    if generation.confidence == "low":
        lines = ["", "", "DRAFT — low confidence, not the final answer.", generation.low_confidence_reason]
        lines += [f"- {i.check}: {i.detail}" for i in generation.issues if i.attempt == len(generation.attempts)]
        return "\n".join(lines)
    return ""
