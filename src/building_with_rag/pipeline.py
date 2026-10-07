"""One shared path for /v1/query and /v1/chat/completions.

retrieve -> answer_events (one generation/validation operation) -> final GenerationResult.
"""

from collections.abc import Iterator

from building_with_rag.contracts import GenerationResult, QueryRequest, QueryResult
from building_with_rag.generation.answer import stream_answer
from building_with_rag.registry import Pattern, run_pattern
from building_with_rag.retrieval.semantic import run_semantic

_REASON_MAX = 200


def retrieve(request: QueryRequest) -> QueryResult:
    """Semantic -> real retrieval; every other mode -> its placeholder. HTTP errors propagate."""
    if request.pattern == Pattern.SEMANTIC:
        return run_semantic(request)
    payload = run_pattern(request.pattern, request.question, request.caller_id)
    return QueryResult(**payload)


def answer_events(question: str, retrieval: QueryResult) -> Iterator[tuple[str, object]]:
    """Yield ("notice"|"text", str) pieces, then ("final", GenerationResult | None).

    Non-semantic modes yield their placeholder message and no generation.
    """
    if retrieval.pattern != Pattern.SEMANTIC.value:
        yield ("text", retrieval.message)
        yield ("final", None)
        return
    yield from stream_answer(question, retrieval.results)


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
