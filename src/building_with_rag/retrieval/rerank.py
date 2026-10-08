"""Hybrid re-ranking: bounded hybrid candidates re-scored by one provider call.

Re-ranking scores rank only; they do not prove a passage is correct or answers the question.
Any failure returns an error: hybrid order is never returned labelled as re-ranked.
"""

import math
import time

import httpx

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.registry import Pattern
from building_with_rag.retrieval import semantic
from building_with_rag.retrieval.hybrid import run_hybrid
from building_with_rag.retrieval.semantic import RetrievalError
from building_with_rag.settings import Settings, get_settings

MAX_CANDIDATES = 20
NOT_SENT = "not_sent_to_reranker"
BELOW_RETURN = "below_return_limit"

_SCORE_NOTE = (
    "Re-ranking scores rank passages only and do not prove a passage is correct or answers "
    "the question."
)


def _not_ready(message: str) -> RetrievalError:
    return RetrievalError(503, "retrieval_not_ready", message)


def validate_settings(s: Settings) -> None:
    """Raise 503 naming the offending setting (never its value)."""
    if not s.rerank_api_key:
        raise _not_ready("RERANK_API_KEY is not set; hybrid-reranked requires it.")
    if not s.rerank_api_base_url:
        raise _not_ready("RERANK_API_BASE_URL is not set; hybrid-reranked requires it.")
    if not s.rerank_model_name:
        raise _not_ready("RERANK_MODEL_NAME is not set; hybrid-reranked requires it.")
    if s.rerank_request_timeout_seconds < 1:
        raise _not_ready("RERANK_REQUEST_TIMEOUT_SECONDS must be at least 1.")
    if not 1 <= s.rerank_return_limit:
        raise _not_ready("RERANK_RETURN_LIMIT must be at least 1.")
    if s.rerank_send_limit < s.rerank_return_limit:
        raise _not_ready("RERANK_SEND_LIMIT must be at least RERANK_RETURN_LIMIT.")
    if s.rerank_candidate_limit < s.rerank_send_limit:
        raise _not_ready("RERANK_CANDIDATE_LIMIT must be at least RERANK_SEND_LIMIT.")
    if s.rerank_candidate_limit > MAX_CANDIDATES:
        raise _not_ready(f"RERANK_CANDIDATE_LIMIT must be at most {MAX_CANDIDATES}.")


def _upstream() -> RetrievalError:
    return RetrievalError(
        502, "retrieval_upstream_error", "Re-ranking request failed; no re-ranked result returned."
    )


def call_rerank(settings: Settings, query: str, documents: list[str]) -> tuple[list[dict], int | None]:
    """One POST {base}/rerank. Return (validated data[], usage_tokens). Raises 502 on any failure."""
    url = settings.rerank_api_base_url.rstrip("/") + "/rerank"
    try:
        resp = httpx.post(
            url,
            json={"model": settings.rerank_model_name, "query": query, "documents": documents},
            headers={"Authorization": f"Bearer {settings.rerank_api_key}"},
            timeout=float(settings.rerank_request_timeout_seconds),
        )
        resp.raise_for_status()
        payload = resp.json()
    except (httpx.HTTPError, ValueError):
        raise _upstream() from None
    try:
        data = payload["data"]
        if not isinstance(data, list) or not data:
            raise ValueError
        seen: set[int] = set()
        for item in data:
            index, score = item["index"], item["relevance_score"]
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(documents):
                raise ValueError
            if index in seen:
                raise ValueError
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError
            seen.add(index)
        if len(seen) != len(documents):  # every sent candidate must be scored; never fill in
            raise ValueError
        usage = payload.get("usage")
        tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
        return data, tokens if isinstance(tokens, int) else None
    except (KeyError, TypeError, ValueError, AttributeError):
        raise _upstream() from None


def select(
    candidates: list[RetrievedChunk],
    scores: dict[int, float],
    send_limit: int,
    return_limit: int,
    limit: int,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """Pure selection. *candidates* are in fused order; *scores* maps sent index -> score.

    Returns (final results by rerank_rank, omitted candidates in fused_rank order).
    """
    sent = candidates[:send_limit]
    omitted = [c.model_copy(update={"omitted_reason": NOT_SENT}) for c in candidates[send_limit:]]
    order = sorted(range(len(sent)), key=lambda i: (-scores[i], sent[i].fused_rank or 0))
    keep = min(limit, return_limit)
    final: list[RetrievedChunk] = []
    for rank, i in enumerate(order, start=1):
        fields = {"rerank_score": float(scores[i]), "rerank_rank": rank}
        if rank <= keep:
            final.append(sent[i].model_copy(update={**fields, "score": float(scores[i])}))
        else:
            omitted.append(sent[i].model_copy(update={**fields, "omitted_reason": BELOW_RETURN}))
    omitted.sort(key=lambda c: c.fused_rank or 0)
    return final, omitted


def _document(chunk: RetrievedChunk) -> str:
    return f"{chunk.heading}\n{chunk.text}"


def run_hybrid_reranked(request: QueryRequest) -> QueryResult:
    semantic._validate_scope(request, "hybrid-reranked")
    settings = get_settings()
    try:
        validate_settings(settings)
    except RetrievalError as exc:
        raise semantic.to_http_error(exc) from None

    hybrid_request = request.model_copy(
        update={"pattern": Pattern.HYBRID, "limit": settings.rerank_candidate_limit}
    )
    hybrid = run_hybrid(hybrid_request)  # HTTP errors pass through unchanged
    candidates = hybrid.results

    ht = hybrid.trace
    trace = {
        "mode": "hybrid-reranked",
        "query": request.question,
        "filters": ht.get("filters"),
        "caller_id": ht.get("caller_id"),
        "result_count": 0,
        "hybrid": {k: ht.get(k) for k in
                   ("embedding", "semantic", "keyword", "fusion", "contribution", "unresolved_hits")},
        "rerank": {
            "model": settings.rerank_model_name,
            "candidate_limit": settings.rerank_candidate_limit,
            "send_limit": settings.rerank_send_limit,
            "return_limit": settings.rerank_return_limit,
            "candidates": len(candidates), "sent": 0, "returned": 0,
            "omitted_before": 0, "omitted_after": 0,
        },
    }
    if not candidates:
        return QueryResult(
            pattern="hybrid-reranked", status="no_results",
            message="No passages matched; nothing to re-rank. " + _SCORE_NOTE, trace=trace,
        )

    sent = candidates[: settings.rerank_send_limit]
    started = time.monotonic()
    try:
        data, tokens = call_rerank(settings, request.question, [_document(c) for c in sent])
    except RetrievalError as exc:
        raise semantic.to_http_error(exc) from None
    scores = {item["index"]: item["relevance_score"] for item in data}

    final, omitted = select(
        candidates, scores, settings.rerank_send_limit, settings.rerank_return_limit, request.limit
    )
    rr = trace["rerank"]
    rr.update({
        "sent": len(sent), "returned": len(final),
        "omitted_before": sum(1 for c in omitted if c.omitted_reason == NOT_SENT),
        "omitted_after": sum(1 for c in omitted if c.omitted_reason == BELOW_RETURN),
        "latency_ms": int((time.monotonic() - started) * 1000),
    })
    if tokens is not None:
        rr["usage_tokens"] = tokens
    trace["result_count"] = len(final)
    return QueryResult(
        pattern="hybrid-reranked", status="ok",
        message=f"Returned {len(final)} re-ranked passage(s). " + _SCORE_NOTE,
        trace=trace, results=final, omitted_candidates=omitted,
    )
