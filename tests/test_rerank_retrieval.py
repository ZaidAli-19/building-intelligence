"""Offline tests for hybrid re-ranking (no network)."""

import httpx
import pytest
from fastapi import HTTPException

from building_with_rag import pipeline
from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.generation.context import build_context
from building_with_rag.retrieval import rerank
from building_with_rag.settings import Settings


def _cand(n: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"c{i}", section_id=f"bns:{i}", act="BNS_2023", text=f"text {i}",
            heading=f"h{i}", score=1.0 / i, fused_score=1.0 / i, fused_rank=i,
            semantic_rank=i, keyword_rank=None,
        )
        for i in range(1, n + 1)
    ]


def _settings(**over) -> Settings:
    base = dict(
        rerank_api_key="k", rerank_api_base_url="https://rerank.invalid/v1",
        rerank_model_name="m", rerank_request_timeout_seconds=5,
        rerank_candidate_limit=4, rerank_send_limit=3, rerank_return_limit=2,
    )
    return Settings(_env_file=None, **{**base, **over})


def _request(limit: int = 5) -> QueryRequest:
    return QueryRequest(question="q", pattern="hybrid-reranked", limit=limit)


@pytest.fixture()
def env(monkeypatch):
    calls = {"hybrid": 0, "rerank": 0}
    state = {"settings": _settings(), "candidates": _cand(4), "scores": {0: 0.1, 1: 0.9, 2: 0.5}}

    def fake_hybrid(request):
        calls["hybrid"] += 1
        assert request.limit == state["settings"].rerank_candidate_limit
        return QueryResult(
            pattern="hybrid", status="ok", message="", trace={"filters": {}, "caller_id": "x"},
            results=state["candidates"],
        )

    def fake_call(settings, query, documents):
        calls["rerank"] += 1
        return [{"index": i, "relevance_score": s} for i, s in state["scores"].items()], 7

    monkeypatch.setattr(rerank, "get_settings", lambda: state["settings"])
    monkeypatch.setattr(rerank, "run_hybrid", fake_hybrid)
    monkeypatch.setattr(rerank, "call_rerank", fake_call)
    return calls, state


def test_reorders_and_keeps_fused_fields(env) -> None:
    result = rerank.run_hybrid_reranked(_request())
    assert result.status == "ok"
    assert [r.chunk_id for r in result.results] == ["c2", "c3"]
    assert [r.rerank_rank for r in result.results] == [1, 2]
    assert all(r.score == r.rerank_score for r in result.results)
    assert result.results[0].fused_rank == 2 and result.results[0].fused_score == 0.5
    assert result.trace["rerank"]["usage_tokens"] == 7


def test_omitted_before_and_after(env) -> None:
    result = rerank.run_hybrid_reranked(_request())
    by_id = {c.chunk_id: c for c in result.omitted_candidates}
    assert [c.chunk_id for c in result.omitted_candidates] == ["c1", "c4"]  # fused order
    assert by_id["c4"].omitted_reason == "not_sent_to_reranker"
    assert by_id["c4"].rerank_score is None and by_id["c4"].rerank_rank is None
    assert by_id["c1"].omitted_reason == "below_return_limit"
    assert by_id["c1"].rerank_rank == 3 and by_id["c1"].rerank_score == 0.1
    rr = result.trace["rerank"]
    assert rr["returned"] + rr["omitted_before"] + rr["omitted_after"] == rr["candidates"]


def test_empty_key_is_503_and_nothing_called(env) -> None:
    calls, state = env
    state["settings"] = _settings(rerank_api_key="")
    with pytest.raises(HTTPException) as exc:
        rerank.run_hybrid_reranked(_request())
    assert exc.value.status_code == 503 and exc.value.detail["code"] == "retrieval_not_ready"
    assert "RERANK_API_KEY" in exc.value.detail["message"]
    assert calls == {"hybrid": 0, "rerank": 0}


def test_invalid_limits_is_503(env) -> None:
    _, state = env
    state["settings"] = _settings(rerank_send_limit=9)  # SEND > CANDIDATE
    with pytest.raises(HTTPException) as exc:
        rerank.run_hybrid_reranked(_request())
    assert exc.value.status_code == 503 and "RERANK_" in exc.value.detail["message"]


@pytest.mark.parametrize("reply", [
    {"data": [{"index": 0, "relevance_score": 0.1}, {"index": 0, "relevance_score": 0.2}]},  # duplicate
    {"data": [{"index": 9, "relevance_score": 0.1}]},  # out of range
    {"data": [{"index": 0}]},  # missing score
    {"data": []},
])
def test_invalid_reply_is_502(monkeypatch, reply) -> None:
    class Resp:
        def raise_for_status(self): ...
        def json(self): return reply

    monkeypatch.setattr(rerank.httpx, "post", lambda *a, **k: Resp())
    with pytest.raises(rerank.RetrievalError) as exc:
        rerank.call_rerank(_settings(), "q", ["a", "b"])
    assert exc.value.status_code == 502 and exc.value.code == "retrieval_upstream_error"


@pytest.mark.parametrize("failure", ["timeout", "status"])
def test_provider_failure_is_502_no_results(monkeypatch, failure) -> None:
    def post(*a, **k):
        if failure == "timeout":
            raise httpx.ReadTimeout("slow")
        raise httpx.HTTPStatusError("bad", request=httpx.Request("POST", "https://x"),
                                    response=httpx.Response(500))

    monkeypatch.setattr(rerank.httpx, "post", post)
    with pytest.raises(rerank.RetrievalError) as exc:
        rerank.call_rerank(_settings(), "q", ["a"])
    assert exc.value.status_code == 502
    assert "rerank.invalid" not in exc.value.message


def test_context_uses_only_final_results_and_routing(env, monkeypatch) -> None:
    result = rerank.run_hybrid_reranked(_request())
    entries, _, _ = build_context(result.results)
    assert [e["chunk_id"] for e in entries] == ["c2", "c3"]
    monkeypatch.setattr(pipeline, "run_hybrid_reranked", lambda request: result)
    assert pipeline.retrieve(_request()) is result
