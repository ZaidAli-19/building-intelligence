"""Offline tests for decomposition and HyDE (fake provider and MongoDB; no network)."""

import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from building_with_rag.app import create_app
from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.pipeline import answer_events, answerable, retrieve
from building_with_rag.retrieval import decomposition, expansion, hyde, semantic
from building_with_rag.settings import get_settings


def _chunk(cid, act="BNS_2023", score=0.9):
    return RetrievedChunk(chunk_id=cid, section_id=cid.replace("-0", ""), act=act,
                          text="t " + cid, heading="h", score=score)


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "generation_api_base_url", "http://fake")
    monkeypatch.setattr(settings, "generation_api_key", "k")
    monkeypatch.setattr(semantic, "_clients", lambda: (object(), object()))
    monkeypatch.setattr(semantic, "_ensure_ready", lambda db: None)


def _replies(monkeypatch, *replies):
    queue = list(replies)
    monkeypatch.setattr(expansion, "generation_step", lambda system, user: queue.pop(0))


def _search(monkeypatch, table):
    monkeypatch.setattr(expansion, "search", lambda db, v, text, f, limit: table[text])


def _req(pattern, q="compare BNS and IPC murder"):
    return QueryRequest(question=q, pattern=pattern)


@pytest.mark.parametrize("reply", [
    "", "not json", "[]", '{"subquestions": []}', '{"subquestions": ["a", "A "]}',
    '{"subquestions": ["a","b","c","d"]}', '{"subquestions": [""]}', '{"subquestions": [1]}',
])
def test_invalid_decomposition_is_clarify(monkeypatch, reply) -> None:
    _replies(monkeypatch, reply)
    result = decomposition.run_decomposition(_req("decomposition"))
    assert result.status == "clarify" and result.results == [] and result.subquestions == []
    assert not answerable(result)


def test_valid_decomposition_ok_with_deduped_union(monkeypatch) -> None:
    shared = _chunk("bns:103-0")
    _replies(monkeypatch, '{"subquestions": ["q1", "q2"]}',
             '{"steps": [{"index": 0, "sufficient": true}, {"index": 1, "sufficient": true}]}')
    _search(monkeypatch, {
        "q1": [shared, _chunk("bns:104-0")],
        "q2": [shared, _chunk("ipc:302-0", "IPC_1860")],
    })
    result = decomposition.run_decomposition(_req("decomposition"))
    assert result.status == "ok"
    assert [s.status for s in result.subquestions] == ["evidenced", "evidenced"]
    # interleaved by step, deduplicated by chunk_id
    assert [c.chunk_id for c in result.results] == ["bns:103-0", "bns:104-0", "ipc:302-0"]


def test_step_without_evidence_is_partial_never_ok(monkeypatch) -> None:
    _replies(monkeypatch, '{"subquestions": ["q1", "q2"]}',
             '{"steps": [{"index": 0, "sufficient": true}]}')
    _search(monkeypatch, {"q1": [_chunk("bns:103-0")], "q2": []})
    result = decomposition.run_decomposition(_req("decomposition"))
    assert result.status == "partial_answer" and "q2" in result.message
    assert result.subquestions[1].status == "no_evidence" and result.subquestions[1].reason
    assert [c.chunk_id for c in result.results] == ["bns:103-0"]
    assert answerable(result)


def test_all_steps_insufficient_is_clarify(monkeypatch) -> None:
    _replies(monkeypatch, '{"subquestions": ["q1"]}',
             '{"steps": [{"index": 0, "sufficient": false, "reason": "off topic"}]}')
    _search(monkeypatch, {"q1": [_chunk("bns:1-0")]})
    result = decomposition.run_decomposition(_req("decomposition"))
    assert result.status == "clarify" and result.results == []
    assert not answerable(result)


def test_invalid_sufficiency_reply_is_502(monkeypatch) -> None:
    _replies(monkeypatch, '{"subquestions": ["q1"]}', "garbage")
    _search(monkeypatch, {"q1": [_chunk("bns:1-0")]})
    with pytest.raises(HTTPException) as exc:
        decomposition.run_decomposition(_req("decomposition"))
    assert exc.value.status_code == 502


@pytest.mark.parametrize("reply", ["", "   ", "NOT_POSSIBLE", "x" * 1600])
def test_hyde_bad_text_is_unavailable_without_search(monkeypatch, reply) -> None:
    _replies(monkeypatch, reply)
    monkeypatch.setattr(expansion, "search", lambda *a: pytest.fail("searched"))
    result = hyde.run_hyde(_req("hyde"))
    assert result.status == "hyde_unavailable" and result.results == []
    assert result.hyde_hypothetical_text_debug is None


def test_hyde_hypothetical_text_only_in_debug_field(monkeypatch) -> None:
    secret = "A person who dishonestly moves property commits a taking."
    _replies(monkeypatch, secret)
    _search(monkeypatch, {
        "bike q": [_chunk("bns:303-0", score=0.5)],
        secret: [_chunk("bns:303-0", score=0.8), _chunk("ipc:378-0", "IPC_1860", 0.7)],
    })
    result = hyde.run_hyde(_req("hyde", "bike q"))
    assert result.status == "ok"
    assert [c.chunk_id for c in result.results] == ["bns:303-0", "ipc:378-0"]
    assert result.results[0].score == 0.8
    assert result.hyde_hypothetical_text_debug == secret
    dumped = result.model_dump()
    dumped.pop("hyde_hypothetical_text_debug")
    assert secret not in json.dumps(dumped)
    assert len(result.hyde_direct_candidates) == 1 and len(result.hyde_query_candidates) == 2


@pytest.mark.parametrize("pattern,runner", [
    ("decomposition", decomposition.run_decomposition), ("hyde", hyde.run_hyde),
])
def test_missing_generation_key_is_503_before_provider(monkeypatch, pattern, runner) -> None:
    monkeypatch.setattr(get_settings(), "generation_api_key", "")
    monkeypatch.setattr(expansion.httpx, "post", lambda *a, **k: pytest.fail("provider called"))
    with pytest.raises(HTTPException) as exc:
        runner(_req(pattern))
    assert exc.value.status_code == 503 and "GENERATION_API_KEY" in exc.value.detail["message"]


def test_pipeline_routes_and_skips_model_on_non_answerable(monkeypatch) -> None:
    called = []

    def fake(pattern, status):
        def run(request):
            called.append(pattern)
            return QueryResult(pattern=pattern, status=status, message="m", trace={})
        return run

    monkeypatch.setattr("building_with_rag.pipeline.run_decomposition", fake("decomposition", "clarify"))
    monkeypatch.setattr("building_with_rag.pipeline.run_hyde", fake("hyde", "hyde_unavailable"))
    for pattern in ("decomposition", "hyde"):
        result = retrieve(_req(pattern))
        assert list(answer_events("q", result)) == [("text", "m"), ("final", None)]
    assert called == ["decomposition", "hyde"]


def test_partial_answer_prefix_names_missing_part(monkeypatch) -> None:
    monkeypatch.setattr("building_with_rag.pipeline.stream_answer", lambda q, r: iter([("final", None)]))
    result = QueryResult(pattern="decomposition", status="partial_answer",
                         message="Partial: no evidence for q2.", trace={})
    first = next(iter(answer_events("q", result)))
    assert first[0] == "text" and "q2" in first[1]


def test_models_and_missing_config_chat_error(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "generation_api_key", "")
    client = TestClient(create_app())
    assert len(client.get("/v1/models").json()["data"]) == 7
    resp = client.post("/v1/chat/completions", json={
        "model": "rag-hyde", "messages": [{"role": "user", "content": "q"}]})
    assert resp.status_code == 503
