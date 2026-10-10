"""Offline tests for the bounded agentic controller (fake adapters; no network)."""

import json

import pytest
from fastapi import HTTPException

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.pipeline import answer_events, answerable
from building_with_rag.retrieval import agentic
from building_with_rag.settings import get_settings


def _chunk(cid="bns:303-0", text="passage"):
    return RetrievedChunk(chunk_id=cid, section_id=cid.split("-")[0], act="BNS_2023",
                          text=text, heading="h", score=0.9)


def _res(pattern, status="ok", chunks=None, **extra):
    return QueryResult(pattern=pattern, status=status, message="m", trace={"secret": "x"},
                       results=chunks if chunks is not None else [_chunk()], **extra)


@pytest.fixture()
def calls(monkeypatch):
    """Record (tool, request) for each adapter call; behaviour set per test via .reply."""
    class Log(list):
        pass

    log = Log()
    replies = log.replies = {}

    def make(name, pattern):
        def adapter(request):
            log.append((name, request))
            reply = replies.get(name, _res(pattern))
            if isinstance(reply, Exception):
                raise reply
            return reply
        return adapter, pattern

    for name, pattern in (("structuredLookup", "structured"), ("semanticRetrieve", "semantic"),
                          ("hydeRetrieve", "hyde")):
        monkeypatch.setitem(agentic._TOOLS, name, make(name, pattern))
    return log


def _run(question, **kw):
    return agentic.run_agentic(QueryRequest(question=question, pattern="agentic", **kw))


def test_validation_rejects_without_tool_calls_or_question_text(calls) -> None:
    cases = {
        "section 103": "bare_section_number",
        "BNS and IPC section 103": "ambiguous_request",
        "BNS section 1 and section 2": "ambiguous_request",
    }
    for question, reason in cases.items():
        r = _run(question)
        assert r.status == "clarify" and r.results == [] and r.trace["stop_reason"] == reason
        assert r.trace["call_count"] == 0 and r.trace["steps"] == []
        assert question.strip() not in json.dumps(r.trace)
    assert calls == []


def test_exact_lookup_calls_structured_once(calls) -> None:
    calls.replies["structuredLookup"] = _res("structured", chunks=[_chunk("bns:103")])
    r = _run("What does BNS section 103 say?")
    assert [c[0] for c in calls] == ["structuredLookup"]
    assert r.status == "ready_for_synthesis" and r.trace["path"] == "exact_lookup"
    assert r.trace["call_count"] == 1 and [c.section_id for c in r.results] == ["bns:103"]


def test_missing_record_is_partial_with_no_fallback(calls) -> None:
    calls.replies["structuredLookup"] = _res("structured", "not_found", [])
    r = _run("What does IPC section 4 say?")
    assert r.status == "partial_result" and r.trace["stop_reason"] == "record_not_found"
    assert r.results == [] and [c[0] for c in calls] == ["structuredLookup"]
    assert not answerable(r)
    assert list(answer_events("q", r)) == [("text", r.message), ("final", None)]


def test_ordinary_question_calls_semantic_once(calls) -> None:
    r = _run("What is the punishment for theft?")
    assert [c[0] for c in calls] == ["semanticRetrieve"]
    assert r.status == "ready_for_synthesis" and r.trace["path"] == "semantic"
    assert r.trace["steps"][0]["adequate"] is True and answerable(r)


def test_vocabulary_path_uses_hyde_only_when_direct_inadequate(calls) -> None:
    q = "What happens to someone who secretly takes a neighbour's bike?"
    r = _run(q)
    assert [c[0] for c in calls] == ["semanticRetrieve"] and r.trace["path"] == "vocab_mismatch"
    calls.clear()
    calls.replies["semanticRetrieve"] = _res("semantic", "no_results", [])
    calls.replies["hydeRetrieve"] = _res(
        "hyde", chunks=[_chunk()], hyde_hypothetical_text_debug="HYPOTHETICAL",
        hyde_direct_candidates=[_chunk()], hyde_query_candidates=[_chunk()])
    r = _run(q)
    assert [c[0] for c in calls] == ["semanticRetrieve", "hydeRetrieve"]
    assert r.trace["call_count"] == 2 and r.trace["hyde_calls"] == 1
    assert r.status == "ready_for_synthesis"
    assert r.hyde_hypothetical_text_debug is None
    assert r.hyde_direct_candidates == [] and r.hyde_query_candidates == []
    assert "HYPOTHETICAL" not in r.model_dump_json() and "secret" not in json.dumps(r.trace)
    assert q not in json.dumps(r.trace)


def test_both_inadequate_is_clarify(calls) -> None:
    calls.replies["semanticRetrieve"] = _res("semantic", "no_results", [])
    calls.replies["hydeRetrieve"] = _res("hyde", "hyde_unavailable", [])
    r = _run("someone tries to grab my bag")
    assert r.status == "clarify" and r.trace["stop_reason"] == "no_usable_evidence"
    assert r.trace["call_count"] == 2 and r.results == []


def test_scope_is_propagated_to_every_call(calls) -> None:
    calls.replies["semanticRetrieve"] = _res("semantic", "no_results", [])
    _run("someone tries to grab my bag", filters={"act": ["IPC_1860"]}, limit=3)
    assert len(calls) == 2
    for _, request in calls:
        assert request.filters.act == ["IPC_1860"] and request.limit == 3


def test_deadline_expiry_stops_with_reason(calls, monkeypatch) -> None:
    clock = iter([0.0, 100.0, 100.0, 100.0])
    monkeypatch.setattr(agentic, "_now", lambda: next(clock))
    r = _run("What is the punishment for theft?")
    assert r.trace["stop_reason"] == "deadline_exceeded" and r.trace["call_count"] == 0
    assert calls == []


def test_tool_error_and_bad_result(calls) -> None:
    calls.replies["semanticRetrieve"] = HTTPException(502, detail={"code": "x", "message": "secret"})
    with pytest.raises(HTTPException) as exc:
        _run("What is the punishment for theft?")
    assert exc.value.status_code == 502
    calls.replies["semanticRetrieve"] = _res("hyde")  # wrong pattern
    r = _run("What is the punishment for theft?")
    assert r.status == "partial_result" and r.trace["stop_reason"] == "tool_error"
    assert r.trace["steps"][0]["tool_status"] == "bad_result"


def test_invalid_deadline_setting_is_503(calls, monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "agentic_deadline_seconds", "0")
    with pytest.raises(HTTPException) as exc:
        _run("What is the punishment for theft?")
    assert exc.value.status_code == 503 and calls == []


def test_answerable_only_for_real_passages() -> None:
    def r(status, n):
        return QueryResult(pattern="agentic", status=status, message="m", trace={},
                           results=[_chunk()] * n)
    assert answerable(r("ready_for_synthesis", 1)) and answerable(r("partial_result", 1))
    assert not answerable(r("clarify", 1)) and not answerable(r("partial_result", 0))


def test_vocab_direct_step_needs_a_legal_term(calls) -> None:
    calls.replies["semanticRetrieve"] = _res("semantic", chunks=[_chunk(text="about taking things")])
    calls.replies["hydeRetrieve"] = _res("hyde", chunks=[_chunk("bns:303-0", "Theft is ...")])
    r = _run("What happens to someone who secretly takes a neighbours bike?")
    assert [c[0] for c in calls] == ["semanticRetrieve", "hydeRetrieve"]
    assert r.trace["steps"][0]["adequate"] is False and r.status == "ready_for_synthesis"
    calls.clear()
    calls.replies["semanticRetrieve"] = _res("semantic", chunks=[_chunk(text="Theft is ...")])
    _run("What happens to someone who secretly takes a neighbours bike?")
    assert [c[0] for c in calls] == ["semanticRetrieve"]
