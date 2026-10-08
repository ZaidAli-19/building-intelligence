"""Offline tests for structured exact retrieval (fake `sections` collection, no network)."""

import pytest
from fastapi import HTTPException

from building_with_rag import pipeline
from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.generation.context import build_context
from building_with_rag.retrieval import semantic, structured
from building_with_rag.retrieval.structured import classify


@pytest.mark.parametrize("question,act,number", [
    ("What does BNS section 103 say?", "BNS_2023", 103),
    ("IPC sec. 302 punishment", "IPC_1860", 302),
    ("Indian Penal Code section 420", "IPC_1860", 420),
])
def test_exact_lookup_classified(question, act, number) -> None:
    s = classify(question)
    assert (s.status, s.act, s.section_number) == ("ok", act, number)


@pytest.mark.parametrize("question", [
    "What does section 103 say?",
    "Compare BNS section 103 and IPC section 302 with section 103",
    "BNS section 103 and section 104",
    "BNS section 103A",
])
def test_unclear_requests_need_clarification(question) -> None:
    s = classify(question)
    assert s.status == "clarification_needed"


def test_aggregation_filter_and_open_question_are_recommendations() -> None:
    assert classify("How many sections are in the BNS?").intent == "aggregation"
    assert classify("List all sections in chapter XVII").intent == "filter"
    open_q = classify("What is the punishment for theft?")
    assert open_q.status == "recommendation" and open_q.intent is None


class FakeSections:
    def __init__(self, doc=None):
        self.doc, self.calls = doc, []

    def find_one(self, predicate, projection=None):
        self.calls.append(predicate)
        return self.doc


class FakeDb:
    def __init__(self, sections):
        self.sections = sections

    def __getitem__(self, name):
        assert name == "sections"
        return self.sections


DOC = {
    "section_id": "bns:103", "act": "BNS_2023", "act_label": "Bharatiya Nyaya Sanhita, 2023",
    "status": "in_force", "chapter": "VI", "chapter_title": "T", "section_number": 103,
    "heading": "Punishment for murder", "text": "Whoever commits murder ...",
    "source_pdf": "p.pdf", "source_sha256": "x", "source_status_version": "v1", "needs_review": False,
}


def _run(monkeypatch, question, doc=DOC, **extra):
    sections = FakeSections(doc)
    monkeypatch.setattr(semantic, "get_db", lambda: FakeDb(sections))
    request = QueryRequest(question=question, pattern="structured", **extra)
    return structured.run_structured(request), sections


def test_predicate_has_only_validated_fields(monkeypatch) -> None:
    result, sections = _run(monkeypatch, 'BNS section 103 {"$ne": null}', chapter="VI")
    assert result.status == "ok"
    assert sections.calls == [{
        "act": "BNS_2023", "section_number": 103,
        "access_level": {"$in": ["public"]}, "chapter": "VI",
    }]
    assert "$ne" not in str(sections.calls)
    chunk = result.results[0]
    assert chunk.chunk_id == "bns:103" and chunk.score == 1.0
    assert result.trace["mongodb_called"] is True
    assert result.trace["record"]["source_status_version"] == "v1"


def test_status_filter_narrows_predicate(monkeypatch) -> None:
    _, sections = _run(monkeypatch, "BNS section 103", filters={"status": ["in_force"]})
    assert sections.calls[0]["status"] == {"$in": ["in_force"]}


def test_non_ok_signals_make_no_collection_call(monkeypatch) -> None:
    for q in ("What does section 103 say?", "How many sections are there?", "theft penalty"):
        result, sections = _run(monkeypatch, q)
        assert result.status in ("clarification_needed", "recommendation")
        assert result.results == [] and result.trace["mongodb_called"] is False
        assert sections.calls == []


def test_missing_record_is_not_found(monkeypatch) -> None:
    result, _ = _run(monkeypatch, "IPC section 4", doc=None)
    assert result.status == "not_found" and result.results == []
    assert result.trace["mongodb_called"] is True


def test_scope_and_chapter_validation(monkeypatch) -> None:
    for extra in ({"required_acts": ["BNS_2023"]}, {"caller_id": "someone-else"}):
        with pytest.raises(HTTPException) as exc:
            _run(monkeypatch, "BNS section 103", **extra)
        assert exc.value.status_code == 422
    with pytest.raises(HTTPException) as exc:
        _run(monkeypatch, "BNS section 103", chapter='{"$ne": 1}')
    assert exc.value.detail["code"] == "unsupported_option"


def test_pipeline_routes_and_context(monkeypatch) -> None:
    result, _ = _run(monkeypatch, "BNS section 103")
    monkeypatch.setattr(pipeline, "run_structured", lambda request: result)
    request = QueryRequest(question="BNS section 103", pattern="structured")
    assert pipeline.retrieve(request) is result
    entries, _, _ = build_context(result.results)
    assert [e["label"] for e in entries] == ["E1"]

    unclear = QueryResult(pattern="structured", status="clarification_needed",
                          message="Which act?", trace={})
    events = list(pipeline.answer_events("q", unclear))
    assert events == [("text", "Which act?"), ("final", None)]
