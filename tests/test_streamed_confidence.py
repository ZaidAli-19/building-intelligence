"""Offline tests for streamed answers and confidence (fake provider, no network)."""

import json

import pytest
from fastapi.testclient import TestClient

from building_with_rag import pipeline
from building_with_rag.app import create_app
from building_with_rag.contracts import QueryResult, RetrievedChunk
from building_with_rag.generation import answer


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1", section_id="bns:303", act="BNS_2023", heading="Theft", score=0.8,
        text="Whoever commits theft shall be punished with imprisonment up to three years.",
        section_number=303,
    )


GOOD = "Under BNS_2023, theft is punished with imprisonment up to three years [E1]."
BAD = "Theft is punished with death [E9]."
SUPPORTED = json.dumps({"verdicts": [{"claim": 0, "supported": True}]})


class FakeProvider:
    """Scripted attempts: each is a list of text pieces, or an Exception raised after them."""

    def __init__(self, attempts, verdict=SUPPORTED):
        self.attempts = list(attempts)
        self.verdict = verdict
        self.stream_calls = 0

    def stream(self, messages):
        self.stream_calls += 1
        script = self.attempts.pop(0)
        for piece in script:
            if isinstance(piece, Exception):
                raise piece
            yield piece

    def complete(self, messages):
        return self.verdict


@pytest.fixture()
def provider(monkeypatch):
    def install(attempts, verdict=SUPPORTED):
        fake = FakeProvider(attempts, verdict)
        monkeypatch.setattr(answer, "_stream_completion", fake.stream)
        monkeypatch.setattr(answer, "_complete", fake.complete)
        return fake
    return install


def _run(results=None):
    events = list(answer.stream_answer("What is theft?", results or [_chunk()]))
    final = events[-1][1]
    text = "".join(p for k, p in events if k == "text")
    return events, final, text


def test_passing_attempt_single_call_high(provider) -> None:
    fake = provider([[GOOD[:30], GOOD[30:]]])
    events, final, text = _run()
    assert fake.stream_calls == 1
    assert text == final.text == GOOD
    assert final.outcome == "answered" and final.confidence == "high"
    assert final.citations[0].section_id == "bns:303"
    assert [a.status for a in final.attempts] == ["passed"]
    assert events[0] == ("notice", answer.DRAFT_LINE)


def test_failed_attempt_then_passing_retry(provider) -> None:
    fake = provider([[BAD], [GOOD]])
    events, final, _ = _run()
    assert fake.stream_calls == 2
    assert [k for k, p in events if p == answer.DRAFT_LINE] == ["notice", "notice"]
    assert any("Retrying (attempt 2 of 2)" in str(p) for k, p in events if k == "notice")
    assert final.confidence == "high" and final.outcome == "answered"
    assert [a.status for a in final.attempts] == ["failed", "passed"]
    assert final.issues[0].attempt == 1 and final.issues[0].check == "citation_labels"


def test_both_attempts_fail_low_confidence(provider) -> None:
    provider([[BAD], [BAD]])
    _, final, _ = _run()
    assert final.outcome == "malformed" and final.confidence == "low"
    assert final.text == "" and final.draft_answer == BAD
    assert final.low_confidence_reason and "E9" in final.issues[-1].detail
    footer = pipeline.render_footer(final)
    assert "DRAFT — low confidence, not the final answer." in footer


def test_provider_failure_mid_stream(provider) -> None:
    provider([["Partial answer text that began streaming ", answer._Unavailable("boom")]])
    events, final, _ = _run()
    assert final.outcome == "unavailable" and final.confidence is None
    assert final.draft_answer.startswith("Partial answer")
    assert events[-2] == ("notice", answer.UNAVAILABLE_AFTER_TEXT)


def test_insufficient_evidence_not_streamed(provider) -> None:
    provider([["INSUFFICIENT_EVIDENCE: nothing about GST"]])
    events, final, _ = _run()
    assert final.outcome == "insufficient_evidence" and final.confidence is None
    assert all(k == "final" for k, _ in events)
    assert "nothing about GST" in pipeline.render_footer(final)


def test_chat_stream_ends_with_stop_and_done_after_failure(provider, monkeypatch) -> None:
    provider([["Partial answer text that began streaming ", answer._Unavailable("boom")]])
    retrieval = QueryResult(
        pattern="semantic", status="ok", message="ok", trace={}, results=[_chunk()]
    )
    monkeypatch.setattr(pipeline, "run_semantic", lambda request: retrieval)
    client = TestClient(create_app())
    with client.stream(
        "POST", "/v1/chat/completions",
        json={"model": "rag-semantic", "stream": True,
              "messages": [{"role": "user", "content": "What is theft?"}]},
    ) as response:
        assert response.status_code == 200
        raw = b"".join(response.iter_bytes()).decode()
    assert "unchecked draft" in raw
    assert '"finish_reason": "stop"' in raw
    assert raw.strip().endswith("data: " + "[" + "DONE" + "]")
