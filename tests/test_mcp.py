"""Offline tests for the MCP client mode (fake transport; no network)."""

import pytest
from fastapi import HTTPException

from building_with_rag.contracts import QueryRequest
from building_with_rag.pipeline import answerable
from building_with_rag.retrieval import mcp_client
from building_with_rag.settings import get_settings


def _section(sid="bns:103", n=103, text="passage"):
    return {"section_id": sid, "act": "BNS_2023", "act_label": "BNS", "status": "in_force",
            "chapter": "VI", "chapter_title": "t", "section_number": n, "heading": "Murder",
            "text": text, "text_available": bool(text), "needs_review": False}


@pytest.fixture()
def net(monkeypatch):
    monkeypatch.setattr(get_settings(), "mcp_server_url", "http://fake/mcp")
    monkeypatch.setattr(get_settings(), "mcp_deadline_seconds", "10")
    class Log(list):
        pass

    calls, replies = Log(), {}

    def transport(url, name, args, timeout):
        calls.append((name, args))
        reply = replies.get(name, {"status": "ok", "section": _section()})
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(mcp_client, "_transport", transport)
    calls.replies = replies
    return calls


def _run(q, **kw):
    return mcp_client.run_mcp(QueryRequest(question=q, pattern="mcp", **kw))


def test_validation_makes_no_call_and_traces_no_question(net):
    for q in ("section 103", "BNS section 1 and IPC section 2", "BNS sections 1 and 2"):
        r = _run(q)
        assert r.status == "clarify" and r.trace["call_count"] == 0
        assert q not in str(r.trace)
    assert net == []


def test_exact_lookup_one_call(net):
    r = _run("What does BNS section 103 say?")
    assert net == [("get_section", {"act": "bns", "section_number": 103})]
    assert r.status == "ready_for_synthesis" and r.trace["path"] == "exact_lookup"
    assert [c.section_id for c in r.results] == ["bns:103"] and r.results[0].act == "BNS_2023"
    assert answerable(r)


def test_not_found_and_missing_text(net):
    net.replies["get_section"] = {"status": "not_found", "section": None}
    r = _run("What does IPC section 4 say?")
    assert r.trace["stop_reason"] == "record_not_found" and r.results == [] and len(net) == 1
    net.replies["get_section"] = {"status": "ok", "section": _section("ipc:4", 4, "")}
    r = _run("What does IPC section 4 say?")
    assert r.trace["stop_reason"] == "section_text_missing" and r.results == []


def test_heading_path_two_calls_and_ambiguous(net):
    item = {"section_id": "bns:103", "section_number": 103, "heading": "Murder", "chapter": "VI"}
    net.replies["search_sections_by_heading"] = {"status": "ok", "items": [item], "total": 1}
    r = _run('Show the BNS section titled "Murder"')
    assert [c[0] for c in net] == ["search_sections_by_heading", "get_section"]
    assert r.status == "ready_for_synthesis" and r.trace["call_count"] == 2
    net.clear()
    net.replies["search_sections_by_heading"] = {"status": "ok", "total": 2, "items": [
        item, {**item, "section_id": "bns:104", "section_number": 104}]}
    r = _run('BNS heading "Murder"')
    assert r.status == "clarify" and r.trace["stop_reason"] == "ambiguous_heading" and len(net) == 1


def test_unsupported_and_allowlist(net):
    assert _run("What is the punishment for theft?").trace["stop_reason"] == "unsupported_question"
    run = mcp_client._Run("u", 5.0)
    with pytest.raises(mcp_client._Stop):
        run.call("drop_collection", {"act": "bns"})
    assert net == []


def test_deadline_and_errors(net, monkeypatch):
    net.replies["get_section"] = TimeoutError()
    assert _run("BNS section 103").trace["stop_reason"] == "deadline_exceeded"
    net.replies["get_section"] = ConnectionError()
    with pytest.raises(HTTPException) as e:
        _run("BNS section 103")
    assert e.value.status_code == 502 and e.value.detail["code"] == "mcp_upstream_error"
    net.replies["get_section"] = {"status": "ok", "section": _section("ipc:103")}
    with pytest.raises(HTTPException) as e:
        _run("BNS section 103")
    assert e.value.detail["code"] == "mcp_bad_response"
    ticks = iter([0.0, 99.0, 99.0])
    monkeypatch.setattr(mcp_client, "_now", lambda: next(ticks))
    net.replies["get_section"] = {"status": "ok", "section": _section()}
    assert _run("BNS section 103").trace["stop_reason"] == "deadline_exceeded"


def test_unset_url_is_503(monkeypatch):
    monkeypatch.setattr(get_settings(), "mcp_server_url", "")
    with pytest.raises(HTTPException) as e:
        _run("BNS section 103")
    assert e.value.status_code == 503


def _range_reply(nums, text="passage"):
    return {"status": "ok", "items": [_section(f"bns:{n}", n, text) for n in nums]}


def test_range_path(net):
    net.replies["get_sections_in_range"] = _range_reply(range(10, 20))
    r = _run("What do BNS sections 10 to 19 say?")
    assert net == [("get_sections_in_range", {"act": "bns", "start": 10, "end": 19})]
    assert r.status == "ready_for_synthesis" and r.trace["path"] == "range_lookup"
    assert len(r.results) == 10 and answerable(r)
    net.replies["get_sections_in_range"] = _range_reply([10, 11, 13])
    r = _run("Show BNS sections 10-13")
    assert r.status == "partial_result" and r.trace["stop_reason"] == "range_partial"
    assert "12" in r.message and len(r.results) == 3


def test_range_rejected_without_call(net):
    for q, reason in (("BNS sections 19 to 10", "invalid_range"),
                      ("BNS sections 1 to 100", "range_too_large"),
                      ("sections 10 to 19", "ambiguous_request"),
                      ("BNS section 1 to IPC section 5", "ambiguous_request")):
        assert _run(q).trace["stop_reason"] == reason
    assert net == []
    net.replies["get_sections_in_range"] = _range_reply([10, 99])
    with pytest.raises(HTTPException):
        _run("BNS sections 10 to 19")


def test_chapter_and_section_listing(net):
    net.replies["list_chapters"] = {"status": "ok", "chapters": [
        {"chapter": "I", "chapter_title": "PRELIMINARY", "first_section": 1, "last_section": 3,
         "section_count": 3}]}
    r = _run("What chapters does the BNS have?")
    assert net == [("list_chapters", {"act": "bns"})]
    assert r.status == "listing" and r.trace["path"] == "chapter_list" and r.results == []
    assert "PRELIMINARY" in r.message and not answerable(r)
    net.clear()
    net.replies["list_sections"] = {"status": "ok", "total": 358, "items": [
        {"section_id": f"bns:{n}", "section_number": n, "heading": "h", "chapter": "I"}
        for n in (1, 2, 3)]}
    r = _run("List the first 3 BNS sections")
    assert net == [("list_sections", {"act": "bns", "limit": 3, "offset": 0})]
    assert r.status == "listing" and r.trace["path"] == "section_list" and "bns:3" in r.message
    assert _run("List the first 99 BNS sections").trace["stop_reason"] == "invalid_limit"
    assert _run("What chapters are there?").trace["stop_reason"] == "unsupported_question"
    net.replies["list_sections"] = {"status": "ok", "items": [
        {"section_id": "ipc:1", "section_number": 1, "heading": "h"}]}
    with pytest.raises(HTTPException):
        _run("List the first 1 BNS sections")


def test_pdf_only_on_slash_pdf(net):
    from fastapi.testclient import TestClient

    from building_with_rag import export_pdf
    from building_with_rag.app import create_app

    net.replies["get_sections_in_range"] = _range_reply(range(103, 106))
    r = _run("What do BNS sections 103 to 105 say?")
    assert r.status == "ready_for_synthesis" and "pdf" not in r.trace  # no PDF without /pdf
    r = _run("What do BNS sections 103 to 105 say? /pdf")
    assert r.status == "pdf_ready" and r.trace["pdf"] == "ready" and len(r.results) == 3
    assert not answerable(r) and "/v1/exports/" in r.message and "/pdf" not in r.trace.values()
    export_id = r.message.split("/v1/exports/")[1].split(".pdf")[0]
    client = TestClient(create_app())
    pdf = client.get(f"/v1/exports/{export_id}.pdf")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    assert export_pdf.get(export_id) is not None
    assert client.get("/v1/exports/" + "0" * 32 + ".pdf").status_code == 404
    r = _run("section 103 /pdf")  # nothing to export: no PDF, normal clarify
    assert r.status == "clarify" and r.trace["pdf"] == "skipped"
