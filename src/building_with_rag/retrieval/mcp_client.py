"""MCP client mode: a deterministic controller over two allowlisted tools of the separate
Legal Sections MCP server. No LLM planner, no discovery, no fallback to another mode.

The server's section text is the only evidence and is treated as data, never instructions.
The trace never holds question text, credentials or the server URL.
"""

import asyncio
import json
import re
import time

from fastapi import HTTPException

from building_with_rag import export_pdf
from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.retrieval import semantic, structured
from building_with_rag.settings import get_settings

MAX_CALLS = 2
DEADLINE_RANGE = (1.0, 60.0)
_TOOLS = frozenset({
    "get_section", "search_sections_by_heading", "get_sections_in_range",
    "list_chapters", "list_sections",
})
_ACT_TO_TOOL = {"BNS_2023": "bns", "IPC_1860": "ipc"}
_TOOL_TO_ACT = {v: k for k, v in _ACT_TO_TOOL.items()}
_SECTION_ID = re.compile(r"^(bns|ipc):(\d+)$")
_QUOTED = re.compile(r"[\"“]([^\"”]{2,80})[\"”]")
_SEARCH_LIMIT = 5
MAX_RANGE_SPAN = 25  # same cap as the server
MAX_SECTION = 511
_RANGE = re.compile(
    r"(?:\bsections?\b|\bsec\b\.?|\bs\.|§)\s*(\d+)\s*(?:to|through|until|-|–|—)\s*(\d+)\b",
    re.IGNORECASE,
)
_PDF_CMD = re.compile(r"(?:^|\s)/pdf\s*$", re.IGNORECASE)  # only an explicit trailing /pdf
_CHAPTERS = re.compile(r"\bchapters?\b", re.IGNORECASE)
_LISTING = re.compile(r"\b(?:list|show|give)\b.*\bsections\b", re.IGNORECASE)
_FIRST_N = re.compile(r"\bfirst\s+(\d{1,3})\b", re.IGNORECASE)
LIST_DEFAULT, LIST_MAX = 20, 50  # list_sections page size (server cap is 50)
_CLARIFY = (
    "rag-mcp answers questions that name one act (BNS or IPC) and one section number, "
    "one act plus a section range (e.g. BNS sections 10 to 19, at most 25), "
    "one act plus a quoted section heading, or ask for an act's chapters or its first N "
    "sections (e.g. List the first 10 BNS sections)."
)

_now = time.monotonic  # replaceable clock


class _Stop(Exception):
    def __init__(self, reason: str):
        self.reason = reason


def _http(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _settings() -> tuple[str, float]:
    s = get_settings()
    if not s.mcp_server_url.strip():
        raise _http(503, "retrieval_not_ready", "MCP_SERVER_URL is not set.")
    try:
        deadline = float(s.mcp_deadline_seconds)
    except (TypeError, ValueError):
        deadline = -1.0
    if not DEADLINE_RANGE[0] <= deadline <= DEADLINE_RANGE[1]:
        raise _http(503, "retrieval_not_ready", "MCP_DEADLINE_SECONDS must be a number from 1 to 60.")
    return s.mcp_server_url.strip(), deadline


def _payload(result) -> dict:
    """Tool result -> envelope dict (structured content, else JSON text)."""
    data = getattr(result, "structuredContent", None)
    if data is None:
        texts = [c.text for c in result.content if getattr(c, "type", "") == "text"]
        data = json.loads(texts[0]) if texts else None
    if isinstance(data, dict) and set(data) == {"result"} and isinstance(data["result"], dict):
        data = data["result"]
    if not isinstance(data, dict):
        raise TypeError("not an object")
    return data


async def _call_async(url: str, name: str, args: dict) -> dict:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url) as (read, write, _), ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool(name, args)
    if result.isError:
        raise ValueError("tool error")
    return _payload(result)


def _transport(url: str, name: str, args: dict, timeout: float) -> dict:
    """The single network seam (tests replace it). Raises TimeoutError, ConnectionError, ValueError."""
    try:
        return asyncio.run(asyncio.wait_for(_call_async(url, name, args), timeout))
    except TimeoutError:
        raise
    except (ValueError, TypeError):
        raise ValueError("bad response") from None
    except Exception:  # noqa: BLE001 - never surface driver/URL details
        raise ConnectionError("mcp request failed") from None


class _Run:
    def __init__(self, url: str, deadline: float):
        self.url = url
        self.deadline = deadline
        self.started = _now()
        self.steps: list[dict] = []
        self.calls = 0

    def call(self, name: str, args: dict) -> dict:
        if name not in _TOOLS:
            raise _Stop("tool_error")
        if self.calls >= MAX_CALLS:
            raise _Stop("call_budget_exhausted")
        remaining = self.deadline - (_now() - self.started)
        if remaining <= 0:
            raise _Stop("deadline_exceeded")
        self.calls += 1
        step = {"tool": name, "tool_status": "error", "result_count": 0}
        self.steps.append(step)
        try:
            data = _transport(self.url, name, args, remaining)
        except TimeoutError:
            step["tool_status"] = "timeout"
            raise _Stop("deadline_exceeded") from None
        except ConnectionError:
            step["tool_status"] = "unreachable"
            raise _http(502, "mcp_upstream_error", "MCP server request failed.") from None
        except ValueError:
            step["tool_status"] = "bad_result"
            raise _http(502, "mcp_bad_response", "MCP server returned an unexpected response.") from None
        status = data.get("status")
        step["tool_status"] = str(status)[:30]
        if status == "upstream_error":
            raise _http(502, "mcp_upstream_error", "MCP server could not reach its database.")
        if status not in ("ok", "not_found"):  # includes validation_error: a client bug
            raise _http(502, "mcp_bad_response", "MCP server returned an unexpected response.")
        return data

    def trace(self, validation: str, path: str, reason: str, count: int) -> dict:
        return {
            "mode": "mcp", "validation": validation, "path": path, "steps": self.steps,
            "call_count": self.calls, "deadline_seconds": self.deadline,
            "elapsed_ms": int((_now() - self.started) * 1000), "stop_reason": reason,
            "caller_id": get_settings().webui_demo_caller_id, "result_count": count,
        }


def _out(run, status, message, validation, path, reason, results=None) -> QueryResult:
    results = results or []
    return QueryResult(
        pattern="mcp", status=status, message=message,
        trace=run.trace(validation, path, reason, len(results)), results=results,
    )


def _bad() -> HTTPException:
    return _http(502, "mcp_bad_response", "MCP server returned an unexpected response.")


def _valid_id(section_id, act_tool: str, number: int | None = None) -> bool:
    m = _SECTION_ID.match(section_id) if isinstance(section_id, str) else None
    return bool(m) and m.group(1) == act_tool and (number is None or int(m.group(2)) == number)


def _chunk(section: dict, act_tool: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=section["section_id"], section_id=section["section_id"],
        act=_TOOL_TO_ACT[act_tool], text=section.get("text") or "",
        heading=section.get("heading") or "", score=1.0,  # exact match, not a similarity
        act_label=section.get("act_label"), status=section.get("status"),
        chapter=None if section.get("chapter") is None else str(section["chapter"]),
        chapter_title=section.get("chapter_title"),
        section_number=section.get("section_number"),
        source_pdf=section.get("source_pdf"), needs_review=section.get("needs_review"),
    )


def _fetch(run: _Run, path: str, act_tool: str, number: int, section_id: str | None = None):
    """One get_section call, inspected; returns a final QueryResult."""
    data = run.call("get_section", {"act": act_tool, "section_number": number})
    if data["status"] == "not_found":
        return _out(run, "partial_result",
                    "This corpus has no record for that act and section; nothing was guessed.",
                    "passed", path, "record_not_found")
    section = data.get("section")
    if not isinstance(section, dict) or not _valid_id(section.get("section_id"), act_tool, number):
        raise _bad()
    if section_id is not None and section["section_id"] != section_id:
        raise _bad()
    run.steps[-1]["result_count"] = 1
    if section.get("text_available") is False or not str(section.get("text") or "").strip():
        return _out(run, "partial_result",
                    "The record exists but has no extractable text; there is no evidence to answer from.",
                    "passed", path, "section_text_missing")
    chunk = _chunk(section, act_tool)
    return _out(run, "ready_for_synthesis", f"MCP ({path}): 1 section from the legal sections server.",
                "passed", path, "evidence_adequate", [chunk])


def _heading(run: _Run, act_tool: str, phrase: str):
    data = run.call("search_sections_by_heading",
                    {"act": act_tool, "query": phrase, "limit": _SEARCH_LIMIT})
    items = data.get("items")
    if data["status"] != "ok" or not isinstance(items, list):
        raise _bad()
    for it in items:
        if not isinstance(it, dict) or not _valid_id(it.get("section_id"), act_tool):
            raise _bad()
    run.steps[-1]["result_count"] = len(items)
    total = data.get("total", len(items))
    if not items:
        return _out(run, "partial_result", "No section heading in this corpus matches that phrase.",
                    "passed", "heading_search", "no_heading_match")
    if len(items) > 1 or (isinstance(total, int) and total > 1):
        listing = "; ".join(f"{i['section_id']} {i.get('heading') or '(no heading)'}" for i in items)
        return _out(run, "clarify", f"Several headings match; name one section. Matches: {listing}",
                    "passed", "heading_search", "ambiguous_heading")
    item = items[0]
    return _fetch(run, "heading_search", act_tool, int(item["section_id"].split(":")[1]),
                  item["section_id"])


def _range(run: _Run, act_tool: str, start: int, end: int):
    """One get_sections_in_range call; gaps are named, never hidden."""
    data = run.call("get_sections_in_range", {"act": act_tool, "start": start, "end": end})
    items = data.get("items")
    if data["status"] != "ok" or not isinstance(items, list) or len(items) > end - start + 1:
        raise _bad()
    sections = {}
    for it in items:
        sid = it.get("section_id") if isinstance(it, dict) else None
        n = it.get("section_number") if isinstance(it, dict) else None
        if not (_valid_id(sid, act_tool, n) and start <= n <= end) or n in sections:
            raise _bad()
        sections[n] = it
    run.steps[-1]["result_count"] = len(sections)
    usable = [n for n in sorted(sections)
              if sections[n].get("text_available") is not False
              and str(sections[n].get("text") or "").strip()]
    gaps = [n for n in range(start, end + 1) if n not in usable]
    if not usable:
        return _out(run, "partial_result", "The corpus has no usable text for that range; nothing was guessed.",
                    "passed", "range_lookup", "record_not_found")
    chunks = [_chunk(sections[n], act_tool) for n in usable]
    if gaps:
        shown = ", ".join(str(n) for n in gaps[:30])
        return _out(run, "partial_result", f"No usable record for section(s) {shown}; the rest follow.",
                    "passed", "range_lookup", "range_partial", chunks)
    return _out(run, "ready_for_synthesis", f"MCP (range_lookup): {len(chunks)} sections from the legal sections server.",
                "passed", "range_lookup", "evidence_adequate", chunks)


def _listing(run: _Run, path: str, message: str, count: int):
    """Headings/chapters are shown directly; they are not statute text, so no answer step."""
    run.steps[-1]["result_count"] = count
    return _out(run, "listing", message, "passed", path, "listing_returned")


def _chapters(run: _Run, act_tool: str):
    data = run.call("list_chapters", {"act": act_tool})
    rows = data.get("chapters")
    if data["status"] != "ok" or not isinstance(rows, list) or len(rows) > 100:
        raise _bad()
    lines = []
    for r in rows:
        if not isinstance(r, dict) or not all(
            k in r for k in ("chapter", "chapter_title", "first_section", "last_section")
        ):
            raise _bad()
        lines.append(f"Chapter {r['chapter']} - {r['chapter_title']}: sections "
                     f"{r['first_section']}-{r['last_section']}")
    if not lines:
        return _out(run, "partial_result", "The corpus returned no chapters.", "passed",
                    "chapter_list", "no_sections")
    head = f"{act_tool.upper()} has {len(lines)} chapters (from the legal sections server):"
    return _listing(run, "chapter_list", head + "\n" + "\n".join(lines), len(lines))


def _sections(run: _Run, act_tool: str, limit: int):
    data = run.call("list_sections", {"act": act_tool, "limit": limit, "offset": 0})
    items = data.get("items")
    if data["status"] != "ok" or not isinstance(items, list) or len(items) > limit:
        raise _bad()
    lines = []
    for it in items:
        if not isinstance(it, dict) or not _valid_id(it.get("section_id"), act_tool):
            raise _bad()
        lines.append(f"{it['section_id']} - {(it.get('heading') or '(no heading)')[:80]}")
    if not lines:
        return _out(run, "partial_result", "The corpus returned no sections.", "passed",
                    "section_list", "no_sections")
    total = data.get("total")
    head = f"First {len(lines)} {act_tool.upper()} sections" + (
        f" of {total}" if isinstance(total, int) else "") + " (headings only):"
    return _listing(run, "section_list", head + "\n" + "\n".join(lines), len(lines))


def _attach_pdf(question: str, result: QueryResult) -> QueryResult:
    """`/pdf` requested: replace the answer step by a download link when there is content."""
    listing = result.status == "listing"
    if result.status not in ("ready_for_synthesis", "partial_result", "listing") or (
        not listing and not result.results
    ):
        result.message += " (No PDF: nothing to export.)"
        result.trace["pdf"] = "skipped"
        return result
    path = str(result.trace.get("path"))
    note = "" if result.status == "ready_for_synthesis" else result.message
    data = export_pdf.build_pdf(question, path, result.results, note, listing)
    base = get_settings().public_base_url.rstrip("/")
    link = f"{base}/v1/exports/{export_pdf.put(data)}.pdf"
    count = len(result.results) or result.trace.get("result_count", 0)
    gap = f" Note: {result.message}" if result.status == "partial_result" else ""
    result.message = f"PDF ready ({count} item(s)): {link}{gap}"
    result.status = "pdf_ready"
    result.trace["pdf"] = "ready"
    return result


def run_mcp(request: QueryRequest) -> QueryResult:
    question = request.question.strip()
    wants_pdf = _PDF_CMD.search(question) is not None
    if wants_pdf:
        question = _PDF_CMD.sub("", question).strip()
    result = _run_mcp(request, question)
    return _attach_pdf(question, result) if wants_pdf else result


def _run_mcp(request: QueryRequest, question: str) -> QueryResult:
    url, deadline = _settings()
    semantic._validate_scope(request, "mcp")
    run = _Run(url, deadline)

    rng = _RANGE.search(question) if question else None
    if rng:
        acts = [code for pattern, code in structured._ACTS if pattern.search(question)]
        start, end = int(rng.group(1)), int(rng.group(2))
        reason = None
        if len(acts) != 1:
            reason = "ambiguous_request"
        elif not (1 <= start <= end <= MAX_SECTION):
            reason = "invalid_range"
        elif end - start + 1 > MAX_RANGE_SPAN:
            reason = "range_too_large"
        if reason:
            return _out(run, "clarify", _CLARIFY, reason, "none", reason)
        try:
            return _range(run, _ACT_TO_TOOL[acts[0]], start, end)
        except _Stop as stop:
            if stop.reason == "tool_error":
                raise _bad() from None
            return _out(run, "clarify", "The MCP request deadline was reached; try again.",
                        "passed", "range_lookup", stop.reason)

    signals = structured.classify(question) if question else None
    if signals is None:
        return _out(run, "clarify", _CLARIFY, "empty_question", "none", "ambiguous_request")
    if signals.status == "clarification_needed":
        bare = signals.reason.startswith("Which act?")
        reason = "bare_section_number" if bare else "ambiguous_request"
        return _out(run, "clarify", _CLARIFY, reason, "none", reason)

    path = "none"
    try:
        if signals.status == "ok":
            path = "exact_lookup"
            return _fetch(run, path, _ACT_TO_TOOL[signals.act], signals.section_number)
        acts = [code for pattern, code in structured._ACTS if pattern.search(question)]
        quoted = _QUOTED.findall(question)
        if len(acts) == 1 and len(quoted) == 1:
            path = "heading_search"
            return _heading(run, _ACT_TO_TOOL[acts[0]], quoted[0].strip())
        if len(acts) == 1 and _CHAPTERS.search(question):
            path = "chapter_list"
            return _chapters(run, _ACT_TO_TOOL[acts[0]])
        if len(acts) == 1 and _LISTING.search(question):
            first = _FIRST_N.search(question)
            limit = int(first.group(1)) if first else LIST_DEFAULT
            if not 1 <= limit <= LIST_MAX:
                return _out(run, "clarify", _CLARIFY, "invalid_limit", "none", "invalid_limit")
            path = "section_list"
            return _sections(run, _ACT_TO_TOOL[acts[0]], limit)
        return _out(run, "clarify", _CLARIFY, "passed", "none", "unsupported_question")
    except _Stop as stop:
        message = ("The MCP request deadline was reached; try again."
                   if stop.reason == "deadline_exceeded" else _CLARIFY)
        if stop.reason == "tool_error":
            raise _bad() from None
        return _out(run, "clarify", message, "passed", path, stop.reason)
