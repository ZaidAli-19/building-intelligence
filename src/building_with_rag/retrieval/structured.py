"""Structured exact retrieval: rule-based classifier + one read-only `sections` lookup.

Only validated StructuredSignals values (act, section_number, optional chapter) and
server-fixed filters may reach MongoDB; raw question text never does. No LLM, no Voyage.
"""

import re

from fastapi import HTTPException
from pymongo.errors import PyMongoError

from building_with_rag.contracts import (
    QueryRequest,
    QueryResult,
    RetrievedChunk,
    StructuredSignals,
)
from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.retrieval import semantic
from building_with_rag.retrieval.semantic import RetrievalError
from building_with_rag.settings import get_settings

_CHAPTER = re.compile(r"^[A-Za-z0-9 .\-]{1,40}$")
_AGGREGATION = re.compile(r"\b(how many|count|total number)\b", re.IGNORECASE)
_FILTER = re.compile(r"\b(list|all sections|which sections)\b|\bsections?\s+(?:in|under)\b", re.IGNORECASE)
_SECTION_REF = re.compile(r"(?:\bsections?\b|\bsec\b\.?|\bs\.|§)\s*(\d+)([A-Za-z]*)", re.IGNORECASE)
_FOLLOWING_NUMBER = re.compile(r"\s*(?:,|and|&|or)\s*(?:sections?\s*|sec\.?\s*|s\.\s*|§\s*)?(\d+)([A-Za-z]*)", re.IGNORECASE)
_ACTS = (
    (re.compile(r"\bBNS\b|\bBharatiya\s+Nyaya\s+Sanhita\b", re.IGNORECASE), "BNS_2023"),
    (re.compile(r"\bIPC\b|\bIndian\s+Penal\s+Code\b", re.IGNORECASE), "IPC_1860"),
)

_MESSAGE_NOTE = (
    "Exact record from the supplied corpus; its status and source_status_version describe the "
    "source document and do not claim current legal applicability."
)


def _signals(**kw) -> StructuredSignals:
    return StructuredSignals(**kw)


def classify(question: str, chapter: str | None = None) -> StructuredSignals:
    """Pure rule-based classifier. Acts and sections are never guessed."""
    if _AGGREGATION.search(question):
        return _signals(
            intent="aggregation", chapter=chapter, status="recommendation",
            reason="Counting questions are recognised but not executed; use semantic or hybrid.",
        )
    if _FILTER.search(question):
        return _signals(
            intent="filter", chapter=chapter, status="recommendation",
            reason="Listing and filtering questions are recognised but not executed; "
            "use semantic or hybrid.",
        )

    refs: list[tuple[str, str]] = []
    for m in _SECTION_REF.finditer(question):
        refs.append((m.group(1), m.group(2)))
        pos = m.end()
        while (n := _FOLLOWING_NUMBER.match(question, pos)):
            refs.append((n.group(1), n.group(2)))
            pos = n.end()
    acts = [code for pattern, code in _ACTS if pattern.search(question)]

    if not refs:
        return _signals(
            chapter=chapter, status="recommendation",
            reason="No section reference found; for open questions use semantic or hybrid.",
        )

    def clarify(reason: str) -> StructuredSignals:
        return _signals(intent="exact_lookup", chapter=chapter, status="clarification_needed",
                        reason=reason)

    if any(suffix for _, suffix in refs):
        return clarify("Only plain integer section numbers are supported (e.g. 103, not 103A).")
    numbers = {int(number) for number, _ in refs}
    if len(numbers) > 1:
        return clarify("Several section numbers were given; ask for one section at a time.")
    number = next(iter(numbers))
    if not 1 <= number <= 999:
        return clarify("The section number must be between 1 and 999.")
    if not acts:
        return clarify("Which act? Say BNS or IPC; the act is never guessed.")
    if len(acts) > 1:
        return clarify("Both BNS and IPC were named; ask about one act at a time.")
    return _signals(
        intent="exact_lookup", act=acts[0], section_number=number, chapter=chapter,
        status="ok", reason=f"Exact lookup of {acts[0]} section {number}.",
    )


def _validate_scope(request: QueryRequest) -> None:
    problems = []
    if request.caller_id is not None and request.caller_id != get_settings().webui_demo_caller_id:
        problems.append("caller_id is not the permitted demo caller")
    if request.required_acts is not None:
        problems.append("required_acts is not supported by structured mode")
    if request.chapter is not None and not _CHAPTER.match(request.chapter):
        raise HTTPException(status_code=422, detail={
            "code": "unsupported_option",
            "message": "chapter must be 1-40 characters of letters, digits, spaces, '.' and '-'.",
        })
    if problems:
        raise HTTPException(status_code=422, detail="; ".join(problems) + ".")


def _result(status: str, message: str, trace: dict, results=None) -> QueryResult:
    return QueryResult(
        pattern="structured", status=status, message=message, trace=trace, results=results or []
    )


def run_structured(request: QueryRequest) -> QueryResult:
    _validate_scope(request)
    signals = classify(request.question, request.chapter)
    filters = semantic._effective_filters(request)
    trace = {
        "mode": "structured",
        "signals": signals.model_dump(),
        "mongodb_called": False,
        "collection": schema.SECTIONS_COLLECTION,
        "filters": filters,
        "caller_id": get_settings().webui_demo_caller_id,
        "result_count": 0,
    }
    if signals.status != "ok":
        return _result(signals.status, signals.reason, trace)
    if filters["act"] and signals.act not in filters["act"]:
        return _result(
            "clarification_needed",
            f"The question names {signals.act}, which the act filter excludes; "
            "adjust the filter or the question.",
            trace,
        )

    predicate = {
        "act": signals.act,
        "section_number": signals.section_number,
        "access_level": {"$in": filters["access_level"]},
    }
    if filters["status"]:
        predicate["status"] = {"$in": filters["status"]}
    if signals.chapter:
        predicate["chapter"] = signals.chapter

    try:
        db = semantic.get_db()
        trace["mongodb_called"] = True
        doc = db[schema.SECTIONS_COLLECTION].find_one(predicate, {"provenance": 0})
    except RetrievalError as exc:
        raise semantic.to_http_error(exc) from None
    except PyMongoError:
        raise semantic.to_http_error(
            RetrievalError(502, "retrieval_upstream_error", "MongoDB request failed.")
        ) from None

    if doc is None:
        return _result(
            "not_found",
            "This corpus has no such record (this says nothing about whether the law has such "
            "a section).",
            trace,
        )
    chunk = RetrievedChunk(
        chunk_id=doc["section_id"],
        section_id=doc["section_id"],
        act=doc.get("act") or signals.act,
        text=doc.get("text") or "",
        heading=doc.get("heading") or "",
        score=1.0,  # marks an exact match, not a similarity
        act_label=doc.get("act_label"),
        status=doc.get("status"),
        chapter=doc.get("chapter"),
        chapter_title=doc.get("chapter_title"),
        section_number=doc.get("section_number"),
        source_pdf=doc.get("source_pdf"),
        source_sha256=doc.get("source_sha256"),
        needs_review=doc.get("needs_review"),
    )
    trace["result_count"] = 1
    trace["record"] = {
        "section_id": doc["section_id"],
        "status": doc.get("status"),
        "source_status_version": doc.get("source_status_version"),
    }
    return _result("ok", _MESSAGE_NOTE, trace, [chunk])
