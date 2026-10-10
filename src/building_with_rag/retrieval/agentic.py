"""Bounded agentic RAG: a deterministic, read-only controller over three allowlisted adapters.

No LLM planner and no dynamic tools. Runs only when the caller selects the `agentic` mode.
Real corpus passages returned by an adapter are the only evidence; HyDE debug text, candidate
lists and adapter traces are dropped before the result is returned.
"""

import logging
import os
import re
import sys
import time
import uuid

from fastapi import HTTPException

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.retrieval import hyde, semantic, structured
from building_with_rag.settings import get_settings

MAX_CALLS = 2
MAX_HYDE_CALLS = 1
DEADLINE_RANGE = (1.0, 120.0)

# Fixed allowlist: tool name -> (adapter, expected QueryResult.pattern).
_TOOLS = {
    "structuredLookup": (structured.run_structured, "structured"),
    "semanticRetrieve": (semantic.run_semantic, "semantic"),
    "hydeRetrieve": (hyde.run_hyde, "hyde"),
}

# Prepared vocabulary-mismatch rule: lay cues present and no corpus legal term present.
LAY_CUES = ("secretly takes", "snatch", "grab", "steals", "cheated", "kill", "hit", "hurt")
LEGAL_TERMS = ("theft", "murder", "robbery", "cheating", "culpable homicide", "extortion")
_LAY = re.compile(r"\b(?:" + "|".join(re.escape(c) for c in LAY_CUES) + r")\b", re.IGNORECASE)
_LEGAL = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in LEGAL_TERMS) + r")\b", re.IGNORECASE)

_CLARIFY_MESSAGE = (
    "Please rephrase the question, or name the act (BNS or IPC) and one section number."
)

_now = time.monotonic  # replaceable clock

# Colour-coded controller log. Never logs question text, credentials or HyDE text.
_COLORS = {
    "START": "\x1b[1;34m", "VALIDATE": "\x1b[34m", "PATH": "\x1b[1;35m", "CALL": "\x1b[36m",
    "RESULT": "\x1b[32m", "INADEQUATE": "\x1b[33m", "STOP": "\x1b[1;31m", "FINAL": "\x1b[1;32m",
}
_RESET = "\x1b[0m"
log = logging.getLogger("building_with_rag.agentic")


def _setup_log() -> None:
    if log.handlers:
        return
    if sys.platform == "win32":
        os.system("")  # enable ANSI colours in the Windows console
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def _event(rid: str, kind: str, message: str) -> None:
    log.info("%s[agentic %s] %-10s %s%s", _COLORS.get(kind, ""), rid, kind, message, _RESET)


_setup_log()


class _Stop(Exception):
    def __init__(self, reason: str):
        self.reason = reason


def _deadline_seconds() -> float:
    raw = get_settings().agentic_deadline_seconds
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = -1.0
    if not DEADLINE_RANGE[0] <= value <= DEADLINE_RANGE[1]:
        raise HTTPException(status_code=503, detail={
            "code": "retrieval_not_ready",
            "message": "AGENTIC_DEADLINE_SECONDS must be a number from 1 to 120.",
        })
    return value


def vocabulary_mismatch(question: str) -> bool:
    return bool(_LAY.search(question)) and not _LEGAL.search(question)


def _usable(result: QueryResult) -> list[RetrievedChunk]:
    return [c for c in result.results if c.text.strip()]


def _well_formed(result: QueryResult, pattern: str) -> bool:
    return result.pattern == pattern and all(c.chunk_id and c.section_id for c in result.results)


class _Run:
    def __init__(self, request: QueryRequest, deadline: float):
        self.request = request
        self.deadline = deadline
        self.started = _now()
        self.rid = uuid.uuid4().hex[:6]
        self.steps: list[dict] = []
        self.calls = 0
        self.hyde_calls = 0
        self.http_error: HTTPException | None = None

    def call(self, name: str, need_legal_term: bool = False) -> tuple[QueryResult, bool]:
        """One allowlisted call: budget and deadline first, then inspect the returned result.

        need_legal_term (vocabulary-mismatch direct step): adequacy also requires a passage
        that uses a corpus legal term, i.e. the lay wording actually reached the legal text.
        """
        if self.calls >= MAX_CALLS or (name == "hydeRetrieve" and self.hyde_calls >= MAX_HYDE_CALLS):
            _event(self.rid, "STOP", f"call budget exhausted before {name} "
                   f"(calls={self.calls}/{MAX_CALLS}, hyde={self.hyde_calls}/{MAX_HYDE_CALLS})")
            raise _Stop("call_budget_exhausted")
        elapsed = _now() - self.started
        if elapsed >= self.deadline:
            _event(self.rid, "STOP", f"deadline exceeded before {name} "
                   f"({elapsed:.1f}s >= {self.deadline:.0f}s)")
            raise _Stop("deadline_exceeded")
        adapter, pattern = _TOOLS[name]
        self.calls += 1
        self.hyde_calls += name == "hydeRetrieve"
        step = {"adapter": name, "tool_status": "error", "adequate": False, "result_count": 0}
        self.steps.append(step)
        _event(self.rid, "CALL", f"{name} -> call {self.calls}/{MAX_CALLS} "
               f"(elapsed {elapsed:.1f}s of {self.deadline:.0f}s)")
        try:
            result = adapter(self.request.model_copy(deep=True))
        except HTTPException as exc:
            step["tool_status"] = f"http_{exc.status_code}"
            self.http_error = exc
            _event(self.rid, "STOP", f"{name} failed: HTTP {exc.status_code}")
            raise _Stop("tool_error") from None
        if not _well_formed(result, pattern):
            step["tool_status"] = "bad_result"
            _event(self.rid, "STOP", f"{name} returned a malformed result")
            raise _Stop("tool_error")
        usable = _usable(result)
        adequate = result.status == "ok" and bool(usable)
        if adequate and need_legal_term:
            adequate = any(_LEGAL.search(f"{c.heading} {c.text}") for c in usable)
        step.update(tool_status=result.status, adequate=adequate, result_count=len(usable))
        ids = ",".join(c.section_id for c in usable[:5])
        _event(self.rid, "RESULT" if adequate else "INADEQUATE",
               f"{name}: status={result.status} passages={len(usable)} adequate={adequate}"
               + (f" sections=[{ids}]" if ids else ""))
        return result, adequate

    def trace(self, validation: str, path: str, stop_reason: str, result_count: int) -> dict:
        return {
            "mode": "agentic", "validation": validation, "path": path, "steps": self.steps,
            "call_count": self.calls, "hyde_calls": self.hyde_calls,
            "deadline_seconds": self.deadline,
            "elapsed_ms": int((_now() - self.started) * 1000), "stop_reason": stop_reason,
            "caller_id": get_settings().webui_demo_caller_id, "result_count": result_count,
        }


def _out(run: _Run, status, message, validation, path, reason, results=None) -> QueryResult:
    results = results or []
    _event(run.rid, "FINAL" if status == "ready_for_synthesis" else "STOP",
           f"status={status} stop_reason={reason} path={path} calls={run.calls} "
           f"hyde={run.hyde_calls} passages={len(results)}")
    return QueryResult(
        pattern="agentic", status=status, message=message,
        trace=run.trace(validation, path, reason, len(results)), results=results,
    )


def _ready(run, path, adapter, result) -> QueryResult:
    passages = _usable(result)
    message = f"Agentic ({path}): {len(passages)} real passage(s) from {adapter}."
    return _out(run, "ready_for_synthesis", message, "passed", path, "evidence_adequate", passages)


def run_agentic(request: QueryRequest) -> QueryResult:
    deadline = _deadline_seconds()
    semantic._validate_scope(request, "agentic")
    run = _Run(request, deadline)
    question = request.question.strip()
    _event(run.rid, "START", f"agentic request received (question_chars={len(question)}, "
           f"deadline={deadline:.0f}s, budget={MAX_CALLS} calls / {MAX_HYDE_CALLS} HyDE)")

    # 1. Validate before any tool call.
    signals = structured.classify(question) if question else None
    if signals is None:
        return _out(run, "clarify", _CLARIFY_MESSAGE, "empty_question", "none", "ambiguous_request")
    if signals.status == "clarification_needed":
        bare = signals.reason.startswith("Which act?")
        reason = "bare_section_number" if bare else "ambiguous_request"
        _event(run.rid, "VALIDATE", f"rejected: {reason}; no tool called")
        return _out(run, "clarify", _CLARIFY_MESSAGE, reason, "none", reason)

    _event(run.rid, "VALIDATE", "passed")
    if signals.status == "ok":
        path = "exact_lookup"
    elif vocabulary_mismatch(question):
        path = "vocab_mismatch"
    else:
        path = "semantic"

    _event(run.rid, "PATH", {
        "exact_lookup": "act + section named -> structuredLookup",
        "vocab_mismatch": "lay wording, no legal term -> semanticRetrieve, then hydeRetrieve if inadequate",
        "semantic": "ordinary question -> semanticRetrieve",
    }[path])
    try:
        return _follow(run, path)
    except _Stop as stop:
        if stop.reason == "tool_error" and run.http_error is not None:
            semantic._ready = False
            raise run.http_error from None
        status = "partial_result" if stop.reason == "tool_error" else "clarify"
        return _out(run, status, _CLARIFY_MESSAGE, "passed", path, stop.reason)


def _follow(run: _Run, path: str) -> QueryResult:
    if path == "exact_lookup":
        result, adequate = run.call("structuredLookup")
        if adequate:
            return _ready(run, path, "structuredLookup", result)
        if result.status == "not_found":
            return _out(
                run, "partial_result",
                "This corpus has no record for that act and section; nothing was guessed.",
                "passed", path, "record_not_found",
            )
        return _out(run, "clarify", _CLARIFY_MESSAGE, "passed", path, "adapter_clarification")

    result, adequate = run.call("semanticRetrieve", need_legal_term=path == "vocab_mismatch")
    if adequate:
        return _ready(run, path, "semanticRetrieve", result)
    if path == "vocab_mismatch":
        result, adequate = run.call("hydeRetrieve")
        if adequate:
            return _ready(run, path, "hydeRetrieve", result)  # drops all HyDE-only fields
    return _out(run, "clarify", _CLARIFY_MESSAGE, "passed", path, "no_usable_evidence")

