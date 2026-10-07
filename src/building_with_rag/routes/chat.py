"""OpenAI-compatible chat adapter.

Maps the selected rag-<pattern> model to the same QueryRequest and the same
retrieve -> answer_events path as /v1/query, with server-side demo caller_id and
generate_answer. Only answer text streams; retrieval stays non-streamed. Supports normal
JSON responses and role/content/stop SSE frames; no custom SSE events. The OpenAI-style
error envelope applies before streaming begins.
"""

import hmac
import json
import time
import uuid
from collections.abc import Iterator

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from building_with_rag.contracts import ChatCompletionRequest, QueryRequest, SemanticFilters
from building_with_rag.pipeline import answer_events, render_footer, retrieve
from building_with_rag.registry import MODEL_ID_TO_PATTERN
from building_with_rag.settings import get_settings

router = APIRouter()

_FINISH_STOP = "stop"
_DONE = "[" + "DONE" + "]"


def _error(status: int, message: str, code: str, kind: str = "invalid_request_error") -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={"error": {"message": message, "type": kind, "code": code}},
    )


def _completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def _check_api_key(authorization: str | None) -> None:
    expected = get_settings().capstone_api_key
    if not expected:
        return
    supplied = ""
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise _error(401, "Invalid API key.", "invalid_api_key", "authentication_error")


def _unique(*lists: list[str]) -> list[str]:
    seen: list[str] = []
    for values in lists:
        for v in values:
            if v not in seen:
                seen.append(v)
    return seen


def _build_query(request: ChatCompletionRequest) -> QueryRequest:
    pattern = MODEL_ID_TO_PATTERN.get(request.model)
    if pattern is None:
        raise _error(400, f"Model '{request.model}' not found.", "model_not_found")
    latest_user = next((m.content for m in reversed(request.messages) if m.role == "user"), None)
    if latest_user is None:
        raise _error(400, "At least one user message is required.", "missing_user_message")
    options = request.rag_options
    nested = options.filters if options and options.filters else None
    try:
        return QueryRequest(
            question=latest_user,
            pattern=pattern,
            caller_id=get_settings().webui_demo_caller_id,  # server-side demo caller
            filters=(
                SemanticFilters(
                    act=_unique(options.act, nested.act if nested else []),
                    status=_unique(options.status, nested.status if nested else []),
                    access_level=options.access_level,
                )
                if options
                else None
            ),
            limit=options.limit if options else 5,
            generate_answer=True,  # server-side decision; adapter never trusts client
            required_acts=options.required_acts if options else None,
            chapter=options.chapter if options else None,
        )
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first["loc"])
        raise _error(400, f"Invalid request ({loc}): {first['msg']}", "invalid_request") from None


def _prepare(request: ChatCompletionRequest):
    """Retrieve (non-streamed) and return the event iterator; raises before any streaming."""
    query_request = _build_query(request)
    try:
        retrieval = retrieve(query_request)
    except HTTPException as exc:
        detail = exc.detail
        if isinstance(detail, dict):
            raise _error(
                exc.status_code, str(detail.get("message", "Retrieval failed.")),
                str(detail.get("code", "retrieval_error")), "server_error",
            ) from None
        raise _error(exc.status_code, str(detail), "invalid_request") from None
    return answer_events(query_request.question, retrieval)


def _pieces(events: Iterator[tuple[str, object]]) -> Iterator[str]:
    """Forward text/notice pieces, then the footer rendered from the same final result."""
    emitted = False
    for kind, payload in events:
        if kind in ("text", "notice"):
            emitted = True
            yield payload
        elif kind == "final":
            footer = render_footer(payload)
            if footer:
                yield footer if emitted else footer.lstrip("\n")


def _json_response(request: ChatCompletionRequest, text: str) -> dict:
    return {
        "id": _completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": index,
                "message": {"role": "assistant", "content": text},
                "finish_reason": _FINISH_STOP,
            }
            for index in range(request.n)
        ],
    }


def _sse_frame(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/v1/chat/completions")
def chat_completions(
    request: ChatCompletionRequest, authorization: str | None = Header(default=None)
):
    _check_api_key(authorization)
    events = _prepare(request)  # may raise the error envelope; nothing streamed yet
    if not request.stream:
        return _json_response(request, "".join(_pieces(events)))

    def generate():
        base = {"id": _completion_id(), "created": int(time.time()), "model": request.model}

        def frame(delta: dict, finish: str | None) -> str:
            return _sse_frame({
                **base,
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": i, "delta": delta, "finish_reason": finish}
                    for i in range(request.n)
                ],
            })

        yield frame({"role": "assistant"}, None)
        for piece in _pieces(events):
            yield frame({"content": piece}, None)
        yield frame({}, _FINISH_STOP)
        yield "data: " + _DONE + "\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
