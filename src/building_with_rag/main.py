import json
import time
import uuid
from collections.abc import Iterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from building_with_rag.models import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelCard,
    ModelList,
    OpenAIErrorBody,
    OpenAIErrorEnvelope,
    QueryRequest,
    QueryResult,
)
from building_with_rag.patterns import run_pattern
from building_with_rag.registry import MODES, MODES_BY_MODEL_ID
from building_with_rag.settings import get_settings

app = FastAPI(title="Building with RAG")


class ChatError(Exception):
    def __init__(self, status: int, message: str, code: str | None = None, param: str | None = None):
        self.status, self.message, self.code, self.param = status, message, code, param


def _error(status: int, message: str, code: str | None = None, param: str | None = None):
    body = OpenAIErrorEnvelope(error=OpenAIErrorBody(message=message, code=code, param=param))
    return JSONResponse(status_code=status, content=body.model_dump())


@app.exception_handler(ChatError)
async def _chat_error(_: Request, exc: ChatError):
    return _error(exc.status, exc.message, exc.code, exc.param)


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    if request.url.path == "/v1/chat/completions":
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        return _error(400, f"Invalid request: {loc}: {first.get('msg', 'invalid')}", param=loc or None)
    return JSONResponse(status_code=422, content={"detail": json.loads(json.dumps(exc.errors(), default=str))})


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/query", response_model=QueryResult)
def query(request: QueryRequest) -> QueryResult:
    return run_pattern(request)


@app.get("/v1/models", response_model=ModelList)
def models() -> ModelList:
    return ModelList(data=[ModelCard(id=m.model_id) for m in MODES.values()])


def _answer_text(result: QueryResult) -> str:
    return result.message


def _sse(chunk: ChatCompletionChunk) -> str:
    return f"data: {chunk.model_dump_json()}\n\n"


def _stream(completion_id: str, created: int, model: str, text: str) -> Iterator[str]:
    def chunk(delta: ChatCompletionDelta, finish: str | None = None) -> str:
        return _sse(
            ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=model,
                choices=[ChatCompletionChunkChoice(delta=delta, finish_reason=finish)],
            )
        )

    yield chunk(ChatCompletionDelta(role="assistant"))
    yield chunk(ChatCompletionDelta(content=text))
    yield chunk(ChatCompletionDelta(), "stop")
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    mode = MODES_BY_MODEL_ID.get(request.model)
    if mode is None:
        raise ChatError(404, f"The model '{request.model}' does not exist.", "model_not_found", "model")
    question = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
    opts = request.rag_options
    try:
        query_request = QueryRequest(
            question=question,
            pattern=mode.pattern,
            caller_id=get_settings().webui_demo_caller_id,  # server-side, never browser-supplied
            filters=opts.filters if opts else None,
            limit=(opts.limit if opts and opts.limit else 5),
            generate_answer=True,  # server-side
            required_acts=(opts.required_acts or []) if opts else [],
            chapter=opts.chapter if opts else None,
        )
    except ValueError as exc:
        raise ChatError(400, f"Invalid request: {exc.errors()[0]['msg']}" if hasattr(exc, "errors") else "Invalid request", param="messages") from exc

    text = _answer_text(run_pattern(query_request))
    completion_id, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
    if request.stream:
        return StreamingResponse(
            _stream(completion_id, created, request.model, text), media_type="text/event-stream"
        )
    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[ChatCompletionChoice(message=ChatMessage(role="assistant", content=text))],
    )
