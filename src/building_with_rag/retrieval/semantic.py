"""Semantic retrieval: embed the question, run $vectorSearch, resolve passages.

Scores only rank similarity; they never prove a passage is correct.
"""

from threading import Lock

from fastapi import HTTPException
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from voyageai import Client as VoyageClient
from voyageai.error import VoyageError

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.settings import get_settings

INPUT_TYPE = "query"
MIN_CANDIDATES = 50
MAX_CANDIDATES = 200
CANDIDATE_MULTIPLIER = 10
MONGO_TIMEOUT_MS = 5000
VOYAGE_TIMEOUT_S = 15.0

_SCORE_NOTE = "Scores rank similarity only and do not prove a passage is correct or answers the question."

_lock = Lock()
_mongo: MongoClient | None = None
_voyage: VoyageClient | None = None
_ready = False


class RetrievalError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def to_http_error(exc: RetrievalError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": exc.message})


def num_candidates(limit: int) -> int:
    """limit*10 clamped to [max(limit, 50), 200]."""
    return min(MAX_CANDIDATES, max(limit, MIN_CANDIDATES, limit * CANDIDATE_MULTIPLIER))


def _not_ready(message: str) -> RetrievalError:
    return RetrievalError(503, "retrieval_not_ready", message)


def _clients():
    global _mongo, _voyage
    settings = get_settings()
    if not settings.mongodb_uri:
        raise _not_ready("MONGODB_URI is not set.")
    if not settings.voyage_api_key:
        raise _not_ready("VOYAGE_API_KEY is not set.")
    with _lock:
        if _mongo is None:
            _mongo = MongoClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=MONGO_TIMEOUT_MS,
                connectTimeoutMS=MONGO_TIMEOUT_MS,
                socketTimeoutMS=MONGO_TIMEOUT_MS * 3,
            )
        if _voyage is None:
            _voyage = VoyageClient(api_key=settings.voyage_api_key, timeout=VOYAGE_TIMEOUT_S)
    db_name = settings.mongodb_test_db_name if settings.app_env == "testing" else settings.mongodb_db_name
    return _mongo[db_name], _voyage


def get_db():
    """MongoDB database only (no Voyage); raises 503 when MONGODB_URI is unset."""
    global _mongo
    settings = get_settings()
    if not settings.mongodb_uri:
        raise _not_ready("MONGODB_URI is not set.")
    with _lock:
        if _mongo is None:
            _mongo = MongoClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=MONGO_TIMEOUT_MS,
                connectTimeoutMS=MONGO_TIMEOUT_MS,
                socketTimeoutMS=MONGO_TIMEOUT_MS * 3,
            )
    name = settings.mongodb_test_db_name if settings.app_env == "testing" else settings.mongodb_db_name
    return _mongo[name]


def _ensure_ready(db) -> None:
    global _ready
    if _ready:
        return
    try:
        index = next(
            (i for i in db[schema.EMBEDDINGS_COLLECTION].list_search_indexes()
             if i.get("name") == schema.VECTOR_INDEX_NAME),
            None,
        )
        sample = db[schema.EMBEDDINGS_COLLECTION].find_one({}, {"model": 1, "dimensions": 1})
    except PyMongoError:
        raise RetrievalError(502, "retrieval_upstream_error", "MongoDB request failed.") from None
    if index is None or not index.get("queryable"):
        raise _not_ready(f"Vector index '{schema.VECTOR_INDEX_NAME}' is missing or not queryable.")
    if sample is None:
        raise _not_ready("Embeddings collection is empty; run the Story 2.2 ingestion.")
    if sample.get("model") != schema.EMBEDDING_MODEL or sample.get("dimensions") != schema.EMBEDDING_DIMENSIONS:
        raise _not_ready("Stored embeddings do not match the expected model/dimensions.")
    _ready = True


def _effective_filters(request: QueryRequest) -> dict[str, list[str]]:
    f = request.filters
    return {
        "act": list(f.act) if f else [],
        "status": list(f.status) if f else [],
        "access_level": [schema.DEFAULT_ACCESS_LEVEL],  # server-fixed; callers can only narrow
    }


def _mongo_filter(filters: dict[str, list[str]]) -> dict:
    clauses = [{field: {"$in": values}} for field, values in filters.items() if values]
    return {"$and": clauses}


def _validate_scope(request: QueryRequest, mode: str = "semantic") -> None:
    problems = []
    if request.caller_id is not None and request.caller_id != get_settings().webui_demo_caller_id:
        problems.append("caller_id is not the permitted demo caller")
    if request.required_acts is not None:
        problems.append(f"required_acts is not supported by {mode} mode")
    if request.chapter is not None:
        problems.append(f"chapter is not supported by {mode} mode")
    if problems:
        raise HTTPException(status_code=422, detail="; ".join(problems) + ".")


def embed_question(voyage, question: str) -> list[float]:
    """Embed the raw question (input_type="query"); fail on a wrong dimension."""
    try:
        vector = voyage.embed(
            texts=[question], model=schema.EMBEDDING_MODEL, input_type=INPUT_TYPE
        ).embeddings[0]
    except VoyageError:
        raise RetrievalError(502, "retrieval_upstream_error", "Voyage embedding request failed.") from None
    if len(vector) != schema.EMBEDDING_DIMENSIONS:
        raise _not_ready(
            f"Query embedding has {len(vector)} dimensions; expected {schema.EMBEDDING_DIMENSIONS}."
        )
    return vector


def vector_search(db, vector, limit: int, candidates: int, mongo_filter: dict) -> list[dict]:
    """$vectorSearch on embeddings with the filter inside the stage; descending score order."""
    pipeline = [
        {"$vectorSearch": {
            "index": schema.VECTOR_INDEX_NAME,
            "path": "vector",
            "queryVector": vector,
            "numCandidates": candidates,
            "limit": limit,
            "filter": mongo_filter,
        }},
        {"$project": {"_id": 0, "chunk_id": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]
    return list(db[schema.EMBEDDINGS_COLLECTION].aggregate(pipeline))


def resolve_hits(db, hits: list[dict]) -> tuple[list[RetrievedChunk], int]:
    """Resolve {chunk_id, score} hits through chunks and sections; unresolved hits are omitted."""
    chunk_ids = [h["chunk_id"] for h in hits]
    chunks = {c["chunk_id"]: c for c in db[schema.CHUNKS_COLLECTION].find({"chunk_id": {"$in": chunk_ids}})}
    section_ids = list({c["section_id"] for c in chunks.values() if c.get("section_id")})
    sections = {s["section_id"]: s for s in db[schema.SECTIONS_COLLECTION].find({"section_id": {"$in": section_ids}})}
    results = []
    unresolved = 0
    for hit in hits:
        chunk = chunks.get(hit["chunk_id"])
        section = sections.get(chunk.get("section_id")) if chunk else None
        if chunk is None or section is None:
            unresolved += 1
            continue
        results.append(RetrievedChunk(
            chunk_id=chunk["chunk_id"],
            section_id=section["section_id"],
            act=section.get("act") or chunk.get("act") or "",
            text=chunk.get("text") or "",
            heading=section.get("heading") or "",
            score=hit["score"],
            chunk_index=chunk.get("chunk_index"),
            act_label=section.get("act_label"),
            status=section.get("status"),
            chapter=section.get("chapter"),
            chapter_title=section.get("chapter_title"),
            section_number=section.get("section_number"),
            source_pdf=section.get("source_pdf"),
            source_sha256=section.get("source_sha256"),
            needs_review=section.get("needs_review"),
        ))
    return results, unresolved


def _result(request, status, message, trace, results) -> QueryResult:
    return QueryResult(
        pattern="semantic", status=status, message=message, trace=trace, results=results
    )


def run_semantic(request: QueryRequest) -> QueryResult:
    global _ready
    _validate_scope(request)
    try:
        return _run(request)
    except RetrievalError as exc:
        _ready = False
        raise to_http_error(exc) from None


def _run(request: QueryRequest) -> QueryResult:
    global _ready
    db, voyage = _clients()
    _ensure_ready(db)

    filters = _effective_filters(request)
    limit = request.limit
    candidates = num_candidates(limit)
    mongo_filter = _mongo_filter(filters)
    trace = {
        "mode": "semantic",
        "query": request.question,
        "embedding": {
            "model": schema.EMBEDDING_MODEL,
            "input_type": INPUT_TYPE,
            "dimensions": schema.EMBEDDING_DIMENSIONS,
        },
        "index": schema.VECTOR_INDEX_NAME,
        "limit": limit,
        "num_candidates": candidates,
        "filters": filters,
        "caller_id": get_settings().webui_demo_caller_id,
        "result_count": 0,
        "ignored": [],
        "unresolved_hits": 0,
    }

    try:
        searchable = db[schema.EMBEDDINGS_COLLECTION].count_documents(mongo_filter, limit=1)
    except PyMongoError:
        _ready = False
        raise RetrievalError(502, "retrieval_upstream_error", "MongoDB request failed.") from None
    if not searchable:
        return _result(
            request, "no_results",
            "No passages match the given filters. " + _SCORE_NOTE, trace, [],
        )

    vector = embed_question(voyage, request.question)
    try:
        hits = vector_search(db, vector, limit, candidates, mongo_filter)
        results, unresolved = resolve_hits(db, hits)
    except PyMongoError:
        raise RetrievalError(502, "retrieval_upstream_error", "MongoDB request failed.") from None
    trace["result_count"] = len(results)
    trace["unresolved_hits"] = unresolved
    return _result(
        request, "ok", f"Returned {len(results)} passage(s). " + _SCORE_NOTE, trace, results
    )
