"""Hybrid retrieval: vector route + Atlas Search keyword route, fused by RRF.

Fusion ranks only (raw BM25 and cosine scores are not comparable); the fused score ranks
passages and does not prove a passage is correct or answers the question.
"""

from threading import Lock

from pymongo.errors import PyMongoError

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.retrieval import semantic
from building_with_rag.retrieval.semantic import RetrievalError
from building_with_rag.settings import get_settings

RRF_K = 60
KEYWORD_COMMAND = "uv run python -m building_with_rag.ingestion.keyword_index"

_SCORE_NOTE = (
    "The fused score ranks passages only and does not prove a passage is correct or answers "
    "the question."
)

_lock = Lock()
_ready = False


def route_depth(limit: int) -> int:
    return max(limit, min(50, max(20, 4 * limit)))


def fuse(semantic_hits: list[dict], keyword_hits: list[dict], limit: int) -> list[dict]:
    """Reciprocal Rank Fusion over ranks. Inputs are best-first lists of {chunk_id, score}.

    Returns the top *limit* entries: chunk_id, per-route score/rank (None if absent),
    fused_score, fused_rank (1-based over the full fused list).
    """
    merged: dict[str, dict] = {}
    for route, hits in (("semantic", semantic_hits), ("keyword", keyword_hits)):
        for rank, hit in enumerate(hits, start=1):
            entry = merged.setdefault(hit["chunk_id"], {
                "chunk_id": hit["chunk_id"],
                "semantic_score": None, "semantic_rank": None,
                "keyword_score": None, "keyword_rank": None,
                "fused_score": 0.0,
            })
            if entry[f"{route}_rank"] is not None:
                continue  # a route lists a chunk once
            entry[f"{route}_score"] = hit["score"]
            entry[f"{route}_rank"] = rank
            entry["fused_score"] += 1.0 / (RRF_K + rank)
    ordered = sorted(
        merged.values(),
        key=lambda e: (
            -e["fused_score"],
            e["semantic_rank"] if e["semantic_rank"] is not None else float("inf"),
            e["chunk_id"],
        ),
    )
    for position, entry in enumerate(ordered, start=1):
        entry["fused_rank"] = position
    return ordered[:limit]


def _ensure_ready(db) -> None:
    global _ready
    if _ready:
        return
    semantic._ensure_ready(db)  # vector index queryable, embeddings non-empty and matching
    try:
        index = next(
            (i for i in db[schema.CHUNKS_COLLECTION].list_search_indexes()
             if i.get("name") == schema.KEYWORD_INDEX_NAME),
            None,
        )
    except PyMongoError:
        raise RetrievalError(502, "retrieval_upstream_error", "MongoDB request failed.") from None
    if index is None or not index.get("queryable"):
        raise RetrievalError(
            503, "retrieval_not_ready",
            f"Keyword index '{schema.KEYWORD_INDEX_NAME}' is missing or not queryable; "
            f"run: {KEYWORD_COMMAND}",
        )
    with _lock:
        _ready = True


def keyword_search(db, question: str, filters: dict[str, list[str]], depth: int) -> list[dict]:
    """Atlas Search `text` operator on chunks.text; filters inside compound.filter."""
    pipeline = [
        {"$search": {
            "index": schema.KEYWORD_INDEX_NAME,
            "compound": {
                "must": [{"text": {"query": question, "path": "text"}}],
                "filter": [
                    {"in": {"path": field, "value": values}}
                    for field, values in filters.items() if values
                ],
            },
        }},
        {"$limit": depth},
        {"$project": {"_id": 0, "chunk_id": 1, "score": {"$meta": "searchScore"}}},
    ]
    return list(db[schema.CHUNKS_COLLECTION].aggregate(pipeline))


def _result(status: str, message: str, trace: dict, results: list[RetrievedChunk]) -> QueryResult:
    return QueryResult(pattern="hybrid", status=status, message=message, trace=trace, results=results)


def run_hybrid(request: QueryRequest) -> QueryResult:
    global _ready
    semantic._validate_scope(request, "hybrid")
    try:
        return _run(request)
    except RetrievalError as exc:
        _ready = False
        raise semantic.to_http_error(exc) from None


def _run(request: QueryRequest) -> QueryResult:
    db, voyage = semantic._clients()
    _ensure_ready(db)

    filters = semantic._effective_filters(request)
    mongo_filter = semantic._mongo_filter(filters)
    limit = request.limit
    depth = route_depth(limit)
    candidates = semantic.num_candidates(depth)
    trace = {
        "mode": "hybrid",
        "query": request.question,
        "embedding": {
            "model": schema.EMBEDDING_MODEL,
            "input_type": semantic.INPUT_TYPE,
            "dimensions": schema.EMBEDDING_DIMENSIONS,
        },
        "filters": filters,
        "caller_id": get_settings().webui_demo_caller_id,
        "result_count": 0,
        "unresolved_hits": 0,
        "semantic": {
            "index": schema.VECTOR_INDEX_NAME, "limit": depth,
            "num_candidates": candidates, "hit_count": 0,
        },
        "keyword": {
            "index": schema.KEYWORD_INDEX_NAME, "path": "text", "operator": "text",
            "limit": depth, "hit_count": 0,
        },
        "fusion": {"method": "rrf", "k": RRF_K, "weights": {"semantic": 1.0, "keyword": 1.0},
                   "route_depth": depth},
        "contribution": {"both": 0, "semantic_only": 0, "keyword_only": 0},
    }

    vector = semantic.embed_question(voyage, request.question)
    try:
        semantic_hits = semantic.vector_search(db, vector, depth, candidates, mongo_filter)
        keyword_hits = keyword_search(db, request.question, filters, depth)
    except PyMongoError:
        raise RetrievalError(502, "retrieval_upstream_error", "MongoDB request failed.") from None
    trace["semantic"]["hit_count"] = len(semantic_hits)
    trace["keyword"]["hit_count"] = len(keyword_hits)

    fused = fuse(semantic_hits, keyword_hits, limit)
    if not fused:
        return _result("no_results", "No passages matched by either route. " + _SCORE_NOTE, trace, [])

    try:
        resolved, unresolved = semantic.resolve_hits(
            db, [{"chunk_id": f["chunk_id"], "score": f["fused_score"]} for f in fused]
        )
    except PyMongoError:
        raise RetrievalError(502, "retrieval_upstream_error", "MongoDB request failed.") from None
    by_id = {f["chunk_id"]: f for f in fused}
    results = []
    for chunk in resolved:
        f = by_id[chunk.chunk_id]
        results.append(chunk.model_copy(update={
            "semantic_score": f["semantic_score"], "semantic_rank": f["semantic_rank"],
            "keyword_score": f["keyword_score"], "keyword_rank": f["keyword_rank"],
            "fused_score": f["fused_score"], "fused_rank": f["fused_rank"],
        }))
        if f["semantic_rank"] is not None and f["keyword_rank"] is not None:
            trace["contribution"]["both"] += 1
        elif f["semantic_rank"] is not None:
            trace["contribution"]["semantic_only"] += 1
        else:
            trace["contribution"]["keyword_only"] += 1
    trace["result_count"] = len(results)
    trace["unresolved_hits"] = unresolved
    return _result("ok", f"Returned {len(results)} passage(s). " + _SCORE_NOTE, trace, results)
