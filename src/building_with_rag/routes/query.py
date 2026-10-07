"""Query endpoint: semantic is handled by retrieval; other modes use run_pattern."""

from fastapi import APIRouter

from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.registry import Pattern, run_pattern
from building_with_rag.retrieval.semantic import run_semantic

router = APIRouter()


@router.post("/v1/query")
def query(request: QueryRequest) -> QueryResult:
    if request.pattern == Pattern.SEMANTIC:
        return run_semantic(request)
    payload = run_pattern(request.pattern, request.question, request.caller_id)
    return QueryResult(**payload)
