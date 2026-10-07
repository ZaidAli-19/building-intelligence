"""Query endpoint: semantic is handled by retrieval; other modes use run_pattern."""

from fastapi import APIRouter

from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.generation.answer import generate_answer
from building_with_rag.registry import Pattern, run_pattern
from building_with_rag.retrieval.semantic import run_semantic

router = APIRouter()

_OUTCOME_MESSAGES = {
    "answered": "Answer generated from the cited passages.",
    "insufficient_evidence": "The retrieved evidence is insufficient to answer; no answer given.",
    "unavailable": "Answer generation is unavailable; retrieved passages are still returned.",
    "malformed": "The model output was invalid and was discarded; passages are still returned.",
}


@router.post("/v1/query")
def query(request: QueryRequest) -> QueryResult:
    if request.pattern == Pattern.SEMANTIC:
        result = run_semantic(request)
        if request.generate_answer:
            result.generation = generate_answer(request.question, result.results)
            result.message += " " + _OUTCOME_MESSAGES[result.generation.outcome]
        return result
    payload = run_pattern(request.pattern, request.question, request.caller_id)
    return QueryResult(**payload)
