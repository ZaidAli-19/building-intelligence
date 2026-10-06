from building_with_rag.models import QueryRequest, QueryResult
from building_with_rag.registry import MODES


def run_pattern(request: QueryRequest) -> QueryResult:
    """Single path for /v1/query and /v1/chat/completions. Placeholders for now."""
    mode = MODES[request.pattern]
    return QueryResult(
        pattern=mode.pattern,
        status="not_implemented",
        message=f"The '{mode.pattern}' pattern is not implemented yet.",
        trace={"model_id": mode.model_id, "caller_id": request.caller_id},
    )
