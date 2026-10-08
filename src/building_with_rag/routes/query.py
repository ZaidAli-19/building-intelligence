"""Query endpoint: same retrieve -> answer path as chat, returned as one QueryResult."""

from fastapi import APIRouter

from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.pipeline import REAL_PATTERNS, answer_events, retrieve

router = APIRouter()

_OUTCOME_MESSAGES = {
    "answered": "Answer generated from the cited passages; evidence check passed (confidence: high).",
    "insufficient_evidence": "The retrieved evidence is insufficient to answer; no answer given.",
    "unavailable": "Answer generation is unavailable; retrieved passages are still returned.",
    "malformed": "No validated answer: the generated text failed the evidence check or was invalid; "
    "passages are still returned.",
}
_LOW_CONFIDENCE = (
    "The draft failed the evidence check (confidence: low); see generation.draft_answer and "
    "generation.issues. It is not a validated answer."
)


@router.post("/v1/query")
def query(request: QueryRequest) -> QueryResult:
    result = retrieve(request)
    if request.pattern in REAL_PATTERNS and request.generate_answer:
        for kind, payload in answer_events(request.question, result):
            if kind == "final":
                result.generation = payload
        generation = result.generation
        sentence = _LOW_CONFIDENCE if generation.confidence == "low" else _OUTCOME_MESSAGES[generation.outcome]
        result.message += " " + sentence
    return result
