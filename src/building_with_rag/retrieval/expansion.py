"""Shared helpers for decomposition and HyDE: one bounded generation step and one vector search."""

import json
import re

import httpx
from pymongo.errors import PyMongoError

from building_with_rag.contracts import RetrievedChunk
from building_with_rag.retrieval import semantic
from building_with_rag.retrieval.semantic import RetrievalError
from building_with_rag.settings import get_settings

STEP_TIMEOUT_SECONDS = 60.0
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def require_generation_settings() -> None:
    """503 naming the missing variable, before any provider call."""
    settings = get_settings()
    if not settings.generation_api_base_url:
        raise RetrievalError(503, "retrieval_not_ready", "GENERATION_API_BASE_URL is not set.")
    if not settings.generation_api_key:
        raise RetrievalError(503, "retrieval_not_ready", "GENERATION_API_KEY is not set.")


def generation_step(system: str, user: str) -> str:
    """One non-streamed chat-completions call; returns message text. Provider failure -> 502."""
    require_generation_settings()
    settings = get_settings()
    url = settings.generation_api_base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": settings.generation_model_name,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {settings.generation_api_key}"}
    try:
        resp = httpx.post(url, json=body, headers=headers, timeout=STEP_TIMEOUT_SECONDS)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
        raise RetrievalError(
            502, "retrieval_upstream_error", "Generation service request failed."
        ) from None
    return content if isinstance(content, str) else ""


def parse_json_object(text: str) -> dict | None:
    """Parse a JSON object reply (tolerating a code fence); None when not an object."""
    try:
        value = json.loads(_FENCE.sub("", text.strip()))
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def search(db, voyage, text: str, mongo_filter: dict, limit: int) -> list[RetrievedChunk]:
    """Embed text (voyage-3.5, query) and run the existing vector search; real passages only."""
    vector = semantic.embed_question(voyage, text)
    try:
        hits = semantic.vector_search(db, vector, limit, semantic.num_candidates(limit), mongo_filter)
        results, _ = semantic.resolve_hits(db, hits)
    except PyMongoError:
        raise RetrievalError(502, "retrieval_upstream_error", "MongoDB request failed.") from None
    return results
