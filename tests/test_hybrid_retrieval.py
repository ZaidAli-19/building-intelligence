"""Offline tests for hybrid fusion and run_hybrid (no network)."""

import pytest
from fastapi import HTTPException

from building_with_rag.contracts import QueryRequest, RetrievedChunk
from building_with_rag.retrieval import hybrid, semantic


def _hits(*ids):
    return [{"chunk_id": c, "score": 1.0 - i / 10} for i, c in enumerate(ids)]


def test_chunk_in_both_routes_ranks_first_with_rrf_score() -> None:
    fused = hybrid.fuse(_hits("a", "b"), _hits("c", "a"), limit=5)
    top = fused[0]
    assert top["chunk_id"] == "a"
    assert top["semantic_rank"] == 1 and top["keyword_rank"] == 2
    assert top["fused_score"] == pytest.approx(1 / 61 + 1 / 62)
    assert [f["fused_rank"] for f in fused] == [1, 2, 3]


def test_single_route_chunk_leaves_other_route_none() -> None:
    fused = {f["chunk_id"]: f for f in hybrid.fuse(_hits("a"), _hits("b"), limit=5)}
    assert fused["a"]["keyword_score"] is None and fused["a"]["keyword_rank"] is None
    assert fused["b"]["semantic_score"] is None and fused["b"]["semantic_rank"] is None


def test_tie_order_is_deterministic() -> None:
    # equal fused score: semantic rank wins (missing last), then chunk_id
    fused = hybrid.fuse(_hits("z"), _hits("a"), limit=5)
    assert [f["chunk_id"] for f in fused] == ["z", "a"]
    again = hybrid.fuse(_hits("z"), _hits("a"), limit=5)
    assert again == fused
    both_keyword = hybrid.fuse([], _hits("b") + _hits("a"), limit=5)
    assert [f["chunk_id"] for f in both_keyword][:1] == ["b"]


def _request() -> QueryRequest:
    return QueryRequest(question="criminal breach of trust", pattern="hybrid", limit=2)


def _fake_env(monkeypatch):
    monkeypatch.setattr(hybrid, "_ready", False)
    monkeypatch.setattr(semantic, "_clients", lambda: (object(), object()))
    monkeypatch.setattr(semantic, "embed_question", lambda voyage, q: [0.0])
    monkeypatch.setattr(
        semantic, "vector_search", lambda db, v, limit, cands, f: _hits("a", "b")
    )
    monkeypatch.setattr(hybrid, "keyword_search", lambda db, q, filters, depth: _hits("b", "c"))

    def resolve(db, hits):
        return [
            RetrievedChunk(chunk_id=h["chunk_id"], section_id="bns:316", act="BNS_2023",
                           text="t", heading="h", score=h["score"])
            for h in hits
        ], 0

    monkeypatch.setattr(semantic, "resolve_hits", resolve)


def test_run_hybrid_sets_fused_score_and_route_fields(monkeypatch) -> None:
    _fake_env(monkeypatch)
    monkeypatch.setattr(hybrid, "_ensure_ready", lambda db: None)
    result = hybrid.run_hybrid(_request())
    assert result.pattern == "hybrid" and result.status == "ok"
    assert len(result.results) == 2
    first = result.results[0]
    assert first.chunk_id == "b"  # found by both routes
    assert first.score == first.fused_score
    assert first.semantic_rank == 2 and first.keyword_rank == 1 and first.fused_rank == 1
    assert result.trace["contribution"] == {"both": 1, "semantic_only": 1, "keyword_only": 0}
    assert result.trace["fusion"]["k"] == 60


def test_missing_keyword_index_is_not_ready(monkeypatch) -> None:
    _fake_env(monkeypatch)
    monkeypatch.setattr(semantic, "_ensure_ready", lambda db: None)

    class Chunks:
        def list_search_indexes(self):
            return []

    class Db:
        def __getitem__(self, name):
            return Chunks()

    monkeypatch.setattr(semantic, "_clients", lambda: (Db(), object()))
    with pytest.raises(HTTPException) as exc:
        hybrid.run_hybrid(_request())
    assert exc.value.status_code == 503
    assert exc.value.detail["code"] == "retrieval_not_ready"
    assert "keyword_index" in exc.value.detail["message"]
