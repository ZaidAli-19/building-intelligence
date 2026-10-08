"""Bounded, labelled evidence context built from retrieval results (semantic or hybrid)."""

from building_with_rag.contracts import RetrievedChunk

MAX_PASSAGES = 5
MAX_CHARS = 12_000


def build_context(results: list[RetrievedChunk]) -> tuple[list[dict], int, int]:
    """Return (entries, omitted_count, total_chars). Passages are never cut mid-text."""
    entries: list[dict] = []
    chars = 0
    for chunk in results[:MAX_PASSAGES]:
        if chars + len(chunk.text) > MAX_CHARS:
            break
        entries.append({
            "label": f"E{len(entries) + 1}",
            "chunk_id": chunk.chunk_id,
            "section_id": chunk.section_id,
            "act": chunk.act,
            "act_label": chunk.act_label,
            "heading": chunk.heading,
            "chapter": chunk.chapter,
            "section_number": chunk.section_number,
            "status": chunk.status,
            "source_pdf": chunk.source_pdf,
            "needs_review": chunk.needs_review,
            "text": chunk.text,
        })
        chars += len(chunk.text)
    return entries, len(results) - len(entries), chars
