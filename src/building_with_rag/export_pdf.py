"""PDF of a `rag-mcp` retrieval (the MCP server's sections only) plus a small in-memory store.

Built only when the user asks for it with `/pdf`; nothing is written to disk.
"""

import io
import threading
import uuid
from collections import OrderedDict
from html import escape

import pymupdf

from building_with_rag.contracts import RetrievedChunk

MAX_EXPORTS = 20
_NOTE = (
    "Source: Legal Sections MCP server (BNS 2023 / IPC 1860 corpus). Section status describes "
    "the source document and does not claim current legal applicability."
)
_CSS = (
    "body{font-family:sans-serif;font-size:10pt} h1{font-size:16pt} "
    "h2{font-size:11.5pt;margin-top:12pt;margin-bottom:2pt} "
    ".meta{color:#555;font-size:8.5pt;margin-top:0} p{margin-top:3pt;margin-bottom:3pt} "
    ".note{color:#444;font-size:8.5pt}"
)

_store: OrderedDict[str, bytes] = OrderedDict()
_lock = threading.Lock()


def _paragraphs(text: str) -> str:
    return "".join(f"<p>{escape(p)}</p>" for p in text.split("\n") if p.strip())


def _html(question: str, path: str, results: list[RetrievedChunk], message: str,
          listing: bool) -> str:
    parts = [
        f"<h1>Legal sections</h1><p class='meta'>Question: {escape(question)}</p>",
        f"<p class='meta'>Path: {escape(path)} · {len(results)} section(s)</p>",
    ]
    if listing:
        parts.append(_paragraphs(message))
    else:
        if message:
            parts.append(f"<p class='note'><b>Note:</b> {escape(message)}</p>")
        for c in results:
            meta = [c.section_id, c.act_label or c.act]
            if c.chapter:
                meta.append(f"Chapter {c.chapter}" + (f" - {c.chapter_title}" if c.chapter_title else ""))
            if c.status:
                meta.append(c.status)
            if c.needs_review:
                meta.append("needs review")
            parts.append(
                f"<h2>Section {c.section_number}: {escape(c.heading or '(no heading)')}</h2>"
                f"<p class='meta'>{escape(' · '.join(meta))}</p>{_paragraphs(c.text)}"
            )
    parts.append(f"<p class='note'>{escape(_NOTE)}</p>")
    return "".join(parts)


def build_pdf(question: str, path: str, results: list[RetrievedChunk], message: str = "",
              listing: bool = False) -> bytes:
    story = pymupdf.Story(html=_html(question, path, results, message, listing), user_css=_CSS)
    buffer = io.BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    page = pymupdf.paper_rect("a4")
    where = page + (48, 48, -48, -48)
    more = 1
    while more:
        device = writer.begin_page(page)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()
    return buffer.getvalue()


def put(data: bytes) -> str:
    """Keep the newest MAX_EXPORTS PDFs in memory; return an unguessable id."""
    export_id = uuid.uuid4().hex
    with _lock:
        _store[export_id] = data
        while len(_store) > MAX_EXPORTS:
            _store.popitem(last=False)
    return export_id


def get(export_id: str) -> bytes | None:
    with _lock:
        return _store.get(export_id)
