# Handout 7.1 — Legal Sections MCP Server and RAG MCP Client

Paste this prompt into your coding assistant from the capstone repository:

---

Create `docs/stories/story-7-1-mcp-integration.md`. Create the story only; do **not** implement it. Read `AGENTS.md`, `CLAUDE.md`, `docs/architecture.md`, all prior stories (especially 5.1 and 6.1), `contracts.py`, `registry.py`, `pipeline.py`, `retrieval/structured.py`, `retrieval/agentic.py`, `ingestion/mongodb_schema.py`, `settings.py`, and `.env.example` first. Preserve all seven working modes, `POST /v1/query`, `GET /v1/models`, OpenAI-compatible `POST /v1/chat/completions`, response shapes, citations, and Open WebUI streaming.

Write a focused story in short sections: purpose, prerequisites, work to do, completion checks, and handover. It covers two deliverables that share one MongoDB database:

**A. New, separate Python application: the Legal Sections MCP server** (its own directory and `pyproject.toml` beside this repo, e.g. `../legal-sections-mcp/`; not inside `src/building_with_rag/`). It exposes an MCP server (official `mcp` Python SDK, `FastMCP`, streamable HTTP transport plus stdio for local debugging) and is **read-only** over the existing `building_with_rag` database, `sections` collection only (no `chunks`, `embeddings`, Voyage, LLM, or vector/keyword index; no writes; no schema or index changes). It reuses the existing `MONGODB_URI` / `MONGODB_DB_NAME` variable names and the fixed server-side filter `access_level = public`; callers can never widen it. At least five tools with typed inputs and structured outputs, for example:

1. `list_sections(act, chapter?, limit, offset)` — paginated section headings of one act (`ipc` or `bns`).
2. `get_section(act, section_number)` — one section's full record; act-qualified, never a bare number.
3. `list_chapters(act)` — chapter number/title and section range per chapter.
4. `search_sections_by_heading(act, query, limit)` — case-insensitive heading match (plain substring, escaped; no user-supplied regex or operators).
5. `get_sections_in_range(act, start, end)` — bounded range (hard cap on span).
6. Optional: `compare_ipc_bns(ipc_section)` — only if the corpus provides a mapping; otherwise omit and say so.

Each tool validates input (act ∈ {`ipc`,`bns`}, positive integers, caps on `limit`/span), returns `section_id` in the existing `bns:`/`ipc:` form, never returns Mongo `_id`/`provenance`, and returns a typed `not_found` / validation result instead of raising raw errors. Raw user text must never enter a Mongo predicate as an operator. Document known corpus gaps (11 IPC sections without extractable text) in tool results (`needs_review`/empty text flagged, not hidden).

**B. MCP client scenario inside this RAG application.** Add an explicitly selectable `mcp` mode and model ID `rag-mcp` to the shared registry (additive; no existing mode, route, request/result field, provider setting, or Open WebUI Function changes). It is deterministic application code, not an LLM planner and not automatic routing: validate → pick one fixed path → call the MCP server through an allowlist of its tools (at most two calls, hard deadline via a new application setting such as `MCP_SERVER_URL` / `MCP_DEADLINE_SECONDS`, added to `settings.py` and `.env.example`) → inspect each result → stop with an explicit reason. Map MCP results into the existing `QueryResult`/`RetrievedChunk` shapes so the grounded-answer, confidence, citation, and streaming paths are reused unchanged; the MCP server's section text is the only evidence. Reuse `structured.classify` for act/section detection. Missing/unreachable MCP server → explicit 503/502 in the existing error style, no silent fallback to another mode. Trace is non-sensitive (no question text, no credentials).

Do not add: auth systems, dynamic tool registration, write tools, a database other than the existing one, a second embedding/LLM provider, retries, parallel or background work, memory, caching layers, deployment/Docker, production observability, broad tests, or an evaluation harness.

Keep completion checks lightweight: start the MCP server alone and call each tool once with an MCP inspector or a minimal client (compact output, never full section text); `rag-mcp` exact-section question end to end through `/v1/query` and one chat stream; a not-found section; an invalid act; MCP server stopped → explicit error; `GET /v1/models` lists eight modes and the earlier seven are unchanged. Offline tests with a fake MCP client for the RAG side and `mongomock` or a fake collection for the server side, few cases, run only this story's tests plus ruff.

Update `docs/architecture.md` with an "MCP integration (Story 7.1)" section (two-application diagram in words, tool table, trust boundary, settings, outcomes, limitations) and `docs/manual-tests.md`. Follow `CLAUDE.md` style: compact docs, no pasted passages, story report limited to the stated short format. After creating the story, report its path only.
