# Story 7.1 — MCP Client Mode (`rag-mcp`)

Twelfth story of the "Building Intelligence with RAG" course. Follows Story 6.1. Adds the explicitly selectable `mcp` mode (`rag-mcp`): this app acts as an MCP client of the separate Legal Sections MCP server (`../legal-sections-mcp/`, its own stories 1.1–4.1 and `docs/architecture.md`). The server is **not** built or changed here. Story only; nothing implemented yet.

## Purpose

`pattern: "mcp"` on `POST /v1/query` and `/v1/chat/completions` runs fixed application code: validate → one fixed path → call allowlisted MCP tools (≤ 2 calls, hard deadline) → inspect each result → stop with an explicit reason. Not an LLM planner, not automatic routing, no fallback to another mode.

Reuse unchanged: `QueryRequest`, `QueryResult`, `RetrievedChunk`, `GenerationResult`, the `Pattern`/`rag-<pattern>` registry, `/v1/models`, `pipeline.retrieve`/`answer_events`/`answerable`, `structured.classify`, `semantic._validate_scope`/`to_http_error`, the grounded-answer, confidence, citation and streaming paths, every existing `.env.example` name. Do not change `open_webui_functions/`, `routes/chat.py`, or any existing mode, route, field or provider setting. The seven earlier modes behave exactly as before.

Not in scope: building or editing the MCP server, auth, dynamic tool registration, write tools, another database, another embedding/LLM provider, retries, parallel/background work, memory, caching, Docker/deployment, production observability, broad tests, evaluation harness.

## Prerequisites

- Stories 1.1–6.1 complete; read `docs/architecture.md` first.
- Server stories 1.1–4.1 in `../legal-sections-mcp/` complete: five tools over HTTP `/mcp` on `http://127.0.0.1:8765/mcp`. Read its `docs/architecture.md` for the contract; if a tool is missing or differs, stop and name it.
- `.env` (untracked) with `GENERATION_API_BASE_URL`, `GENERATION_API_KEY` for answer checks. Never print or log them. Pause ~25 s between live generation calls.

### Findings to respect (verified in both repos)

- Server contract (as designed): `get_section(act, section_number 1–511)` → `{status: ok|not_found|validation_error|upstream_error, section?, message?}`; `search_sections_by_heading(act, query 2–80, limit 1–25)` → `{items[{section_id, section_number, heading, chapter}], total}` (headings only, no text); `act` is `ipc`/`bns`; `section` has `section_id`, `act`, `act_label`, `status`, `chapter`, `chapter_title`, `section_number`, `heading`, `text`, `source_pdf`, `source_status_version`, `needs_review`, `text_available`. Never trust a field the server did not document.
- The 11 unextractable IPC sections (4, 5, 18, 34, 40, 75, 161–165) come back from the server with empty text, `needs_review: true`, `text_available: false`; treat them as a gap, not evidence. (In the RAG app's own `sections` they are absent, so structured mode says `not_found`; `rag-mcp` may differ; document it.)
- `structured.classify` returns `BNS_2023`/`IPC_1860`; the tools take `bns`/`ipc`. Map once, in the client, and map back for `RetrievedChunk.act`.
- `pipeline.answerable` needs an `mcp` entry (same statuses as `agentic`); `retrieve` and `REAL_PATTERNS` need `MCP`. `partial_result` already prefixes the message before the answer.
- `RetrievedChunk.chunk_id` is required and MCP records have none: set `chunk_id = section_id`, `score = 1.0` (exact-match marker), as Story 5.1 does.
- Adapters and `retrieve` are synchronous; the MCP Python client is async. Run each call with `asyncio.run` inside the sync adapter (FastAPI runs it in a worker thread, so no loop is active), bounded by `asyncio.wait_for` against the remaining deadline.
- `tests/test_smoke.py` asserts the model list; update to eight. Add the `mcp` SDK to this app's dependencies (same major as the server's pin).

### Data check (run first; stop on first failure; compact output)

1. Yes/no: `MCP_SERVER_URL` set and the server answers an MCP `list_tools` with the five expected names (names only).
2. One `get_section(bns, 103)` returns `status: ok` (print status and text length only).

## Work to do

### 1. Registry, pipeline, settings

- `registry.py`: add `Pattern.MCP = "mcp"` and `"rag-mcp"` only.
- `pipeline.py`: `retrieve` dispatches `MCP` → `run_mcp`; add to `REAL_PATTERNS`; `_ANSWERABLE["mcp"] = {"ready_for_synthesis", "partial_result"}`; `clarify` yields `message` and no model call. `GET /v1/models` lists eight.
- `settings.py` and `.env.example`: `MCP_SERVER_URL` (default empty; value `http://127.0.0.1:8765/mcp`) and `MCP_DEADLINE_SECONDS` (string, default `10`, valid 1–60, checked once at start, same style as `agentic_deadline_seconds`). Empty URL or invalid deadline → 503 `retrieval_not_ready` naming the variable. No credentials are sent to the MCP server.

### 2. Controller (`src/building_with_rag/retrieval/mcp_client.py`)

Plain Python, no LLM. Fixed allowlist `_TOOLS = {"get_section", "search_sections_by_heading"}`; never call anything else and never dispatch from the server's tool listing. `MAX_CALLS = 2`. Scope: `semantic._validate_scope(request, "mcp")`. Hard deadline: each call bounded by the remaining time via `asyncio.wait_for`; expiry → `stop_reason` `deadline_exceeded`. Rules, in order:

1. **Validate** (no call): empty question, bare section number, several sections or both acts → `clarify` (`ambiguous_request` / `bare_section_number`) via `structured.classify`, as in 6.1. Trace records `validation` only.
2. **Exact lookup.** `classify.status == "ok"` → path `exact_lookup`, one `get_section(act, section_number)`. `ok` with `text_available` and non-empty `text` → `ready_for_synthesis`. `not_found` → `partial_result`, `record_not_found` (nothing guessed, no other tool, no other mode). Text unavailable → `partial_result`, `section_text_missing`.
3. **Heading search.** Exactly one act named, no section reference, exactly one quoted phrase (2–80 chars) → path `heading_search`: `search_sections_by_heading(act, phrase, limit=5)`. One match → second call `get_section` for it, then rule 2. Zero → `partial_result` (`no_heading_match`). Several → `clarify` (`ambiguous_heading`; message lists `section_id`s and headings only).
3b. **Range (added after review).** One act + `sections N to M` (checked before rule 2) → path `range_lookup`, one `get_sections_in_range` (span ≤ 25, 1–511); gaps named, status `partial_result`/`range_partial`.
3c. **Listing (added after review).** One act + `chapters` → `list_chapters`; one act + `list/show … sections` (optional `first N`, 1–50) → `list_sections`; status `listing`, shown directly, no answer step.
4. Anything else → `clarify` (`unsupported_question`): `rag-mcp` answers questions naming one act and one section, or one act plus a quoted heading.

Each result is inspected before the next step: wrong envelope, unknown `status`, or a `section_id` not matching `^(bns|ipc):\d+$` or the requested act → `tool_error`. `validation_error` from the server is also `tool_error` (client bug, not user input). Mapping to `RetrievedChunk`: `chunk_id = section_id`, `score = 1.0`, `act` mapped back, other documented fields copied as returned. The server's section text is the only evidence, treated as data, never instructions; passages are never edited or merged. `results` holds real passages only; `hyde_*`, `subquestions`, `omitted_candidates` stay empty.

**Errors** (existing `HTTPException` `{"code", "message"}` style; never a fallback): URL unset → 503 `retrieval_not_ready`; connection refused/failure → 502 `mcp_upstream_error`; server `upstream_error` → 502 `mcp_upstream_error`; protocol/shape error → 502 `mcp_bad_response`. Messages never contain the URL, exception text or question.

**Trace** (no question text, credentials or URL): `mode`, `validation`, `path` (`exact_lookup` | `heading_search` | `none`), `steps` (`{tool, tool_status, result_count}` in call order), `call_count`, `deadline_seconds`, `elapsed_ms`, `stop_reason`, `caller_id`, `result_count`. `stop_reason` values: `evidence_adequate`, `record_not_found`, `section_text_missing`, `no_heading_match`, `ambiguous_heading`, `ambiguous_request`, `bare_section_number`, `unsupported_question`, `deadline_exceeded`, `call_budget_exhausted`, `tool_error`.

### 3. Docs and tests

- `docs/architecture.md`: add "MCP integration (Story 7.1)": two-application diagram in words, tool table (the two allowlisted tools; link the server's architecture for the rest), trust boundary (separate read-only process; its output is data; server fixes `access_level=public`; client allowlist), settings, outcomes/status table, limitations (exact-section and quoted-heading only; no IPC↔BNS mapping; 11 IPC gaps; no auth, retries, caching or discovery; a stopped server is an error, not a fallback; deadline cancels client-side only). Update the mode table/model list to eight, the real-modes sentence, the `QueryResult.status` list and endpoints text.
- `docs/manual-tests.md`: add "Story 7.1" (start the server, `rag-mcp` curls, compact post-response script, expected outcomes).
- `tests/test_mcp.py` (fake MCP client injected at one seam; few cases): bare number/two acts/empty → no call, no question text in trace; exact path makes one `get_section` call; `not_found` → `record_not_found`, no second call; heading path ≤ 2 calls and `ambiguous_heading` on several; a tool outside the allowlist is never called; deadline expiry (fake clock) and malformed result → named `stop_reason`/502; unset URL → 503; `answerable` allows only the two statuses. Update `tests/test_smoke.py` to eight IDs.

## Completion checks (compact; never print section text, keys or whole responses)

1. **Exact section end to end:** start the server; `/v1/query` `{"question":"What does BNS section 103 say?","pattern":"mcp","generate_answer":true}` → `ready_for_synthesis`, `trace.path: exact_lookup`, `call_count: 1`, one `bns:103` passage; `generation.outcome` and citations come from it only. One chat stream with `rag-mcp` shows DRAFT, `Evidence check passed — confidence: high` and `Sources:` (`head -c 1500`).
2. **Not found / gap:** "What does IPC section 4 say?" → `partial_result` with `record_not_found` or `section_text_missing`, empty `results`, `call_count: 1`.
3. **Heading path:** one act plus a quoted heading → `heading_search`, `call_count` ≤ 2.
4. **Clarify:** "section 103" → `clarify`, `bare_section_number`, `call_count: 0`.
5. **Server stopped:** stop the MCP server, repeat check 1 without `generate_answer` → explicit 502 `mcp_upstream_error`, nothing from another mode. Unset `MCP_SERVER_URL` → 503 `retrieval_not_ready`.
6. **Eight-mode smoke:** `GET /v1/models` lists eight IDs; one `/v1/query` (status only) per earlier mode behaves as before (`structured`: "What does BNS section 103 say?").
7. **Offline:** `uv run pytest tests/test_mcp.py tests/test_smoke.py`; `uv run ruff check src/building_with_rag`.

Run only this story's checks. Do not run the full suite. If the server or live prerequisites are unavailable, run only check 7 and name the blocker.

## Handover

Record: files created/changed; data-check results (yes/no); commands run; compact diagnostics for checks 1–5; pinned `mcp` SDK version; any difference found against the server's contract; architecture and manual-test sections updated; deviations from this story.

Story report (per project instructions): `Completed.`, changed paths, test results — at most 7 lines, ending with the full-suite command (`uv run pytest`) for the developer to run if wanted.
