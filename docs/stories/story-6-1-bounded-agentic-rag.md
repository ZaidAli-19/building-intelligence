# Story 6.1 — Bounded Agentic RAG

Eleventh and final story of the "Building Intelligence with RAG" course. Follows Story 5.2. Adds the explicitly selectable `agentic` mode (`rag-agentic`): a deterministic, read-only controller over three existing adapters, with a hard call budget and deadline.

## Purpose

`pattern: "agentic"` on `POST /v1/query` and `/v1/chat/completions` runs fixed application code: validate → pick one fixed path → call allowlisted adapters (at most two calls, at most one HyDE) → inspect each result → stop with an explicit reason. It is not an LLM planner, not automatic routing (no other mode ever reaches it), and not a new application.

Allowlist (fixed, no dynamic registration): `structuredLookup` → `retrieval.structured.run_structured`; `semanticRetrieve` → `retrieval.semantic.run_semantic`; `hydeRetrieve` → `retrieval.hyde.run_hyde`. Nothing else: no raw database, shell, HTTP, write, browse, retry, parallel, background or recursive work, memory, re-ranking, GraphRAG, evaluation harness.

Reuse, do not rename or replace: `QueryRequest`, `QueryResult`, `RetrievedChunk`, `GenerationResult`, the `Pattern`/`rag-<pattern>` registry, `/v1/models`, `pipeline.retrieve`/`answer_events`/`answerable`, `semantic._validate_scope`/`_effective_filters`, `to_http_error`, the shared grounded-answer, confidence, citation and chat streaming paths, and every `.env.example` provider name. Do not change `open_webui_functions/` (its Pipe and valves already offer `agentic`). Prior modes behave exactly as before.

## Prerequisites

- Stories 1.1–5.2 complete. Read `docs/architecture.md` first.
- `.env` (untracked) with `MONGODB_URI`, `VOYAGE_API_KEY`, `GENERATION_API_BASE_URL`, `GENERATION_API_KEY`. Never print or log them. Pause ~25 s between live calls.
- `sections`, `chunks`, `embeddings` populated; `vector_index` queryable.

### Findings to respect (verified in the repo)

- `agentic`/`rag-agentic` is **not** in `registry.py` yet (six modes). Add `Pattern.AGENTIC` and `"rag-agentic"`; `/v1/models` then lists seven. The brief calls the ID "existing": it is only in the Open WebUI valves, not the API.
- Open WebUI valves also list `graph` and `reranked`, which the API does not accept; leave them alone.
- No controller trace contract exists. Use `QueryResult.trace` (already a free-form dict) — no new top-level field.
- Every adapter raises HTTP errors (`HTTPException`) for missing configuration or upstream failure; an adapter's `QueryResult.status` carries the soft outcomes (`ok`, `not_found`, `clarification_needed`, `no_results`, `hyde_unavailable`).
- `pipeline.answerable` limits which statuses reach the answer step per mode; `agentic` needs an entry (see §3).
- `docs/manual-tests.md` already has a Story 6.1 recipe (keys `path`, `stop_reason`, `call_count`, `hyde_calls`, `steps[adapter, tool_status, adequate]`; `AGENTIC_DEADLINE_SECONDS`). Match it, and fix it if the built shape differs.
- `tests/test_smoke.py` asserts the model list; update the count to seven.

### Data check (run first; stop on first failure; compact output)

1. Yes/no only: the four `.env` variables above set.
2. `GET /healthz` ok; one `semantic` query returns `ok`.

## Work to do

### 1. Controller (`src/building_with_rag/retrieval/agentic.py`)

`run_agentic(request) -> QueryResult`, plain Python, no LLM. A tiny `_TOOLS` dict maps the three allowlisted names to the functions above; the controller can call nothing else.

- **Setting:** `AGENTIC_DEADLINE_SECONDS` (non-provider application setting; default 20, valid 1–120) added to `settings.py` and `.env.example`; invalid → 503 `retrieval_not_ready` naming it. Checked once at start, before any adapter call.
- **Scope:** `semantic._validate_scope(request, "agentic")` (demo `caller_id`, no `required_acts`, no `chapter`); every adapter call receives a copy of the same `QueryRequest` (same filters, `limit`, `caller_id`; server-fixed `access_level=public` applies inside each adapter). The controller never widens scope or sets filters.
- **Budget and deadline:** `MAX_CALLS = 2`, `MAX_HYDE_CALLS = 1`. Before each call check the budget and a monotonic deadline; on expiry or exhaustion stop with `stop_reason` `deadline_exceeded` or `call_budget_exhausted` (no further call; a call already in flight is not interrupted, its result is still inspected). No retries, threads or async.
- **Tool failure:** an adapter `HTTPException` or a result not matching its expected shape (wrong `pattern`, a passage without `chunk_id`/`section_id`/`text`) stops the controller with `stop_reason` `tool_error` (record the adapter name and HTTP status only, never the message body); `status: "partial_result"` if earlier steps produced usable evidence, otherwise re-raise the HTTP error for 503/502 (no fabricated evidence, no fallback).

Fixed rules, in order:

1. **Validate.** Reject with `clarify` and no tool call: a bare section number (a section reference such as `section 103`/`§103`/`s. 103` with no act token, or a number with no act), several section numbers or both acts in one request, an empty or whitespace-only question after trimming. `stop_reason` `bare_section_number` / `ambiguous_request`. Trace records `validation` (`passed` | the reason) only — no question text.
2. **Exact lookup.** If the question names exactly one act (`BNS`/`IPC` or full act name) and one integer section (reuse `structured.classify`; `status == "ok"`): path `exact_lookup`, one `structuredLookup` call. `ok` with a record → `ready_for_synthesis`. `not_found` → `partial_result`, `stop_reason` `record_not_found` (no semantic fallback, no guess). `clarification_needed` from the adapter → `clarify`.
3. **Vocabulary mismatch (non-exact).** Prepared deterministic rule: the question contains one of a small fixed list of lay cues (e.g. `secretly takes`, `snatch`, `grab`, `cheat`, `kill`, `hit`, `hurt`, `cheated`, `steals`) **and** none of the legal terms the corpus uses for them (`theft`, `murder`, `robbery`, `cheating`, `hurt`, `culpable homicide`, …; keep both lists short, in one module-level constant each, case-insensitive whole-word match). When it applies: path `vocab_mismatch`, call `semanticRetrieve` first; if the direct result fails the adequacy rule, call `hydeRetrieve` once (the last allowed call). Final evidence = the passages of the last adapter result that was adequate, else of the best available non-empty result (HyDE result if non-empty, else the semantic one); passages are copied as returned, never merged, re-ranked or edited.
4. **Ordinary question.** Path `semantic`: one `semanticRetrieve` call.
5. **Adequacy rule (single, documented):** a result is adequate when its `status == "ok"` and it has at least one passage with non-empty `text` under the existing scope. Adequate → `ready_for_synthesis`. No usable passages → `partial_result` when some earlier step returned usable evidence, else `clarify` (message asks the user to rephrase or name the act and section).
- `hyde_unavailable`, `no_results`, `not_found` are inspected as inadequate, not errors.

Result: `QueryResult(pattern="agentic", status, message, trace, results)` with `status` one of `ready_for_synthesis`, `partial_result`, `clarify`. `results` = real passages only. `hyde_hypothetical_text_debug` is **always** `None` and `hyde_*` candidate lists, `subquestions`, `omitted_candidates` stay empty: the controller drops the HyDE adapter's debug text, candidate lists and trace before returning, and never logs it.

Trace (non-sensitive; no question text, credentials or hypothetical text): `mode`, `validation`, `path` (`exact_lookup` | `semantic` | `vocab_mismatch` | `none`), `steps` (list of `{adapter, tool_status, adequate, result_count}` in call order), `call_count`, `hyde_calls`, `deadline_seconds`, `elapsed_ms`, `stop_reason`, `caller_id`, `result_count`. Adapter traces are not copied in.

### 2. Registry and routing

- `registry.py`: add `Pattern.AGENTIC` and `"rag-agentic"` only.
- `pipeline.py`: `retrieve` dispatches `AGENTIC` → `run_agentic`; add to `REAL_PATTERNS`; `_ANSWERABLE["agentic"] = {"ready_for_synthesis", "partial_result"}` so only real passages reach the grounded answer; `clarify` yields `message` and no model call. For `partial_result` the answer step is prefixed with the message (existing `partial_answer` mechanism: extend its status check to include `partial_result`) so a gap is never hidden.
- Grounded answer, confidence, citations and chat streaming are unchanged; `routes/chat.py` needs no edit (model ID → pattern through the registry). Other modes never call `run_agentic`.

### 3. Docs and tests

- `docs/architecture.md`: add "Bounded agentic RAG (Story 6.1)" — allowlist, fixed rules, adequacy rule, budget/deadline, trace keys, statuses, outcomes, evidence boundary (only adapter-returned corpus passages; hypothetical text never exposed). Update the mode table/model list to seven, the `QueryResult.status` list, endpoints text, and the real-modes sentence. Limitations: no LLM planning, fixed lay-cue rule misses phrasing, at most two calls, adequacy is existence only (no relevance judgment), single adapter result drives synthesis, a call in flight is not interrupted by the deadline, no memory, no automatic routing, no production security or observability.
- `docs/manual-tests.md`: keep the Story 6.1 section aligned with the built trace.
- `tests/test_smoke.py`: model list is seven IDs.

## Completion checks (compact; never print passages, vectors, keys, hypothetical text or whole responses)

Inspect `status`, `trace.path`, `trace.steps`, `trace.stop_reason`, `trace.call_count`, ids only (jq or the manual-test Post-response script).

1. **Act-qualified lookup:** "What does BNS section 103 say?" → `ready_for_synthesis`, `path: exact_lookup`, `call_count: 1`, one `bns:103` passage. "What does IPC section 4 say?" (absent from the corpus) → `partial_result`, `record_not_found`, empty `results`, no semantic call.
2. **Ordinary explanation:** "What is the punishment for theft?" → `path: semantic`, one `semanticRetrieve` step adequate, `ready_for_synthesis`, `call_count: 1`; with `generate_answer: true`, `generation.outcome` and citations come from the returned passages only.
3. **Vocabulary mismatch:** "What happens to someone who secretly takes a neighbour's bike?" → `path: vocab_mismatch`; semantic first, `hyde_calls: 1` only if the direct step was inadequate; `call_count ≤ 2`; `hyde_hypothetical_text_debug` is null and no hypothetical text appears anywhere in the response.
4. **Ambiguous input:** "section 103" → `clarify`, `stop_reason: bare_section_number`, `call_count: 0`, empty `steps`, trace has no question text.
5. **Chat (one answerable case):** `rag-agentic` for check 2 (stream, `head -c 1500`) shows DRAFT, `Evidence check passed — confidence: high` and `Sources:`; for check 4 the clarification text only. Open WebUI lists and streams `rag-agentic`.
6. **Seven-mode smoke:** one `/v1/query` (status only) and one chat message per `semantic`, `hybrid`, `hybrid-reranked`, `structured` ("What does BNS section 103 say?"), `decomposition`, `hyde`, `agentic`; earlier response shapes unchanged; `GET /v1/models` lists seven.
7. **Offline tests** (`tests/test_agentic.py`; fake adapters; few cases): validation rejects bare number/two acts/empty with zero adapter calls and no question text in the trace; exact path calls only `structuredLookup` once, `not_found` → `record_not_found` with no semantic call; ordinary path one `semanticRetrieve`; vocabulary path calls HyDE only when the direct result is inadequate and never more than two calls or one HyDE; deadline expiry (fake clock) and tool error stop with the named `stop_reason`; HyDE debug text, candidate lists and adapter traces are absent from the result; filters/`caller_id` identical on every adapter call; `pipeline.answerable` allows only the two statuses.
8. `uv run ruff check src/building_with_rag` passes.

Run only this story's checks (new test file, `tests/test_smoke.py`, ruff). Do not run the full suite. If live prerequisites are unavailable, run only checks 7–8 and name the blocked prerequisite.

## Handover

Record: files created/changed; data-check results (yes/no); commands run; compact diagnostics for checks 1–4; the lay-cue and legal-term lists as built; architecture and manual-test sections updated; deviations from this story.

Story report (per project instructions): `Completed.`, changed paths, test results — at most 7 lines, ending with the full-suite command (`uv run pytest`) for the developer to run if wanted.
