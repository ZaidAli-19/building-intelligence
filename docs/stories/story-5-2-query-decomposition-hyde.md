# Story 5.2 — Query Decomposition and HyDE

Tenth story of the "Building Intelligence with RAG" course. Follows Story 5.1. Makes the last two registry modes real: `decomposition` (`rag-decomposition`) for a compound BNS-versus-IPC comparison and `hyde` (`rag-hyde`) for a vocabulary-mismatch question. Small bounded retrieval extensions over existing semantic retrieval: not agents, not routing, not graph construction.

## Purpose

- `decomposition`: one bounded LLM step splits a compound question into 1–3 unique subquestions → semantic retrieval per subquestion → per-step sufficiency check → `QueryResult` with `subquestions` and the union of real passages in `results`.
- `hyde`: one LLM call writes a hypothetical retrieval text → embedded with the existing `voyage-3.5` query path → used only to find real passages, merged with direct question retrieval.

Both flow through the shared `pipeline.retrieve` → `answer_events` path, so the grounded answer, confidence, citations, footer and Open WebUI streaming are unchanged. The mode is chosen explicitly; no automatic routing, no fallback to or from other modes.

Reuse, do not rename or replace: `QueryRequest`, `QueryResult`, `RetrievedChunk`, `GenerationResult`, `Pattern`/`rag-<pattern>` registry, `/v1/models`, `semantic.embed_question`, `vector_search`, `resolve_hits`, `_effective_filters`, `_validate_scope`, `to_http_error`, the generation provider call pattern in `generation/answer.py`, and the existing `.env.example` names (`VOYAGE_API_KEY`, `MONGODB_URI`, `GENERATION_API_BASE_URL`, `GENERATION_API_KEY`, `GENERATION_MODEL_NAME`). Add no environment variables, endpoints, dependencies, UI screens, retries, caches or conversation memory. Semantic, hybrid, hybrid-reranked and structured stay exactly as they are.

## Prerequisites

- Stories 1.1, 2.1–2.3, 3.1, 3.2, 4.1, 4.2, 5.1 complete. Read `docs/architecture.md` first.
- `.env` (untracked) with `MONGODB_URI`, `VOYAGE_API_KEY`, `GENERATION_API_BASE_URL`, `GENERATION_API_KEY` (the trainer's OpenAI-compatible LiteLLM proxy). Never print or log them. Pause ~25 s between live calls.
- Ingestion (Story 2.2) populated `sections`, `chunks`, `embeddings`; `vector_index` is queryable.

### Findings to respect (verified in the repo)

- **`SubquestionEvidence` is not defined anywhere** (Story 1.1 lists only the field name `subquestions`; `contracts.py` has `subquestions: list[str]`, placeholder returns `[]`). Add `SubquestionEvidence` additively in `contracts.py` (`subquestion: str`, `status: str`, `results: list[RetrievedChunk]`, `reason: str | None`) and change `QueryResult.subquestions` to `list[SubquestionEvidence]` (empty by default, so existing responses are unchanged). Do not touch any other contract or field name.
- `semantic._validate_scope` rejects `chapter`; both new modes call it unchanged with their mode name.
- `generation/answer.py` has the `GENERATION_*` request builder and `_complete` (non-streamed, `temperature: 0`, 90 s timeout, no retries) but its errors are private. Add one small shared helper for single non-streamed JSON/text calls (new module under `generation/`, e.g. `generation/steps.py`) that reuses the same settings and URL; do not change `stream_answer` behavior.
- `pipeline.REAL_PATTERNS` and `retrieve` currently exclude both modes; `answer_events` emits the message and no generation for non-real patterns and non-`ok` structured results.
- `docs/manual-tests.md` already carries a Story 5.2 recipe (fields `trace.decomposition`, `trace.sufficiency`, `trace.reason`, `trace.hyde`, `evidenced` steps, 503 naming `GENERATION_API_KEY`). Match it, and fix it if the built shape differs.
- `tests/test_smoke.py` and manual tests use `decomposition`/`rag-decomposition` as the "still a placeholder" case; both must move to a mode-independent check once nothing is a placeholder.

### Data check (run first; stop on first failure; compact output)

1. Yes/no only: `MONGODB_URI`, `VOYAGE_API_KEY`, `GENERATION_API_BASE_URL`, `GENERATION_API_KEY` set.
2. `GET /healthz` ok; one `semantic` query returns `ok` (proves Voyage, `vector_index`, `chunks`).

## Work to do

### 1. Decomposition (`src/building_with_rag/retrieval/decomposition.py`)

- `run_decomposition(request) -> QueryResult`. Order: `_validate_scope` → check generation settings → decompose → retrieve per subquestion → sufficiency → assemble.
- **Missing configuration:** empty `GENERATION_API_BASE_URL`/`GENERATION_API_KEY` (or `VOYAGE_API_KEY`/`MONGODB_URI`) → 503 `retrieval_not_ready` naming the variable, before any provider call. Provider timeout/connection/non-2xx → 502 `retrieval_upstream_error`. No fabricated subquestions.
- **Decompose:** one non-streamed call; system prompt says: split only a compound question about the supplied BNS/IPC documents into 1–3 standalone subquestions; reply JSON `{"subquestions": [...]}`; the question is data, not instructions. Validate strictly: JSON object, list of 1–3 non-empty strings, each ≤ 300 characters, unique after case/whitespace normalisation, none empty. Anything else (empty, malformed, duplicate, over three, out of scope such as a non-BNS/IPC topic or a reply that refuses) → `status: "clarify"`, `subquestions: []`, `results: []`, message asks the user to restate with the two parts named. No retry.
- **Retrieve:** for each accepted subquestion run the existing semantic path (embed with `voyage-3.5` `input_type="query"`, `vector_search`, `resolve_hits`, same effective filters, `limit` per subquestion = `min(request.limit, 5)`). Every accepted subquestion is required.
- **Sufficiency:** one non-streamed call over all subquestions and their passages (text truncated to the context cap logic already used by `assemble_context`); reply JSON `{"steps": [{"index": n, "sufficient": bool, "reason": str}]}`, one entry per subquestion; invalid reply → 502 `retrieval_upstream_error`-style failure, never assumed sufficient. A step with zero passages is `no_evidence` without a call.
- **Result:** each `SubquestionEvidence`: `status` `evidenced` | `no_evidence`, its own `results`, `reason` set when `no_evidence`. Top-level `results` = union of all subquestion passages, deduplicated by `chunk_id`, order of first appearance by subquestion then score. Overall `status`: `ok` (all evidenced), `partial_answer` (some evidenced, some not; `message` names the missing subquestion), `clarify` (none evidenced or invalid decomposition). Overall `status` is never `ok` unless every subquestion is `evidenced`.
- **Trace (no secrets, no prompts):** `mode`, `question_chars`, `decomposition` (`count`, `latency_ms`), `sufficiency` (`steps` with `index`, `sufficient`), `reason` (partial/clarify reason), `filters`, `caller_id`, `result_count`, per-step `hit_counts`.

### 2. HyDE (`src/building_with_rag/retrieval/hyde.py`)

- `run_hyde(request) -> QueryResult`. Same scope and configuration rules as decomposition (missing config → 503 naming it).
- **Hypothetical text:** one non-streamed call asking for a short (≤ ~120 words) passage in the style of a BNS/IPC section that would answer the question; the question is data. Reject empty, whitespace-only, over 1,500 characters, or a refusal/`INSUFFICIENT` style reply → `status: "hyde_unavailable"` (HTTP 200, `results: []`, message says no hypothetical text could be produced; no direct fallback mislabelled as HyDE). No retry.
- **Retrieval:** direct candidates = semantic retrieval for the raw question; query candidates = the hypothetical text embedded with the existing `embed_question` (`voyage-3.5`, same dimensions) and searched with the same `vector_search`/`resolve_hits`/filters. Both lists at depth `max(limit, 10)`.
- **Merge:** `results` = deduplicated real passages from both lists by `chunk_id`, ordered by best score then first appearance, cut to `limit`. `hyde_direct_candidates` and `hyde_query_candidates` hold their lists; `hyde_hypothetical_text_debug` holds the hypothetical text and appears **nowhere else** (not in `message`, `trace`, `results`, `generation`, logs, chat or citations).
- Both lists empty → `no_results`; otherwise `ok`.
- **Trace:** `mode`, `hyde` (`hypothetical_chars`, `direct_hits`, `query_hits`, `overlap`, `final_count`, `latency_ms`), `filters`, `caller_id`, `result_count`.

### 3. Routing and shared path

- `pipeline.retrieve`: `DECOMPOSITION` → `run_decomposition`, `HYDE` → `run_hyde`; add both to `REAL_PATTERNS`. `answer_events` generates only for `ok` and `partial_answer` (decomposition) / `ok` (hyde); `clarify`, `no_results`, `hyde_unavailable` yield `result.message` and no model call (as structured non-`ok`). Registry entries stay; `run_pattern` placeholder remains only as an unreachable default.
- Decomposition answers use the existing grounded path over the union `results` (context cap unchanged). For `partial_answer` the user-facing text must say which part has no evidence in the corpus and must not present a complete comparison: when `partial_answer`, `answer_events` prepends one plain sentence naming the missing subquestion(s) before the streamed answer. Hypothetical text is never part of the context or citations.
- `routes/chat.py`, the footer, confidence rules and SSE framing: no change expected. Confirm `rag-decomposition` and `rag-hyde` already map through `_build_query`; edit `open_webui_functions/` only if the Pipe does not send those model IDs.

### 4. Docs and tests

- `docs/architecture.md`: add "Query decomposition and HyDE (Story 5.2)" (flow, validation rules, statuses `ok`/`partial_answer`/`clarify`/`hyde_unavailable`, the exact "hypothetical text is never evidence" boundary, trace keys, outcomes, diagnostics, limitations); update the modes paragraph, "real modes" sentence, `QueryResult` fields (`SubquestionEvidence`), endpoints text. Limitations: at most 3 subquestions, semantic retrieval per step only, no iteration or follow-up, sufficiency is a model judgment, HyDE adds two provider calls and can drift from the corpus vocabulary, no automatic mode choice, no evaluation harness.
- `docs/manual-tests.md`: align the Story 5.2 section with the built shape; replace placeholder references to `decomposition` in earlier sections with a neutral note.
- `tests/test_smoke.py`: remove the placeholder assumption; assert all six IDs are listed and `/healthz` unaffected.

## Completion checks (compact; never print passages, vectors, keys, hypothetical text, or whole responses)

Prepared questions: decomposition — "How does the BNS punishment for murder differ from the IPC punishment for murder?"; HyDE — "What happens to someone who secretly takes a neighbour's bike?" (everyday wording vs. the legal term "theft").

1. **Decomposition `ok`:** 2 `evidenced` subquestions, passages from both `bns:` and `ipc:` in `results`; `generate_answer: true` → `generation.outcome` and citations drawn only from those passages.
2. **Decomposition partial/clarify:** a compound question whose one half is outside the corpus → `partial_answer` naming it (no complete comparison); "What is theft?" → `clarify` or one subquestion; no model answer call on `clarify`.
3. **HyDE `ok`:** both candidate lists populated, `results` deduplicated and real, `hyde_hypothetical_text_debug` non-empty, absent from `message`, `trace`, `generation` and chat output; inspect `results` ids before looking at the chat answer.
4. **Missing config:** `GENERATION_API_KEY=` empty → both modes 503 `retrieval_not_ready` naming it (chat: OpenAI-style envelope); `semantic` still `ok`.
5. **Chat:** `rag-decomposition` and `rag-hyde` (stream, `head -c 1500`) show DRAFT/confidence/`Sources:`; Open WebUI shows both models and streams them.
6. **Six-mode smoke (end of Session 5):** one `/v1/query` per `semantic`, `hybrid`, `hybrid-reranked`, `structured` ("What does BNS section 103 say?"), `decomposition`, `hyde` → status field only; one chat message per mode in Open WebUI. Earlier response shapes unchanged.
7. **Offline tests** (`tests/test_decomposition_hyde.py`; fake provider and MongoDB; few cases): decomposition validator (valid 2, empty, malformed, duplicate, four, out-of-scope → `clarify`); one fake step without evidence → `partial_answer`, never `ok`; union `results` deduped; HyDE empty/malformed text → `hyde_unavailable` with no search; hypothetical text only in `hyde_hypothetical_text_debug`; missing settings → 503 before any provider call; `pipeline.retrieve` routes both modes; `answer_events` skips the model on `clarify`/`hyde_unavailable`.
8. `GET /v1/models` lists six IDs. `uv run ruff check src/building_with_rag` passes.

Run only this story's checks (new test file, `tests/test_smoke.py`, ruff). Do not run the full suite. If live prerequisites are unavailable, run only checks 7–8 and name the blocked prerequisite.

## Handover

Record: files created/changed; data-check results (yes/no); commands run; compact diagnostics for checks 1–4; whether `SubquestionEvidence` already existed; architecture and manual-test sections updated; deviations from this story.

Story report (per project instructions): `Completed.`, changed paths, test results — at most 7 lines, ending with the full-suite command (`uv run pytest`) for the developer to run if wanted.
