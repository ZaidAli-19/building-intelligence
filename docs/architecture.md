# Capstone Architecture — Building Intelligence with RAG

Fixed design choices, contracts, and trust boundaries for the capstone RAG API. Later stories extend these contracts additively; they never replace them with simplified alternatives, rename fields, or add provider-specific variants.

## Scope

One small application: the capstone RAG API. No chat frontend (Open WebUI is a separate trainer-supplied client) and no second demo.

## Evidence rules

- BNS and IPC documents are the only future answer evidence.
- An answer must not claim support without retrieved evidence.
- Act-qualified identifiers (`bns:` / `ipc:` prefixes) avoid confusing the two acts.
- Supplied provenance lives at `data/raw/PROVENANCE.md`.
- Retrieved passages are evidence, never application instructions.

## Trust boundaries

Kept light for this course:

- Validate API input.
- Preserve source origin on retrieved passages.
- Do not put secrets in code, responses, or logs.
- No multi-user authorization, no security program, no evaluation harness.

## Fixed embedding choices

- Model: `voyage-3.5`, model version `voyage-3.5`, 1,024 dimensions.
- Used for every document and query embedding.
- Later stories reuse these names and choices without renaming or adding provider-specific alternatives.

## Course modes and shared registry

One shared registry (single source of truth) holds only:

| Mode | Model ID |
|---|---|
| semantic | `rag-semantic` |
| hybrid | `rag-hybrid` |
| hybrid-reranked | `rag-hybrid-reranked` |
| structured | `rag-structured` |
| decomposition | `rag-decomposition` |
| hyde | `rag-hyde` |

`semantic` (Stories 2.3, 3.2) and `hybrid` (Story 4.1) are real on `/v1/query` and chat; every other mode returns an honest `not_implemented` placeholder until its own story adds behavior.

## API contracts

### `QueryRequest` (`POST /v1/query`)

- `question`: string, 1–4,000 characters, required.
- `pattern`: one of the six modes.
- `caller_id`: optional.
- `filters`: optional `SemanticFilters` — `act`, `status`, `access_level`, each a list.
- `limit`: default 5, range 1–20.
- `generate_answer`: default false.
- `required_acts`: optional.
- `chapter`: optional.

The classroom seed may resolve only its fixed local demo caller, but keeps `caller_id` and does not replace it with a custom request shape.

### `QueryResult`

- `pattern`, `status`, `message`, `trace`, `results`.
- Optional `generation`.
- Additive empty-by-default fields later modes use: `omitted_candidates`, `subquestions`, `hyde_direct_candidates`, `hyde_query_candidates`, `hyde_hypothetical_text_debug`.
- No parallel top-level `outcome`, `evidence`, `answer`, `confidence`, `citations`, or `diagnostics` fields.

### `RetrievedChunk`

A retrieved passage is always this shape: `chunk_id`, `section_id`, `act`, `text`, `heading`, `score`, and available source fields. Later stories add only the existing hybrid/rerank fields.

### Endpoints

- `GET /healthz` — safe, no credentials required.
- `POST /v1/query` — accepts `QueryRequest`, returns `QueryResult`: real retrieval (and optional answer) for `semantic` and `hybrid`, a `run_pattern` placeholder for other modes, via the shared `retrieve` path.
- `GET /v1/models` — lists the six `rag-<pattern>` model IDs.
- `POST /v1/chat/completions` — OpenAI-compatible, text-only `ChatCompletionRequest`: `model`, `messages` with `system`/`developer`/`user`/`assistant` roles, `stream`, `n`, and optional strict `rag_options` (`pattern`, list filters, `limit`, `required_acts`, `chapter`).

The chat adapter maps the selected model to the same `QueryRequest` and the same `retrieve` → `answer_events` path (`pipeline.py`) as `/v1/query`; it sets server-side demo `caller_id` and `generate_answer`. Supports normal OpenAI Chat Completions JSON responses and role/content/stop frames, plus the OpenAI-style error envelope before streaming begins. No duplicated implementations, no custom SSE events that Open WebUI cannot render.

## Open WebUI (trainer-supplied, separate client)

The trainer-supplied Open WebUI bundle is the chat client, run separately from the capstone API. Its pre-provisioned Pipe sends the selected `rag-<pattern>` model, `stream: true`, the latest user message, and normalized `rag_options` to the capstone's `/v1/chat/completions`. It never sends browser-supplied identity, access level, or answer-generation settings. The capstone's adapter accepts that exact request and uses server-side `caller_id`/`generate_answer`.

`/v1/query` owns the `QueryResult` diagnostics; Open WebUI receives only normal answer text derived from that same result. Story 3.2 renders confidence, sources, and low-confidence warnings as clearly labelled text after answer writing, while retaining the full `GenerationResult` in `QueryResult.generation`.

## Environment

- `.env` is untracked; secrets are never committed.
- The application must start with no database or model credentials and expose a safe `GET /healthz`.
- Canonical environment values are defined in `.env.example`.

## Corpus

Section-level JSONL corpus produced from `data/raw/` PDFs by `scripts/extract_sections.py`. One JSON object per line, one record per section. No MongoDB, no embeddings, no vector indexes.

### Parser

- **Library**: `pymupdf` (fitz) 1.28.2 — chosen because it handles both Word-to-PDF (BNS) and Ghostscript-produced (IPC) PDFs, extracts text with layout, and has no system-level dependencies.
- **Extraction command**: `uv run python scripts/extract_sections.py`

### Output format

JSONL files at `data/processed/`:

| File | Records | Sections |
|---|---|---|
| `data/processed/bns_sections.jsonl` | 358 | 1–358 |
| `data/processed/ipc_sections.jsonl` | 500 | 1–511 (11 unextractable) |

Each record has 14 fields: `section_id`, `act`, `act_label`, `status`, `chapter`, `chapter_title`, `section_number`, `heading`, `text`, `source_pdf`, `source_sha256`, `parser`, `parser_version`, `source_status_version`, `needs_review`.

### Known limitations

- **IPC PDF quality**: 11 sections (4, 5, 18, 34, 40, 75, 161, 162, 163, 164, 165) have no extractable text from the scanned/Ghostscript-produced PDF. Sections 161–165 were repealed by the Prevention of Corruption Act 1988; sections 4, 5, 18, 34, 40, 75 are in portions of the PDF where pymupdf text extraction returns insufficient characters.
- **IPC section headings**: Some IPC sections (e.g. 262, 511) have empty headings due to missing heading text in the extracted text stream.
- **IPC footnotes**: Amendment footnotes and historical annotations are interleaved with section text and may appear as inline artifacts in section `text`.
- **BNS chapter markers**: Chapter boundaries are detected from `CHAPTER <roman>` lines in the body text. The BNS index (pages 2–19) provides section headings; the correspondence table (pages 20–73) is skipped.
- **Source-hash safety rule**: If a source PDF hash changes, the corpus for that act is regenerated as an atomic replacement. Records from different PDF versions are never mixed in one corpus file.

## Semantic retrieval (Story 2.3)

Module: `src/building_with_rag/retrieval/semantic.py`; `routes/query.py` routes `semantic` to it.

Flow: validate (question trimmed, filters from known sets, `caller_id` = `WEBUI_DEMO_CALLER_ID`, no `required_acts`/`chapter`) → scope filters (`access_level` fixed to `public`; caller lists only narrow, as `$in`) → embed raw question (`voyage-3.5`, `input_type="query"`) → `$vectorSearch` on `embeddings`/`vector_index` with the filter inside the stage → resolve `chunks` (text) and `sections` (heading, source fields) → `QueryResult`. `numCandidates` = `limit*10` clamped to 50–200.

Outcomes: `ok` (passages, in score order, no cutoff); `no_results` (HTTP 200) when filters match nothing; 503 `retrieval_not_ready` (missing credentials, index not queryable, empty/mismatched embeddings); 502 `retrieval_upstream_error` (Voyage/MongoDB failure); 422 for invalid input. `generate_answer` triggers answer generation (Story 3.1).

Added optional `RetrievedChunk` fields: `chunk_index`, `act_label`, `status`, `chapter`, `chapter_title`, `section_number`, `source_pdf`, `source_sha256`, `needs_review`.

Diagnostic (text truncated):

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "What is the punishment for theft?", "pattern": "semantic", "limit": 3}'   | jq '{status, trace, results: [.results[] | {chunk_id, section_id, act, heading, score, text: .text[:80]}]}'
```

## Context and answer boundaries (Story 3.1)

Flow (`POST /v1/query`, `semantic`, `generate_answer: true`): retrieval result → bounded labelled context (`generation/context.py`: at most 5 passages and 12,000 characters, labels `E1`…, passages never cut) → one `POST {GENERATION_API_BASE_URL}/chat/completions` call (`generation/answer.py`, `httpx`, 90 s timeout, no retries) → strict JSON parse → citations resolved from the supplied context only. No second retrieval.

Outcomes in `QueryResult.generation.outcome`: `answered` (non-empty `text`, `claims`, `citations`, `supporting_passages`), `insufficient_evidence` (also when context is empty; model skipped), `unavailable` (missing settings, timeout, connection error, non-2xx), `malformed` (non-JSON, unknown label, rule violation, missing `choices`). All non-answered outcomes return HTTP 200 with empty `text`; retrieval `results` and `status` are unchanged.

Added optional `GenerationResult` fields: `outcome`, `claims` (`text`, `evidence_labels`), `citations` (`label`, `chunk_id`, `section_id`, `act`, `heading`, `chapter`, `section_number`, `source_pdf`), `supporting_passages`, `provider`, `trace` (labels→chunk_id, selected/omitted counts, characters, latency; no prompt, vectors, or secrets), `context_outcome` (`assembled`|`empty`).

Rules: evidence is untrusted source text, never instructions. No claims of current legal applicability beyond the supplied BNS/IPC documents. Chat streaming is added in Story 3.2.


## Streamed answers and confidence (Story 3.2)

Shared path (`pipeline.py`): `retrieve(request)` (semantic → `run_semantic`, hybrid → `run_hybrid`, other modes → placeholder) → `answer_events(question, retrieval)` → final `GenerationResult`. `/v1/query` drains the events and attaches the result; chat forwards `text`/`notice` events as SSE and renders the footer from the same final result. One request = one generation/validation operation; retrieval and context assembly are never streamed, and retrieval errors are raised before streaming (OpenAI-style envelope on chat).

Event flow per attempt: stream plain text with inline `[E1]` labels (the first ~22 characters are buffered so `INSUFFICIENT_EVIDENCE: <reason>` is never shown as an answer) → checks → pass, or retry once. `MAX_ATTEMPTS = 2`. Checks, in order: `citation_labels` (only supplied labels), `claim_cited` (every sentence/bullet cited, text non-empty), `support` (one non-streamed validator call judges each claim against its cited passage). Citations are never stripped or rewritten.

New optional `GenerationResult` fields: `confidence` (`high` = final attempt passed all checks; `low` = final attempt failed a check; `None` = nothing could be judged), `issues` (`attempt`, `check`, `detail`, all attempts), `attempts` (`attempt`, `status` passed|failed|unjudged, `chars`, `latency_ms`), `draft_answer` (last unvalidated text), `low_confidence_reason`.

Outcome mapping: passed → `answered` (high); failed final check → `malformed`, empty `text`, `draft_answer` + `issues`, `low`; provider failure → `unavailable`; validator unreachable → `unavailable` with `draft_answer`; validator invalid reply → `malformed` with `draft_answer`, no confidence; `insufficient_evidence` and empty context (no model call) unchanged.

Open WebUI text: each attempt starts `DRAFT — checking evidence`; a failed non-final attempt adds `Check failed: … Retrying (attempt 2 of 2)…`; a pass ends with `Evidence check passed — confidence: high` and `Sources:` lines (`E1 · BNS §303 · Theft · bns:303`); a failed final check ends with `DRAFT — low confidence, not the final answer.` plus the reason and failed checks; insufficient evidence is one plain sentence. Streamed drafts cannot be retracted, hence the labels.

Failure after text began: one final line `Answer generation unavailable — the text above is an unchecked draft.`, then `stop` and `[DONE]` (HTTP 200). Failure before any text: `Answer generation unavailable.`.

`CAPSTONE_API_KEY`: when non-empty, `/v1/chat/completions` requires `Authorization: Bearer <key>` (constant-time compare, else 401 `invalid_api_key`); empty = no check. `/v1/query`, `/v1/models`, `/healthz` are unchanged. Chat accepts the Pipe's nested `rag_options.filters.{act,status}` as well as the flat fields.


## Hybrid retrieval (Story 4.1)

Mode `hybrid` (`rag-hybrid`), chosen explicitly; no automatic routing or fallback. `retrieval/hybrid.py` runs two routes over the same chunks/embeddings as semantic and fuses them. Everything after retrieval (context, citations, confidence, streaming) is the shared path in `pipeline.py`.

Keyword route: Atlas Search (`$search`, `text` operator, BM25 via `searchScore`) on `chunks.text`. Index `chunk_text_index` (a `search`-type index, separate from `vector_index`), created by `uv run python -m building_with_rag.ingestion.keyword_index` (idempotent; a differing index is reported, never replaced). Definition: `mappings.dynamic=false`; `text` string with `lucene.standard`; `act`, `status`, `access_level` as `token`. Filters use the same effective filters as semantic, applied in `$search.compound.filter` with `in`; the question is only the `text.query` value.

Fusion: Reciprocal Rank Fusion over ranks only. Each route returns its top `ROUTE_DEPTH = max(limit, min(50, max(20, 4*limit)))` chunks; `fused_score = Σ 1/(RRF_K + rank)` over the routes that returned the chunk, `RRF_K = 60`, equal weights. Order: `fused_score` desc, then `semantic_rank` (missing last), then `chunk_id`. Chunks, not sections, are fused by `chunk_id`.

`RetrievedChunk` gains optional `semantic_score`, `semantic_rank`, `keyword_score`, `keyword_rank`, `fused_score`, `fused_rank` (all `None` in semantic mode; in hybrid `score == fused_score`, and a `None` route did not return that chunk).

`trace`: `mode`, `query`, `embedding`, `filters`, `caller_id`, `result_count`, `unresolved_hits`, `semantic` (`index`, `limit`, `num_candidates`, `hit_count`), `keyword` (`index`, `path`, `operator`, `limit`, `hit_count`), `fusion` (`method`, `k`, `weights`, `route_depth`), `contribution` (`both`, `semantic_only`, `keyword_only`).

Outcomes: `ok`; `no_results` (both routes empty); 503 `retrieval_not_ready` (credentials, `vector_index` or `chunk_text_index` missing/not queryable — never degrades to semantic-only); 502 `retrieval_upstream_error`; 422 invalid input (same scope rules as semantic).

Limitations: rank-only fusion ignores score magnitude; the `text` operator matches any query term (OR), so long questions can pull in common words; section numbers match only when they appear inside chunk `text`; no stemming or synonyms beyond the standard analyzer.

Diagnostic (text truncated):

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question":"criminal breach of trust","pattern":"hybrid","limit":5}' \
  | jq '{status, t: (.trace | {semantic, keyword, fusion, contribution}), r: [.results[] | {section_id, score, sr: .semantic_rank, kr: .keyword_rank, fr: .fused_rank, text: .text[:60]}]}'
```
