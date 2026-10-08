# Manual tests

## Story 1.1 — Architecture and Project Seed

What it adds: FastAPI project seed with health, query, model-listing, and OpenAI-compatible chat endpoints — originally all returning `not_implemented` placeholders (every mode is real since Story 5.2).

Prerequisite: start the API — `uv run uvicorn building_with_rag.app:app --host 127.0.0.1 --port 8000`

### Health

```bash
curl -s http://127.0.0.1:8000/healthz
```

Expected: `{"status":"ok"}`.

### List models

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -c "import json,sys; [print(m['id']) for m in json.load(sys.stdin)['data']]"
```

Expected: six lines: `rag-semantic`, `rag-hybrid`, `rag-hybrid-reranked`, `rag-structured`, `rag-decomposition`, `rag-hyde`.

### Query — each RAG mode (semantic)

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "semantic"}'
```

Expected: `"status":"ok"` with ranked passages (see Story 2.3 below); needs Story 2.2 ingestion and `.env` credentials, otherwise HTTP 503.

### Query — hybrid

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "criminal breach of trust", "pattern": "hybrid", "limit": 3}' \
  | jq '{status, r: [.results[] | {section_id, score, sr: .semantic_rank, kr: .keyword_rank, fr: .fused_rank}]}'
```

Expected: `"status":"ok"` with fused results (real since Story 4.1; needs `chunk_text_index`, else 503).

### Query — hybrid-reranked

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "hybrid-reranked", "limit": 3}' \
  | jq '{status, t: .trace.rerank, r: [.results[] | {section_id, fr: .fused_rank, rr: .rerank_rank}]}'
```

Expected: `"status":"ok"` with re-ranked results (real since Story 4.2; needs `RERANK_API_KEY`, else 503 `retrieval_not_ready`).

### Query — decomposition / hyde

Real since Story 5.2; see the Story 5.2 section (needs `.env` with generation settings).

### Query — empty question (failure)

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "", "pattern": "semantic"}'
```

Expected: 422 validation error (question below min_length 1).

### Chat completions — JSON (non-streaming)

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-structured", "messages": [{"role": "user", "content": "What does section 103 say?"}]}'
```

Expected: `"object":"chat.completion"`, `"finish_reason":"stop"`, content is the structured clarification (act is missing); no credentials or network needed.

### Chat completions — streaming

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-structured", "messages": [{"role": "user", "content": "What does section 103 say?"}], "stream": true}'
```

Expected: SSE `data:` frames with `delta` role then content, ending with `data: [DONE]`.

### Chat completions — invalid model (failure)

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'
```

Expected: 400 with `"type":"invalid_request_error"`, `"code":"model_not_found"`.

### Chat completions — no user message (failure)

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-semantic", "messages": [{"role": "system", "content": "You are helpful."}]}'
```

Expected: 400 with `"code":"missing_user_message"`.

## Story 2.1 — Document Ingestion

What it adds: PDF extraction script that produces inspectable JSONL section-level corpora from the BNS and IPC bare act PDFs.

Prerequisite: Story 1.1 complete, `data/raw/` PDFs and `PROVENANCE.md` present, `uv sync` done.

### Run extraction

```bash
uv run python scripts/extract_sections.py
```

Expected: prints parser name/version, record counts (~358 BNS, ~500 IPC), count of `needs_review` flags, count of empty-text records. Both `data/processed/bns_sections.jsonl` and `data/processed/ipc_sections.jsonl` exist.

### Re-run (skip)

```bash
uv run python scripts/extract_sections.py
```

Expected: prints "BNS corpus up to date — skipping" and "IPC corpus up to date — skipping". No records appended or overwritten.

## Story 2.3 — Semantic Retrieval

What it adds: `pattern: "semantic"` on `POST /v1/query` embeds the question, runs a MongoDB vector search, and returns ranked source passages.

Prerequisite: Story 2.2 ingestion done, `MONGODB_URI` and `VOYAGE_API_KEY` set in `.env`, API started as in Story 1.1.

```bash
# Success
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for theft?", "pattern": "semantic", "limit": 3}' \
  | jq '{status, trace, results: [.results[] | {chunk_id, section_id, act, heading, score, text: .text[:80]}]}'
```

Expected: `status: "ok"`, `pattern: "semantic"`, 1–3 results in non-increasing `score` order, populated `trace` (mode, embedding, index, limit, num_candidates, filters, result_count).

```bash
# No results (IPC is repealed, so the filters match nothing)
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for theft?", "pattern": "semantic", "filters": {"act": ["IPC_1860"], "status": ["in_force"]}}' \
  | jq '{status, message, results: (.results | length)}'
```

Expected: HTTP 200, `status: "no_results"`, 0 results.

```bash
# Rejected filter (operator-shaped value)
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "semantic", "filters": {"act": {"$ne": "x"}}}'

# Rejected limit (max 20)
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "semantic", "limit": 21}'
```

Expected: `422` for each. With an empty `VOYAGE_API_KEY`, the success request returns HTTP 503 (`retrieval_not_ready`), not `no_results`.
## Story 3.1 — Grounded Answer Generation

What it adds: `generate_answer: true` on semantic `POST /v1/query` fills `generation` with a grounded, non-streaming answer and resolved citations, or an honest `insufficient_evidence`, `unavailable`, or `malformed` outcome.

Prerequisite: Story 2.3 prerequisites, plus `GENERATION_API_BASE_URL` and `GENERATION_API_KEY` set in `.env` (OpenAI-compatible LiteLLM proxy; `GENERATION_MODEL_NAME` defaults to `gpt-4o-mini`). Restart the API after changing `.env`.

```bash
# Answerable question
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for theft under the BNS?", "pattern": "semantic", "limit": 5, "generate_answer": true}' \
  | jq '{status, g: (.generation | {outcome, model, provider, context_outcome, text: (.text[:300]), claims: [.claims[] | {t: .text[:80], e: .evidence_labels}], citations: [.citations[] | {label, chunk_id, section_id, act, heading}], trace}), ctx: [.results[] | {chunk_id, section_id, act, score}]}'
```

Expected: `generation.outcome: "answered"`, non-empty `text`, each claim has evidence labels, every citation `chunk_id` appears in `ctx` and `section_id` prefix matches `act` (`bns:` for `BNS_2023`).

```bash
# Unsupported question (insufficient evidence)
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "What is the GST rate on restaurant services?", "pattern": "semantic", "limit": 5, "generate_answer": true}' \
  | jq '{status, g: (.generation | {outcome, text, claims: (.claims | length), citations: (.citations | length)}), results: (.results | length)}'
```

Expected: HTTP 200, `generation.outcome: "insufficient_evidence"`, empty `text`, 0 claims, 0 citations; retrieved `results` are still returned.

```bash
# Without generate_answer: no model call
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "semantic", "limit": 3}' \
  | jq '{status, generation, results: (.results | length)}'
```

Expected: `status: "ok"`, `generation: null`, 1–3 results.

Unavailable check: set `GENERATION_API_KEY=` empty in `.env`, restart the API, and rerun the first command. Expected: HTTP 200, `generation.outcome: "unavailable"`, empty `text`, `results` still present.

## Story 3.2 — Streamed Answers with Confidence

What it adds: semantic chat streams a DRAFT-prefixed answer, then a confidence/sources footer; `/v1/query` returns the same result with `confidence`, `issues`, `attempts`.

Prerequisite: Story 3.1 prerequisites. Optional `CAPSTONE_API_KEY` (chat then needs `Authorization: Bearer <key>`).

```bash
Q='What is the punishment for theft under the BNS?'
curl -sN http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"model\":\"rag-semantic\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}]}" \
  | grep '^data: {' | sed 's/^data: //' | jq -rj '.choices[0].delta.content // empty' | head -c 1500
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d "{\"question\":\"$Q\",\"pattern\":\"semantic\",\"generate_answer\":true}" \
  | jq '{status, g: (.generation | {outcome, confidence, attempts, issues, citations: [.citations[] | {label, section_id}]})}'
```

Expected: stream shows `DRAFT — checking evidence`, labelled answer, `confidence: high`, `Sources:`; `/v1/query` agrees (`answered`, `high`, same citations).

Unsupported: use "What is the GST rate on restaurant services?" — chat ends with the insufficient-evidence sentence (no confidence); `/v1/query` shows `insufficient_evidence` with `results` intact.

Stream shape: `curl -sN ... | grep -c '^data: '` — role chunk, content chunks, `finish_reason: "stop"`, `data: [DONE]` last. With `CAPSTONE_API_KEY` set, a wrong Bearer → 401 `invalid_api_key`.

Offline: `uv run pytest tests/test_streamed_confidence.py`

## Story 4.1 — Hybrid Search

What it adds: `rag-hybrid` / `pattern: "hybrid"` fuses Atlas Search keyword hits on `chunks.text` with semantic hits (RRF, k=60).

Prerequisite: Story 3.2 prerequisites; Atlas Search available. Voyage free tier is 3 requests/minute — space runs ~25 s apart.

```bash
uv run python -m building_with_rag.ingestion.keyword_index   # twice: created, then reusing
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question":"criminal breach of trust","pattern":"hybrid","limit":5}' \
  | jq '{status, t: (.trace | {semantic, keyword, fusion, contribution}), r: [.results[] | {section_id, score, sr: .semantic_rank, kr: .keyword_rank, fr: .fused_rank, text: .text[:60]}]}'
```

Expected: index ends `chunk_text_index: READY (queryable)`; `status: ok`, ≤ `limit` results, `score` non-increasing and equal to fused score, each result has `sr` and/or `kr`.

Compare modes: run one paraphrased and one exact-term question with `"pattern":"semantic"` and `"hybrid"`, `generate_answer: true`, `limit: 5`; note section order, contributing route(s), and `generation.outcome`.

Chat: `rag-hybrid` streams DRAFT, confidence and `Sources:` as `rag-semantic` does; `rag-decomposition` is real since Story 5.2.

Failure: with `chunk_text_index` missing, hybrid → 503 `retrieval_not_ready` (names the keyword-index command) while semantic stays `ok`.

Offline: `uv run pytest tests/test_hybrid_retrieval.py`

## Story 4.2 — Hybrid re-ranking

Prerequisite: Story 4.1 prerequisites plus `RERANK_API_KEY` in `.env`. Space Voyage runs ~25 s apart.

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question":"What is the difference between culpable homicide and murder?","pattern":"hybrid-reranked","limit":5}' \
  | jq '{status, t: .trace.rerank, r: [(.results[]|.+{kept:true}), (.omitted_candidates[]|.+{kept:false})] | map({section_id, kept, fr: .fused_rank, fs: .fused_score, rr: .rerank_rank, rs: .rerank_score, why: .omitted_reason, text: .text[:50]}) | sort_by(.fr)}'
```

Expected: `status: ok`; results ≤ `min(limit, RERANK_RETURN_LIMIT)` ordered by `rerank_rank`, `score == rerank_score`; omitted records carry `omitted_reason`; results + omitted = `trace.rerank.candidates`.

Compare: run `hybrid` and `hybrid-reranked` with `generate_answer: true`, `limit: 5`; note section order (`fr → rr`), omitted IDs, `generation.outcome`, `trace.rerank.latency_ms`.

Chat: `rag-hybrid-reranked` streams DRAFT, confidence and `Sources:`; cited chunks come from `results` only.

Missing key: with `RERANK_API_KEY=` empty and the API restarted, `hybrid-reranked` → 503 `retrieval_not_ready` (query) / OpenAI-style error (chat); `hybrid` stays `ok`.

Offline: `uv run pytest tests/test_rerank_retrieval.py tests/test_hybrid_retrieval.py`

## Story 5.1 — Structured exact retrieval

Postman: `POST http://127.0.0.1:8000/v1/query`, header `Content-Type: application/json`, Body → raw → JSON. Send each body below; check `status`, `trace.signals`, `trace.mongodb_called`, `trace.record`, `results`.

| # | Body | Expected |
|---|---|---|
| 1 | `{"question": "What does BNS section 103 say?", "pattern": "structured"}` | `status: ok`; `results[0].section_id: bns:103`; `mongodb_called: true`; `trace.record` has `status`, `source_status_version` |
| 2 | `{"question": "IPC section 302", "pattern": "structured"}` | `ok`; `ipc:302`; `mongodb_called: true` |
| 3 | `{"question": "What does BNS section 103 say?", "pattern": "structured", "generate_answer": true}` | as #1, plus `generation.outcome` reported; `generation.citations` only `bns:103` |
| 4 | `{"question": "What does section 103 say?", "pattern": "structured"}` | `clarification_needed`; `results: []`; `mongodb_called: false`; `trace.signals.reason` names missing act |
| 5 | `{"question": "IPC section 4", "pattern": "structured"}` | HTTP 200; `not_found`; `results: []`; `mongodb_called: true` |
| 6 | `{"question": "BNS section 999", "pattern": "structured"}` | same as #5 |
| 7 | `{"question": "What is the punishment for theft?", "pattern": "structured"}` | `recommendation`; `mongodb_called: false` |

Chat: `POST http://127.0.0.1:8000/v1/chat/completions`, same header, body `{"model": "rag-structured", "messages": [{"role": "user", "content": "What does BNS section 103 say?"}]}` (omit `stream` so Postman shows one JSON reply). `choices[0].message.content` shows DRAFT/confidence/`Sources:`. With content `What does section 103 say?` it is the clarification text only. If `CAPSTONE_API_KEY` is set, add `Authorization: Bearer <key>`.

Optional compact view: in the request's **Scripts → Post-response** tab (Tests in older Postman) paste, then open the Console (View → Show Postman Console) instead of reading the full body:

```js
const b = pm.response.json();
console.log(JSON.stringify({status: b.status, signals: b.trace.signals, mongodb_called: b.trace.mongodb_called,
  record: b.trace.record, results: b.results.map(r => ({section_id: r.section_id, text: r.text.slice(0, 60)}))}));
```

Offline: `uv run pytest tests/test_structured_retrieval.py tests/test_smoke.py`

## Story 5.2 — Query decomposition and HyDE

Prerequisite: `.env` with `MONGODB_URI`, `VOYAGE_API_KEY`, `GENERATION_API_BASE_URL`, `GENERATION_API_KEY`. Pause ~25 s between live calls.

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question":"How does the BNS punishment for murder differ from the IPC punishment for murder?","pattern":"decomposition","generate_answer":true}' \
  | jq '{status, reason: .trace.reason, t: (.trace | {decomposition, sufficiency}), s: [.subquestions[] | {subquestion, status, reason, ids: [.results[].section_id]}], g: (.generation | {outcome, confidence, text: .text[:200]})}'
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question":"What happens to someone who secretly takes a neighbour'"'"'s bike?","pattern":"hyde","generate_answer":true}' \
  | jq '{status, t: .trace.hyde, direct: [.hyde_direct_candidates[].section_id], hyde: [.hyde_query_candidates[].section_id], final: [.results[] | {section_id, score}], debug_chars: (.hyde_hypothetical_text_debug | length), g: (.generation | {outcome, confidence, text: .text[:200]})}'
```

Expected: decomposition `ok` with 2 `evidenced` steps from both acts; a half with no corpus evidence gives `partial_answer` naming it; "What is theft?" gives `clarify` or one subquestion. HyDE `ok`, both candidate lists populated, `debug_chars > 0`, hypothetical text absent from `message`, `trace`, `generation` and chat.

Missing config: with `GENERATION_API_KEY=` empty, both modes return 503 `retrieval_not_ready` naming it (chat: OpenAI-style envelope) while `semantic` stays `ok`.

Chat: `rag-decomposition` / `rag-hyde` stream DRAFT, confidence and `Sources:`; `partial_answer` prints the unsupported-part message first.

Offline: `uv run pytest tests/test_decomposition_hyde.py tests/test_smoke.py`

## Story 6.1 — Bounded agentic RAG

Prerequisite: `.env` with `MONGODB_URI`, `VOYAGE_API_KEY`, `GENERATION_API_BASE_URL`, `GENERATION_API_KEY`; optional `AGENTIC_DEADLINE_SECONDS` (default 20, valid 1-120; invalid -> 503 `retrieval_not_ready`). Pause ~25 s between live calls.

Postman: Import → Raw text → paste one curl (no shell functions or pipes). Each is a plain single-line curl; `\u0027` is a JSON-escaped apostrophe (keeps the single-quoted shell string valid).

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question":"What does BNS section 103 say?","pattern":"agentic","generate_answer":true}'
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question":"What does IPC section 4 say?","pattern":"agentic"}'
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question":"What is the punishment for theft?","pattern":"agentic","generate_answer":true}'
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question":"What happens to someone who secretly takes a neighbour\u0027s bike?","pattern":"agentic","generate_answer":true}'
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question":"section 103","pattern":"agentic"}'
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"rag-agentic","messages":[{"role":"user","content":"What is the punishment for theft?"}]}'
```

Expected, in order: `ready_for_synthesis`, `path: exact_lookup`, `call_count: 1`, `bns:103`; `partial_result`, `record_not_found`, empty `results`; `ready_for_synthesis`, `path: semantic`, `generation.outcome: answered`; `path: vocab_mismatch` (`hyde_calls: 1` only if the direct step was inadequate; `hyde_hypothetical_text_debug: null`); `clarify`, `bare_section_number`, `call_count: 0`, empty `steps`; chat content with DRAFT, `Evidence check passed — confidence: high` and `Sources:` (omit `stream` for one JSON reply; add `Authorization: Bearer <key>` if `CAPSTONE_API_KEY` is set).

Compact view: paste in **Scripts → Post-response** and read the Console (query requests only):

```js
const b = pm.response.json();
console.log(JSON.stringify({status: b.status, path: b.trace.path, stop: b.trace.stop_reason, calls: b.trace.call_count, hyde: b.trace.hyde_calls,
  steps: (b.trace.steps || []).map(s => [s.adapter, s.tool_status, s.adequate]), ids: b.results.map(r => r.section_id),
  dbg: b.hyde_hypothetical_text_debug, g: b.generation ? {o: b.generation.outcome, c: b.generation.confidence, n: b.generation.citations.length} : null}));
```

Chat: `rag-agentic` streams DRAFT text, `Evidence check passed — confidence: high` and `Sources:`; non-answer statuses print `message` only.

Offline: `uv run pytest tests/test_smoke.py`