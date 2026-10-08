# Manual tests

## Story 1.1 — Architecture and Project Seed

What it adds: FastAPI project seed with health, query, model-listing, and OpenAI-compatible chat endpoints — all returning honest `not_implemented` placeholders.

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

Expected (Story 2.3): `"status":"ok"` with ranked passages in `results` (requires `.env` credentials and Story 2.2 data; see `docs/architecture.md` for the truncated `jq` form).

### Query — hybrid

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "criminal breach of trust", "pattern": "hybrid", "limit": 5}'
```

Expected (Story 4.1): `"pattern":"hybrid"`, `"status":"ok"`, fused `results` with `semantic_rank`/`keyword_rank`/`fused_rank` (requires the keyword index; see Story 4.1).

### Query — hybrid-reranked

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "hybrid-reranked", "limit": 5}'
```

Expected (Story 4.2): `"pattern":"hybrid-reranked"`, `"status":"ok"`, `results` ordered by `rerank_rank` with `omitted_candidates` for the rest (requires `RERANK_API_KEY`; see Story 4.2).

### Query — structured

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "structured"}'
```

Expected: `"status":"not_implemented"`, message references `structured`.

### Query — decomposition

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "decomposition"}'
```

Expected: `"status":"not_implemented"`, message references `decomposition`.

### Query — hyde

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "hyde"}'
```

Expected: `"status":"not_implemented"`, message references `hyde`.

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
  -d '{"model": "rag-structured", "messages": [{"role": "user", "content": "What is theft?"}]}'
```

Expected: `"object":"chat.completion"`, `"finish_reason":"stop"`, content contains `not implemented yet`.

### Chat completions — streaming

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-structured", "messages": [{"role": "user", "content": "What is theft?"}], "stream": true}'
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
  -d '{"model": "rag-structured", "messages": [{"role": "system", "content": "You are helpful."}]}'
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
## Story 2.2 — MongoDB, Chunks, Embeddings, and Vector Index

What it adds: an ingestion runner that loads the JSONL corpora into MongoDB as `sources`, `sections`, `chunks`, and `embeddings`, embeds every chunk with Voyage, and creates the Atlas `vector_index` on `embeddings.vector`.

Prerequisite: Story 2.1 complete (both JSONL files present), and `.env` holds `MONGODB_URI` (Atlas cluster), `MONGODB_DB_NAME`, `VOYAGE_API_KEY`. First run takes roughly 35–40 minutes on a free Voyage key.

### Run ingestion

```bash
uv run python -m building_with_rag.ingestion.ingest
```

Expected: steps 1–10 print in order. `sources=2`, `sections=858` with the supplied PDFs, `chunks` > 0, and `embeddings == chunks: True`. `bns:1 chunks` shows one or more with `all linked: True`, sample vector length 1024, index status `READY`, and the sample query for "punishment for theft" prints `chunk_id`, `section_id`, heading, and score (or `vector query pending — index not ready`).

### Re-run (skip)

```bash
uv run python -m building_with_rag.ingestion.ingest
```

Expected: sources and sections report skipped, chunks report all skipped with 0 inserted/replaced, `Embeddings: 0 inserted, <n> skipped, 0 deleted` (no Voyage calls), and the existing `vector_index` is reused rather than recreated.

### Missing MongoDB URI (failure)

```bash
MONGODB_URI= uv run python -m building_with_rag.ingestion.ingest
```

Expected: stops at step 1 with `MongoDB unavailable: MONGODB_URI is empty. Set it in .env and re-run.` Nothing is written and no Voyage call is made.

## Story 2.3 — Semantic Retrieval

What it adds: `POST /v1/query` with `pattern: "semantic"` embeds the question, runs a MongoDB vector search, and returns ranked source passages with a diagnostic `trace`.

Prerequisite: Story 2.2 data loaded, `.env` holds `MONGODB_URI` and `VOYAGE_API_KEY`, API started as in Story 1.1.

```bash
# Success
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for theft?", "pattern": "semantic", "limit": 3}'

# No match (IPC is repealed)
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for theft?", "pattern": "semantic", "filters": {"act": ["IPC_1860"], "status": ["in_force"]}}'

# Invalid filter (failure)
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for theft?", "pattern": "semantic", "filters": {"act": {"$ne": "x"}}}'
```

Expected: first returns `"pattern":"semantic"`, `"status":"ok"`, at most 3 `results` in descending `score` order with `chunk_id`, `section_id` (e.g. `bns:303`), `act`, `heading`, `text`, and a populated `trace`. Second returns HTTP 200, `"status":"no_results"`, empty `results`. Third returns 422 validation error. With an empty `VOYAGE_API_KEY` the query returns 503 `retrieval_not_ready`, not `no_results`.


## Story 3.1 — Grounded Answer Generation

What it adds: `POST /v1/query` with `pattern: "semantic"` and `generate_answer: true` also returns `generation`, a non-streaming answer built only from the retrieved passages, with resolved citations.

Prerequisite: Story 2.3 working, API started as in Story 1.1, `.env` holds `GENERATION_API_BASE_URL`, `GENERATION_API_KEY`, and `GENERATION_MODEL_NAME`. Restart the API after any `.env` change.

```bash
# Success
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for theft under the BNS?", "pattern": "semantic", "limit": 5, "generate_answer": true}'

# Unsupported question (edge case)
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the GST rate on restaurant services?", "pattern": "semantic", "limit": 5, "generate_answer": true}'
```

Expected: first returns HTTP 200, retrieval `"status":"ok"`, `generation.outcome` `"answered"` with non-empty `text`, `claims` with `evidence_labels`, and `citations` (e.g. `bns:303`, `BNS_2023`). Second returns `generation.outcome` `"insufficient_evidence"`, empty `text`, no claims or citations, with `results` still present. If the generation service is unreachable, slow, or rejects the key, `generation.outcome` is `"unavailable"` (HTTP 200, `results` still returned; see `generation.trace.error`). With `generate_answer` omitted, `generation` is null.


## Story 3.2 — Streamed Answers with Confidence

What it adds: `rag-semantic` on `/v1/chat/completions` now retrieves, streams a cited answer, validates it, and ends with a confidence footer, sharing one path with `/v1/query`.

Prerequisite: API started as in Story 1.1 and `.env` holds the `GENERATION_*` values; when `CAPSTONE_API_KEY` is set, add `-H "Authorization: Bearer <key>"` to the chat commands.

```bash
# Success: streamed, validated answer
curl -sN http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-semantic", "stream": true, "messages": [{"role": "user", "content": "What is the punishment for theft under the BNS?"}]}'

# Edge case: unsupported question
curl -sN http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-semantic", "stream": true, "messages": [{"role": "user", "content": "What is the GST rate on restaurant services?"}]}'

# Failure: unknown model
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'
```

Expected: first streams `DRAFT — checking evidence`, text with `[E1]`-style labels, then `Evidence check passed — confidence: high` and `Sources:` lines; a failed check shows `Check failed: … Retrying (attempt 2 of 2)…` or ends with `DRAFT — low confidence, not the final answer.`. Second ends with the insufficient-evidence sentence and no confidence. Both streams end with `finish_reason: "stop"` and `data: [DONE]`. Third returns 400 `model_not_found`. With `CAPSTONE_API_KEY` set, a missing or wrong bearer token returns 401 `invalid_api_key`. Other modes (e.g. `rag-structured`) on chat still return the `not_implemented` placeholder.


## Story 4.1 — Hybrid Search

What it adds: `pattern: "hybrid"` (`rag-hybrid`) fuses Atlas Search keyword hits on chunk text with the semantic vector hits by Reciprocal Rank Fusion.

Prerequisite: Story 2.3 working, API started as in Story 1.1; create the keyword index once with the first command (waits until `READY`).

```bash
uv run python -m building_with_rag.ingestion.keyword_index

# Success
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "criminal breach of trust", "pattern": "hybrid", "limit": 5}'

# Edge case: filters leave nothing (IPC is repealed)
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "criminal breach of trust", "pattern": "hybrid", "filters": {"act": ["IPC_1860"], "status": ["in_force"]}}'

# Chat with the hybrid model
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-hybrid", "stream": true, "messages": [{"role": "user", "content": "What is criminal breach of trust?"}]}'
```

Expected: the index command ends with `chunk_text_index: READY (queryable)` (a second run says `reused`). Success returns `"pattern":"hybrid"`, `"status":"ok"`, at most 5 results in non-increasing `score` where `score == fused_score`, each with `semantic_rank` and/or `keyword_rank` and `fused_rank`, and `trace` showing `semantic`, `keyword`, `fusion`, and `contribution`. Edge case returns HTTP 200 with `"status":"no_results"` and empty `results`. Chat streams the same DRAFT/confidence/Sources text as `rag-semantic` and ends with `data: [DONE]`. If the keyword index is missing, hybrid returns 503 `retrieval_not_ready` naming the index command. `structured`, `decomposition`, and `hyde` still return `not_implemented`.


## Story 4.2 — Hybrid Re-ranking

What it adds: `pattern: "hybrid-reranked"` (`rag-hybrid-reranked`) re-scores a bounded set of hybrid candidates with one re-ranking call, so the answer is built from the passages the re-ranker ranks highest.

Prerequisite: Story 4.1 working (including the keyword index), `.env` holds `RERANK_API_KEY`, API started as in Story 1.1.

```bash
# Success
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the difference between culpable homicide and murder?", "pattern": "hybrid-reranked", "limit": 5}'

# Chat with the re-ranked model
curl -sN http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-hybrid-reranked", "stream": true, "messages": [{"role": "user", "content": "What is the punishment for theft under the BNS?"}]}'

# Failure: with RERANK_API_KEY empty (restart the API after changing .env)
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "hybrid-reranked"}'
```

Expected: success returns `"status":"ok"`, at most `min(limit, RERANK_RETURN_LIMIT)` `results` with `rerank_rank` 1..n and `score == rerank_score`, each keeping `fused_rank`/`fused_score`; `omitted_candidates` hold the rest with `omitted_reason` (`not_sent_to_reranker` or `below_return_limit`); `trace.rerank` shows the counts and `latency_ms`. Chat streams DRAFT/confidence/`Sources:` text and ends with `data: [DONE]`. The failure returns HTTP 503 `retrieval_not_ready` ("RERANK_API_KEY is not set; hybrid-reranked requires it."), not `no_results`. A provider failure returns 502 `retrieval_upstream_error`. `structured`, `decomposition`, and `hyde` still return `not_implemented`.
