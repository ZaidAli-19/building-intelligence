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
  -d '{"question": "What is theft?", "pattern": "hybrid"}'
```

Expected: `"status":"not_implemented"`, message references `hybrid`.

### Query — hybrid-reranked

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "hybrid-reranked"}'
```

Expected: `"status":"not_implemented"`, message references `hybrid-reranked`.

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
  -d '{"model": "rag-semantic", "messages": [{"role": "user", "content": "What is theft?"}]}'
```

Expected: `"object":"chat.completion"`, `"finish_reason":"stop"`, content contains `not implemented yet`.

### Chat completions — streaming

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-semantic", "messages": [{"role": "user", "content": "What is theft?"}], "stream": true}'
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
