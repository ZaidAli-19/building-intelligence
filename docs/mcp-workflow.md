# MCP integration — end-to-end workflow

How the Legal Sections MCP server, the RAG app (`rag-mcp` mode) and Open WebUI fit together, in the order they were built and wired. Design contracts: `docs/architecture.md` (section "MCP integration (Story 7.1)") and `../legal-sections-mcp/docs/architecture.md`.

## 1. Picture

```
Open WebUI (8080)  --OpenAI-compatible-->  RAG app (8000)  --MCP streamable HTTP-->  MCP server (8765/mcp)  --read-only-->  MongoDB
  model: rag-mcp       http://127.0.0.1:8000/v1   retrieval/mcp_client.py   http://127.0.0.1:8765/mcp   legal-sections-mcp   sections collection
```

The RAG app is the MCP **client**; the server is a separate process. The grounded answer, confidence, citations and streaming are the app's existing shared path. The MCP server's section text is the only evidence.

## 2. The three URLs

| Where | Setting | Value | Purpose |
|---|---|---|---|
| RAG app `.env` (untracked) | `MCP_SERVER_URL` | `http://127.0.0.1:8765/mcp` | where the app finds the MCP server; empty → 503 `retrieval_not_ready` |
| RAG app `.env` | `PUBLIC_BASE_URL` | `http://127.0.0.1:8000` | prefix of the `/pdf` download link (optional) |
| RAG app `.env` | `MCP_DEADLINE_SECONDS` | `10` (valid 1–60) | hard deadline for MCP calls |
| Open WebUI → Admin Panel → Settings → Connections → OpenAI API | connection URL | `http://127.0.0.1:8000/v1` | where Open WebUI finds the RAG app; key = `CAPSTONE_API_KEY` if set |
| MCP server `.env` | `MONGODB_URI`, `MONGODB_DB_NAME` | same database as the RAG app | read-only access to `sections` |
| MCP server `.env` | `MCP_HOST` / `MCP_PORT` | `127.0.0.1` / `8765` | where the server listens (endpoint `/mcp`) |

`.env.example` in the RAG repo lists the two new `MCP_*` names, but the app reads only `.env`; copy them there. Never commit `.env` or print keys.

## 3. Build order

### A. MCP server (`../legal-sections-mcp/`, separate repo, own stories 1.1–4.1)

1. Project seed: `uv`, `pyproject.toml`, `mcp` SDK (`FastMCP`), `pymongo`, `pydantic`; settings and a lazy Mongo client.
2. Read-only repository over `sections` only; every query forced to `access_level = "public"`; `_id` and `provenance` never returned; heading search uses `re.escape`.
3. Five tools with typed inputs and a `status` envelope (`ok`, `not_found`, `validation_error`, `upstream_error`):
   `get_section`, `list_sections`, `list_chapters`, `search_sections_by_heading`, `get_sections_in_range` (span ≤ 25).
   `compare_ipc_bns` is omitted: the corpus has no IPC↔BNS mapping.
4. Transports: streamable HTTP at `/mcp` (default) and `--stdio`; manual tests and README.
5. The 11 unextractable IPC sections (4, 5, 18, 34, 40, 75, 161–165) are flagged (`needs_review`, `text_available: false`), not hidden.

### B. RAG app (this repo, Story 7.1: `docs/stories/story-7-1-mcp-integration.md`)

1. Story written first (client only; the server was already its own project).
2. Dependency: `mcp` SDK (`uv add "mcp>=1.2,<2"`).
3. Registry: `Pattern.MCP` and model ID `rag-mcp` (`registry.py`); `GET /v1/models` now lists eight.
4. Settings: `mcp_server_url`, `mcp_deadline_seconds` in `settings.py`; names added to `.env.example`.
5. Controller `src/building_with_rag/retrieval/mcp_client.py`: deterministic, no LLM, fixed allowlist of all five server tools; at most 2 calls; per-call deadline; every result inspected (wrong shape or foreign `section_id` → 502 `mcp_bad_response`); errors 503/502, never a fallback to another mode. Rules, in order:
   1. validate (bare section number, several sections, both acts → `clarify`, no call);
   2. range such as "BNS sections 10 to 19" → one `get_sections_in_range` (reversed, beyond 1–511 or wider than 25 → `clarify`; gaps named, `partial_result`);
   3. one act + `chapters` → `list_chapters`; one act + "list/show ... sections" (optional "first N") → `list_sections` (status `listing`, shown directly, no generated answer);
   4. one act + one section → one `get_section`;
   5. one act + one quoted heading → `search_sections_by_heading`, then `get_section` only for a single match;
   6. anything else → `clarify` (`unsupported_question`).
6. Pipeline: `retrieve` dispatches `MCP`; `REAL_PATTERNS` and the answerable statuses (`ready_for_synthesis`, `partial_result`) include it.
7. Range fix: the shared answer step capped context at 5 passages / 12,000 characters, so the model reported "insufficient evidence" for 10-section ranges. `build_context` and `stream_answer` gained optional limits (defaults unchanged); `pipeline.py` passes 25 passages / 60,000 characters only for `mcp`.
8. Tests `tests/test_mcp.py` (fake transport, 9 cases), `tests/test_smoke.py` updated to eight models.
9. Docs: `docs/architecture.md` and `docs/manual-tests.md` updated.

### C. Open WebUI

1. Start it: `sh open_webui_functions/manage_open_webui.sh start`; open `http://127.0.0.1:8080` (not `localhost`).
2. The existing "RAG options" chip does **not** offer `mcp` (its pattern list is in `open_webui_functions/capstone_retrieval_controls.py`, deliberately unchanged), so `rag-mcp` is reached by a direct connection instead:
   Admin Panel → Settings → Connections → OpenAI API → **+** → URL `http://127.0.0.1:8000/v1` → key (`CAPSTONE_API_KEY` or any value) → save.
3. New chat → model picker → `rag-mcp`. If missing: gear icon on the connection → Verify; confirm `GET http://127.0.0.1:8000/v1/models` lists `rag-mcp`.

## 4. Run order (every session)

1. MCP server: `cd ../legal-sections-mcp && uv run python -m legal_sections_mcp` (listens on 8765).
2. RAG app: `uv run uvicorn building_with_rag.app:app --port 8000` (restart after any code or `.env` change).
3. Open WebUI: `sh open_webui_functions/manage_open_webui.sh start`.

## 5. What to ask `rag-mcp`

| Question | Expected |
|---|---|
| What does BNS section 103 say? | answered, high confidence, `bns:103` |
| What do BNS sections 303 to 305 say? | answered, 3 sources |
| Show IPC sections 1 to 10 | sections 4 and 5 named as missing |
| What does IPC section 4 say? | "no such record" |
| Show the BNS section titled "Murder" | lists matching ids; name one |
| section 103 / BNS sections 19 to 10 / BNS sections 1 to 100 | `clarify`, no tool call |
| What chapters does the BNS have? | `listing`: chapter lines (`list_chapters`) |
| List the first 10 IPC sections | `listing`: headings only (`list_sections`) |
| What do BNS sections 303 to 305 say? /pdf | `pdf_ready` with a download link; PDF only when `/pdf` is typed |
| What is the punishment for theft? | unsupported (use semantic or hybrid) |
| any question with the MCP server stopped | 502 `mcp_upstream_error` |

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "MCP_SERVER_URL is not set" (503) | add it to the app's `.env`, restart the app |
| 502 `mcp_upstream_error` | MCP server not running or wrong port; check `.env` URL |
| `rag-mcp` not in Open WebUI | connection missing/disabled, or app not running; verify `/v1/models` |
| Range answered "insufficient evidence", sees only 5 sections | app still on old code; restart it |
| Large ranges (~10 sections) give a low-confidence draft | shared evidence check rejects an uncited intro sentence; ask for 2–5 sections |
| Port 8000/8765 already in use | an earlier instance is still running; stop it, then restart |

## 7. Checks (this story only)

`uv run pytest tests/test_mcp.py tests/test_smoke.py` and `uv run ruff check src/building_with_rag`. Full suite, if wanted: `uv run pytest`.
