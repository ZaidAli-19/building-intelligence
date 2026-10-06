from fastapi.testclient import TestClient

from building_with_rag.main import app

client = TestClient(app)
MODEL_IDS = {
    "rag-semantic",
    "rag-hybrid",
    "rag-hybrid-reranked",
    "rag-structured",
    "rag-decomposition",
    "rag-hyde",
}


def test_healthz():
    assert client.get("/healthz").json() == {"status": "ok"}


def test_query_placeholder():
    r = client.post("/v1/query", json={"question": "What is murder?", "pattern": "semantic"})
    assert r.status_code == 200
    assert r.json()["status"] == "not_implemented"
    assert r.json()["results"] == []


def test_models():
    assert {m["id"] for m in client.get("/v1/models").json()["data"]} == MODEL_IDS


def test_chat_json_and_sse():
    body = {
        "model": "rag-semantic",
        "messages": [{"role": "user", "content": "hi"}],
        "rag_options": {"pattern": "semantic", "filters": {"act": ["BNS"]}, "limit": 5},
    }
    r = client.post("/v1/chat/completions", json=body)
    assert "not implemented" in r.json()["choices"][0]["message"]["content"]
    s = client.post("/v1/chat/completions", json={**body, "stream": True})
    assert s.text.rstrip().endswith("data: [DONE]")
    assert '"finish_reason":"stop"' in s.text


def test_chat_unknown_model_envelope():
    r = client.post(
        "/v1/chat/completions",
        json={"model": "nope", "messages": [{"role": "user", "content": "x"}]},
    )
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"
