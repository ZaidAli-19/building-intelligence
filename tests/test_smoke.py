"""Minimal smoke tests: seed behavior only (health, models, chat errors)."""

import pytest
from fastapi.testclient import TestClient

from building_with_rag.app import create_app
from building_with_rag.registry import PATTERN_MODEL_IDS


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_lists_eight(client: TestClient) -> None:
    response = client.get("/v1/models")
    assert response.status_code == 200
    ids = [m["id"] for m in response.json()["data"]]
    assert ids == list(PATTERN_MODEL_IDS.values())


def test_chat_unknown_model_error(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-bogus", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["error"]["type"] == "invalid_request_error"
