"""Parser tests for grounded answers (no network)."""

import json

from building_with_rag.generation.answer import parse_output

LABELS = {"E1", "E2"}


def _payload(**over):
    base = {
        "outcome": "answered",
        "answer": "Theft is punished.",
        "claims": [{"text": "Theft is punished.", "evidence": ["E1"]}],
        "reason": "",
    }
    return json.dumps({**base, **over})


def test_unknown_label_malformed() -> None:
    assert parse_output(_payload(claims=[{"text": "x", "evidence": ["E9"]}]), LABELS) is None


def test_non_json_malformed() -> None:
    assert parse_output("The answer is theft.", LABELS) is None


def test_answered_without_claims_malformed() -> None:
    assert parse_output(_payload(claims=[]), LABELS) is None


def test_valid_answered_with_fence() -> None:
    parsed = parse_output("```json\n" + _payload() + "\n```", LABELS)
    assert parsed["outcome"] == "answered"
    assert parsed["claims"][0]["evidence"] == ["E1"]
