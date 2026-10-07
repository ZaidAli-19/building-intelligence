"""Offline tests for claim splitting and citation checks (no network)."""

from building_with_rag.generation.answer import _check_attempt, split_claims

SUPPLIED = {"E1", "E2"}


def test_claims_carry_labels_including_trailing_label_group() -> None:
    claims = split_claims("Theft is punished. [E1]\n- Robbery is separate [E2].")
    assert [c["labels"] for c in claims] == [["E1"], ["E2"]]
    assert claims[0]["text"] == "Theft is punished."


def test_unknown_label_flagged() -> None:
    text = "Theft is punished [E9]."
    issues = _check_attempt(text, split_claims(text), SUPPLIED)
    assert [i.check for i in issues] == ["citation_labels"]


def test_uncited_statement_flagged() -> None:
    text = "Theft is punished [E1]. It is also serious."
    issues = _check_attempt(text, split_claims(text), SUPPLIED)
    assert [i.check for i in issues] == ["claim_cited"]


def test_valid_text_passes() -> None:
    text = "Theft is punished [E1]. Robbery is separate [E1, E2]."
    assert _check_attempt(text, split_claims(text), SUPPLIED) == []
