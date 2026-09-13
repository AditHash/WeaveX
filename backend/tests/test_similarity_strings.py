"""Phase 15 tests — string similarity (Levenshtein + Jaccard).

Numbers below were computed with a standalone script before writing
these assertions, using PLAN.md's own GPT-5/Apple-style examples.
"""

import pytest

from app.entity_resolution import (
    jaccard_token_similarity,
    levenshtein_distance,
    levenshtein_similarity,
)


# ------------------------------------------------------- levenshtein_distance


def test_distance_zero_for_identical_strings():
    assert levenshtein_distance("FastAPI", "FastAPI") == 0


def test_distance_counts_single_substitution():
    assert levenshtein_distance("Pydantic", "Pydnatic") == 2  # transposition = 2 edits


def test_distance_from_empty_string():
    assert levenshtein_distance("", "GPT5") == 4
    assert levenshtein_distance("GPT5", "") == 4


# ----------------------------------------------------- levenshtein_similarity


def test_similarity_one_for_identical_strings():
    assert levenshtein_similarity("FastAPI", "FastAPI") == 1.0


def test_similarity_high_for_near_typo():
    # "GPT-5" -> "GPT5": one character removed, similarity 0.8.
    assert levenshtein_similarity("GPT-5", "GPT5") == pytest.approx(0.8)


def test_similarity_low_for_unrelated_words():
    assert levenshtein_similarity("FastAPI", "Django") == pytest.approx(0.0)


def test_similarity_of_two_empty_strings_is_one():
    assert levenshtein_similarity("", "") == 1.0


# ------------------------------------------------------ jaccard_token_similarity


def test_jaccard_one_for_identical_token_sets():
    assert jaccard_token_similarity("GPT-5", "GPT-5") == 1.0


def test_jaccard_catches_containment_that_levenshtein_misses():
    # This is the whole reason two metrics exist: "GPT-5" vs
    # "OpenAI GPT-5" scores LOW on edit distance (0.4167, dominated by
    # the extra word) but the shared token "gpt-5" gives Jaccard 0.5 —
    # correctly recognizing the containment relationship.
    a, b = "GPT-5", "OpenAI GPT-5"
    assert levenshtein_similarity(a, b) == pytest.approx(0.4167, abs=1e-3)
    assert jaccard_token_similarity(a, b) == pytest.approx(0.5)


def test_jaccard_zero_for_no_shared_tokens():
    assert jaccard_token_similarity("FastAPI", "Django") == 0.0


def test_jaccard_one_for_two_empty_strings():
    assert jaccard_token_similarity("", "") == 1.0


def test_jaccard_zero_when_one_side_empty():
    assert jaccard_token_similarity("GPT-5", "") == 0.0
