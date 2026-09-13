"""Phase 10 tests — cosine_similarity, tested independently of chunking."""

import pytest

from app.chunking import cosine_similarity


def test_identical_vectors_have_similarity_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_opposite_vectors_have_similarity_negative_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_ignores_magnitude_only_direction_matters():
    # [2,0] and [10,0] point the same direction, just different lengths.
    assert cosine_similarity([2.0, 0.0], [10.0, 0.0]) == pytest.approx(1.0)


def test_zero_vector_returns_zero_not_an_error():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])
