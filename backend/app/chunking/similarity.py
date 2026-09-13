"""Cosine similarity.

Shared by semantic chunking V2 (this phase) and, later, the vector engine
(Phase 22) — written here once so Phase 22 reuses this exact function
rather than reimplementing the same formula.
"""

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """How similar two vectors' directions are, ignoring magnitude.

    What problem it solves: two paragraph embeddings pointing the same
    direction means "about the same topic" — that's the signal semantic
    chunking uses to decide whether two paragraphs belong in one chunk.

    Formula: (A . B) / (||A|| * ||B||)

    Range: 1 = identical direction, 0 = orthogonal/unrelated, -1 =
    opposite direction. Real embedding models mostly produce positive
    values for related text.

    Complexity: O(d), d = vector dimension — one pass over both vectors.

    Limitations: requires equal-length vectors (raises otherwise); a
    zero vector returns similarity 0.0 rather than raising a
    division-by-zero error, since "no direction" has no meaningful
    similarity to compare.
    """
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} != {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)
