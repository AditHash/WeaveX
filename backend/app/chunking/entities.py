"""Crude candidate-entity extraction — a cheap proxy for real NER.

Not real entity recognition (that's Phase 12's LLM-based extraction).
This is a regex heuristic: sequences of capitalized words. Used only as
an extra chunk-boundary signal for Phase 11 — "these two paragraphs
mention the same capitalized phrase" is useful even before any real
entity extraction exists in the system.
"""

import re

_CAPITALIZED_PHRASE = re.compile(r"\b[A-Z][^\W\d_]*(?:\s+[A-Z][^\W\d_]*)*\b")


def extract_candidate_entities(text: str) -> set[str]:
    """Extract capitalized-word-sequence candidates from text.

    What problem it solves: entity-aware chunking needs a "do these two
    paragraphs talk about the same thing" signal, but real entity
    extraction doesn't exist until Phase 12. This is the cheap stand-in.

    Complexity: O(n) in text length (one regex pass).

    Limitations (real, documented rather than hidden): the first word of
    every sentence is capitalized regardless of whether it's a real
    entity, so this can false-positive on two paragraphs that merely
    share a sentence-starter. It can't detect an entity that's never
    capitalized, or normalize case variants ("FastAPI" vs "fastapi"). The
    leading character must be an ASCII capital (A-Z) — a name accented
    from its very first letter (e.g. "Álvaro") won't match — but interior
    characters allow any Unicode letter, so accented names like
    "Sebastián Ramírez" are captured correctly (verified against this
    project's own real corpus, which caught an earlier ASCII-only regex
    silently dropping that exact name entirely). Good enough as a
    boundary-decision *signal*, not as ground truth.
    """
    return set(_CAPITALIZED_PHRASE.findall(text))
