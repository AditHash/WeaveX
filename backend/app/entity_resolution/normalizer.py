"""Entity name normalization.

Produces a comparable canonical key from a raw entity name string, so
"GPT-5", "GPT 5", "gpt5" can be recognized as likely the same entity in
later resolution stages (Phase 16 exact match, Phase 15 similarity).
This is NOT the display name (Entity.canonical_name keeps its original
casing/formatting) — it's purely an internal comparison key.
"""

import re
import unicodedata

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_entity_name(name: str) -> str:
    """Produce a normalized comparison key for an entity name.

    What problem it solves: entity resolution needs to compare mentions
    for likely-same-entity, but "FastAPI", "Fast API", "fastapi" are all
    different Python strings. This maps all of them to one key.

    Algorithm: unicode NFC -> lowercase -> strip everything that isn't a
    letter or digit (punctuation AND whitespace both removed, not just
    collapsed — "Fast API" and "FastAPI" must normalize identically).

    Complexity: O(n) in name length.

    Limitations: this is normalization for EXACT-match comparison only
    (resolution stage 1). It deliberately loses information (case,
    punctuation, spacing) that near-miss cases might still need — e.g.
    "Apple" vs "Apple Inc." normalize to different keys on purpose, since
    conflating them here would be exactly the kind of over-eager merge
    string similarity (Phase 15) and semantic similarity (later) exist
    to handle more carefully instead.
    """
    normalized = unicodedata.normalize("NFC", name).lower()
    return _NON_ALNUM.sub("", normalized)
