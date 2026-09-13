"""Phase 14 tests — entity name normalization.

Uses the exact worked examples from PLAN.md/CLAUDE.md's own entity
resolution sections (GPT-5, FastAPI, Apple variants) rather than
fabricated names.
"""

from app.entity_resolution import normalize_entity_name


def test_fastapi_variants_normalize_identically():
    assert (
        normalize_entity_name("FastAPI")
        == normalize_entity_name("Fast API")
        == normalize_entity_name("fastapi")
    )


def test_gpt5_variants_normalize_identically():
    variants = ["GPT-5", "GPT 5", "GPT5", "gpt5"]
    normalized = {normalize_entity_name(v) for v in variants}
    assert len(normalized) == 1


def test_apple_and_apple_inc_do_not_normalize_identically():
    # Deliberate: PLAN.md is explicit these need similarity/semantic
    # matching (later stages), not exact normalization — conflating them
    # here would be an over-eager merge.
    assert normalize_entity_name("Apple") != normalize_entity_name("Apple Inc.")


def test_unicode_accented_names_normalize_consistently():
    import unicodedata

    composed = "Sebastián Ramírez"
    decomposed = unicodedata.normalize("NFD", composed)
    assert normalize_entity_name(composed) == normalize_entity_name(decomposed)


def test_normalization_is_idempotent():
    once = normalize_entity_name("FastAPI")
    twice = normalize_entity_name(once)
    assert once == twice
