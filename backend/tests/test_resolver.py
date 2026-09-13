"""Phase 16 tests — entity resolution.

Uses real corpus entities (FastAPI, Sebastián Ramírez) and PLAN.md's own
GPT-5 example where relevant. Similarity numbers verified with a
standalone script before writing assertions.
"""

from app.entity_resolution import resolve_entity
from app.graph import Graph
from app.models import Entity, EntityType

from tests.fixtures import FastAPICorpus


def test_first_mention_of_new_entity():
    graph = Graph()
    result = resolve_entity(graph, "FastAPI", EntityType.TECHNOLOGY)
    assert result.is_new is True
    assert result.matched_via == "new"
    assert result.entity.canonical_name == "FastAPI"


def test_exact_normalized_match_ignores_case_and_spacing():
    corpus = FastAPICorpus()
    graph = corpus.build_graph()  # has FastAPI with aliases ["Fast API", "fastapi"]

    result = resolve_entity(graph, "fastapi", EntityType.TECHNOLOGY)
    assert result.is_new is False
    assert result.matched_via == "exact"
    assert result.entity.id == corpus.entities["FastAPI"].id


def test_alias_match():
    # NOTE: FastAPICorpus's own aliases ("Fast API", "fastapi") both
    # normalize identically to the canonical_name "FastAPI" — they always
    # hit the "exact" path first and can never exercise the alias-only
    # branch (an earlier version of this test wrongly assumed otherwise).
    # A real alias whose normalized form actually DIFFERS from the
    # canonical name is needed to test that branch at all.
    graph = Graph()
    entity = Entity(
        canonical_name="GPT-5",
        entity_type=EntityType.TECHNOLOGY,
        aliases=["OpenAI GPT-5"],  # normalizes to "openaigpt5" != "gpt5"
    )
    graph.add_entity(entity)

    result = resolve_entity(graph, "OpenAI GPT-5", EntityType.TECHNOLOGY)
    assert result.is_new is False
    assert result.matched_via == "alias"
    assert result.entity.id == entity.id


def test_type_mismatch_never_merges_regardless_of_name():
    corpus = FastAPICorpus()
    graph = corpus.build_graph()

    # Same normalized name as the real Technology entity, but a
    # different type — must NOT match, type is a hard filter.
    result = resolve_entity(graph, "FastAPI", EntityType.PERSON)
    assert result.is_new is True
    assert result.matched_via == "new"


def test_borderline_similarity_does_not_auto_merge():
    # NOTE: "GPT-5" vs "GPT5" was the original example here, but
    # normalize_entity_name strips punctuation, so those two normalize
    # IDENTICALLY ("gpt5") and always hit the "exact" path — they can
    # never reach similarity comparison at all (this exact fact is what
    # test_entity_normalizer.py's test_gpt5_variants_normalize_identically
    # already proves; this test's original premise contradicted it).
    # "Pydantic" vs "Pydanti" normalizes to two DIFFERENT strings, so it
    # actually reaches the similarity stage — sim = 0.875, inside the
    # [0.75, 0.95) "candidate" zone, deliberately NOT auto-merged per
    # CLAUDE.md's conservative-by-design resolution stages.
    graph = Graph()
    resolve_result = resolve_entity(graph, "Pydantic", EntityType.TECHNOLOGY)
    graph.add_entity(resolve_result.entity)

    result = resolve_entity(graph, "Pydanti", EntityType.TECHNOLOGY)
    assert result.is_new is True
    assert result.matched_via == "new"


def test_accented_name_variant_does_not_auto_merge_either():
    # Real name from this project's own corpus. Accent-stripped variant
    # scores 0.8824 — still below the 0.95 auto-merge bar, so this stays
    # conservative even for a real name, not just the synthetic GPT-5 case.
    corpus = FastAPICorpus()
    graph = corpus.build_graph()

    result = resolve_entity(graph, "Sebastian Ramirez", EntityType.PERSON)
    assert result.is_new is True
    assert result.matched_via == "new"


def test_high_similarity_auto_merges():
    # Mechanism test (not corpus-derived — needs a long enough string for
    # a single-character edit to clear the 0.95 bar; verified via script:
    # sim("MicroservicesArchitecture", "MicroservicesArchitectur") = 0.96).
    graph = Graph()
    first = resolve_entity(graph, "MicroservicesArchitecture", EntityType.CONCEPT)
    graph.add_entity(first.entity)

    result = resolve_entity(graph, "MicroservicesArchitectur", EntityType.CONCEPT)
    assert result.is_new is False
    assert result.matched_via == "similarity"
    assert result.entity.id == first.entity.id
