"""Phase 17/18/19 tests — GraphBuilder (construction + dedup + evidence).

Uses real chunks from FastAPICorpus and extraction results shaped like
what extract_from_chunk would actually return for that chunk's content
(see test_extractor.py for the LLM-call layer this builds on top of).
The dedup test is CLAUDE.md's own worked example almost verbatim: the
same fact asserted from two different chunks must produce ONE
relationship with TWO evidence rows, not two relationships.
"""

from app.extraction import ExtractionResult
from app.graph import Graph, GraphBuilder

from tests.fixtures import FastAPICorpus


def test_new_entities_and_relationships_are_created():
    corpus = FastAPICorpus()
    chunk = corpus.chunks[1]  # "FastAPI uses Pydantic...Starlette..."
    extraction = ExtractionResult.model_validate(
        {
            "entities": [
                {"name": "FastAPI", "entity_type": "Technology"},
                {"name": "Pydantic", "entity_type": "Technology"},
            ],
            "relationships": [
                {
                    "source": "FastAPI", "target": "Pydantic", "relationship_type": "USES",
                    "evidence": "FastAPI uses Pydantic for data validation.",
                    "confidence": 0.94,
                }
            ],
        }
    )

    graph = Graph()
    builder = GraphBuilder(graph)
    result = builder.build_from_extraction(extraction, chunk)

    assert len(result.new_entities) == 2
    assert len(result.new_relationships) == 1
    assert len(result.new_evidence) == 1
    assert len(graph) == 2
    assert len(graph.list_relationships()) == 1

    rel = graph.list_relationships()[0]
    evidence = graph.get_evidence(rel.id)
    assert len(evidence) == 1
    assert evidence[0].chunk_id == chunk.id
    assert evidence[0].text == "FastAPI uses Pydantic for data validation."


def test_same_fact_from_two_chunks_dedupes_into_one_relationship():
    # CLAUDE.md's own example: FastAPI uses Pydantic asserted in chunk_001
    # and chunk_008 must become ONE relationship with evidence from BOTH
    # chunks, not two separate relationships.
    corpus = FastAPICorpus()
    chunk_a, chunk_b = corpus.chunks[0], corpus.chunks[1]

    def extraction_for(evidence_text: str) -> ExtractionResult:
        return ExtractionResult.model_validate(
            {
                "entities": [
                    {"name": "FastAPI", "entity_type": "Technology"},
                    {"name": "Pydantic", "entity_type": "Technology"},
                ],
                "relationships": [
                    {
                        "source": "FastAPI", "target": "Pydantic",
                        "relationship_type": "USES",
                        "evidence": evidence_text, "confidence": 0.9,
                    }
                ],
            }
        )

    graph = Graph()
    builder = GraphBuilder(graph)

    result_a = builder.build_from_extraction(extraction_for("FastAPI uses Pydantic."), chunk_a)
    result_b = builder.build_from_extraction(extraction_for("...relies on Pydantic..."), chunk_b)

    assert len(graph.list_relationships()) == 1  # not 2
    assert len(result_a.new_relationships) == 1
    assert len(result_b.new_relationships) == 0  # reused, not created
    assert len(result_b.reused_relationships) == 1

    rel = graph.list_relationships()[0]
    evidence = graph.get_evidence(rel.id)
    assert len(evidence) == 2
    assert {e.chunk_id for e in evidence} == {chunk_a.id, chunk_b.id}


def test_different_surface_form_becomes_an_alias_not_a_duplicate_entity():
    corpus = FastAPICorpus()
    chunk = corpus.chunks[0]

    extraction_1 = ExtractionResult.model_validate(
        {"entities": [{"name": "FastAPI", "entity_type": "Technology"}], "relationships": []}
    )
    extraction_2 = ExtractionResult.model_validate(
        {"entities": [{"name": "Fast API", "entity_type": "Technology"}], "relationships": []}
    )

    graph = Graph()
    builder = GraphBuilder(graph)
    builder.build_from_extraction(extraction_1, chunk)
    result_2 = builder.build_from_extraction(extraction_2, chunk)

    assert len(graph) == 1  # still one entity
    assert len(result_2.new_entities) == 0
    assert len(result_2.reused_entities) == 1
    entity = graph.list_entities()[0]
    assert entity.canonical_name == "FastAPI"
    assert "Fast API" in entity.aliases


def test_same_name_different_type_stays_separate_entities():
    corpus = FastAPICorpus()
    chunk = corpus.chunks[0]

    extraction = ExtractionResult.model_validate(
        {
            "entities": [
                {"name": "FastAPI", "entity_type": "Technology"},
                {"name": "FastAPI", "entity_type": "Person"},  # contrived, but proves the filter
            ],
            "relationships": [],
        }
    )

    graph = Graph()
    builder = GraphBuilder(graph)
    result = builder.build_from_extraction(extraction, chunk)

    assert len(result.new_entities) == 2
    assert len(graph) == 2


def test_relationship_with_unknown_endpoint_is_skipped_not_crashed():
    # Defensive case: validator.py should catch this upstream, but the
    # builder must not crash if it somehow receives a relationship whose
    # source/target wasn't in this extraction's own entity list.
    corpus = FastAPICorpus()
    chunk = corpus.chunks[0]

    extraction = ExtractionResult(entities=[], relationships=[])
    # Bypass Pydantic validation deliberately isn't possible here since
    # ExtractedRelationship requires source/target strings regardless —
    # simulate the "not in name_to_entity" case with a relationship whose
    # names simply weren't declared as entities in the same result.
    from app.extraction import ExtractedRelationship
    extraction.relationships.append(
        ExtractedRelationship(
            source="Ghost", target="Phantom", relationship_type="RELATED_TO",
            evidence="text", confidence=0.5,
        )
    )

    graph = Graph()
    builder = GraphBuilder(graph)
    result = builder.build_from_extraction(extraction, chunk)  # must not raise

    assert result.new_relationships == []
    assert len(graph) == 0
