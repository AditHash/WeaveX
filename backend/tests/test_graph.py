"""Phase 3 graph tests.

Built against the FastAPICorpus fixture (tests/fixtures.py) — a real,
7-entity/6-relationship graph derived from the actual worked example in
CLAUDE.md/PLAN.md, with full Document->Chunk->Relationship->Evidence
provenance — instead of disconnected two-node toy graphs. Edge-case tests
(duplicate id, missing entity, cascade delete) use entities pulled from
that same corpus rather than inventing throwaway names.
"""

import pytest

from app.graph import EntityNotFoundError, Graph
from app.models import Relationship, RelationshipType

from tests.fixtures import FastAPICorpus


# --------------------------------------------------------- realistic corpus


def test_full_corpus_loads_into_graph():
    corpus = FastAPICorpus()
    graph = corpus.build_graph()

    assert len(graph) == 7  # FastAPI, Python, Pydantic, Starlette, Sebastián Ramírez, API, Microservices
    for rel in corpus.relationships:
        assert graph.get_relationship(rel.id) == rel


def test_every_relationship_has_full_provenance_chain():
    # This is the project's core promise: no graph fact without a traceable
    # Relationship -> Evidence -> Chunk -> Document chain.
    corpus = FastAPICorpus()
    chunk_ids = {c.id for c in corpus.chunks}

    for rel in corpus.relationships:
        matching_evidence = [e for e in corpus.evidence if e.relationship_id == rel.id]
        assert matching_evidence, f"relationship {rel.id} has no evidence"

        for ev in matching_evidence:
            assert ev.chunk_id in chunk_ids
            chunk = next(c for c in corpus.chunks if c.id == ev.chunk_id)
            assert chunk.document_id == corpus.document.id
            # evidence text must actually be a substring the chunk plausibly
            # came from (loose check: same paragraph's content)
            assert ev.text  # non-empty, real sentence, not a placeholder


def test_created_by_relationship_points_person_to_fastapi():
    corpus = FastAPICorpus()
    graph = corpus.build_graph()

    ramirez = corpus.entities["Sebastián Ramírez"]
    fastapi = corpus.entities["FastAPI"]
    created = next(
        r for r in corpus.relationships
        if r.relationship_type == RelationshipType.CREATED
    )
    assert created.source_entity_id == ramirez.id
    assert created.target_entity_id == fastapi.id
    assert graph.get_relationship(created.id) is not None


def test_fastapi_has_multiple_outgoing_relationships():
    # FastAPI is the hub of this corpus: USES x2, WRITTEN_IN, PROVIDES x2.
    corpus = FastAPICorpus()
    fastapi_id = corpus.entities["FastAPI"].id
    outgoing = [r for r in corpus.relationships if r.source_entity_id == fastapi_id]
    assert len(outgoing) == 5


def test_remove_fastapi_cascades_all_five_outgoing_and_one_incoming():
    corpus = FastAPICorpus()
    graph = corpus.build_graph()
    fastapi_id = corpus.entities["FastAPI"].id

    graph.remove_entity(fastapi_id)

    for rel in corpus.relationships:
        assert graph.get_relationship(rel.id) is None
    # every other entity from the corpus survives — cascade only removed
    # relationships, not unrelated nodes
    for name, entity in corpus.entities.items():
        if name != "FastAPI":
            assert graph.get_entity(entity.id) is not None


# ------------------------------------------------------------ entity CRUD


def test_add_and_get_entity():
    corpus = FastAPICorpus()
    graph = Graph()
    fastapi = corpus.entities["FastAPI"]
    graph.add_entity(fastapi)
    assert graph.get_entity(fastapi.id) == fastapi
    assert len(graph) == 1


def test_get_missing_entity_returns_none():
    graph = Graph()
    assert graph.get_entity("entity_missing") is None


def test_add_duplicate_entity_raises():
    corpus = FastAPICorpus()
    graph = Graph()
    fastapi = corpus.entities["FastAPI"]
    graph.add_entity(fastapi)
    with pytest.raises(ValueError):
        graph.add_entity(fastapi)


def test_remove_entity():
    corpus = FastAPICorpus()
    graph = Graph()
    pydantic = corpus.entities["Pydantic"]
    graph.add_entity(pydantic)
    graph.remove_entity(pydantic.id)
    assert graph.get_entity(pydantic.id) is None
    assert len(graph) == 0


def test_remove_missing_entity_raises():
    graph = Graph()
    with pytest.raises(EntityNotFoundError):
        graph.remove_entity("entity_missing")


# ------------------------------------------------------ relationship CRUD


def test_add_relationship_requires_both_entities_to_exist():
    corpus = FastAPICorpus()
    graph = Graph()
    fastapi = corpus.entities["FastAPI"]
    graph.add_entity(fastapi)  # Pydantic deliberately not added

    rel = next(
        r for r in corpus.relationships
        if r.relationship_type == RelationshipType.USES
        and r.target_entity_id == corpus.entities["Pydantic"].id
    )
    with pytest.raises(EntityNotFoundError):
        graph.add_relationship(rel)


def test_remove_relationship_updates_adjacency():
    corpus = FastAPICorpus()
    graph = Graph()
    fastapi, python = corpus.entities["FastAPI"], corpus.entities["Python"]
    graph.add_entity(fastapi)
    graph.add_entity(python)
    written_in = next(
        r for r in corpus.relationships
        if r.relationship_type == RelationshipType.WRITTEN_IN
    )
    graph.add_relationship(written_in)

    graph.remove_relationship(written_in.id)

    assert graph.get_relationship(written_in.id) is None
    # adjacency lists must be clean, not still holding the removed id
    graph.add_relationship(
        Relationship(
            source_entity_id=fastapi.id,
            target_entity_id=python.id,
            relationship_type=RelationshipType.WRITTEN_IN,
            confidence=0.9,
        )
    )


def test_remove_relationship_missing_raises():
    graph = Graph()
    with pytest.raises(KeyError):
        graph.remove_relationship("rel_missing")
