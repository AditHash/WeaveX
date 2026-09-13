"""Phase 2 model tests.

Covers: default ID generation/uniqueness, the validation rules each model
enforces, and round-trip (de)serialization — since every model will be
written to/read from SQLite as JSON later (Phase 6).
"""

import pytest
from pydantic import ValidationError

from app.models import (
    Chunk,
    Document,
    Entity,
    EntityType,
    Evidence,
    Relationship,
    RelationshipType,
    VectorRecord,
)


# ---------------------------------------------------------------- Document


def test_document_auto_id_and_defaults():
    doc = Document(title="FastAPI Intro", content="FastAPI is...")
    assert doc.id.startswith("doc_")
    assert doc.metadata == {}
    assert doc.created_at is not None


def test_document_ids_are_unique():
    a = Document(title="A", content="x")
    b = Document(title="B", content="y")
    assert a.id != b.id


def test_document_round_trip():
    doc = Document(title="A", content="x", metadata={"lang": "en"})
    restored = Document.model_validate(doc.model_dump())
    assert restored == doc


# ------------------------------------------------------------------ Chunk


def test_chunk_requires_valid_offsets():
    chunk = Chunk(
        document_id="doc_1",
        text="FastAPI uses Pydantic.",
        chunk_index=0,
        start_offset=0,
        end_offset=23,
    )
    assert chunk.id.startswith("chunk_")


def test_chunk_rejects_end_before_start():
    with pytest.raises(ValidationError):
        Chunk(
            document_id="doc_1",
            text="broken",
            chunk_index=0,
            start_offset=50,
            end_offset=10,
        )


def test_chunk_rejects_negative_index():
    with pytest.raises(ValidationError):
        Chunk(
            document_id="doc_1",
            text="x",
            chunk_index=-1,
            start_offset=0,
            end_offset=1,
        )


# ----------------------------------------------------------------- Entity


def test_entity_defaults():
    entity = Entity(canonical_name="FastAPI", entity_type=EntityType.TECHNOLOGY)
    assert entity.id.startswith("entity_")
    assert entity.aliases == []
    assert entity.description is None


def test_entity_rejects_unknown_type():
    with pytest.raises(ValidationError):
        Entity(canonical_name="FastAPI", entity_type="NotARealType")


def test_entity_keeps_aliases():
    entity = Entity(
        canonical_name="FastAPI",
        entity_type=EntityType.TECHNOLOGY,
        aliases=["Fast API", "fastapi"],
    )
    assert entity.aliases == ["Fast API", "fastapi"]


# ----------------------------------------------------------- Relationship


def test_relationship_defaults():
    rel = Relationship(
        source_entity_id="entity_1",
        target_entity_id="entity_2",
        relationship_type=RelationshipType.USES,
        confidence=0.94,
    )
    assert rel.id.startswith("rel_")


def test_relationship_rejects_unknown_type():
    with pytest.raises(ValidationError):
        Relationship(
            source_entity_id="entity_1",
            target_entity_id="entity_2",
            relationship_type="INVENTED",
            confidence=0.9,
        )


@pytest.mark.parametrize("bad_confidence", [-0.1, 1.1])
def test_relationship_rejects_confidence_out_of_range(bad_confidence):
    with pytest.raises(ValidationError):
        Relationship(
            source_entity_id="entity_1",
            target_entity_id="entity_2",
            relationship_type=RelationshipType.USES,
            confidence=bad_confidence,
        )


# --------------------------------------------------------------- Evidence


def test_evidence_links_relationship_to_chunk():
    ev = Evidence(
        relationship_id="rel_1",
        chunk_id="chunk_1",
        text="FastAPI uses Pydantic for data validation.",
        confidence=0.94,
    )
    assert ev.id.startswith("evidence_")


def test_multiple_evidence_can_share_one_relationship():
    # This is the whole point of splitting Evidence out of Relationship:
    # 3 chunks, same fact, 1 relationship, 3 evidence rows.
    ev1 = Evidence(relationship_id="rel_1", chunk_id="chunk_1", text="a", confidence=0.9)
    ev2 = Evidence(relationship_id="rel_1", chunk_id="chunk_8", text="b", confidence=0.8)
    assert ev1.relationship_id == ev2.relationship_id
    assert ev1.chunk_id != ev2.chunk_id


# ------------------------------------------------------------ VectorRecord


def test_vector_record_defaults():
    vec = VectorRecord(chunk_id="chunk_1", vector=[0.1, 0.2, 0.3])
    assert vec.id.startswith("vec_")


def test_vector_record_rejects_empty_vector():
    with pytest.raises(ValidationError):
        VectorRecord(chunk_id="chunk_1", vector=[])
