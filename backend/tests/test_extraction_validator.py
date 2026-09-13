"""Phase 12 tests — cross-field extraction validation."""

from app.extraction import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    validate_extraction_result,
)


def _result(entities, relationships):
    return ExtractionResult(entities=entities, relationships=relationships)


def test_valid_result_has_no_errors():
    result = _result(
        entities=[
            ExtractedEntity(name="FastAPI", entity_type="Technology"),
            ExtractedEntity(name="Pydantic", entity_type="Technology"),
        ],
        relationships=[
            ExtractedRelationship(
                source="FastAPI", target="Pydantic", relationship_type="USES",
                evidence="FastAPI uses Pydantic for data validation.", confidence=0.94,
            )
        ],
    )
    assert validate_extraction_result(result) == []


def test_relationship_with_unknown_source_is_an_error():
    result = _result(
        entities=[ExtractedEntity(name="Pydantic", entity_type="Technology")],
        relationships=[
            ExtractedRelationship(
                source="FastAPI", target="Pydantic", relationship_type="USES",
                evidence="FastAPI uses Pydantic.", confidence=0.9,
            )
        ],
    )
    errors = validate_extraction_result(result)
    assert len(errors) == 1
    assert "FastAPI" in errors[0]


def test_relationship_with_unknown_target_is_an_error():
    result = _result(
        entities=[ExtractedEntity(name="FastAPI", entity_type="Technology")],
        relationships=[
            ExtractedRelationship(
                source="FastAPI", target="Pydantic", relationship_type="USES",
                evidence="FastAPI uses Pydantic.", confidence=0.9,
            )
        ],
    )
    errors = validate_extraction_result(result)
    assert len(errors) == 1
    assert "Pydantic" in errors[0]


def test_empty_evidence_is_an_error():
    result = _result(
        entities=[
            ExtractedEntity(name="FastAPI", entity_type="Technology"),
            ExtractedEntity(name="Pydantic", entity_type="Technology"),
        ],
        relationships=[
            ExtractedRelationship(
                source="FastAPI", target="Pydantic", relationship_type="USES",
                evidence="   ", confidence=0.9,
            )
        ],
    )
    errors = validate_extraction_result(result)
    assert any("evidence" in e for e in errors)


def test_result_with_no_relationships_is_valid():
    result = _result(
        entities=[ExtractedEntity(name="FastAPI", entity_type="Technology")],
        relationships=[],
    )
    assert validate_extraction_result(result) == []
