"""Phase 12 tests — extraction Pydantic schemas."""

import pytest
from pydantic import ValidationError

from app.extraction import ExtractedEntity, ExtractedRelationship, ExtractionResult


def test_extracted_entity_rejects_unknown_type():
    with pytest.raises(ValidationError):
        ExtractedEntity(name="FastAPI", entity_type="NotARealType")


def test_extracted_relationship_rejects_unknown_type():
    with pytest.raises(ValidationError):
        ExtractedRelationship(
            source="FastAPI", target="Pydantic", relationship_type="INVENTED",
            evidence="FastAPI uses Pydantic.", confidence=0.9,
        )


@pytest.mark.parametrize("bad_confidence", [-0.1, 1.1])
def test_extracted_relationship_rejects_confidence_out_of_range(bad_confidence):
    with pytest.raises(ValidationError):
        ExtractedRelationship(
            source="FastAPI", target="Pydantic", relationship_type="USES",
            evidence="FastAPI uses Pydantic.", confidence=bad_confidence,
        )


def test_extraction_result_defaults_to_empty():
    result = ExtractionResult()
    assert result.entities == []
    assert result.relationships == []


def test_extraction_result_accepts_real_corpus_shaped_data():
    # Matches tests/fixtures.py's FastAPICorpus mapping exactly (same
    # entity/relationship types already established and tested there).
    result = ExtractionResult.model_validate(
        {
            "entities": [
                {"name": "FastAPI", "entity_type": "Technology"},
                {"name": "Pydantic", "entity_type": "Technology"},
            ],
            "relationships": [
                {
                    "source": "FastAPI",
                    "target": "Pydantic",
                    "relationship_type": "USES",
                    "evidence": "FastAPI uses Pydantic for data validation.",
                    "confidence": 0.94,
                }
            ],
        }
    )
    assert len(result.entities) == 2
    assert result.relationships[0].relationship_type == "USES"
