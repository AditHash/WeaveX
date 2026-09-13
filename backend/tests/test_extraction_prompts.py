"""Phase 12 tests — extraction prompt construction."""

from app.extraction import build_extraction_prompt
from app.models.schema import EntityType, RelationshipType

from tests.fixtures import FASTAPI_CORPUS_TEXT


def test_prompt_includes_chunk_text_verbatim():
    prompt = build_extraction_prompt(FASTAPI_CORPUS_TEXT)
    assert FASTAPI_CORPUS_TEXT in prompt


def test_prompt_lists_every_allowed_entity_type():
    prompt = build_extraction_prompt("some text")
    for entity_type in EntityType:
        assert entity_type.value in prompt


def test_prompt_lists_every_allowed_relationship_type():
    prompt = build_extraction_prompt("some text")
    for relationship_type in RelationshipType:
        assert relationship_type.value in prompt


def test_prompt_instructs_json_only_output():
    prompt = build_extraction_prompt("some text")
    assert "JSON" in prompt
