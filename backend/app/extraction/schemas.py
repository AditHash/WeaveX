"""Pydantic schemas the LLM extraction output must conform to.

Reuses EntityType/RelationshipType from app.models.schema (Phase 2's
centralized vocabulary) instead of redefining it — the LLM is
constrained to exactly the same type list the rest of the system uses,
not a free-text schema of its own invention.
"""

from pydantic import BaseModel, Field

from app.models.schema import EntityType, RelationshipType


class ExtractedEntity(BaseModel):
    name: str
    entity_type: EntityType


class ExtractedRelationship(BaseModel):
    source: str
    target: str
    relationship_type: RelationshipType
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)
