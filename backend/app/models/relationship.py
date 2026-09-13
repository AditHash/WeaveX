"""Relationship — a typed, directed edge between two entities.

Deliberately holds NO evidence text and NO single source_chunk_id. That
lives on Evidence instead, so multiple chunks that assert the same fact
(e.g. three chunks all saying "FastAPI uses Pydantic") collapse into one
logical Relationship with multiple Evidence rows — this is what makes
relationship deduplication possible (Phase 18) instead of one edge per
mention.

confidence here is the relationship's aggregate confidence (e.g. derived
from its evidence), not any single extraction's confidence.
"""

from typing import Any

from pydantic import BaseModel, Field

from app.models.ids import new_id
from app.models.schema import RelationshipType


class Relationship(BaseModel):
    id: str = Field(default_factory=lambda: new_id("rel"))
    source_entity_id: str
    target_entity_id: str
    relationship_type: RelationshipType
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
