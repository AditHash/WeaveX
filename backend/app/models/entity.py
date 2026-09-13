"""Entity — one resolved real-world thing.

Before entity resolution (Phase 16) runs, extraction produces raw mentions
("FastAPI", "fastapi", "Fast API"). After resolution, those collapse into
ONE Entity with a canonical_name and the surface forms kept as aliases —
never discarded, since they're needed to match future mentions.
"""

from typing import Any

from pydantic import BaseModel, Field

from app.models.ids import new_id
from app.models.schema import EntityType


class Entity(BaseModel):
    id: str = Field(default_factory=lambda: new_id("entity"))
    canonical_name: str
    entity_type: EntityType
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
