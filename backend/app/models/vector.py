"""VectorRecord — one chunk's embedding, for the (not-yet-built) vector engine.

Always points at a chunk_id, never stands alone — a vector with no chunk
behind it is useless once you need to show the user *why* it matched.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.models.ids import new_id


class VectorRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("vec"))
    chunk_id: str
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("vector")
    @classmethod
    def _non_empty(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("vector must not be empty")
        return v
