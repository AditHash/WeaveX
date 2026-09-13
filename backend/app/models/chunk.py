"""Chunk — the unit everything downstream (extraction, embedding) operates on.

A chunk always knows exactly where it came from (document_id + character
offsets). That's what makes provenance possible later: a relationship
points at an Evidence row, which points at a chunk_id, which points back
here, which points at the source document.
"""

from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.models.ids import new_id


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: new_id("chunk"))
    document_id: str
    text: str
    chunk_index: int = Field(ge=0)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _end_after_start(self) -> "Chunk":
        if self.end_offset < self.start_offset:
            raise ValueError(
                f"end_offset ({self.end_offset}) must be >= "
                f"start_offset ({self.start_offset})"
            )
        return self
