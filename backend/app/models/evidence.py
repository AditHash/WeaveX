"""Evidence — the provenance link: Relationship -> Evidence -> Chunk -> Document.

One row = one piece of textual proof, from one chunk, for one relationship.
A relationship with no evidence should never be allowed into the graph
(see CLAUDE.md/PLAN.md "provenance is not optional") — that rule is
enforced by the graph builder later, not here; this model just carries
the data.
"""

from pydantic import BaseModel, Field

from app.models.ids import new_id


class Evidence(BaseModel):
    id: str = Field(default_factory=lambda: new_id("evidence"))
    relationship_id: str
    chunk_id: str
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
