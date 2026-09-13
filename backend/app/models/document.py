"""Document — the source text a user ingests.

Everything else in WeaveX (chunk, entity mention, evidence) ultimately
traces back to one of these. It is the root of the provenance chain.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.ids import new_id


class Document(BaseModel):
    id: str = Field(default_factory=lambda: new_id("doc"))
    title: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
