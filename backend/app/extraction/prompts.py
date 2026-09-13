"""Prompt construction for LLM semantic extraction.

A pure function — given chunk text, build the instruction string. No
network call, no LLM client here (that's extractor.py's job) — kept
separate so the prompt itself is readable and testable without mocking
an LLM.
"""

from app.models.schema import EntityType, RelationshipType

_ENTITY_TYPES = ", ".join(t.value for t in EntityType)
_RELATIONSHIP_TYPES = ", ".join(t.value for t in RelationshipType)


def build_extraction_prompt(chunk_text: str) -> str:
    """Build the extraction prompt for one chunk of text.

    What problem it solves: the LLM must not invent its own schema —
    CLAUDE.md is explicit that entity/relationship types are centralized
    and controlled. This prompt embeds the exact allowed type lists so
    the model is steered toward them; validator.py then enforces it for
    real, since instruction-following alone is never guaranteed.
    """
    return f"""Extract entities and relationships from the text below.

Allowed entity types: {_ENTITY_TYPES}
Allowed relationship types: {_RELATIONSHIP_TYPES}

Rules:
- Only use the allowed types listed above. Do not invent new types.
- Every relationship must include the exact sentence (evidence) that supports it.
- Every relationship must include a confidence score between 0.0 and 1.0.
- Return ONLY valid JSON matching this shape, no other text:

{{
  "entities": [{{"name": "...", "entity_type": "..."}}],
  "relationships": [{{"source": "...", "target": "...", "relationship_type": "...", "evidence": "...", "confidence": 0.0}}]
}}

Text:
{chunk_text}
"""
