"""Extraction validation beyond Pydantic's structural checks.

Pydantic (schemas.py) already rejects unknown entity/relationship types
and out-of-range confidence at parse time. What's left is a cross-field
check Pydantic can't express on its own: every relationship's source/
target must name an entity that ALSO appears in the same extraction's
entity list. A relationship citing an entity the extraction never
mentioned is structurally valid JSON but logically broken — per
CLAUDE.md, "invalid extraction should never directly enter the graph".
"""

from app.extraction.schemas import ExtractionResult


def validate_extraction_result(result: ExtractionResult) -> list[str]:
    """Return validation error messages for one extraction (empty = valid).

    What problem it solves: an LLM can return well-formed JSON that
    passes Pydantic but is still internally inconsistent — e.g. a
    relationship citing "Pydantic" as source when "Pydantic" was never
    listed in this same extraction's entities. Structural validity isn't
    the same as referential validity.

    Complexity: O(n) — builds one set of known names, then one pass over
    relationships.
    """
    errors: list[str] = []
    known_names = {e.name for e in result.entities}

    for rel in result.relationships:
        if rel.source not in known_names:
            errors.append(
                f"relationship source '{rel.source}' is not in the extracted entities"
            )
        if rel.target not in known_names:
            errors.append(
                f"relationship target '{rel.target}' is not in the extracted entities"
            )
        if not rel.evidence.strip():
            errors.append(
                f"relationship {rel.source}->{rel.target} has empty evidence"
            )

    return errors
