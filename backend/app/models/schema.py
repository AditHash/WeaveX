"""Centralized entity/relationship type vocabulary.

The LLM extractor must not be allowed to invent arbitrary types — every
extracted entity/relationship type has to be one of these, or it gets
rejected during extraction validation (Phase 13). Keeping the vocabulary
here, in one file, means there's exactly one place to look when deciding
"is this a valid type" and exactly one place to extend it.
"""

from enum import StrEnum


class EntityType(StrEnum):
    PERSON = "Person"
    COMPANY = "Company"
    ORGANIZATION = "Organization"
    PRODUCT = "Product"
    TECHNOLOGY = "Technology"
    FRAMEWORK = "Framework"
    PROGRAMMING_LANGUAGE = "ProgrammingLanguage"
    DATABASE = "Database"
    PROJECT = "Project"
    CONCEPT = "Concept"
    LOCATION = "Location"


class RelationshipType(StrEnum):
    USES = "USES"
    BUILDS = "BUILDS"
    CREATED = "CREATED"
    WORKS_FOR = "WORKS_FOR"
    DEPENDS_ON = "DEPENDS_ON"
    WRITTEN_IN = "WRITTEN_IN"
    IMPLEMENTS = "IMPLEMENTS"
    PROVIDES = "PROVIDES"
    PART_OF = "PART_OF"
    RELATED_TO = "RELATED_TO"
    LOCATED_IN = "LOCATED_IN"
