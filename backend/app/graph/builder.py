"""Graph builder — turns one chunk's validated ExtractionResult into
graph mutations (add_entity/add_relationship) plus Evidence rows.

Combines three responsibilities the spec lists as separate phases
(17 graph construction, 18 relationship dedup, 19 evidence/provenance)
because they're inseparable in practice: you can't correctly decide
whether a relationship already exists (dedup) without first resolving
both its endpoints to canonical entities (Phase 16), and every
relationship this builder creates or reuses needs an Evidence row
attached in the same step, or a relationship could land in the graph
with no provenance at all — which CLAUDE.md forbids outright.
"""

from dataclasses import dataclass, field

from app.entity_resolution import resolve_entity
from app.extraction import ExtractionResult
from app.graph.graph import Graph
from app.models import Chunk, Entity, Evidence, Relationship, RelationshipType


@dataclass
class BuildResult:
    """What one build_from_extraction() call actually did — for
    logging/inspection, not required by the graph itself."""

    new_entities: list[Entity] = field(default_factory=list)
    reused_entities: list[Entity] = field(default_factory=list)
    new_relationships: list[Relationship] = field(default_factory=list)
    reused_relationships: list[Relationship] = field(default_factory=list)
    new_evidence: list[Evidence] = field(default_factory=list)


class GraphBuilder:
    def __init__(self, graph: Graph) -> None:
        self._graph = graph

    def build_from_extraction(
        self, extraction: ExtractionResult, chunk: Chunk
    ) -> BuildResult:
        """Fold one chunk's extraction result into the graph.

        What problem it solves: connects the extraction pipeline
        (Phase 12) to the graph engine (Phases 3-6) — nothing before
        this actually puts LLM output into the graph.

        Algorithm:
          1. For each extracted entity, resolve_entity() against the
             graph; add it if new, else reuse the match (recording the
             surface form as a new alias if it differs from what's
             already there).
          2. For each extracted relationship, look up its resolved
             source/target entities from step 1.
          3. Look for an EXISTING relationship with the same
             (source_id, target_id, relationship_type). If found, don't
             create a duplicate edge (dedup) — just attach a new Evidence
             row to it. If not found, create the Relationship AND its
             first Evidence row together, so a relationship is never
             added without provenance.

        Complexity: O(e + r*k), e = entities in this extraction, r =
        relationships in this extraction, k = existing relationships
        touching a given source entity (dedup lookup via get_outgoing,
        itself an O(1) index access, then O(k) to scan for a type+target
        match).

        Limitations:
          - No locking / concurrency handling — matches this project's
            single-threaded, in-memory v0 scope.
          - Relationship confidence is NOT recomputed as an aggregate of
            its evidence when reusing a relationship for a dedup match
            (kept as the original value) — CLAUDE.md's own "Open design
            decisions" section flags flat vs. aggregate confidence as an
            explicitly unresolved question, not something to silently
            pick an answer for here.
        """
        result = BuildResult()
        name_to_entity: dict[str, Entity] = {}

        for extracted_entity in extraction.entities:
            resolution = resolve_entity(
                self._graph, extracted_entity.name, extracted_entity.entity_type
            )
            entity = resolution.entity

            if resolution.is_new:
                self._graph.add_entity(entity)
                result.new_entities.append(entity)
            else:
                result.reused_entities.append(entity)
                self._maybe_add_alias(entity, extracted_entity.name)

            name_to_entity[extracted_entity.name] = entity

        for extracted_rel in extraction.relationships:
            source = name_to_entity.get(extracted_rel.source)
            target = name_to_entity.get(extracted_rel.target)
            if source is None or target is None:
                # validator.py should already have caught this upstream;
                # skip defensively rather than crash the pipeline.
                continue

            existing = self._find_existing_relationship(
                source.id, target.id, extracted_rel.relationship_type
            )

            if existing is not None:
                relationship = existing
                result.reused_relationships.append(relationship)
            else:
                relationship = Relationship(
                    source_entity_id=source.id,
                    target_entity_id=target.id,
                    relationship_type=extracted_rel.relationship_type,
                    confidence=extracted_rel.confidence,
                )
                self._graph.add_relationship(relationship)
                result.new_relationships.append(relationship)

            evidence = Evidence(
                relationship_id=relationship.id,
                chunk_id=chunk.id,
                text=extracted_rel.evidence,
                confidence=extracted_rel.confidence,
            )
            self._graph.add_evidence(evidence)
            result.new_evidence.append(evidence)

        return result

    def _maybe_add_alias(self, entity: Entity, surface_form: str) -> None:
        if surface_form == entity.canonical_name or surface_form in entity.aliases:
            return
        entity.aliases.append(surface_form)

    def _find_existing_relationship(
        self, source_id: str, target_id: str, relationship_type: RelationshipType
    ) -> Relationship | None:
        for rel in self._graph.get_outgoing(source_id):
            if (
                rel.target_entity_id == target_id
                and rel.relationship_type == relationship_type
            ):
                return rel
        return None
