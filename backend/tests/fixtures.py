"""Realistic test data.

Built from the actual worked example in CLAUDE.md/PLAN.md — the FastAPI
introduction paragraph and its expected extraction result — not throwaway
placeholder names. This exercises the full provenance chain the way real
ingested data will: Document -> Chunk -> Entity/Relationship -> Evidence,
with real character offsets and real evidence sentences, not bare
Entity/Relationship objects floating with no source text behind them.
"""

from app.graph import Graph
from app.models import (
    Chunk,
    Document,
    Entity,
    EntityType,
    Evidence,
    Relationship,
    RelationshipType,
)

FASTAPI_CORPUS_TEXT = (
    "FastAPI is a modern Python web framework.\n\n"
    "FastAPI uses Pydantic for data validation and Starlette for web "
    "functionality.\n\n"
    "Sebastián Ramírez created FastAPI.\n\n"
    "FastAPI is commonly used to build APIs and microservices."
)


class FastAPICorpus:
    """Everything built from FASTAPI_CORPUS_TEXT, wired together like real
    ingested data would be: a Document, its Chunks (one per paragraph),
    seven Entities, six Relationships, and one Evidence row per
    relationship pointing at the chunk + exact sentence it came from.
    """

    def __init__(self) -> None:
        self.document = Document(
            title="FastAPI Introduction", content=FASTAPI_CORPUS_TEXT
        )
        self.chunks = self._build_chunks()
        self.entities = self._build_entities()
        self.relationships: list[Relationship] = []
        self.evidence: list[Evidence] = []
        self._build_relationships_and_evidence()

    def _build_chunks(self) -> list[Chunk]:
        paragraphs = [p.strip() for p in FASTAPI_CORPUS_TEXT.split("\n\n")]
        chunks = []
        search_from = 0
        for i, para in enumerate(paragraphs):
            start = FASTAPI_CORPUS_TEXT.index(para, search_from)
            end = start + len(para)
            chunks.append(
                Chunk(
                    document_id=self.document.id,
                    text=para,
                    chunk_index=i,
                    start_offset=start,
                    end_offset=end,
                )
            )
            search_from = end
        return chunks

    def _build_entities(self) -> dict[str, Entity]:
        return {
            "FastAPI": Entity(
                canonical_name="FastAPI",
                entity_type=EntityType.TECHNOLOGY,
                aliases=["Fast API", "fastapi"],
            ),
            "Python": Entity(
                canonical_name="Python", entity_type=EntityType.PROGRAMMING_LANGUAGE
            ),
            "Pydantic": Entity(
                canonical_name="Pydantic", entity_type=EntityType.TECHNOLOGY
            ),
            "Starlette": Entity(
                canonical_name="Starlette", entity_type=EntityType.TECHNOLOGY
            ),
            "Sebastián Ramírez": Entity(
                canonical_name="Sebastián Ramírez",
                entity_type=EntityType.PERSON,
            ),
            "API": Entity(canonical_name="API", entity_type=EntityType.CONCEPT),
            "Microservices": Entity(
                canonical_name="Microservices", entity_type=EntityType.CONCEPT
            ),
        }

    def _add_fact(
        self,
        source: str,
        target: str,
        relationship_type: RelationshipType,
        confidence: float,
        chunk: Chunk,
        evidence_text: str,
    ) -> None:
        rel = Relationship(
            source_entity_id=self.entities[source].id,
            target_entity_id=self.entities[target].id,
            relationship_type=relationship_type,
            confidence=confidence,
        )
        ev = Evidence(
            relationship_id=rel.id,
            chunk_id=chunk.id,
            text=evidence_text,
            confidence=confidence,
        )
        self.relationships.append(rel)
        self.evidence.append(ev)

    def _build_relationships_and_evidence(self) -> None:
        # Matches CLAUDE.md section 4 "Example" exactly.
        self._add_fact(
            "FastAPI", "Python", RelationshipType.WRITTEN_IN, 0.95,
            self.chunks[0], "FastAPI is a modern Python web framework.",
        )
        self._add_fact(
            "FastAPI", "Pydantic", RelationshipType.USES, 0.94,
            self.chunks[1], "FastAPI uses Pydantic for data validation.",
        )
        self._add_fact(
            "FastAPI", "Starlette", RelationshipType.USES, 0.93,
            self.chunks[1], "FastAPI uses ... Starlette for web functionality.",
        )
        self._add_fact(
            "Sebastián Ramírez", "FastAPI", RelationshipType.CREATED, 0.97,
            self.chunks[2], "Sebastián Ramírez created FastAPI.",
        )
        self._add_fact(
            "FastAPI", "API", RelationshipType.PROVIDES, 0.85,
            self.chunks[3], "FastAPI is commonly used to build APIs and microservices.",
        )
        self._add_fact(
            "FastAPI", "Microservices", RelationshipType.PROVIDES, 0.85,
            self.chunks[3], "FastAPI is commonly used to build APIs and microservices.",
        )

    def build_graph(self) -> Graph:
        """A populated Graph with every entity + relationship added."""
        graph = Graph()
        for entity in self.entities.values():
            graph.add_entity(entity)
        for rel in self.relationships:
            graph.add_relationship(rel)
        return graph
