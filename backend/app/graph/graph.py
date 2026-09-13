"""In-memory knowledge graph: storage + adjacency index + basic CRUD.

Traversal (BFS, subgraph, find_path) is a separate phase — this file only
owns the container and its integrity rules:

  - a relationship can't reference an entity that doesn't exist
  - removing an entity removes every relationship touching it (no orphans)

Both are enforced here, at write time, rather than discovered later during
traversal.
"""

from app.models import Entity, Relationship


class EntityNotFoundError(KeyError):
    """Raised when an operation references an entity_id that isn't in the graph."""


class Graph:
    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._relationships: dict[str, Relationship] = {}
        # adjacency index: entity_id -> list of relationship_ids
        self._outgoing: dict[str, list[str]] = {}
        self._incoming: dict[str, list[str]] = {}

    # ------------------------------------------------------------ entities

    def add_entity(self, entity: Entity) -> None:
        if entity.id in self._entities:
            raise ValueError(f"entity '{entity.id}' already exists")
        self._entities[entity.id] = entity
        self._outgoing[entity.id] = []
        self._incoming[entity.id] = []

    def get_entity(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def remove_entity(self, entity_id: str) -> None:
        if entity_id not in self._entities:
            raise EntityNotFoundError(entity_id)

        # cascade: drop every relationship touching this entity, in either
        # direction, so nothing is left pointing at a deleted node
        touching = set(self._outgoing[entity_id]) | set(self._incoming[entity_id])
        for rel_id in touching:
            self.remove_relationship(rel_id)

        del self._entities[entity_id]
        del self._outgoing[entity_id]
        del self._incoming[entity_id]

    # ------------------------------------------------------- relationships

    def add_relationship(self, relationship: Relationship) -> None:
        if relationship.id in self._relationships:
            raise ValueError(f"relationship '{relationship.id}' already exists")
        if relationship.source_entity_id not in self._entities:
            raise EntityNotFoundError(relationship.source_entity_id)
        if relationship.target_entity_id not in self._entities:
            raise EntityNotFoundError(relationship.target_entity_id)

        self._relationships[relationship.id] = relationship
        self._outgoing[relationship.source_entity_id].append(relationship.id)
        self._incoming[relationship.target_entity_id].append(relationship.id)

    def get_relationship(self, relationship_id: str) -> Relationship | None:
        return self._relationships.get(relationship_id)

    def remove_relationship(self, relationship_id: str) -> None:
        rel = self._relationships.get(relationship_id)
        if rel is None:
            raise KeyError(relationship_id)

        self._outgoing[rel.source_entity_id].remove(relationship_id)
        self._incoming[rel.target_entity_id].remove(relationship_id)
        del self._relationships[relationship_id]

    # ------------------------------------------------------------- sizing

    def __len__(self) -> int:
        """Number of entities (matches len(graph.entities) intuition)."""
        return len(self._entities)
