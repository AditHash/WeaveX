"""SQLite persistence for the graph engine.

Snapshot-based, not write-ahead: save_graph() replaces the on-disk state
with a full snapshot of the in-memory graph; load_graph() rebuilds a
Graph from that snapshot. A write-ahead log (durable per-mutation
persistence) is explicitly deferred to v2 per CLAUDE.md 4.2 — this is
enough to satisfy the actual Phase 6 milestone: create graph, save, stop
the program, start it again, load, keep traversing.

All SQL lives here. Graph (app/graph/graph.py) has zero knowledge of
SQLite — this repository only calls Graph's public methods
(list_entities/list_relationships to read, add_entity/add_relationship
to write).

Only entities/relationships/aliases are persisted. documents/chunks/
evidence tables are deliberately not built yet: nothing produces those
objects until the chunking/extraction/provenance phases, and Graph
itself doesn't hold them today — persisting tables nothing writes to
would be exactly the premature-scaffolding rule 22 warns against.
"""

import json
import sqlite3
from pathlib import Path

from app.graph import Graph
from app.models import Entity, EntityType, Relationship, RelationshipType

SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    description TEXT,
    metadata TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entity_aliases (
    entity_id TEXT NOT NULL REFERENCES entities(id),
    alias TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    source_entity_id TEXT NOT NULL REFERENCES entities(id),
    target_entity_id TEXT NOT NULL REFERENCES entities(id),
    relationship_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    metadata TEXT NOT NULL
);
"""


class SQLiteGraphRepository:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SQLiteGraphRepository":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def save_graph(self, graph: Graph) -> None:
        """Overwrite the on-disk snapshot with the graph's current state."""
        cur = self._conn.cursor()
        cur.execute("DELETE FROM entity_aliases")
        cur.execute("DELETE FROM relationships")
        cur.execute("DELETE FROM entities")

        for entity in graph.list_entities():
            cur.execute(
                "INSERT INTO entities "
                "(id, canonical_name, entity_type, description, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    entity.id,
                    entity.canonical_name,
                    entity.entity_type.value,
                    entity.description,
                    json.dumps(entity.metadata),
                ),
            )
            for alias in entity.aliases:
                cur.execute(
                    "INSERT INTO entity_aliases (entity_id, alias) VALUES (?, ?)",
                    (entity.id, alias),
                )

        for rel in graph.list_relationships():
            cur.execute(
                "INSERT INTO relationships "
                "(id, source_entity_id, target_entity_id, relationship_type, "
                " confidence, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    rel.id,
                    rel.source_entity_id,
                    rel.target_entity_id,
                    rel.relationship_type.value,
                    rel.confidence,
                    json.dumps(rel.metadata),
                ),
            )

        self._conn.commit()

    def load_graph(self) -> Graph:
        """Rebuild a Graph from the on-disk snapshot."""
        graph = Graph()
        cur = self._conn.cursor()

        aliases_by_entity: dict[str, list[str]] = {}
        for entity_id, alias in cur.execute(
            "SELECT entity_id, alias FROM entity_aliases ORDER BY rowid"
        ):
            aliases_by_entity.setdefault(entity_id, []).append(alias)

        for entity_id, canonical_name, entity_type, description, metadata in cur.execute(
            "SELECT id, canonical_name, entity_type, description, metadata FROM entities"
        ):
            entity = Entity(
                id=entity_id,
                canonical_name=canonical_name,
                entity_type=EntityType(entity_type),
                aliases=aliases_by_entity.get(entity_id, []),
                description=description,
                metadata=json.loads(metadata),
            )
            graph.add_entity(entity)

        for (
            rel_id,
            source_id,
            target_id,
            rel_type,
            confidence,
            metadata,
        ) in cur.execute(
            "SELECT id, source_entity_id, target_entity_id, relationship_type, "
            "confidence, metadata FROM relationships"
        ):
            relationship = Relationship(
                id=rel_id,
                source_entity_id=source_id,
                target_entity_id=target_id,
                relationship_type=RelationshipType(rel_type),
                confidence=confidence,
                metadata=json.loads(metadata),
            )
            graph.add_relationship(relationship)

        return graph
