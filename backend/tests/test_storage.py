"""Phase 6 tests — SQLite persistence.

Exercises the actual milestone from PLAN.md #18: create graph -> save ->
stop -> start -> load -> traverse. Uses the real FastAPICorpus fixture so
the round trip is checked against real entities/relationships/aliases/
confidences, and "traverse after reload" is checked with the same
traversal functions (get_neighbors, find_path) Phase 4/5 already proved
correct — not a bespoke equality check invented just for persistence.

Every repository is opened with `with` (or closed manually) — SQLite
holds an OS-level file lock on Windows, and pytest's tmp_path cleanup
(unlike this file's earlier tempfile.TemporaryDirectory attempt) doesn't
surface that as a failure, but leaving connections open is still a real
resource leak worth avoiding.
"""

from app.graph import Graph, find_path
from app.models import Entity, EntityType
from app.storage import SQLiteGraphRepository

from tests.fixtures import FastAPICorpus


def test_save_and_load_round_trip_preserves_everything(tmp_path):
    corpus = FastAPICorpus()
    graph = corpus.build_graph()
    db_path = tmp_path / "weavex.db"

    with SQLiteGraphRepository(db_path) as repo:
        repo.save_graph(graph)

    # Fresh repository instance, same file — simulates the process
    # restarting rather than reusing any in-memory state.
    with SQLiteGraphRepository(db_path) as repo:
        reloaded_graph = repo.load_graph()

    assert len(reloaded_graph) == len(graph)
    for entity in graph.list_entities():
        restored = reloaded_graph.get_entity(entity.id)
        assert restored is not None
        assert restored == entity  # full Pydantic equality: type, aliases, metadata

    assert len(reloaded_graph.list_relationships()) == len(graph.list_relationships())
    for rel in graph.list_relationships():
        assert reloaded_graph.get_relationship(rel.id) == rel


def test_reloaded_graph_is_fully_traversable(tmp_path):
    # The actual point of persistence: not just "data comes back", but
    # "the graph still behaves like a graph" after reload.
    corpus = FastAPICorpus()
    graph = corpus.build_graph()
    ids = {name: e.id for name, e in corpus.entities.items()}
    db_path = tmp_path / "weavex.db"

    with SQLiteGraphRepository(db_path) as repo:
        repo.save_graph(graph)

    with SQLiteGraphRepository(db_path) as repo:
        reloaded = repo.load_graph()

    outgoing = reloaded.get_outgoing(ids["FastAPI"])
    assert len(outgoing) == 5

    path = find_path(reloaded, ids["Sebastián Ramírez"], ids["Pydantic"])
    assert [e.canonical_name for e in path] == [
        "Sebastián Ramírez", "FastAPI", "Pydantic",
    ]


def test_aliases_round_trip_in_order(tmp_path):
    corpus = FastAPICorpus()
    fastapi = corpus.entities["FastAPI"]
    assert fastapi.aliases == ["Fast API", "fastapi"]  # fixture's actual order

    graph = Graph()
    graph.add_entity(fastapi)
    db_path = tmp_path / "aliases.db"

    with SQLiteGraphRepository(db_path) as repo:
        repo.save_graph(graph)

    with SQLiteGraphRepository(db_path) as repo:
        reloaded = repo.load_graph()

    restored = reloaded.get_entity(fastapi.id)
    assert restored.aliases == ["Fast API", "fastapi"]


def test_save_is_a_full_snapshot_not_an_accumulation(tmp_path):
    # Saving graph A, then a smaller graph B to the same file, must leave
    # only B's data behind — not A ∪ B. This is what makes save_graph
    # safe to call repeatedly rather than only once at shutdown.
    db_path = tmp_path / "weavex.db"

    graph_a = Graph()
    solo = Entity(canonical_name="Solo Entity", entity_type=EntityType.CONCEPT)
    graph_a.add_entity(solo)
    with SQLiteGraphRepository(db_path) as repo:
        repo.save_graph(graph_a)

    corpus = FastAPICorpus()
    graph_b = corpus.build_graph()
    with SQLiteGraphRepository(db_path) as repo:
        repo.save_graph(graph_b)

    with SQLiteGraphRepository(db_path) as repo:
        reloaded = repo.load_graph()

    assert len(reloaded) == 7  # only graph_b's entities
    assert reloaded.get_entity(solo.id) is None  # graph_a's entity is gone


def test_empty_graph_round_trips(tmp_path):
    db_path = tmp_path / "empty.db"
    with SQLiteGraphRepository(db_path) as repo:
        repo.save_graph(Graph())

    with SQLiteGraphRepository(db_path) as repo:
        reloaded = repo.load_graph()

    assert len(reloaded) == 0
    assert reloaded.list_relationships() == []


def test_entity_with_none_description_round_trips(tmp_path):
    db_path = tmp_path / "weavex.db"
    entity = Entity(canonical_name="Nameless", entity_type=EntityType.CONCEPT)
    assert entity.description is None

    graph = Graph()
    graph.add_entity(entity)
    with SQLiteGraphRepository(db_path) as repo:
        repo.save_graph(graph)

    with SQLiteGraphRepository(db_path) as repo:
        reloaded = repo.load_graph()

    assert reloaded.get_entity(entity.id).description is None
