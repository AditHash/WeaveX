"""Phase 5 tests — bfs_subgraph / dfs_visit / find_path.

Built on the same FastAPICorpus fixture as Phases 3-4 (real 7-entity,
6-relationship graph with real per-relationship confidences), so depth
limits, relationship-type filters, and confidence filters are exercised
against actual worked-example values, not fabricated round numbers.
"""

import pytest

from app.graph import EntityNotFoundError, Graph, bfs_subgraph, dfs_visit, find_path
from app.models import Entity, EntityType, RelationshipType

from tests.fixtures import FastAPICorpus


def _graph_and_ids():
    corpus = FastAPICorpus()
    graph = corpus.build_graph()
    ids = {name: e.id for name, e in corpus.entities.items()}
    return corpus, graph, ids


# ------------------------------------------------------------- bfs_subgraph


def test_bfs_depth_zero_returns_only_start_node():
    _, graph, ids = _graph_and_ids()
    sub = bfs_subgraph(graph, ids["FastAPI"], depth=0)
    assert {e.canonical_name for e in sub.entities} == {"FastAPI"}
    assert sub.relationships == []


def test_bfs_depth_one_from_fastapi_reaches_all_direct_neighbors():
    _, graph, ids = _graph_and_ids()
    sub = bfs_subgraph(graph, ids["FastAPI"], depth=1)
    names = {e.canonical_name for e in sub.entities}
    # FastAPI + its 5 outgoing targets + its 1 incoming source, all at 1 hop
    assert names == {
        "FastAPI", "Python", "Pydantic", "Starlette",
        "API", "Microservices", "Sebastián Ramírez",
    }
    assert len(sub.relationships) == 6  # every relationship in the corpus


def test_bfs_depth_two_from_python_reaches_whole_corpus():
    # Python -> FastAPI (1 hop) -> everything else (2 hops), since FastAPI
    # is the hub every other entity connects through.
    _, graph, ids = _graph_and_ids()
    sub = bfs_subgraph(graph, ids["Python"], depth=2)
    assert len(sub.entities) == 7


def test_bfs_filters_by_relationship_type():
    _, graph, ids = _graph_and_ids()
    sub = bfs_subgraph(
        graph, ids["FastAPI"], depth=1, relationship_type=RelationshipType.USES
    )
    names = {e.canonical_name for e in sub.entities}
    assert names == {"FastAPI", "Pydantic", "Starlette"}
    assert len(sub.relationships) == 2


def test_bfs_filters_by_min_confidence():
    # PROVIDES edges (API, Microservices) are both confidence 0.85; raising
    # the bar to 0.9 should exclude them but keep USES (0.93/0.94),
    # WRITTEN_IN (0.95), CREATED (0.97).
    _, graph, ids = _graph_and_ids()
    sub = bfs_subgraph(graph, ids["FastAPI"], depth=1, min_confidence=0.9)
    names = {e.canonical_name for e in sub.entities}
    assert "API" not in names
    assert "Microservices" not in names
    assert {"Python", "Pydantic", "Starlette", "Sebastián Ramírez"} <= names


def test_bfs_missing_entity_raises():
    _, graph, _ = _graph_and_ids()
    with pytest.raises(EntityNotFoundError):
        bfs_subgraph(graph, "entity_missing", depth=1)


# ------------------------------------------------------------------ dfs_visit


def test_dfs_reaches_whole_connected_corpus():
    _, graph, ids = _graph_and_ids()
    visited = dfs_visit(graph, ids["FastAPI"])
    assert len(visited) == 7


def test_dfs_respects_max_depth():
    _, graph, ids = _graph_and_ids()
    visited = dfs_visit(graph, ids["FastAPI"], max_depth=0)
    assert {e.canonical_name for e in visited} == {"FastAPI"}


def test_dfs_missing_entity_raises():
    _, graph, _ = _graph_and_ids()
    with pytest.raises(EntityNotFoundError):
        dfs_visit(graph, "entity_missing")


# ------------------------------------------------------------------ find_path


def test_find_path_same_entity():
    _, graph, ids = _graph_and_ids()
    path = find_path(graph, ids["FastAPI"], ids["FastAPI"])
    assert [e.canonical_name for e in path] == ["FastAPI"]


def test_find_path_direct_edge():
    _, graph, ids = _graph_and_ids()
    path = find_path(graph, ids["FastAPI"], ids["Pydantic"])
    assert [e.canonical_name for e in path] == ["FastAPI", "Pydantic"]


def test_find_path_two_hops_through_hub():
    # Sebastián Ramírez -> FastAPI -> Pydantic (Ramírez has no direct edge
    # to Pydantic, only via the FastAPI hub).
    _, graph, ids = _graph_and_ids()
    path = find_path(graph, ids["Sebastián Ramírez"], ids["Pydantic"])
    assert [e.canonical_name for e in path] == [
        "Sebastián Ramírez", "FastAPI", "Pydantic",
    ]


def test_find_path_is_shortest_not_just_any():
    # Python and Starlette are both only reachable via FastAPI - shortest
    # path must be exactly 2 hops, not a longer detour.
    _, graph, ids = _graph_and_ids()
    path = find_path(graph, ids["Python"], ids["Starlette"])
    assert len(path) == 3
    assert path[0].canonical_name == "Python"
    assert path[-1].canonical_name == "Starlette"


def test_find_path_returns_none_when_disconnected():
    # Deliberately isolated: two real Entity objects, added to a fresh
    # Graph with no relationship between them, to exercise the "no path
    # exists" case (the real corpus is fully connected, so it can't
    # produce this case on its own).
    graph = Graph()
    a = Entity(canonical_name="Island A", entity_type=EntityType.CONCEPT)
    b = Entity(canonical_name="Island B", entity_type=EntityType.CONCEPT)
    graph.add_entity(a)
    graph.add_entity(b)

    assert find_path(graph, a.id, b.id) is None


def test_find_path_missing_entity_raises():
    _, graph, ids = _graph_and_ids()
    with pytest.raises(EntityNotFoundError):
        find_path(graph, "entity_missing", ids["FastAPI"])
    with pytest.raises(EntityNotFoundError):
        find_path(graph, ids["FastAPI"], "entity_missing")
