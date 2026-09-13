"""Phase 4 tests — get_outgoing / get_incoming / get_neighbors.

Built on the same FastAPICorpus fixture as Phase 3 (real 7-entity,
6-relationship graph), so these exercise a node (FastAPI) with real
mixed in/out degree instead of a single fabricated edge.
"""

import pytest

from app.graph import EntityNotFoundError, Graph
from app.models import RelationshipType

from tests.fixtures import FastAPICorpus


def _graph_and_ids():
    corpus = FastAPICorpus()
    graph = corpus.build_graph()
    ids = {name: e.id for name, e in corpus.entities.items()}
    return corpus, graph, ids


# ---------------------------------------------------------------- outgoing


def test_get_outgoing_for_hub_node():
    corpus, graph, ids = _graph_and_ids()
    outgoing = graph.get_outgoing(ids["FastAPI"])
    # FastAPI --WRITTEN_IN--> Python, --USES--> Pydantic, --USES--> Starlette,
    # --PROVIDES--> API, --PROVIDES--> Microservices  (5 outgoing edges)
    assert len(outgoing) == 5
    assert all(r.source_entity_id == ids["FastAPI"] for r in outgoing)


def test_get_outgoing_for_leaf_node_is_empty():
    _, graph, ids = _graph_and_ids()
    # Python is only ever a target (WRITTEN_IN), never a source
    assert graph.get_outgoing(ids["Python"]) == []


def test_get_outgoing_missing_entity_raises():
    _, graph, _ = _graph_and_ids()
    with pytest.raises(EntityNotFoundError):
        graph.get_outgoing("entity_missing")


# ---------------------------------------------------------------- incoming


def test_get_incoming_for_fastapi():
    _, graph, ids = _graph_and_ids()
    incoming = graph.get_incoming(ids["FastAPI"])
    # only Sebastián Ramírez --CREATED--> FastAPI points at FastAPI
    assert len(incoming) == 1
    assert incoming[0].relationship_type == RelationshipType.CREATED
    assert incoming[0].source_entity_id == ids["Sebastián Ramírez"]


def test_get_incoming_for_source_only_node_is_empty():
    _, graph, ids = _graph_and_ids()
    # Sebastián Ramírez only ever points outward (CREATED), nothing points at him
    assert graph.get_incoming(ids["Sebastián Ramírez"]) == []


def test_get_incoming_missing_entity_raises():
    _, graph, _ = _graph_and_ids()
    with pytest.raises(EntityNotFoundError):
        graph.get_incoming("entity_missing")


# --------------------------------------------------------------- neighbors


def test_get_neighbors_is_undirected():
    corpus, graph, ids = _graph_and_ids()
    neighbors = graph.get_neighbors(ids["FastAPI"])
    neighbor_names = {n.canonical_name for n in neighbors}
    # 5 outgoing targets + 1 incoming source (Sebastián Ramírez) = 6 distinct
    assert neighbor_names == {
        "Python", "Pydantic", "Starlette", "API", "Microservices",
        "Sebastián Ramírez",
    }


def test_get_neighbors_filtered_by_relationship_type():
    _, graph, ids = _graph_and_ids()
    neighbors = graph.get_neighbors(ids["FastAPI"], relationship_type=RelationshipType.USES)
    neighbor_names = {n.canonical_name for n in neighbors}
    assert neighbor_names == {"Pydantic", "Starlette"}


def test_get_neighbors_filtered_by_type_with_no_matches():
    _, graph, ids = _graph_and_ids()
    neighbors = graph.get_neighbors(ids["Python"], relationship_type=RelationshipType.USES)
    assert neighbors == []


def test_get_neighbors_missing_entity_raises():
    _, graph, _ = _graph_and_ids()
    with pytest.raises(EntityNotFoundError):
        graph.get_neighbors("entity_missing")


def test_get_neighbors_isolated_node_after_removal():
    _, graph, ids = _graph_and_ids()
    graph.remove_entity(ids["Python"])  # removes FastAPI--WRITTEN_IN-->Python
    neighbors = graph.get_neighbors(ids["FastAPI"])
    assert "Python" not in {n.canonical_name for n in neighbors}
