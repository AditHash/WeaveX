"""Phase 24 tests — GraphRetriever, on the real FastAPICorpus graph."""

from app.models import RelationshipType
from app.retrieval import GraphRetriever, find_mentioned_entities

from tests.fixtures import FastAPICorpus


def _graph():
    corpus = FastAPICorpus()
    return corpus.build_graph(), corpus


# --------------------------------------------------- find_mentioned_entities


def test_finds_entity_mentioned_by_canonical_name():
    graph, corpus = _graph()
    matches = find_mentioned_entities(graph, "What does FastAPI use for data validation?")
    names = {e.canonical_name for e in matches}
    assert "FastAPI" in names
    # "Pydantic" isn't literally in the query text, so it shouldn't match —
    # this heuristic is substring-based, not semantic.
    assert "Pydantic" not in names


def test_finds_entity_mentioned_by_alias():
    graph, _ = _graph()
    matches = find_mentioned_entities(graph, "What does Fast API use?")
    names = {e.canonical_name for e in matches}
    assert "FastAPI" in names


def test_no_match_returns_empty_list():
    graph, _ = _graph()
    assert find_mentioned_entities(graph, "What is the weather today?") == []


def test_finds_multiple_mentioned_entities():
    graph, _ = _graph()
    matches = find_mentioned_entities(graph, "Did Sebastián Ramírez create FastAPI?")
    names = {e.canonical_name for e in matches}
    assert names == {"FastAPI", "Sebastián Ramírez"}


# --------------------------------------------------------------- retrieve


def test_retrieve_returns_hub_subgraph_at_depth_one():
    graph, _ = _graph()
    retriever = GraphRetriever(graph)
    sub = retriever.retrieve("What does FastAPI use for data validation?", depth=1)

    names = {e.canonical_name for e in sub.entities}
    # FastAPI is the hub — depth 1 reaches the whole 7-entity corpus.
    assert names == {
        "FastAPI", "Python", "Pydantic", "Starlette",
        "API", "Microservices", "Sebastián Ramírez",
    }


def test_retrieve_filters_by_relationship_type():
    graph, _ = _graph()
    retriever = GraphRetriever(graph)
    sub = retriever.retrieve(
        "What does FastAPI use?", depth=1, relationship_type=RelationshipType.USES
    )
    names = {e.canonical_name for e in sub.entities}
    assert names == {"FastAPI", "Pydantic", "Starlette"}


def test_retrieve_returns_empty_subgraph_when_no_entity_mentioned():
    graph, _ = _graph()
    retriever = GraphRetriever(graph)
    sub = retriever.retrieve("What is the weather today?")
    assert sub.entities == []
    assert sub.relationships == []


def test_retrieve_merges_results_from_multiple_start_entities_without_duplicates():
    graph, _ = _graph()
    retriever = GraphRetriever(graph)
    # Both "FastAPI" and "Pydantic" are mentioned — their depth-1
    # subgraphs overlap heavily (both reach each other); the merged
    # result must not contain duplicate entities.
    sub = retriever.retrieve("How does FastAPI relate to Pydantic?", depth=1)

    entity_ids = [e.id for e in sub.entities]
    assert len(entity_ids) == len(set(entity_ids))  # no duplicates
