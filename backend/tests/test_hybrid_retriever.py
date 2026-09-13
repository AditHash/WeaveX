"""Phase 26 tests — HybridRetriever, on the real FastAPICorpus graph +
vector store, reusing rankings already verified in Phase 24/25's tests.
"""

from app.models import RelationshipType, VectorRecord
from app.retrieval import GraphRetriever, HybridRetriever, VectorRetriever
from app.vector import VectorStore

from tests.fixtures import FastAPICorpus
from tests.test_semantic_chunker import fake_embed


def _hybrid_retriever() -> tuple[HybridRetriever, FastAPICorpus]:
    corpus = FastAPICorpus()
    graph = corpus.build_graph()
    store = VectorStore()
    for chunk in corpus.chunks:
        store.add(VectorRecord(chunk_id=chunk.id, vector=fake_embed(chunk.text)))

    retriever = HybridRetriever(
        GraphRetriever(graph), VectorRetriever(store, fake_embed)
    )
    return retriever, corpus


def test_retrieve_returns_both_graph_and_vector_results():
    retriever, corpus = _hybrid_retriever()
    result = retriever.retrieve(
        "What does FastAPI use for data validation?", graph_depth=1, vector_k=4
    )

    # Graph side: same depth-1 hub result already verified in Phase 24.
    names = {e.canonical_name for e in result.subgraph.entities}
    assert names == {
        "FastAPI", "Python", "Pydantic", "Starlette",
        "API", "Microservices", "Sebastián Ramírez",
    }

    # Vector side: same top-ranked chunk already verified in Phase 25.
    assert result.vector_results[0].record.chunk_id == corpus.chunks[1].id


def test_graph_depth_and_vector_k_are_passed_through():
    retriever, _ = _hybrid_retriever()
    result = retriever.retrieve("FastAPI", graph_depth=0, vector_k=2)

    assert {e.canonical_name for e in result.subgraph.entities} == {"FastAPI"}
    assert len(result.vector_results) == 2


def test_relationship_type_filter_applies_to_graph_side_only():
    retriever, _ = _hybrid_retriever()
    result = retriever.retrieve(
        "What does FastAPI use?",
        graph_depth=1,
        vector_k=4,
        relationship_type=RelationshipType.USES,
    )
    names = {e.canonical_name for e in result.subgraph.entities}
    assert names == {"FastAPI", "Pydantic", "Starlette"}
    assert len(result.vector_results) == 4  # vector side unaffected by the filter


def test_vector_results_still_returned_when_no_entity_mentioned():
    # No known entity named in this query -> empty subgraph, but vector
    # search doesn't depend on entity matching at all and should still
    # return results. Proves the two retrievers are truly independent.
    retriever, _ = _hybrid_retriever()
    result = retriever.retrieve("What is the weather today?", vector_k=4)

    assert result.subgraph.entities == []
    assert len(result.vector_results) == 4
