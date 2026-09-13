"""Phase 27 tests — rank_hybrid_result, on the real FastAPICorpus graph
and vector store.
"""

from app.models import VectorRecord
from app.retrieval import (
    GraphRetriever,
    HybridRetrievalResult,
    HybridRetriever,
    VectorRetriever,
    rank_hybrid_result,
)
from app.vector import SearchResult, VectorStore

from tests.fixtures import FastAPICorpus
from tests.test_semantic_chunker import fake_embed


def _hybrid_result_and_graph():
    corpus = FastAPICorpus()
    graph = corpus.build_graph()
    store = VectorStore()
    for chunk in corpus.chunks:
        store.add(VectorRecord(chunk_id=chunk.id, vector=fake_embed(chunk.text)))

    retriever = HybridRetriever(GraphRetriever(graph), VectorRetriever(store, fake_embed))
    result = retriever.retrieve(
        "What does FastAPI use for data validation?", graph_depth=1, vector_k=4
    )
    return graph, result, corpus


def test_ranked_items_are_sorted_descending_by_final_score():
    graph, result, _ = _hybrid_result_and_graph()
    ranked = rank_hybrid_result(graph, result)

    scores = [item.final_score for item in ranked]
    assert scores == sorted(scores, reverse=True)


def test_graph_relationship_item_carries_its_own_confidence():
    graph, result, corpus = _hybrid_result_and_graph()
    ranked = rank_hybrid_result(graph, result)

    uses_pydantic = next(
        item for item in ranked
        if item.relationship is not None
        and item.relationship.target_entity_id == corpus.entities["Pydantic"].id
    )
    assert uses_pydantic.graph_score == 1.0
    assert uses_pydantic.confidence == 0.94  # fixtures.py's actual value for this fact


def test_every_item_traces_back_to_a_known_source():
    # Every ranked item must originate from either the subgraph's
    # relationships or the vector results -- nothing fabricated.
    graph, result, _ = _hybrid_result_and_graph()
    ranked = rank_hybrid_result(graph, result)

    known_relationship_ids = {r.id for r in result.subgraph.relationships}
    known_chunk_ids = {r.record.chunk_id for r in result.vector_results}

    for item in ranked:
        if item.relationship is not None:
            assert item.relationship.id in known_relationship_ids
        if item.vector_score > 0:
            assert item.chunk_id in known_chunk_ids


def test_chunk_that_is_both_graph_evidence_and_vector_hit_merges_into_one_item():
    corpus = FastAPICorpus()
    graph = corpus.build_graph()
    uses_pydantic = next(
        r for r in corpus.relationships
        if r.relationship_type == "USES"
        and r.target_entity_id == corpus.entities["Pydantic"].id
    )
    evidence_chunk_id = graph.get_evidence(uses_pydantic.id)[0].chunk_id

    result = HybridRetrievalResult(
        subgraph=type("S", (), {"entities": [], "relationships": [uses_pydantic]})(),
        vector_results=[
            SearchResult(
                record=VectorRecord(chunk_id=evidence_chunk_id, vector=[1.0]), score=0.9
            )
        ],
    )

    ranked = rank_hybrid_result(graph, result)
    assert len(ranked) == 1
    merged = ranked[0]
    assert merged.graph_score == 1.0
    assert merged.vector_score == 0.9
    assert merged.chunk_id == evidence_chunk_id


def test_relationship_with_multiple_evidence_chunks_stays_unmerged():
    from app.models import Evidence

    corpus = FastAPICorpus()
    graph = corpus.build_graph()
    rel = next(
        r for r in corpus.relationships
        if r.relationship_type == "USES"
        and r.target_entity_id == corpus.entities["Pydantic"].id
    )
    # Add a second evidence row from a different chunk -- now this
    # relationship has no single chunk_id to key a merge on.
    graph.add_evidence(
        Evidence(relationship_id=rel.id, chunk_id="chunk_other", text="more", confidence=0.8)
    )

    result = HybridRetrievalResult(
        subgraph=type("S", (), {"entities": [], "relationships": [rel]})(),
        vector_results=[],
    )
    ranked = rank_hybrid_result(graph, result)

    assert len(ranked) == 1
    assert ranked[0].chunk_id is None
    assert ranked[0].text == ""  # ambiguous which evidence to show, left blank


def test_custom_weights_change_final_score():
    graph, result, _ = _hybrid_result_and_graph()
    graph_heavy = rank_hybrid_result(
        graph, result, graph_weight=1.0, vector_weight=0.0, confidence_weight=0.0
    )
    vector_heavy = rank_hybrid_result(
        graph, result, graph_weight=0.0, vector_weight=1.0, confidence_weight=0.0
    )

    assert graph_heavy[0].final_score != vector_heavy[0].final_score
