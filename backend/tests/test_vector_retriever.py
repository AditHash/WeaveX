"""Phase 25 tests — VectorRetriever, on the real FastAPICorpus chunks
and the same query/ranking already verified in test_vector_store.py.
"""

from app.models import VectorRecord
from app.retrieval import VectorRetriever
from app.vector import VectorStore

from tests.fixtures import FastAPICorpus
from tests.test_semantic_chunker import fake_embed


def _populated_retriever() -> tuple[VectorRetriever, FastAPICorpus]:
    corpus = FastAPICorpus()
    store = VectorStore()
    for chunk in corpus.chunks:
        store.add(VectorRecord(chunk_id=chunk.id, vector=fake_embed(chunk.text)))
    return VectorRetriever(store, fake_embed), corpus


def test_retrieve_ranks_real_corpus_chunks_by_relevance():
    # Same numbers already verified in test_vector_store.py: the
    # Pydantic/Starlette chunk (P2) should rank highest for a data
    # validation question.
    retriever, corpus = _populated_retriever()
    results = retriever.retrieve("What does FastAPI use for data validation?", k=4)

    assert results[0].record.chunk_id == corpus.chunks[1].id


def test_retrieve_respects_k():
    retriever, _ = _populated_retriever()
    results = retriever.retrieve("FastAPI", k=2)
    assert len(results) == 2


def test_embed_fn_called_once_per_query_not_per_stored_record():
    corpus = FastAPICorpus()
    store = VectorStore()
    for chunk in corpus.chunks:
        store.add(VectorRecord(chunk_id=chunk.id, vector=fake_embed(chunk.text)))

    calls = []

    def counting_embed(text: str) -> list[float]:
        calls.append(text)
        return fake_embed(text)

    retriever = VectorRetriever(store, counting_embed)
    retriever.retrieve("What does FastAPI use?", k=4)

    assert calls == ["What does FastAPI use?"]  # exactly one call, for the query


def test_retrieve_on_empty_store_returns_empty_list():
    store = VectorStore()
    retriever = VectorRetriever(store, fake_embed)
    assert retriever.retrieve("anything", k=5) == []
