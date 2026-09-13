"""Phase 21-23 tests — VectorStore (storage + cosine similarity reuse +
top-K search). These three phases landed as one cohesive class: cosine
similarity was already built in Phase 10 (reused here verbatim, not
reimplemented) and top-K search is simply what .search() does — there
was no separate unit of work to split out.

Query-vs-chunk similarity numbers were computed with a standalone script
before writing assertions (same fake_embed vocab as test_semantic_chunker).
"""

import pytest

from app.models import VectorRecord
from app.vector import VectorStore

from tests.fixtures import FastAPICorpus
from tests.test_semantic_chunker import fake_embed


def _populated_store() -> tuple[VectorStore, FastAPICorpus]:
    corpus = FastAPICorpus()
    store = VectorStore()
    for chunk in corpus.chunks:
        store.add(VectorRecord(chunk_id=chunk.id, vector=fake_embed(chunk.text)))
    return store, corpus


def test_add_and_get():
    store = VectorStore()
    record = VectorRecord(chunk_id="chunk_1", vector=[0.1, 0.2, 0.3])
    store.add(record)
    assert store.get(record.id) == record
    assert len(store) == 1


def test_add_duplicate_id_raises():
    store = VectorStore()
    record = VectorRecord(chunk_id="chunk_1", vector=[0.1, 0.2])
    store.add(record)
    with pytest.raises(ValueError):
        store.add(record)


def test_remove():
    store = VectorStore()
    record = VectorRecord(chunk_id="chunk_1", vector=[0.1])
    store.add(record)
    store.remove(record.id)
    assert store.get(record.id) is None
    assert len(store) == 0


def test_remove_missing_raises():
    store = VectorStore()
    with pytest.raises(KeyError):
        store.remove("vec_missing")


def test_search_ranks_real_corpus_chunks_by_relevance():
    # Query about data validation should rank the Pydantic/Starlette
    # chunk highest — verified via script: P2=0.6325 > P4=0.4082 >
    # P1=P3=0.3536.
    store, corpus = _populated_store()
    query_vector = fake_embed("What does FastAPI use for data validation?")

    results = store.search(query_vector, k=4)

    assert results[0].record.chunk_id == corpus.chunks[1].id
    assert results[0].score == pytest.approx(0.6325, abs=1e-3)
    assert results[1].record.chunk_id == corpus.chunks[3].id  # microservices chunk, second


def test_search_respects_k():
    store, _ = _populated_store()
    results = store.search(fake_embed("FastAPI"), k=2)
    assert len(results) == 2


def test_search_k_larger_than_store_returns_all():
    store, _ = _populated_store()
    results = store.search(fake_embed("FastAPI"), k=100)
    assert len(results) == 4


def test_search_empty_store_returns_empty_list():
    store = VectorStore()
    assert store.search([1.0, 0.0], k=5) == []
