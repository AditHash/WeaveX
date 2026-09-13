"""Phase 10 tests — semantic chunking V2 (embedding-based boundaries).

embed_fn here is a deterministic bag-of-words counter over real corpus
vocabulary — NOT a real embedding model (that's Phase 20's job), but a
real, non-random function computed FROM the actual FastAPICorpus
paragraph text, so the resulting similarities are genuine signal (shared
topic -> higher similarity) rather than hand-typed magic numbers. Exact
threshold values below were computed with this same function beforehand
(not hand-estimated) — see the similarity numbers in each test's comment.
"""

from app.chunking import chunk_document_semantic
from app.models import Document

from tests.fixtures import FASTAPI_CORPUS_TEXT

_VOCAB = [
    "fastapi", "python", "framework", "pydantic", "starlette", "validation",
    "sebastián", "ramírez", "created", "microservices", "web", "apis",
]


def fake_embed(text: str) -> list[float]:
    """Bag-of-words vector over _VOCAB. Deterministic, real text in ->
    real (if simple) vector out. Computed similarities between the
    corpus's 4 real paragraphs (verified with a standalone script, not
    hand-estimated):
      P1<->P2 ("FastAPI...framework." <-> "FastAPI uses Pydantic..."): 0.4472
      P2<->P3 ("...web functionality." <-> "Sebastián Ramírez created..."): 0.2236
      P3<->P4 ("...created FastAPI." <-> "...APIs and microservices."): 0.2887
    """
    lowered = text.lower()
    return [float(lowered.count(word)) for word in _VOCAB]


def _document() -> Document:
    return Document(title="test", content=FASTAPI_CORPUS_TEXT)


def test_high_threshold_merges_nothing():
    # default threshold=0.75 exceeds all three real pairwise similarities
    # (max is 0.4472) -> every paragraph stays its own chunk.
    chunks = chunk_document_semantic(_document(), fake_embed, max_chunk_size=1000)
    assert len(chunks) == 4


def test_zero_threshold_merges_everything_up_to_size_cap():
    # threshold=0.0 <= every similarity -> merge decision is purely the
    # size cap, same as V1's behavior would be at this size.
    chunks = chunk_document_semantic(
        _document(), fake_embed, max_chunk_size=1000, similarity_threshold=0.0
    )
    assert len(chunks) == 1
    assert chunks[0].text == FASTAPI_CORPUS_TEXT


def test_selective_threshold_merges_only_the_most_similar_pair():
    # 0.35 sits between sim(P3,P4)=0.2887 and sim(P1,P2)=0.4472, so only
    # P1+P2 (the most topically related pair — both about what FastAPI
    # is/uses) merge; P3 (who created it) and P4 (what it's used for)
    # stay separate, matching the real topic shifts in this corpus.
    chunks = chunk_document_semantic(
        _document(), fake_embed, max_chunk_size=1000, similarity_threshold=0.35
    )
    assert len(chunks) == 3
    assert "FastAPI is a modern Python web framework." in chunks[0].text
    assert "FastAPI uses Pydantic" in chunks[0].text
    assert chunks[1].text == "Sebastián Ramírez created FastAPI."
    assert chunks[2].text == "FastAPI is commonly used to build APIs and microservices."


def test_size_cap_overrides_high_similarity():
    # threshold=0.4 alone would merge P1+P2 (sim=0.4472 >= 0.4), but
    # max_chunk_size=30 is smaller than P1 alone (41 chars) -> the size
    # cap blocks every merge regardless of similarity.
    chunks = chunk_document_semantic(
        _document(), fake_embed, max_chunk_size=30, similarity_threshold=0.4
    )
    assert len(chunks) == 4


def test_chunk_offsets_match_document_content():
    chunks = chunk_document_semantic(_document(), fake_embed, max_chunk_size=1000)
    document = _document()
    for chunk in chunks:
        assert chunk.text == FASTAPI_CORPUS_TEXT[chunk.start_offset : chunk.end_offset]


def test_chunk_indices_sequential():
    chunks = chunk_document_semantic(
        _document(), fake_embed, max_chunk_size=1000, similarity_threshold=0.35
    )
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_embed_fn_called_once_per_paragraph():
    calls = []

    def counting_embed(text: str) -> list[float]:
        calls.append(text)
        return fake_embed(text)

    chunk_document_semantic(_document(), counting_embed, max_chunk_size=1000)
    assert len(calls) == 4  # once per paragraph, not once per chunk or per pair
