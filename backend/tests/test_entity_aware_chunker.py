"""Phase 11 tests — semantic chunking V3 (entity-aware boundaries).

Reuses fake_embed from test_semantic_chunker.py for the real-corpus
tests. The OR-logic test below uses the exact "FastAPI uses Pydantic." /
"Pydantic provides validation." scenario PLAN.md itself describes as the
reason entity continuity is needed — with a deliberately orthogonal
embed_fn so the test proves the entity signal (not similarity) causes
the merge.
"""

from app.chunking import chunk_document_entity_aware
from app.models import Document

from tests.fixtures import FASTAPI_CORPUS_TEXT
from tests.test_semantic_chunker import fake_embed


def _document(content: str = FASTAPI_CORPUS_TEXT) -> Document:
    return Document(title="test", content=content)


def test_entity_overlap_merges_everything_at_default_threshold():
    # At threshold=0.75, V2 alone merges nothing (max real similarity is
    # 0.4472 — see test_semantic_chunker.py). Every adjacent paragraph
    # pair in this corpus shares the entity "FastAPI" though (verified:
    # P1-P2, P2-P3, P3-P4 all overlap on {"FastAPI"}), so V3 merges
    # everything the size cap allows.
    chunks = chunk_document_entity_aware(_document(), fake_embed, max_chunk_size=1000)
    assert len(chunks) == 1
    assert chunks[0].text == FASTAPI_CORPUS_TEXT


def test_size_cap_still_overrides_entity_overlap():
    chunks = chunk_document_entity_aware(_document(), fake_embed, max_chunk_size=30)
    assert len(chunks) == 4


def test_shared_entity_merges_despite_zero_similarity():
    # The exact scenario PLAN.md describes: two sentences about the same
    # subject (Pydantic) phrased differently enough that similarity alone
    # wouldn't catch it. embed_fn here is deliberately orthogonal
    # (similarity = 0.0) to prove the MERGE happens because of the shared
    # entity, not despite the test accidentally having high similarity.
    text = "FastAPI uses Pydantic.\n\nPydantic provides validation."

    def orthogonal_embed(t: str) -> list[float]:
        return [1.0, 0.0] if "uses" in t else [0.0, 1.0]

    chunks = chunk_document_entity_aware(
        _document(text), orthogonal_embed, max_chunk_size=1000, similarity_threshold=0.9
    )
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_no_shared_entity_and_low_similarity_does_not_merge():
    # Contrast case: no shared entity, low similarity -> must NOT merge,
    # confirming the OR logic isn't accidentally an always-merge bug.
    text = "FastAPI uses Pydantic.\n\nThe weather is nice today."

    def orthogonal_embed(t: str) -> list[float]:
        return [1.0, 0.0] if "Pydantic" in t else [0.0, 1.0]

    chunks = chunk_document_entity_aware(
        _document(text), orthogonal_embed, max_chunk_size=1000, similarity_threshold=0.9
    )
    assert len(chunks) == 2


def test_chunk_offsets_match_document_content():
    chunks = chunk_document_entity_aware(_document(), fake_embed, max_chunk_size=1000)
    for chunk in chunks:
        assert chunk.text == FASTAPI_CORPUS_TEXT[chunk.start_offset : chunk.end_offset]


def test_chunk_indices_sequential():
    chunks = chunk_document_entity_aware(_document(), fake_embed, max_chunk_size=30)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
