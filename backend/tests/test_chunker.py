"""Phase 9 tests — semantic chunking V1 (paragraph-based).

Uses the real FastAPICorpus text for standard packing behavior, and a
real long single-paragraph passage lifted verbatim from CLAUDE.md itself
(the project's own spec, section 1) to exercise the "one paragraph alone
exceeds max_chunk_size" edge case — real prose long enough to actually
trigger it, not a fabricated wall of characters.
"""

from app.chunking import chunk_document
from app.models import Document

from tests.fixtures import FASTAPI_CORPUS_TEXT

# Verbatim from CLAUDE.md section 1 "What this project is" — a single
# paragraph, no blank lines inside, deliberately long (~400 chars).
LONG_REAL_PARAGRAPH = (
    "A pipeline that takes plain unstructured text, extracts entities and "
    "relationships from it using an LLM, stores them in a custom-built "
    "graph engine (no Neo4j, no third-party graph DB), stores chunk "
    "embeddings in a custom-built vector engine (NavixDB, or a "
    "scoped-down version of it), and serves hybrid (vector + graph) "
    "retrieval through an API, with a frontend that renders the "
    "knowledge graph interactively."
)


def _document(content: str) -> Document:
    return Document(title="test", content=content)


# ---------------------------------------------------------- offset invariant


def test_every_chunk_text_matches_its_own_offsets():
    # The core provenance invariant: chunk.text must be an exact
    # substring of document.content at [start_offset:end_offset].
    document = _document(FASTAPI_CORPUS_TEXT)
    chunks = chunk_document(document, max_chunk_size=60)
    for chunk in chunks:
        assert chunk.text == document.content[chunk.start_offset : chunk.end_offset]
        assert chunk.document_id == document.id


def test_chunk_indices_are_sequential_from_zero():
    document = _document(FASTAPI_CORPUS_TEXT)
    chunks = chunk_document(document, max_chunk_size=60)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# ------------------------------------------------------------- packing sizes


def test_large_max_size_merges_whole_corpus_into_one_chunk():
    document = _document(FASTAPI_CORPUS_TEXT)
    chunks = chunk_document(document, max_chunk_size=1000)
    assert len(chunks) == 1
    assert chunks[0].text == FASTAPI_CORPUS_TEXT
    assert chunks[0].start_offset == 0
    assert chunks[0].end_offset == len(FASTAPI_CORPUS_TEXT)


def test_small_max_size_produces_one_chunk_per_paragraph():
    # 60 chars is smaller than every individual paragraph gap needed to
    # merge two together (the corpus's 4 paragraphs are 42/80/35/59
    # chars), so packing can't merge any pair -> 4 chunks, one per
    # paragraph, in document order.
    document = _document(FASTAPI_CORPUS_TEXT)
    chunks = chunk_document(document, max_chunk_size=60)

    assert len(chunks) == 4
    assert chunks[0].text == "FastAPI is a modern Python web framework."
    assert chunks[1].text == (
        "FastAPI uses Pydantic for data validation and Starlette for "
        "web functionality."
    )
    assert chunks[2].text == "Sebastián Ramírez created FastAPI."
    assert chunks[3].text == "FastAPI is commonly used to build APIs and microservices."


def test_medium_max_size_merges_some_paragraphs_not_others():
    # Paragraph spans (chars, incl. blank-line separator to the next):
    # P1=41, P2=78 (P1+P2 combined span=121), P3=34, P4=57 (P3+P4
    # combined span=93). max=100 sits between those two combined spans:
    # too small to merge P1+P2 (121), big enough to merge P3+P4 (93).
    document = _document(FASTAPI_CORPUS_TEXT)
    chunks = chunk_document(document, max_chunk_size=100)

    assert len(chunks) == 3
    assert chunks[0].text == "FastAPI is a modern Python web framework."
    assert chunks[1].text == (
        "FastAPI uses Pydantic for data validation and Starlette for "
        "web functionality."
    )
    assert "Sebastián Ramírez created FastAPI." in chunks[2].text
    assert "FastAPI is commonly used to build APIs and microservices." in chunks[2].text


# ------------------------------------------------------- oversized paragraph


def test_paragraph_longer_than_max_size_becomes_its_own_oversized_chunk():
    document = _document(LONG_REAL_PARAGRAPH)
    assert len(LONG_REAL_PARAGRAPH) > 100  # sanity: this text really is long

    chunks = chunk_document(document, max_chunk_size=100)

    assert len(chunks) == 1
    assert chunks[0].text == LONG_REAL_PARAGRAPH
    assert len(chunks[0].text) > 100  # documented V1 limitation: not split further


def test_oversized_paragraph_does_not_get_merged_with_neighbors():
    text = f"Short intro.\n\n{LONG_REAL_PARAGRAPH}\n\nShort outro."
    document = _document(text)

    chunks = chunk_document(document, max_chunk_size=100)

    assert len(chunks) == 3
    assert chunks[0].text == "Short intro."
    assert chunks[1].text == LONG_REAL_PARAGRAPH
    assert chunks[2].text == "Short outro."


# ---------------------------------------------------------------------- misc


def test_default_max_chunk_size_comes_from_settings():
    from app.config import settings

    document = _document(FASTAPI_CORPUS_TEXT)
    default_chunks = chunk_document(document)
    explicit_chunks = chunk_document(document, max_chunk_size=settings.chunk_size)
    assert [c.text for c in default_chunks] == [c.text for c in explicit_chunks]


def test_single_paragraph_document():
    document = _document("Just one paragraph, nothing else.")
    chunks = chunk_document(document, max_chunk_size=1000)
    assert len(chunks) == 1
    assert chunks[0].text == "Just one paragraph, nothing else."
