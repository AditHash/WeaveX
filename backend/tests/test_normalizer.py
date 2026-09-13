"""Phase 7 tests — text normalization.

Deliberately dirties the real FASTAPI_CORPUS_TEXT (CRLF endings, doubled
spaces, trailing whitespace, extra blank lines, decomposed unicode
accents) and checks normalize_text recovers exactly the clean canonical
text — instead of testing against unrelated made-up strings. A few
narrow rule-specific tests use small literals where the rule being
checked (e.g. "don't touch newlines") needs an isolated case the real
corpus doesn't happen to exercise.
"""

import unicodedata

from app.chunking import normalize_text

from tests.fixtures import FASTAPI_CORPUS_TEXT


def _dirty(text: str) -> str:
    """Corrupt clean text the way real-world ingestion would: CRLF line
    endings, doubled internal spaces, trailing whitespace per line, extra
    blank lines between paragraphs, and decomposed unicode accents."""
    dirty = text.replace("\n\n", "\r\n\r\n\r\n")  # too many blank lines + CRLF
    dirty = dirty.replace(" ", "  ")  # double every space
    dirty = "\n".join(line + "   " for line in dirty.split("\n"))  # trailing ws
    return unicodedata.normalize("NFD", dirty)  # decompose accented chars


def test_normalize_recovers_clean_corpus_text():
    dirty = _dirty(FASTAPI_CORPUS_TEXT)
    assert dirty != FASTAPI_CORPUS_TEXT  # sanity: dirtying actually changed it

    assert normalize_text(dirty) == FASTAPI_CORPUS_TEXT


def test_normalize_is_idempotent():
    once = normalize_text(FASTAPI_CORPUS_TEXT)
    twice = normalize_text(once)
    assert once == twice


def test_normalize_composes_accented_characters():
    decomposed = unicodedata.normalize("NFD", "Sebastián Ramírez")
    assert decomposed != "Sebastián Ramírez"  # sanity: decomposition changed the bytes
    assert normalize_text(decomposed) == "Sebastián Ramírez"


def test_normalize_collapses_internal_whitespace_but_not_newlines():
    text = "FastAPI   uses    Pydantic.\n\nStarlette provides web functionality."
    result = normalize_text(text)
    assert result == "FastAPI uses Pydantic.\n\nStarlette provides web functionality."


def test_normalize_caps_blank_lines_at_one():
    text = "Paragraph one.\n\n\n\n\nParagraph two."
    assert normalize_text(text) == "Paragraph one.\n\nParagraph two."


def test_normalize_preserves_structural_separators():
    # Repeated punctuation used as a section divider must survive — collapsing
    # it is Phase 8's structure-detection job, not normalization's.
    text = "FASTAPI ARCHITECTURE\n====================\n\nFastAPI is a framework."
    result = normalize_text(text)
    assert "====================" in result


def test_normalize_strips_leading_and_trailing_whitespace():
    text = "\n\n  FastAPI is a framework.  \n\n"
    assert normalize_text(text) == "FastAPI is a framework."
