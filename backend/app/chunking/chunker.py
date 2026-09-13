"""Semantic chunking V1 — paragraph-based chunking.

Splits a document's (already-normalized) text into paragraphs and
greedily packs consecutive paragraphs into chunks under a max size, so
each chunk stays LLM-context-friendly while never splitting mid-
paragraph in V1.

Progression (per project goal): V1 paragraph-based (this) -> V2
embedding-based semantic boundary detection -> V3 entity-aware -> V4
relationship-aware. Later versions replace the boundary-decision rule;
the packing/offset-tracking mechanics built here carry forward.
"""

from app.config import settings
from app.models import Chunk, Document


def chunk_document(document: Document, max_chunk_size: int | None = None) -> list[Chunk]:
    """Split a document into paragraph-respecting chunks.

    What problem it solves: bounded LLM input without cutting sentences —
    naive fixed-size splitting would break mid-sentence and destroy the
    sentence-level evidence extraction depends on.

    Algorithm: split on blank lines (the paragraph boundary normalize_text
    guarantees), then greedily accumulate paragraphs into the current
    chunk while its span stays under max_chunk_size; start a new chunk
    when adding the next paragraph would exceed it.

    Complexity: O(n) in document length — one pass to locate paragraphs
    (one str.index call per paragraph from the previous search position),
    one pass to pack them.

    Limitations (V1, documented rather than silently handled):
      - A single paragraph longer than max_chunk_size becomes its own
        oversized chunk rather than being split further — no sentence-
        level splitting yet.
      - Assumes document.content is already normalized; does not call
        normalize_text itself (would risk shifting offsets away from
        whatever content the caller actually has).
      - Boundary decision is purely size-based — no semantic-similarity
        notion yet (that's V2).
    """
    if max_chunk_size is None:
        max_chunk_size = settings.chunk_size

    paragraphs = _split_paragraphs(document.content)

    chunks: list[Chunk] = []
    current_start: int | None = None
    current_end: int | None = None
    chunk_index = 0

    def flush() -> None:
        nonlocal chunk_index
        chunks.append(
            Chunk(
                document_id=document.id,
                text=document.content[current_start:current_end],
                chunk_index=chunk_index,
                start_offset=current_start,
                end_offset=current_end,
            )
        )
        chunk_index += 1

    for start, end, _text in paragraphs:
        if current_start is None:
            current_start, current_end = start, end
            continue

        if end - current_start <= max_chunk_size:
            current_end = end
        else:
            flush()
            current_start, current_end = start, end

    if current_start is not None:
        flush()

    return chunks


def _split_paragraphs(text: str) -> list[tuple[int, int, str]]:
    """Return (start_offset, end_offset, text) per blank-line-separated
    paragraph, offsets relative to the original `text`."""
    raw_paragraphs = [p for p in text.split("\n\n") if p.strip()]
    results: list[tuple[int, int, str]] = []
    search_from = 0
    for para in raw_paragraphs:
        stripped = para.strip("\n")
        start = text.index(stripped, search_from)
        end = start + len(stripped)
        results.append((start, end, stripped))
        search_from = end
    return results
