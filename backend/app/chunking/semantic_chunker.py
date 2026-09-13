"""Semantic chunking V2 — embedding-based boundary detection.

V1 (chunker.py) decides chunk boundaries purely by character count. V2
replaces that decision rule with cosine similarity between consecutive
paragraph embeddings — merge two paragraphs if they're actually about
the same thing, not just because they're short. max_chunk_size still
applies as a hard cap either way.

Embeddings are NOT generated here. Generating an embedding is a model-
inference call (an external API) — that's explicitly Phase 20's job
("Embedding generation"), not this one's, even though this phase number
(10) comes long before it (20) in the build order. Resolution:
chunk_document_semantic takes an `embed_fn` callable as a parameter
(dependency injection) instead of calling an API itself. That lets the
boundary-detection algorithm be built and fully tested now, without
blocking on Phase 20's real client and without this phase prematurely
building that client itself.
"""

from collections.abc import Callable

from app.chunking.paragraphs import split_paragraphs
from app.chunking.similarity import cosine_similarity
from app.config import settings
from app.models import Chunk, Document

EmbedFn = Callable[[str], list[float]]


def chunk_document_semantic(
    document: Document,
    embed_fn: EmbedFn,
    max_chunk_size: int | None = None,
    similarity_threshold: float = 0.75,
) -> list[Chunk]:
    """Split a document into chunks using paragraph-embedding similarity.

    What problem it solves: V1 merges paragraphs purely by character
    count, which can lump unrelated short paragraphs together or split a
    single train of thought just because it's long. V2 only merges
    paragraphs that are actually about the same thing.

    Algorithm:
      1. Split into paragraphs (same offset-tracking split as V1).
      2. Embed each paragraph via embed_fn.
      3. Walk consecutive paragraphs; merge paragraph i+1 into the
         current chunk if cosine_similarity(embedding[i], embedding[i+1])
         >= similarity_threshold AND the merged span still fits under
         max_chunk_size. Otherwise flush and start a new chunk.

    Complexity: O(n) embed_fn calls (the expensive part in practice —
    each is a network call to a real embedding model) + O(n) similarity
    comparisons, each O(d) in embedding dimension.

    Limitations:
      - Only compares ADJACENT paragraphs (i vs i+1), not all pairs —
        cheap, and matches how documents are actually written/read, but
        a paragraph echoing something from three paragraphs back won't
        be detected as related. Multi-paragraph topic tracking is what
        Phase 11 (entity-aware chunking) adds on top of this.
      - similarity_threshold=0.75 is a starting guess, not evaluated
        against real data — CLAUDE.md's own "Open design decisions"
        section explicitly flags this exact number as something to
        revisit once there's an evaluation set.
      - Like V1, a single paragraph longer than max_chunk_size becomes
        its own oversized chunk rather than being split further.
    """
    if max_chunk_size is None:
        max_chunk_size = settings.chunk_size

    paragraphs = split_paragraphs(document.content)
    if not paragraphs:
        return []

    embeddings = [embed_fn(text) for _start, _end, text in paragraphs]

    chunks: list[Chunk] = []
    current_start, current_end = paragraphs[0][0], paragraphs[0][1]
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

    for i in range(1, len(paragraphs)):
        start, end, _text = paragraphs[i]
        similarity = cosine_similarity(embeddings[i - 1], embeddings[i])
        fits_size = (end - current_start) <= max_chunk_size

        if similarity >= similarity_threshold and fits_size:
            current_end = end
        else:
            flush()
            current_start, current_end = start, end

    flush()
    return chunks
