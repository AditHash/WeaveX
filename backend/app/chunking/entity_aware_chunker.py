"""Semantic chunking V3 — entity-aware boundary detection.

V2 (semantic_chunker.py) merges consecutive paragraphs by embedding
similarity alone. V3 adds a second, independent signal: shared candidate
entities (extract_candidate_entities) between them. Two paragraphs merge
if EITHER signal says they're related — high semantic similarity, OR a
shared capitalized phrase — still bounded by max_chunk_size either way.

Why OR, not AND: similarity and entity overlap catch different cases.
"FastAPI uses Pydantic." followed by "Pydantic provides validation." may
have only moderate embedding similarity (different sentence subjects)
but obviously belong together — entity overlap ("Pydantic") is what
catches that. PLAN.md's own worked example describes exactly this
scenario as the reason entity continuity is needed alongside similarity;
requiring both signals to agree (AND) would miss it.
"""

from collections.abc import Callable

from app.chunking.entities import extract_candidate_entities
from app.chunking.paragraphs import split_paragraphs
from app.chunking.similarity import cosine_similarity
from app.config import settings
from app.models import Chunk, Document

EmbedFn = Callable[[str], list[float]]


def chunk_document_entity_aware(
    document: Document,
    embed_fn: EmbedFn,
    max_chunk_size: int | None = None,
    similarity_threshold: float = 0.75,
) -> list[Chunk]:
    """Split a document into chunks using similarity OR entity overlap.

    What problem it solves: V2 alone under-merges paragraphs that share a
    subject but phrase it differently enough to score low on embedding
    similarity. Entity overlap catches that class of case.

    Algorithm: same paragraph split + per-paragraph embedding as V2, plus
    a candidate-entity set per paragraph. Merge paragraph i+1 into the
    current chunk if (similarity >= threshold OR entity sets overlap)
    AND the merged span still fits max_chunk_size.

    Complexity: O(n) embed_fn calls + O(n) regex passes (entity
    extraction) + O(n) comparisons, each O(d) similarity + O(k) set
    intersection (k = candidate entities per paragraph, small in practice).

    Limitations:
      - Inherits extract_candidate_entities's known false-positive risk
        (shared sentence-starter capitalization) — see that function's
        docstring.
      - Still only compares ADJACENT paragraphs, same as V2.
      - Like V1/V2, an oversized single paragraph becomes its own chunk
        rather than being split further.
    """
    if max_chunk_size is None:
        max_chunk_size = settings.chunk_size

    paragraphs = split_paragraphs(document.content)
    if not paragraphs:
        return []

    embeddings = [embed_fn(text) for _start, _end, text in paragraphs]
    entity_sets = [extract_candidate_entities(text) for _start, _end, text in paragraphs]

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
        shares_entity = bool(entity_sets[i - 1] & entity_sets[i])
        fits_size = (end - current_start) <= max_chunk_size

        if (similarity >= similarity_threshold or shares_entity) and fits_size:
            current_end = end
        else:
            flush()
            current_start, current_end = start, end

    flush()
    return chunks
