"""Retrieval ranking.

Combines HybridRetriever's two differently-shaped outputs (a Subgraph of
relationships+evidence, and a list of vector chunk hits) into one ranked
list, using a configurable weighted score. This is also where the real
overlap between the two sides — deliberately deferred from Phase 26 —
actually gets merged: a chunk that is BOTH a vector hit AND evidence for
a returned graph relationship becomes one item with both scores, not two.
"""

from dataclasses import dataclass

from app.graph import Graph
from app.models import Relationship
from app.retrieval.hybrid_retriever import HybridRetrievalResult

DEFAULT_GRAPH_WEIGHT = 0.4
DEFAULT_VECTOR_WEIGHT = 0.4
DEFAULT_CONFIDENCE_WEIGHT = 0.2


@dataclass
class RankedItem:
    relationship: Relationship | None = None  # set if this item carries a graph fact
    chunk_id: str | None = None  # set if this item has a known source chunk
    text: str = ""  # evidence text (single-chunk case), for display
    graph_score: float = 0.0
    vector_score: float = 0.0
    confidence: float = 0.0
    final_score: float = 0.0


def rank_hybrid_result(
    graph: Graph,
    result: HybridRetrievalResult,
    graph_weight: float = DEFAULT_GRAPH_WEIGHT,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    confidence_weight: float = DEFAULT_CONFIDENCE_WEIGHT,
) -> list[RankedItem]:
    """Score and order a HybridRetrievalResult into one ranked list.

    What problem it solves: HybridRetriever hands back two independent
    lists; something has to decide display/prompt order, and merge the
    case where a chunk is relevant on both axes rather than showing it
    twice.

    Algorithm:
      1. One RankedItem per graph relationship: graph_score = 1.0 (found
         by traversal at all), confidence = the relationship's own
         confidence. Its chunk_id is taken from Evidence.chunk_id ONLY
         when the relationship has exactly one evidence row — a
         relationship with evidence from multiple chunks has no single
         chunk to key on, so it's left unmerged (a documented
         limitation, not guessed at).
      2. For each vector SearchResult: if one or more relationship items
         share its chunk_id, apply its score to ALL of them (a single
         chunk commonly supports more than one extracted fact — e.g.
         one sentence asserting both "USES Pydantic" and "USES
         Starlette" — so this is a one-to-many match, not one-to-one).
         Otherwise add a new chunk-only item.
      3. final_score = graph_weight*graph_score + vector_weight*
         vector_score + confidence_weight*confidence, for every item.
      4. Sort descending by final_score.

    Complexity: O(g+v) to build and merge items (a dict of chunk_id ->
    list[RankedItem], not a nested scan), O(n log n) to sort.

    Limitations: weights are NOT evaluated against real data — CLAUDE.md's
    own "Open design decisions" section explicitly flags confidence/graph/
    vector weighting as unresolved. These defaults (equal graph/vector
    weight, confidence as a smaller tiebreaker) are a starting guess.
    """
    all_items: list[RankedItem] = []
    items_by_chunk: dict[str, list[RankedItem]] = {}

    for rel in result.subgraph.relationships:
        evidence = graph.get_evidence(rel.id)
        chunk_ids = {e.chunk_id for e in evidence}
        item = RankedItem(
            relationship=rel,
            text=evidence[0].text if len(evidence) == 1 else "",
            graph_score=1.0,
            confidence=rel.confidence,
        )
        if len(chunk_ids) == 1:
            item.chunk_id = next(iter(chunk_ids))
            items_by_chunk.setdefault(item.chunk_id, []).append(item)
        all_items.append(item)

    for search_result in result.vector_results:
        chunk_id = search_result.record.chunk_id
        matching = items_by_chunk.get(chunk_id)
        if matching:
            for item in matching:
                item.vector_score = search_result.score
        else:
            chunk_item = RankedItem(chunk_id=chunk_id, vector_score=search_result.score)
            all_items.append(chunk_item)
            items_by_chunk.setdefault(chunk_id, []).append(chunk_item)

    for item in all_items:
        item.final_score = (
            graph_weight * item.graph_score
            + vector_weight * item.vector_score
            + confidence_weight * item.confidence
        )

    all_items.sort(key=lambda i: i.final_score, reverse=True)
    return all_items
