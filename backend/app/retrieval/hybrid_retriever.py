"""Hybrid retrieval — combines graph facts and vector-similar chunks
into one result for the RAG generator (Phase 29) to ground an answer in.

Wraps GraphRetriever (Phase 24) and VectorRetriever (Phase 25) rather
than reimplementing either — this is purely a combination layer.
"""

from dataclasses import dataclass, field

from app.graph import Subgraph
from app.models import RelationshipType
from app.retrieval.graph_retriever import GraphRetriever
from app.retrieval.vector_retriever import VectorRetriever
from app.vector import SearchResult


@dataclass
class HybridRetrievalResult:
    subgraph: Subgraph
    vector_results: list[SearchResult] = field(default_factory=list)


class HybridRetriever:
    def __init__(
        self, graph_retriever: GraphRetriever, vector_retriever: VectorRetriever
    ) -> None:
        self._graph_retriever = graph_retriever
        self._vector_retriever = vector_retriever

    def retrieve(
        self,
        query: str,
        graph_depth: int = 2,
        vector_k: int = 10,
        relationship_type: RelationshipType | None = None,
    ) -> HybridRetrievalResult:
        """Run both retrieval strategies and return both results together.

        What problem it solves: graph retrieval answers "how are these
        entities connected" but misses relevant text that never happens
        to name a known entity; vector retrieval answers "what text
        reads similarly" but has no notion of relationship structure.
        Neither alone is enough — CLAUDE.md 4.4 requires both.

        Algorithm: run both retrievers independently (neither depends on
        the other's output) and return both results as-is.

        Complexity: O(graph retrieval) + O(vector retrieval) — the costs
        already documented in Phase 24/25, simply added since the two
        run independently.

        Limitations, both deliberate:
          - No deduplication between the two result shapes here. A
            Subgraph (entities/relationships) and a list of chunk search
            results aren't the same shape, so there's nothing to
            naturally merge between them yet — real overlap (a chunk
            that also happens to be evidence for a returned graph fact)
            is left for Phase 28's context builder, which is where both
            shapes actually get unified into prompt text anyway.
          - No ranking/weighting between the two result types — that's
            Phase 27 ("Ranking"), a deliberately separate concern from
            just gathering candidates.
        """
        subgraph = self._graph_retriever.retrieve(
            query, depth=graph_depth, relationship_type=relationship_type
        )
        vector_results = self._vector_retriever.retrieve(query, k=vector_k)
        return HybridRetrievalResult(subgraph=subgraph, vector_results=vector_results)
