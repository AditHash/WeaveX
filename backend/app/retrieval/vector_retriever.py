"""Vector retrieval — given a question, find the most relevant chunks.

Wraps VectorStore (Phase 21-23) with an embed_fn — same dependency-
injection pattern used throughout this codebase (chunking V2/V3,
extraction, the real HuggingFace/Groq client wrappers) — so this stays
decoupled from whichever embedding provider is actually configured.
"""

from app.embedding.client import EmbedFn
from app.vector import SearchResult, VectorStore


class VectorRetriever:
    def __init__(self, store: VectorStore, embed_fn: EmbedFn) -> None:
        self._store = store
        self._embed_fn = embed_fn

    def retrieve(self, query: str, k: int = 10) -> list[SearchResult]:
        """Find the top-k stored chunks most semantically similar to a query.

        What problem it solves: turns a free-text question into the
        vector-search half of hybrid retrieval (Phase 26) — "which
        stored text reads like this question", independent of any
        entity/relationship structure.

        Algorithm: embed the query with the SAME embed_fn used to embed
        the stored chunks — embedding-space consistency matters, mixing
        vectors from two different models would make cosine similarity
        meaningless — then delegate to VectorStore.search().

        Complexity: one embed_fn call (network-bound in practice) plus
        VectorStore.search()'s O(N*d + N log N).

        Limitations: purely semantic — unlike GraphRetriever (Phase 24),
        it has no notion of relationship type or direction, just textual
        similarity. That's exactly why hybrid retrieval (Phase 26)
        combines both instead of picking one.
        """
        query_vector = self._embed_fn(query)
        return self._store.search(query_vector, k)
