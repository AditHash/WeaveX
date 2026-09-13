"""Vector storage — brute-force, in-memory.

Stores VectorRecord objects and finds the top-K most similar to a query
vector via cosine_similarity (Phase 10's function — reused here, not
reimplemented, exactly the reuse that function's own docstring
promised).

Brute-force scan is intentional for v0: CLAUDE.md 4.3 is explicit an ANN
index is a later optimization (Phase 39, only once correctness is proven
and performance is actually measured in Phase 38) — building one now
would be the premature optimization rule 20 warns against.
"""

from dataclasses import dataclass

from app.chunking.similarity import cosine_similarity
from app.models import VectorRecord


@dataclass
class SearchResult:
    record: VectorRecord
    score: float


class VectorStore:
    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    def add(self, record: VectorRecord) -> None:
        """Store one embedding. O(1)."""
        if record.id in self._records:
            raise ValueError(f"vector record '{record.id}' already exists")
        self._records[record.id] = record

    def get(self, record_id: str) -> VectorRecord | None:
        return self._records.get(record_id)

    def remove(self, record_id: str) -> None:
        if record_id not in self._records:
            raise KeyError(record_id)
        del self._records[record_id]

    def search(self, query_vector: list[float], k: int) -> list[SearchResult]:
        """Top-k most similar stored vectors to query_vector.

        What problem it solves: "which stored chunks are most relevant
        to this query" — the core operation retrieval (Phase 25) needs.

        Algorithm: brute force — compare the query against every stored
        vector, sort by similarity, return the top k.

        Complexity: O(N*d + N log N), N = stored vectors, d = vector
        dimension — the textbook brute-force bound. An ANN index
        (HNSW/IVF) would trade an index-build cost for sub-linear query
        time; that's Phase 39's job, not this one's.

        Limitations: no metadata filtering (e.g. "only chunks from
        document X") — callers filter results themselves for now.
        """
        scored = [
            SearchResult(
                record=record,
                score=cosine_similarity(query_vector, record.vector),
            )
            for record in self._records.values()
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    def __len__(self) -> int:
        return len(self._records)
