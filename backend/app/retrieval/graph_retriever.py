"""Graph retrieval — given a question, find relevant graph facts.

Entity identification here is a cheap word-boundary matching heuristic
against known entity names/aliases, not real NLP entity linking — that
would need an LLM or dedicated NER model. Extraction (Phase 12) already
does real LLM-based entity recognition, but on ingested DOCUMENT text;
recognizing entity mentions inside a QUESTION at query time is a related
but separate problem this phase deliberately doesn't solve with the same
machinery.
"""

import re
import unicodedata

from app.graph import Graph, Subgraph, bfs_subgraph
from app.models import Entity, Relationship, RelationshipType


def _light_normalize(text: str) -> str:
    """Lowercase + unicode-compose, WITHOUT stripping spaces/punctuation.

    Deliberately NOT entity_resolution.normalize_entity_name — that
    function strips all spacing/punctuation down to one contiguous
    token, which is right for exact-name-equality comparison but wrong
    here: it would collapse "FastAPI" and "API" into substrings of each
    other with no word boundary left to tell them apart (a real bug this
    module used to have — see find_mentioned_entities below).
    """
    return unicodedata.normalize("NFC", text).lower()


def find_mentioned_entities(graph: Graph, query: str) -> list[Entity]:
    """Find graph entities plausibly mentioned in a free-text query.

    What problem it solves: graph traversal needs a starting node, but a
    query arrives as free text ("What does FastAPI use for validation?"),
    not an entity_id.

    Algorithm: light-normalize the query (case/unicode only, spaces and
    punctuation preserved); for each graph entity, regex-search for its
    normalized canonical_name OR any alias as a WHOLE WORD (\\b...\\b) in
    the query. Cheap, deterministic, no LLM call.

    Complexity: O(k*q), k = entities in the graph, q = query length
    (each check is a bounded regex scan).

    Limitations: catches "FastAPI" in a query containing "FastAPI" as a
    standalone word, but not typos, pronouns ("it"), or paraphrases
    ("the framework"). Good enough as a first-pass entity linker; a real
    system's NER/embedding-based query linking is out of this phase's
    scope.

    Correctness note: word-boundary matching (not raw substring
    containment) is required, not a style choice — this corpus's own
    entity "API" is a literal substring of "FastAPI" ("fast-API"), so
    substring matching would wrongly flag "API" as mentioned in every
    FastAPI question. \\b anchors on both sides reject that false match
    while still matching "API" as its own word elsewhere.
    """
    normalized_query = _light_normalize(query)
    matches = []
    for entity in graph.list_entities():
        candidates = [entity.canonical_name, *entity.aliases]
        for candidate in candidates:
            if not candidate:
                continue
            pattern = r"\b" + re.escape(_light_normalize(candidate)) + r"\b"
            if re.search(pattern, normalized_query):
                matches.append(entity)
                break
    return matches


class GraphRetriever:
    def __init__(self, graph: Graph) -> None:
        self._graph = graph

    def retrieve(
        self,
        query: str,
        depth: int = 2,
        relationship_type: RelationshipType | None = None,
    ) -> Subgraph:
        """Turn a free-text question into the relevant slice of the graph.

        What problem it solves: produces the graph-facts half of hybrid
        retrieval (Phase 26) — grounding context a RAG generator can cite.

        Algorithm: find_mentioned_entities() for starting points, then
        bfs_subgraph() (Phase 5) from EACH of them up to `depth` hops,
        merging results (entities/relationships deduplicated by id).

        Complexity: O(m * (V' + E')), m = mentioned entities found, each
        bfs_subgraph call bounded as documented in Phase 5.

        Limitations: if no known entity is found in the query, returns an
        empty Subgraph rather than guessing — Phase 26's hybrid retriever
        is expected to fall back to vector-only results in that case,
        that fallback is not this class's job.
        """
        start_entities = find_mentioned_entities(self._graph, query)
        if not start_entities:
            return Subgraph()

        merged_entities: dict[str, Entity] = {}
        merged_relationships: dict[str, Relationship] = {}

        for entity in start_entities:
            sub = bfs_subgraph(
                self._graph, entity.id, depth=depth, relationship_type=relationship_type
            )
            for e in sub.entities:
                merged_entities[e.id] = e
            for r in sub.relationships:
                merged_relationships[r.id] = r

        return Subgraph(
            entities=list(merged_entities.values()),
            relationships=list(merged_relationships.values()),
        )
