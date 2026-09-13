"""Entity resolution.

Decides whether a newly extracted entity mention refers to an entity the
graph already has, or is genuinely new. Staged, cheapest-first, per
CLAUDE.md's "cheap deterministic methods first": exact normalized match,
then alias match, then string similarity. Semantic/LLM-assisted
resolution (the spec's final stage, for ambiguous cases like "Apple" vs
"Apple Inc.") is deliberately NOT built into this resolver — it would
need an embed_fn/llm_fn dependency the same way chunking V2/V3 and
extraction do, and this project doesn't need it wired in yet. This
resolver is the deterministic floor those stages would sit on top of.

Pure decision function: does NOT mutate the graph or write a new alias
onto a matched entity. That belongs to Phase 17's graph builder, which
has to update-and-persist the result anyway.
"""

from dataclasses import dataclass

from app.entity_resolution.normalizer import normalize_entity_name
from app.entity_resolution.similarity import levenshtein_similarity
from app.graph import Graph
from app.models import Entity, EntityType

AUTO_MERGE_THRESHOLD = 0.95
CANDIDATE_THRESHOLD = 0.75


@dataclass
class ResolutionResult:
    entity: Entity
    is_new: bool
    matched_via: str  # "exact" | "alias" | "similarity" | "new"


def resolve_entity(graph: Graph, name: str, entity_type: EntityType) -> ResolutionResult:
    """Decide which existing entity (if any) a mention refers to.

    What problem it solves: without this, every mention of "FastAPI"
    across different chunks creates a duplicate node instead of one
    canonical entity accumulating evidence.

    Algorithm (cheapest first):
      1. Exact normalized match against existing entities' canonical
         names AND their existing aliases.
      2. String similarity (Levenshtein) against existing entities of the
         SAME entity_type only — a Person and a Technology never merge
         regardless of name similarity, type is a hard filter.
      3. Otherwise: new entity.

    Scores in [CANDIDATE_THRESHOLD, AUTO_MERGE_THRESHOLD) are a known
    "maybe" zone but are NOT auto-merged. CLAUDE.md explicitly separates
    "candidate" from "automatic merge"; without a human/LLM-assisted step
    to adjudicate them (deferred, see module docstring), auto-merging a
    borderline match risks exactly the over-merge failure mode PLAN.md
    warns collapses distinct entities — so this resolver stays
    conservative and creates a new entity instead of guessing.

    Complexity: O(k), k = existing entities of the same type (compared
    against each one). Fine at this project's scale; would need an index
    (e.g. by normalized-name prefix) at real scale — not needed yet.
    """
    normalized_name = normalize_entity_name(name)

    for entity in graph.list_entities():
        if entity.entity_type != entity_type:
            continue
        if normalize_entity_name(entity.canonical_name) == normalized_name:
            return ResolutionResult(entity=entity, is_new=False, matched_via="exact")
        if any(
            normalize_entity_name(alias) == normalized_name for alias in entity.aliases
        ):
            return ResolutionResult(entity=entity, is_new=False, matched_via="alias")

    best_match: Entity | None = None
    best_score = 0.0
    for entity in graph.list_entities():
        if entity.entity_type != entity_type:
            continue
        score = levenshtein_similarity(name, entity.canonical_name)
        if score > best_score:
            best_score, best_match = score, entity

    if best_match is not None and best_score >= AUTO_MERGE_THRESHOLD:
        return ResolutionResult(entity=best_match, is_new=False, matched_via="similarity")

    new_entity = Entity(canonical_name=name, entity_type=entity_type)
    return ResolutionResult(entity=new_entity, is_new=True, matched_via="new")
