"""Graph traversal: BFS subgraph extraction, DFS, and BFS shortest path.

Kept separate from graph.py: Graph owns storage/CRUD/adjacency, this file
owns algorithms that operate on it through its public query methods
(get_outgoing/get_incoming) — never touching Graph's internals directly.
"""

from collections import deque
from dataclasses import dataclass, field

from app.graph.graph import EntityNotFoundError, Graph
from app.models import Entity, Relationship, RelationshipType


@dataclass
class Subgraph:
    """The induced subgraph a traversal touched: every entity visited and
    every relationship walked to reach them."""

    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)


def _passes_filters(
    rel: Relationship,
    relationship_type: RelationshipType | None,
    min_confidence: float | None,
) -> bool:
    if relationship_type is not None and rel.relationship_type != relationship_type:
        return False
    if min_confidence is not None and rel.confidence < min_confidence:
        return False
    return True


def _neighbor_edges(graph: Graph, entity_id: str) -> list[Relationship]:
    return graph.get_outgoing(entity_id) + graph.get_incoming(entity_id)


def _other_end(rel: Relationship, from_id: str) -> str:
    return rel.target_entity_id if rel.source_entity_id == from_id else rel.source_entity_id


def bfs_subgraph(
    graph: Graph,
    start_id: str,
    depth: int,
    relationship_type: RelationshipType | None = None,
    min_confidence: float | None = None,
) -> Subgraph:
    """Breadth-first traversal from start_id up to `depth` hops.

    See module/phase docs for why BFS is used here. relationship_type and
    min_confidence prune which edges are followed (and therefore which
    entities get reached) — an edge that fails the filter is neither
    included in the result nor traversed through.
    """
    start_entity = graph.get_entity(start_id)
    if start_entity is None:
        raise EntityNotFoundError(start_id)

    visited_entities: dict[str, Entity] = {start_id: start_entity}
    visited_rel_ids: dict[str, Relationship] = {}
    queue: deque[tuple[str, int]] = deque([(start_id, 0)])

    while queue:
        current_id, current_depth = queue.popleft()
        if current_depth >= depth:
            continue

        for rel in _neighbor_edges(graph, current_id):
            if not _passes_filters(rel, relationship_type, min_confidence):
                continue
            visited_rel_ids[rel.id] = rel

            neighbor_id = _other_end(rel, current_id)
            if neighbor_id not in visited_entities:
                visited_entities[neighbor_id] = graph.get_entity(neighbor_id)
                queue.append((neighbor_id, current_depth + 1))

    return Subgraph(
        entities=list(visited_entities.values()),
        relationships=list(visited_rel_ids.values()),
    )


def dfs_visit(graph: Graph, start_id: str, max_depth: int | None = None) -> list[Entity]:
    """Depth-first traversal order starting at start_id.

    See module/phase docs: included for completeness (spec's required
    algorithm set + the classic BFS/DFS contrast), not because retrieval
    currently needs DFS order specifically.
    """
    if graph.get_entity(start_id) is None:
        raise EntityNotFoundError(start_id)

    visited: dict[str, Entity] = {}
    stack: list[tuple[str, int]] = [(start_id, 0)]

    while stack:
        current_id, current_depth = stack.pop()
        if current_id in visited:
            continue
        visited[current_id] = graph.get_entity(current_id)

        if max_depth is not None and current_depth >= max_depth:
            continue

        for rel in _neighbor_edges(graph, current_id):
            neighbor_id = _other_end(rel, current_id)
            if neighbor_id not in visited:
                stack.append((neighbor_id, current_depth + 1))

    return list(visited.values())


def find_path(graph: Graph, source_id: str, target_id: str) -> list[Entity] | None:
    """Shortest path (fewest hops) from source_id to target_id, or None.

    See module/phase docs for why BFS guarantees shortest, not just any,
    path. Returns one such path (there may be multiple of equal length).
    """
    if graph.get_entity(source_id) is None:
        raise EntityNotFoundError(source_id)
    if graph.get_entity(target_id) is None:
        raise EntityNotFoundError(target_id)

    if source_id == target_id:
        return [graph.get_entity(source_id)]

    came_from: dict[str, str] = {}
    visited = {source_id}
    queue: deque[str] = deque([source_id])

    while queue:
        current_id = queue.popleft()
        for rel in _neighbor_edges(graph, current_id):
            neighbor_id = _other_end(rel, current_id)
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            came_from[neighbor_id] = current_id

            if neighbor_id == target_id:
                path_ids = [target_id]
                while path_ids[-1] != source_id:
                    path_ids.append(came_from[path_ids[-1]])
                path_ids.reverse()
                return [graph.get_entity(eid) for eid in path_ids]

            queue.append(neighbor_id)

    return None
