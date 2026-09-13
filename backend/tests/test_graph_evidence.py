"""Phase 17 tests — Graph evidence storage (add_evidence/get_evidence,
cascade delete through relationship removal).

Built on the real FastAPICorpus fixture, which already carries real
Evidence rows for its 6 relationships (fixtures.py's own _add_fact()).
"""

import pytest

from app.models import Evidence

from tests.fixtures import FastAPICorpus


def test_corpus_evidence_attaches_to_graph():
    # build_graph() already wires in corpus.evidence (fixtures.py) — one
    # row per relationship, six relationships.
    corpus = FastAPICorpus()
    graph = corpus.build_graph()

    assert len(graph.list_evidence()) == 6


def test_get_evidence_for_relationship():
    corpus = FastAPICorpus()
    graph = corpus.build_graph()

    uses_pydantic = next(
        r for r in corpus.relationships
        if r.relationship_type == "USES"
        and r.target_entity_id == corpus.entities["Pydantic"].id
    )
    evidence = graph.get_evidence(uses_pydantic.id)
    assert len(evidence) == 1
    assert evidence[0].text == "FastAPI uses Pydantic for data validation."


def test_get_evidence_for_relationship_with_none_returns_empty_list():
    # Every relationship in the corpus already has one evidence row via
    # build_graph() — build a fresh graph with a relationship but
    # deliberately no evidence attached, to exercise the empty case.
    corpus = FastAPICorpus()
    graph = corpus.build_graph()
    rel = corpus.relationships[0]
    graph.remove_relationship(rel.id)  # drops its evidence too (cascade)
    graph.add_relationship(rel)  # re-add the same relationship, now bare

    assert graph.get_evidence(rel.id) == []


def test_add_evidence_requires_existing_relationship():
    corpus = FastAPICorpus()
    graph = corpus.build_graph()
    orphan_evidence = Evidence(
        relationship_id="rel_missing", chunk_id="chunk_1",
        text="some text", confidence=0.9,
    )
    with pytest.raises(KeyError):
        graph.add_evidence(orphan_evidence)


def test_multiple_evidence_rows_for_one_relationship():
    # build_graph() already attaches one corpus evidence row to every
    # relationship — check the two NEW rows are both present alongside
    # it, not that they're the only two (they aren't).
    corpus = FastAPICorpus()
    graph = corpus.build_graph()
    rel = corpus.relationships[0]

    ev1 = Evidence(relationship_id=rel.id, chunk_id="chunk_a", text="a", confidence=0.9)
    ev2 = Evidence(relationship_id=rel.id, chunk_id="chunk_b", text="b", confidence=0.8)
    graph.add_evidence(ev1)
    graph.add_evidence(ev2)

    result = graph.get_evidence(rel.id)
    assert {ev1.id, ev2.id} <= {e.id for e in result}


def test_removing_relationship_cascades_its_evidence():
    corpus = FastAPICorpus()
    graph = corpus.build_graph()
    rel = corpus.relationships[0]
    ev = Evidence(relationship_id=rel.id, chunk_id="chunk_a", text="a", confidence=0.9)
    graph.add_evidence(ev)

    graph.remove_relationship(rel.id)

    assert ev.id not in {e.id for e in graph.list_evidence()}


def test_removing_entity_cascades_through_relationship_to_evidence():
    # Two levels of cascade: remove_entity -> remove_relationship ->
    # evidence cascade — proves the chain works end to end, not just the
    # direct relationship-removal case above.
    corpus = FastAPICorpus()
    graph = corpus.build_graph()
    fastapi_id = corpus.entities["FastAPI"].id
    rel = next(r for r in corpus.relationships if r.source_entity_id == fastapi_id)
    ev = Evidence(relationship_id=rel.id, chunk_id="chunk_a", text="a", confidence=0.9)
    graph.add_evidence(ev)

    graph.remove_entity(fastapi_id)

    assert ev.id not in {e.id for e in graph.list_evidence()}
