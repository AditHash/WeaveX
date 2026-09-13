# WeaveX

**From-scratch knowledge graph + hybrid RAG engine.**

> Weaving unstructured knowledge into an intelligent, queryable graph.

Takes plain unstructured text and turns it into an interconnected, evidence-backed
knowledge graph — then uses that graph together with vector retrieval to answer
questions with traceable provenance. The graph engine and vector engine are built
from scratch (no Neo4j, no FAISS/Chroma/Pinecone, no GraphRAG framework); an LLM
is used only for extraction and answer generation.

```text
Plain Text
    ↓
Intelligent Chunking
    ↓
Semantic Extraction
    ↓
Entity Resolution
    ↓
Knowledge Graph
    ↓
Graph + Vector Retrieval
    ↓
Hybrid RAG
    ↓
Grounded Answer
```

See [CLAUDE.md](CLAUDE.md) / [PLAN.md](PLAN.md) for the full spec and build order.

## Repository layout

```text
weavex/
├── backend/
├── frontend/
├── tests/
├── evaluation/
├── data/
├── scripts/
├── docs/
├── README.md
└── PLAN.md
```

## Status

`backend/` scaffolded (uv + FastAPI). Core engine (graph, extraction, entity
resolution) not yet built — see current milestone in PLAN.md.
