# CLAUDE.md — Unstructured-to-Graph RAG

This file is the project spec and working contract for building this system. Read it fully before writing code. When a decision isn't covered here, prefer the simplest option that keeps the "built from scratch" constraint intact, and flag the tradeoff rather than silently picking one.

## 1. What this project is

A pipeline that takes plain unstructured text, extracts entities and relationships from it using an LLM, stores them in a **custom-built graph engine** (no Neo4j, no third-party graph DB), stores chunk embeddings in a **custom-built vector engine** (NavixDB, or a scoped-down version of it), and serves hybrid (vector + graph) retrieval through an API, with a frontend that renders the knowledge graph interactively.

The point of the project is to demonstrate: (1) systems engineering — building a graph storage engine and reusing a vector engine from scratch, and (2) applied AI engineering — LLM-based structured extraction, entity resolution, and hybrid retrieval design. Both halves matter equally. Neither should be sacrificed to ship faster.

## 2. Explicit scope rules

**Build from scratch (this is the actual project):**
- Graph storage engine (nodes, edges, adjacency index, traversal, persistence)
- Vector storage engine (reuse NavixDB, or a minimal version scoped to this project if NavixDB isn't ready)
- Entity resolution / deduplication logic
- Hybrid retrieval merge logic (vector hits + graph expansion)

**Allowed as infrastructure/utilities (not the thing being demonstrated):**
- LLM API (Claude or GPT) for extraction and final answer generation — this is a foundation-model dependency, not a storage/algorithm dependency. Do not attempt to train or self-host a model for this.
- FastAPI for the API layer
- A frontend rendering/charting library (e.g. D3, react-force-graph, Cytoscape.js) for the graph visualization — building a force-directed layout algorithm from scratch is out of scope; using a library to *draw* the graph your engine produces is fine.
- Standard Python stdlib / typing / async tooling

**Explicitly NOT allowed:**
- Neo4j, ArangoDB, or any pre-built graph database
- ChromaDB, Pinecone, Weaviate, FAISS, pgvector, or any pre-built vector database
- Any "knowledge graph" or "GraphRAG" framework/library that does extraction, storage, or retrieval for you (e.g. LlamaIndex KG Index, Microsoft GraphRAG package) — these can be *read* for design ideas, never imported as dependencies

If a milestone seems to require one of the disallowed items, stop and re-scope the milestone instead of reaching for the shortcut.

## 3. Architecture

```
Frontend (graph viz)
      |
      v
API layer (FastAPI)
  - POST /documents   -> ingestion, kicks off async processing
  - GET  /documents/:id/status
  - POST /query        -> hybrid retrieval + answer generation
  - GET  /graph         -> subgraph for a given query/entity (for frontend rendering)
      |
      v
Processing pipeline (async workers)
  - chunk text
  - call LLM for entity/relation extraction (per chunk)
  - resolve/merge entities across chunks
  - write into graph engine
  - embed chunks, write into vector engine
      |                                   |
      v                                   v
Graph engine (custom)              Vector engine (NavixDB / custom)
  - node/edge store                  - embedding store
  - adjacency index                  - similarity search (HNSW)
  - traversal (BFS/k-hop)
  - persistence (disk)
```

## 4. Component specs

### 4.1 Extraction pipeline
- Input: raw text document.
- Chunk into ~500–1000 token windows with overlap; retain `doc_id`, `chunk_id`, `position` metadata on every chunk.
- Per chunk, call the LLM with a structured-output prompt requesting:
  - `entities`: list of `{name, type}`
  - `relations`: list of `{source, relation, target, evidence}` (evidence = the exact sentence supporting the relation)
- Validate the LLM's JSON against a schema (Pydantic). On malformed output, retry once with a stricter reminder, then skip and log the chunk rather than crash the pipeline.
- **Entity resolution**: after raw extraction across all chunks, merge entity mentions that refer to the same real-world thing.
  - Pass 1: exact/normalized string match (lowercase, strip punctuation).
  - Pass 2: embedding similarity on remaining unmatched entities (cosine similarity above a threshold → same cluster).
  - Each resolved entity becomes one graph node; keep all original surface forms as an alias list on the node.
- Output of this stage: a set of graph `add_node` / `add_edge` calls, each edge tagged with its source chunk id and the evidence text (provenance is not optional — every edge must be traceable back to the text that produced it).

### 4.2 Graph engine (custom-built)
- **Node/edge store**: dict-based, `nodes: dict[str, Node]`, `edges: dict[str, Edge]`. `Node = {id, type, name, aliases, properties}`. `Edge = {id, source, target, relation, evidence, source_chunk_id, confidence}`.
- **Adjacency index**: `dict[node_id, list[edge_id]]` for both outgoing and incoming edges — O(1) neighbor lookup, not a scan over all edges.
- **Traversal engine**: BFS with a `max_hops` parameter and a visited-set to avoid cycles; returns the induced subgraph (nodes + edges touched) for a given starting node or set of nodes.
- **Persistence**: serialize to JSON (or a simple binary format) on write, load on startup. A write-ahead log is a v2 concern, not v0.
- **Explicitly deferred to v2+**: a query DSL (e.g. Cypher-like pattern matching), community detection (Leiden), transaction/concurrency handling.

### 4.3 Vector engine
- Reuse NavixDB if it's stable enough to import as a module; otherwise build the minimal subset needed here: store chunk embeddings, support top-k cosine similarity search. Don't let this component block the rest of the pipeline — if NavixDB isn't ready, stub it with a naive linear-scan similarity search first and swap in the real engine later. The interface (`add(id, vector)`, `search(query_vector, k)`) should stay stable regardless of which implementation sits behind it.

### 4.4 Retrieval logic
- Given a query: embed it, run vector search for top-k candidate chunks/entities, then run graph traversal from the matched entities up to `k` hops to pull in connected context.
- Merge and deduplicate the vector results and graph-expansion results before passing to the LLM for answer generation.
- Return both the generated answer and the subgraph actually used, so the frontend can visualize exactly what informed the answer.

### 4.5 API layer
- `POST /documents` — accepts raw text, returns a job id, triggers async processing.
- `GET /documents/:id/status` — processing status (queued/running/done/failed).
- `POST /query` — `{question: str}` → `{answer: str, subgraph: {nodes, edges}}`.
- `GET /graph` — full or filtered graph dump for exploratory frontend browsing.

### 4.6 Frontend
- Graph rendering via a force-directed layout library (react-force-graph or Cytoscape.js).
- Click-to-expand nodes (fetch neighbors on demand via the graph engine's traversal, not by loading the whole graph at once).
- Highlight the subgraph returned by `/query` so a user can see the reasoning path behind an answer.

## 5. Milestones

- **v0** — Prove extraction quality on a small, single-topic corpus. Chunking → LLM extraction → entity resolution → in-memory graph (dict-based, no persistence yet) → print/inspect the resulting graph. No API, no frontend, no vector engine yet. Success criterion: manually reading the extracted graph, does it look right?
- **v1** — Add the graph engine's persistence layer, add the vector engine (stub or NavixDB), wire up hybrid retrieval, expose it through the FastAPI endpoints. Success criterion: ask a question, get an answer with a traceable subgraph.
- **v2** — Frontend graph visualization wired to `/query` and `/graph`. Success criterion: a demoable end-to-end flow — upload text, ask a question, see the answer and the graph.
- **v3 (optional, stretch)** — query DSL, community detection, larger corpus, ingestion pipeline hardening (retries, dedup across re-ingested documents).

## 6. Open design decisions (revisit as the project develops)

- Confidence scoring on extracted relations — flat accept/reject vs. a numeric score used to weight retrieval.
- How aggressively to merge entities in resolution (over-merging collapses distinct entities; under-merging fragments the graph).
- Whether chunk overlap size needs tuning per corpus type.
- Whether NavixDB is ready to import directly or needs its own stub period first.
