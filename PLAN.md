# KnowledgeGraph-RAG — Development Plan

A from-scratch system that transforms unstructured plain text into an evidence-backed knowledge graph and uses that graph together with vector retrieval for RAG.

The purpose of this project is to understand and implement the core mechanics ourselves.

We will use external models for things that are fundamentally model-inference tasks, such as LLM generation and embeddings. We will NOT use third-party graph databases, vector databases, RAG frameworks, or graph-RAG frameworks.

---

# 0. Final System

The system we are building is:

```text
                    PLAIN TEXT CORPUS
                           |
                           v
                  +------------------+
                  | Document Manager |
                  +--------+---------+
                           |
                           v
                  +------------------+
                  | Intelligent      |
                  | Chunking Engine  |
                  +--------+---------+
                           |
                           v
                  +------------------+
                  | Semantic         |
                  | Extraction       |
                  +--------+---------+
                           |
                           v
                  +------------------+
                  | Entity           |
                  | Resolution       |
                  +--------+---------+
                           |
                           v
                  +------------------+
                  | Knowledge Graph  |
                  | Builder          |
                  +--------+---------+
                           |
               +-----------+-----------+
               |                       |
               v                       v
        +-------------+         +-------------+
        | Graph Store |         | Vector Store|
        +-------------+         +-------------+
               |                       |
               +-----------+-----------+
                           |
                           v
                  +------------------+
                  | Retrieval Engine |
                  +--------+---------+
                           |
              +------------+------------+
              |                         |
              v                         v
       Graph Retrieval            Vector Retrieval
              |                         |
              +------------+------------+
                           |
                           v
                  +------------------+
                  | Hybrid Retrieval |
                  +--------+---------+
                           |
                           v
                  +------------------+
                  | Context Builder  |
                  +--------+---------+
                           |
                           v
                         LLM
                           |
                           v
                   Answer + Evidence
```

The frontend comes after the core engine.

---

# 1. Development Philosophy

We will build this in layers.

Do not start with:

```text
FastAPI
React
Docker
LLM
Vector DB
Graph DB
```

all at once.

Instead:

```text
Algorithm
   ↓
Core engine
   ↓
Persistence
   ↓
Model integration
   ↓
Retrieval
   ↓
API
   ↓
Frontend
```

Every major subsystem should work independently before connecting it to the next subsystem.

The most important rule:

> If we cannot explain how a component works, we should not hide it behind a third-party abstraction.

---

# 2. What We Are Allowed to Use

Allowed:

```text
Python
SQLite
JSON
Pydantic
FastAPI
React
TypeScript
HTTP libraries
An LLM API
An embedding model/API
Basic utility libraries
```

Not allowed as core implementations:

```text
Neo4j
Pinecone
Weaviate
Chroma
FAISS
Milvus
LangChain
LlamaIndex
Graph-RAG frameworks
Vector database libraries
Graph database libraries
```

We are building the following ourselves:

```text
Intelligent chunking
Entity extraction pipeline
Entity normalization
Entity resolution
Graph representation
Graph traversal
Graph persistence layer
Vector storage
Vector similarity search
Graph retrieval
Vector retrieval
Hybrid retrieval
Context construction
RAG orchestration
Evaluation
```

---

# 3. Phase 1 — Project Skeleton

First create the repository.

```text
knowledge-graph-rag/

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

Initially only work inside:

```text
backend/
```

Create:

```text
backend/
└── app/
    ├── models/
    ├── chunking/
    ├── extraction/
    ├── entity_resolution/
    ├── graph/
    ├── vector/
    ├── retrieval/
    ├── rag/
    ├── storage/
    ├── config.py
    └── main.py
```

Do not create unnecessary infrastructure yet.

---

# 4. Phase 2 — Define the Data Model

Before writing algorithms, define what the system stores.

We need:

```text
Document
Chunk
Entity
Relationship
Evidence
VectorRecord
```

Create:

```text
models/
├── document.py
├── chunk.py
├── entity.py
├── relationship.py
├── evidence.py
└── vector.py
```

---

# 5. Document Model

Build:

```python
class Document:
    id
    title
    content
    metadata
```

Example:

```json
{
  "id": "doc_001",
  "title": "FastAPI Introduction",
  "content": "FastAPI is...",
  "metadata": {}
}
```

Requirements:

- unique ID
- title
- complete source text
- metadata

---

# 6. Chunk Model

Build:

```python
class Chunk:
    id
    document_id
    text
    chunk_index
    start_offset
    end_offset
    metadata
```

The chunk must always know where it came from.

For example:

```text
chunk_007
   |
   +-- document_id = doc_001
   +-- start = 1432
   +-- end = 2148
```

This is necessary for provenance.

---

# 7. Entity Model

Build:

```python
class Entity:
    id
    canonical_name
    entity_type
    aliases
    description
    metadata
```

Example:

```json
{
  "id": "entity_001",
  "canonical_name": "FastAPI",
  "entity_type": "Technology",
  "aliases": [
    "fastapi",
    "Fast API"
  ]
}
```

---

# 8. Relationship Model

Build:

```python
class Relationship:
    id
    source_entity_id
    target_entity_id
    relationship_type
    confidence
    metadata
```

A relationship must NOT simply be:

```text
FastAPI -> Pydantic
```

It must contain the relationship type:

```text
FastAPI --USES--> Pydantic
```

---

# 9. Evidence Model

Build:

```python
class Evidence:
    id
    relationship_id
    chunk_id
    text
    confidence
```

This allows:

```text
FastAPI
   |
  USES
   |
Pydantic
   |
   +-- Evidence
         |
         +-- chunk_004
         +-- "FastAPI uses Pydantic..."
```

This becomes extremely important later for RAG citations.

---

# 10. Phase 3 — Build the Graph Engine

Do this before using an LLM.

Create:

```text
graph/
├── graph.py
├── traversal.py
├── repository.py
└── models.py
```

Start with an in-memory graph.

---

# 11. Graph Data Structure

Implement:

```python
class Graph:
    ...
```

Internally maintain:

```text
entities
relationships
adjacency
```

For example:

```text
entities:

entity_001 -> FastAPI
entity_002 -> Pydantic
entity_003 -> Python
```

Relationships:

```text
rel_001 -> FastAPI USES Pydantic
rel_002 -> FastAPI WRITTEN_IN Python
```

Adjacency:

```text
FastAPI
  |
  +-- rel_001
  +-- rel_002
```

---

# 12. Implement Basic Graph Operations

Implement:

```python
add_entity()
get_entity()
remove_entity()

add_relationship()
get_relationship()
remove_relationship()
```

Then:

```python
get_neighbors()
get_incoming()
get_outgoing()
```

---

# 13. Implement Graph Traversal

Start with BFS.

Implement:

```python
get_subgraph(entity_id, depth)
```

Example:

```text
FastAPI

depth=1

FastAPI
 ├── Pydantic
 ├── Starlette
 └── Python
```

depth=2:

```text
FastAPI
 ├── Pydantic
 │    └── Validation
 │
 ├── Starlette
 │    └── Web
 │
 └── Python
      └── Programming Language
```

---

# 14. Implement Path Finding

Implement:

```python
find_path(source, target)
```

Example:

```text
FastAPI
   |
USES
   |
Pydantic
   |
PROVIDES
   |
Validation
```

Query:

```text
find_path(FastAPI, Validation)
```

returns:

```text
FastAPI
 -> Pydantic
 -> Validation
```

---

# 15. Graph Engine Milestone

At this point we should be able to manually create:

```python
graph.add_entity(...)
graph.add_relationship(...)
```

and query:

```python
graph.get_neighbors(...)
graph.get_subgraph(...)
graph.find_path(...)
```

No LLM.

No API.

No frontend.

If this does not work correctly, stop here and fix it.

---

# 16. Phase 4 — Graph Persistence

Now add SQLite.

Create:

```text
storage/
└── sqlite.py
```

Tables:

```text
documents
chunks
entities
entity_aliases
relationships
evidence
```

The graph engine should NOT directly execute SQL.

Use:

```text
Graph Engine
     |
Graph Repository
     |
SQLite Repository
     |
SQLite
```

---

# 17. Repository Interfaces

Create interfaces such as:

```python
class EntityRepository:
    save()
    get()
    delete()
    search()
```

```python
class RelationshipRepository:
    save()
    get()
    delete()
    find_between()
```

```python
class ChunkRepository:
    save()
    get()
```

This separates graph logic from persistence.

---

# 18. Persistence Milestone

Test:

```text
Create graph
      ↓
Save
      ↓
Stop program
      ↓
Start program
      ↓
Load graph
      ↓
Traverse graph
```

The graph should survive restarts.

---

# 19. Phase 5 — Text Normalization

Now begin the text pipeline.

Create:

```text
chunking/
├── normalizer.py
├── structure.py
├── similarity.py
├── segmenter.py
├── chunker.py
└── models.py
```

First implement text normalization.

Handle:

```text
extra whitespace
line breaks
unicode normalization
empty lines
repeated separators
```

Do not destroy meaningful formatting.

---

# 20. Phase 6 — Document Structure Detection

Before semantic chunking, detect structure.

Identify:

```text
titles
headings
paragraphs
lists
code blocks
quotes
tables if possible
```

For plain text, use heuristics.

For example:

```text
FASTAPI ARCHITECTURE
====================

FastAPI is...

Components
----------

FastAPI uses...
```

should become:

```text
Document
 ├── FASTAPI ARCHITECTURE
 │    └── paragraph
 │
 └── Components
      └── paragraph
```

Do not use an LLM for every structural decision.

---

# 21. Phase 7 — Semantic Segmentation

Now build the actual intelligent chunker.

The goal:

> Keep semantically related information together while respecting context-size limits.

The chunker should consider:

```text
semantic similarity
section boundaries
paragraph boundaries
entity continuity
topic continuity
token/character limit
```

---

# 22. Semantic Chunking Algorithm — V1

Start simple.

Split the document into paragraphs.

Then:

```text
Paragraph 1
Paragraph 2
Paragraph 3
Paragraph 4
...
```

Generate embeddings for each paragraph.

Calculate similarity:

```text
P1 <-> P2 = 0.91
P2 <-> P3 = 0.88
P3 <-> P4 = 0.31
```

If similarity is high:

```text
merge
```

If similarity drops:

```text
new chunk
```

So:

```text
P1 + P2 + P3
```

becomes:

```text
Chunk 1
```

and:

```text
P4
```

starts:

```text
Chunk 2
```

---

# 23. Chunk Boundary Detection

Build:

```python
detect_boundary(previous_unit, current_unit)
```

The result should consider:

```text
semantic similarity
heading changes
maximum size
entity continuity
```

Example:

```text
score > threshold
    => continue chunk

score < threshold
    => create new chunk
```

---

# 24. Semantic Chunking — V2

After V1 works, add entity continuity.

Suppose:

```text
Paragraph 1:
FastAPI uses Pydantic.

Paragraph 2:
Pydantic provides validation.

Paragraph 3:
Pydantic models can be customized.
```

Even if Paragraph 2 and 3 have moderate semantic similarity, the repeated entity:

```text
Pydantic
```

is a signal that they belong together.

So:

```text
semantic similarity
+
entity continuity
```

helps determine the boundary.

---

# 25. Semantic Chunking — V3

Add relationship continuity.

If:

```text
P1:
FastAPI uses Pydantic.

P2:
Pydantic provides validation.

P3:
This validation is performed before the request handler executes.
```

These three sentences form one logical chain.

The chunker should try to preserve the chain.

This is where the chunker begins becoming graph-aware.

---

# 26. Chunk Validation

After generating chunks, validate:

```text
No empty chunks
No extremely small chunks
No oversized chunks
No broken sentences where avoidable
No unnecessary section mixing
```

Store metadata:

```text
chunk_id
document_id
section
start_offset
end_offset
semantic_score
```

---

# 27. Intelligent Chunking Milestone

Given:

```text
large plain text
```

the system should produce:

```text
Chunk 1
Chunk 2
Chunk 3
...
```

and we should be able to inspect them manually.

Create a debugging command:

```bash
python inspect_chunks.py document.txt
```

Output:

```text
Chunk 1
-------
Topic: Authentication
Size: 812 tokens

...

Chunk 2
-------
Topic: Database
Size: 746 tokens

...
```

Do not move forward until the chunks make sense.

---

# 28. Phase 8 — Semantic Extraction

Now integrate the LLM.

Create:

```text
extraction/
├── extractor.py
├── schemas.py
├── prompts.py
└── validator.py
```

The extractor receives:

```text
Chunk
```

and produces:

```text
ExtractionResult
```

---

# 29. Define the Knowledge Schema

Do not let the LLM invent entity types.

Start with:

```text
Person
Company
Organization
Product
Technology
Framework
ProgrammingLanguage
Database
Project
Concept
Location
```

Relationship types:

```text
USES
BUILDS
CREATED
WORKS_FOR
DEPENDS_ON
WRITTEN_IN
IMPLEMENTS
PROVIDES
PART_OF
RELATED_TO
LOCATED_IN
```

Keep this controlled.

---

# 30. LLM Extraction Schema

Create Pydantic models:

```python
class ExtractedEntity:
    name
    entity_type
```

```python
class ExtractedRelationship:
    source
    target
    relationship_type
    evidence
    confidence
```

```python
class ExtractionResult:
    entities
    relationships
```

The LLM must return this structure.

---

# 31. Extraction Validation

Validate:

```text
entity type exists
relationship type exists
source entity exists
target entity exists
confidence is valid
evidence exists
```

Invalid extraction should never directly enter the graph.

---

# 32. Extraction Milestone

Given:

```text
FastAPI uses Pydantic for data validation.
```

we should obtain:

```text
Entities:

FastAPI
Pydantic
Data Validation

Relations:

FastAPI --USES--> Pydantic
Pydantic --PROVIDES--> Data Validation
```

and evidence:

```text
"FastAPI uses Pydantic for data validation."
```

---

# 33. Phase 9 — Entity Normalization

Create:

```text
entity_resolution/
├── normalizer.py
├── similarity.py
└── resolver.py
```

Normalize:

```text
GPT-5
GPT 5
GPT5
gpt-5
```

into comparable representations.

Implement:

```text
lowercase
whitespace normalization
punctuation normalization
unicode normalization
```

---

# 34. Entity Resolution — V1

Exact normalized match.

Example:

```text
FastAPI
fastapi
Fast API
```

should be candidates for the same entity.

---

# 35. Entity Resolution — V2

Implement string similarity.

Build your own:

```text
Levenshtein distance
token similarity
Jaccard similarity
```

Calculate:

```text
similarity(A, B)
```

Then use thresholds.

For example:

```text
> 0.95
automatic merge

0.75 - 0.95
candidate

< 0.75
different
```

The exact values should be evaluated rather than blindly assumed.

---

# 36. Entity Resolution — V3

Add semantic similarity.

Only ambiguous cases should reach an LLM.

Example:

```text
Apple
Apple Inc.
Apple company
```

The system can determine whether they refer to the same entity based on context.

The important principle:

```text
cheap deterministic methods first
        ↓
semantic similarity
        ↓
LLM only when necessary
```

---

# 37. Entity Resolution Milestone

After processing multiple chunks:

```text
Chunk 1:
FastAPI

Chunk 2:
Fast API

Chunk 3:
fastapi
```

the graph should contain:

```text
ONE canonical entity:

FastAPI
```

with aliases:

```text
Fast API
fastapi
```

---

# 38. Phase 10 — Graph Construction

Connect:

```text
Chunk
 ↓
Extraction
 ↓
Entity Resolution
 ↓
Graph Builder
```

The graph builder must:

```text
create entities
resolve entities
create relationships
deduplicate relationships
attach evidence
store confidence
```

---

# 39. Relationship Deduplication

Suppose:

```text
Chunk 1:
FastAPI uses Pydantic.

Chunk 2:
FastAPI uses Pydantic.

Chunk 3:
FastAPI uses Pydantic.
```

Do not create three separate logical relationships.

Create:

```text
FastAPI --USES--> Pydantic
```

with:

```text
Evidence:
chunk_001
chunk_008
chunk_019
```

---

# 40. Graph Construction Milestone

At this stage:

```text
plain text
   ↓
intelligent chunks
   ↓
entities
   ↓
relationships
   ↓
entity resolution
   ↓
graph
```

should work end-to-end.

This is the most important milestone in the project.

---

# 41. Phase 11 — Vector Engine

Now build the vector system.

Create:

```text
vector/
├── models.py
├── store.py
├── similarity.py
└── search.py
```

---

# 42. Vector Record

Create:

```python
class VectorRecord:
    id
    chunk_id
    vector
    metadata
```

The vector must always point to a chunk.

```text
vector
  |
  v
chunk
  |
  v
document
```

---

# 43. Vector Storage

Initially use a simple structure.

For example:

```python
vectors = {
    "chunk_001": [...],
    "chunk_002": [...],
    "chunk_003": [...]
}
```

Persist it using SQLite or a binary file format.

Do not use FAISS or another vector engine.

---

# 44. Implement Cosine Similarity

Implement:

```python
cosine_similarity(a, b)
```

Formula:

```text
       A · B
------------------
||A|| × ||B||
```

Test this independently.

Examples:

```text
same vector -> 1
orthogonal vectors -> 0
opposite vectors -> -1
```

---

# 45. Implement Top-K Search

Build:

```python
search(query_vector, k)
```

Algorithm:

```text
query
 ↓
compare against every vector
 ↓
calculate similarity
 ↓
sort
 ↓
return top K
```

This is intentionally brute-force.

Do not optimize yet.

---

# 46. Vector Engine Milestone

Given:

```text
"What does FastAPI use for validation?"
```

the vector engine should return relevant chunks:

```text
chunk_004
chunk_019
chunk_022
```

with scores.

---

# 47. Phase 12 — Graph Retrieval

Create:

```text
retrieval/
├── graph_retriever.py
├── vector_retriever.py
├── hybrid_retriever.py
└── ranking.py
```

GraphRetriever should accept a question.

Example:

```text
"What technologies does FastAPI use?"
```

Extract the relevant entity:

```text
FastAPI
```

Then traverse:

```text
FastAPI
 ├── USES → Pydantic
 ├── USES → Starlette
 └── WRITTEN_IN → Python
```

---

# 48. Graph Retrieval Strategies

Start with:

```text
entity
 ↓
neighbors
```

Then support:

```text
depth=1
depth=2
```

Then relationship filtering:

```text
USES
DEPENDS_ON
BUILDS
```

Then confidence filtering:

```text
confidence >= threshold
```

---

# 49. Vector Retrieval

Implement:

```python
VectorRetriever.retrieve(query, top_k)
```

Pipeline:

```text
query
 ↓
embedding
 ↓
vector search
 ↓
top-k chunks
```

---

# 50. Phase 13 — Hybrid Retrieval

Now combine both systems.

```text
Question
   |
   +------------------+
   |                  |
   v                  v
Graph Search      Vector Search
   |                  |
   v                  v
Graph Facts       Source Chunks
   |                  |
   +---------+--------+
             |
             v
       Ranking Engine
             |
             v
       Context Builder
```

---

# 51. Retrieval Ranking

Each result should have:

```text
graph_score
vector_score
confidence
source_quality
final_score
```

Build a configurable scoring function.

For example:

```text
final_score =
    graph_weight * graph_score
    +
    vector_weight * vector_score
    +
    confidence_weight * confidence
```

Do not assume the first weights are optimal.

Evaluate them.

---

# 52. Phase 14 — Context Builder

Create:

```python
ContextBuilder
```

It receives:

```text
graph results
vector results
evidence
```

and produces a structured context.

Example:

```text
QUESTION:

What does FastAPI use for validation?


GRAPH FACTS:

FastAPI --USES--> Pydantic


SOURCE:

"FastAPI uses Pydantic for data validation."


SOURCE:

"Pydantic provides validation models."


PROVENANCE:

chunk_004
chunk_011
```

---

# 53. Phase 15 — RAG Generator

Create:

```python
RAGGenerator
```

Input:

```text
question
context
```

Output:

```text
answer
citations
```

The LLM should be instructed:

```text
Only answer using supplied context.

Do not invent facts.

Do not invent relationships.

If evidence is insufficient, say so.

Return source references where possible.
```

---

# 54. RAG Milestone

The system should support:

```text
Question:

What does FastAPI use for data validation?
```

Answer:

```text
FastAPI uses Pydantic for data validation.

Evidence:
chunk_004
```

And the system should be able to retrieve the original chunk.

---

# 55. Phase 16 — Full Ingestion Pipeline

Now combine everything.

Create:

```python
DocumentProcessor
```

Pipeline:

```text
Document
   |
   v
Normalize
   |
   v
Structure Detection
   |
   v
Intelligent Chunking
   |
   +----------------------+
   |                      |
   v                      v
Extraction             Embedding
   |                      |
   v                      v
Entity Resolution     Vector Store
   |
   v
Graph Builder
   |
   v
Graph Store
```

---

# 56. Processing State

Documents should have:

```text
CREATED
PROCESSING
CHUNKING
EXTRACTING
RESOLVING
BUILDING_GRAPH
EMBEDDING
COMPLETED
FAILED
```

The processor should update the state.

---

# 57. Phase 17 — API

Now expose the engine through FastAPI.

Create:

```text
api/
├── documents.py
├── entities.py
├── graph.py
├── search.py
└── rag.py
```

Endpoints:

```text
POST /documents

GET /documents

GET /documents/{id}

POST /documents/{id}/process

GET /documents/{id}/chunks

GET /entities/{id}

GET /entities/{id}/neighbors

GET /entities/{id}/subgraph

GET /relationships/{id}

GET /relationships/{id}/evidence

POST /search/vector

POST /search/graph

POST /search/hybrid

POST /rag/query
```

---

# 58. Phase 18 — Graph API

The frontend should request small subgraphs.

For example:

```text
GET /entities/entity_001/subgraph?depth=2
```

Response:

```json
{
  "nodes": [],
  "edges": []
}
```

Never send the entire graph unnecessarily.

---

# 59. Phase 19 — Frontend

Only now create:

```text
frontend/
```

Use:

```text
React
TypeScript
```

The frontend should not contain graph logic.

It should call the API.

---

# 60. Frontend Pages

Build:

```text
Documents
Graph Explorer
Entity Details
Search
RAG Chat
```

---

# 61. Graph Explorer

Display:

```text
       Pydantic
           |
          USES
           |
           v
        FastAPI
           |
          USES
           |
           v
       Starlette
```

Interactions:

```text
click node
expand node
collapse node
search entity
filter relationship
filter entity type
```

---

# 62. Evidence Panel

When clicking:

```text
FastAPI --USES--> Pydantic
```

show:

```text
Relationship:
USES

Source:
FastAPI

Target:
Pydantic

Evidence:
"FastAPI uses Pydantic for data validation."

Document:
FastAPI Introduction

Chunk:
chunk_004

Confidence:
0.94
```

---

# 63. RAG Chat

Create:

```text
Question input
Answer
Sources
Graph facts
```

The user should be able to click a source and jump to the corresponding chunk.

---

# 64. Phase 20 — Evaluation

Create:

```text
evaluation/
├── dataset/
├── extraction.py
├── entity_resolution.py
├── retrieval.py
└── rag.py
```

---

# 65. Extraction Evaluation

Measure:

```text
Entity Precision
Entity Recall
Entity F1

Relationship Precision
Relationship Recall
Relationship F1
```

Use a manually verified dataset.

---

# 66. Entity Resolution Evaluation

Create examples:

```text
FastAPI
fastapi
Fast API

GPT-5
GPT5
GPT 5

Apple
Apple Inc.
```

Measure:

```text
correct merges
incorrect merges
missed merges
```

---

# 67. Retrieval Evaluation

For each question define the correct chunks.

Measure:

```text
Recall@K
Precision@K
MRR
```

Do this separately for:

```text
Vector retrieval
Graph retrieval
Hybrid retrieval
```

This will show whether hybrid retrieval actually improves the system.

---

# 68. RAG Evaluation

Measure:

```text
Answer correctness
Evidence correctness
Faithfulness
Citation correctness
```

Create questions where:

```text
answer exists
answer does not exist
multiple entities are involved
multi-hop reasoning is required
```

---

# 69. Phase 21 — Performance

Only after correctness.

Measure:

```text
document processing time
chunking time
extraction time
entity resolution time
graph insertion time
embedding time
vector search time
graph search time
RAG latency
```

Do not optimize based on guesses.

Measure first.

---

# 70. Phase 22 — Advanced Vector Index

Only after brute-force search works.

Implement an ANN index yourself.

Possible progression:

```text
V1
Brute force

V2
KD-tree

V3
HNSW

V4
Index persistence
```

HNSW is particularly interesting because it teaches how modern vector databases perform approximate nearest-neighbor search.

---

# 71. Phase 23 — Advanced Graph Algorithms

Extend the graph engine with:

```text
DFS
BFS
shortest path
weighted path
relationship filtering
confidence-aware traversal
centrality
connected components
community detection
```

These are optional advanced milestones.

---

# 72. Phase 24 — Graph Query Language

Build a small query language.

Example:

```text
GET FastAPI
```

```text
GET NEIGHBORS FastAPI
```

```text
GET NEIGHBORS FastAPI DEPTH 2
```

```text
FIND PATH FastAPI Pydantic
```

Eventually:

```text
FIND FastAPI -[USES]-> *
```

This is optional but would make the project substantially more interesting.

---

# 73. Phase 25 — Contradiction Detection

Later, support:

```text
Document A:
FastAPI uses Pydantic.

Document B:
FastAPI does not use Pydantic.
```

The graph should not silently overwrite information.

Instead:

```text
FastAPI
   |
   +-- USES --> Pydantic
   |
   +-- NOT_USES --> Pydantic
```

with separate evidence.

This introduces:

```text
conflicting facts
source provenance
source confidence
temporal information
```

---

# 74. Phase 26 — Temporal Knowledge

Eventually support:

```text
FastAPI version 0.x
FastAPI version 1.x
```

Relationships can have:

```text
valid_from
valid_until
version
timestamp
```

Then queries could become:

```text
What did Project X depend on in 2025?
```

This is an advanced feature.

---

# 75. Final Repository

Eventually:

```text
knowledge-graph-rag/

├── backend/
│   └── app/
│       ├── api/
│       ├── models/
│       ├── ingestion/
│       ├── chunking/
│       ├── extraction/
│       ├── entity_resolution/
│       ├── graph/
│       ├── vector/
│       ├── retrieval/
│       ├── rag/
│       ├── storage/
│       ├── evaluation/
│       ├── config.py
│       └── main.py
│
├── frontend/
│   └── src/
│       ├── graph/
│       ├── documents/
│       ├── search/
│       ├── chat/
│       ├── entities/
│       └── api/
│
├── tests/
│
├── evaluation/
│
├── data/
│
├── scripts/
│
├── docs/
│
├── README.md
└── PLAN.md
```

---

# 76. Exact Build Order

The actual implementation order is:

```text
01. Project skeleton

02. Data models

03. In-memory graph

04. Graph operations

05. BFS traversal

06. Path finding

07. Graph persistence

08. Text normalization

09. Document structure detection

10. Paragraph segmentation

11. Embedding-based semantic similarity

12. Intelligent chunking V1

13. Chunk validation

14. Intelligent chunking V2
    + entity continuity

15. Intelligent chunking V3
    + relationship continuity

16. LLM extraction

17. Extraction schema validation

18. Entity normalization

19. String similarity

20. Entity resolution

21. Graph construction

22. Relationship deduplication

23. Evidence/provenance

24. Embedding generation

25. Vector storage

26. Cosine similarity

27. Top-K vector search

28. Graph retrieval

29. Vector retrieval

30. Retrieval ranking

31. Hybrid retrieval

32. Context builder

33. RAG generation

34. Full document processor

35. FastAPI API

36. API tests

37. React frontend

38. Graph visualization

39. Evidence UI

40. RAG chat UI

41. Evaluation framework

42. Retrieval evaluation

43. RAG evaluation

44. Performance profiling

45. ANN index

46. Advanced graph algorithms

47. Graph query language

48. Contradiction handling

49. Temporal knowledge
```

---

# 77. The Three Major Milestones

Do not think about the entire project at once.

There are three major checkpoints.

## Milestone 1 — Knowledge Graph

```text
Plain Text
   ↓
Intelligent Chunking
   ↓
Extraction
   ↓
Entity Resolution
   ↓
Graph
```

At this point you have built the knowledge graph engine.

---

## Milestone 2 — Retrieval Engine

```text
Graph
   +
Vector Store
   |
   v
Hybrid Retrieval
```

At this point you have built your retrieval engine.

---

## Milestone 3 — Graph-RAG

```text
Question
   ↓
Graph + Vector Retrieval
   ↓
Context
   ↓
LLM
   ↓
Answer + Evidence
```

At this point you have the complete core system.

The frontend comes after this.

---

# 78. What You Should Personally Implement

The following are the parts where you should write the actual algorithms yourself:

```text
✓ Semantic chunking

✓ Chunk boundary detection

✓ Entity normalization

✓ Entity resolution

✓ Graph representation

✓ Graph traversal

✓ Path finding

✓ Relationship deduplication

✓ Evidence tracking

✓ Vector storage

✓ Cosine similarity

✓ Top-K search

✓ Graph retrieval

✓ Vector retrieval

✓ Hybrid ranking

✓ Context construction

✓ Evaluation
```

These are the parts where you should understand the implementation line by line.

---

# 79. What You Don't Need to Reinvent

Do not waste months rebuilding unrelated infrastructure.

You don't need to write:

```text
✓ HTTP server from TCP
✓ JSON parser
✓ SQLite
✓ React
✓ Transformer architecture
✓ GPU kernels
✓ CUDA
```

The purpose is to build the knowledge/retrieval system, not recreate the entire software ecosystem.

---

# 80. Recommended First Coding Session

Do NOT start with FastAPI.

Do NOT start with React.

Do NOT start with the LLM.

Start with:

```text
backend/app/models/
backend/app/graph/
```

Implement:

```python
Entity
Relationship
Graph
```

Then manually create:

```text
FastAPI
Pydantic
Python
```

and:

```text
FastAPI --USES--> Pydantic
FastAPI --WRITTEN_IN--> Python
```

Then implement:

```python
graph.get_neighbors("fastapi")
```

and:

```python
graph.find_path("fastapi", "python")
```

Once that works, move to chunking.

---

# 81. First End-to-End Target

The first meaningful version of the system should eventually allow:

```bash
python process.py input.txt
```

and produce:

```text
Document processed.

Documents:       1
Chunks:          17
Entities:        42
Relationships:   67
Evidence:        67
Vectors:         17
```

Then:

```bash
python query.py "What does FastAPI use for validation?"
```

produces:

```text
Answer:

FastAPI uses Pydantic for data validation.

Graph facts:

FastAPI --USES--> Pydantic

Evidence:

"FastAPI uses Pydantic for data validation."

Source:

document_001
chunk_007
```

And eventually:

```bash
python server.py
```

opens the complete web application where the user can visually explore the graph and ask RAG questions.

---

# 82. Definition of Done

The project is considered complete when:

```text
[ ] Plain text can be ingested

[ ] Documents are stored

[ ] Intelligent chunks are generated

[ ] Chunk boundaries preserve semantic context

[ ] Entities are extracted

[ ] Relationships are extracted

[ ] Extraction is schema validated

[ ] Entities are normalized

[ ] Duplicate entities are resolved

[ ] Relationships are deduplicated

[ ] Every relationship has evidence

[ ] Graph can be traversed

[ ] Paths can be discovered

[ ] Graph is persisted

[ ] Embeddings are generated

[ ] Vectors are stored

[ ] Cosine similarity works

[ ] Top-K vector search works

[ ] Graph retrieval works

[ ] Vector retrieval works

[ ] Hybrid retrieval works

[ ] Context is constructed

[ ] RAG answers questions

[ ] Answers contain evidence

[ ] Evidence links to chunks

[ ] Chunks link to documents

[ ] API exposes the system

[ ] Frontend visualizes graph

[ ] Users can expand graph nodes

[ ] Users can inspect relationships

[ ] Users can inspect evidence

[ ] Users can ask RAG questions

[ ] Evaluation dataset exists

[ ] Retrieval metrics exist

[ ] RAG evaluation exists
```

---

# 83. The Core Learning Path

The most important learning progression is:

```text
Data Structures
      ↓
Graphs
      ↓
Graph Algorithms
      ↓
Text Processing
      ↓
Semantic Similarity
      ↓
Information Extraction
      ↓
Entity Resolution
      ↓
Knowledge Representation
      ↓
Vector Search
      ↓
Information Retrieval
      ↓
Hybrid Retrieval
      ↓
RAG
      ↓
Distributed/Production Systems
```

This is why we are deliberately not starting with Neo4j + Pinecone + LangChain.

By the end, you should understand not only how to use those systems, but what they are actually doing underneath.

---

# 84. The Most Important Rule

Do not build the project as:

```text
"How do I make a Graph-RAG application?"
```

Build it as:

```text
"How do I implement every important subsystem
that makes Graph-RAG possible?"
```

That mindset changes the entire project.

The final application is only the demonstration.

The real project is the engine underneath it.