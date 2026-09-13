# KnowledgeGraph-RAG

A from-scratch knowledge extraction, graph construction, graph retrieval, vector retrieval, and hybrid RAG system built from unstructured plain text.

The goal of this project is not to assemble existing RAG/Graph-RAG frameworks.

The goal is to understand and implement the core mechanisms ourselves.

---

# 1. Project Goal

Build a system that accepts unstructured plain text and converts it into an interconnected, queryable knowledge graph.

The system should then use that graph, together with semantic vector retrieval, to answer questions about the original corpus.

The complete system should look like:

```text
                    UNSTRUCTURED TEXT
                           |
                           v
                    +-------------+
                    |   Chunker   |
                    +-------------+
                           |
                           v
                 +-------------------+
                 | Semantic Extractor|
                 +-------------------+
                           |
              +------------+------------+
              |                         |
              v                         v
          Entities                 Relations
              |                         |
              +------------+------------+
                           |
                           v
                  Entity Resolution
                           |
                           v
                    Graph Builder
                           |
              +------------+------------+
              |                         |
              v                         v
        Graph Storage             Vector Storage
              |                         |
              +------------+------------+
                           |
                           v
                     Query System
                           |
              +------------+------------+
              |                         |
              v                         v
       Graph Retrieval           Vector Retrieval
              |                         |
              +------------+------------+
                           |
                           v
                   Context Builder
                           |
                           v
                         LLM
                           |
                           v
                        Answer
```

The frontend should visualize the graph and allow users to interact with the knowledge base.

---

# 2. What "From Scratch" Means

This project should not use existing solutions for the core functionality.

Do NOT make the core system dependent on:

- Neo4j
- Pinecone
- Weaviate
- Chroma
- FAISS
- LangChain
- LlamaIndex
- existing Graph-RAG frameworks
- existing vector databases
- graph databases

You can use:

- Python
- FastAPI
- PostgreSQL or SQLite as basic persistence
- an LLM API for language understanding
- an embedding model/API
- React/TypeScript for the frontend

The important distinction is:

```text
Infrastructure:
    allowed

Core algorithms:
    implemented by us
```

For example, SQLite is fine as a persistence mechanism.

But the graph data model and graph traversal engine should be ours.

Similarly, an embedding model is fine.

But vector storage and similarity search should be implemented by us.

---

# 3. Core Requirements

The final system should support:

1. Plain text ingestion
2. Text cleaning
3. Intelligent chunking
4. Semantic entity extraction
5. Semantic relationship extraction
6. Entity normalization
7. Entity resolution
8. Duplicate entity merging
9. Graph construction
10. Graph persistence
11. Graph traversal
12. Subgraph extraction
13. Provenance tracking
14. Vector embedding generation
15. Vector storage
16. Vector similarity search
17. Graph-based retrieval
18. Vector-based retrieval
19. Hybrid retrieval
20. Context construction
21. RAG answer generation
22. Answer provenance
23. Graph visualization
24. Interactive graph exploration
25. API access
26. Evaluation

---

# 4. Example

Input:

```text
FastAPI is a modern Python web framework.

FastAPI uses Pydantic for data validation and Starlette
for web functionality.

Sebastián Ramírez created FastAPI.

FastAPI is commonly used to build APIs and microservices.
```

The system should extract something similar to:

```text
Entities:

FastAPI
Type: Technology

Python
Type: ProgrammingLanguage

Pydantic
Type: Technology

Starlette
Type: Technology

Sebastián Ramírez
Type: Person

API
Type: Concept

Microservices
Type: Concept
```

Relationships:

```text
FastAPI --WRITTEN_IN--> Python

FastAPI --USES--> Pydantic

FastAPI --USES--> Starlette

Sebastián Ramírez --CREATED--> FastAPI

FastAPI --USED_FOR--> API

FastAPI --USED_FOR--> Microservices
```

Each relationship should retain evidence.

Example:

```text
FastAPI --USES--> Pydantic

Evidence:
"FastAPI uses Pydantic for data validation."

Source:
document_001

Chunk:
chunk_004

Confidence:
0.94
```

This is important.

The graph should not simply know that two entities are connected.

It should know:

```text
WHY are they connected?
WHERE did the relationship come from?
HOW confident are we?
```

---

# 5. System Architecture

The application should be divided into several independent components.

```text
project/
|
+-- ingestion/
|
+-- chunking/
|
+-- extraction/
|
+-- entity_resolution/
|
+-- graph/
|
+-- vector/
|
+-- retrieval/
|
+-- rag/
|
+-- api/
|
+-- evaluation/
|
+-- frontend/
|
+-- tests/
```

The core architecture should be:

```text
                  +----------------+
                  | Plain Text     |
                  +-------+--------+
                          |
                          v
                  +---------------+
                  | Ingestion     |
                  +-------+-------+
                          |
                          v
                  +---------------+
                  | Chunking      |
                  +-------+-------+
                          |
                          v
                  +---------------+
                  | Extraction    |
                  +-------+-------+
                          |
                          v
                  +---------------+
                  | Entity        |
                  | Resolution    |
                  +-------+-------+
                          |
             +------------+------------+
             |                         |
             v                         v
      +-------------+           +-------------+
      | Graph Store |           | Vector Store|
      +-------------+           +-------------+
             |                         |
             +------------+------------+
                          |
                          v
                  +---------------+
                  | Retrieval     |
                  +-------+-------+
                          |
                          v
                  +---------------+
                  | Context       |
                  | Builder       |
                  +-------+-------+
                          |
                          v
                  +---------------+
                  | LLM           |
                  +-------+-------+
                          |
                          v
                       Answer
```

---

# 6. Data Model

Everything should have stable IDs.

Do not use entity names as primary identifiers.

Use IDs such as:

```text
entity_01
entity_02
chunk_001
document_001
relation_001
```

---

# 7. Document

A document represents an input text source.

```python
Document
```

Fields:

```text
id
title
content
created_at
updated_at
metadata
```

Example:

```json
{
  "id": "doc_001",
  "title": "FastAPI Introduction",
  "content": "...",
  "metadata": {}
}
```

---

# 8. Chunk

Large documents must be divided into chunks.

```python
Chunk
```

Fields:

```text
id
document_id
text
start_offset
end_offset
chunk_index
metadata
```

Example:

```json
{
  "id": "chunk_001",
  "document_id": "doc_001",
  "text": "FastAPI uses Pydantic...",
  "chunk_index": 4
}
```

The chunk is extremely important because it becomes the provenance unit.

---

# 9. Entity

An entity represents a real concept/object/person/technology/etc.

```python
Entity
```

Fields:

```text
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
    "Fast API",
    "fastapi"
  ]
}
```

---

# 10. Relationship

A relationship connects two entities.

```python
Relationship
```

Fields:

```text
id
source_entity_id
target_entity_id
relationship_type
confidence
evidence
source_chunk_id
metadata
```

Example:

```json
{
  "id": "rel_001",
  "source_entity_id": "entity_001",
  "target_entity_id": "entity_002",
  "relationship_type": "USES",
  "confidence": 0.94,
  "evidence": "FastAPI uses Pydantic for data validation.",
  "source_chunk_id": "chunk_004"
}
```

---

# 11. Graph Representation

Initially, implement the graph using adjacency structures.

For example:

```python
class Graph:
    nodes = {}
    edges = {}
```

An adjacency list could look like:

```python
adjacency = {
    "entity_001": [
        "relation_001",
        "relation_004"
    ]
}
```

Or:

```python
adjacency = {
    "fastapi": {
        "uses": ["pydantic", "starlette"],
        "written_in": ["python"]
    }
}
```

However, use proper entity and relationship objects internally rather than making the name itself the data model.

---

# 12. Graph Operations

Implement these yourself.

Basic operations:

```python
add_entity(entity)

get_entity(entity_id)

remove_entity(entity_id)

add_relationship(relation)

get_relationship(relation_id)

remove_relationship(relation_id)
```

Traversal:

```python
get_neighbors(entity_id)

get_outgoing_edges(entity_id)

get_incoming_edges(entity_id)

get_neighbors_by_relationship(entity_id, relation_type)
```

Subgraph:

```python
get_subgraph(entity_id, depth=1)

get_subgraph(entity_id, depth=2)
```

Path finding:

```python
find_path(source_id, target_id)
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

The engine should be able to discover:

```text
FastAPI -> Pydantic -> Validation
```

---

# 13. Graph Traversal

Implement BFS first.

Do not immediately implement complicated graph algorithms.

For example:

```text
start = FastAPI

depth 1:

FastAPI
 |
 +-- Pydantic
 +-- Starlette
 +-- Python

depth 2:

FastAPI
 |
 +-- Pydantic
 |     |
 |     +-- Validation
 |
 +-- Starlette
 |
 +-- Python
       |
       +-- Programming Language
```

BFS is enough for the initial Graph-RAG system.

Later implement:

- DFS
- shortest path
- weighted traversal
- relationship filtering
- confidence filtering
- path ranking

---

# 14. Semantic Extraction

This is one of the most important components.

Input:

```text
FastAPI uses Pydantic for data validation.
```

Output:

```json
{
  "entities": [
    {
      "name": "FastAPI",
      "type": "Technology"
    },
    {
      "name": "Pydantic",
      "type": "Technology"
    },
    {
      "name": "Data Validation",
      "type": "Concept"
    }
  ],
  "relationships": [
    {
      "source": "FastAPI",
      "target": "Pydantic",
      "type": "USES"
    },
    {
      "source": "Pydantic",
      "target": "Data Validation",
      "type": "PROVIDES"
    }
  ]
}
```

The LLM should NOT be allowed to arbitrarily invent an unlimited schema.

Define the schema ourselves.

For example:

```text
Entity Types:

Person
Company
Product
Technology
ProgrammingLanguage
Framework
Database
Concept
Project
Organization
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

This makes the graph consistent.

---

# 15. Structured Extraction

Create a strict Pydantic schema.

For example:

```python
class ExtractedEntity(BaseModel):
    name: str
    entity_type: str


class ExtractedRelationship(BaseModel):
    source: str
    target: str
    relationship_type: str
    evidence: str
    confidence: float


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]
```

The LLM output must be validated against this schema.

Invalid output should not enter the graph.

---

# 16. Entity Resolution

This is another major component.

Suppose the corpus contains:

```text
GPT-5
GPT 5
GPT5
OpenAI GPT-5
```

These may all refer to the same entity.

The system needs to determine:

```text
GPT-5
   ^
   |
+--+--+--+
|  |  |  |
GPT5
GPT 5
OpenAI GPT-5
```

and produce:

```text
Canonical Entity:

GPT-5
```

with aliases:

```text
GPT5
GPT 5
OpenAI GPT-5
```

---

# 17. Entity Resolution Strategy

Do not start with an LLM.

Implement deterministic matching first.

Stage 1:

Normalize text.

```text
"GPT-5"
"GPT 5"
"gpt5"
```

Normalize into comparable forms.

For example:

```text
lowercase
remove punctuation
normalize whitespace
```

Stage 2:

Exact normalized match.

Stage 3:

Alias match.

Stage 4:

String similarity.

Implement similarity yourself.

Possible algorithms:

```text
Levenshtein distance
Jaccard similarity
token similarity
```

Stage 5:

Semantic similarity.

Only after these should you consider asking an LLM to resolve ambiguous cases.

---

# 18. Graph Construction

After extraction:

```text
ExtractionResult
        |
        v
Entity Resolution
        |
        v
Canonical Entities
        |
        v
Relationship Validation
        |
        v
Graph Builder
```

The graph builder should:

1. Create missing entities
2. Resolve existing entities
3. Create relationships
4. Attach provenance
5. Merge duplicate relationships
6. Update confidence
7. Preserve evidence

---

# 19. Relationship Deduplication

Suppose multiple chunks say:

```text
FastAPI uses Pydantic.
```

You should not create:

```text
FastAPI --USES--> Pydantic

FastAPI --USES--> Pydantic

FastAPI --USES--> Pydantic

FastAPI --USES--> Pydantic
```

Instead maintain one logical relationship:

```text
FastAPI --USES--> Pydantic
```

with multiple pieces of evidence:

```text
Evidence 1 -> chunk_001
Evidence 2 -> chunk_014
Evidence 3 -> chunk_038
```

This is important for provenance.

---

# 20. Evidence Model

A relationship should eventually have:

```python
Evidence
```

Fields:

```text
id
relationship_id
chunk_id
text
confidence
```

Therefore:

```text
Entity A
   |
   | USES
   |
Entity B
   |
   +---- Evidence
          |
          +-- chunk_001
          +-- chunk_019
          +-- chunk_043
```

This enables explainable Graph-RAG.

---

# 21. Persistence

Do not build a database engine.

Use a simple persistence layer.

For MVP:

```text
SQLite
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

The graph engine sits above the database.

Architecture:

```text
Graph API
    |
Graph Engine
    |
Graph Repository
    |
SQLite
```

The graph engine should NOT know SQL details.

For example:

```python
class GraphRepository:
    def save_entity(...)
    def get_entity(...)
    def save_relationship(...)
    def get_neighbors(...)
```

Then:

```text
Graph Engine
      |
GraphRepository
      |
SQLiteRepository
```

This keeps the architecture clean.

---

# 22. Vector System

The vector system should also be implemented ourselves.

Pipeline:

```text
Chunk
  |
  v
Embedding Model
  |
  v
Vector
  |
  v
Your Vector Store
```

A vector record:

```python
class VectorRecord:
    id: str
    chunk_id: str
    vector: list[float]
```

Store:

```text
chunk_id
embedding
metadata
```

---

# 23. Similarity Search

Implement cosine similarity.

Given:

```text
query_vector
document_vector
```

calculate:

```text
cosine_similarity =
    dot(query, document)
    /
    (norm(query) * norm(document))
```

Then:

```text
query
 |
 v
embedding
 |
 v
compare against stored vectors
 |
 v
sort by similarity
 |
 v
top K
```

Initially a brute-force scan is completely acceptable.

For example:

```text
10,000 vectors

query
 |
 +-- compare with vector 1
 +-- compare with vector 2
 +-- compare with vector 3
 ...
 +-- compare with vector 10,000
 |
sort
 |
top 10
```

Do NOT prematurely optimize.

Later you can implement:

```text
KD-tree
HNSW
IVF
product quantization
```

as learning extensions.

---

# 24. Why We Need Both Graph and Vector Retrieval

These solve different problems.

Vector search answers:

```text
"What text is semantically similar to my question?"
```

Graph search answers:

```text
"What entities are connected?"

"How are A and B related?"

"What technologies does X depend on?"

"What projects are connected to Y?"
```

Example question:

```text
"What technologies does FastAPI use?"
```

Graph retrieval can directly traverse:

```text
FastAPI
 |
 +-- USES --> Pydantic
 |
 +-- USES --> Starlette
```

Vector search may retrieve chunks discussing FastAPI.

Both are useful.

---

# 25. Graph Retrieval

Create:

```python
GraphRetriever
```

Input:

```text
query
```

First determine which entities are relevant.

For example:

```text
"What technologies does FastAPI use?"
```

Extract:

```text
FastAPI
```

Then retrieve:

```text
FastAPI
   |
   +-- USES --> Pydantic
   +-- USES --> Starlette
```

The graph retriever should support:

```python
retrieve_neighbors()

retrieve_relationships()

retrieve_subgraph()

find_path()

retrieve_by_entity_type()

retrieve_by_relationship_type()
```

---

# 26. Vector Retrieval

Create:

```python
VectorRetriever
```

Input:

```text
"What does FastAPI use for data validation?"
```

Generate query embedding.

Search vectors.

Return:

```text
chunk_004
chunk_019
chunk_021
```

Each result should contain:

```text
chunk_id
text
score
document_id
metadata
```

---

# 27. Hybrid Retrieval

Create:

```python
HybridRetriever
```

It should combine:

```text
Graph Retrieval
+
Vector Retrieval
```

Example:

```text
Question
   |
   +------------------+
   |                  |
   v                  v
Graph Search      Vector Search
   |                  |
   v                  v
Entities          Relevant chunks
Relations
   |                  |
   +--------+---------+
            |
            v
      Context Builder
```

---

# 28. Retrieval Ranking

The system should eventually rank retrieved information.

For example:

```text
final_score =
    graph_score * 0.5
    +
    vector_score * 0.5
```

Do not hardcode this forever.

Create a configurable ranking system.

For example:

```python
class RetrievalScore:
    graph_score: float
    vector_score: float
    confidence: float
    final_score: float
```

Later experiment with different weighting strategies.

---

# 29. Context Builder

The context builder converts retrieval results into information the LLM can consume.

Example:

```text
QUESTION:

What technologies does FastAPI use?

GRAPH CONTEXT:

FastAPI --USES--> Pydantic
FastAPI --USES--> Starlette

SOURCE CONTEXT:

"FastAPI uses Pydantic for data validation and Starlette
for web functionality."

PROVENANCE:

chunk_004
```

Then send this context to the LLM.

---

# 30. RAG Answer Generation

The LLM receives:

```text
Question
+
Graph context
+
Source chunks
+
Evidence
```

It should answer only from retrieved information.

The answer should ideally include citations.

Example:

```text
FastAPI uses Pydantic for data validation and Starlette
for web functionality.

Sources:
- chunk_004
```

---

# 31. Hallucination Protection

The RAG layer should instruct the model:

```text
Answer only using the supplied context.

If the context does not contain enough information,
say that the information is unavailable.

Do not invent relationships.
Do not invent entities.
Do not infer unsupported facts.
```

The system should distinguish:

```text
FACT FROM CORPUS
```

from:

```text
MODEL KNOWLEDGE
```

---

# 32. Graph Query Language

Eventually implement a simple query interface.

For example:

```text
GET ENTITY FastAPI

GET NEIGHBORS FastAPI

GET NEIGHBORS FastAPI DEPTH 2

GET RELATIONSHIPS FastAPI

FIND PATH FastAPI Pydantic
```

Later you can design your own mini graph query language.

Example:

```text
FIND FastAPI -[USES]-> *
```

or:

```text
FIND FastAPI -[*]-> Pydantic
```

This is optional for the MVP but would be a valuable extension.

---

# 33. API Layer

Use FastAPI.

The backend should expose:

```text
POST /documents
GET  /documents
GET  /documents/{id}

POST /documents/{id}/process

GET /documents/{id}/chunks

GET /entities
GET /entities/{id}

GET /entities/{id}/neighbors

GET /entities/{id}/subgraph

GET /relationships/{id}

GET /relationships/{id}/evidence

POST /search/vector

POST /search/graph

POST /search/hybrid

POST /rag/query

GET /graph
```

---

# 34. Document Processing API

Example:

```http
POST /documents
```

Request:

```json
{
  "title": "FastAPI Notes",
  "text": "FastAPI is..."
}
```

Response:

```json
{
  "document_id": "doc_001",
  "status": "created"
}
```

Then:

```http
POST /documents/doc_001/process
```

Processing:

```text
Document
   |
Chunk
   |
Extract
   |
Resolve
   |
Build Graph
   |
Generate Embeddings
```

---

# 35. Graph API

Example:

```http
GET /entities/entity_001/subgraph?depth=2
```

Response:

```json
{
  "nodes": [
    {
      "id": "entity_001",
      "name": "FastAPI",
      "type": "Technology"
    }
  ],
  "edges": [
    {
      "source": "entity_001",
      "target": "entity_002",
      "type": "USES"
    }
  ]
}
```

The frontend can directly visualize this response.

---

# 36. Frontend

Build a real graph explorer.

Recommended:

```text
React
TypeScript
```

The graph itself should be rendered using a visualization library only.

The graph logic must remain in the backend.

The frontend should provide:

```text
Document upload
|
Graph explorer
|
Entity search
|
Entity details
|
Relationship details
|
Evidence viewer
|
RAG chat
```

---

# 37. Graph UI

The graph should look like:

```text
                   Python
                     |
                  WRITTEN_IN
                     |
                     v
                  FastAPI
                 /       \
              USES       USES
               /           \
              v             v
          Pydantic       Starlette
              |
           PROVIDES
              |
              v
        Data Validation
```

Nodes should display:

```text
name
type
```

Edges should display:

```text
relationship
```

Clicking an edge should display:

```text
Relationship:
FastAPI USES Pydantic

Evidence:
"FastAPI uses Pydantic for data validation."

Source:
FastAPI Notes

Chunk:
chunk_004

Confidence:
0.94
```

---

# 38. Interactive Graph

The user should be able to:

```text
click node
```

and see:

```text
Entity Information
```

They should be able to:

```text
expand node
```

which retrieves its neighbors.

This avoids sending the entire graph to the browser.

For example:

```text
GET /entities/fastapi/neighbors
```

Then dynamically add nodes.

---

# 39. Search UI

Provide:

```text
[ Search knowledge base... ]
```

Search should return:

```text
Entities
Relationships
Documents
Chunks
```

Example:

```text
Search: Pydantic

Entities:
Pydantic

Relationships:
FastAPI -> USES -> Pydantic

Documents:
FastAPI Architecture

Chunks:
"FastAPI uses Pydantic..."
```

---

# 40. RAG Chat UI

Example:

```text
+------------------------------------------+
| Knowledge Graph                          |
|                                          |
|      FastAPI                             |
|       /    \                             |
|    USES   USES                           |
|     /        \                           |
| Pydantic   Starlette                     |
|                                          |
+------------------------------------------+

+------------------------------------------+
| Ask a question                           |
|                                          |
| What does FastAPI use for validation?    |
|                                          |
| [ Send ]                                 |
+------------------------------------------+

Answer:

FastAPI uses Pydantic for data validation.

Sources:
chunk_004
```

Clicking `chunk_004` should show the original text.

---

# 41. Provenance

Provenance should be a first-class concept.

Every generated fact should be traceable:

```text
Answer
  |
  v
Relationship
  |
  v
Evidence
  |
  v
Chunk
  |
  v
Document
```

This is one of the most important differences between a toy knowledge graph and a useful RAG system.

---

# 42. Processing Pipeline

The document processing pipeline should eventually look like:

```text
Document
   |
   v
Normalize
   |
   v
Chunk
   |
   +--------------------+
   |                    |
   v                    v
Extract entities    Generate embeddings
   |
Extract relations
   |
Validate
   |
Entity resolution
   |
Graph construction
   |
Persist
```

Embedding generation can happen independently from graph extraction.

---

# 43. Processing State

Documents should have states.

```text
CREATED
PROCESSING
EXTRACTING
RESOLVING
BUILDING_GRAPH
EMBEDDING
COMPLETED
FAILED
```

This allows the frontend to show processing progress.

---

# 44. Error Handling

The pipeline must handle:

```text
LLM timeout
invalid LLM output
embedding failure
database failure
duplicate document
empty document
invalid relationship
unknown entity type
unknown relationship type
```

Do not allow a failed extraction to corrupt the graph.

---

# 45. Configuration

Use environment variables.

Example:

```text
LLM_PROVIDER
LLM_MODEL
LLM_API_KEY

EMBEDDING_PROVIDER
EMBEDDING_MODEL

DATABASE_PATH

CHUNK_SIZE
CHUNK_OVERLAP

MAX_GRAPH_DEPTH
TOP_K
```

Do not hardcode secrets.

---

# 46. Testing

Tests are a major part of this project.

Test:

```text
chunking
entity normalization
entity resolution
graph insertion
graph traversal
path finding
vector similarity
vector retrieval
hybrid retrieval
provenance
API endpoints
```

Example graph test:

```text
A -> B
B -> C
C -> D
```

Ask:

```text
neighbors(A)
```

Expected:

```text
B
```

Ask:

```text
subgraph(A, depth=2)
```

Expected:

```text
A
B
C
```

Ask:

```text
find_path(A, D)
```

Expected:

```text
A -> B -> C -> D
```

---

# 47. Evaluation Dataset

Create a small manually verified dataset.

Example:

```text
20 documents
100 entities
200 relationships
50 questions
```

For each question define the expected answer/evidence.

Example:

```text
Question:
What does FastAPI use for data validation?

Expected entity:
Pydantic

Expected relationship:
FastAPI --USES--> Pydantic

Expected source:
chunk_004
```

Measure:

```text
entity extraction precision
entity extraction recall

relationship precision
relationship recall

entity resolution accuracy

vector retrieval recall@K

graph retrieval accuracy

hybrid retrieval accuracy

answer faithfulness
```

---

# 48. Metrics

For extraction:

```text
Precision
Recall
F1
```

For retrieval:

```text
Recall@K
Precision@K
MRR
```

For Graph-RAG:

```text
Answer correctness
Evidence correctness
Faithfulness
```

Do not rely solely on LLM-as-judge.

Maintain deterministic evaluation wherever possible.

---

# 49. Logging

Every processing stage should produce logs.

Example:

```text
[INGESTION]
Document doc_001 received

[CHUNKING]
Created 14 chunks

[EXTRACTION]
chunk_001 -> 5 entities, 4 relationships

[RESOLUTION]
"Fast API" -> FastAPI

[GRAPH]
Created 5 entities
Created 4 relationships

[VECTOR]
Generated 14 embeddings

[COMPLETE]
Document doc_001 processed
```

---

# 50. Suggested Repository Structure

```text
knowledge-graph-rag/
|
+-- backend/
|   |
|   +-- app/
|       |
|       +-- api/
|       |   +-- documents.py
|       |   +-- entities.py
|       |   +-- graph.py
|       |   +-- search.py
|       |   +-- rag.py
|       |
|       +-- ingestion/
|       |   +-- loader.py
|       |
|       +-- chunking/
|       |   +-- chunker.py
|       |
|       +-- extraction/
|       |   +-- extractor.py
|       |   +-- schemas.py
|       |   +-- prompts.py
|       |
|       +-- entity_resolution/
|       |   +-- normalizer.py
|       |   +-- similarity.py
|       |   +-- resolver.py
|       |
|       +-- graph/
|       |   +-- models.py
|       |   +-- graph.py
|       |   +-- traversal.py
|       |   +-- repository.py
|       |
|       +-- vector/
|       |   +-- models.py
|       |   +-- store.py
|       |   +-- similarity.py
|       |   +-- search.py
|       |
|       +-- retrieval/
|       |   +-- graph_retriever.py
|       |   +-- vector_retriever.py
|       |   +-- hybrid_retriever.py
|       |   +-- ranking.py
|       |
|       +-- rag/
|       |   +-- context.py
|       |   +-- generator.py
|       |
|       +-- storage/
|       |   +-- database.py
|       |
|       +-- config.py
|       +-- main.py
|
+-- frontend/
|   |
|   +-- src/
|       +-- components/
|       +-- graph/
|       +-- search/
|       +-- chat/
|       +-- documents/
|       +-- api/
|
+-- tests/
|
+-- evaluation/
|
+-- scripts/
|
+-- data/
|
+-- docs/
|
+-- docker/
|
+-- README.md
```

---

# 51. Development Phases

Do not build everything simultaneously.

Build vertically.

## Phase 1 — Graph Core

Goal:

```text
Python objects
    |
    v
Graph
    |
    v
Traversal
```

Implement:

```text
Entity
Relationship
Graph
add_entity
add_relationship
neighbors
subgraph
BFS
path finding
```

No LLM.

No frontend.

No RAG.

---

# 52. Phase 2 — Persistence

Add:

```text
SQLite
```

Implement:

```text
GraphRepository
```

The graph should survive application restarts.

Test:

```text
create graph
restart application
load graph
traverse graph
```

---

# 53. Phase 3 — Text Processing

Implement:

```text
plain text
    |
cleaning
    |
chunking
```

Test chunking independently.

Do not involve the LLM yet.

---

# 54. Phase 4 — Semantic Extraction

Add the LLM.

Implement:

```text
chunk
 |
LLM
 |
structured JSON
 |
Pydantic validation
 |
ExtractionResult
```

At this stage, simply print the extraction result.

Do not insert it into the graph immediately.

First verify extraction quality.

---

# 55. Phase 5 — Entity Resolution

Implement:

```text
normalization
exact matching
alias matching
string similarity
```

Then connect extraction to the graph.

Pipeline:

```text
Text
 |
Chunk
 |
Extract
 |
Resolve
 |
Graph
```

At the end of this phase:

```text
plain text -> knowledge graph
```

should work.

This is the first major milestone.

---

# 56. Phase 6 — Provenance

Add:

```text
document
chunk
evidence
```

Every graph relationship must point back to the text that produced it.

Milestone:

```text
Graph relationship
        |
        v
Evidence
        |
        v
Original chunk
        |
        v
Original document
```

---

# 57. Phase 7 — Vector Engine

Implement:

```text
embedding generation
vector storage
cosine similarity
top-k search
```

Do not use a vector database.

Milestone:

```text
query
 |
embedding
 |
your vector store
 |
top-k chunks
```

---

# 58. Phase 8 — Retrieval

Implement:

```text
GraphRetriever
VectorRetriever
```

Test independently.

For example:

```text
Question:
What does FastAPI use?
```

Graph retrieval:

```text
FastAPI
 |
 +-- Pydantic
 +-- Starlette
```

Vector retrieval:

```text
chunk_004
chunk_009
chunk_013
```

---

# 59. Phase 9 — Hybrid RAG

Combine:

```text
Graph
+
Vector
+
Evidence
```

Build:

```text
HybridRetriever
ContextBuilder
RAGGenerator
```

Milestone:

```text
Question
   |
Graph + Vector Retrieval
   |
Context
   |
LLM
   |
Answer + Evidence
```

---

# 60. Phase 10 — FastAPI

Expose the system through APIs.

At this point the core engine already exists.

FastAPI becomes the interface to your engine.

---

# 61. Phase 11 — Frontend

Only now build the frontend.

The frontend should consume your APIs.

Build:

```text
Document page
Graph explorer
Entity page
Relationship/evidence panel
Search
RAG chat
```

The frontend is not responsible for knowledge extraction or graph logic.

---

# 62. Phase 12 — Optimization

Only after everything works.

Investigate:

```text
faster vector search
graph indexing
caching
batch extraction
parallel processing
incremental updates
memory usage
large corpus handling
```

This is where you can start implementing more advanced data structures.

---

# 63. Phase 13 — Production Architecture

After the single-process version works:

```text
API
 |
Queue
 |
Workers
 |
Graph Store
 |
Vector Store
```

Implement background processing.

For example:

```text
POST /documents
        |
        v
Processing Job
        |
        v
Worker
        |
        +-- Chunk
        +-- Extract
        +-- Resolve
        +-- Graph
        +-- Embeddings
```

Do not start here.

---

# 64. What Should Be Built Without Libraries

The following should be yours:

```text
Chunking algorithm
Entity normalization
Entity resolution
Graph data structure
Graph storage abstraction
Graph traversal
Path finding
Relationship management
Evidence management
Vector storage
Cosine similarity
Vector retrieval
Hybrid retrieval
Retrieval ranking
Context construction
RAG orchestration
Evaluation
```

---

# 65. What Can Use Existing Libraries

Use libraries where the goal is not to reinvent unrelated infrastructure.

Examples:

```text
FastAPI
Pydantic
SQLite
React
TypeScript
HTTP libraries
JSON libraries
logging libraries
```

For model inference:

```text
LLM API
Embedding model/API
```

The intelligence around those models should still be yours.

---

# 66. What We Are Explicitly NOT Building

Do not attempt to build:

```text
GPT-style transformer training system
GPU kernel
CUDA implementation
distributed database
full SQL database
production-grade ANN index immediately
browser rendering engine
HTTP server from TCP sockets
```

Those are different projects.

The objective is to understand and implement the knowledge graph + retrieval system.

---

# 67. Final Architecture

The final system should conceptually look like:

```text
                         USER
                           |
             +-------------+-------------+
             |                           |
             v                           v
        Graph Explorer               RAG Chat
             |                           |
             +-------------+-------------+
                           |
                         API
                           |
                    +------+------+
                    |             |
                    v             v
               Graph Engine   RAG Engine
                    |             |
                    |       +-----+-----+
                    |       |           |
                    v       v           v
                Graph     Vector     Context
                Store     Store      Builder
                    |       |           |
                    +-------+-----------+
                            |
                            v
                           LLM


                 INGESTION PIPELINE

Plain Text
    |
    v
Document
    |
    v
Chunker
    |
    +---------------------+
    |                     |
    v                     v
Extractor             Embedder
    |                     |
    v                     v
Entities              Vectors
Relations                 |
    |                     |
    v                     v
Entity Resolver      Vector Store
    |
    v
Graph Builder
    |
    v
Graph Store
```

---

# 68. MVP Definition

The MVP is NOT:

```text
beautiful frontend
+
Neo4j
+
Pinecone
+
LangChain
```

The MVP is:

```text
plain text
    |
    v
chunking
    |
    v
semantic extraction
    |
    v
entity resolution
    |
    v
your graph
    |
    v
your graph traversal
    |
    v
your vector store
    |
    v
hybrid retrieval
    |
    v
RAG answer
```

with provenance.

If this works reliably, the project is already successful.

---

# 69. First Milestone

The first thing to build is NOT the UI.

Create a Python program:

```text
input.txt
    |
    v
chunker
    |
    v
extractor
    |
    v
entity resolver
    |
    v
graph builder
    |
    v
graph.json
```

Then run:

```text
python main.py input.txt
```

and produce:

```text
graph.json
```

containing:

```json
{
  "entities": [],
  "relationships": []
}
```

Then write:

```text
python query.py graph.json FastAPI
```

and get:

```text
FastAPI
 |
 +-- USES --> Pydantic
 |
 +-- USES --> Starlette
 |
 +-- WRITTEN_IN --> Python
```

Once this works, you have the core of the project.

Everything else builds on top of this.

---

# 70. Long-Term Extensions

After the core system works, possible advanced features include:

```text
Temporal knowledge graphs

Entity versioning

Contradiction detection

Confidence propagation

Relationship weighting

Graph embeddings

Graph neural networks

Community detection

Centrality analysis

Multi-hop reasoning

Query planning

Graph-aware reranking

Hybrid retrieval optimization

Incremental graph updates

Knowledge graph snapshots

Fact verification

Contradictory evidence

Source reliability scoring

User-specific knowledge graphs

Multi-document entity resolution

Streaming ingestion

Distributed graph processing

ANN vector indexing

Custom graph query language
```

These are extensions, not MVP requirements.

---

# 71. The Main Engineering Principle

The project should be built in layers.

Never let one component do everything.

Bad:

```text
RAGService
    |
    +-- calls LLM
    +-- chunks text
    +-- creates graph
    +-- stores vectors
    +-- searches graph
    +-- generates answer
```

Good:

```text
DocumentProcessor
       |
       +-- Chunker
       +-- Extractor
       +-- EntityResolver
       +-- GraphBuilder
       +-- Embedder

GraphEngine
       |
       +-- GraphStore
       +-- Traversal

VectorEngine
       |
       +-- VectorStore
       +-- Similarity

RetrievalEngine
       |
       +-- GraphRetriever
       +-- VectorRetriever
       +-- HybridRetriever

RAGEngine
       |
       +-- ContextBuilder
       +-- AnswerGenerator
```

Each component should have one clear responsibility.

---

# 72. Final Success Criteria

The project is complete when a user can provide a corpus such as:

```text
100+ plain-text documents
```

and the system can:

```text
1. Ingest the documents

2. Split them into chunks

3. Extract entities

4. Extract relationships

5. Resolve duplicate entities

6. Build a connected graph

7. Preserve evidence for every relationship

8. Generate embeddings

9. Store vectors

10. Search vectors

11. Traverse the graph

12. Combine graph + vector retrieval

13. Generate an answer

14. Show supporting evidence

15. Visualize the relevant graph

16. Click through from:
        answer
          -> evidence
          -> chunk
          -> document
```

The final user experience should be:

```text
                "What does Project X depend on?"
                              |
                              v
                       Query Engine
                              |
                 +------------+------------+
                 |                         |
                 v                         v
             Graph Search             Vector Search
                 |                         |
                 +------------+------------+
                              |
                              v
                       Relevant Context
                              |
                              v
                             LLM
                              |
                              v
                    "Project X depends on
                     A, B and C because..."
                              |
                              v
                         Evidence
                              |
                              v
                         Graph View
```

That is the project.

The important part is that **you own the entire pipeline from text → knowledge → graph → retrieval → answer**.

The LLM is a component you call for language understanding, not the application itself.

The databases are storage mechanisms, not the intelligence of the system.

The graph visualization is a UI, not the graph engine.

And the RAG answer is the final consumer of all the engineering you built underneath it.