# Knowledge Graph Schema

**Category**: Long Term (4-6 months)
**Priority**: 🟢 LOW-MEDIUM
**Status**: Proposal
**Dependencies**: [001](./001_reasoning_as_a_service.md), [002](./002_structured_protocol_levels.md), [003](./003_proactive_memory.md) should be stable

---

## Vision

Upgrade from flat vector storage to **cognitive knowledge graph** that represents not just facts, but **relationships, dependencies, contradictions, and temporal evolution** of knowledge.

**Inspired by**: HippoRAG (biological memory architecture) + semantic web principles

**Core Concept**: Knowledge is a graph, not a list

**Value Proposition**:
- Detect contradictions automatically
- Track knowledge evolution over time
- Understand dependencies (X requires Y)
- Causal reasoning (A caused B)
- Analogical reasoning (X similar to Y)

---

## Current State Analysis

### What Exists Today

Current Zephyra storage (`/home/user/uni-memory/app/main.py`):

```sql
CREATE TABLE memories (
    id UUID,
    owner_key VARCHAR(255),
    content TEXT,
    embedding vector(384),
    timestamp TIMESTAMP,
    ...
);
```

**Limitations**:
- **Flat structure**: Memories are independent blobs
- **No relationships**: Can't express "A implies B" or "A contradicts C"
- **No temporal reasoning**: Can't track "I believed X, now I believe Y"
- **No dependency tracking**: Can't represent "To do A, first do B and C"

**Example of what's missing**:
```
Memory 1: "Python 2 is the standard" (2015)
Memory 2: "Python 3 is the standard" (2020)

Current system: Both stored, no connection
Desired: TEMPORAL edge showing belief evolution + CONTRADICTS edge
```

---

## Proposed Architecture

### Node Types

#### FACT
Atomic, verifiable truth

**Examples**:
- "Paris is the capital of France"
- "TCP uses 3-way handshake"
- "React hooks introduced in v16.8"

**Metadata**:
- `confidence_score`: 0.0-1.0 (how certain)
- `source`: user/inferred/external
- `verified_at`: timestamp of last verification

#### PROCEDURE
Step-by-step how-to knowledge

**Examples**:
- "To deploy: 1) Run tests 2) Build 3) Push"
- "Git workflow: feature branch → PR → review → merge"

**Metadata**:
- `steps`: List of ordered steps
- `success_rate`: How often this procedure works

#### PATTERN
Detected regularity or heuristic

**Examples**:
- "When test fails with error X, solution is usually Y"
- "Code reviews on Friday get fewer comments"
- "Bugs in auth module tend to be race conditions"

**Metadata**:
- `occurrence_count`: How many times seen
- `strength`: 0.0-1.0 (how reliable)

#### FAILURE
Failed attempt with context

**Examples**:
- "Tried iterative BST, got stack overflow"
- "Approach A failed because of memory leak"

**Metadata**:
- `approach`: What was tried
- `error_type`: Category of failure
- `context`: Conditions when failed

#### HYPOTHESIS
Unverified claim or prediction

**Examples**:
- "Refactoring this will improve performance"
- "This bug is probably caused by race condition"

**Metadata**:
- `confidence`: Likelihood of being true
- `verification_status`: pending/confirmed/rejected

---

### Edge Types

#### IMPLIES (Logical)
A → B (if A is true, then B is true)

**Examples**:
- FACT("Variable is null") → FACT("Will throw NullPointerException")
- PROCEDURE("Run tests") → FACT("Code quality maintained")

#### CONTRADICTS (Inconsistency)
A ⊥ B (A and B cannot both be true)

**Examples**:
- FACT("Python 2 is standard") ⊥ FACT("Python 3 is standard")
- HYPOTHESIS("Bug is in frontend") ⊥ HYPOTHESIS("Bug is in backend")

**Use**: Detect inconsistencies, prompt user to resolve

#### PREREQUISITE (Dependency)
B requires A (must have A before doing B)

**Examples**:
- PROCEDURE("Deploy to prod") requires FACT("Tests passed")
- PROCEDURE("Implement feature X") requires FACT("Auth system exists")

**Use**: Dependency resolution, planning

#### TEMPORAL (Time-ordered)
A before B (A happened/was known before B)

**Examples**:
- FACT("Python 2 standard") before FACT("Python 3 standard")
- FAILURE("Approach A") before PATTERN("Approach A usually fails")

**Use**: Knowledge evolution tracking

#### SIMILAR_TO (Analogical)
A ≈ B (A and B share properties)

**Examples**:
- PATTERN("Bug X") similar to PATTERN("Bug Y")
- PROCEDURE("Setup React") similar to PROCEDURE("Setup Vue")

**Use**: Transfer learning, analogical reasoning

#### CAUSED_BY (Causal)
B caused by A (A is the reason for B)

**Examples**:
- FAILURE("Memory leak") caused_by FACT("Forgot to close connection")
- PATTERN("Slow query") caused_by FACT("Missing index")

**Use**: Root cause analysis

---

### Graph Schema (PostgreSQL)

#### knowledge_nodes Table

```sql
CREATE TABLE knowledge_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_key VARCHAR(255) NOT NULL,

    -- Node classification
    node_type VARCHAR(50) NOT NULL,
    -- Values: 'FACT', 'PROCEDURE', 'PATTERN', 'FAILURE', 'HYPOTHESIS'

    -- Content
    content TEXT NOT NULL,
    embedding vector(384),

    -- Metadata
    confidence_score FLOAT DEFAULT 1.0,
    source VARCHAR(50) DEFAULT 'user',
    -- Values: 'user', 'inferred', 'external', 'system'

    access_frequency INTEGER DEFAULT 0,
    -- For forgetting curve (prune low-access nodes)

    last_verified TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- Type-specific metadata (JSONB for flexibility)
    metadata JSONB,
    -- Example for PROCEDURE: {"steps": [...], "success_rate": 0.85}
    -- Example for PATTERN: {"occurrence_count": 12, "strength": 0.9}
    -- Example for FAILURE: {"approach": "X", "error_type": "Y"}

    FOREIGN KEY (owner_key) REFERENCES api_keys(key)
);

-- Indexes
CREATE INDEX idx_knowledge_nodes_owner ON knowledge_nodes(owner_key);
CREATE INDEX idx_knowledge_nodes_type ON knowledge_nodes(owner_key, node_type);
CREATE INDEX idx_knowledge_nodes_embedding ON knowledge_nodes USING ivfflat (embedding vector_cosine_ops);
-- Note: Can upgrade to HNSW later for better performance

CREATE INDEX idx_knowledge_nodes_confidence ON knowledge_nodes(owner_key, confidence_score DESC);
CREATE INDEX idx_knowledge_nodes_access ON knowledge_nodes(owner_key, access_frequency DESC);
CREATE INDEX idx_knowledge_nodes_verified ON knowledge_nodes(owner_key, last_verified DESC);
```

#### knowledge_edges Table

```sql
CREATE TABLE knowledge_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_key VARCHAR(255) NOT NULL,

    -- Edge endpoints
    source_node_id UUID NOT NULL,
    target_node_id UUID NOT NULL,

    -- Edge classification
    edge_type VARCHAR(50) NOT NULL,
    -- Values: 'IMPLIES', 'CONTRADICTS', 'PREREQUISITE', 'TEMPORAL', 'SIMILAR_TO', 'CAUSED_BY'

    -- Edge properties
    weight FLOAT DEFAULT 1.0,
    -- Strength of relationship (0.0-1.0)

    confidence FLOAT DEFAULT 1.0,
    -- How certain we are about this edge

    created_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(50) DEFAULT 'user',
    -- Values: 'user', 'inferred', 'system'

    -- Metadata
    metadata JSONB,

    FOREIGN KEY (owner_key) REFERENCES api_keys(key),
    FOREIGN KEY (source_node_id) REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_node_id) REFERENCES knowledge_nodes(id) ON DELETE CASCADE,

    -- Prevent duplicate edges
    UNIQUE(source_node_id, target_node_id, edge_type)
);

-- Indexes
CREATE INDEX idx_knowledge_edges_owner ON knowledge_edges(owner_key);
CREATE INDEX idx_knowledge_edges_source ON knowledge_edges(source_node_id);
CREATE INDEX idx_knowledge_edges_target ON knowledge_edges(target_node_id);
CREATE INDEX idx_knowledge_edges_type ON knowledge_edges(owner_key, edge_type);
CREATE INDEX idx_knowledge_edges_both ON knowledge_edges(source_node_id, target_node_id);
```

---

## Graph Operations

### 1. Traversal Queries

#### Find all nodes that IMPLY a given node
```sql
SELECT n.*
FROM knowledge_nodes n
JOIN knowledge_edges e ON e.source_node_id = n.id
WHERE e.target_node_id = %s
  AND e.edge_type = 'IMPLIES'
  AND e.owner_key = %s;
```

#### Find all PREREQUISITEs for a PROCEDURE
```sql
WITH RECURSIVE prerequisites AS (
    -- Base case: direct prerequisites
    SELECT target_node_id, 1 as depth
    FROM knowledge_edges
    WHERE source_node_id = %s
      AND edge_type = 'PREREQUISITE'
      AND owner_key = %s

    UNION

    -- Recursive case: prerequisites of prerequisites
    SELECT e.target_node_id, p.depth + 1
    FROM knowledge_edges e
    JOIN prerequisites p ON e.source_node_id = p.target_node_id
    WHERE e.edge_type = 'PREREQUISITE'
      AND e.owner_key = %s
      AND p.depth < 5  -- Prevent infinite loops
)
SELECT n.*, p.depth
FROM knowledge_nodes n
JOIN prerequisites p ON n.id = p.target_node_id
ORDER BY p.depth;
```

#### Detect CONTRADICTIONS
```sql
SELECT n1.content AS statement1,
       n2.content AS statement2,
       e.confidence AS contradiction_confidence
FROM knowledge_edges e
JOIN knowledge_nodes n1 ON e.source_node_id = n1.id
JOIN knowledge_nodes n2 ON e.target_node_id = n2.id
WHERE e.edge_type = 'CONTRADICTS'
  AND e.owner_key = %s
  AND e.confidence > 0.7
ORDER BY e.confidence DESC;
```

### 2. Temporal Reasoning

#### Knowledge Evolution Timeline
```sql
SELECT n.*, e.created_at AS superseded_at
FROM knowledge_nodes n
LEFT JOIN knowledge_edges e ON e.source_node_id = n.id
                            AND e.edge_type = 'TEMPORAL'
WHERE n.owner_key = %s
  AND n.node_type = 'FACT'
ORDER BY n.created_at;
```

#### Current Beliefs (latest in TEMPORAL chain)
```sql
-- Nodes with no outgoing TEMPORAL edges (i.e., most recent)
SELECT n.*
FROM knowledge_nodes n
LEFT JOIN knowledge_edges e ON e.source_node_id = n.id
                            AND e.edge_type = 'TEMPORAL'
WHERE n.owner_key = %s
  AND n.node_type IN ('FACT', 'HYPOTHESIS')
  AND e.id IS NULL;
```

### 3. Pattern Discovery

#### Find PATTERNs similar to current problem
```sql
SELECT n.*, 1 - (n.embedding <=> %s::vector) AS similarity
FROM knowledge_nodes n
WHERE n.owner_key = %s
  AND n.node_type = 'PATTERN'
  AND n.access_frequency > 2  -- Must have occurred multiple times
ORDER BY n.embedding <=> %s::vector
LIMIT 10;
```

#### Identify causal chains (A caused B caused C)
```sql
WITH RECURSIVE causal_chain AS (
    -- Base case
    SELECT source_node_id, target_node_id, 1 AS depth,
           ARRAY[source_node_id, target_node_id] AS path
    FROM knowledge_edges
    WHERE source_node_id = %s
      AND edge_type = 'CAUSED_BY'
      AND owner_key = %s

    UNION

    -- Recursive case
    SELECT e.source_node_id, e.target_node_id, c.depth + 1,
           c.path || e.target_node_id
    FROM knowledge_edges e
    JOIN causal_chain c ON e.source_node_id = c.target_node_id
    WHERE e.edge_type = 'CAUSED_BY'
      AND e.owner_key = %s
      AND c.depth < 5
      AND NOT (e.target_node_id = ANY(c.path))  -- Prevent cycles
)
SELECT n.content, c.depth
FROM causal_chain c
JOIN knowledge_nodes n ON n.id = c.target_node_id
ORDER BY c.depth;
```

---

## Migration Strategy

### Phase 4.1: Schema Creation (Week 17-20)

**Approach**: Additive migration (no downtime)

1. Create `knowledge_nodes` and `knowledge_edges` tables
2. Existing `memories` table stays intact
3. Dual-write: new data goes to both `memories` and `knowledge_nodes`

#### Migration SQL

```sql
-- migrations/004_knowledge_graph.sql

-- 1. Create tables
-- (schemas defined above)

-- 2. Backfill existing memories as FACT nodes
INSERT INTO knowledge_nodes (id, owner_key, node_type, content, embedding, created_at, source)
SELECT id, owner_key, 'FACT', content, embedding, timestamp, 'user'
FROM memories
WHERE content IS NOT NULL;

-- 3. Create TEMPORAL edges for memories from same session
WITH ordered_memories AS (
    SELECT id, session_id, owner_key, timestamp,
           LAG(id) OVER (PARTITION BY session_id, owner_key ORDER BY timestamp) AS prev_id
    FROM memories
    WHERE session_id IS NOT NULL
)
INSERT INTO knowledge_edges (owner_key, source_node_id, target_node_id, edge_type, created_by)
SELECT owner_key, prev_id, id, 'TEMPORAL', 'system'
FROM ordered_memories
WHERE prev_id IS NOT NULL;

-- 4. Infer SIMILAR_TO edges for highly similar memories (> 0.95 similarity)
-- (Run as background job to avoid long migration)
```

**Rollback Plan**: Drop `knowledge_nodes` and `knowledge_edges` tables, continue with `memories` only

---

### Phase 4.2: Graph Queries (Week 21-24)

**API Endpoints**:

#### POST /v1/graph/traverse
```json
{
  "start_node_id": "node_abc123",
  "edge_type": "IMPLIES",
  "max_depth": 3,
  "direction": "outgoing"
}
```

**Response**:
```json
{
  "nodes": [
    {
      "id": "node_abc123",
      "type": "FACT",
      "content": "User is authenticated",
      "depth": 0
    },
    {
      "id": "node_def456",
      "type": "FACT",
      "content": "User has access to dashboard",
      "depth": 1
    }
  ],
  "edges": [
    {
      "source": "node_abc123",
      "target": "node_def456",
      "type": "IMPLIES",
      "confidence": 0.95
    }
  ]
}
```

#### POST /v1/graph/contradictions
```json
{
  "min_confidence": 0.7
}
```

**Response**:
```json
{
  "contradictions": [
    {
      "node1": {
        "id": "node_123",
        "content": "Python 2 is the standard",
        "created_at": "2015-01-01"
      },
      "node2": {
        "id": "node_456",
        "content": "Python 3 is the standard",
        "created_at": "2020-01-01"
      },
      "confidence": 0.92,
      "resolution": "temporal_evolution",
      "suggestion": "Python 3 is more recent. Archive Python 2 fact?"
    }
  ]
}
```

---

## Integration with Reasoning (001)

### Context Retrieval

**Before** (flat vector search):
```python
# Just return top-K similar memories
results = search_memories(query, k=5)
```

**After** (graph-enhanced):
```python
# 1. Find similar nodes
similar_nodes = graph.search_similar(query, k=3)

# 2. For each node, get connected nodes
context_nodes = []
for node in similar_nodes:
    context_nodes.append(node)

    # Add nodes that IMPLY this one (additional context)
    implications = graph.traverse(node.id, edge_type="IMPLIES", direction="incoming")
    context_nodes.extend(implications[:2])

    # Add PREREQUISITEs if it's a PROCEDURE
    if node.type == "PROCEDURE":
        prereqs = graph.traverse(node.id, edge_type="PREREQUISITE")
        context_nodes.extend(prereqs)

# 3. Build context with richer information
context = build_graph_context(context_nodes)
```

**Benefit**: Reasoning gets not just similar facts, but also implications and dependencies

---

## Database Schema Changes

### New Tables

- `knowledge_nodes` (main node storage)
- `knowledge_edges` (relationships)

### Extend Existing

```sql
-- Link memories to knowledge_nodes
ALTER TABLE memories ADD COLUMN knowledge_node_id UUID REFERENCES knowledge_nodes(id);
CREATE INDEX idx_memories_knowledge_node ON memories(knowledge_node_id);
```

---

## Implementation Phases

### Week 17-20: Schema + Migration

- [ ] Create `knowledge_nodes` table
- [ ] Create `knowledge_edges` table
- [ ] Backfill existing memories as FACT nodes
- [ ] Create initial TEMPORAL edges
- [ ] Performance testing on large graphs (100k+ nodes)

### Week 21-24: Graph Queries

- [ ] Implement traversal algorithms (BFS/DFS)
- [ ] Contradiction detection
- [ ] Temporal reasoning queries
- [ ] API endpoints
- [ ] Query performance optimization

### Week 25-28: Integration + Optimization

- [ ] Integrate graph queries with reasoning loop
- [ ] Segment → Node conversion logic
- [ ] Edge creation during reasoning
- [ ] Cleanup: prune low-access nodes
- [ ] Documentation

---

## Testing Strategy

### Unit Tests

```python
@pytest.mark.asyncio
async def test_graph_traversal():
    # Create nodes
    fact1 = await graph.create_node("FACT", "User is authenticated")
    fact2 = await graph.create_node("FACT", "User has access")

    # Create edge
    await graph.create_edge(fact1.id, fact2.id, "IMPLIES")

    # Traverse
    results = await graph.traverse(fact1.id, edge_type="IMPLIES")
    assert len(results) == 1
    assert results[0].id == fact2.id

@pytest.mark.asyncio
async def test_contradiction_detection():
    fact1 = await graph.create_node("FACT", "Python 2 is standard")
    fact2 = await graph.create_node("FACT", "Python 3 is standard")

    await graph.create_edge(fact1.id, fact2.id, "CONTRADICTS", confidence=0.9)

    contradictions = await graph.find_contradictions(min_confidence=0.8)
    assert len(contradictions) == 1
```

### Performance Tests

```python
@pytest.mark.asyncio
async def test_large_graph_performance():
    # Create 10k nodes
    nodes = []
    for i in range(10000):
        node = await graph.create_node("FACT", f"Fact {i}")
        nodes.append(node)

    # Create 20k edges (random connections)
    for i in range(20000):
        src = random.choice(nodes)
        tgt = random.choice(nodes)
        await graph.create_edge(src.id, tgt.id, "IMPLIES")

    # Test traversal performance
    start = time.time()
    results = await graph.traverse(nodes[0].id, max_depth=3)
    duration = time.time() - start

    assert duration < 0.2  # < 200ms for depth-3 traversal
```

---

## Pricing Implications

No new tier limits. Graph is **infrastructure upgrade**, transparent to users.

**Enterprise feature**: Self-hosted Neo4j option for massive graphs (> 1M nodes)

---

## Success Metrics

- **Graph query latency**: < 200ms (p95) for depth-3 traversal
- **Contradiction detection**: Find 90%+ of true contradictions
- **Migration success**: 100% of memories migrated with 0 data loss
- **Performance**: No degradation vs vector-only search

---

## Alternative: Neo4j Integration

### Option B: Neo4j Sidecar

**Pros**:
- Graph-native database
- Cypher query language (expressive)
- Better performance for complex traversals

**Cons**:
- New dependency (operational complexity)
- Sync required between PostgreSQL and Neo4j
- Higher cost

**When to consider**: If PostgreSQL graph queries become bottleneck (> 500ms p95)

**Migration path**:
```python
# Dual-write to both PostgreSQL and Neo4j
async def create_node(content, node_type):
    # PostgreSQL
    pg_node = await pg.create_node(content, node_type)

    # Neo4j
    neo4j_node = await neo4j.create_node(content, node_type, id=pg_node.id)

    return pg_node
```

---

## Related Documents

- [001_reasoning_as_a_service.md](./001_reasoning_as_a_service.md) - Graph-enhanced context retrieval
- [003_proactive_memory.md](./003_proactive_memory.md) - CAUSED_BY edges for failure analysis
- [005_hybrid_memory_agent.md](./005_hybrid_memory_agent.md) - Graph traversal module
- [006_implementation_roadmap.md](./006_implementation_roadmap.md) - Phase 4 timeline

---

*Document Status*: ✅ Ready for Phase 4
*Last Updated*: 2026-02-13
*Recommendation*: Start with PostgreSQL (Option A), migrate to Neo4j only if needed
