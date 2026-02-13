# Structured Protocol Levels

**Category**: Medium Term (1-2 months after Phase 1)
**Priority**: 🟡 MEDIUM
**Status**: Proposal
**Dependencies**: [001_reasoning_as_a_service.md](./001_reasoning_as_a_service.md) (must be stable)

---

## Vision

Optimize memory operations by introducing a **3-level protocol** that automatically routes queries to the most cost-effective operation level while maintaining quality.

**Problem**: Not all memory operations need expensive LLM calls. Simple fact lookups can be served via SQL queries, while only complex synthesis requires full text generation.

**Solution**: Structured protocol with automatic level selection based on query complexity.

**Expected Impact**: **40%+ cost savings** vs always using Level 3 (full LLM synthesis)

---

## Current State Analysis

### What Exists Today

In the current Zephyra implementation (`/home/user/uni-memory/app/main.py`):

- **All queries go through LLM synthesis** (`execute_llm_call()`)
- No differentiation between simple vs complex operations
- Vector search exists but always followed by LLM generation
- High token usage even for trivial queries

**Example inefficiency**:
```python
# Current: Simple fact lookup still uses full LLM call
query = "What is the capital of France?"
results = search_memories(query)  # Vector search
answer = llm.synthesize(query, results)  # Unnecessary LLM call
# Could just return results[0].content directly!
```

### Cost Analysis

**Current approach** (always Level 3):
- Query: "What is my API key?"
- Vector search: free (local embeddings)
- LLM synthesis: ~200 tokens × $0.60/1M = $0.00012
- **Latency**: ~2-3s

**With 3-level protocol**:
- Level 1 (SQL): ~$0, ~100ms
- Savings: $0.00012 per query
- At scale (10k queries/day): **$438/year saved**

---

## Proposed Architecture

### Level Definitions

#### Level 1: Quick Ops (No LLM)

**Characteristics**:
- Deterministic SQL queries
- No embeddings, no text generation
- < 100ms latency
- $0 cost

**Operations**:
- `GET(entity_id)` → Retrieve fact by ID
- `CHECK(claim)` → Consistency score via exact match
- `COUNT(filter)` → Count memories matching filter
- `EXISTS(condition)` → Boolean check

**Use Cases**:
- "Show me memory #12345"
- "How many memories do I have about Python?"
- "Do I have any memories from 2025-01-15?"

**Implementation**:
```python
class Level1Operations:
    async def get(self, entity_id: str, owner_key: str) -> dict:
        """Direct SQL query by ID"""
        query = "SELECT * FROM memories WHERE id = %s AND owner_key = %s"
        result = await db.fetch_one(query, (entity_id, owner_key))
        return result

    async def check(self, claim: str, owner_key: str) -> float:
        """Consistency score via exact/fuzzy match"""
        query = "SELECT content FROM memories WHERE owner_key = %s"
        memories = await db.fetch_all(query, (owner_key,))

        # Simple string similarity (no LLM)
        max_similarity = 0.0
        for mem in memories:
            similarity = fuzz.ratio(claim, mem['content']) / 100.0
            max_similarity = max(max_similarity, similarity)

        return max_similarity

    async def count(self, filters: dict, owner_key: str) -> int:
        """Count memories matching filters"""
        query = "SELECT COUNT(*) FROM memories WHERE owner_key = %s"
        conditions = []
        params = [owner_key]

        if 'workspace' in filters:
            conditions.append("workspace = %s")
            params.append(filters['workspace'])

        if 'date_from' in filters:
            conditions.append("timestamp >= %s")
            params.append(filters['date_from'])

        if conditions:
            query += " AND " + " AND ".join(conditions)

        result = await db.fetch_one(query, tuple(params))
        return result['count']
```

---

#### Level 2: Moderate Ops (Encoder Only)

**Characteristics**:
- Vector similarity search (pgvector)
- Local embeddings (SentenceTransformer)
- No text generation
- < 500ms latency
- $0 cost (no API calls)

**Operations**:
- `SEARCH(query_embedding)` → Ranked results by similarity
- `SUGGEST(context_embedding)` → Proactive hints based on context
- `CLUSTER(embeddings)` → Group similar memories
- `NEIGHBORS(entity_id)` → Find related memories

**Use Cases**:
- "Find memories similar to this problem"
- "What did I learn about React hooks?"
- "Show me related failures to this error"

**Implementation**:
```python
class Level2Operations:
    def __init__(self, embedding_model, db):
        self.embeddings = embedding_model
        self.db = db

    async def search(self, query: str, owner_key: str, limit: int = 10) -> List[dict]:
        """Semantic search via pgvector"""
        # Generate embedding locally (no API call)
        query_embedding = self.embeddings.encode(query)

        # pgvector cosine similarity
        sql = """
        SELECT id, content, workspace, timestamp,
               1 - (embedding <=> %s::vector) AS similarity
        FROM memories
        WHERE owner_key = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """

        results = await self.db.fetch_all(
            sql,
            (query_embedding.tolist(), owner_key, query_embedding.tolist(), limit)
        )

        return results

    async def suggest(self, context: str, owner_key: str, limit: int = 3) -> List[str]:
        """Proactive hints based on context embedding"""
        context_embedding = self.embeddings.encode(context)

        # Search for patterns/procedures similar to current context
        sql = """
        SELECT content
        FROM memories
        WHERE owner_key = %s
          AND metadata->>'type' IN ('pattern', 'procedure')
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """

        results = await self.db.fetch_all(
            sql,
            (owner_key, context_embedding.tolist(), limit)
        )

        return [r['content'] for r in results]

    async def neighbors(self, entity_id: str, owner_key: str, k: int = 5) -> List[dict]:
        """Find K nearest neighbors to a memory"""
        # Get embedding of target memory
        target = await self.db.fetch_one(
            "SELECT embedding FROM memories WHERE id = %s AND owner_key = %s",
            (entity_id, owner_key)
        )

        # Find nearest neighbors
        sql = """
        SELECT id, content, 1 - (embedding <=> %s::vector) AS similarity
        FROM memories
        WHERE owner_key = %s AND id != %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """

        neighbors = await self.db.fetch_all(
            sql,
            (target['embedding'], owner_key, entity_id, target['embedding'], k)
        )

        return neighbors
```

---

#### Level 3: Complex Ops (Full Generation)

**Characteristics**:
- Full LLM synthesis with retrieved context
- Text generation
- < 3s latency
- $0.01-0.10 cost per query (API calls)

**Operations**:
- `SYNTHESIZE(query, context)` → Strategic response
- `EXPLAIN(contradiction)` → Clarification of inconsistencies
- `REASON(problem, context)` → Multi-step reasoning (from 001)
- `SUMMARIZE(memories)` → Concise summary of multiple memories

**Use Cases**:
- "Explain how I solved this problem last time"
- "What's the contradiction between these two memories?"
- "Synthesize everything I know about X"

**Implementation**:
```python
class Level3Operations:
    def __init__(self, llm_client, memory_service):
        self.llm = llm_client
        self.memory = memory_service

    async def synthesize(self, query: str, owner_key: str) -> dict:
        """Full RAG synthesis"""
        # Retrieve relevant memories (Level 2)
        level2 = Level2Operations(embeddings, db)
        relevant = await level2.search(query, owner_key, limit=5)

        # Build context
        context = "\n\n".join([f"- {m['content']}" for m in relevant])

        # LLM synthesis
        prompt = f"""Based on these memories:

{context}

Answer the query: {query}

Provide a clear, concise answer."""

        response = await self.llm.generate(prompt, max_tokens=500)

        return {
            "answer": response.text,
            "sources": [m['id'] for m in relevant],
            "tokens_used": response.tokens_used,
            "level_used": 3
        }

    async def explain(self, contradiction: str, owner_key: str) -> str:
        """Explain contradiction"""
        prompt = f"""Two memories contradict each other:

{contradiction}

Explain the contradiction and suggest how to resolve it."""

        response = await self.llm.generate(prompt, max_tokens=300)
        return response.text

    async def summarize(self, memory_ids: List[str], owner_key: str) -> str:
        """Summarize multiple memories"""
        # Fetch memories
        memories = []
        for id in memory_ids:
            mem = await self.memory.get_by_id(id, owner_key)
            memories.append(mem['content'])

        context = "\n\n".join(memories)

        prompt = f"""Summarize these memories concisely:

{context}

Summary (max 3 paragraphs):"""

        response = await self.llm.generate(prompt, max_tokens=400)
        return response.text
```

---

### Automatic Level Selection

**Query Complexity Classifier** (heuristic-based MVP):

```python
class ProtocolRouter:
    def __init__(self, level1, level2, level3):
        self.level1 = level1
        self.level2 = level2
        self.level3 = level3

    async def route(self, query: str, owner_key: str, auto_level: bool = True) -> dict:
        """Route query to appropriate level"""

        if not auto_level:
            # User explicitly requested level 3
            return await self.level3.synthesize(query, owner_key)

        # Classify query
        level = self._classify_query(query)

        if level == 1:
            # Try Level 1 operations
            if self._is_get_query(query):
                entity_id = self._extract_entity_id(query)
                result = await self.level1.get(entity_id, owner_key)
                return {"answer": result, "level_used": 1, "tokens_used": 0}

            elif self._is_count_query(query):
                count = await self.level1.count({}, owner_key)
                return {"answer": f"You have {count} memories", "level_used": 1, "tokens_used": 0}

        if level <= 2:
            # Try Level 2 operations
            results = await self.level2.search(query, owner_key)

            if len(results) > 0 and results[0]['similarity'] > 0.9:
                # High confidence: return top result directly
                return {
                    "answer": results[0]['content'],
                    "level_used": 2,
                    "tokens_used": 0,
                    "sources": [results[0]['id']]
                }

        # Fallback to Level 3
        return await self.level3.synthesize(query, owner_key)

    def _classify_query(self, query: str) -> int:
        """Classify query complexity (heuristic)"""
        query_lower = query.lower()

        # Level 1 indicators
        if any(keyword in query_lower for keyword in ['show me memory', 'get memory', 'memory #']):
            return 1

        if any(keyword in query_lower for keyword in ['how many', 'count', 'do i have']):
            return 1

        # Level 2 indicators
        if any(keyword in query_lower for keyword in ['find', 'search', 'similar', 'related']):
            return 2

        # Level 3 indicators
        if any(keyword in query_lower for keyword in ['explain', 'why', 'how', 'synthesize', 'summarize']):
            return 3

        # Default: Level 2 (semantic search)
        return 2

    def _is_get_query(self, query: str) -> bool:
        return 'memory #' in query.lower() or 'show me memory' in query.lower()

    def _extract_entity_id(self, query: str) -> str:
        import re
        match = re.search(r'#([a-f0-9-]+)', query)
        return match.group(1) if match else None
```

---

## Database Schema Changes

No new tables needed. Optimization only.

**Indexes to add** (if not exist):
```sql
-- Level 1 optimizations
CREATE INDEX IF NOT EXISTS idx_memories_workspace ON memories(owner_key, workspace);
CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(owner_key, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_memories_metadata_type ON memories(owner_key, (metadata->>'type'));

-- Level 2 optimizations (already exists)
-- HNSW index on embedding column (already created in init_db)
```

---

## API Design

### POST /v1/memory/protocol

Unified protocol endpoint with automatic level selection.

**Request**:
```json
{
  "query": "Find memories about Python",
  "auto_level": true,
  "max_results": 10,
  "context": {
    "workspace": "coding",
    "prefer_recent": true
  }
}
```

**Response** (Level 2 used):
```json
{
  "answer": [
    {
      "id": "mem_abc123",
      "content": "Python list comprehensions are faster than loops",
      "similarity": 0.92,
      "timestamp": "2026-02-10T14:30:00Z"
    }
  ],
  "level_used": 2,
  "tokens_used": 0,
  "latency_ms": 245,
  "explanation": "High-confidence semantic search result, no LLM synthesis needed"
}
```

**Response** (Level 3 fallback):
```json
{
  "answer": "Based on your memories, you've learned that Python list comprehensions offer better performance than traditional loops, with typical speedups of 20-30%. You also noted that they're more Pythonic and readable for simple transformations.",
  "level_used": 3,
  "tokens_used": 87,
  "latency_ms": 1850,
  "sources": ["mem_abc123", "mem_def456"],
  "explanation": "Query required synthesis of multiple memories"
}
```

### Individual Level Endpoints

For explicit level control:

#### POST /v1/memory/protocol/get (Level 1)
```json
{
  "entity_id": "mem_abc123"
}
```

#### POST /v1/memory/protocol/search (Level 2)
```json
{
  "query": "Python performance tips",
  "limit": 10
}
```

#### POST /v1/memory/protocol/synthesize (Level 3)
```json
{
  "query": "Explain my Python learning journey",
  "max_tokens": 500
}
```

---

## Integration Points

### 1. Reasoning Loop (from 001)

**Integration**: Reasoning loop uses protocol for memory retrieval

```python
# In reasoning_loop.py
async def retrieve_context(query: str, session_id: str, owner_key: str):
    # Use Level 2 for fast retrieval (no synthesis needed)
    protocol = ProtocolRouter(level1, level2, level3)
    result = await protocol.route(
        query=query,
        owner_key=owner_key,
        auto_level=True
    )

    # Reasoning loop handles synthesis, so we only need facts
    if result['level_used'] <= 2:
        return result['answer']
    else:
        # Level 3 was used, extract sources
        return result.get('sources', [])
```

**Benefit**: Reduces token usage in reasoning loop by 30-40%

---

## Implementation Phases

### Week 1: Level 1 + Level 2

- [ ] Implement `Level1Operations` class
- [ ] Implement `Level2Operations` class
- [ ] Add database indexes
- [ ] Unit tests for both levels

### Week 2: Level 3 + Router

- [ ] Implement `Level3Operations` class
- [ ] Implement `ProtocolRouter` with heuristics
- [ ] Integration tests

### Week 3: API + Optimization

- [ ] `POST /v1/memory/protocol` endpoint
- [ ] Individual level endpoints
- [ ] Cost tracking (tokens saved)
- [ ] Performance tuning

### Week 4: Testing + Docs

- [ ] Load testing (compare latency across levels)
- [ ] A/B test: auto_level vs always Level 3
- [ ] API documentation
- [ ] User guide

---

## Testing Strategy

### Unit Tests

```python
@pytest.mark.asyncio
async def test_level1_get():
    level1 = Level1Operations(db)
    result = await level1.get("mem_123", "api_key_123")
    assert result['id'] == "mem_123"

@pytest.mark.asyncio
async def test_level2_search():
    level2 = Level2Operations(embeddings, db)
    results = await level2.search("Python", "api_key_123")
    assert len(results) > 0
    assert results[0]['similarity'] > 0.5
```

### Performance Tests

```python
@pytest.mark.asyncio
async def test_latency_level1():
    start = time.time()
    await level1.get("mem_123", "api_key_123")
    latency = time.time() - start
    assert latency < 0.1  # < 100ms

@pytest.mark.asyncio
async def test_latency_level2():
    start = time.time()
    await level2.search("Python", "api_key_123")
    latency = time.time() - start
    assert latency < 0.5  # < 500ms
```

### Cost Analysis Test

```python
@pytest.mark.asyncio
async def test_cost_savings():
    queries = [
        "Show me memory #123",          # L1
        "Find Python memories",         # L2
        "Explain my learning journey"   # L3
    ]

    total_tokens_auto = 0
    total_tokens_always_l3 = 0

    for query in queries:
        # With auto-level
        result_auto = await router.route(query, auto_level=True)
        total_tokens_auto += result_auto['tokens_used']

        # Always L3
        result_l3 = await level3.synthesize(query)
        total_tokens_always_l3 += result_l3['tokens_used']

    savings = (total_tokens_always_l3 - total_tokens_auto) / total_tokens_always_l3
    assert savings > 0.4  # > 40% savings
```

---

## Pricing Implications

No changes to tier limits. Protocol optimization is **transparent cost reduction**.

**User benefit**: Faster responses, lower costs (if BYO API keys)

---

## Success Metrics

- **Cost Savings**: > 40% vs always-L3
- **Latency Reduction**: 50%+ queries < 500ms (vs 2-3s for L3)
- **Accuracy**: > 90% of auto-routed queries return correct level
- **User Satisfaction**: No degradation in answer quality

---

## Related Documents

- [001_reasoning_as_a_service.md](./001_reasoning_as_a_service.md) - Reasoning loop integration
- [003_proactive_memory.md](./003_proactive_memory.md) - Proactive suggestions use L2
- [006_implementation_roadmap.md](./006_implementation_roadmap.md) - Phase 2 timeline

---

*Document Status*: ✅ Ready for Phase 2
*Last Updated*: 2026-02-13
*Dependencies*: 001 must be stable first
