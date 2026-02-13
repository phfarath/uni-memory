# Proactive Memory

**Category**: Medium-Long Term (2-4 months after Phase 1)
**Priority**: 🟡 MEDIUM-HIGH
**Status**: Proposal
**Dependencies**: [001_reasoning_as_a_service.md](./001_reasoning_as_a_service.md) (needs reasoning sessions)

---

## Vision

Transform memory from **passive storage** to **active cognitive partner** that proactively interrupts reasoning with helpful suggestions based on past failures and patterns.

**Inspired by**: Extended Mind Theory (Clark & Chalmers, 1998) - external tools become part of cognitive system

**Core Concept**: Memory that "speaks back"

**Example**:
```
Reasoning Agent: "I'll try approach A to solve this..."

Memory System: "⚠️ WAIT - Approach A failed in 3 previous attempts
                with similar inputs.

                Pattern detected: When characteristic Z is present,
                approach B has 80% success rate.

                Evidence: [mem_123, mem_456, mem_789]"
```

**Competitive Differentiation**: No existing memory platform (Mem0, Zep, Pinecone) offers proactive suggestions.

---

## Current State Analysis

### What Exists Today

In current Zephyra (`/home/user/uni-memory/app/main.py`):

- **Passive retrieval**: User/agent explicitly queries memory
- **No pattern detection**: Failures stored but not analyzed
- **No proactive suggestions**: Memory never interrupts reasoning
- **No outcome tracking**: Don't know if retrieved memories were helpful

**Current flow**:
```python
# User asks → Memory responds
query = "How do I solve X?"
memories = search_memories(query)
answer = synthesize(memories)
# Memory is reactive, not proactive
```

### What's Missing

- ❌ Failure pattern detection
- ❌ Interruption budget system
- ❌ Proactive suggestion generation
- ❌ Outcome tracking (success/failure)
- ❌ User feedback loop

---

## Proposed Architecture

### High-Level Flow

```mermaid
sequenceDiagram
    participant Agent as Reasoning Agent
    participant Loop as Reasoning Loop
    participant Memory as Proactive Memory
    participant Pattern as Pattern Detector

    Agent->>Loop: Start reasoning on problem
    Loop->>Memory: Store segment 0
    Loop->>Memory: Generate segment 1

    Memory->>Pattern: Analyze current context
    Pattern->>Pattern: Search for failure patterns
    Pattern-->>Memory: Pattern found (confidence 0.85)

    Memory->>Memory: Check interrupt budget (3/5 left)
    Memory->>Memory: confidence > threshold? YES

    Memory->>Loop: ⚠️ INTERRUPT with suggestion
    Loop->>Agent: Show suggestion in trace

    Agent->>Loop: Continue (with or without suggestion)
    Loop->>Memory: Store outcome + feedback
```

### Components

#### 1. Pattern Detector

**Responsibility**: Identify recurring failure patterns

**Algorithm**:
```python
class FailurePatternDetector:
    def __init__(self, memory_service, embeddings):
        self.memory = memory_service
        self.embeddings = embeddings

    async def detect_pattern(
        self,
        current_context: str,
        owner_key: str,
        threshold: float = 0.85
    ) -> Optional[dict]:
        """Detect if current context matches known failure pattern"""

        # 1. Encode current context
        context_embedding = self.embeddings.encode(current_context)

        # 2. Search for similar failures
        sql = """
        SELECT content, metadata
        FROM memories
        WHERE owner_key = %s
          AND metadata->>'type' = 'failure'
          AND metadata->>'outcome' = 'failed'
        ORDER BY embedding <=> %s::vector
        LIMIT 10
        """

        failures = await self.memory.db.fetch_all(
            sql,
            (owner_key, context_embedding.tolist())
        )

        if not failures:
            return None

        # 3. Check similarity
        top_failure = failures[0]
        similarity = 1 - cosine_distance(
            context_embedding,
            top_failure['embedding']
        )

        if similarity < threshold:
            return None

        # 4. Extract pattern
        pattern = {
            "type": "failure_pattern",
            "confidence": similarity,
            "failed_approach": top_failure['metadata'].get('approach'),
            "failure_count": len([f for f in failures if similarity > 0.80]),
            "suggested_alternative": top_failure['metadata'].get('alternative'),
            "evidence": [f['id'] for f in failures[:3]]
        }

        return pattern

    async def suggest_alternative(self, pattern: dict, owner_key: str) -> str:
        """Generate suggestion based on pattern"""

        if pattern['suggested_alternative']:
            return pattern['suggested_alternative']

        # Search for successful approaches with similar context
        sql = """
        SELECT content, metadata
        FROM memories
        WHERE owner_key = %s
          AND metadata->>'type' = 'success'
          AND metadata->>'context_similarity' > '0.7'
        LIMIT 5
        """

        successes = await self.memory.db.fetch_all(sql, (owner_key,))

        if successes:
            # Extract common approach from successes
            approaches = [s['metadata'].get('approach') for s in successes]
            # Return most common
            from collections import Counter
            most_common = Counter(approaches).most_common(1)[0][0]
            return most_common

        return "Try a different approach based on past failures"
```

---

#### 2. Interruption Policy

**Responsibility**: Decide when to interrupt vs stay silent

**MVP**: Heuristic-based rules
**Future**: RL-learned policy (Phase 3.2)

**Interrupt Budget System**:
```python
class InterruptionPolicy:
    def __init__(self, budget: int = 5, threshold: float = 0.8):
        self.budget = budget  # Max interrupts per session
        self.threshold = threshold  # Confidence needed to interrupt
        self.interrupts_used = 0

    def should_interrupt(
        self,
        pattern: dict,
        segment_number: int,
        reasoning_state: dict
    ) -> bool:
        """Decide if we should interrupt"""

        # 1. Budget check
        if self.interrupts_used >= self.budget:
            return False

        # 2. Confidence check
        if pattern['confidence'] < self.threshold:
            return False

        # 3. Timing check (don't interrupt too early)
        if segment_number < 1:
            return False

        # 4. Heuristic: only interrupt if failure_count >= 2
        if pattern.get('failure_count', 0) < 2:
            return False

        # 5. Heuristic: don't interrupt twice in a row
        if segment_number - reasoning_state.get('last_interrupt_segment', -999) < 3:
            return False

        return True

    def record_interrupt(self, segment_number: int):
        """Track that we interrupted"""
        self.interrupts_used += 1
        return segment_number
```

---

#### 3. Proactive Memory Service

**Responsibility**: Orchestrate pattern detection + interruption

**Integration with Reasoning Loop** (from 001):
```python
# In reasoning_loop.py
async def solve_with_proactive(
    problem: str,
    config: ReasoningConfig,
    owner_key: str,
    session_id: str,
    enable_proactive: bool = True
):
    context = problem
    segments = []
    interruptions = []

    # Initialize proactive components
    pattern_detector = FailurePatternDetector(memory, embeddings)
    interrupt_policy = InterruptionPolicy(budget=5, threshold=0.8)

    for i in range(config.max_segments):
        # 1. Generate reasoning segment
        segment = await llm.generate(context, max_tokens=1000)

        # 2. PROACTIVE CHECK (new!)
        if enable_proactive and i > 0:
            pattern = await pattern_detector.detect_pattern(segment, owner_key)

            if pattern and interrupt_policy.should_interrupt(pattern, i, {}):
                # Generate suggestion
                suggestion = await pattern_detector.suggest_alternative(pattern, owner_key)

                interruption = {
                    "segment": i,
                    "pattern": pattern,
                    "suggestion": suggestion,
                    "timestamp": datetime.utcnow()
                }

                interruptions.append(interruption)
                interrupt_policy.record_interrupt(i)

                # Inject suggestion into context
                context += f"\n\n💡 MEMORY INSIGHT: {suggestion}"

        # 3. Continue normal reasoning flow
        if is_complete(segment):
            return Result(status="completed", interruptions=interruptions, ...)

        summary = await summarize(segment)
        memory.store(summary, metadata={"type": "reasoning_summary"})

        # 4. Build context for next iteration
        context = build_context(problem, summary, relevant)
        segments.append(segment)

    return Result(status="incomplete", interruptions=interruptions, ...)
```

---

## Database Schema Changes

### Extend memories Table

```sql
-- Add outcome tracking
ALTER TABLE memories ADD COLUMN outcome VARCHAR(50);
-- Values: 'success', 'failure', 'unknown' (default)

ALTER TABLE memories ADD COLUMN feedback_score INTEGER;
-- User rating: -1 (not helpful), 0 (neutral), +1 (helpful)

-- Add indexes
CREATE INDEX idx_memories_outcome ON memories(owner_key, (metadata->>'type'), outcome);
CREATE INDEX idx_memories_feedback ON memories(owner_key, feedback_score);
```

### New Table: interruption_events

```sql
CREATE TABLE interruption_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    owner_key VARCHAR(255) NOT NULL,
    segment_number INTEGER NOT NULL,
    pattern_type VARCHAR(50),
    confidence FLOAT,
    suggestion TEXT,
    user_action VARCHAR(50), -- 'accepted', 'rejected', 'ignored'
    feedback_score INTEGER, -- -1, 0, +1
    created_at TIMESTAMP DEFAULT NOW(),

    FOREIGN KEY (session_id) REFERENCES reasoning_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (owner_key) REFERENCES api_keys(key)
);

CREATE INDEX idx_interruption_events_session ON interruption_events(session_id);
CREATE INDEX idx_interruption_events_feedback ON interruption_events(owner_key, feedback_score);
```

---

## API Design

### Extended /v1/reasoning/solve

**Request**:
```json
{
  "problem": "Implement binary search tree",
  "config": {
    "max_segments": 10,
    "enable_proactive": true,
    "interrupt_budget": 5,
    "interrupt_threshold": 0.8
  }
}
```

**Response**:
```json
{
  "status": "completed",
  "answer": "...",
  "reasoning_trace": [...],
  "interruptions": [
    {
      "segment": 2,
      "pattern": {
        "type": "failure_pattern",
        "confidence": 0.87,
        "failed_approach": "iterative_approach",
        "failure_count": 3,
        "evidence": ["mem_123", "mem_456", "mem_789"]
      },
      "suggestion": "⚠️ Approach 'iterative_approach' failed 3 times. Try 'recursive_approach' instead (80% success rate).",
      "timestamp": "2026-02-13T15:30:00Z"
    }
  ],
  "proactive_stats": {
    "patterns_detected": 2,
    "interrupts_triggered": 1,
    "budget_remaining": 4
  }
}
```

### POST /v1/memory/feedback

User provides feedback on memory/suggestion helpfulness.

**Request**:
```json
{
  "memory_id": "mem_123",
  "feedback_score": 1,
  "comment": "This suggestion saved me 2 hours!"
}
```

**Response**:
```json
{
  "status": "recorded",
  "memory_id": "mem_123",
  "updated_confidence": 0.92
}
```

---

## Implementation Phases

### Phase 3.1: MVP (Weeks 9-12, Heuristic-Based)

#### Week 9-10: Pattern Detection

- [ ] Extend `memories` table (outcome, feedback_score)
- [ ] Create `interruption_events` table
- [ ] Implement `FailurePatternDetector` class
- [ ] Store failures with `type=failure` metadata
- [ ] Unit tests for pattern detection

**Deliverables**:
- ✅ Failure patterns detected with 70%+ accuracy
- ✅ Pattern search < 200ms

#### Week 11-12: Interrupt Budget System

- [ ] Implement `InterruptionPolicy` class
- [ ] Heuristic rules (budget, threshold, timing)
- [ ] Suggestion generation
- [ ] Integration with reasoning loop
- [ ] API: `enable_proactive` flag

**Deliverables**:
- ✅ Interrupt budget enforced
- ✅ Suggestions appear in reasoning trace

---

### Phase 3.2: RL-Learned Policy (Weeks 13-16, Future)

#### Week 13-14: Dataset Collection

- [ ] Collect interruption outcomes
- [ ] Track user actions (accepted/rejected/ignored)
- [ ] Build dataset: (state, interrupt, outcome, feedback)
- [ ] Export for ML training

**Deliverables**:
- ✅ 500+ interruption events collected
- ✅ Dataset with labels

#### Week 15-16: RL Training

- [ ] Train binary classifier: `should_interrupt(state)`
- [ ] Features: pattern confidence, segment number, failure count, user history
- [ ] Reward: user feedback (+1 helpful, -1 not helpful)
- [ ] Replace heuristic policy with learned policy

**Deliverables**:
- ✅ Learned policy > 80% accuracy
- ✅ Online learning enabled (improve over time)

---

## Testing Strategy

### Unit Tests

```python
@pytest.mark.asyncio
async def test_pattern_detection():
    detector = FailurePatternDetector(memory, embeddings)

    # Store 3 failures with same approach
    for i in range(3):
        await memory.store(
            "Tried iterative approach, got stack overflow",
            metadata={"type": "failure", "approach": "iterative_approach"},
            outcome="failed"
        )

    # Check pattern detection
    pattern = await detector.detect_pattern(
        "I'll try iterative approach",
        owner_key="test"
    )

    assert pattern is not None
    assert pattern['confidence'] > 0.8
    assert pattern['failure_count'] == 3

@pytest.mark.asyncio
async def test_interrupt_budget():
    policy = InterruptionPolicy(budget=3)

    # Should allow first 3 interrupts
    for i in range(3):
        assert policy.should_interrupt({"confidence": 0.9}, i+1, {}) == True
        policy.record_interrupt(i+1)

    # Should block 4th interrupt (budget exceeded)
    assert policy.should_interrupt({"confidence": 0.9}, 4, {}) == False
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_proactive_interruption():
    # Setup: store failures
    await memory.store(
        "Approach X failed",
        metadata={"type": "failure", "approach": "X"},
        outcome="failed"
    )

    # Solve with proactive enabled
    result = await reasoning_loop.solve_with_proactive(
        problem="Try approach X",
        config=ReasoningConfig(enable_proactive=True),
        owner_key="test",
        session_id="test_session"
    )

    # Verify interruption occurred
    assert len(result.interruptions) > 0
    assert "Approach X failed" in result.interruptions[0]['suggestion']
```

### A/B Testing

```python
# Group A: Proactive enabled
group_a_results = []
for user in group_a_users:
    result = await solve_with_proactive(enable_proactive=True)
    group_a_results.append(result)

# Group B: Proactive disabled
group_b_results = []
for user in group_b_users:
    result = await solve_with_proactive(enable_proactive=False)
    group_b_results.append(result)

# Metrics
group_a_completion_rate = sum(r.status == "completed" for r in group_a_results) / len(group_a_results)
group_b_completion_rate = sum(r.status == "completed" for r in group_b_results) / len(group_b_results)

group_a_satisfaction = sum(r.feedback_score for r in group_a_results) / len(group_a_results)
group_b_satisfaction = sum(r.feedback_score for r in group_b_results) / len(group_b_results)
```

---

## User Experience

### Interruption UI (Reasoning Trace)

```json
{
  "reasoning_trace": [
    {
      "segment_number": 0,
      "content": "I'll implement this using approach A...",
      "summary": "Decided to use approach A"
    },
    {
      "segment_number": 1,
      "content": "Implementing approach A...",
      "summary": "Started implementation"
    },
    {
      "segment_number": 2,
      "interruption": {
        "icon": "💡",
        "title": "Memory Insight",
        "message": "⚠️ Approach A failed in 3 previous attempts with similar inputs.\n\nPattern: When input size > 1000, approach B has 80% success rate.\n\nEvidence: [View Past Failures]",
        "suggestion": "Consider using approach B instead",
        "actions": ["Accept", "Reject", "Show Evidence"]
      },
      "content": "After considering the suggestion, I'll try approach B...",
      "summary": "Switched to approach B"
    }
  ]
}
```

### Feedback Collection

After session completes:
```
🎯 How helpful were the memory insights?

Insight #1: "Switch from approach A to B"
[ Not Helpful ] [ Somewhat Helpful ] [ Very Helpful ]

Comment (optional): ___________________________
```

---

## Pricing Implications

No new tier limits. Proactive memory is **value-add feature** included in existing tiers.

**Positioning**: "Pro tier includes proactive memory suggestions"

---

## Success Metrics

### Technical Metrics

- **Pattern Detection Accuracy**: > 70% (detect true failures)
- **False Positive Rate**: < 20% (don't interrupt unnecessarily)
- **Latency**: < 200ms for pattern check (non-blocking)
- **Interrupt Rate**: 10-20% of segments (not too spammy)

### User Experience Metrics

- **Helpfulness**: 60%+ users rate suggestions as helpful
- **Adoption**: < 10% disable proactive mode
- **Completion Rate**: 10%+ higher with proactive enabled
- **Time Saved**: 20%+ reduction in reasoning segments (due to better approaches)

### Business Metrics

- **Conversion**: 30%+ of users who experience proactive upgrade to Pro
- **Retention**: 80%+ of Pro users keep proactive enabled
- **NPS Impact**: +5 points from proactive feature

---

## Related Documents

- [001_reasoning_as_a_service.md](./001_reasoning_as_a_service.md) - Reasoning loop integration
- [002_structured_protocol_levels.md](./002_structured_protocol_levels.md) - Level 2 for pattern search
- [004_knowledge_graph.md](./004_knowledge_graph.md) - Graph edges for causal patterns
- [006_implementation_roadmap.md](./006_implementation_roadmap.md) - Phase 3 timeline

---

*Document Status*: ✅ Ready for Phase 3
*Last Updated*: 2026-02-13
*Dependencies*: 001 reasoning loop must be stable
