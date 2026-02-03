# Auto-Capture Context - Implementation Report

**Feature**: Automatic Context Capture
**Spec**: `docs/futures/001_v1.0_auto_capture.md`
**Status**: Implemented
**Date**: 2026-02-03

---

## Overview

Implemented automatic capture of commands, file edits, errors, and decisions without manual `remember()` calls. The system monitors user activities and automatically creates memories based on configurable event detectors.

---

## Components Implemented

### 1. Database Schema

**File**: `app/main.py` (init_db function)

#### New Table: `auto_capture_events`
```sql
CREATE TABLE IF NOT EXISTS auto_capture_events (
    id SERIAL PRIMARY KEY,
    owner_key TEXT NOT NULL,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_data JSONB NOT NULL,
    captured_at TIMESTAMPTZ DEFAULT NOW(),
    processed BOOLEAN DEFAULT FALSE,
    memory_id INTEGER REFERENCES memories(id)
);
```

#### Indexes
- `idx_auto_capture_session` - For session-based queries
- `idx_auto_capture_processed` - For background worker efficiency
- `idx_auto_capture_owner` - For multi-tenant filtering

#### Tier Definition Update
Added `max_auto_capture_per_day` column to `tier_definitions`:
| Tier | Limit |
|------|-------|
| free | 100/day |
| pro | 5,000/day |
| team | 50,000/day |
| root | unlimited (-1) |

---

### 2. Auto-Capture Engine

**File**: `app/auto_capture.py` (NEW)

#### Event Detectors

All detectors implement the `EventDetector` abstract base class:

```python
class EventDetector(ABC):
    def detect(self, event_data: dict) -> bool
    def format_memory(self, event_data: dict) -> str
    def event_type(self) -> str
```

##### CommandDetector
Detects relevant shell commands:
- Git: `commit`, `push`, `merge`, `pull`, `checkout`, `branch`, `rebase`, `stash`, `tag`
- Package managers: `npm install`, `pip install`, `yarn add`, `pnpm install`
- Testing: `pytest`, `npm test`, `cargo test`, `go test`
- Containers: `docker build`, `docker run`, `docker-compose`
- Infrastructure: `kubectl apply`, `terraform apply`
- Build tools: `make`, `cmake`, `cargo build`, `go build`

##### FileEditDetector
Detects edits to source files:
- **Relevant extensions**: `.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.go`, `.rs`, `.java`, `.kt`, `.sql`, `.yaml`, `.json`, `.tf`, `.sh`
- **Ignored patterns**: `node_modules/`, `__pycache__/`, `.git/`, `dist/`, `build/`, `venv/`, `package-lock.json`, etc.

##### ErrorDetector
Detects error messages containing:
- `error:`, `exception:`, `traceback`, `failed`, `fatal`
- `segfault`, `panic:`, `TypeError`, `ImportError`
- HTTP errors: `404`, `500`, `502`, `503`

##### DecisionDetector
Detects architectural decisions (English & Portuguese):
- English: "i'll use", "decided to use", "architecture", "design pattern", "migrate to"
- Portuguese: "vou usar", "decidimos usar", "arquitetura", "migrar para"

#### AutoCaptureEngine Class

```python
class AutoCaptureEngine:
    def register_detector(self, detector: EventDetector)
    def enable_for_session(self, session_id: str, owner_key: str)
    def disable_for_session(self, session_id: str)
    def is_enabled(self, session_id: str) -> bool
    def get_owner_key(self, session_id: str) -> Optional[str]
    def process_event(self, event_data: dict) -> Optional[str]
    def _sanitize_content(self, content: str) -> str
```

#### Security: Sensitive Data Filtering

Before persistence, content is sanitized to remove:
- API keys (`api_key=...`, `sk_...`)
- Passwords (`password:...`, `pwd=...`)
- Tokens (`token=...`, `Bearer ...`)
- AWS credentials
- Private keys

---

### 3. Rate Limiter Integration

**File**: `app/rate_limiter.py`

#### New Action Type
```python
ACTION_AUTO_CAPTURE = "auto_capture"
```

#### Endpoint Mappings
```python
"POST /v1/auto-capture/enable": (ACTION_AUTO_CAPTURE, 1),
"POST /v1/auto-capture/disable": (ACTION_AUTO_CAPTURE, 1),
"POST /v1/auto-capture/event": (ACTION_AUTO_CAPTURE, 1),
"GET /v1/auto-capture/status": (ACTION_REQUEST, 1),
"tool:enable_auto_capture": (ACTION_AUTO_CAPTURE, 1),
"tool:disable_auto_capture": (ACTION_AUTO_CAPTURE, 1),
```

#### Usage Stats
Added `auto_capture` to `get_full_usage_stats()` response.

---

### 4. REST API Endpoints

**File**: `app/main.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/auto-capture/enable` | POST | Enable auto-capture for a session |
| `/v1/auto-capture/disable` | POST | Disable auto-capture for a session |
| `/v1/auto-capture/status` | GET | Get status and event statistics |
| `/v1/auto-capture/event` | POST | Capture an event (if enabled) |

#### Request/Response Examples

**Enable Auto-Capture**
```bash
curl -X POST http://localhost:8001/v1/auto-capture/enable \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "my-session"}'

# Response: {"status": "enabled", "session_id": "my-session"}
```

**Capture Event**
```bash
curl -X POST http://localhost:8001/v1/auto-capture/event \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session",
    "event_type": "command",
    "event_data": {"command": "git commit -m fix: login timeout"}
  }'

# Response: {"status": "captured", "event_id": 123, "session_id": "my-session"}
```

**Get Status**
```bash
curl "http://localhost:8001/v1/auto-capture/status?session_id=my-session" \
  -H "x-api-key: $API_KEY"

# Response:
# {
#   "session_id": "my-session",
#   "enabled": true,
#   "total_events": 15,
#   "processed_events": 12,
#   "pending_events": 3
# }
```

---

### 5. MCP Tools

**File**: `app/main.py`

| Tool | Description |
|------|-------------|
| `enable_auto_capture` | Enable auto-capture for a session |
| `disable_auto_capture` | Disable auto-capture for a session |
| `auto_capture_status` | Check status and statistics |

#### Example MCP Call
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "enable_auto_capture",
    "arguments": {"session_id": "claude-session"}
  }
}
```

---

### 6. Background Worker

**File**: `app/main.py`

#### Scheduler Configuration
```python
schedule.every(30).seconds.do(process_pending_events, ...)
schedule.every().sunday.at("03:00").do(cleanup_old_auto_capture_events, ...)
```

#### Processing Logic (`app/auto_capture.py`)
1. Fetch up to 100 unprocessed events
2. For each event:
   - Parse event_data JSON
   - Process through AutoCaptureEngine
   - If relevant, create memory via `add_memory_trace_logic()`
   - Mark event as processed
3. Log processing statistics

#### Cleanup Logic
- Runs weekly (Sunday 3 AM)
- Deletes processed events older than 30 days

---

### 7. Dependencies

**File**: `requirements.txt`

Added:
```
schedule==1.2.0
```

---

### 8. Migration Script

**File**: `migrations/001_auto_capture.sql`

Standalone SQL script for manual database migration (for existing deployments).

---

### 9. Test Suite

**File**: `tests/test_auto_capture.py`

| Test | Description |
|------|-------------|
| `test_enable_auto_capture` | Enable returns status=enabled |
| `test_disable_auto_capture` | Disable returns status=disabled |
| `test_auto_capture_status` | Status endpoint with event stats |
| `test_capture_event` | Event captured when enabled |
| `test_capture_event_disabled_session` | Event skipped when disabled |
| `test_mcp_enable_auto_capture` | MCP tool works |
| `test_usage_includes_auto_capture` | Usage stats show auto_capture |
| `test_command_detector` | Unit test for CommandDetector |
| `test_file_edit_detector` | Unit test for FileEditDetector |
| `test_error_detector` | Unit test for ErrorDetector |
| `test_sanitize_sensitive_data` | Unit test for sanitization |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Activity                             │
│   (commands, file edits, errors, decisions)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    REST API / MCP Tools                          │
│   POST /v1/auto-capture/event                                    │
│   tool:enable_auto_capture                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   auto_capture_events Table                      │
│   (event_type, event_data, processed=FALSE)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (every 30s)
┌─────────────────────────────────────────────────────────────────┐
│                    Background Worker                             │
│   process_pending_events()                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   AutoCaptureEngine                              │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│   │ Command     │ │ FileEdit    │ │ Error       │               │
│   │ Detector    │ │ Detector    │ │ Detector    │ ...           │
│   └─────────────┘ └─────────────┘ └─────────────┘               │
│                         │                                        │
│                         ▼                                        │
│               _sanitize_content()                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     memories Table                               │
│   (workspace='auto-capture', role='system')                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Files Changed

| File | Type | Description |
|------|------|-------------|
| `app/main.py` | Modified | Schema, endpoints, MCP tools, worker |
| `app/rate_limiter.py` | Modified | ACTION_AUTO_CAPTURE, mappings, stats |
| `app/auto_capture.py` | **New** | Engine, detectors, worker functions |
| `requirements.txt` | Modified | Added schedule==1.2.0 |
| `migrations/001_auto_capture.sql` | **New** | Migration script |
| `tests/test_auto_capture.py` | **New** | Test suite |

---

## Verification

### Run Tests
```bash
python tests/test_auto_capture.py
```

### Manual Smoke Test
```bash
# 1. Enable
curl -X POST http://localhost:8001/v1/auto-capture/enable \
  -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"session_id": "test"}'

# 2. Capture event
curl -X POST http://localhost:8001/v1/auto-capture/event \
  -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"session_id": "test", "event_type": "command", "event_data": {"command": "git commit -m test"}}'

# 3. Wait 30s for background processing

# 4. Check memories
curl http://localhost:8001/v1/memories -H "x-api-key: $API_KEY"
```

### Verify Rate Limiting
```bash
curl http://localhost:8001/v1/usage -H "x-api-key: $API_KEY"
# Should show "auto_capture" in usage stats
```

---

## Rollback Procedure

### Code
```bash
git revert <commit-hash>
```

### Database
```sql
DROP INDEX IF EXISTS idx_auto_capture_session;
DROP INDEX IF EXISTS idx_auto_capture_processed;
DROP INDEX IF EXISTS idx_auto_capture_owner;
DROP TABLE IF EXISTS auto_capture_events;
ALTER TABLE tier_definitions DROP COLUMN IF EXISTS max_auto_capture_per_day;
```

---

## Future Improvements

1. **Custom Detectors**: Allow users to register custom event detectors via API
2. **Event Batching**: Support batch event submission for efficiency
3. **Real-time Processing**: Option to process events immediately instead of background
4. **Analytics Dashboard**: UI for viewing auto-capture statistics
5. **Detector Configuration**: Per-user detector enable/disable settings
