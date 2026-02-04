-- Migration: Auto-Capture Events Table
-- Date: 2026-02-03
-- Description: Adds auto-capture functionality for automatic context capture

-- 1. Create auto_capture_events table
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

-- 2. Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_auto_capture_session
    ON auto_capture_events(session_id, captured_at DESC);

CREATE INDEX IF NOT EXISTS idx_auto_capture_processed
    ON auto_capture_events(processed, captured_at);

CREATE INDEX IF NOT EXISTS idx_auto_capture_owner
    ON auto_capture_events(owner_key);

-- 3. Add max_auto_capture_per_day column to tier_definitions
ALTER TABLE tier_definitions
    ADD COLUMN IF NOT EXISTS max_auto_capture_per_day INTEGER DEFAULT 1000;

-- 4. Update tier limits for auto-capture
UPDATE tier_definitions SET max_auto_capture_per_day = 100 WHERE tier = 'free';
UPDATE tier_definitions SET max_auto_capture_per_day = 5000 WHERE tier = 'pro';
UPDATE tier_definitions SET max_auto_capture_per_day = 50000 WHERE tier = 'team';
UPDATE tier_definitions SET max_auto_capture_per_day = -1 WHERE tier = 'root';

-- Rollback script (run manually if needed):
-- DROP INDEX IF EXISTS idx_auto_capture_session;
-- DROP INDEX IF EXISTS idx_auto_capture_processed;
-- DROP INDEX IF EXISTS idx_auto_capture_owner;
-- DROP TABLE IF EXISTS auto_capture_events;
-- ALTER TABLE tier_definitions DROP COLUMN IF EXISTS max_auto_capture_per_day;
