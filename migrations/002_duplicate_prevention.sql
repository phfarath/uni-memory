-- Migration: Duplicate Prevention
-- Date: 2026-02-10
-- Description: Adds HNSW vector index for efficient duplicate detection via cosine similarity

-- 1. Create HNSW vector index for cosine similarity searches
-- Used by check_duplicate() in app/duplicate_prevention.py
-- HNSW provides better recall than IVFFlat and works with any number of rows
CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw
    ON memories USING hnsw (embedding vector_cosine_ops);

-- Rollback script (run manually if needed):
-- DROP INDEX IF EXISTS idx_memories_embedding_hnsw;
