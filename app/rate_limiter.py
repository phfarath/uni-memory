"""
Aethera Cortex Rate Limiter Module.
In-memory cache + Postgres-backed rate limiting and usage tracking.
"""

import time
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime, timezone
import threading

logger = logging.getLogger("Aethera.RateLimiter")

# Action types for categorizing API calls
ACTION_REQUEST = "request"
ACTION_MEMORY_WRITE = "memory_write"
ACTION_EMBEDDING = "embedding"
ACTION_LLM_CALL = "llm_call"
ACTION_AUTO_CAPTURE = "auto_capture"

# Endpoint to action type mapping
ENDPOINT_ACTIONS: Dict[str, Tuple[str, int]] = {
    # REST Endpoints
    "GET /v1/memories": (ACTION_REQUEST, 1),
    "POST /v1/chat/completions": (ACTION_LLM_CALL, 1),
    "DELETE /v1/memories": (ACTION_REQUEST, 1),
    "PUT /v1/memories": (ACTION_MEMORY_WRITE, 1),
    # Auto-Capture Endpoints
    "POST /v1/auto-capture/enable": (ACTION_AUTO_CAPTURE, 1),
    "POST /v1/auto-capture/disable": (ACTION_AUTO_CAPTURE, 1),
    "POST /v1/auto-capture/event": (ACTION_AUTO_CAPTURE, 1),
    "GET /v1/auto-capture/status": (ACTION_REQUEST, 1),
    # Duplicate Prevention Endpoints
    "POST /v1/memories/check-duplicate": (ACTION_EMBEDDING, 1),
    # MCP Endpoints
    "POST /mcp": (ACTION_REQUEST, 1),
    "POST /mcp/messages": (ACTION_REQUEST, 1),
    # MCP Tools (detected by tool name)
    "tool:remember": (ACTION_MEMORY_WRITE, 1),
    "tool:recall": (ACTION_EMBEDDING, 1),
    "tool:list_recent": (ACTION_REQUEST, 1),
    "tool:update_memory": (ACTION_MEMORY_WRITE, 1),
    "tool:forget": (ACTION_REQUEST, 1),
    "tool:enable_auto_capture": (ACTION_AUTO_CAPTURE, 1),
    "tool:disable_auto_capture": (ACTION_AUTO_CAPTURE, 1),
}


class RateLimiter:
    """
    In-memory cache + DB-backed rate limiter.
    Thread-safe with periodic cache refresh.
    """
    
    def __init__(self, get_db_connection_func):
        self.get_db_connection = get_db_connection_func
        self._tier_cache: Dict[str, dict] = {}
        self._usage_cache: Dict[str, Dict[str, int]] = {}  # {api_key: {action: count}}
        self._cache_date: Optional[str] = None
        self._lock = threading.Lock()
        self._refresh_tier_cache()
    
    def _refresh_tier_cache(self):
        """Load tier definitions from DB into memory."""
        try:
            conn = self.get_db_connection()
            c = conn.cursor()
            c.execute("""SELECT tier, max_requests_per_day, max_memories,
                                max_embeddings_per_day, max_llm_calls_per_day,
                                COALESCE(max_auto_capture_per_day, 1000)
                         FROM tier_definitions""")
            rows = c.fetchall()
            conn.close()

            with self._lock:
                self._tier_cache = {
                    row[0]: {
                        "max_requests": row[1],
                        "max_memories": row[2],
                        "max_embeddings": row[3],
                        "max_llm_calls": row[4],
                        "max_auto_capture": row[5]
                    }
                    for row in rows
                }
            logger.info(f"[RATE_LIMITER] Tier cache refreshed: {list(self._tier_cache.keys())}")
        except Exception as e:
            logger.error(f"[RATE_LIMITER] Failed to refresh tier cache: {e}")
    
    def _get_today(self) -> str:
        """Get today's date string for cache key."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    def _reset_cache_if_new_day(self):
        """Reset usage cache at midnight."""
        today = self._get_today()
        if self._cache_date != today:
            with self._lock:
                self._usage_cache = {}
                self._cache_date = today
                logger.info(f"[RATE_LIMITER] Usage cache reset for new day: {today}")
    
    def get_usage_from_db(self, api_key: str) -> Dict[str, int]:
        """Get today's usage counts from database."""
        today = self._get_today()
        try:
            conn = self.get_db_connection()
            c = conn.cursor()
            c.execute("""
                SELECT action_type, COUNT(*) 
                FROM usage_logs 
                WHERE api_key = %s 
                  AND DATE(created_at) = %s
                GROUP BY action_type
            """, (api_key, today))
            rows = c.fetchall()
            conn.close()
            
            return {row[0]: row[1] for row in rows}
        except Exception as e:
            logger.error(f"[RATE_LIMITER] Failed to get usage from DB: {e}")
            return {}
    
    def get_memory_count(self, api_key: str) -> int:
        """Get total memories stored for this API key (owner)."""
        try:
            conn = self.get_db_connection()
            c = conn.cursor()
            # Count memories owned by this key (multi-tenant)
            c.execute("SELECT COUNT(*) FROM memories WHERE owner_key = %s", (api_key,))
            count = c.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            logger.error(f"[RATE_LIMITER] Failed to get memory count: {e}")
            return 0
    
    def check_limit(self, api_key: str, action_type: str, tier: str) -> Tuple[bool, dict]:
        """
        Check if the request is within rate limits.
        
        Returns:
            (is_allowed, usage_info) where usage_info contains current/max values
        """
        self._reset_cache_if_new_day()
        
        # Get tier limits
        limits = self._tier_cache.get(tier, self._tier_cache.get("free", {}))
        
        # Unlimited tier (-1)
        max_value = self._get_limit_for_action(limits, action_type)
        if max_value == -1:
            return True, {"used": 0, "limit": -1, "remaining": -1, "unlimited": True}
        
        # Get current usage
        if api_key not in self._usage_cache:
            self._usage_cache[api_key] = self.get_usage_from_db(api_key)
        
        current = self._usage_cache.get(api_key, {}).get(action_type, 0)
        remaining = max(0, max_value - current)
        
        usage_info = {
            "used": current,
            "limit": max_value,
            "remaining": remaining,
            "unlimited": False
        }
        
        is_allowed = current < max_value
        
        if not is_allowed:
            logger.warning(f"[RATE_LIMITER] Limit exceeded: {api_key[:20]}... | {action_type} | {current}/{max_value}")
        
        return is_allowed, usage_info
    
    def _get_limit_for_action(self, limits: dict, action_type: str) -> int:
        """Map action type to the correct limit field."""
        mapping = {
            ACTION_REQUEST: "max_requests",
            ACTION_MEMORY_WRITE: "max_memories",
            ACTION_EMBEDDING: "max_embeddings",
            ACTION_LLM_CALL: "max_llm_calls",
            ACTION_AUTO_CAPTURE: "max_auto_capture"
        }
        return limits.get(mapping.get(action_type, "max_requests"), 100)
    
    def log_usage(self, api_key: str, endpoint: str, method: str, 
                  action_type: str, status: int, metadata: Optional[dict] = None):
        """
        Log API usage to database (fire-and-forget).
        Also updates in-memory cache.
        """
        self._reset_cache_if_new_day()
        
        # Update in-memory cache
        with self._lock:
            if api_key not in self._usage_cache:
                self._usage_cache[api_key] = {}
            self._usage_cache[api_key][action_type] = \
                self._usage_cache[api_key].get(action_type, 0) + 1
        
        # Persist to DB
        try:
            conn = self.get_db_connection()
            c = conn.cursor()
            c.execute("""
                INSERT INTO usage_logs 
                (api_key, endpoint, method, action_type, timestamp, response_status, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (api_key, endpoint, method, action_type, time.time(), status, 
                  None if not metadata else str(metadata)))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"[RATE_LIMITER] Failed to log usage: {e}")
    
    def get_full_usage_stats(self, api_key: str, tier: str) -> dict:
        """Get complete usage stats for /v1/usage endpoint."""
        self._reset_cache_if_new_day()
        
        limits = self._tier_cache.get(tier, self._tier_cache.get("free", {}))
        usage = self.get_usage_from_db(api_key)
        memory_count = self.get_memory_count(api_key)
        
        def stat(action: str, limit_key: str):
            max_val = limits.get(limit_key, 0)
            used = usage.get(action, 0)
            if max_val == -1:
                return {"used": used, "limit": "unlimited", "remaining": "unlimited"}
            return {"used": used, "limit": max_val, "remaining": max(0, max_val - used)}
        
        # Calculate reset time (next midnight UTC)
        now = datetime.now(timezone.utc)
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if now.hour >= 0:
            from datetime import timedelta
            tomorrow = tomorrow + timedelta(days=1)
        
        return {
            "tier": tier,
            "period": self._get_today(),
            "usage": {
                "requests": stat(ACTION_REQUEST, "max_requests"),
                "memories": {
                    "stored": memory_count,
                    "limit": limits.get("max_memories", 1000) if limits.get("max_memories", 0) != -1 else "unlimited"
                },
                "embeddings": stat(ACTION_EMBEDDING, "max_embeddings"),
                "llm_calls": stat(ACTION_LLM_CALL, "max_llm_calls"),
                "auto_capture": stat(ACTION_AUTO_CAPTURE, "max_auto_capture")
            },
            "reset_at": tomorrow.isoformat()
        }


async def cleanup_old_logs(get_db_connection_func, days: int = 90):
    """
    Delete usage logs older than specified days.
    Should be run as a scheduled job (daily at 3 AM).
    """
    try:
        conn = get_db_connection_func()
        c = conn.cursor()
        c.execute(f"DELETE FROM usage_logs WHERE created_at < NOW() - INTERVAL '{days} days'")
        deleted = c.rowcount
        conn.commit()
        conn.close()
        logger.info(f"[CLEANUP] Deleted {deleted} usage logs older than {days} days")
        return deleted
    except Exception as e:
        logger.error(f"[CLEANUP] Failed to cleanup old logs: {e}")
        return 0
