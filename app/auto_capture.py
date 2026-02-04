"""
Aethera Cortex Auto-Capture Module.
Automatic context capture from user activities (commands, file edits, errors, decisions).
"""

import re
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set
from datetime import datetime, timezone

logger = logging.getLogger("Aethera.AutoCapture")

# Sensitive data patterns to filter out before persistence
SENSITIVE_PATTERNS = [
    r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']?[\w\-]+["\']?',
    r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']?[^\s"\']+["\']?',
    r'(?i)(secret|token)\s*[=:]\s*["\']?[\w\-]+["\']?',
    r'(?i)(bearer|authorization)\s+[\w\-\.]+',
    r'sk_[a-zA-Z0-9_\-]{20,}',  # API keys like sk_...
    r'(?i)-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----',
    r'(?i)(aws_access_key_id|aws_secret_access_key)\s*[=:]\s*[\w\-]+',
]


class EventDetector(ABC):
    """Abstract base class for event detectors."""

    @abstractmethod
    def detect(self, event_data: dict) -> bool:
        """Returns True if the event is relevant for capture."""
        pass

    @abstractmethod
    def format_memory(self, event_data: dict) -> str:
        """Formats the event as a memory content string."""
        pass

    @abstractmethod
    def event_type(self) -> str:
        """Returns the event type identifier."""
        pass


class CommandDetector(EventDetector):
    """Detects relevant shell commands executed by the user."""

    RELEVANT_COMMANDS = [
        'git commit', 'git push', 'git merge', 'git pull', 'git checkout',
        'git branch', 'git rebase', 'git stash', 'git tag',
        'npm install', 'npm run', 'npm test', 'npm build',
        'pip install', 'pip uninstall',
        'pytest', 'python -m pytest', 'cargo test', 'go test',
        'docker build', 'docker run', 'docker-compose',
        'kubectl apply', 'kubectl delete', 'kubectl create',
        'terraform apply', 'terraform plan', 'terraform destroy',
        'make', 'cmake', 'cargo build', 'go build',
        'yarn install', 'yarn add', 'yarn test',
        'pnpm install', 'pnpm add',
        'mvn install', 'mvn test', 'gradle build', 'gradle test',
    ]

    def detect(self, event_data: dict) -> bool:
        command = event_data.get('command', '')
        if not command:
            return False
        return any(cmd in command.lower() for cmd in self.RELEVANT_COMMANDS)

    def format_memory(self, event_data: dict) -> str:
        command = event_data.get('command', '')
        output = event_data.get('output', '')
        exit_code = event_data.get('exit_code', 0)

        memory = f"[COMMAND] Executed: {command}"
        if exit_code != 0:
            memory += f" (exit code: {exit_code})"
        if output and len(output) < 200:
            memory += f"\nOutput: {output}"
        return memory

    def event_type(self) -> str:
        return "command"


class FileEditDetector(EventDetector):
    """Detects when important files are edited."""

    RELEVANT_EXTENSIONS = [
        '.py', '.js', '.ts', '.tsx', '.jsx',
        '.go', '.rs', '.java', '.kt', '.scala',
        '.c', '.cpp', '.h', '.hpp',
        '.rb', '.php', '.swift', '.m',
        '.sql', '.graphql', '.proto',
        '.yaml', '.yml', '.json', '.toml',
        '.dockerfile', '.tf', '.tfvars',
        '.sh', '.bash', '.zsh',
    ]

    IGNORE_PATTERNS = [
        'node_modules/', '__pycache__/', '.git/',
        'dist/', 'build/', 'target/', '.next/',
        'venv/', '.venv/', 'env/', '.env/',
        '.pytest_cache/', '.mypy_cache/',
        'coverage/', '.coverage', 'htmlcov/',
        '*.min.js', '*.min.css', '*.map',
        'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
        'Cargo.lock', 'poetry.lock', 'Pipfile.lock',
    ]

    def detect(self, event_data: dict) -> bool:
        file_path = event_data.get('file_path', '')
        if not file_path:
            return False

        # Ignore files in irrelevant directories
        if any(pattern in file_path for pattern in self.IGNORE_PATTERNS):
            return False

        # Check if extension is relevant
        return any(file_path.lower().endswith(ext) for ext in self.RELEVANT_EXTENSIONS)

    def format_memory(self, event_data: dict) -> str:
        file_path = event_data.get('file_path', '')
        lines_added = event_data.get('lines_added', 0)
        lines_removed = event_data.get('lines_removed', 0)
        description = event_data.get('description', '')

        memory = f"[FILE_EDIT] Modified: {file_path}"
        if lines_added or lines_removed:
            memory += f" (+{lines_added} -{lines_removed} lines)"
        if description:
            memory += f"\nDescription: {description}"
        return memory

    def event_type(self) -> str:
        return "file_edit"


class ErrorDetector(EventDetector):
    """Detects errors and their solutions."""

    ERROR_KEYWORDS = [
        'error:', 'error :', 'exception:', 'exception :',
        'traceback', 'stack trace', 'stacktrace',
        'failed', 'failure', 'fatal',
        'segfault', 'segmentation fault',
        'panic:', 'panicked at',
        'undefined reference', 'unresolved',
        'syntax error', 'typeerror', 'nameerror',
        'importerror', 'modulenotfounderror',
        'connectionerror', 'timeout',
        'permission denied', 'access denied',
        'not found', '404', '500', '502', '503',
    ]

    def detect(self, event_data: dict) -> bool:
        message = event_data.get('message', '').lower()
        if not message:
            return False
        return any(kw in message for kw in self.ERROR_KEYWORDS)

    def format_memory(self, event_data: dict) -> str:
        error_msg = event_data.get('message', '')
        solution = event_data.get('solution', '')
        context = event_data.get('context', '')

        memory = f"[ERROR] {error_msg[:500]}"
        if context:
            memory += f"\nContext: {context}"
        if solution:
            memory += f"\nSolution: {solution}"
        return memory

    def event_type(self) -> str:
        return "error"


class DecisionDetector(EventDetector):
    """Detects architectural and design decisions."""

    DECISION_KEYWORDS = [
        # English
        "i'll use", "we'll use", "going to use", "decided to use",
        "let's use", "should use", "will implement",
        "architecture", "design pattern", "strategy",
        "migrate to", "switch to", "replace with",
        "approach", "solution", "trade-off", "tradeoff",
        # Portuguese
        'vou usar', 'vamos usar', 'decidimos usar', 'escolhemos',
        'arquitetura', 'padrao de design', 'estrategia',
        'migrar para', 'trocar para', 'substituir por',
        'abordagem', 'solucao', 'decisao',
    ]

    def detect(self, event_data: dict) -> bool:
        message = event_data.get('message', '').lower()
        if not message:
            return False
        return any(kw in message for kw in self.DECISION_KEYWORDS)

    def format_memory(self, event_data: dict) -> str:
        message = event_data.get('message', '')
        rationale = event_data.get('rationale', '')
        alternatives = event_data.get('alternatives', [])

        memory = f"[DECISION] {message}"
        if rationale:
            memory += f"\nRationale: {rationale}"
        if alternatives:
            memory += f"\nAlternatives considered: {', '.join(alternatives)}"
        return memory

    def event_type(self) -> str:
        return "decision"


class AutoCaptureEngine:
    """
    Engine for automatic context capture.

    Manages event detectors, session state, and event processing.
    """

    def __init__(self):
        self.detectors: List[EventDetector] = []
        self.enabled_sessions: Dict[str, str] = {}  # {session_id: owner_key}
        self._sensitive_patterns = [re.compile(p) for p in SENSITIVE_PATTERNS]

        # Register default detectors
        self.register_detector(CommandDetector())
        self.register_detector(FileEditDetector())
        self.register_detector(ErrorDetector())
        self.register_detector(DecisionDetector())

        logger.info("[AUTO_CAPTURE] Engine initialized with default detectors")

    def register_detector(self, detector: EventDetector):
        """Register a new event detector."""
        self.detectors.append(detector)
        logger.info(f"[AUTO_CAPTURE] Registered detector: {detector.event_type()}")

    def enable_for_session(self, session_id: str, owner_key: str):
        """Enable auto-capture for a session."""
        self.enabled_sessions[session_id] = owner_key
        logger.info(f"[AUTO_CAPTURE] Enabled for session: {session_id}")

    def disable_for_session(self, session_id: str):
        """Disable auto-capture for a session."""
        if session_id in self.enabled_sessions:
            del self.enabled_sessions[session_id]
            logger.info(f"[AUTO_CAPTURE] Disabled for session: {session_id}")

    def is_enabled(self, session_id: str) -> bool:
        """Check if auto-capture is enabled for a session."""
        return session_id in self.enabled_sessions

    def get_owner_key(self, session_id: str) -> Optional[str]:
        """Get the owner key for a session."""
        return self.enabled_sessions.get(session_id)

    def process_event(self, event_data: dict) -> Optional[str]:
        """
        Process an event and decide if it should become a memory.

        Args:
            event_data: Dict containing 'type' and 'data' keys

        Returns:
            Memory content string if relevant, None otherwise
        """
        event_type = event_data.get('type', '')
        data = event_data.get('data', {})

        # Find matching detector
        for detector in self.detectors:
            if detector.event_type() == event_type:
                if detector.detect(data):
                    memory_content = detector.format_memory(data)
                    # Sanitize before returning
                    return self._sanitize_content(memory_content)
                return None

        # If no specific detector matches, try all detectors
        for detector in self.detectors:
            if detector.detect(data):
                memory_content = detector.format_memory(data)
                return self._sanitize_content(memory_content)

        return None

    def _sanitize_content(self, content: str) -> str:
        """Remove sensitive data from content before persistence."""
        sanitized = content
        for pattern in self._sensitive_patterns:
            sanitized = pattern.sub('[REDACTED]', sanitized)
        return sanitized


# Global singleton instance
auto_capture_engine = AutoCaptureEngine()


def process_pending_events(get_db_connection_func, add_memory_func):
    """
    Process pending auto-capture events and create memories.
    Should be called every 30 seconds by the scheduler.

    Args:
        get_db_connection_func: Function to get database connection
        add_memory_func: Function to add memory (add_memory_trace_logic)
    """
    try:
        conn = get_db_connection_func()
        c = conn.cursor()

        # Fetch unprocessed events (batch of 100)
        c.execute("""
            SELECT id, owner_key, session_id, event_type, event_data
            FROM auto_capture_events
            WHERE processed = FALSE
            ORDER BY captured_at ASC
            LIMIT 100
        """)

        events = c.fetchall()

        if not events:
            conn.close()
            return

        processed_count = 0
        memory_count = 0

        for event in events:
            event_id, owner_key, session_id, event_type, event_data = event

            # Parse event_data if it's a string
            if isinstance(event_data, str):
                try:
                    event_data = json.loads(event_data)
                except json.JSONDecodeError:
                    event_data = {}

            # Process event through the engine
            memory_content = auto_capture_engine.process_event({
                'type': event_type,
                'data': event_data
            })

            memory_id = None
            if memory_content:
                try:
                    # Add memory and get the ID
                    add_memory_func(owner_key, session_id, 'system', memory_content, 'auto-capture')
                    memory_count += 1

                    # Get the memory ID (last inserted)
                    c.execute("SELECT id FROM memories WHERE owner_key = %s ORDER BY id DESC LIMIT 1", (owner_key,))
                    row = c.fetchone()
                    if row:
                        memory_id = row[0]
                except Exception as e:
                    logger.error(f"[AUTO_CAPTURE] Failed to create memory: {e}")

            # Mark event as processed
            c.execute("""
                UPDATE auto_capture_events
                SET processed = TRUE, memory_id = %s
                WHERE id = %s
            """, (memory_id, event_id))

            processed_count += 1

        conn.commit()
        conn.close()

        logger.info(f"[AUTO_CAPTURE] Processed {processed_count} events, created {memory_count} memories")

    except Exception as e:
        logger.error(f"[AUTO_CAPTURE] Error processing events: {e}")


def cleanup_old_auto_capture_events(get_db_connection_func, days: int = 30):
    """
    Remove processed events older than specified days.
    Should be run weekly (e.g., Sunday at 3 AM).

    Args:
        get_db_connection_func: Function to get database connection
        days: Number of days to keep processed events (default: 30)
    """
    try:
        conn = get_db_connection_func()
        c = conn.cursor()

        c.execute(f"""
            DELETE FROM auto_capture_events
            WHERE processed = TRUE
            AND captured_at < NOW() - INTERVAL '{days} days'
        """)

        deleted = c.rowcount
        conn.commit()
        conn.close()

        logger.info(f"[AUTO_CAPTURE] Cleanup: {deleted} old events removed")
        return deleted

    except Exception as e:
        logger.error(f"[AUTO_CAPTURE] Cleanup error: {e}")
        return 0
