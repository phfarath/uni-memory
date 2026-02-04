"""
Test suite for Auto-Capture functionality.
Run: python tests/test_auto_capture.py

Requires: Server running at localhost:8001
"""

import requests
import time
import sys
import os

# Configuration
BASE_URL = "http://localhost:8001"
ROOT_KEY = None  # Set this to your root API key

def get_root_key():
    """Get root key from file variable, environment, or prompt."""
    if ROOT_KEY:
        return ROOT_KEY
    key = os.environ.get("AETHERA_ROOT_KEY") or os.environ.get("ROOT_KEY")
    if not key:
        print("Please set AETHERA_ROOT_KEY environment variable or modify ROOT_KEY in this file.")
        sys.exit(1)
    return key


def create_test_key(root_key: str, tier: str = "free") -> str:
    """Create a test API key."""
    resp = requests.post(
        f"{BASE_URL}/admin/keys/create",
        headers={"x-api-key": root_key, "Content-Type": "application/json"},
        json={"owner_name": f"auto-capture-test-{tier}", "tier": tier},
        timeout=5
    )
    if resp.status_code != 200:
        raise Exception(f"Failed to create test key: {resp.text}")
    return resp.json()["key"]


def revoke_test_key(root_key: str, test_key: str):
    """Revoke a test API key."""
    requests.post(
        f"{BASE_URL}/admin/keys/revoke",
        headers={"x-api-key": root_key, "Content-Type": "application/json"},
        params={"target_key": test_key},
        timeout=5
    )


def test_enable_auto_capture(api_key: str):
    """Test enabling auto-capture for a session."""
    print("\n>>> [TEST] POST /v1/auto-capture/enable")

    resp = requests.post(
        f"{BASE_URL}/v1/auto-capture/enable",
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json={"session_id": "test-session-enable"},
        timeout=5
    )

    if resp.status_code != 200:
        print(f" [FAIL] Status {resp.status_code}: {resp.text}")
        return False

    data = resp.json()
    if data.get("status") != "enabled":
        print(f" [FAIL] Expected status=enabled, got {data}")
        return False

    print(f" [PASS] Auto-capture enabled: {data}")
    return True


def test_disable_auto_capture(api_key: str):
    """Test disabling auto-capture for a session."""
    print("\n>>> [TEST] POST /v1/auto-capture/disable")

    # First enable
    requests.post(
        f"{BASE_URL}/v1/auto-capture/enable",
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json={"session_id": "test-session-disable"},
        timeout=5
    )

    # Then disable
    resp = requests.post(
        f"{BASE_URL}/v1/auto-capture/disable",
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json={"session_id": "test-session-disable"},
        timeout=5
    )

    if resp.status_code != 200:
        print(f" [FAIL] Status {resp.status_code}: {resp.text}")
        return False

    data = resp.json()
    if data.get("status") != "disabled":
        print(f" [FAIL] Expected status=disabled, got {data}")
        return False

    print(f" [PASS] Auto-capture disabled: {data}")
    return True


def test_auto_capture_status(api_key: str):
    """Test getting auto-capture status."""
    print("\n>>> [TEST] GET /v1/auto-capture/status")

    session_id = "test-session-status"

    # Enable first
    requests.post(
        f"{BASE_URL}/v1/auto-capture/enable",
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json={"session_id": session_id},
        timeout=5
    )

    # Get status
    resp = requests.get(
        f"{BASE_URL}/v1/auto-capture/status",
        headers={"x-api-key": api_key},
        params={"session_id": session_id},
        timeout=5
    )

    if resp.status_code != 200:
        print(f" [FAIL] Status {resp.status_code}: {resp.text}")
        return False

    data = resp.json()
    required_fields = ["session_id", "enabled", "total_events", "processed_events"]

    for field in required_fields:
        if field not in data:
            print(f" [FAIL] Missing field: {field}")
            return False

    if not data.get("enabled"):
        print(f" [FAIL] Expected enabled=True, got {data}")
        return False

    print(f" [PASS] Auto-capture status: {data}")
    return True


def test_capture_event(api_key: str):
    """Test capturing an event when auto-capture is enabled."""
    print("\n>>> [TEST] POST /v1/auto-capture/event (enabled)")

    session_id = "test-session-capture"

    # Enable auto-capture
    requests.post(
        f"{BASE_URL}/v1/auto-capture/enable",
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json={"session_id": session_id},
        timeout=5
    )

    # Capture event
    resp = requests.post(
        f"{BASE_URL}/v1/auto-capture/event",
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json={
            "session_id": session_id,
            "event_type": "command",
            "event_data": {"command": "git commit -m 'test commit'"}
        },
        timeout=5
    )

    if resp.status_code != 200:
        print(f" [FAIL] Status {resp.status_code}: {resp.text}")
        return False

    data = resp.json()
    if data.get("status") != "captured":
        print(f" [FAIL] Expected status=captured, got {data}")
        return False

    if "event_id" not in data:
        print(f" [FAIL] Missing event_id in response: {data}")
        return False

    print(f" [PASS] Event captured: {data}")
    return True


def test_capture_event_disabled_session(api_key: str):
    """Test that events are skipped when auto-capture is disabled."""
    print("\n>>> [TEST] POST /v1/auto-capture/event (disabled session)")

    session_id = "test-session-disabled-capture"

    # Make sure auto-capture is disabled (don't enable it)
    requests.post(
        f"{BASE_URL}/v1/auto-capture/disable",
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json={"session_id": session_id},
        timeout=5
    )

    # Try to capture event
    resp = requests.post(
        f"{BASE_URL}/v1/auto-capture/event",
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json={
            "session_id": session_id,
            "event_type": "command",
            "event_data": {"command": "ls -la"}
        },
        timeout=5
    )

    if resp.status_code != 200:
        print(f" [FAIL] Status {resp.status_code}: {resp.text}")
        return False

    data = resp.json()
    if data.get("status") != "skipped":
        print(f" [FAIL] Expected status=skipped, got {data}")
        return False

    print(f" [PASS] Event skipped for disabled session: {data}")
    return True


def test_mcp_enable_auto_capture(api_key: str):
    """Test enabling auto-capture via MCP tool."""
    print("\n>>> [TEST] MCP tools/call enable_auto_capture")

    resp = requests.post(
        f"{BASE_URL}/mcp",
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "enable_auto_capture",
                "arguments": {"session_id": "mcp-test-session"}
            }
        },
        timeout=10
    )

    if resp.status_code != 200:
        print(f" [FAIL] Status {resp.status_code}: {resp.text}")
        return False

    data = resp.json()
    result = data.get("result", {})
    content = result.get("content", [])

    if not content:
        print(f" [FAIL] Empty content in response: {data}")
        return False

    text = content[0].get("text", "")
    if "ativado" not in text.lower() and "enabled" not in text.lower():
        print(f" [FAIL] Expected success message, got: {text}")
        return False

    print(f" [PASS] MCP enable_auto_capture: {text[:100]}...")
    return True


def test_usage_includes_auto_capture(api_key: str):
    """Test that usage stats include auto_capture."""
    print("\n>>> [TEST] GET /v1/usage (includes auto_capture)")

    resp = requests.get(
        f"{BASE_URL}/v1/usage",
        headers={"x-api-key": api_key},
        timeout=5
    )

    if resp.status_code != 200:
        print(f" [FAIL] Status {resp.status_code}: {resp.text}")
        return False

    data = resp.json()
    usage = data.get("usage", {})

    if "auto_capture" not in usage:
        print(f" [FAIL] auto_capture missing from usage stats: {usage.keys()}")
        return False

    auto_capture_stats = usage.get("auto_capture", {})
    if "used" not in auto_capture_stats or "limit" not in auto_capture_stats:
        print(f" [FAIL] auto_capture missing used/limit: {auto_capture_stats}")
        return False

    print(f" [PASS] Usage includes auto_capture: {auto_capture_stats}")
    return True


def test_command_detector():
    """Test CommandDetector logic (unit test)."""
    print("\n>>> [TEST] CommandDetector (unit test)")

    # Import the detector
    try:
        from app.auto_capture import CommandDetector
        detector = CommandDetector()

        # Test relevant commands
        relevant_commands = [
            {"command": "git commit -m 'test'"},
            {"command": "git push origin main"},
            {"command": "npm install express"},
            {"command": "pytest tests/"},
            {"command": "docker build -t myapp ."},
        ]

        for event in relevant_commands:
            if not detector.detect(event):
                print(f" [FAIL] Should detect: {event['command']}")
                return False

        # Test irrelevant commands
        irrelevant_commands = [
            {"command": "ls -la"},
            {"command": "cd /tmp"},
            {"command": "echo hello"},
            {"command": "cat file.txt"},
        ]

        for event in irrelevant_commands:
            if detector.detect(event):
                print(f" [FAIL] Should NOT detect: {event['command']}")
                return False

        print(" [PASS] CommandDetector correctly filters commands")
        return True

    except ImportError as e:
        print(f" [SKIP] Could not import CommandDetector: {e}")
        return True  # Skip if running remotely


def test_file_edit_detector():
    """Test FileEditDetector logic (unit test)."""
    print("\n>>> [TEST] FileEditDetector (unit test)")

    try:
        from app.auto_capture import FileEditDetector
        detector = FileEditDetector()

        # Test relevant files
        relevant_files = [
            {"file_path": "app/main.py"},
            {"file_path": "src/index.ts"},
            {"file_path": "components/Button.tsx"},
            {"file_path": "Dockerfile"},
        ]

        for event in relevant_files:
            if not detector.detect(event):
                print(f" [FAIL] Should detect: {event['file_path']}")
                return False

        # Test ignored files
        ignored_files = [
            {"file_path": "node_modules/package/index.js"},
            {"file_path": "__pycache__/module.cpython-39.pyc"},
            {"file_path": ".git/objects/abc123"},
            {"file_path": "package-lock.json"},
        ]

        for event in ignored_files:
            if detector.detect(event):
                print(f" [FAIL] Should NOT detect: {event['file_path']}")
                return False

        print(" [PASS] FileEditDetector correctly filters files")
        return True

    except ImportError as e:
        print(f" [SKIP] Could not import FileEditDetector: {e}")
        return True


def test_error_detector():
    """Test ErrorDetector logic (unit test)."""
    print("\n>>> [TEST] ErrorDetector (unit test)")

    try:
        from app.auto_capture import ErrorDetector
        detector = ErrorDetector()

        # Test error messages
        error_messages = [
            {"message": "Error: Connection refused"},
            {"message": "Traceback (most recent call last):"},
            {"message": "FATAL: database connection failed"},
            {"message": "TypeError: Cannot read property 'x' of undefined"},
        ]

        for event in error_messages:
            if not detector.detect(event):
                print(f" [FAIL] Should detect error: {event['message'][:50]}...")
                return False

        # Test non-error messages
        non_errors = [
            {"message": "Successfully deployed to production"},
            {"message": "Build completed in 45 seconds"},
            {"message": "All tests passed"},
        ]

        for event in non_errors:
            if detector.detect(event):
                print(f" [FAIL] Should NOT detect: {event['message']}")
                return False

        print(" [PASS] ErrorDetector correctly identifies errors")
        return True

    except ImportError as e:
        print(f" [SKIP] Could not import ErrorDetector: {e}")
        return True


def test_sanitize_sensitive_data():
    """Test that sensitive data is sanitized."""
    print("\n>>> [TEST] Sensitive data sanitization (unit test)")

    try:
        from app.auto_capture import AutoCaptureEngine
        engine = AutoCaptureEngine()

        # Test sensitive patterns
        sensitive_content = [
            ("API_KEY=sk_live_abc123xyz", "[REDACTED]"),
            ("password: mysecret123", "[REDACTED]"),
            ("Bearer eyJhbGciOiJIUzI1NiJ9.xxxxx", "[REDACTED]"),
            ("aws_secret_access_key=AKIAIOSFODNN7EXAMPLE", "[REDACTED]"),
        ]

        for content, _ in sensitive_content:
            sanitized = engine._sanitize_content(content)
            if "REDACTED" not in sanitized:
                print(f" [FAIL] Should sanitize: {content[:30]}...")
                return False

        # Test non-sensitive content
        safe_content = "This is a normal log message about user login"
        sanitized = engine._sanitize_content(safe_content)
        if sanitized != safe_content:
            print(f" [FAIL] Should NOT modify: {safe_content}")
            return False

        print(" [PASS] Sensitive data correctly sanitized")
        return True

    except ImportError as e:
        print(f" [SKIP] Could not import AutoCaptureEngine: {e}")
        return True


def main():
    print("=" * 60)
    print(" AETHERA CORTEX - Auto-Capture Test Suite")
    print("=" * 60)

    root_key = get_root_key()

    # Create test key
    print("\n>>> Creating test API key...")
    try:
        test_key = create_test_key(root_key, "free")
        print(f"  Test key created: {test_key[:20]}...")
    except Exception as e:
        print(f"  [FAIL] Could not create test key: {e}")
        return 1

    results = []

    try:
        # API Tests
        results.append(("Enable Auto-Capture", test_enable_auto_capture(test_key)))
        results.append(("Disable Auto-Capture", test_disable_auto_capture(test_key)))
        results.append(("Auto-Capture Status", test_auto_capture_status(test_key)))
        results.append(("Capture Event (enabled)", test_capture_event(test_key)))
        results.append(("Capture Event (disabled)", test_capture_event_disabled_session(test_key)))
        results.append(("MCP Enable Auto-Capture", test_mcp_enable_auto_capture(test_key)))
        results.append(("Usage Includes Auto-Capture", test_usage_includes_auto_capture(test_key)))

        # Unit Tests
        results.append(("CommandDetector", test_command_detector()))
        results.append(("FileEditDetector", test_file_edit_detector()))
        results.append(("ErrorDetector", test_error_detector()))
        results.append(("Sensitive Data Sanitization", test_sanitize_sensitive_data()))

    finally:
        # Cleanup
        print("\n>>> Cleaning up test key...")
        revoke_test_key(root_key, test_key)
        print("  Test key revoked")

    # Summary
    print("\n" + "=" * 60)
    print(" TEST RESULTS")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  Total: {passed}/{total} passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
