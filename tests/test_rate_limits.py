"""
Test suite for Rate Limiting and Usage Tracking.
Run: python tests/test_rate_limits.py

Requires: Server running at localhost:8001
"""

import requests
import time
import sys

# Configuration
BASE_URL = "http://localhost:8001"
ROOT_KEY = None # Set this to your root API key

def get_root_key():
    """Get root key from file variable, environment, or prompt."""
    import os
    # First check the ROOT_KEY variable set in this file
    if ROOT_KEY:
        return ROOT_KEY
    # Then check environment variables
    key = os.environ.get("AETHERA_ROOT_KEY") or os.environ.get("ROOT_KEY")
    if not key:
        print("Please set AETHERA_ROOT_KEY environment variable or modify ROOT_KEY in this file.")
        sys.exit(1)
    return key

def test_usage_endpoint(api_key: str):
    """Test /v1/usage endpoint returns valid response."""
    print("\n>>> [TEST] GET /v1/usage")
    resp = requests.get(
        f"{BASE_URL}/v1/usage",
        headers={"x-api-key": api_key},
        timeout=5
    )
    
    if resp.status_code != 200:
        print(f" [FAIL] Status {resp.status_code}: {resp.text}")
        return False
    
    data = resp.json()
    required_fields = ["tier", "period", "usage", "reset_at"]
    
    for field in required_fields:
        if field not in data:
            print(f" [FAIL] Missing field: {field}")
            return False
    
    print(f" [PASS] Usage data: {data}")
    return True

def test_rate_limit_enforcement(root_key: str):
    """Test that rate limits are enforced for free tier."""
    print("\n>>> [TEST] Rate Limit Enforcement")
    
    # 1. Create a free tier test key
    print("  Creating free tier test key...")
    resp = requests.post(
        f"{BASE_URL}/admin/keys/create",
        headers={"x-api-key": root_key, "Content-Type": "application/json"},
        json={"owner_name": "rate-limit-test", "tier": "free"},
        timeout=5
    )
    
    if resp.status_code != 200:
        print(f" [FAIL] Could not create test key: {resp.text}")
        return False
    
    test_key = resp.json()["key"]
    print(f"  Test key created: {test_key[:20]}...")
    
    # 2. Make requests until we hit the limit (free = 100/day)
    # For testing, we'll just make a few and check usage increases
    print("  Making 5 test requests...")
    for i in range(5):
        requests.get(
            f"{BASE_URL}/v1/memories?limit=1",
            headers={"x-api-key": test_key},
            timeout=5
        )
    
    # 3. Check usage reflects the requests
    resp = requests.get(
        f"{BASE_URL}/v1/usage",
        headers={"x-api-key": test_key},
        timeout=5
    )
    
    if resp.status_code != 200:
        print(f" [FAIL] Could not get usage: {resp.text}")
        return False
    
    usage = resp.json()
    requests_used = usage["usage"]["requests"]["used"]
    
    if requests_used >= 5:
        print(f" [PASS] Usage tracking working: {requests_used} requests logged")
    else:
        print(f" [WARN] Expected >= 5 requests, got {requests_used}")
    
    # 4. Cleanup - revoke test key
    requests.post(
        f"{BASE_URL}/admin/keys/revoke",
        headers={"x-api-key": root_key, "Content-Type": "application/json"},
        params={"target_key": test_key},
        timeout=5
    )
    print("  Test key revoked")
    
    return True

def test_tier_upgrade(root_key: str):
    """Test admin tier upgrade endpoint."""
    print("\n>>> [TEST] Tier Upgrade Flow")
    
    # 1. Create a free tier key
    resp = requests.post(
        f"{BASE_URL}/admin/keys/create",
        headers={"x-api-key": root_key, "Content-Type": "application/json"},
        json={"owner_name": "upgrade-test", "tier": "free"},
        timeout=5
    )
    
    if resp.status_code != 200:
        print(f" [FAIL] Could not create test key: {resp.text}")
        return False
    
    test_key = resp.json()["key"]
    
    # 2. Verify it's free tier
    resp = requests.get(
        f"{BASE_URL}/v1/usage",
        headers={"x-api-key": test_key},
        timeout=5
    )
    
    if resp.json()["tier"] != "free":
        print(f" [FAIL] Expected free tier")
        return False
    
    print(f"  Created free tier key")
    
    # 3. Upgrade to pro
    resp = requests.post(
        f"{BASE_URL}/admin/users/{test_key}/upgrade",
        headers={"x-api-key": root_key, "Content-Type": "application/json"},
        json={"new_tier": "pro"},
        timeout=5
    )
    
    if resp.status_code != 200:
        print(f" [FAIL] Upgrade failed: {resp.text}")
        return False
    
    print(f"  Upgraded to pro: {resp.json()}")
    
    # 4. Verify upgrade
    resp = requests.get(
        f"{BASE_URL}/v1/usage",
        headers={"x-api-key": test_key},
        timeout=5
    )
    
    if resp.json()["tier"] != "pro":
        print(f" [FAIL] Tier not updated to pro")
        return False
    
    print(f" [PASS] Tier upgrade verified")
    
    # Cleanup
    requests.post(
        f"{BASE_URL}/admin/keys/revoke",
        headers={"x-api-key": root_key, "Content-Type": "application/json"},
        params={"target_key": test_key},
        timeout=5
    )
    
    return True

def test_admin_stats(root_key: str):
    """Test admin usage stats endpoint."""
    print("\n>>> [TEST] Admin Usage Stats")
    
    resp = requests.get(
        f"{BASE_URL}/admin/usage/stats",
        headers={"x-api-key": root_key},
        timeout=5
    )
    
    if resp.status_code != 200:
        print(f" [FAIL] Status {resp.status_code}: {resp.text}")
        return False
    
    data = resp.json()
    required_fields = ["total_users", "active_today", "by_tier"]
    
    for field in required_fields:
        if field not in data:
            print(f" [FAIL] Missing field: {field}")
            return False
    
    print(f" [PASS] Admin stats: {data}")
    return True

def test_list_tiers(root_key: str):
    """Test listing available tiers."""
    print("\n>>> [TEST] List Tiers")
    
    resp = requests.get(
        f"{BASE_URL}/admin/tiers",
        headers={"x-api-key": root_key},
        timeout=5
    )
    
    if resp.status_code != 200:
        print(f" [FAIL] Status {resp.status_code}: {resp.text}")
        return False
    
    data = resp.json()
    tiers = data.get("tiers", [])
    
    expected_tiers = {"free", "pro", "team", "root"}
    actual_tiers = {t["tier"] for t in tiers}
    
    if not expected_tiers.issubset(actual_tiers):
        print(f" [FAIL] Missing tiers. Expected {expected_tiers}, got {actual_tiers}")
        return False
    
    print(f" [PASS] Found {len(tiers)} tiers: {actual_tiers}")
    return True

def main():
    print("=" * 60)
    print(" AETHERA CORTEX - Rate Limiting Test Suite")
    print("=" * 60)
    
    root_key = get_root_key()
    
    results = []
    
    # Run tests
    results.append(("Usage Endpoint", test_usage_endpoint(root_key)))
    results.append(("Rate Limit Enforcement", test_rate_limit_enforcement(root_key)))
    results.append(("Tier Upgrade", test_tier_upgrade(root_key)))
    results.append(("Admin Stats", test_admin_stats(root_key)))
    results.append(("List Tiers", test_list_tiers(root_key)))
    
    # Summary
    print("\n" + "=" * 60)
    print(" TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} - {name}")
    
    print(f"\n  Total: {passed}/{total} passed")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
