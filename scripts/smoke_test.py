"""
Production Smoke Test Suite.
Validates the FastAPI application endpoints (/health, /ready, /resolve) across
core Spotify support scenarios: Billing, Playback Troubleshooting, and Greeting.
Verifies citation integrity, grounding scores, and execution latency.
Supports live server (http://localhost:8000) and embedded TestClient fallback.
"""
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def get_client():
    """Attempts connection to live server; falls back to embedded FastAPI TestClient."""
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:8000/health", timeout=1.0) as resp:
            if resp.status == 200:
                print("[INFO] Connected to live server at http://localhost:8000")
                return "live", "http://localhost:8000"
    except Exception:
        pass

    print("[INFO] Live server not detected on port 8000. Using embedded FastAPI TestClient.")
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    client.__enter__()
    return "embedded", client

def make_request(client_type, client_target, method: str, endpoint: str, json_data: Dict[str, Any] = None):
    """Unified request executor for live and embedded client types."""
    import urllib.request
    import json

    t0 = time.perf_counter()
    if client_type == "live":
        url = f"{client_target}{endpoint}"
        headers = {"Content-Type": "application/json"}
        data = json.dumps(json_data).encode("utf-8") if json_data else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                latency = round((time.perf_counter() - t0) * 1000.0, 2)
                return resp.status, body, latency
        except urllib.error.HTTPError as e:
            body = json.loads(e.read().decode("utf-8"))
            latency = round((time.perf_counter() - t0) * 1000.0, 2)
            return e.code, body, latency
    else:
        if method == "GET":
            resp = client_target.get(endpoint)
        else:
            resp = client_target.post(endpoint, json=json_data)
        latency = round((time.perf_counter() - t0) * 1000.0, 2)
        return resp.status_code, resp.json(), latency

def run_smoke_tests():
    print("=" * 75)
    print("  HIVER AI SUPPORT AGENT — PRODUCTION SMOKE TEST SUITE")
    print("=" * 75)

    client_type, client = get_client()

    failures = []
    test_results = []

    # 1. Health Probe
    print("\n[1/5] Testing /health Liveness Endpoint...")
    status_code, body, lat = make_request(client_type, client, "GET", "/health")
    if status_code == 200 and body.get("status") == "healthy":
        print(f"  [OK] /health ({lat:.1f}ms) | status={body.get('status')}, provider={body.get('provider')}")
        test_results.append(("GET /health", status_code, lat, "PASS", "Liveness OK"))
    else:
        print(f"  [FAIL] /health Failed: code={status_code}, body={body}")
        failures.append(f"/health failed with {status_code}")
        test_results.append(("GET /health", status_code, lat, "FAIL", "Invalid health status"))

    # 2. Ready Probe
    print("\n[2/5] Testing /ready Readiness Endpoint...")
    status_code, body, lat = make_request(client_type, client, "GET", "/ready")
    if status_code == 200 and body.get("ready") is True:
        print(f"  [OK] /ready ({lat:.1f}ms) | ready={body.get('ready')}, is_warmed={body.get('is_warmed')}")
        test_results.append(("GET /ready", status_code, lat, "PASS", "Readiness OK"))
    else:
        print(f"  [FAIL] /ready Failed: code={status_code}, body={body}")
        failures.append(f"/ready failed with {status_code}")
        test_results.append(("GET /ready", status_code, lat, "FAIL", "Service not ready"))

    # 3. Test Scenarios for /resolve
    queries = [
        {
            "name": "Billing & PayPal Dispute",
            "query": "I cancelled my Premium subscription last week but you took payment from my PayPal today.",
            "expected_intent": "SUBSCRIPTION_BILLING"
        },
        {
            "name": "Audio Playback Troubleshooting",
            "query": "Songs keep pausing and stuttering after 10 seconds on desktop app.",
            "expected_intent": "PLAYBACK_STREAMING"
        },
        {
            "name": "Friendly Customer Greeting",
            "query": "Hey Spotify, just wanted to say hi!",
            "expected_intent": "GREETING_OR_CHAT"
        }
    ]

    for idx, q_info in enumerate(queries, start=3):
        q_name = q_info["name"]
        q_text = q_info["query"]
        expected_intent = q_info["expected_intent"]
        print(f"\n[{idx}/5] Testing Query: '{q_name}'...")

        status_code, body, lat = make_request(
            client_type, client, "POST", "/resolve",
            {"customer_message": q_text, "use_llm": False}
        )

        if status_code != 200:
            print(f"  [FAIL] POST /resolve Failed: status={status_code}, body={body}")
            failures.append(f"Query '{q_name}' failed with status {status_code}")
            test_results.append((f"POST {q_name}", status_code, lat, "FAIL", f"HTTP {status_code}"))
            continue

        intent = body.get("intent")
        conf = body.get("confidence", 0.0)
        citations = body.get("citations", [])
        grounding = body.get("grounding_score", 0.0)
        res_text = body.get("response", "")

        # Verifications
        has_citations = len(citations) > 0
        grounding_ok = (grounding >= 0.95)
        response_ok = len(res_text) > 20

        if has_citations and grounding_ok and response_ok:
            print(f"  [OK] Resolved in {lat:.1f}ms | Intent: {intent} (conf={conf:.2f})")
            print(f"       Grounding Score: {grounding:.2f} | Citations ({len(citations)}): {citations}")
            print(f"       Snippet: {res_text[:65]}...")
            test_results.append((f"POST {q_name}", status_code, lat, "PASS", f"G={grounding:.2f}, Citations={len(citations)}"))
        else:
            reason = f"citations={len(citations)}, grounding={grounding}"
            print(f"  [FAIL] Query Validation Failed: {reason}")
            failures.append(f"Query '{q_name}' validation failed: {reason}")
            test_results.append((f"POST {q_name}", status_code, lat, "FAIL", reason))

    print("\n" + "=" * 75)
    print("  SMOKE TEST SUMMARY")
    print("=" * 75)
    print(f"{'Endpoint / Test':<32} {'Status':<8} {'Latency':<10} {'Result':<8} {'Notes'}")
    print("-" * 75)
    for name, code, lat, res, notes in test_results:
        print(f"{name:<32} {code:<8} {lat:6.1f}ms   {res:<8} {notes}")
    print("=" * 75)

    if failures:
        print(f"\n[ERROR] {len(failures)} test(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All production smoke tests PASSED successfully.")
        sys.exit(0)

if __name__ == "__main__":
    run_smoke_tests()
