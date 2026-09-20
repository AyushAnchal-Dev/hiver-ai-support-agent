"""
Tests for Structured JSON Logging Schema.
Verifies log records adhere to production schema, field types, and UUIDv4 format.
"""
import json
import uuid
import unittest
from pathlib import Path
from app.orchestrator.metrics_collector import MetricsCollector
from app.orchestrator.contracts import ResolverResult

class TestLoggingSchema(unittest.TestCase):

    def setUp(self):
        root = Path(__file__).resolve().parent.parent
        self.log_file = root / "logs" / "orchestrator.log"
        self.collector = MetricsCollector()
        self.sample_result = ResolverResult(
            customer_message="My account won't log in",
            intent="ACCOUNT_ACCESS_AUTH",
            confidence=0.95,
            response="Go to spotify.com/password-reset /SpotifyCares [Citations: Article ART_AUTH_01]",
            citations=["Article ART_AUTH_01"],
            retrieved_evidence=[],
            policy_ids=["POL_AUTH_01"],
            context_summary="Password recovery steps",
            matched_articles=["ART_AUTH_01"],
            grounding_score=0.98,
            escalation_metadata={"escalation_required": False},
            latency_breakdown={"total": 24.5},
            trace=[],
            resolution_type="SELF_SERVE",
            provider_metadata={"engine": "deterministic_resolver"}
        )

    def test_log_entry_schema(self):
        """Verifies logged entries conform strictly to required JSON schema."""
        self.collector.record_execution(self.sample_result, is_cold=False, cache_hit=True)
        self.assertTrue(self.log_file.exists())

        with open(self.log_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            self.assertGreater(len(lines), 0)
            entry = json.loads(lines[-1])

        required_keys = [
            "request_id", "timestamp", "level", "intent", "confidence",
            "provider", "grounding_score", "cache_hit", "cold_start",
            "latency_ms", "resolution_type", "trace"
        ]
        for key in required_keys:
            self.assertIn(key, entry, f"Missing required log key: {key}")

        self.assertIsInstance(entry["cache_hit"], bool)
        self.assertIsInstance(entry["cold_start"], bool)
        self.assertIsInstance(entry["latency_ms"], (int, float))
        self.assertIsInstance(entry["trace"], list)

    def test_request_id_uuid(self):
        """Verifies request_id is a valid RFC 4122 UUIDv4."""
        self.collector.record_execution(self.sample_result)
        with open(self.log_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            entry = json.loads(lines[-1])

        req_id = entry["request_id"]
        parsed_uuid = uuid.UUID(req_id, version=4)
        self.assertEqual(str(parsed_uuid), req_id)

if __name__ == "__main__":
    unittest.main()
