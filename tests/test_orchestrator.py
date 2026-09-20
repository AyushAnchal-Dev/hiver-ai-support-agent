"""
Unit tests for the Stage 6A Production Resolver Orchestrator.
Tests end-to-end multi-agent pipeline coordination, PipelineState transitions,
ResolverResult schema integrity, latency breakdown, stage ordering, citation preservation,
escalation handling, and disk metrics persistence.
"""
import unittest
import json
from pathlib import Path

from app.orchestrator.state import PipelineState, PipelineStageTrace
from app.orchestrator.contracts import ResolverResult
from app.orchestrator.resolver_orchestrator import ResolverOrchestrator
from app.llm.mock_provider import MockProvider

class TestResolverOrchestrator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Instantiate orchestrator with lazy initialization to avoid redundant vector index rebuilds
        cls.orchestrator = ResolverOrchestrator(
            llm_provider=MockProvider(latency_seconds=0.001)
        )

    def test_orchestrator_end_to_end_deterministic(self):
        """Verify full 6-stage execution returns complete ResolverResult schema in deterministic mode."""
        query = "How do I reset my forgotten password?"
        result = self.orchestrator.resolve(customer_message=query, use_llm=False)

        self.assertIsInstance(result, ResolverResult)
        self.assertEqual(result.customer_message, query)
        self.assertEqual(result.intent, "ACCOUNT_ACCESS_AUTH")
        self.assertGreater(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertIn("password", result.response.lower())
        self.assertIn("[Citations:", result.response)
        self.assertGreater(len(result.citations), 0)

        # Verify expanded fields required by Stage 6A
        self.assertIsInstance(result.context_summary, str)
        self.assertIn("snippets", result.context_summary)
        self.assertIsInstance(result.matched_articles, list)
        self.assertIsInstance(result.provider_metadata, dict)
        self.assertEqual(result.provider_metadata.get("engine"), "deterministic_resolver")
        self.assertIsNotNone(result.grounding_report)
        self.assertGreaterEqual(result.grounding_score, 0.0)
        self.assertIsInstance(result.escalation_metadata, dict)
        self.assertIn("escalation_required", result.escalation_metadata)

    def test_stage_ordering(self):
        """Verify PipelineState traces capture the strict 6-stage execution sequence."""
        query = "Can I recover a playlist I accidentally deleted yesterday?"
        result = self.orchestrator.resolve(customer_message=query, use_llm=False)

        expected_stages = [
            "intent_triage",
            "hybrid_retrieval",
            "context_compression",
            "policy_retrieval",
            "response_synthesis",
            "grounding_validation"
        ]

        actual_stages = [t.stage_name for t in result.trace]
        self.assertEqual(actual_stages, expected_stages)

    def test_latency_integrity(self):
        """Verify every pipeline stage records non-negative latency and breakdown sums correctly."""
        query = "Songs keep pausing after 10 seconds on desktop app."
        result = self.orchestrator.resolve(customer_message=query, use_llm=False)

        breakdown = result.latency_breakdown
        self.assertIn("total", breakdown)
        self.assertGreater(breakdown["total"], 0.0)

        # Check all 6 stages present in breakdown
        for stage_name in [
            "intent_triage",
            "hybrid_retrieval",
            "context_compression",
            "policy_retrieval",
            "response_synthesis",
            "grounding_validation"
        ]:
            self.assertIn(stage_name, breakdown)
            self.assertGreaterEqual(breakdown[stage_name], 0.0)

        # Sum of stages matches reported total
        stages_sum = sum(breakdown[s] for s in breakdown if s != "total")
        self.assertAlmostEqual(stages_sum, breakdown["total"], places=1)

    def test_citation_preservation(self):
        """Verify citations in ResolverResult strictly reference valid thread and policy identifiers."""
        query = "Why was I charged twice for Spotify Premium this month?"
        result = self.orchestrator.resolve(customer_message=query, use_llm=False)

        self.assertGreater(len(result.citations), 0)
        has_thread_or_policy = any(
            c.startswith("Thread #") or c.startswith("Policy ") for c in result.citations
        )
        self.assertTrue(has_thread_or_policy, f"Citations {result.citations} missing valid tags.")

    def test_escalation_metadata_handling(self):
        """Verify billing disputes trigger escalation metadata while self-serve does not."""
        # Case A: Billing cancellation dispute
        billing_query = "I cancelled my Premium subscription last week but you took payment from my PayPal today."
        billing_res = self.orchestrator.resolve(customer_message=billing_query, use_llm=False)
        self.assertTrue(billing_res.escalation_metadata.get("escalation_required"))
        self.assertEqual(billing_res.escalation_metadata.get("target_queue"), "billing_tier2")
        self.assertTrue(billing_res.escalation_metadata.get("dm_required"))

        # Case B: Greeting / self-serve
        greet_query = "Hey Spotify, just wanted to say hi!"
        greet_res = self.orchestrator.resolve(customer_message=greet_query, use_llm=False)
        self.assertFalse(greet_res.escalation_metadata.get("escalation_required"))

    def test_llm_mode_orchestration(self):
        """Verify resolve() with use_llm=True produces valid synthesized response with token telemetry."""
        query = "My app crashes every time I tap the search bar."
        result = self.orchestrator.resolve(customer_message=query, use_llm=True)

        self.assertIsInstance(result, ResolverResult)
        self.assertGreater(len(result.response), 20)
        self.assertIn("[Citations:", result.response)
        self.assertEqual(result.provider_metadata.get("engine"), "MOCK")
        self.assertIn("prompt_tokens", result.provider_metadata)
        self.assertIn("completion_tokens", result.provider_metadata)
        self.assertGreater(result.provider_metadata["total_tokens"], 0)

    def test_runtime_metrics_generation(self):
        """Verify execution generates and updates metrics/pipeline_metrics.json, latency_metrics.json, and logs/."""
        query = "How to clear cache on Android?"
        self.orchestrator.resolve(customer_message=query, use_llm=False)

        metrics_dir = self.orchestrator.settings.metrics_dir
        logs_dir = self.orchestrator.settings.logs_dir

        pipeline_metrics_file = metrics_dir / "pipeline_metrics.json"
        latency_metrics_file = metrics_dir / "latency_metrics.json"
        log_file = logs_dir / "orchestrator.log"

        self.assertTrue(pipeline_metrics_file.exists())
        self.assertTrue(latency_metrics_file.exists())
        self.assertTrue(log_file.exists())

        # Verify pipeline metrics JSON schema
        with open(pipeline_metrics_file, "r", encoding="utf-8") as f:
            p_data = json.load(f)
        self.assertIn("total_requests", p_data)
        self.assertIn("success_rate", p_data)
        self.assertIn("mean_grounding_score", p_data)
        self.assertGreater(p_data["total_requests"], 0)

        # Verify latency metrics JSON schema
        with open(latency_metrics_file, "r", encoding="utf-8") as f:
            l_data = json.load(f)
        self.assertIn("sample_size", l_data)
        self.assertIn("stages", l_data)
        self.assertIn("end_to_end_total", l_data["stages"])
        self.assertIn("p50_ms", l_data["stages"]["end_to_end_total"])

        # Verify log file contains records
        with open(log_file, "r", encoding="utf-8") as f:
            log_content = f.read()
        self.assertIn("Intent:", log_content)
        self.assertIn("Total Latency:", log_content)

    def test_serialization_to_dict(self):
        """Verify ResolverResult.to_dict() produces valid JSON-serializable structure."""
        query = "Music stuttering over bluetooth."
        result = self.orchestrator.resolve(customer_message=query, use_llm=False)
        d = result.to_dict()

        self.assertIsInstance(d, dict)
        serialized = json.dumps(d)
        self.assertIsInstance(serialized, str)
        self.assertIn("customer_message", d)
        self.assertIn("context_summary", d)
        self.assertIn("matched_articles", d)

if __name__ == "__main__":
    unittest.main()
