"""
Tests for ResolverOrchestrator Warmup Interface and Subsystem Preloading.
Verifies is_warmed state flag, warmup_duration_ms metric, and custom warmup query handling.
"""
import unittest
from unittest.mock import MagicMock
from app.orchestrator.resolver_orchestrator import ResolverOrchestrator

class TestOrchestratorWarmup(unittest.TestCase):

    def setUp(self):
        self.mock_retriever = MagicMock()
        self.mock_policy = MagicMock()
        self.orch = ResolverOrchestrator(
            retriever_agent=self.mock_retriever,
            policy_agent=self.mock_policy,
            lazy_init_retriever=True
        )

    def test_warmup_interface_attributes(self):
        """Verifies warmup interface attributes exist and have correct initial types."""
        self.assertFalse(self.orch.is_warmed)
        self.assertEqual(self.orch.warmup_duration_ms, 0.0)

    def test_warmup_execution_marks_warm(self):
        """Verifies warmup() executes initialization and sets is_warmed=True."""
        self.orch.warmup()
        self.assertTrue(self.orch.is_warmed)
        self.assertGreaterEqual(self.orch.warmup_duration_ms, 0.0)
        self.mock_retriever.initialize.assert_called_once()
        self.mock_policy.evaluate_policy.assert_called()

    def test_warmup_with_custom_queries(self):
        """Verifies warmup accepts custom domain queries and executes successfully."""
        custom_queries = [
            "Why was I charged twice for Spotify Premium?",
            "How do I reset my password?"
        ]
        self.orch.warmup(queries=custom_queries)
        self.assertTrue(self.orch.is_warmed)
        self.assertGreaterEqual(self.orch.warmup_duration_ms, 0.0)

if __name__ == "__main__":
    unittest.main()
