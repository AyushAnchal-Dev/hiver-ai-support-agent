"""
Tests for Canonical Fallback Article Grounding.
Verifies intent-to-article mappings use valid canonical IDs and never ungrounded default placeholders.
"""
import unittest
from unittest.mock import MagicMock
from app.orchestrator.resolver_orchestrator import ResolverOrchestrator, INTENT_TO_CANONICAL_ARTICLE

class TestFallbackArticles(unittest.TestCase):

    def test_intent_to_canonical_article_mapping(self):
        """Verifies all mapped articles are canonical knowledge base IDs and not ART_DEFAULT."""
        self.assertNotIn("ART_DEFAULT", INTENT_TO_CANONICAL_ARTICLE.values())
        for intent, article_id in INTENT_TO_CANONICAL_ARTICLE.items():
            self.assertTrue(article_id.startswith("ART_"), f"Invalid article prefix for {intent}: {article_id}")
            self.assertNotEqual(article_id, "ART_DEFAULT")

    def test_orchestrator_error_uses_canonical_article(self):
        """Verifies when pipeline encounters an error, fallback response cites a canonical article."""
        failing_intent_agent = MagicMock()
        failing_intent_agent.predict.side_effect = RuntimeError("Fatal classifier crash")
        failing_intent_agent.predict_intent.side_effect = RuntimeError("Fatal classifier crash")

        orch = ResolverOrchestrator(
            intent_agent=failing_intent_agent,
            lazy_init_retriever=True
        )
        result = orch.resolve("Cannot log in to my account")

        self.assertIn("ART_BUG_01", result.response)
        self.assertNotIn("ART_DEFAULT", result.response)
        self.assertIn("Article ART_BUG_01", result.citations)

if __name__ == "__main__":
    unittest.main()
