"""
Tests for ContextBuilder Evidence Memoization Cache.
Verifies repeated retrieved conversation sets reuse compressed context and track hit rates.
"""
import unittest
from app.agents.context_builder import ContextBuilder
from app.agents.retriever_agent import RetrievedConversation

class TestContextCache(unittest.TestCase):

    def setUp(self):
        self.builder = ContextBuilder(cache_size=64)
        self.sample_convs = [
            RetrievedConversation(
                thread_id="T1001",
                intent="ACCOUNT_ACCESS_AUTH",
                similarity_score=0.92,
                customer_issue_summary="Cannot log into my Spotify account",
                resolution_summary="Sent password reset link",
                retrieved_turns=[
                    {"turn_id": 1, "role": "customer", "text": "I can't log in to Spotify"},
                    {"turn_id": 2, "role": "brand", "text": "Head over to spotify.com/password-reset"}
                ]
            )
        ]

    def tearDown(self):
        self.builder.clear_cache()

    def test_memoization_cache_hit(self):
        """Verifies second compression of identical conversations hits memoization cache."""
        ctx1 = self.builder.build_context(self.sample_convs)
        self.assertEqual(self.builder.cache_hits, 0)
        self.assertEqual(self.builder.cache_misses, 1)

        ctx2 = self.builder.build_context(self.sample_convs)
        self.assertEqual(self.builder.cache_hits, 1)
        self.assertEqual(self.builder.cache_misses, 1)
        self.assertIs(ctx1, ctx2)
        self.assertAlmostEqual(self.builder.cache_hit_rate, 0.5)

    def test_cache_clear_and_hit_rate(self):
        """Verifies clear_cache resets in-memory cache and returns 0 hit rate."""
        self.builder.build_context(self.sample_convs)
        self.builder.build_context(self.sample_convs)
        self.assertGreater(self.builder.cache_hit_rate, 0.0)

        self.builder.clear_cache()
        self.assertEqual(self.builder.cache_hits, 0)
        self.assertEqual(self.builder.cache_misses, 0)
        self.assertEqual(self.builder.cache_hit_rate, 0.0)
        self.assertEqual(len(self.builder._cache), 0)

if __name__ == "__main__":
    unittest.main()
