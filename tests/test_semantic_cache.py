"""
Tests for Semantic Dense Retrieval LRU Cache.
Verifies cache hits, FIFO/LRU eviction policy, and cache reset operations.
"""
import unittest
from unittest.mock import MagicMock
import numpy as np
from app.retriever.semantic_search import SemanticSearch

class TestSemanticSearchCache(unittest.TestCase):

    def setUp(self):
        SemanticSearch.clear_cache()
        self.mock_model = MagicMock()
        self.mock_model.encode.side_effect = lambda q, **kwargs: np.array([0.1, 0.2, 0.3], dtype=np.float32)
        SemanticSearch._shared_models["test-model"] = self.mock_model
        self.retriever = SemanticSearch(model_name="test-model")

    def tearDown(self):
        SemanticSearch.clear_cache()

    def test_lru_cache_hit(self):
        """Verifies repeated query returns cached embedding and increments cache_hits."""
        q = "how do I reset my password?"
        v1 = self.retriever.encode_query(q)
        initial_hits = SemanticSearch.cache_hits
        v2 = self.retriever.encode_query(q)
        
        self.assertEqual(SemanticSearch.cache_hits, initial_hits + 1)
        np.testing.assert_array_equal(v1, v2)
        self.mock_model.encode.assert_called_once()

    def test_lru_cache_eviction(self):
        """Verifies cache evicts oldest item when max capacity is reached."""
        orig_max = SemanticSearch._max_cache_size
        try:
            SemanticSearch._max_cache_size = 2
            self.retriever.encode_query("query 1")
            self.retriever.encode_query("query 2")
            self.assertIn("query 1", SemanticSearch._embedding_cache)
            self.assertIn("query 2", SemanticSearch._embedding_cache)

            # Insert third query to trigger eviction of query 1
            self.retriever.encode_query("query 3")
            self.assertNotIn("query 1", SemanticSearch._embedding_cache)
            self.assertIn("query 2", SemanticSearch._embedding_cache)
            self.assertIn("query 3", SemanticSearch._embedding_cache)
            self.assertEqual(len(SemanticSearch._embedding_cache), 2)
        finally:
            SemanticSearch._max_cache_size = orig_max

    def test_cache_clear(self):
        """Verifies clear_cache resets dictionary and telemetry counters."""
        self.retriever.encode_query("sample query")
        self.assertGreater(len(SemanticSearch._embedding_cache), 0)
        
        SemanticSearch.clear_cache()
        self.assertEqual(len(SemanticSearch._embedding_cache), 0)
        self.assertEqual(SemanticSearch.cache_hits, 0)
        self.assertEqual(SemanticSearch.cache_misses, 0)

if __name__ == "__main__":
    unittest.main()
