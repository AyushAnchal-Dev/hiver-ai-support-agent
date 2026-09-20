"""
Tests for BM25 Lexical Inverted Index Corpus Cache.
Verifies corpus fingerprinting, index reuse, and cache reset functionality.
"""
import unittest
from app.retriever.lexical_search import BM25LexicalSearch

class TestBM25Cache(unittest.TestCase):

    def setUp(self):
        BM25LexicalSearch.reset_cache()
        self.corpus = [
            "spotify payment failed can not update card",
            "offline playback tracks grayed out premium",
            "need help resetting password cannot log in"
        ]
        self.doc_ids = ["doc1", "doc2", "doc3"]

    def tearDown(self):
        BM25LexicalSearch.reset_cache()

    def test_corpus_caching_avoids_rebuild(self):
        """Verifies fitting identical corpus reuses inverted index and increments cache_hits."""
        bm25_1 = BM25LexicalSearch()
        bm25_1.fit(self.corpus, doc_ids=self.doc_ids)
        self.assertEqual(BM25LexicalSearch.cache_misses, 1)
        self.assertEqual(BM25LexicalSearch.cache_hits, 0)

        # Second instance fitting exact same corpus
        bm25_2 = BM25LexicalSearch()
        bm25_2.fit(self.corpus, doc_ids=self.doc_ids)
        self.assertEqual(BM25LexicalSearch.cache_hits, 1)
        self.assertEqual(bm25_2.num_docs, len(self.corpus))
        self.assertEqual(bm25_2.idf, bm25_1.idf)

    def test_reset_cache(self):
        """Verifies reset_cache purges corpus index cache and resets metrics."""
        bm25 = BM25LexicalSearch()
        bm25.fit(self.corpus, doc_ids=self.doc_ids)
        self.assertGreater(len(BM25LexicalSearch._shared_corpus_cache), 0)

        BM25LexicalSearch.reset_cache()
        self.assertEqual(len(BM25LexicalSearch._shared_corpus_cache), 0)
        self.assertEqual(BM25LexicalSearch.cache_hits, 0)
        self.assertEqual(BM25LexicalSearch.cache_misses, 0)

if __name__ == "__main__":
    unittest.main()
