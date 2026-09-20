"""
Unit tests for the Stage 4 Retriever Agent & Hybrid Vector Knowledge Base.
Tests FAISS vector index creation, BM25 lexical search, hybrid reranking, and domain-specific query retrieval.
"""
import unittest
import os
from pathlib import Path
import numpy as np

from app.retriever.vector_store import FaissVectorStore
from app.retriever.lexical_search import BM25LexicalSearch
from app.retriever.reranker import IntentAwareReranker
from app.agents.retriever_agent import RetrieverAgent, RetrievedConversation

class TestRetrieverSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root_dir = Path(__file__).resolve().parent.parent
        cls.corpus_csv = cls.root_dir / "data" / "thread_corpus.csv"
        cls.flat_index = cls.root_dir / "data" / "faiss_flat.index"
        cls.hnsw_index = cls.root_dir / "data" / "faiss_hnsw.index"

        # Initialize lightweight retriever agent
        cls.agent = RetrieverAgent(index_type="flat")
        if cls.corpus_csv.exists():
            cls.agent.initialize()

    def test_faiss_vector_store_build_and_load(self):
        """Verify building, saving, and reloading FAISS IndexFlatIP and IndexHNSWFlat."""
        dim = 384
        n_samples = 100
        np.random.seed(42)
        dummy_embeddings = np.random.randn(n_samples, dim).astype("float32")
        # Normalize L2
        norms = np.linalg.norm(dummy_embeddings, axis=1, keepdims=True)
        dummy_embeddings = dummy_embeddings / norms

        test_flat_path = str(self.root_dir / "data" / "test_flat.index")
        test_hnsw_path = str(self.root_dir / "data" / "test_hnsw.index")

        try:
            # 1. Flat index
            store_flat = FaissVectorStore.build_flat_index(dummy_embeddings)
            store_flat.save(test_flat_path)
            loaded_flat = FaissVectorStore.load(test_flat_path, index_type="flat")
            self.assertEqual(loaded_flat.dimension, dim)
            self.assertEqual(loaded_flat.index.ntotal, n_samples)

            # Search query
            q_vec = dummy_embeddings[0]
            scores, indices = loaded_flat.search(q_vec, top_k=5)
            self.assertEqual(len(scores), 5)
            self.assertEqual(indices[0], 0)  # Top match should be itself
            self.assertAlmostEqual(scores[0], 1.0, places=4)

            # 2. HNSW index
            store_hnsw = FaissVectorStore.build_hnsw_index(dummy_embeddings, m=16, ef_construction=32)
            store_hnsw.save(test_hnsw_path)
            loaded_hnsw = FaissVectorStore.load(test_hnsw_path, index_type="hnsw")
            self.assertEqual(loaded_hnsw.dimension, dim)
            self.assertEqual(loaded_hnsw.index.ntotal, n_samples)

        finally:
            for p in [test_flat_path, test_hnsw_path]:
                if os.path.exists(p):
                    os.remove(p)

    def test_bm25_lexical_search(self):
        """Verify BM25 lexical inverted index indexing and term-based retrieval."""
        docs = [
            "My Spotify premium account was charged twice on my credit card this month.",
            "The audio keeps stuttering and music pauses randomly on desktop app.",
            "How do I reset my account password? I am locked out of Spotify.",
            "Can you bring back the Android home screen widget please?"
        ]
        doc_ids = ["doc_01", "doc_02", "doc_03", "doc_04"]

        bm25 = BM25LexicalSearch()
        bm25.fit(docs, doc_ids=doc_ids)
        self.assertEqual(bm25.num_docs, 4)

        # Query for billing
        results = bm25.search("charged twice on card", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0][0], "doc_01")
        self.assertGreater(results[0][1], 0.0)

        # Query for password
        results_pw = bm25.search("reset password", top_k=2)
        self.assertEqual(results_pw[0][0], "doc_03")

    def test_top5_retrieval_structure(self):
        """Verify retrieve() returns exactly 5 RetrievedConversation objects with complete schema."""
        if not self.corpus_csv.exists() or not self.flat_index.exists():
            self.skipTest("Corpus or FAISS index not built yet.")

        query = "Why did my monthly subscription increase from 9.99 to 12.99?"
        results = self.agent.retrieve(query, top_k=5)

        self.assertEqual(len(results), 5)
        for r in results:
            self.assertIsInstance(r, RetrievedConversation)
            self.assertIsInstance(r.thread_id, str)
            self.assertIsInstance(r.intent, str)
            self.assertIsInstance(r.similarity_score, float)
            self.assertIsInstance(r.customer_issue_summary, str)
            self.assertIsInstance(r.resolution_summary, str)
            self.assertIsInstance(r.retrieved_turns, list)
            # Stage 4 Extended Schema & Metadata
            self.assertIsInstance(r.semantic_score, float)
            self.assertIsInstance(r.lexical_score, float)
            self.assertIsInstance(r.rrf_score, float)
            self.assertIsInstance(r.intent_match, bool)
            self.assertIsInstance(r.retrieval_confidence, float)
            self.assertGreaterEqual(r.retrieval_confidence, 0.0)
            self.assertLessEqual(r.retrieval_confidence, 1.0)
            self.assertIsInstance(r.resolution_type, str)
            self.assertIsInstance(r.contains_dm_request, bool)
            self.assertIsInstance(r.contains_link, bool)
            self.assertIsInstance(r.ranking_score, float)

    def test_descending_similarity(self):
        """Verify retrieved conversations are sorted in strict descending order of ranking score."""
        if not self.corpus_csv.exists() or not self.flat_index.exists():
            self.skipTest("Corpus or FAISS index not built yet.")

        query = "The music keeps stopping every time my phone screen locks."
        results = self.agent.retrieve(query, top_k=5)

        self.assertGreater(len(results), 1)
        for i in range(len(results) - 1):
            self.assertGreaterEqual(
                results[i].ranking_score,
                results[i + 1].ranking_score,
                f"Rank {i} ranking score ({results[i].ranking_score}) < Rank {i+1} ({results[i+1].ranking_score})"
            )

    def test_billing_retrieval_domain(self):
        """Verify a subscription billing query retrieves relevant billing conversations."""
        if not self.corpus_csv.exists() or not self.flat_index.exists():
            self.skipTest("Corpus or FAISS index not built yet.")

        query = "I cancelled my Premium subscription last week but you took payment from my PayPal today."
        results = self.agent.retrieve(query, top_k=5)

        self.assertGreater(len(results), 0)
        # Check that top result is SUBSCRIPTION_BILLING
        self.assertEqual(results[0].intent, "SUBSCRIPTION_BILLING")
        self.assertGreaterEqual(results[0].similarity_score, 0.60)

    def test_playback_retrieval_domain(self):
        """Verify a streaming playback query retrieves relevant playback conversations."""
        if not self.corpus_csv.exists() or not self.flat_index.exists():
            self.skipTest("Corpus or FAISS index not built yet.")

        query = "Songs keep pausing and buffering every 10 seconds while listening over WiFi."
        results = self.agent.retrieve(query, top_k=5)

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].intent, "PLAYBACK_STREAMING")
        self.assertGreaterEqual(results[0].similarity_score, 0.50)

    def test_unknown_query_retrieval(self):
        """Verify out-of-domain query executes without crashing and maintains low similarity."""
        if not self.corpus_csv.exists() or not self.flat_index.exists():
            self.skipTest("Corpus or FAISS index not built yet.")

        nonsense = "What is the boiling point of liquid nitrogen and who won the election?"
        results = self.agent.retrieve(nonsense, top_k=5)

        self.assertEqual(len(results), 5)
        # Check that similarity score is relatively lower than prototypical queries
        self.assertLess(results[0].similarity_score, 0.75)

    def test_hybrid_reranker_fusion(self):
        """Verify IntentAwareReranker properly combines ranks and applies intent alignment."""
        sem_results = [("t1", 0.90, 0), ("t2", 0.80, 1), ("t3", 0.70, 2)]
        lex_results = [("t3", 10.5, 2), ("t1", 5.2, 0), ("t4", 3.1, 3)]
        metadata = {
            "t1": {"intent": "SUBSCRIPTION_BILLING", "customer_issue_summary": "Charged twice"},
            "t2": {"intent": "ACCOUNT_ACCESS_AUTH", "customer_issue_summary": "Password issue"},
            "t3": {"intent": "SUBSCRIPTION_BILLING", "customer_issue_summary": "Refund request"},
            "t4": {"intent": "PLAYBACK_STREAMING", "customer_issue_summary": "Buffer issue"}
        }

        reranker = IntentAwareReranker()
        reranked = reranker.rerank(
            semantic_results=sem_results,
            lexical_results=lex_results,
            metadata_map=metadata,
            query_intent="SUBSCRIPTION_BILLING",
            top_k=3
        )

        self.assertEqual(len(reranked), 3)
        # t1 has top semantic rank, 2nd lexical rank, and matches intent -> should be #1
        self.assertEqual(reranked[0]["thread_id"], "t1")
        self.assertTrue(reranked[0]["intent_aligned"])

if __name__ == "__main__":
    unittest.main()
