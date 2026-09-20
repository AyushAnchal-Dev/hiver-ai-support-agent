"""
Unit tests for conversation thread graph reconstruction.
"""
import unittest
from app.data.conversation_graph import reconstruct_conversation_threads

class TestReconstruction(unittest.TestCase):

    def setUp(self):
        # Synthetic mock tweets representing a 3-turn exchange
        self.mock_tweets = {
            "101": {
                "tweet_id": "101",
                "author_id": "cust_001",
                "inbound": True,
                "created_at": "Tue Oct 31 10:00:00 +0000 2017",
                "text": "@SpotifyCares why are my offline songs not playing?",
                "response_tweet_id": "102",
                "in_response_to_tweet_id": None
            },
            "102": {
                "tweet_id": "102",
                "author_id": "SpotifyCares",
                "inbound": False,
                "created_at": "Tue Oct 31 10:05:00 +0000 2017",
                "text": "@cust_001 Hi! Have you checked your offline storage limit? ^JK",
                "response_tweet_id": "103",
                "in_response_to_tweet_id": "101"
            },
            "103": {
                "tweet_id": "103",
                "author_id": "cust_001",
                "inbound": True,
                "created_at": "Tue Oct 31 10:10:00 +0000 2017",
                "text": "@SpotifyCares Yes, I have 20GB free space on SD card",
                "response_tweet_id": None,
                "in_response_to_tweet_id": "102"
            }
        }
        self.children_map = {
            "101": ["102"],
            "102": ["103"]
        }
        self.target_roots = {"101"}

    def test_reconstruction_integrity(self):
        """Verify thread turns are collected and correctly structured."""
        threads = reconstruct_conversation_threads(
            tweets_dict=self.mock_tweets,
            target_roots=self.target_roots,
            children_map=self.children_map,
            brand_handle="SpotifyCares"
        )
        self.assertEqual(len(threads), 1)
        thread = threads[0]

        # Verify thread attributes
        self.assertEqual(thread.thread_id, "101")
        self.assertEqual(thread.root_tweet_id, "101")
        self.assertEqual(thread.customer_id, "cust_001")
        self.assertEqual(thread.brand, "SpotifyCares")
        self.assertEqual(thread.conversation_length, 3)
        self.assertEqual(thread.maximum_depth, 3)
        self.assertEqual(thread.customer_turns, 2)
        self.assertEqual(thread.brand_turns, 1)
        self.assertEqual(thread.status, "resolved_publicly")

    def test_chronological_ordering_and_dual_text(self):
        """Verify turns are strictly ordered and preserve both raw and clean text."""
        threads = reconstruct_conversation_threads(
            tweets_dict=self.mock_tweets,
            target_roots=self.target_roots,
            children_map=self.children_map,
            brand_handle="SpotifyCares"
        )
        turns = threads[0].turns
        self.assertEqual(len(turns), 3)

        # Turn 1
        self.assertEqual(turns[0].turn_id, 1)
        self.assertEqual(turns[0].tweet_id, "101")
        self.assertEqual(turns[0].role, "customer")
        self.assertEqual(turns[0].raw_text, "@SpotifyCares why are my offline songs not playing?")
        self.assertEqual(turns[0].clean_text, "@SpotifyCares why are my offline songs not playing?")

        # Turn 2
        self.assertEqual(turns[1].turn_id, 2)
        self.assertEqual(turns[1].tweet_id, "102")
        self.assertEqual(turns[1].role, "brand")
        self.assertIn("^JK", turns[1].raw_text)
        self.assertNotIn("^JK", turns[1].clean_text)

        # Turn 3
        self.assertEqual(turns[2].turn_id, 3)
        self.assertEqual(turns[2].tweet_id, "103")
        self.assertEqual(turns[2].role, "customer")

if __name__ == "__main__":
    unittest.main()
