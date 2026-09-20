"""
Validation tests for dataset exports and conversation JSON schema conformity.
"""
import os
import json
import csv
import unittest
from pathlib import Path

class TestSchemasAndArtifacts(unittest.TestCase):

    def setUp(self):
        self.data_dir = Path(__file__).resolve().parent.parent / "data"
        self.raw_csv = self.data_dir / "spotify_raw.csv"
        self.threads_csv = self.data_dir / "spotify_threads.csv"
        self.convs_json = self.data_dir / "spotify_conversations.json"
        self.metadata_json = self.data_dir / "spotify_metadata.json"

    def test_artifacts_exist_and_non_empty(self):
        """Verify production Stage 2 inference artifacts exist and have non-zero file sizes."""
        for p in [self.threads_csv, self.convs_json, self.metadata_json]:
            self.assertTrue(p.exists(), f"File missing: {p}")
            self.assertGreater(p.stat().st_size, 0, f"File is empty: {p}")

    def test_metadata_json_schema(self):
        """Verify data/spotify_metadata.json contains all required metadata fields."""
        with open(self.metadata_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        required_keys = [
            "dataset_name", "source_dataset", "target_brand",
            "total_tweets", "total_threads", "average_thread_length",
            "median_thread_length", "generation_timestamp", "schema_version"
        ]
        for key in required_keys:
            self.assertIn(key, meta, f"Missing required metadata key: {key}")

        self.assertEqual(meta["target_brand"], "SpotifyCares")
        self.assertGreater(meta["total_tweets"], 50000)
        self.assertGreater(meta["total_threads"], 10000)

    def test_conversations_json_schema(self):
        """Verify data/spotify_conversations.json matches the hierarchical schema with raw & clean text."""
        with open(self.convs_json, "r", encoding="utf-8") as f:
            # Load first 20 records to keep test lightweight
            convs = json.load(f)[:20]

        self.assertGreater(len(convs), 0)

        required_thread_keys = [
            "thread_id", "root_tweet_id", "customer_id",
            "brand", "status", "created_at", "conversation_length",
            "maximum_depth", "customer_turns", "brand_turns", "turns"
        ]
        required_turn_keys = [
            "turn_id", "tweet_id", "author_id", "role",
            "created_at", "raw_text", "clean_text",
            "in_response_to_tweet_id", "turn_depth"
        ]

        for conv in convs:
            for k in required_thread_keys:
                self.assertIn(k, conv, f"Missing thread key: {k}")
            
            self.assertEqual(conv["brand"], "SpotifyCares")
            self.assertIsInstance(conv["turns"], list)
            self.assertGreaterEqual(len(conv["turns"]), 1)

            for turn in conv["turns"]:
                for tk in required_turn_keys:
                    self.assertIn(tk, turn, f"Missing turn key: {tk}")
                
                # Check that both raw_text and clean_text exist and are strings
                self.assertIsInstance(turn["raw_text"], str)
                self.assertIsInstance(turn["clean_text"], str)
                self.assertIn(turn["role"], ["customer", "brand"])

    def test_threads_csv_schema(self):
        """Verify data/spotify_threads.csv contains both raw_text and clean_text columns."""
        with open(self.threads_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

        expected_headers = [
            "thread_id", "turn_id", "tweet_id", "author_id", "role",
            "created_at", "raw_text", "clean_text", "in_response_to_tweet_id", "turn_depth"
        ]
        for col in expected_headers:
            self.assertIn(col, header, f"Missing column in spotify_threads.csv: {col}")

if __name__ == "__main__":
    unittest.main()
