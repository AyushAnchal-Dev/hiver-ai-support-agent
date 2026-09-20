"""
Tests for PromptBuilder Template Manifest and Integrity Verification.
Verifies cryptographic hashing of prompt templates and JSON persistence.
"""
import json
import unittest
from pathlib import Path
from app.prompts.prompt_builder import PromptBuilder

class TestPromptManifest(unittest.TestCase):

    def setUp(self):
        self.builder = PromptBuilder(version="v1")

    def test_manifest_generation(self):
        """Verifies PromptBuilder computes valid SHA-256 hashes and composite checksum."""
        manifest = self.builder.get_version_manifest()
        self.assertEqual(manifest["version"], "v1")
        self.assertIn("version_checksum", manifest)
        self.assertEqual(len(manifest["version_checksum"]), 64)
        self.assertGreaterEqual(manifest["template_count"], 4)
        self.assertIn("system_prompt.md", manifest["templates"])
        self.assertIn("resolver_prompt.md", manifest["templates"])

    def test_manifest_file_persistence(self):
        """Verifies save_manifest creates and writes valid JSON manifest file."""
        saved_path = self.builder.save_manifest()
        self.assertTrue(Path(saved_path).exists())

        with open(saved_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["version"], "v1")
        self.assertIn("version_checksum", data)
        self.assertIn("templates", data)

if __name__ == "__main__":
    unittest.main()
