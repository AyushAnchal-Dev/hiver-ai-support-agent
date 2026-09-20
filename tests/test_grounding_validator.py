"""
Unit tests for GroundingValidator.
Verifies brand voice whitelist, claim taxonomy classification,
and anti-hallucination remediation.
"""
import unittest
from pathlib import Path

from app.agents.context_builder import CompressedContext, EvidenceSnippet
from app.agents.policy_agent import PolicyContext
from app.agents.grounding_validator import GroundingValidator, GroundingReport

class TestGroundingValidator(unittest.TestCase):

    def setUp(self):
        self.validator = GroundingValidator()
        self.mock_context = CompressedContext(
            snippets=[
                EvidenceSnippet(
                    snippet_id="T101-Turn1",
                    thread_id="101",
                    turn_id=1,
                    role="brand",
                    intent="PLAYBACK_STREAMING",
                    content="Toggle Hardware Acceleration off in Spotify Desktop settings.",
                    snippet_type="resolution"
                )
            ],
            cited_thread_ids=["101"],
            total_chars=100,
            estimated_tokens=25,
            has_dm_deflection=False,
            has_link=False,
            top_intent="PLAYBACK_STREAMING",
            mean_retrieval_confidence=0.90
        )
        self.mock_policy = PolicyContext(
            intent="PLAYBACK_STREAMING",
            policy_ids=["POL_PLAY_01"],
            actionable_steps=["Check your internet connection and toggle WiFi off and on."],
            canonical_urls=["https://support.spotify.com/article/playback/"],
            escalation_required=False,
            escalation_reason=None,
            target_queue=None,
            dm_required=False,
            matched_policies=[{"policy_id": "POL_PLAY_01", "description": "Playback diagnostic rule", "mandatory_instruction": "Restart app"}],
            matched_articles=[{"article_id": "ART_PLAY_01", "title": "Fix playback issues", "summary": "Guide to fix stuttering and buffering."}]
        )

    def test_whitelist_empathy_phrases(self):
        """Verify approved brand voice empathy phrases are whitelisted as brand_voice_claim."""
        response = "Hey there! We hear you, and that doesn't sound right. Let us know how you get on! /SpotifyCares"
        report = self.validator.validate_response(response, self.mock_context, self.mock_policy)
        self.assertTrue(report.is_grounded)
        self.assertEqual(len(report.unsupported_claims_before), 0)
        self.assertTrue(all(c["claim_type"] == "brand_voice_claim" for c in report.claim_classifications))

    def test_unsupported_factual_claim_detection(self):
        """Verify prohibited phone numbers and external links are classified as unsupported and remediated."""
        response = "Hey there! Please call 1-800-555-9999 or visit http://fake-site.com for a full refund."
        report = self.validator.validate_response(response, self.mock_context, self.mock_policy)
        self.assertFalse(report.is_grounded)
        self.assertTrue(report.remediation_applied)
        self.assertGreater(len(report.unsupported_claims_before), 0)
        self.assertNotIn("1-800-555-9999", report.final_response)
        self.assertNotIn("fake-site.com", report.final_response)

    def test_article_and_policy_claim_classification(self):
        """Verify valid policy steps and article guidance are classified into policy_claim / article_claim."""
        response = "Check your internet connection and toggle WiFi off and on. Guide to fix stuttering and buffering."
        report = self.validator.validate_response(response, self.mock_context, self.mock_policy)
        claim_types = [c["claim_type"] for c in report.claim_classifications]
        self.assertTrue(any(t in ["policy_claim", "article_claim"] for t in claim_types))

if __name__ == "__main__":
    unittest.main()
