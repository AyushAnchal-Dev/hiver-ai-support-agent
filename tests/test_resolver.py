"""
Unit tests for Stage 5: Resolver Agent, Context Builder, Policy Agent, and Grounding Validator.
Verifies grounded response synthesis, citation integrity, anti-hallucination validation,
escalation routing, and confidence calibration.
"""
import unittest
from pathlib import Path

from app.agents.context_builder import ContextBuilder, CompressedContext, EvidenceSnippet
from app.agents.policy_agent import PolicyAgent, PolicyContext
from app.agents.grounding_validator import GroundingValidator, GroundingReport
from app.agents.resolver_agent import ResolverAgent, ResolverResponse
from app.agents.retriever_agent import RetrievedConversation

class TestResolverSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root_dir = Path(__file__).resolve().parent.parent
        cls.context_builder = ContextBuilder(max_chars=2000, max_snippets=6)
        cls.policy_agent = PolicyAgent(knowledge_dir=str(cls.root_dir / "knowledge"))
        cls.validator = GroundingValidator()
        cls.resolver = ResolverAgent(brand_signature="/SpotifyCares")

    def _create_mock_context(self, intent: str, thread_id: str = "12345") -> CompressedContext:
        """Helper to create a representative compressed context."""
        snippets = [
            EvidenceSnippet(
                snippet_id=f"T{thread_id}-Turn1",
                thread_id=thread_id,
                turn_id=1,
                role="customer",
                intent=intent,
                content="My songs keep stopping every few seconds during streaming.",
                snippet_type="symptom"
            ),
            EvidenceSnippet(
                snippet_id=f"T{thread_id}-Turn2",
                thread_id=thread_id,
                turn_id=2,
                role="brand",
                intent=intent,
                content="Hey! Could you try toggling Hardware Acceleration off in settings? That often clears up audio pauses.",
                snippet_type="resolution"
            )
        ]
        return CompressedContext(
            snippets=snippets,
            cited_thread_ids=[thread_id],
            total_chars=180,
            estimated_tokens=42,
            has_dm_deflection=False,
            has_link=False,
            top_intent=intent,
            mean_retrieval_confidence=0.85
        )

    def test_playback_troubleshooting(self):
        """Verify playback failure query generates multi-step diagnostic guidance with citations."""
        query = "Music keeps stuttering and pausing every 10 seconds on desktop app."
        intent = "PLAYBACK_STREAMING"
        context = self._create_mock_context(intent, thread_id="99101")
        policy = self.policy_agent.evaluate_policy(query, intent)

        response: ResolverResponse = self.resolver.generate_response(
            customer_message=query,
            predicted_intent=intent,
            context=context,
            policy_context=policy
        )

        self.assertIsInstance(response, ResolverResponse)
        self.assertIn("Hardware Acceleration", response.response)
        self.assertEqual(response.resolution_type, "troubleshooting_steps")
        self.assertFalse(response.escalation_required)
        self.assertGreaterEqual(response.confidence, 0.70)
        self.assertIn("Thread #99101", response.citations)
        self.assertTrue(any("POL_PLAY" in c or "ART_PLAY" in c for c in response.citations))

    def test_billing_escalation(self):
        """Verify billing cancellation/refund query triggers DM deflection and tier-2 escalation."""
        query = "I cancelled my Premium subscription last month but you charged my card $9.99 again. I want a refund."
        intent = "SUBSCRIPTION_BILLING"
        context = self._create_mock_context(intent, thread_id="66201")
        policy = self.policy_agent.evaluate_policy(query, intent)

        response: ResolverResponse = self.resolver.generate_response(
            customer_message=query,
            predicted_intent=intent,
            context=context,
            policy_context=policy
        )

        self.assertTrue(response.escalation_required)
        self.assertEqual(response.target_queue, "billing_tier2")
        self.assertEqual(response.resolution_type, "dm_deflection")
        self.assertIn("Direct Message", response.response)
        self.assertTrue(any("POL_BILL_01" in c or "ART_BILL" in c for c in response.citations))

    def test_greeting_chat_response(self):
        """Verify social chit-chat receives polite acknowledgement without ungrounded troubleshooting steps."""
        query = "Hello Spotify! Hope you have a wonderful Friday!"
        intent = "GREETING_OR_CHAT"
        context = self._create_mock_context(intent, thread_id="77301")
        policy = self.policy_agent.evaluate_policy(query, intent)

        response: ResolverResponse = self.resolver.generate_response(
            customer_message=query,
            predicted_intent=intent,
            context=context,
            policy_context=policy
        )

        self.assertEqual(response.resolution_type, "direct_resolution")
        self.assertFalse(response.escalation_required)
        self.assertNotIn("Hardware Acceleration", response.response)
        self.assertNotIn("reinstall", response.response.lower())

    def test_unknown_query_fallback(self):
        """Verify out-of-domain or unclassified query triggers diagnostic clarification prompt."""
        query = "Can you help me solve my quadratic equation x^2 + 5x + 6 = 0?"
        intent = "UNKNOWN_OTHER"
        context = self._create_mock_context(intent, thread_id="88401")
        policy = self.policy_agent.evaluate_policy(query, intent)

        response: ResolverResponse = self.resolver.generate_response(
            customer_message=query,
            predicted_intent=intent,
            context=context,
            policy_context=policy
        )

        self.assertFalse(response.escalation_required)
        self.assertTrue("details" in response.response.lower() or "help" in response.response.lower())

    def test_feature_request_handling(self):
        """Verify feature feedback / UI request is routed to Spotify Community Ideas board."""
        query = "Please add a feature to sort playlists by release date and BPM."
        intent = "UI_FEEDBACK_FEATURE_REQUEST"
        context = self._create_mock_context(intent, thread_id="55102")
        policy = self.policy_agent.evaluate_policy(query, intent)

        response: ResolverResponse = self.resolver.generate_response(
            customer_message=query,
            predicted_intent=intent,
            context=context,
            policy_context=policy
        )

        self.assertEqual(response.resolution_type, "link_redirection")
        self.assertIn("Community Idea Submissions", response.response)
        self.assertIn("<LINK:URL>", response.response)

    def test_hallucination_validator_rejection(self):
        """Verify GroundingValidator intercepts ungrounded external claims like phone numbers or fake URLs."""
        hallucinated_text = (
            "Hey there! We can fix this right now. Please call us at 1-800-555-SPOTIFY or visit http://freepremium-scam.com to claim your free refund. /SpotifyCares"
        )
        context = self._create_mock_context("PLAYBACK_STREAMING", thread_id="123")
        policy = self.policy_agent.evaluate_policy("test", "PLAYBACK_STREAMING")

        report: GroundingReport = self.validator.validate_response(
            proposed_response=hallucinated_text,
            context=context,
            policy=policy
        )

        self.assertFalse(report.is_grounded)
        self.assertTrue(report.remediation_applied)
        self.assertGreater(len(report.unsupported_claims), 0)
        self.assertNotIn("1-800-555-SPOTIFY", report.final_response)
        self.assertNotIn("freepremium-scam.com", report.final_response)

    def test_confidence_calibration(self):
        """Verify response confidence is strictly bounded [0.0, 1.0] and sensitive to evidence quality."""
        query = "Playback is pausing on WiFi."
        intent = "PLAYBACK_STREAMING"
        context = self._create_mock_context(intent, thread_id="99999")
        policy = self.policy_agent.evaluate_policy(query, intent)

        res = self.resolver.generate_response(query, intent, context, policy)
        self.assertGreaterEqual(res.confidence, 0.0)
        self.assertLessEqual(res.confidence, 1.0)
        self.assertIsInstance(res.confidence, float)

    def test_evidence_citation_integrity(self):
        """Verify every citation in the response exists in either the evidence snippets or policy context."""
        query = "How do I reset my Spotify password?"
        intent = "ACCOUNT_ACCESS_AUTH"
        context = self._create_mock_context(intent, thread_id="33333")
        policy = self.policy_agent.evaluate_policy(query, intent)

        res = self.resolver.generate_response(query, intent, context, policy)
        self.assertGreater(len(res.citations), 0)
        for citation in res.citations:
            if citation.startswith("Thread #"):
                tid = citation.replace("Thread #", "")
                self.assertIn(tid, context.cited_thread_ids)
            elif citation.startswith("Policy "):
                pid = citation.replace("Policy ", "")
                self.assertIn(pid, policy.policy_ids)

    def test_context_compression_budget(self):
        """Verify ContextBuilder compresses conversations within budget and strips duplicates."""
        builder = ContextBuilder(max_chars=300, max_snippets=3)
        mock_convs = [
            RetrievedConversation(
                thread_id="101",
                intent="PLAYBACK_STREAMING",
                similarity_score=0.88,
                customer_issue_summary="Music pauses",
                resolution_summary="Disable hardware acceleration",
                retrieved_turns=[
                    {"turn_id": 1, "role": "customer", "clean_text": "Music keeps pausing constantly.", "is_inbound": True},
                    {"turn_id": 2, "role": "brand", "clean_text": "Hey there! Try disabling Hardware Acceleration in settings. /GK", "is_inbound": False},
                    {"turn_id": 3, "role": "brand", "clean_text": "Try disabling Hardware Acceleration in settings now. /GK", "is_inbound": False}
                ],
                semantic_score=0.88,
                lexical_score=12.0,
                rrf_score=0.03,
                intent_match=True,
                retrieval_confidence=0.85
            )
        ]

        compressed = builder.build_context(mock_convs, query_intent="PLAYBACK_STREAMING")
        self.assertLessEqual(compressed.total_chars, 300)
        self.assertLessEqual(len(compressed.snippets), 2)  # Turn 3 is a near-duplicate of Turn 2
        self.assertEqual(compressed.cited_thread_ids, ["101"])

    def test_citation_validity_against_corpus_and_policies(self):
        """Verify generated citations strictly match valid corpus thread IDs and knowledge base policy IDs."""
        query = "Can you help me cancel my subscription and avoid being charged?"
        intent = "SUBSCRIPTION_BILLING"
        context = self._create_mock_context(intent, thread_id="101")
        policy = self.policy_agent.evaluate_policy(query, intent)

        res = self.resolver.generate_response(query, intent, context, policy)
        self.assertGreater(len(res.citations), 0)
        for c in res.citations:
            if c.startswith("Thread #"):
                tid = c.replace("Thread #", "")
                self.assertEqual(tid, "101")
            elif c.startswith("Policy "):
                pid = c.replace("Policy ", "")
                self.assertIn(pid, policy.policy_ids)
                self.assertTrue(pid.startswith("POL_") or pid.startswith("ART_") or pid.startswith("SOP_") or pid.startswith("ESC_"))

    def test_escalation_routing_queues(self):
        """Verify escalation routing maps sensitive scenarios to correct specialized queues."""
        # 1. Billing dispute
        query_billing = "I was charged twice after cancelling my account, refund me now."
        policy_billing = self.policy_agent.evaluate_policy(query_billing, "SUBSCRIPTION_BILLING")
        self.assertTrue(policy_billing.escalation_required)
        self.assertEqual(policy_billing.target_queue, "billing_tier2")

        # 2. Account compromise / security
        query_security = "Someone hacked my account and changed the email address!"
        policy_security = self.policy_agent.evaluate_policy(query_security, "ACCOUNT_ACCESS_AUTH")
        self.assertTrue(policy_security.escalation_required)
        self.assertEqual(policy_security.target_queue, "account_security_tier2")

        # 3. Safety emergency
        query_safety = "I am so depressed by these ads I might harm myself"
        policy_safety = self.policy_agent.evaluate_policy(query_safety, "GENERAL_COMPLAINT")
        self.assertTrue(policy_safety.escalation_required)
        self.assertEqual(policy_safety.target_queue, "human_support")

        # 4. Standard self-serve playback query
        query_playback = "Songs keep stopping midway on desktop."
        policy_playback = self.policy_agent.evaluate_policy(query_playback, "PLAYBACK_STREAMING")
        self.assertFalse(policy_playback.escalation_required)
        self.assertIsNone(policy_playback.target_queue)

    def test_grounding_failures_remediation(self):
        """Verify GroundingValidator detects prohibited patterns, flags unsupported claims, and applies remediation."""
        prohibited_text = (
            "We have credited your credit card with $50 refund guaranteed! "
            "Please call 1-800-222-3344 or visit http://scam-site.org/claim immediately."
        )
        context = self._create_mock_context("SUBSCRIPTION_BILLING", thread_id="101")
        policy = self.policy_agent.evaluate_policy("refund", "SUBSCRIPTION_BILLING")

        report = self.validator.validate_response(
            proposed_response=prohibited_text,
            context=context,
            policy=policy
        )
        self.assertFalse(report.is_grounded)
        self.assertTrue(report.remediation_applied)
        self.assertNotIn("1-800-222-3344", report.final_response)
        self.assertNotIn("scam-site.org", report.final_response)
        self.assertGreater(len(report.unsupported_claims), 0)

    def test_deterministic_pipeline_reproducibility(self):
        """Verify ResolverAgent execution is strictly deterministic and reproducible across multiple runs."""
        query = "Music keeps buffering and skipping tracks on desktop app."
        intent = "PLAYBACK_STREAMING"
        context = self._create_mock_context(intent, thread_id="888")
        policy = self.policy_agent.evaluate_policy(query, intent)

        res1 = self.resolver.generate_response(query, intent, context, policy)
        res2 = self.resolver.generate_response(query, intent, context, policy)

        self.assertEqual(res1.response, res2.response)
        self.assertEqual(res1.citations, res2.citations)
        self.assertEqual(res1.resolution_type, res2.resolution_type)
        self.assertEqual(res1.confidence, res2.confidence)

    def test_account_access_subtypes(self):
        """Verify ACCOUNT_ACCESS_AUTH subtypes routing prevents wrongful 14-day travel restriction advice."""
        # 1. General login error
        query_login = "I can't log in with my username and password on the mobile app."
        policy_login = self.policy_agent.evaluate_policy(query_login, "ACCOUNT_ACCESS_AUTH")
        context_login = self._create_mock_context("ACCOUNT_ACCESS_AUTH", thread_id="701")
        res_login = self.resolver.generate_response(query_login, "ACCOUNT_ACCESS_AUTH", context_login, policy_login)
        self.assertNotIn("14 days", res_login.response)
        self.assertNotIn("traveling abroad", res_login.response.lower())

        # 2. Travel restriction query
        query_travel = "I am traveling abroad on vacation and Spotify stopped working after 2 weeks."
        policy_travel = self.policy_agent.evaluate_policy(query_travel, "ACCOUNT_ACCESS_AUTH")
        context_travel = self._create_mock_context("ACCOUNT_ACCESS_AUTH", thread_id="702")
        res_travel = self.resolver.generate_response(query_travel, "ACCOUNT_ACCESS_AUTH", context_travel, policy_travel)
        self.assertTrue("14 days" in res_travel.response or "travel" in res_travel.response.lower())

        # 3. Password reset query
        query_pw = "How can I reset my Spotify password?"
        policy_pw = self.policy_agent.evaluate_policy(query_pw, "ACCOUNT_ACCESS_AUTH")
        context_pw = self._create_mock_context("ACCOUNT_ACCESS_AUTH", thread_id="703")
        res_pw = self.resolver.generate_response(query_pw, "ACCOUNT_ACCESS_AUTH", context_pw, policy_pw)
        self.assertIn("password", res_pw.response.lower())
        self.assertNotIn("14 days", res_pw.response)

    def test_brand_voice_not_hallucination(self):
        """Verify approved brand voice empathy phrases are whitelisted and not penalized as hallucinations."""
        polite_response = (
            "Hey there! We hear you, and that doesn't sound right. "
            "Could you try toggling Hardware Acceleration off in settings? "
            "Could you give us a few more details about your device model and app version so we can look into this for you? "
            "Let us know how you get on! /SpotifyCares [Citations: Policy POL_PLAY_01]"
        )
        context = self._create_mock_context("PLAYBACK_STREAMING", thread_id="801")
        policy = self.policy_agent.evaluate_policy("playback glitch", "PLAYBACK_STREAMING")

        report = self.validator.validate_response(
            proposed_response=polite_response,
            context=context,
            policy=policy
        )
        self.assertTrue(report.is_grounded)
        self.assertEqual(len(report.unsupported_claims_before), 0)
        self.assertEqual(report.final_grounding_score, 1.0)
        self.assertTrue(any(c["claim_type"] == "brand_voice_claim" for c in report.claim_classifications))

    def test_context_two_snippets_per_thread(self):
        """Verify ContextBuilder restricts evidence extraction to at most 2 snippets per thread (1 symptom, 1 resolution)."""
        builder = ContextBuilder(max_chars=2000, max_snippets=8)
        mock_conv = RetrievedConversation(
            thread_id="901",
            intent="PLAYBACK_STREAMING",
            similarity_score=0.90,
            customer_issue_summary="Audio stuttering",
            resolution_summary="Toggle hardware acceleration",
            retrieved_turns=[
                {"turn_id": 1, "role": "customer", "clean_text": "Customer symptom turn 1", "is_inbound": True},
                {"turn_id": 2, "role": "customer", "clean_text": "Customer symptom turn 2", "is_inbound": True},
                {"turn_id": 3, "role": "brand", "clean_text": "Brand resolution turn 1 with steps.", "is_inbound": False},
                {"turn_id": 4, "role": "brand", "clean_text": "Brand resolution turn 2 with more steps.", "is_inbound": False},
            ],
            semantic_score=0.90,
            lexical_score=15.0,
            rrf_score=0.03,
            intent_match=True,
            retrieval_confidence=0.88
        )
        compressed = builder.build_context([mock_conv], query_intent="PLAYBACK_STREAMING")
        snippets_for_thread = [s for s in compressed.snippets if s.thread_id == "901"]
        self.assertLessEqual(len(snippets_for_thread), 2)
        roles = [s.role for s in snippets_for_thread]
        self.assertIn("customer", roles)
        self.assertIn("brand", roles)

    def test_confusion_matrix_metrics(self):
        """Verify calculation of binary escalation metrics from confusion matrix."""
        tp, fp, tn, fn = 4, 1, 24, 1
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)

        self.assertAlmostEqual(accuracy, 28 / 30, places=4)
        self.assertAlmostEqual(precision, 0.80, places=4)
        self.assertAlmostEqual(recall, 0.80, places=4)
        self.assertAlmostEqual(f1, 0.80, places=4)

if __name__ == "__main__":
    unittest.main()

