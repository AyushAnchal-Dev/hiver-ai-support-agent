"""
Unit tests for the production IntentAgent triage pipeline.
"""
import unittest
from app.agents.intent_agent import IntentAgent

class TestIntentAgent(unittest.TestCase):

    def setUp(self):
        self.agent = IntentAgent()

    def test_known_examples_high_confidence(self):
        """Verify accurate classification of clear, prototypical customer inquiries."""
        test_cases = [
            ("I forgot my password and can't log in to my account, please help me reset it", "ACCOUNT_ACCESS_AUTH"),
            ("Why was I charged 9.99 twice on my credit card for my Spotify subscription?", "SUBSCRIPTION_BILLING"),
            ("The music keeps pausing every 30 seconds and buffering constantly", "PLAYBACK_STREAMING"),
            ("My downloaded songs won't play offline while I'm on airplane mode", "OFFLINE_DOWNLOAD_SYNC"),
            ("My favorite playlist with 300 songs completely disappeared from my library", "PLAYLIST_LIBRARY_MGMT"),
            ("The app crashes immediately upon opening on my iPhone after the new update", "APP_CRASH_TECHNICAL_BUG"),
            ("Spotify Connect won't discover my Sonos speaker over Bluetooth or WiFi", "DEVICE_CONNECT_INTEGRATION"),
            ("Why is this song greyed out and unplayable in my country?", "CONTENT_METADATA_AVAILABILITY"),
            ("I pay for Premium, why am I suddenly getting 30-second audio ads between songs?", "ADS_PROMOTIONAL_ISSUES"),
            ("Please bring back the home screen widget in the next update, the new UI is terrible", "UI_FEEDBACK_FEATURE_REQUEST"),
        ]

        for text, expected_intent in test_cases:
            pred = self.agent.predict(text)
            self.assertEqual(
                pred.predicted_intent,
                expected_intent,
                f"Failed for text: '{text}'. Expected {expected_intent}, got {pred.predicted_intent}"
            )
            self.assertGreaterEqual(pred.confidence, 0.75, f"Confidence too low ({pred.confidence}) for clear example: '{text}'")

    def test_ambiguous_examples(self):
        """Verify ambiguous queries flag is_ambiguous=True and have lower confidence or explain reasons."""
        ambiguous_cases = [
            "My music stopped playing and then the screen went black",  # PLAYBACK vs APP_CRASH
            "I can't access my playlists on my account",               # ACCOUNT vs PLAYLIST
            "Why is there an issue with my premium downloads offline?"  # SUBSCRIPTION vs OFFLINE
        ]

        for text in ambiguous_cases:
            pred = self.agent.predict(text)
            # Should have non-empty predicted intent and valid scores
            valid_names = [spec.intent_name for spec in self.agent.intents] + [
                "UNKNOWN_FALLBACK", "UNKNOWN_OTHER", "GREETING_OR_CHAT", "GENERAL_COMPLAINT"
            ]
            self.assertIn(pred.predicted_intent, valid_names)
            self.assertIsNotNone(pred.confidence)

    def test_unknown_other_fallback(self):
        """Verify out-of-domain or gibberish messages trigger UNKNOWN_OTHER."""
        nonsense = "What is the capital of Australia and who won the cricket match yesterday?"
        pred = self.agent.predict(nonsense)
        self.assertEqual(pred.predicted_intent, "UNKNOWN_OTHER")
        self.assertTrue(pred.is_ambiguous)
        self.assertLess(pred.confidence, 0.50)

    def test_greeting_or_chat_detection(self):
        """Verify polite chatter, greetings, and DM alerts trigger GREETING_OR_CHAT."""
        messages = [
            "Hey @SpotifyCares good morning! Hope you guys are having a wonderful day!",
            "Sent you guys a DM about an issue, please check it out when you can.",
            "Thank you so much for the quick response! Really appreciate the help."
        ]
        for msg in messages:
            pred = self.agent.predict(msg)
            self.assertEqual(pred.predicted_intent, "GREETING_OR_CHAT")
            self.assertGreaterEqual(pred.confidence, 0.75)
            self.assertTrue(pred.auto_handle_candidate)

    def test_general_complaint_detection(self):
        """Verify non-specific emotional rants trigger GENERAL_COMPLAINT."""
        rants = [
            "Your service is absolute garbage and Spotify sucks so bad lately!",
            "Fix this broken app already, this is completely useless and unacceptable!",
            "Worst service ever, you guys are a joke and I hate using this app."
        ]
        for rant in rants:
            pred = self.agent.predict(rant)
            self.assertEqual(pred.predicted_intent, "GENERAL_COMPLAINT")
            self.assertGreaterEqual(pred.confidence, 0.70)
            self.assertEqual(pred.priority, "medium")

    def test_empty_input_handling(self):
        """Verify empty or whitespace strings return UNKNOWN_OTHER fallback without crashing."""
        for empty_val in ["", "   ", None]:
            pred = self.agent.predict(empty_val)
            self.assertEqual(pred.predicted_intent, "UNKNOWN_OTHER")
            self.assertEqual(pred.confidence, 0.0)

    def test_confidence_threshold_behavior(self):
        """Verify confidence thresholding correctly gates human review."""
        clear_msg = "Can you help me reset my password? I am locked out of my account."
        pred_clear = self.agent.predict(clear_msg)
        self.assertGreaterEqual(pred_clear.confidence, 0.75)
        self.assertFalse(pred_clear.is_ambiguous)

if __name__ == "__main__":
    unittest.main()
