"""
Unit tests for data cleaning and text normalization pipeline.
"""
import unittest
from app.cleaning.normalizer import clean_text
from app.cleaning.filters import contains_dm_deflection, contains_profanity, is_informative_turn

class TestCleaningPipeline(unittest.TestCase):

    def test_raw_text_preservation(self):
        """Verify that the cleaning pipeline does not mutate the input raw text."""
        raw = "@SpotifyCares @12345 My app keeps crashing! https://t.co/abc ^JK"
        clean = clean_text(raw)
        
        # Raw text should be unchanged
        self.assertEqual(raw, "@SpotifyCares @12345 My app keeps crashing! https://t.co/abc ^JK")
        # Clean text should have transformed handles, URLs, and stripped signature
        self.assertIn("@customer", clean)
        self.assertIn("@SpotifyCares", clean)
        self.assertIn("<LINK:URL>", clean)
        self.assertNotIn("^JK", clean)

    def test_handle_masking(self):
        """Verify customer numeric IDs are masked to @customer while @SpotifyCares is preserved."""
        raw = "@115712 @115713 and @SpotifyCares please check @random_user"
        clean = clean_text(raw)
        self.assertIn("@customer @customer", clean)
        self.assertIn("@SpotifyCares", clean)
        self.assertIn("@mention", clean)

    def test_url_categorization(self):
        """Verify Spotify help links vs general links are semantically categorized."""
        spotify_help = "Check our guide at https://support.spotify.com/article/offline-sync for help."
        general_url = "Look at this screenshot https://t.co/photo123"
        
        clean_help = clean_text(spotify_help)
        clean_general = clean_text(general_url)

        self.assertIn("<LINK:HELP_PORTAL>", clean_help)
        self.assertNotIn("https://support.spotify.com", clean_help)
        self.assertIn("<LINK:URL>", clean_general)

    def test_signature_stripping(self):
        """Verify corporate representative signatures are cleanly removed."""
        raw1 = "@12345 We'd love to help with your playlist! ^JK"
        raw2 = "@12345 Please check your storage settings. -AA"
        raw3 = "@12345 Thanks for reaching out. ^Osebi"

        self.assertFalse(clean_text(raw1).endswith("^JK"))
        self.assertFalse(clean_text(raw2).endswith("-AA"))
        self.assertFalse(clean_text(raw3).endswith("^Osebi"))

    def test_html_unescaping(self):
        """Verify HTML entities are properly unescaped."""
        raw = "Me &amp; my family love Spotify &lt;3 but it&#39;s pausing"
        clean = clean_text(raw)
        self.assertIn("Me & my family", clean)
        self.assertIn("<3", clean)
        self.assertIn("it's", clean)

    def test_dm_deflection_filter(self):
        """Verify deflection detector catches DM, Direct Message, and Private Message."""
        self.assertTrue(contains_dm_deflection("Please shoot us a DM with your email"))
        self.assertTrue(contains_dm_deflection("Send us a Direct Message so we can look into this"))
        self.assertTrue(contains_dm_deflection("Could you send a private message?"))
        self.assertFalse(contains_dm_deflection("Try restarting your phone and reinstalling the app."))

    def test_profanity_filter(self):
        """Verify profanity detection."""
        self.assertTrue(contains_profanity("Your service is terrible and worst shit ever"))
        self.assertFalse(contains_profanity("Can someone please assist me with my account?"))

if __name__ == "__main__":
    unittest.main()
