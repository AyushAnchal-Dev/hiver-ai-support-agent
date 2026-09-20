"""
Tests for LLM Provider Retry and Exponential Backoff Logic.
Verifies retry recovery on transient errors, failure on unretryable auth errors, and retry exhaustion.
"""
import unittest
from app.llm.mock_provider import MockProvider
from app.llm.exceptions import LLMAuthenticationError, LLMUnavailableError

class TestRetryLogic(unittest.TestCase):

    def test_transient_failure_retries_and_succeeds(self):
        """Verifies transient failures retry according to backoff schedule and succeed."""
        provider = MockProvider(
            transient_failures=2,
            max_retries=3,
            retry_delays=[0.001, 0.001, 0.001]
        )
        response = provider.generate("help with billing issue")
        self.assertIsNotNone(response.text)
        self.assertEqual(provider.transient_failures, 0)

    def test_auth_failure_no_retry(self):
        """Verifies authentication failures (401/403) fail immediately without retrying."""
        provider = MockProvider(
            simulate_error="auth",
            max_retries=3,
            retry_delays=[0.001, 0.001, 0.001]
        )
        with self.assertRaises(LLMAuthenticationError) as ctx:
            provider.generate("test query")
        self.assertEqual(ctx.exception.status_code, 401)

    def test_exhausted_retries_raises(self):
        """Verifies exceeding max retries raises the terminal LLMUnavailableError."""
        provider = MockProvider(
            transient_failures=5,
            max_retries=3,
            retry_delays=[0.001, 0.001, 0.001]
        )
        with self.assertRaises(LLMUnavailableError):
            provider.generate("test query")

if __name__ == "__main__":
    unittest.main()
