"""
Unit tests for the Stage 6A LLM Abstraction Layer and Prompt Engineering Engine.
Tests provider exception hierarchy, timeout handling, authentication failure,
mock determinism, token streaming, prompt versioning, and settings loading.
"""
import unittest
import tempfile
from pathlib import Path

from app.settings import Settings, get_settings, load_settings
from app.llm.exceptions import (
    LLMProviderError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm.contracts import StreamChunk, LLMGenerationResponse
from app.llm.base_provider import BaseLLMProvider
from app.llm.mock_provider import MockProvider
from app.llm.openai_provider import OpenAIProvider
from app.llm.gemini_provider import GeminiProvider
from app.llm.factory import get_llm_provider
from app.prompts.prompt_builder import PromptBuilder

class TestLLMProviders(unittest.TestCase):

    def test_provider_exceptions_hierarchy(self):
        """Verify custom provider exceptions inherit properly from LLMProviderError."""
        self.assertTrue(issubclass(LLMAuthenticationError, LLMProviderError))
        self.assertTrue(issubclass(LLMRateLimitError, LLMProviderError))
        self.assertTrue(issubclass(LLMTimeoutError, LLMProviderError))
        self.assertTrue(issubclass(LLMUnavailableError, LLMProviderError))

        err = LLMAuthenticationError("Auth failed", provider="OPENAI", status_code=401)
        self.assertEqual(err.provider, "OPENAI")
        self.assertEqual(err.status_code, 401)
        self.assertIn("Auth failed", str(err))

    def test_mock_determinism(self):
        """Verify MockProvider generates deterministic responses and token metrics."""
        provider = MockProvider(model_name="mock-test", latency_seconds=0.0)
        prompt = "I need help with my billing charge after cancellation."

        res1 = provider.generate(prompt)
        res2 = provider.generate(prompt)

        self.assertEqual(res1.text, res2.text)
        self.assertEqual(res1.prompt_tokens, res2.prompt_tokens)
        self.assertEqual(res1.completion_tokens, res2.completion_tokens)
        self.assertGreater(res1.prompt_tokens, 0)
        self.assertGreater(res1.completion_tokens, 0)
        self.assertEqual(res1.finish_reason, "stop")
        self.assertIn("[Citations:", res1.text)

    def test_stream_chunks_with_token_metadata(self):
        """Verify streaming yields sequential StreamChunk objects with token metadata."""
        provider = MockProvider(model_name="mock-stream", latency_seconds=0.0)
        prompt = "How do I reset my password?"

        chunks = list(provider.stream_generate(prompt))
        self.assertGreater(len(chunks), 1)

        # Validate chunk sequencing
        for i, chunk in enumerate(chunks):
            self.assertIsInstance(chunk, StreamChunk)
            self.assertEqual(chunk.chunk_index, i)
            if i == len(chunks) - 1:
                self.assertTrue(chunk.is_final)
                self.assertEqual(chunk.finish_reason, "stop")
            else:
                self.assertFalse(chunk.is_final)

        # Reconstructed text matches word-by-word
        full_text = "".join(c.text for c in chunks)
        self.assertIn("password", full_text.lower())

    def test_timeout_handling(self):
        """Verify provider raises LLMTimeoutError when timeout simulation is triggered."""
        provider = MockProvider(simulate_error="timeout")
        with self.assertRaises(LLMTimeoutError) as ctx:
            provider.generate("Test prompt")
        self.assertEqual(ctx.exception.provider, "MOCK")

        with self.assertRaises(LLMTimeoutError):
            list(provider.stream_generate("Test prompt"))

    def test_auth_failure_handling(self):
        """Verify provider raises LLMAuthenticationError when auth simulation is triggered."""
        provider = MockProvider(simulate_error="auth")
        with self.assertRaises(LLMAuthenticationError) as ctx:
            provider.generate("Test prompt")
        self.assertEqual(ctx.exception.status_code, 401)

        # Test OpenAI provider with empty key and fallback disabled
        openai_live = OpenAIProvider(api_key="", fallback_to_mock=False)
        with self.assertRaises(LLMAuthenticationError):
            openai_live.generate("Test prompt")

        # Test Gemini provider with empty key and fallback disabled
        gemini_live = GeminiProvider(api_key="", fallback_to_mock=False)
        with self.assertRaises(LLMAuthenticationError):
            gemini_live.generate("Test prompt")

    def test_openai_and_gemini_fallback_to_mock(self):
        """Verify OpenAI and Gemini adapters gracefully fall back to mock when unconfigured."""
        openai_fb = OpenAIProvider(api_key=None, fallback_to_mock=True)
        res_oa = openai_fb.generate("Billing inquiry")
        self.assertTrue(res_oa.is_fallback)
        self.assertEqual(res_oa.provider, "OPENAI")
        self.assertIn("Direct Message", res_oa.text)

        # Test fallback streaming
        chunks_oa = list(openai_fb.stream_generate("Billing inquiry"))
        self.assertGreater(len(chunks_oa), 0)
        self.assertTrue(chunks_oa[0].metadata.get("fallback"))

        gemini_fb = GeminiProvider(api_key="", fallback_to_mock=True)
        res_gem = gemini_fb.generate("Troubleshoot playback")
        self.assertTrue(res_gem.is_fallback)
        self.assertEqual(res_gem.provider, "GEMINI")
        self.assertIn("diagnostic", res_gem.text.lower())

    def test_factory_resolution(self):
        """Verify get_llm_provider instantiates appropriate providers based on settings."""
        p_mock = get_llm_provider(provider_name="MOCK")
        self.assertIsInstance(p_mock, MockProvider)

        p_openai = get_llm_provider(provider_name="OPENAI")
        self.assertIsInstance(p_openai, OpenAIProvider)

        p_gemini = get_llm_provider(provider_name="GEMINI")
        self.assertIsInstance(p_gemini, GeminiProvider)

    def test_settings_loader(self):
        """Verify get_settings loads immutable settings with required directories."""
        settings = get_settings()
        self.assertIsInstance(settings, Settings)
        self.assertTrue(settings.data_dir.exists())
        self.assertTrue(settings.metrics_dir.exists())
        self.assertTrue(settings.logs_dir.exists())
        self.assertIsInstance(settings.offline_mode, bool)
        self.assertEqual(settings.brand_signature, "/SpotifyCares")

    def test_prompt_builder_versioning_and_templating(self):
        """Verify PromptBuilder loads v1 markdown templates and performs variable substitution."""
        builder = PromptBuilder(version="v1")
        self.assertIn("v1", builder.list_available_versions())

        # Test System Prompt
        sys_prompt = builder.build_system_prompt()
        self.assertIn("@SpotifyCares", sys_prompt)
        self.assertIn("Safety Guardrails", sys_prompt)
        self.assertIn("Citation Formatting", sys_prompt)

        # Test Resolver Prompt Substitution
        class DummyContext:
            snippets = []
            cited_thread_ids = ["T199189"]
            total_chars = 150
            estimated_tokens = 38
        class DummyPolicy:
            actionable_steps = ["Step 1: Check WiFi", "Step 2: Restart"]
            canonical_urls = ["https://support.spotify.com/article/playback"]

        user_prompt = builder.build_resolver_prompt(
            customer_message="Music keeps stopping on my speaker",
            intent="PLAYBACK_STREAMING",
            context=DummyContext(),
            policy=DummyPolicy()
        )
        self.assertIn("Music keeps stopping on my speaker", user_prompt)
        self.assertIn("PLAYBACK_STREAMING", user_prompt)
        self.assertIn("Step 1: Check WiFi", user_prompt)
        self.assertIn("https://support.spotify.com/article/playback", user_prompt)

    def test_prompt_builder_missing_version_raises(self):
        """Verify PromptBuilder raises FileNotFoundError on non-existent version."""
        with self.assertRaises(FileNotFoundError):
            PromptBuilder(version="v999_nonexistent")

if __name__ == "__main__":
    unittest.main()
