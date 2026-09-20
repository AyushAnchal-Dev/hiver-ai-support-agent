"""
Deterministic Mock LLM Provider.
Provides deterministic responses, realistic token metrics, streaming chunks,
and configurable failure modes for reliable unit and integration testing.
"""
import time
from typing import Iterator, Optional, Dict, Any, List

from app.llm.base_provider import BaseLLMProvider
from app.llm.contracts import StreamChunk, LLMGenerationResponse
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)

class MockProvider(BaseLLMProvider):
    """Deterministic local mock LLM provider."""

    def __init__(
        self,
        model_name: str = "mock-grounded-v1",
        simulate_error: Optional[str] = None,
        default_response: Optional[str] = None,
        latency_seconds: float = 0.01,
        transient_failures: int = 0,
        max_retries: int = 3,
        retry_delays: Optional[List[float]] = None
    ):
        self._model_name = model_name
        self.simulate_error = simulate_error
        self.default_response = default_response
        self.latency_seconds = latency_seconds
        self.transient_failures = transient_failures
        self.max_retries = max_retries
        self.retry_delays = retry_delays if retry_delays is not None else [0.2, 0.4, 0.8]

    @property
    def provider_name(self) -> str:
        return "MOCK"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_available(self) -> bool:
        return True

    def _check_simulated_errors(self):
        """Raises simulated exceptions if configured."""
        if self.transient_failures > 0:
            self.transient_failures -= 1
            raise LLMUnavailableError("Mock transient service unavailable (503)", provider=self.provider_name, status_code=503)
        if self.simulate_error == "timeout":
            raise LLMTimeoutError("Mock generation timed out after threshold", provider=self.provider_name)
        elif self.simulate_error == "auth":
            raise LLMAuthenticationError("Mock API key is missing or invalid", provider=self.provider_name, status_code=401)
        elif self.simulate_error == "rate_limit":
            raise LLMRateLimitError("Mock quota limit reached for tier", provider=self.provider_name, status_code=429)
        elif self.simulate_error == "unavailable":
            raise LLMUnavailableError("Mock inference service unavailable", provider=self.provider_name, status_code=503)


    def _synthesize_mock_text(self, prompt: str) -> str:
        """Synthesizes a realistic, grounded response based on prompt contents."""
        if self.default_response:
            return self.default_response

        p_low = prompt.lower()
        if "billing" in p_low or "charge" in p_low or "cancel" in p_low:
            return (
                "Hi there! Payment questions can definitely be stressful, so let's check on your subscription status. "
                "Because billing and account details need to remain private, please send us a Direct Message with your "
                "account's email address and receipt date so our team can assist. "
                "Let us know how you get on! /SpotifyCares [Citations: Thread #T101, Policy POL_BILL_01]"
            )
        elif "password" in p_low or "login" in p_low:
            return (
                "Hi there! Resetting your password should be quick and easy — let's get you back in. "
                "Head over to spotify.com/password-reset to request a secure password recovery link to your registered email. "
                "Let us know how you get on! /SpotifyCares [Citations: Thread #T102, Policy POL_AUTH_01]"
            )
        elif "playback" in p_low or "pause" in p_low or "stutter" in p_low:
            return (
                "Hey there! That playback glitch doesn't sound right at all. Let's troubleshoot this together. "
                "Could you try these quick diagnostic steps? (1) Restart your device (2) Perform a clean reinstall of Spotify (3) Test offline mode. "
                "Let us know how you get on! /SpotifyCares [Citations: Thread #T103, Policy POL_PLAY_01]"
            )
        else:
            return (
                "Hey there! Thanks for reaching out to Spotify support. We're here to lend a hand. "
                "Could you give us a few more details about your device model and app version so we can look into this for you? "
                "Let us know how you get on! /SpotifyCares [Citations: Policy ART_BUG_01]"
            )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMGenerationResponse:
        delays = self.retry_delays if self.retry_delays is not None else [0.2, 0.4, 0.8]
        attempt = 0
        last_error = None

        while attempt <= self.max_retries:
            try:
                self._check_simulated_errors()
                break
            except (LLMRateLimitError, LLMUnavailableError, LLMTimeoutError) as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = delays[min(attempt, len(delays) - 1)]
                    if delay > 0:
                        time.sleep(delay)
                    attempt += 1
                else:
                    raise last_error
            except (LLMAuthenticationError, Exception) as e:
                raise e

        t0 = time.perf_counter()

        if self.latency_seconds > 0:
            time.sleep(self.latency_seconds)

        text = self._synthesize_mock_text(prompt)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        prompt_tokens = max(1, len(prompt.split()) + (len(system_prompt.split()) if system_prompt else 0))
        completion_tokens = max(1, len(text.split()))

        return LLMGenerationResponse(
            text=text,
            model=self._model_name,
            provider=self.provider_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason="stop",
            latency_ms=latency_ms,
            is_fallback=False,
            metadata={"mock": True}
        )

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Iterator[StreamChunk]:
        self._check_simulated_errors()
        text = self._synthesize_mock_text(prompt)
        words = text.split(" ")

        for idx, word in enumerate(words):
            is_final = (idx == len(words) - 1)
            chunk_text = word if is_final else (word + " ")
            yield StreamChunk(
                text=chunk_text,
                chunk_index=idx,
                is_final=is_final,
                finish_reason="stop" if is_final else None,
                token_count=1,
                metadata={"mock": True}
            )
