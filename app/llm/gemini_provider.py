"""
Google Gemini LLM Provider Adapter.
Implements Google Generative Language REST API integration using standard library urllib.
Supports zero-dependency operation with fallback to MockProvider when unconfigured.
"""
import json
import time
import urllib.request
import urllib.error
from typing import Iterator, Optional, Dict, Any, List

from app.llm.base_provider import BaseLLMProvider
from app.llm.contracts import StreamChunk, LLMGenerationResponse
from app.llm.mock_provider import MockProvider
from app.llm.exceptions import (
    LLMProviderError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)

class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider adapter."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
        timeout: float = 15.0,
        fallback_to_mock: bool = True,
        max_retries: int = 3,
        retry_delays: Optional[List[float]] = None,
        backoff_factor: float = 0.2
    ):
        self.api_key = api_key
        self._model_name = model_name
        self.timeout = timeout
        self.fallback_to_mock = fallback_to_mock
        self.max_retries = max_retries
        self.retry_delays = retry_delays if retry_delays is not None else [0.2, 0.4, 0.8]
        self.backoff_factor = backoff_factor
        self._mock_fallback = MockProvider(model_name=f"mock-{model_name}")

    @property
    def provider_name(self) -> str:
        return "GEMINI"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def _build_payload(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.2),
            }
        }
        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }
        if "max_tokens" in kwargs:
            payload["generationConfig"]["maxOutputTokens"] = kwargs["max_tokens"]
        return payload

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMGenerationResponse:
        if not self.is_available:
            if self.fallback_to_mock:
                res = self._mock_fallback.generate(prompt, system_prompt, **kwargs)
                res.provider = self.provider_name
                res.is_fallback = True
                return res
            raise LLMAuthenticationError("GEMINI_API_KEY is missing or empty.", provider=self.provider_name, status_code=401)

        payload = self._build_payload(prompt, system_prompt, **kwargs)
        data = json.dumps(payload).encode("utf-8")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        body = None
        latency_ms = 0.0
        attempt = 0
        last_error = None

        while attempt <= self.max_retries:
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                    body = json.loads(resp.read().decode("utf-8"))
                    break
            except urllib.error.HTTPError as e:
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                if e.code in (400, 401, 403):
                    raise LLMAuthenticationError(f"Gemini Auth Failed: {e.reason}", provider=self.provider_name, status_code=e.code)
                elif e.code == 429:
                    last_error = LLMRateLimitError(f"Gemini Rate Limit Exceeded: {e.reason}", provider=self.provider_name, status_code=e.code)
                elif e.code in (500, 502, 503, 504):
                    last_error = LLMUnavailableError(f"Gemini Service Unavailable: {e.reason}", provider=self.provider_name, status_code=e.code)
                else:
                    raise LLMProviderError(f"Gemini HTTP Error {e.code}: {e.reason}", provider=self.provider_name, status_code=e.code)
            except (urllib.error.URLError, TimeoutError) as e:
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                if "timed out" in str(e).lower():
                    last_error = LLMTimeoutError(f"Gemini request timed out after {self.timeout}s: {e}", provider=self.provider_name)
                else:
                    last_error = LLMUnavailableError(f"Gemini connection error: {e}", provider=self.provider_name)

            if attempt < self.max_retries:
                delays = self.retry_delays
                delay = delays[min(attempt, len(delays) - 1)]
                time.sleep(delay)
                attempt += 1
            else:
                if last_error:
                    raise last_error
                break


        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        candidates = body.get("candidates", [])
        text = ""
        finish_reason = "stop"
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            finish_reason = candidates[0].get("finishReason", "stop")

        usage = body.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)

        return LLMGenerationResponse(
            text=text,
            model=self.model_name,
            provider=self.provider_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            is_fallback=False,
            metadata={"gemini_response": True}
        )

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Iterator[StreamChunk]:
        if not self.is_available:
            if self.fallback_to_mock:
                for chunk in self._mock_fallback.stream_generate(prompt, system_prompt, **kwargs):
                    chunk.metadata["fallback"] = True
                    yield chunk
                return
            raise LLMAuthenticationError("GEMINI_API_KEY is missing or empty.", provider=self.provider_name, status_code=401)

        payload = self._build_payload(prompt, system_prompt, **kwargs)
        data = json.dumps(payload).encode("utf-8")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:streamGenerateContent?key={self.api_key}"
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403):
                raise LLMAuthenticationError(f"Gemini Auth Failed: {e.reason}", provider=self.provider_name, status_code=e.code)
            elif e.code == 429:
                raise LLMRateLimitError(f"Gemini Rate Limit: {e.reason}", provider=self.provider_name, status_code=e.code)
            raise LLMProviderError(f"Gemini HTTP Error {e.code}: {e.reason}", provider=self.provider_name, status_code=e.code)
        except (urllib.error.URLError, TimeoutError) as e:
            if "timed out" in str(e).lower():
                raise LLMTimeoutError(f"Gemini request timed out: {e}", provider=self.provider_name)
            raise LLMUnavailableError(f"Gemini connection error: {e}", provider=self.provider_name)

        chunk_idx = 0
        with resp:
            content = resp.read().decode("utf-8")
            try:
                chunks_json = json.loads(content)
                if isinstance(chunks_json, list):
                    for i, c in enumerate(chunks_json):
                        candidates = c.get("candidates", [])
                        piece = ""
                        finish_reason = None
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            piece = "".join(p.get("text", "") for p in parts)
                            finish_reason = candidates[0].get("finishReason")
                        is_final = (i == len(chunks_json) - 1)
                        yield StreamChunk(
                            text=piece,
                            chunk_index=chunk_idx,
                            is_final=is_final,
                            finish_reason=finish_reason,
                            token_count=1 if piece else 0
                        )
                        chunk_idx += 1
            except Exception:
                # Fallback to single chunk
                yield StreamChunk(text="", chunk_index=0, is_final=True, finish_reason="stop", token_count=0)
