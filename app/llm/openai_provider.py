"""
OpenAI LLM Provider Adapter.
Implements chat completions and SSE streaming using standard library urllib.
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

class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider adapter."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
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
        return "OPENAI"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def _build_payload(self, prompt: str, system_prompt: Optional[str] = None, stream: bool = False, **kwargs) -> Dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.2),
            "stream": stream
        }
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]
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
            raise LLMAuthenticationError("OPENAI_API_KEY is missing or empty.", provider=self.provider_name, status_code=401)

        payload = self._build_payload(prompt, system_prompt, stream=False, **kwargs)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
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
                if e.code in (401, 403):
                    raise LLMAuthenticationError(f"OpenAI Authentication Failed: {e.reason}", provider=self.provider_name, status_code=e.code)
                elif e.code == 429:
                    last_error = LLMRateLimitError(f"OpenAI Rate Limit Exceeded: {e.reason}", provider=self.provider_name, status_code=e.code)
                elif e.code in (500, 502, 503, 504):
                    last_error = LLMUnavailableError(f"OpenAI Service Unavailable: {e.reason}", provider=self.provider_name, status_code=e.code)
                else:
                    raise LLMProviderError(f"OpenAI HTTP Error {e.code}: {e.reason}", provider=self.provider_name, status_code=e.code)
            except (urllib.error.URLError, TimeoutError) as e:
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                if "timed out" in str(e).lower():
                    last_error = LLMTimeoutError(f"OpenAI request timed out after {self.timeout}s: {e}", provider=self.provider_name)
                else:
                    last_error = LLMUnavailableError(f"OpenAI connection error: {e}", provider=self.provider_name)

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
        choice = body["choices"][0]
        text = choice["message"]["content"]
        usage = body.get("usage", {})

        return LLMGenerationResponse(
            text=text,
            model=body.get("model", self.model_name),
            provider=self.provider_name,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
            latency_ms=latency_ms,
            is_fallback=False,
            metadata={"id": body.get("id")}
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
            raise LLMAuthenticationError("OPENAI_API_KEY is missing or empty.", provider=self.provider_name, status_code=401)

        payload = self._build_payload(prompt, system_prompt, stream=True, **kwargs)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            method="POST"
        )

        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise LLMAuthenticationError(f"OpenAI Auth Failed: {e.reason}", provider=self.provider_name, status_code=e.code)
            elif e.code == 429:
                raise LLMRateLimitError(f"OpenAI Rate Limit: {e.reason}", provider=self.provider_name, status_code=e.code)
            raise LLMProviderError(f"OpenAI HTTP Error {e.code}: {e.reason}", provider=self.provider_name, status_code=e.code)
        except (urllib.error.URLError, TimeoutError) as e:
            if "timed out" in str(e).lower():
                raise LLMTimeoutError(f"OpenAI request timed out: {e}", provider=self.provider_name)
            raise LLMUnavailableError(f"OpenAI connection error: {e}", provider=self.provider_name)

        chunk_idx = 0
        with resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    yield StreamChunk(
                        text="",
                        chunk_index=chunk_idx,
                        is_final=True,
                        finish_reason="stop",
                        token_count=0
                    )
                    break
                try:
                    chunk_json = json.loads(data_str)
                    delta = chunk_json["choices"][0].get("delta", {})
                    content_piece = delta.get("content", "")
                    finish_reason = chunk_json["choices"][0].get("finish_reason")
                    is_final = finish_reason is not None

                    yield StreamChunk(
                        text=content_piece,
                        chunk_index=chunk_idx,
                        is_final=is_final,
                        finish_reason=finish_reason,
                        token_count=1 if content_piece else 0,
                        metadata={"id": chunk_json.get("id")}
                    )
                    chunk_idx += 1
                except Exception:
                    continue
