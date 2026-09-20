"""
LLM Abstraction Layer Module.
"""
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

__all__ = [
    "BaseLLMProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "MockProvider",
    "StreamChunk",
    "LLMGenerationResponse",
    "LLMProviderError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "get_llm_provider",
]
