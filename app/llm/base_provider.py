"""
Abstract Base LLM Provider Interface.
Defines the required protocol for LLM text generation and token streaming.
"""
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Dict, Any

from app.llm.contracts import StreamChunk, LLMGenerationResponse

class BaseLLMProvider(ABC):
    """Abstract interface for all LLM inference providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the canonical name of the provider (e.g. 'OPENAI', 'GEMINI', 'MOCK')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the active model identifier."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Indicates whether credentials and network connectivity allow live inference."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMGenerationResponse:
        """
        Synchronously generates a complete completion for the given prompt.
        Returns a typed LLMGenerationResponse with token counts and latency.
        """
        pass

    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Iterator[StreamChunk]:
        """
        Streams generated token chunks incrementally.
        Yields typed StreamChunk instances with chunk_index and metadata.
        """
        pass
