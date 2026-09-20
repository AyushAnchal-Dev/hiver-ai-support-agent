"""
LLM Provider Contracts and Data Models.
Defines output schemas for generation and streaming.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class StreamChunk:
    """A streaming token or delta chunk emitted by an LLM provider."""
    text: str
    chunk_index: int
    is_final: bool = False
    finish_reason: Optional[str] = None
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMGenerationResponse:
    """Complete structured response from an LLM provider."""
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    is_fallback: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
