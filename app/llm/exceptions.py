"""
LLM Provider Exceptions Hierarchy.
Provides standardized error classes for LLM provider integration.
"""

class LLMProviderError(Exception):
    """Base exception for all LLM provider related errors."""
    def __init__(self, message: str, provider: str = "unknown", status_code: int = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.details = details or {}

class LLMAuthenticationError(LLMProviderError):
    """Raised when authentication fails (e.g. invalid or missing API key, HTTP 401/403)."""
    pass

class LLMRateLimitError(LLMProviderError):
    """Raised when rate limits or quotas are exceeded (e.g. HTTP 429)."""
    pass

class LLMTimeoutError(LLMProviderError):
    """Raised when an LLM provider request times out."""
    pass

class LLMUnavailableError(LLMProviderError):
    """Raised when the LLM provider service is temporarily unavailable (e.g. HTTP 503) or offline."""
    pass
