"""
LLM Provider Factory.
Instantiates and configures LLM providers based on settings or explicit parameters.
"""
from typing import Optional, Dict, Any

from app.settings import Settings, get_settings
from app.llm.base_provider import BaseLLMProvider
from app.llm.mock_provider import MockProvider
from app.llm.openai_provider import OpenAIProvider
from app.llm.gemini_provider import GeminiProvider

def get_llm_provider(
    settings: Optional[Settings] = None,
    provider_name: Optional[str] = None,
    **kwargs
) -> BaseLLMProvider:
    """
    Factory function resolving the configured LLM inference provider.
    Priority:
    1. Explicit provider_name argument
    2. settings.llm_provider
    3. Fallback to MockProvider
    """
    app_settings = settings or get_settings()
    p_name = (provider_name or app_settings.llm_provider or "MOCK").upper()

    if p_name == "OPENAI":
        return OpenAIProvider(
            api_key=kwargs.get("api_key", app_settings.openai_api_key),
            model_name=kwargs.get("model_name", app_settings.openai_model),
            timeout=kwargs.get("timeout", app_settings.llm_timeout_seconds),
            fallback_to_mock=kwargs.get("fallback_to_mock", True)
        )
    elif p_name == "GEMINI":
        return GeminiProvider(
            api_key=kwargs.get("api_key", app_settings.gemini_api_key),
            model_name=kwargs.get("model_name", app_settings.gemini_model),
            timeout=kwargs.get("timeout", app_settings.llm_timeout_seconds),
            fallback_to_mock=kwargs.get("fallback_to_mock", True)
        )
    elif p_name == "MOCK":
        return MockProvider(
            model_name=kwargs.get("model_name", "mock-grounded-v1"),
            simulate_error=kwargs.get("simulate_error"),
            default_response=kwargs.get("default_response"),
            latency_seconds=kwargs.get("latency_seconds", 0.005)
        )
    else:
        # Default fallback to mock
        return MockProvider(model_name="mock-fallback-v1")
