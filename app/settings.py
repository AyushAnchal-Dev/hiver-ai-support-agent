"""
Application Settings and Environment Configuration.
Provides a typed Settings dataclass and zero-dependency .env file parser.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

@dataclass(frozen=True)
class Settings:
    """Immutable application settings container."""
    base_dir: Path
    data_dir: Path
    report_dir: Path
    archive_dir: Path
    knowledge_dir: Path
    metrics_dir: Path
    logs_dir: Path

    # Server & Runtime Settings
    host: str = "0.0.0.0"
    port: int = 8000
    enable_warmup: bool = True
    log_level: str = "INFO"

    # LLM Settings
    llm_provider: str = "MOCK"  # "OPENAI", "GEMINI", "MOCK"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-1.5-flash"
    llm_timeout_seconds: float = 15.0
    llm_max_retries: int = 2
    offline_mode: bool = True

    # Prompt Settings
    prompt_version: str = "v1"

    # Brand Settings
    target_brand: str = "SpotifyCares"
    brand_signature: str = "/SpotifyCares"

    def validate(self) -> None:
        """
        Validates environment configuration on startup.
        Raises ValueError or FileNotFoundError with clear, actionable error messages.
        """
        valid_providers = {"MOCK", "OPENAI", "GEMINI"}
        if self.llm_provider not in valid_providers:
            raise ValueError(
                f"Invalid LLM_PROVIDER '{self.llm_provider}'. Expected one of {sorted(valid_providers)}."
            )

        if self.llm_provider == "OPENAI" and not self.offline_mode:
            if not self.openai_api_key or not self.openai_api_key.strip():
                raise ValueError(
                    "OPENAI_API_KEY is required when LLM_PROVIDER='OPENAI' and OFFLINE_MODE=false. "
                    "Please set OPENAI_API_KEY in your environment or .env file."
                )

        if self.llm_provider == "GEMINI" and not self.offline_mode:
            if not self.gemini_api_key or not self.gemini_api_key.strip():
                raise ValueError(
                    "GEMINI_API_KEY is required when LLM_PROVIDER='GEMINI' and OFFLINE_MODE=false. "
                    "Please set GEMINI_API_KEY in your environment or .env file."
                )

        prompt_dir = self.base_dir / "app" / "prompts" / self.prompt_version
        if not prompt_dir.exists():
            alt_prompt_dir = self.base_dir / "prompts" / self.prompt_version
            if not alt_prompt_dir.exists():
                raise FileNotFoundError(
                    f"Prompt template directory for PROMPT_VERSION='{self.prompt_version}' does not exist at {prompt_dir}."
                )

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

def _parse_env_file(filepath: Path) -> Dict[str, str]:
    """Lightweight, zero-dependency .env parser."""
    env_vars: Dict[str, str] = {}
    if not filepath.exists() or not filepath.is_file():
        return env_vars

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    env_vars[key] = val
    except Exception:
        pass
    return env_vars

_SETTINGS_CACHE: Optional[Settings] = None

def load_settings(base_dir: Optional[Path] = None, env_path: Optional[Path] = None, force_reload: bool = False) -> Settings:
    """
    Constructs and returns a typed Settings object.
    Loads values in priority:
    1. OS environment variables
    2. .env file variables
    3. Default fallbacks
    """
    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is not None and not force_reload:
        return _SETTINGS_CACHE

    root = base_dir or Path(__file__).resolve().parent.parent
    env_file = env_path or (root / ".env")
    env_vars = _parse_env_file(env_file)

    def get_val(key: str, default: Any = None) -> Any:
        return os.environ.get(key, env_vars.get(key, default))

    provider = str(get_val("LLM_PROVIDER", "MOCK")).upper()
    openai_key = get_val("OPENAI_API_KEY")
    gemini_key = get_val("GEMINI_API_KEY") or get_val("GOOGLE_API_KEY")

    offline_str = str(get_val("OFFLINE_MODE", "true")).lower()
    offline_mode = offline_str in ("true", "1", "yes")

    if get_val("OFFLINE_MODE") is None:
        if provider == "OPENAI" and not openai_key:
            offline_mode = True
        elif provider == "GEMINI" and not gemini_key:
            offline_mode = True

    host = str(get_val("HOST", "0.0.0.0"))
    port = int(get_val("PORT", 8000))
    warmup_str = str(get_val("ENABLE_WARMUP", "true")).lower()
    enable_warmup = warmup_str in ("true", "1", "yes")
    log_level = str(get_val("LOG_LEVEL", "INFO")).upper()

    settings = Settings(
        base_dir=root,
        data_dir=root / "data",
        report_dir=root / "report",
        archive_dir=root / "archive",
        knowledge_dir=root / "knowledge",
        metrics_dir=root / "metrics",
        logs_dir=root / "logs",
        host=host,
        port=port,
        enable_warmup=enable_warmup,
        log_level=log_level,
        llm_provider=provider,
        openai_api_key=openai_key,
        openai_model=str(get_val("OPENAI_MODEL", "gpt-4o-mini")),
        gemini_api_key=gemini_key,
        gemini_model=str(get_val("GEMINI_MODEL", "gemini-1.5-flash")),
        llm_timeout_seconds=float(get_val("LLM_TIMEOUT_SECONDS", 15.0)),
        llm_max_retries=int(get_val("LLM_MAX_RETRIES", 2)),
        offline_mode=offline_mode,
        prompt_version=str(get_val("PROMPT_VERSION", "v1")),
        target_brand=str(get_val("TARGET_BRAND", "SpotifyCares")),
        brand_signature=str(get_val("BRAND_SIGNATURE", "/SpotifyCares")),
    )

    settings.metrics_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)

    _SETTINGS_CACHE = settings
    return settings

def get_settings() -> Settings:
    """Returns the cached application settings instance."""
    return load_settings()

def validate_environment(settings: Optional[Settings] = None) -> Settings:
    """Validates settings and directory layout on startup."""
    s = settings or get_settings()
    s.validate()
    return s
