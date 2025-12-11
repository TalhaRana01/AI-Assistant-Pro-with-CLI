"""
Configuration management using Pydantic Settings.

Ye file .env se settings load karti hai aur validate karti hai.
"""

from __future__ import annotations

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Attributes:
        openai_api_key: OpenAI API key (secure)
        anthropic_api_key: Anthropic API key (secure)
        default_provider: Default LLM provider to use
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens in response
        cost_warning_threshold: Warning threshold for costs
        cost_limit_threshold: Hard limit for costs
        log_level: Logging level
    """

    # API Keys (SecretStr = kabhi print nahi hoga)
    openai_api_key: SecretStr = Field(
        ...,  # Required field
        description="OpenAI API key from https://platform.openai.com/api-keys",
    )
    anthropic_api_key: SecretStr = Field(
        ...,
        description="Anthropic API key from https://console.anthropic.com/settings/keys",
    )

    # Model Configuration
    default_provider: str = Field(
        default="openai",
        description="Default LLM provider: 'openai' or 'anthropic'",
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature (0.0-2.0)"
    )
    max_tokens: int = Field(default=1000, gt=0, description="Maximum tokens in response")

    # Cost Tracking
    cost_warning_threshold: float = Field(
        default=0.10, ge=0.0, description="Cost warning threshold in USD"
    )
    cost_limit_threshold: float = Field(
        default=1.00, ge=0.0, description="Hard cost limit in USD"
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR",
    )

    # Pydantic settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",  # .env file se load karo
        env_file_encoding="utf-8",
        case_sensitive=False,  # OPENAI_API_KEY = openai_api_key
        extra="ignore",  # Extra fields ignore karo
    )

    @field_validator("default_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate that provider is either 'openai' or 'anthropic'."""
        allowed = {"openai", "anthropic"}
        if v.lower() not in allowed:
            raise ValueError(f"Provider must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}, got '{v}'")
        return v.upper()


# Global settings instance
# Is se poori app mein settings access kar sakte ho
def get_settings() -> Settings:
    """
    Get application settings.
    
    Returns:
        Settings instance loaded from environment variables.
        
    Example:
        >>> settings = get_settings()
        >>> print(settings.temperature)
        0.7
    """
    return Settings()


if __name__ == "__main__":
    # Testing: Direct run karke dekho
    try:
        settings = get_settings()
        print("✅ Configuration loaded successfully!")
        print(f"Default Provider: {settings.default_provider}")
        print(f"Temperature: {settings.temperature}")
        print(f"Max Tokens: {settings.max_tokens}")
        print(f"Log Level: {settings.log_level}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")