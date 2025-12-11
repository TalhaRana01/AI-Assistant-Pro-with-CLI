"""
Abstract base class for LLM providers.

Ye file ek blueprint hai - har provider ko ye implement karna hoga.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """
    Response from LLM API call.
    
    Attributes:
        content: Generated text response
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated
        model: Model name used
        finish_reason: Why generation stopped (e.g., 'stop', 'length')
    """

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    finish_reason: str | None = None

    def __str__(self) -> str:
        return f"LLMResponse(tokens: {self.input_tokens}→{self.output_tokens}, model: {self.model})"


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Har provider (OpenAI, Anthropic) ko ye class inherit karni hogi
    aur required methods implement karne honge.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> None:
        """
        Initialize LLM provider.
        
        Args:
            api_key: API key for authentication
            model: Model name to use
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Send chat completion request.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Override default temperature (optional)
            max_tokens: Override default max_tokens (optional)
            
        Returns:
            LLMResponse object with generated content and token counts
            
        Raises:
            Exception: Various API-related exceptions
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Get provider name.
        
        Returns:
            Provider name (e.g., 'openai', 'anthropic')
        """
        pass

    def __str__(self) -> str:
        return f"{self.provider_name.upper()} Provider (model: {self.model})"


class ProviderError(Exception):
    """Base exception for provider-related errors."""

    pass


class APIConnectionError(ProviderError):
    """Exception raised when API connection fails."""

    pass


class RateLimitError(ProviderError):
    """Exception raised when rate limit is exceeded."""

    pass


class AuthenticationError(ProviderError):
    """Exception raised when authentication fails."""

    pass


class InvalidRequestError(ProviderError):
    """Exception raised when request is invalid."""

    pass


if __name__ == "__main__":
    print("Base LLM Provider Classes")
    print("=" * 60)
    print("\nYe file ek blueprint hai - abstract base class.")
    print("OpenAI aur Anthropic providers isko inherit karenge.\n")
    
    print("Required Methods:")
    print("  1. chat() - Send messages and get response")
    print("  2. count_tokens() - Count tokens in text")
    print("  3. provider_name - Provider ka naam\n")
    
    print("LLMResponse Structure:")
    response = LLMResponse(
        content="Hello! How can I help you?",
        input_tokens=10,
        output_tokens=7,
        model="gpt-4o-mini",
        finish_reason="stop"
    )
    print(f"  {response}")
    print(f"  Content: {response.content}")
    print(f"  Tokens: {response.input_tokens} → {response.output_tokens}")
    
    print("\n" + "=" * 60)