"""
Anthropic (Claude) LLM Provider implementation.

Ye file Anthropic API ko call karti hai with retry logic.
"""

from __future__ import annotations

import asyncio
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from base import (
    APIConnectionError,
    AuthenticationError,
    BaseLLMProvider,
    InvalidRequestError,
    LLMResponse,
    RateLimitError,
)


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic (Claude) LLM provider implementation.
    
    Supports Claude models with async operations and retry logic.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-haiku-20241022",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key
            model: Model name (default: claude-3-5-haiku-20241022)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            max_retries: Maximum retry attempts
        """
        super().__init__(api_key, model, temperature, max_tokens)
        self.max_retries = max_retries
        
        # Initialize async client
        self.client = AsyncAnthropic(api_key=api_key)

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Send chat completion request with retry logic.
        
        Args:
            messages: List of message dictionaries
            temperature: Override temperature (optional)
            max_tokens: Override max_tokens (optional)
            
        Returns:
            LLMResponse with generated content
            
        Raises:
            APIConnectionError: Connection failed
            RateLimitError: Rate limit exceeded
            AuthenticationError: Invalid API key
            InvalidRequestError: Invalid request
        """
        # Use provided values or defaults
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        # Extract system message if present
        system_message = None
        api_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                api_messages.append(msg)

        # Retry loop with exponential backoff
        for attempt in range(self.max_retries):
            try:
                # Anthropic API call
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": api_messages,
                    "temperature": temp,
                    "max_tokens": max_tok,
                }
                
                if system_message:
                    kwargs["system"] = system_message

                response = await self.client.messages.create(**kwargs)

                # Extract response data
                content = response.content[0].text if response.content else ""
                
                # Get token usage
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens

                return LLMResponse(
                    content=content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=response.model,
                    finish_reason=response.stop_reason,
                )

            except anthropic.APIConnectionError as e:
                if attempt == self.max_retries - 1:
                    raise APIConnectionError(f"Connection failed: {str(e)}") from e
                await self._wait_with_backoff(attempt)

            except anthropic.RateLimitError as e:
                if attempt == self.max_retries - 1:
                    raise RateLimitError(f"Rate limit exceeded: {str(e)}") from e
                await self._wait_with_backoff(attempt)

            except anthropic.AuthenticationError as e:
                # No retry for auth errors
                raise AuthenticationError(f"Authentication failed: {str(e)}") from e

            except anthropic.BadRequestError as e:
                # No retry for bad requests
                raise InvalidRequestError(f"Invalid request: {str(e)}") from e

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise APIConnectionError(f"Unexpected error: {str(e)}") from e
                await self._wait_with_backoff(attempt)

        # This should never be reached due to raises above
        raise APIConnectionError("Max retries exceeded")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text (approximate for Anthropic).
        
        Anthropic doesn't provide a public tokenizer, so we approximate.
        Rule of thumb: 1 token ≈ 4 characters
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Approximate number of tokens
        """
        return len(text) // 4

    @staticmethod
    async def _wait_with_backoff(attempt: int) -> None:
        """
        Wait with exponential backoff.
        
        Args:
            attempt: Current attempt number (0-indexed)
        """
        wait_time = 2 ** attempt  # 1, 2, 4, 8 seconds
        await asyncio.sleep(wait_time)

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return "anthropic"

    async def close(self) -> None:
        """Close the async client."""
        await self.client.close()

    async def __aenter__(self) -> AnthropicProvider:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    async def test_anthropic():
        """Test Anthropic provider."""
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        
        if not api_key or api_key.startswith("sk-ant-your-"):
            print("❌ Please set ANTHROPIC_API_KEY in .env file")
            return

        print("Testing Anthropic Provider")
        print("=" * 60)

        async with AnthropicProvider(
            api_key=api_key, 
            model="claude-3-5-haiku-20241022"
        ) as provider:
            print(f"Provider: {provider}")
            print(f"Model: {provider.model}\n")

            # Test token counting
            text = "Hello, how are you?"
            tokens = provider.count_tokens(text)
            print(f"Text: '{text}'")
            print(f"Tokens (approx): {tokens}\n")

            # Test chat completion
            print("Sending chat request...")
            messages = [{"role": "user", "content": "Say hello in 5 words"}]

            try:
                response = await provider.chat(messages)
                print(f"\n✅ Response received:")
                print(f"   Content: {response.content}")
                print(f"   Tokens: {response.input_tokens} → {response.output_tokens}")
                print(f"   Model: {response.model}")
                print(f"   {response}")
            except Exception as e:
                print(f"\n❌ Error: {e}")

    # Run test
    print("Note: This will make an actual API call and cost ~$0.00002\n")
    asyncio.run(test_anthropic())