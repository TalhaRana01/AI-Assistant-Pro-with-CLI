"""
OpenAI LLM Provider implementation.

Ye file OpenAI API ko call karti hai with retry logic.
"""

from __future__ import annotations

import asyncio
from typing import Any

import openai
import tiktoken
from openai import AsyncOpenAI

from base import (
    APIConnectionError,
    AuthenticationError,
    BaseLLMProvider,
    InvalidRequestError,
    LLMResponse,
    RateLimitError,
)


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM provider implementation.
    
    Supports GPT models with async operations and retry logic.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-4o-mini)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            max_retries: Maximum retry attempts
        """
        super().__init__(api_key, model, temperature, max_tokens)
        self.max_retries = max_retries
        
        # Initialize async client
        self.client = AsyncOpenAI(api_key=api_key)
        
        # Initialize tokenizer for this model
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            self.encoding = tiktoken.get_encoding("cl100k_base")

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

        # Retry loop with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok,
                )

                # Extract response data
                choice = response.choices[0]
                content = choice.message.content or ""
                
                # Get token usage
                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0

                return LLMResponse(
                    content=content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=response.model,
                    finish_reason=choice.finish_reason,
                )

            except openai.APIConnectionError as e:
                if attempt == self.max_retries - 1:
                    raise APIConnectionError(f"Connection failed: {str(e)}") from e
                await self._wait_with_backoff(attempt)

            except openai.RateLimitError as e:
                if attempt == self.max_retries - 1:
                    raise RateLimitError(f"Rate limit exceeded: {str(e)}") from e
                await self._wait_with_backoff(attempt)

            except openai.AuthenticationError as e:
                # No retry for auth errors
                raise AuthenticationError(f"Authentication failed: {str(e)}") from e

            except openai.BadRequestError as e:
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
        Count tokens in text using tiktoken.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        try:
            return len(self.encoding.encode(text))
        except Exception:
            # Fallback: approximate
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
        return "openai"

    async def close(self) -> None:
        """Close the async client."""
        await self.client.close()

    async def __aenter__(self) -> OpenAIProvider:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    async def test_openai():
        """Test OpenAI provider."""
        api_key = os.getenv("OPENAI_API_KEY", "")
        
        if not api_key or api_key.startswith("sk-your-"):
            print("❌ Please set OPENAI_API_KEY in .env file")
            return

        print("Testing OpenAI Provider")
        print("=" * 60)

        async with OpenAIProvider(api_key=api_key, model="gpt-4o-mini") as provider:
            print(f"Provider: {provider}")
            print(f"Model: {provider.model}\n")

            # Test token counting
            text = "Hello, how are you?"
            tokens = provider.count_tokens(text)
            print(f"Text: '{text}'")
            print(f"Tokens: {tokens}\n")

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
    print("Note: This will make an actual API call and cost ~$0.00001\n")
    asyncio.run(test_openai())