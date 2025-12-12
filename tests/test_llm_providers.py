"""
Tests for LLM providers.

Mock API calls to avoid actual costs during testing.
"""

from __future__ import annotations

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.llm.base import LLMResponse
from src.llm.openai_provider import OpenAIProvider
from src.llm.anthropic_provider import AnthropicProvider


class TestLLMResponse:
    """Test LLMResponse dataclass."""

    def test_llm_response_creation(self):
        """Test creating an LLM response."""
        response = LLMResponse(
            content="Hello!",
            input_tokens=10,
            output_tokens=5,
            model="gpt-4o-mini",
            finish_reason="stop",
        )

        assert response.content == "Hello!"
        assert response.input_tokens == 10
        assert response.output_tokens == 5
        assert response.model == "gpt-4o-mini"
        assert response.finish_reason == "stop"

    def test_llm_response_str(self):
        """Test string representation."""
        response = LLMResponse(
            content="Test",
            input_tokens=10,
            output_tokens=5,
            model="gpt-4o-mini",
        )

        str_repr = str(response)
        assert "10" in str_repr
        assert "5" in str_repr
        assert "gpt-4o-mini" in str_repr


class TestOpenAIProvider:
    """Test OpenAI provider."""

    def test_provider_initialization(self):
        """Test initializing OpenAI provider."""
        provider = OpenAIProvider(
            api_key="sk-test-key",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000,
        )

        assert provider.api_key == "sk-test-key"
        assert provider.model == "gpt-4o-mini"
        assert provider.temperature == 0.7
        assert provider.max_tokens == 1000
        assert provider.provider_name == "openai"

    def test_count_tokens(self):
        """Test token counting."""
        provider = OpenAIProvider(api_key="sk-test-key")
        
        text = "Hello, world!"
        tokens = provider.count_tokens(text)

        assert tokens > 0
        assert isinstance(tokens, int)

    @pytest.mark.asyncio
    async def test_chat_with_mock(self):
        """Test chat method with mocked API call."""
        provider = OpenAIProvider(api_key="sk-test-key")

        # Mock the API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from GPT!"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.model = "gpt-4o-mini"

        # Patch the API call
        with patch.object(
            provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            messages = [{"role": "user", "content": "Hello"}]
            response = await provider.chat(messages)

            assert isinstance(response, LLMResponse)
            assert response.content == "Hello from GPT!"
            assert response.input_tokens == 10
            assert response.output_tokens == 5
            assert response.model == "gpt-4o-mini"

    def test_provider_str(self):
        """Test string representation."""
        provider = OpenAIProvider(api_key="sk-test-key", model="gpt-4o-mini")
        
        str_repr = str(provider)
        assert "OPENAI" in str_repr
        assert "gpt-4o-mini" in str_repr


class TestAnthropicProvider:
    """Test Anthropic provider."""

    def test_provider_initialization(self):
        """Test initializing Anthropic provider."""
        provider = AnthropicProvider(
            api_key="sk-ant-test-key",
            model="claude-3-5-haiku-20241022",
            temperature=0.7,
            max_tokens=1000,
        )

        assert provider.api_key == "sk-ant-test-key"
        assert provider.model == "claude-3-5-haiku-20241022"
        assert provider.temperature == 0.7
        assert provider.max_tokens == 1000
        assert provider.provider_name == "anthropic"

    def test_count_tokens_approximate(self):
        """Test approximate token counting."""
        provider = AnthropicProvider(api_key="sk-ant-test-key")
        
        text = "Hello, world!"  # 13 characters
        tokens = provider.count_tokens(text)

        # Approximate: 13 / 4 = 3.25 -> 3
        assert tokens == 3

    @pytest.mark.asyncio
    async def test_chat_with_mock(self):
        """Test chat method with mocked API call."""
        provider = AnthropicProvider(api_key="sk-ant-test-key")

        # Mock the API response
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Hello from Claude!"
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.model = "claude-3-5-haiku-20241022"

        # Patch the API call
        with patch.object(
            provider.client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            messages = [{"role": "user", "content": "Hello"}]
            response = await provider.chat(messages)

            assert isinstance(response, LLMResponse)
            assert response.content == "Hello from Claude!"
            assert response.input_tokens == 10
            assert response.output_tokens == 5
            assert response.model == "claude-3-5-haiku-20241022"

    @pytest.mark.asyncio
    async def test_chat_with_system_message(self):
        """Test chat with system message."""
        provider = AnthropicProvider(api_key="sk-ant-test-key")

        # Mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Response"
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 15
        mock_response.usage.output_tokens = 5
        mock_response.model = "claude-3-5-haiku-20241022"

        with patch.object(
            provider.client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_create:
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
            response = await provider.chat(messages)

            # Verify system message was extracted
            call_kwargs = mock_create.call_args[1]
            assert "system" in call_kwargs
            assert call_kwargs["system"] == "You are helpful."
            # Verify only non-system messages passed
            assert len(call_kwargs["messages"]) == 1

    def test_provider_str(self):
        """Test string representation."""
        provider = AnthropicProvider(
            api_key="sk-ant-test-key",
            model="claude-3-5-haiku-20241022"
        )
        
        str_repr = str(provider)
        assert "ANTHROPIC" in str_repr
        assert "claude-3-5-haiku-20241022" in str_repr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])