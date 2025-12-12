"""
Tests for main application logic.

Test karte hain ke AIAssistant class sahi kaam kar rahi hai.
"""

from __future__ import annotations

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.main import AIAssistant
from src. llm.base import LLMResponse


class TestAIAssistant:
    """Test AIAssistant class."""

    @pytest.fixture
    def mock_settings(self, monkeypatch):
        """Mock settings for testing."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        monkeypatch.setenv("DEFAULT_PROVIDER", "openai")

    def test_assistant_initialization(self, mock_settings):
        """Test that assistant initializes correctly."""
        assistant = AIAssistant()

        assert assistant.current_provider_name == "openai"
        assert assistant.conversation is not None
        assert assistant.cost_tracker is not None
        assert len(assistant.conversation.messages) == 1  # System prompt

    @pytest.mark.asyncio
    async def test_initialize_openai_provider(self, mock_settings):
        """Test initializing OpenAI provider."""
        assistant = AIAssistant()
        
        await assistant.initialize_provider("openai")

        assert assistant.provider is not None
        assert assistant.provider.provider_name == "openai"
        assert assistant.current_provider_name == "openai"

    @pytest.mark.asyncio
    async def test_initialize_anthropic_provider(self, mock_settings):
        """Test initializing Anthropic provider."""
        assistant = AIAssistant()
        
        await assistant.initialize_provider("anthropic")

        assert assistant.provider is not None
        assert assistant.provider.provider_name == "anthropic"
        assert assistant.current_provider_name == "anthropic"

    @pytest.mark.asyncio
    async def test_send_message(self, mock_settings):
        """Test sending a message."""
        assistant = AIAssistant()
        await assistant.initialize_provider("openai")

        # Mock the provider's chat method
        mock_response = LLMResponse(
            content="Hello! How can I help?",
            input_tokens=10,
            output_tokens=5,
            model="gpt-4o-mini",
        )
        
        assistant.provider.chat = AsyncMock(return_value=mock_response)

        # Send message
        response = await assistant.send_message("Hello")

        assert response == "Hello! How can I help?"
        assert len(assistant.conversation.messages) == 3  # system + user + assistant
        assert len(assistant.cost_tracker.entries) == 1

    @pytest.mark.asyncio
    async def test_send_message_tracks_cost(self, mock_settings):
        """Test that sending message tracks cost."""
        assistant = AIAssistant()
        await assistant.initialize_provider("openai")

        mock_response = LLMResponse(
            content="Response",
            input_tokens=100,
            output_tokens=50,
            model="gpt-4o-mini",
        )
        
        assistant.provider.chat = AsyncMock(return_value=mock_response)

        await assistant.send_message("Test")

        assert assistant.cost_tracker.get_total_cost() > 0
        assert len(assistant.cost_tracker.entries) == 1

    @pytest.mark.asyncio
    async def test_send_message_raises_on_cost_limit(self, mock_settings):
        """Test that sending message raises error when cost limit reached."""
        assistant = AIAssistant()
        assistant.cost_tracker.limit_threshold = 0.0001  # Very low limit
        
        await assistant.initialize_provider("openai")

        # Add some cost
        assistant.cost_tracker.add_entry("openai", "gpt-4o-mini", 10000, 5000)

        # Should raise error
        with pytest.raises(RuntimeError, match="Cost limit reached"):
            await assistant.send_message("Test")

    def test_handle_command_help(self, mock_settings):
        """Test /help command."""
        assistant = AIAssistant()
        result = assistant.handle_command("/help")

        assert result is not None
        assert "Available Commands" in result
        assert "/help" in result
        assert "/quit" in result

    def test_handle_command_clear(self, mock_settings):
        """Test /clear command."""
        assistant = AIAssistant()
        assistant.conversation.add_user_message("Test")
        
        result = assistant.handle_command("/clear")

        assert "cleared" in result.lower()
        # Only system message should remain
        assert len(assistant.conversation.messages) == 1

    def test_handle_command_quit(self, mock_settings):
        """Test /quit command."""
        assistant = AIAssistant()
        result = assistant.handle_command("/quit")

        assert result == "QUIT"

    def test_handle_command_exit(self, mock_settings):
        """Test /exit command."""
        assistant = AIAssistant()
        result = assistant.handle_command("/exit")

        assert result == "QUIT"

    def test_handle_command_model_switch(self, mock_settings):
        """Test /model command."""
        assistant = AIAssistant()
        result = assistant.handle_command("/model anthropic")

        assert result == "SWITCH_MODEL:anthropic"

    def test_handle_command_model_invalid(self, mock_settings):
        """Test /model command with invalid provider."""
        assistant = AIAssistant()
        result = assistant.handle_command("/model invalid")

        assert "must be" in result

    def test_handle_command_model_missing_arg(self, mock_settings):
        """Test /model command without provider."""
        assistant = AIAssistant()
        result = assistant.handle_command("/model")

        assert "Usage" in result

    def test_handle_command_cost(self, mock_settings):
        """Test /cost command."""
        assistant = AIAssistant()
        assistant.cost_tracker.add_entry("openai", "gpt-4o-mini", 1000, 500)
        
        result = assistant.handle_command("/cost")

        assert "COST SUMMARY" in result
        assert "$" in result

    def test_handle_command_history_empty(self, mock_settings):
        """Test /history command with empty history."""
        assistant = AIAssistant()
        assistant.conversation.clear()
        
        result = assistant.handle_command("/history")

        assert "No conversation history" in result

    def test_handle_command_history_with_messages(self, mock_settings):
        """Test /history command with messages."""
        assistant = AIAssistant()
        assistant.conversation.add_user_message("Hello")
        assistant.conversation.add_assistant_message("Hi")
        
        result = assistant.handle_command("/history")

        assert "Conversation History" in result

    def test_handle_command_unknown(self, mock_settings):
        """Test unknown command."""
        assistant = AIAssistant()
        result = assistant.handle_command("/unknown")

        assert result is None

    @pytest.mark.asyncio
    async def test_cleanup(self, mock_settings):
        """Test cleanup method."""
        assistant = AIAssistant()
        await assistant.initialize_provider("openai")
        
        # Mock the close method
        assistant.provider.close = AsyncMock()

        await assistant.cleanup()

        assistant.provider.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_switch_provider_closes_old(self, mock_settings):
        """Test that switching provider closes old one."""
        assistant = AIAssistant()
        
        # Initialize first provider
        await assistant.initialize_provider("openai")
        old_provider = assistant.provider
        old_provider.close = AsyncMock()

        # Switch to new provider
        await assistant.initialize_provider("anthropic")

        # Old provider should be closed
        old_provider.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])