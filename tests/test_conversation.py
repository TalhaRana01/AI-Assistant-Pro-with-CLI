"""
Tests for conversation management.

Test karte hain ke messages sahi add ho rahe hain aur history maintain ho rahi hai.
"""

from __future__ import annotations

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from src.utils.conversation import Conversation, Message


class TestMessage:
    """Test Message dataclass."""

    def test_message_creation(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello!")

        assert msg.role == "user"
        assert msg.content == "Hello!"

    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        msg = Message(role="assistant", content="Hi there!")
        msg_dict = msg.to_dict()

        assert msg_dict == {"role": "assistant", "content": "Hi there!"}

    def test_message_str_representation(self):
        """Test string representation of message."""
        msg = Message(role="user", content="Test message")
        assert str(msg) == "User: Test message"


class TestConversation:
    """Test Conversation class."""

    def test_conversation_creation_empty(self):
        """Test creating an empty conversation."""
        conv = Conversation()

        assert len(conv.messages) == 0
        assert conv.system_prompt is None

    def test_conversation_creation_with_system_prompt(self):
        """Test creating conversation with system prompt."""
        system_prompt = "You are a helpful assistant."
        conv = Conversation(system_prompt=system_prompt)

        assert len(conv.messages) == 1
        assert conv.messages[0].role == "system"
        assert conv.messages[0].content == system_prompt

    def test_add_user_message(self):
        """Test adding user message."""
        conv = Conversation()
        conv.add_user_message("Hello!")

        assert len(conv.messages) == 1
        assert conv.messages[0].role == "user"
        assert conv.messages[0].content == "Hello!"

    def test_add_assistant_message(self):
        """Test adding assistant message."""
        conv = Conversation()
        conv.add_assistant_message("Hi there!")

        assert len(conv.messages) == 1
        assert conv.messages[0].role == "assistant"
        assert conv.messages[0].content == "Hi there!"

    def test_add_system_message(self):
        """Test adding system message."""
        conv = Conversation()
        conv.add_system_message("System instruction")

        assert len(conv.messages) == 1
        assert conv.messages[0].role == "system"
        assert conv.messages[0].content == "System instruction"

    def test_get_messages(self):
        """Test getting messages in API format."""
        conv = Conversation()
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi")

        messages = conv.get_messages()

        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi"}

    def test_conversation_flow(self):
        """Test a complete conversation flow."""
        conv = Conversation(system_prompt="Be helpful.")

        conv.add_user_message("What is 2+2?")
        conv.add_assistant_message("2+2 equals 4.")
        conv.add_user_message("Thanks!")
        conv.add_assistant_message("You're welcome!")

        assert len(conv.messages) == 5  # 1 system + 4 messages
        assert conv.get_message_count() == 5

    def test_clear_conversation(self):
        """Test clearing conversation history."""
        conv = Conversation()
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi")

        assert len(conv.messages) == 2

        conv.clear()

        assert len(conv.messages) == 0

    def test_clear_conversation_preserves_system_prompt(self):
        """Test that clear preserves system prompt."""
        system_prompt = "You are helpful."
        conv = Conversation(system_prompt=system_prompt)
        conv.add_user_message("Hello")

        assert len(conv.messages) == 2

        conv.clear()

        assert len(conv.messages) == 1
        assert conv.messages[0].role == "system"
        assert conv.messages[0].content == system_prompt

    def test_get_last_user_message(self):
        """Test getting last user message."""
        conv = Conversation()
        conv.add_user_message("First")
        conv.add_assistant_message("Response")
        conv.add_user_message("Second")

        last_user_msg = conv.get_last_user_message()

        assert last_user_msg == "Second"

    def test_get_last_assistant_message(self):
        """Test getting last assistant message."""
        conv = Conversation()
        conv.add_assistant_message("First response")
        conv.add_user_message("Question")
        conv.add_assistant_message("Second response")

        last_assistant_msg = conv.get_last_assistant_message()

        assert last_assistant_msg == "Second response"

    def test_get_last_message_when_empty(self):
        """Test getting last message from empty conversation."""
        conv = Conversation()

        assert conv.get_last_user_message() is None
        assert conv.get_last_assistant_message() is None

    def test_conversation_length(self):
        """Test __len__ method."""
        conv = Conversation()
        assert len(conv) == 0

        conv.add_user_message("Test")
        assert len(conv) == 1

        conv.add_assistant_message("Response")
        assert len(conv) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])