"""
Conversation history management.

Ye file chat history ko manage karti hai - messages store karti hai.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Message:
    """
    Single message in conversation.
    
    Attributes:
        role: Message sender - 'user', 'assistant', or 'system'
        content: Message content text
    """

    role: Literal["user", "assistant", "system"]
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert message to dictionary format."""
        return {"role": self.role, "content": self.content}

    def __str__(self) -> str:
        """String representation of message."""
        return f"{self.role.capitalize()}: {self.content}"


@dataclass
class Conversation:
    """
    Manages conversation history.
    
    Attributes:
        messages: List of messages in conversation
        system_prompt: Optional system prompt for the conversation
    """

    messages: list[Message] = field(default_factory=list)
    system_prompt: str | None = None

    def __post_init__(self) -> None:
        """Add system prompt as first message if provided."""
        if self.system_prompt and not self.messages:
            self.add_system_message(self.system_prompt)

    def add_user_message(self, content: str) -> None:
        """
        Add user message to conversation.
        
        Args:
            content: User message text
            
        Example:
            >>> conv = Conversation()
            >>> conv.add_user_message("Hello!")
        """
        self.messages.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        """
        Add assistant message to conversation.
        
        Args:
            content: Assistant response text
        """
        self.messages.append(Message(role="assistant", content=content))

    def add_system_message(self, content: str) -> None:
        """
        Add system message to conversation.
        
        Args:
            content: System prompt text
        """
        self.messages.append(Message(role="system", content=content))

    def get_messages(self) -> list[dict[str, str]]:
        """
        Get all messages in API-compatible format.
        
        Returns:
            List of message dictionaries
            
        Example:
            >>> conv = Conversation()
            >>> conv.add_user_message("Hi")
            >>> conv.get_messages()
            [{'role': 'user', 'content': 'Hi'}]
        """
        return [msg.to_dict() for msg in self.messages]

    def clear(self) -> None:
        """
        Clear all messages from conversation.
        
        Preserves system prompt if it exists.
        """
        if self.system_prompt:
            self.messages = [Message(role="system", content=self.system_prompt)]
        else:
            self.messages = []

    def get_message_count(self) -> int:
        """Get total number of messages."""
        return len(self.messages)

    def get_last_user_message(self) -> str | None:
        """
        Get the last user message.
        
        Returns:
            Last user message content or None if no user messages
        """
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None

    def get_last_assistant_message(self) -> str | None:
        """
        Get the last assistant message.
        
        Returns:
            Last assistant message content or None if no assistant messages
        """
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.content
        return None

    def __len__(self) -> int:
        """Return number of messages."""
        return len(self.messages)

    def __str__(self) -> str:
        """String representation of conversation."""
        return f"Conversation({len(self.messages)} messages)"


if __name__ == "__main__":
    # Test conversation manager
    print("Testing Conversation Manager\n" + "=" * 50)

    # Create conversation with system prompt
    conv = Conversation(system_prompt="You are a helpful AI assistant.")
    print(f"Created: {conv}")
    print(f"Messages: {conv.get_message_count()}\n")

    # Add messages
    conv.add_user_message("Hello! What's the weather?")
    conv.add_assistant_message("I don't have access to real-time weather data.")
    conv.add_user_message("Can you tell me a joke?")
    conv.add_assistant_message("Why did the programmer quit? Because they didn't get arrays!")

    # Display conversation
    print("Conversation History:")
    for msg in conv.messages:
        print(f"  {msg}")

    print(f"\nTotal messages: {conv.get_message_count()}")
    print(f"Last user message: {conv.get_last_user_message()}")
    print(f"Last assistant message: {conv.get_last_assistant_message()}")

    # Get API format
    print("\nAPI Format:")
    for msg in conv.get_messages():
        print(f"  {msg}")

    # Clear conversation
    print("\n" + "=" * 50)
    print("Clearing conversation...")
    conv.clear()
    print(f"Messages after clear: {conv.get_message_count()}")
    print("System prompt preserved:", conv.messages[0].content if conv.messages else "No")