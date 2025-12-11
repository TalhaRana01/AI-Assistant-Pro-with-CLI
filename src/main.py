"""
Main CLI application for AI Assistant.

Ye main entry point hai - user yahan se interact karega.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Literal

from config import get_settings
from llm.anthropic_provider import AnthropicProvider
from llm.base import BaseLLMProvider
from llm.openai_provider import OpenAIProvider
from utils.conversation import Conversation
from utils.cost_tracker import CostTracker
from utils.logger import get_logger, log_api_call, log_error

# Initialize logger
logger = get_logger()

ProviderType = Literal["openai", "anthropic"]


class AIAssistant:
    """
    Main AI Assistant application.
    
    Manages conversation, cost tracking, and provider switching.
    """

    def __init__(self) -> None:
        """Initialize AI Assistant."""
        # Load settings
        self.settings = get_settings()
        logger.info("Settings loaded successfully")

        # Initialize conversation
        self.conversation = Conversation(
            system_prompt="You are a helpful AI assistant. Be concise and friendly."
        )

        # Initialize cost tracker
        self.cost_tracker = CostTracker(
            warning_threshold=self.settings.cost_warning_threshold,
            limit_threshold=self.settings.cost_limit_threshold,
        )

        # Initialize provider
        self.current_provider_name: ProviderType = self.settings.default_provider
        self.provider: BaseLLMProvider | None = None

    async def initialize_provider(self, provider_name: ProviderType) -> None:
        """
        Initialize LLM provider.
        
        Args:
            provider_name: Provider to initialize ('openai' or 'anthropic')
        """
        # Close existing provider if any
        if self.provider:
            try:
                await self.provider.close()
            except Exception:
                pass

        # Create new provider
        if provider_name == "openai":
            self.provider = OpenAIProvider(
                api_key=self.settings.openai_api_key.get_secret_value(),
                model="gpt-4o-mini",
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            )
        elif provider_name == "anthropic":
            self.provider = AnthropicProvider(
                api_key=self.settings.anthropic_api_key.get_secret_value(),
                model="claude-3-5-haiku-20241022",
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            )
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

        self.current_provider_name = provider_name
        logger.info(f"Initialized provider: {self.provider}")

    async def send_message(self, user_message: str) -> str:
        """
        Send message to LLM and get response.
        
        Args:
            user_message: User's message
            
        Returns:
            Assistant's response
            
        Raises:
            Exception: If API call fails
        """
        if not self.provider:
            raise RuntimeError("Provider not initialized")

        # Check cost limit
        if self.cost_tracker.should_stop():
            raise RuntimeError(
                f"Cost limit reached! Total: ${self.cost_tracker.get_total_cost():.6f}"
            )

        # Add user message to conversation
        self.conversation.add_user_message(user_message)

        # Get messages for API
        messages = self.conversation.get_messages()

        # Call LLM API
        response = await self.provider.chat(messages)

        # Add assistant response to conversation
        self.conversation.add_assistant_message(response.content)

        # Track cost
        cost = self.cost_tracker.add_entry(
            provider=self.provider.provider_name,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

        # Log API call
        log_api_call(
            logger,
            provider=self.provider.provider_name,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=cost,
        )

        # Check for cost warning
        if self.cost_tracker.should_warn():
            total_cost = self.cost_tracker.get_total_cost()
            logger.warning(f"Cost warning! Total: ${total_cost:.6f}")

        return response.content

    def handle_command(self, command: str) -> str | None:
        """
        Handle special commands.
        
        Args:
            command: Command string (e.g., '/help', '/clear')
            
        Returns:
            Command response or None if command not recognized
        """
        command = command.strip().lower()

        if command == "/help":
            return """
Available Commands:
  /help        - Show this help message
  /clear       - Clear conversation history
  /quit, /exit - Exit the application
  /model <provider> - Switch provider (openai or anthropic)
  /cost        - Show cost summary
  /history     - Show conversation history
"""

        elif command == "/clear":
            self.conversation.clear()
            logger.info("Conversation history cleared")
            return "✅ Conversation history cleared!"

        elif command in ["/quit", "/exit"]:
            return "QUIT"

        elif command.startswith("/model"):
            parts = command.split()
            if len(parts) != 2:
                return "❌ Usage: /model <openai|anthropic>"

            provider = parts[1].lower()
            if provider not in ["openai", "anthropic"]:
                return "❌ Provider must be 'openai' or 'anthropic'"

            return f"SWITCH_MODEL:{provider}"

        elif command == "/cost":
            return self.cost_tracker.get_summary()

        elif command == "/history":
            if not self.conversation.messages:
                return "No conversation history yet."

            history = ["Conversation History:", "=" * 60]
            for msg in self.conversation.messages:
                history.append(f"{msg.role.upper()}: {msg.content[:100]}...")
            history.append("=" * 60)
            return "\n".join(history)

        return None

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.provider:
            try:
                await self.provider.close()
                logger.info("Provider closed")
            except Exception as e:
                log_error(logger, e, "Error closing provider")


async def main() -> None:
    """Main application entry point."""
    print("=" * 70)
    print("🤖 AI ASSISTANT - Multi-Provider CLI")
    print("=" * 70)
    print("Type '/help' for available commands")
    print("Type '/quit' or '/exit' to exit")
    print("=" * 70)
    print()

    assistant = AIAssistant()

    try:
        # Initialize default provider
        await assistant.initialize_provider(assistant.current_provider_name)
        print(f"✅ Connected to {assistant.current_provider_name.upper()}")
        print(f"📊 Model: {assistant.provider.model}")
        print()

        # Main interaction loop
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Check if it's a command
                if user_input.startswith("/"):
                    result = assistant.handle_command(user_input)

                    if result == "QUIT":
                        print("\n👋 Goodbye! Thanks for using AI Assistant.")
                        break

                    elif result and result.startswith("SWITCH_MODEL:"):
                        provider_name = result.split(":")[1]
                        print(f"\n🔄 Switching to {provider_name.upper()}...")
                        await assistant.initialize_provider(provider_name)
                        print(f"✅ Now using {provider_name.upper()}")
                        print(f"📊 Model: {assistant.provider.model}\n")

                    elif result:
                        print(f"\n{result}\n")

                    continue

                # Send message to LLM
                print("\n🤔 Thinking...\n")
                response = await assistant.send_message(user_input)
                print(f"Assistant: {response}\n")

                # Show cost if warning threshold reached
                if assistant.cost_tracker.should_warn():
                    total_cost = assistant.cost_tracker.get_total_cost()
                    print(f"⚠️  Cost Warning: ${total_cost:.6f}\n")

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Use /quit to exit gracefully.")
                continue

            except RuntimeError as e:
                print(f"\n❌ Error: {e}\n")
                if "Cost limit" in str(e):
                    break

            except Exception as e:
                log_error(logger, e, "Error processing message")
                print(f"\n❌ Error: {e}\n")
                print("Please try again or use /quit to exit.\n")

    except Exception as e:
        log_error(logger, e, "Fatal error in main loop")
        print(f"\n❌ Fatal error: {e}")

    finally:
        # Cleanup
        await assistant.cleanup()

        # Show final cost summary
        if assistant.cost_tracker.entries:
            print("\n" + assistant.cost_tracker.get_summary())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)