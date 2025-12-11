"""
Cost tracking and estimation for LLM API calls.

Ye file API calls ka cost calculate karti hai tokens ke base pe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import tiktoken


# Pricing information (per 1 million tokens)
PRICING = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
}

ProviderType = Literal["openai", "anthropic"]
ModelType = Literal["gpt-4o-mini", "claude-3-5-haiku-20241022"]


@dataclass
class CostEntry:
    """
    Single cost entry for an API call.
    
    Attributes:
        provider: LLM provider name
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cost: Cost in USD
    """

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float

    def __str__(self) -> str:
        return (
            f"{self.provider}/{self.model}: "
            f"{self.input_tokens}→{self.output_tokens} tokens, "
            f"${self.cost:.6f}"
        )


@dataclass
class CostTracker:
    """
    Tracks costs across multiple API calls.
    
    Attributes:
        entries: List of cost entries
        warning_threshold: Warning threshold in USD
        limit_threshold: Hard limit in USD
    """

    entries: list[CostEntry] = field(default_factory=list)
    warning_threshold: float = 0.10
    limit_threshold: float = 1.00

    def add_entry(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Add a cost entry and return the cost.
        
        Args:
            provider: LLM provider name
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in USD
            
        Example:
            >>> tracker = CostTracker()
            >>> cost = tracker.add_entry("openai", "gpt-4o-mini", 1000, 500)
            >>> print(f"${cost:.6f}")
            $0.000450
        """
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        entry = CostEntry(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )
        self.entries.append(entry)
        return cost

    @staticmethod
    def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for a given model and token counts.
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in USD
            
        Example:
            >>> cost = CostTracker.calculate_cost("gpt-4o-mini", 1000, 500)
            >>> print(f"${cost:.6f}")
            $0.000450
        """
        if model not in PRICING:
            # Default to gpt-4o-mini pricing if model unknown
            model = "gpt-4o-mini"

        pricing = PRICING[model]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def get_total_cost(self) -> float:
        """
        Get total cost across all entries.
        
        Returns:
            Total cost in USD
        """
        return sum(entry.cost for entry in self.entries)

    def get_total_tokens(self) -> tuple[int, int]:
        """
        Get total input and output tokens.
        
        Returns:
            Tuple of (total_input_tokens, total_output_tokens)
        """
        total_input = sum(entry.input_tokens for entry in self.entries)
        total_output = sum(entry.output_tokens for entry in self.entries)
        return total_input, total_output

    def should_warn(self) -> bool:
        """Check if cost exceeds warning threshold."""
        return self.get_total_cost() >= self.warning_threshold

    def should_stop(self) -> bool:
        """Check if cost exceeds limit threshold."""
        return self.get_total_cost() >= self.limit_threshold

    def get_summary(self) -> str:
        """
        Get a formatted summary of costs.
        
        Returns:
            Formatted cost summary string
        """
        total_cost = self.get_total_cost()
        total_input, total_output = self.get_total_tokens()

        summary = [
            "=" * 60,
            "💰 SESSION COST SUMMARY",
            "=" * 60,
            f"Total API Calls: {len(self.entries)}",
            f"Total Tokens: {total_input:,} input → {total_output:,} output",
            f"Total Cost: ${total_cost:.6f}",
            f"",
            f"Warning Threshold: ${self.warning_threshold:.2f}",
            f"Limit Threshold: ${self.limit_threshold:.2f}",
        ]

        if self.should_stop():
            summary.append(f"\n⚠️  LIMIT EXCEEDED! Cost has reached ${total_cost:.6f}")
        elif self.should_warn():
            summary.append(f"\n⚠️  Warning: Cost is ${total_cost:.6f}")

        summary.append("=" * 60)
        return "\n".join(summary)

    def reset(self) -> None:
        """Clear all cost entries."""
        self.entries = []

    def __str__(self) -> str:
        total_cost = self.get_total_cost()
        return f"CostTracker({len(self.entries)} calls, ${total_cost:.6f})"


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens for
        model: Model name (for encoding)
        
    Returns:
        Number of tokens
        
    Example:
        >>> count_tokens("Hello, world!")
        4
    """
    try:
        # OpenAI models ke liye tiktoken use karo
        if "gpt" in model.lower():
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        else:
            # Anthropic models ke liye approximate
            # Roughly 1 token = 4 characters
            return len(text) // 4
    except Exception:
        # Fallback: approximate calculation
        return len(text) // 4


if __name__ == "__main__":
    # Test cost tracker
    print("Testing Cost Tracker\n")

    tracker = CostTracker(warning_threshold=0.001, limit_threshold=0.01)

    # Test token counting
    text = "Hello, how are you doing today? This is a test message."
    tokens = count_tokens(text, "gpt-4o-mini")
    print(f"Text: '{text}'")
    print(f"Tokens: {tokens}\n")

    # Test cost calculation
    print("Cost Calculations:")
    cost1 = tracker.calculate_cost("gpt-4o-mini", 1000, 500)
    print(f"GPT-4o-mini (1000→500 tokens): ${cost1:.6f}")

    cost2 = tracker.calculate_cost("claude-3-5-haiku-20241022", 1000, 500)
    print(f"Claude Haiku (1000→500 tokens): ${cost2:.6f}\n")

    # Test tracker
    print("Adding entries to tracker:")
    for i in range(5):
        cost = tracker.add_entry("openai", "gpt-4o-mini", 1000, 500)
        print(f"  Entry {i + 1}: ${cost:.6f}")

    print(f"\n{tracker}")
    print(f"\nShould warn: {tracker.should_warn()}")
    print(f"Should stop: {tracker.should_stop()}")

    # Print summary
    print(f"\n{tracker.get_summary()}")